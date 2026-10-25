import asyncio
import base64
import dataclasses
import json
import io
import os
import time

import httpx
import numpy as np
import pytest
from PIL import Image

# Nothing listens here, so the startup gallery refresh fails fast and leaves an empty gallery.
os.environ.update(IMMICH_URL="http://127.0.0.1:9", IMMICH_ML_URL="http://127.0.0.1:9",
                  CPAI_URL="http://cpai.test", IMMICHEYE_LOG_RECOGNITIONS="0")

import respx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from immicheye import config, server  # noqa: E402
from immicheye import blueiris  # noqa: E402
from immicheye.alerts import AlertNames, alert_key  # noqa: E402
from immicheye.blueiris import LateNames, merge_memo  # noqa: E402
from immicheye.gallery import Gallery, GalleryBuilder, labels_for  # noqa: E402
from immicheye.immich import FaceRef, Immich, Person  # noqa: E402
from immicheye.ml import Face  # noqa: E402
from immicheye.tiles import tile_boxes  # noqa: E402


def unit(*v):
    a = np.zeros(512, dtype=np.float32)
    a[: len(v)] = v
    return a / np.linalg.norm(a)


def jpeg():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, "JPEG")
    return buf.getvalue()


def test_labels():
    people = [Person("1", "Cal Smith"), Person("2", "Cal Brown"), Person("3", "Bea"),
              Person("4", "Dan2"), Person("5", "Eve Ellen"), Person("6", "Bea")]
    full = labels_for(people + [Person("7", "Cal Smith (Dad)")], {}, "full")
    assert full["3"] == "bea" and full["5"] == "eve_ellen" and full["7"] == "cal_smith_dad"
    labels = labels_for(people, {"Eve Ellen": "eve"})
    assert labels == {"1": "cal_smith", "2": "cal_brown", "3": "bea", "4": "dan", "5": "eve", "6": "bea"}


def test_match_uses_top_k_mean():
    g = Gallery(labels=["a", "b"], owners=np.array([0, 0, 1]),
                embeddings=np.stack([unit(1, 0), unit(1, 1), unit(0, 1)]))
    m = g.match(unit(1, 0), top_k=2)
    assert [x.label for x in m] == ["a", "b"]
    assert m[0].score == pytest.approx((1 + 2 ** -0.5) / 2, abs=1e-5)
    assert m[1].score == pytest.approx(0, abs=1e-6)


class FakeML:
    def __init__(self, faces=None, error=None):
        self._faces, self.error = faces or [], error

    async def faces(self, jpeg, min_score):
        if self.error:
            raise self.error
        return self._faces


@pytest.fixture
def client():
    with TestClient(server.app) as c:
        server.state["gallery"] = Gallery(
            labels=["bea", "cal"], owners=np.array([0, 1]), embeddings=np.stack([unit(1, 0), unit(0, 1)]))
        yield c


def post(c):
    return c.post("/v1/vision/face/recognize",
                  data={"min_confidence": "0.40"}, files={"image": ("Cam1.jpg", jpeg(), "image/jpeg")})


def test_recognize_known_and_unknown(client):
    server.app.state.ml = FakeML([
        Face((10, 20, 50, 70), 0.9, unit(0.1, 1)),   # close to cal
        Face((-5, 0, 30, 30), 0.8, unit(1, 1)),      # between both: no margin -> unknown
    ])
    body = post(client).json()
    assert body["success"] is True and body["code"] == 200 and body["command"] == "recognize"
    assert body["message"] == "A face was recognised" and body["count"] == 2
    known, other = body["predictions"]
    assert known["userid"] == "cal" and known["confidence"] > 0.9
    assert (known["x_min"], known["y_min"], known["x_max"], known["y_max"]) == (10, 20, 50, 70)
    assert other == {"confidence": 0, "userid": "unknown", "x_min": 0, "y_min": 0, "x_max": 30, "y_max": 30}


def test_recognize_no_face_matches_cpai(client):
    server.app.state.ml = FakeML([])
    body = post(client).json()
    assert body["success"] is False and body["error"] == "No face found in image" and body["code"] == 500


def no_fallback(monkeypatch):
    monkeypatch.setattr(server, "settings", dataclasses.replace(server.settings, cpai_fallback=False))


def test_ml_down_without_fallback_is_no_face(client, monkeypatch):
    no_fallback(monkeypatch)
    server.app.state.ml = FakeML(error=ConnectionError("down"))
    body = post(client).json()
    assert body["success"] is False and body["error"] == "ImmichEye: Immich ML unavailable"


def test_empty_gallery_without_fallback_is_no_face(client, monkeypatch):
    no_fallback(monkeypatch)
    server.state["gallery"] = Gallery()
    assert post(client).json()["error"] == "ImmichEye: gallery not built yet"


@respx.mock
def test_cpai_fallback(client, monkeypatch):
    monkeypatch.setattr(server, "settings", dataclasses.replace(server.settings, cpai_fallback=True))
    server.app.state.ml = FakeML(error=ConnectionError("down"))
    route = respx.post("http://cpai.test/v1/vision/face/recognize").respond(json={"success": True, "from": "cpai"})
    assert post(client).json()["from"] == "cpai"
    sent = route.calls.last.request
    assert b"Cam1.jpg" in sent.content and sent.headers["content-type"].startswith("multipart/form-data")


def test_gallery_from_cache(tmp_path):
    s = dataclasses.replace(server.settings, data_dir=tmp_path)
    b = GalleryBuilder(s, None, None)
    b.cache = {
        "f1": {"person_id": "p1", "asset_id": "a", "embedding": unit(1, 0).tolist(), "label": "bea"},
        "f2": {"person_id": "p2", "asset_id": "a", "embedding": unit(0, 1).tolist(), "label": "cal"},
        "f3": {"person_id": "p3", "asset_id": "a", "embedding": unit(0, 1).tolist(), "label": "cal"},
        "f4": {"person_id": "p4", "asset_id": "a", "embedding": None, "label": "eve"},   # no face found
    }
    b._save_cache()
    g = GalleryBuilder(s, None, None).from_cache()
    assert g.labels == ["bea", "cal"] and g.faces_per_label == {"bea": 1, "cal": 2}
    assert g.match(unit(0, 1), 3)[0].label == "cal"


@respx.mock
def test_delete_and_register_never_reach_cpai(client):
    cpai = respx.route(host="cpai.test").respond(json={"success": True})
    assert client.post("/v1/vision/face/delete", data={"userid": "cal"}).json()["success"] is True
    reg = client.post("/v1/vision/face/register", data={"userid": "cal"},
                      files={"image": ("x.jpg", jpeg(), "image/jpeg")}).json()
    assert reg["success"] is False and "Immich" in reg["error"]
    assert not cpai.called


def test_list(client):
    body = client.post("/v1/vision/face/list").json()
    assert body["success"] is True and body["faces"] == ["bea", "cal"]


def test_clusters_are_deterministic():
    from immicheye.cpai_sync import clusters
    rng = np.random.default_rng(0)
    vecs = np.stack([unit(*rng.normal(size=8)) for _ in range(30)])
    ids = [f"f{i:02d}" for i in range(30)]
    a = clusters(ids, vecs, 4)
    b = clusters(ids[::-1], vecs[::-1], 4)
    assert a == b and sorted(sum(a, [])) == sorted(ids) and len(a) == 4


@respx.mock
@pytest.mark.parametrize("mode", ["managed", "exclusive"])
def test_cpai_sync(tmp_path, mode):
    import asyncio
    import httpx
    from immicheye.cpai_sync import CpaiSync
    s = dataclasses.replace(server.settings, data_dir=tmp_path, cpai_sync=mode, cpai_prototypes=2)
    crops = tmp_path / "crops"
    crops.mkdir()
    cache = {}
    for i in range(16):
        fid = f"j{i:02d}"
        cache[fid] = {"person_id": "p", "asset_id": "a", "label": "cal",
                      "embedding": unit(1, (i % 2) * 3, i / 100).tolist()}   # two looks
        (crops / f"{fid}.jpg").write_bytes(jpeg())
    (tmp_path / "cpai_sync.json").write_text('{"cal1": "stale", "eve1": "gone"}')
    cpai = {"faces": ["cal1", "Cal2", "eve1", "Dan4"]}   # cal1/eve1 ours; Cal2, Dan4 the user's
    respx.post("http://cpai.test/v1/vision/face/list").mock(side_effect=lambda r: httpx.Response(200, json={"success": True, **cpai}))
    reg = respx.post("http://cpai.test/v1/vision/face/register").respond(json={"success": True})
    dele = respx.post("http://cpai.test/v1/vision/face/delete").respond(json={"success": True})

    async def go():
        async with httpx.AsyncClient() as c:
            await CpaiSync(s, c).sync(cache, crops)
    asyncio.run(go())

    registered = [r.request.content.split(b'name="userid"\r\n\r\n')[1].split(b"\r\n")[0].decode()
                  for r in reg.calls]
    assert registered == ["cal1", "cal3"]          # reuses its own cal1, skips the user's Cal2
    deleted = sorted(r.request.content.decode().split("userid=")[1] for r in dele.calls)
    assert deleted == (["eve1"] if mode == "managed" else ["Cal2", "Dan4", "eve1"])
    state = json.loads((tmp_path / "cpai_sync.json").read_text())
    assert sorted(state) == ["cal1", "cal3"]


def test_saves_face_crops(client, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "settings", dataclasses.replace(server.settings, data_dir=tmp_path, save_faces=1))
    server.app.state.ml = FakeML([Face((10, 20, 50, 70), 0.9, unit(0.1, 1))])
    post(client)
    post(client)
    crops = list((tmp_path / "faces").glob("*.jpg"))
    assert len(crops) == 1 and "_cal_cal-" in crops[0].name and crops[0].name.endswith("_40px.jpg")


def test_tile_boxes_cover_large_frames_with_overlap():
    from immicheye.tiles import tile_boxes
    assert tile_boxes(1280, 720, 1280) == [] and tile_boxes(3840, 2160, 0) == []
    boxes = tile_boxes(3840, 2160, 1280)
    assert len(boxes) == 8 and all(x2 - x1 == 1280 and y2 - y1 == 1280 for x1, y1, x2, y2 in boxes)
    assert {b[2] for b in boxes} >= {3840} and {b[3] for b in boxes} >= {2160}
    xs = sorted({b[0] for b in boxes})
    assert all(b - a <= 1280 - 320 for a, b in zip(xs, xs[1:]))   # neighbours overlap by at least 320


def test_merge_moves_tile_faces_and_drops_cut_and_duplicate_faces():
    from immicheye.tiles import merge
    e = unit(1, 0)
    whole = [Face((100, 100, 300, 300), 0.8, e)]
    tiled = [
        ((0, 0, 1280, 1280), [Face((110, 105, 290, 295), 0.9, e),     # the whole-frame face again
                              Face((1240, 500, 1278, 540), 0.9, e)]),  # cut off by the inner edge
        ((960, 0, 2240, 1280), [Face((280, 500, 318, 540), 0.85, e)]),  # same small face, whole
    ]
    faces = merge(whole, tiled, 2240, 1280)
    assert sorted(f.box for f in faces) == [(110, 105, 290, 295), (1240, 500, 1278, 540)]


def test_gallery_includes_configured_lowres_copies(tmp_path):
    s = dataclasses.replace(server.settings, data_dir=tmp_path, lowres_px=(32,))
    b = GalleryBuilder(s, None, None)
    b.cache = {
        "f1": {"person_id": "p1", "asset_id": "a", "embedding": unit(1, 0).tolist(), "label": "bea",
               "lowres": {"32": unit(1, 0.2).tolist(), "48": unit(1, 0.1).tolist()}},
        "f2": {"person_id": "p2", "asset_id": "a", "embedding": unit(0, 1).tolist(), "label": "cal",
               "lowres": {"32": None}},
    }
    g = b.from_cache()
    assert len(g.owners) == 3 and g.faces_per_label == {"bea": 1, "cal": 1}


def test_shrink_scales_face_to_width():
    from immicheye.gallery import shrink
    crop = Image.new("RGB", (220, 260))   # a 100 px face plus margin
    small = Image.open(io.BytesIO(shrink(crop, 32)))
    assert small.width == 70 and shrink(Image.new("RGB", (60, 70)), 32) is None


def test_title_case_labels():
    people = [Person("1", "cal smith"), Person("2", "Bea"), Person("3", "Dan Lee (DJ)"),
              Person("4", "Eve O'Neil"), Person("5", "Cal Brown")]
    assert labels_for(people, {}, "full", "title") == {
        "1": "Cal_Smith", "2": "Bea", "3": "Dan_Lee_DJ", "4": "Eve_O_Neil", "5": "Cal_Brown"}
    first = labels_for(people, {}, "first", "title")
    assert first["2"] == "Bea" and first["1"] == "Cal_Smith" and first["3"] == "Dan"


@respx.mock
def test_cpai_sync_label_case_change_keeps_numbers(tmp_path):
    import asyncio
    import httpx
    from immicheye.cpai_sync import CpaiSync
    s = dataclasses.replace(server.settings, data_dir=tmp_path, cpai_sync="managed", cpai_prototypes=1)
    crops = tmp_path / "crops"
    crops.mkdir()
    cache = {f"f{i}": {"person_id": "p", "asset_id": "a", "label": "Cal_Smith", "embedding": unit(1, i).tolist()}
             for i in range(8)}
    for fid in cache:
        (crops / f"{fid}.jpg").write_bytes(jpeg())
    (tmp_path / "cpai_sync.json").write_text('{"cal_smith1": "old"}')
    respx.post("http://cpai.test/v1/vision/face/list").respond(json={"success": True, "faces": ["cal_smith1"]})
    reg = respx.post("http://cpai.test/v1/vision/face/register").respond(json={"success": True})
    dele = respx.post("http://cpai.test/v1/vision/face/delete").respond(json={"success": True})

    async def go():
        async with httpx.AsyncClient() as c:
            await CpaiSync(s, c).sync(cache, crops)
    asyncio.run(go())
    assert [r.request.content.split(b'name="userid"\r\n\r\n')[1].split(b"\r\n")[0] for r in reg.calls] == [b"Cal_Smith1"]
    assert [r.request.content for r in dele.calls] == [b"userid=cal_smith1"]
    assert sorted(json.loads((tmp_path / "cpai_sync.json").read_text())) == ["Cal_Smith1"]


def test_space_separated_labels(monkeypatch):
    from immicheye import config
    people = [Person("1", "cal smith"), Person("2", "Bea Jones (Mum)")]
    assert labels_for(people, {}, "full", "title", " ") == {"1": "Cal Smith", "2": "Bea Jones Mum"}
    assert labels_for(people, {}, "full", "lower", " ")["1"] == "cal smith"
    monkeypatch.setenv("IMMICHEYE_LABEL_SEPARATOR", "space")
    assert config.load().label_sep == " "
    monkeypatch.setenv("IMMICHEYE_LABEL_SEPARATOR", "-")
    with pytest.raises(ValueError):
        config.load()


EVENTS = "http://events.test/api/events"


def events_on(client, **kw):
    sink = server.app.state.events
    sink.s = dataclasses.replace(server.settings, events_url=EVENTS, **kw)
    return sink


def settle(sink):
    """Wait for the background sender to finish what was queued."""
    for _ in range(200):
        if not sink.pending:
            return
        time.sleep(0.01)
    raise AssertionError("events still pending")


def post_as(c, filename):
    buf = io.BytesIO()
    Image.new("RGB", (200, 120), "gray").save(buf, "JPEG")
    return c.post("/v1/vision/face/recognize", files={"image": (filename, buf.getvalue(), "image/jpeg")})


@respx.mock
def test_events_off_by_default(client):
    route = respx.post(EVENTS).respond(202)
    server.app.state.ml = FakeML([Face((10, 20, 50, 70), 0.9, unit(0.1, 1))])
    post_as(client, "Front.20261025_142653Z.jpg")
    settle(server.app.state.events)
    assert not route.called and server.app.state.events.status()["url_set"] is False


@respx.mock
def test_event_posted_for_each_frame_with_faces(client):
    sink = events_on(client, events_token="tok", events_frame_px=64)
    route = respx.post(EVENTS).respond(202, json={"accepted": 1})
    server.app.state.ml = FakeML([Face((10, 20, 50, 70), 0.9, unit(0.1, 1))])
    reply = post_as(client, "Front.20261025_142653Z.3870.3-1.jpg").json()
    settle(sink)
    assert route.call_count == 1
    req = route.calls.last.request
    assert req.headers["authorization"] == "Bearer tok"
    ev = json.loads(req.content)
    assert ev["v"] == 1 and ev["camera"] == "Front" and ev["size"] == [200, 120] and ev["model"] == "buffalo_l"
    (f,) = ev["faces"]
    emb = np.frombuffer(base64.b64decode(f["embedding"]), dtype="<f4")
    assert emb.shape == (512,) and np.linalg.norm(emb) == pytest.approx(1, abs=1e-5)
    assert f["result"] == reply["predictions"][0]["userid"] == "cal" and f["score"] > 0.9
    assert Image.open(io.BytesIO(base64.b64decode(f["crop"]))).size == (75, 95)   # box grown by 25 px, clamped at 0
    assert max(Image.open(io.BytesIO(base64.b64decode(ev["frame"]))).size) == 64
    assert sink.status()["sent"] == 1

    server.app.state.ml = FakeML([])   # no face: nothing sent
    post_as(client, "Front.20261025_142700Z.jpg")
    settle(sink)
    assert route.call_count == 1


@respx.mock
@pytest.mark.parametrize("fail", [httpx.Response(500), httpx.ConnectError("down")])
def test_event_failure_does_not_change_reply(client, fail):
    sink = events_on(client)
    server.app.state.ml = FakeML([Face((10, 20, 50, 70), 0.9, unit(0.1, 1))])
    expected = post_as(client, "Front.x.jpg").json()["predictions"]
    settle(sink)
    respx.post(EVENTS).mock(side_effect=[fail])
    before = client.get("/immicheye/status").json()["events"]["failed"]
    assert post_as(client, "Front.y.jpg").json()["predictions"] == expected
    settle(sink)
    assert client.get("/immicheye/status").json()["events"]["failed"] == before + 1


@respx.mock
def test_events_only_for_chosen_cameras(client):
    sink = events_on(client, events_cameras=("Front",))
    route = respx.post(EVENTS).respond(202)
    server.app.state.ml = FakeML([Face((10, 20, 50, 70), 0.9, unit(0.1, 1))])
    post_as(client, "Drive.20261025_142653Z.jpg")
    post_as(client, "Front.20261025_142653Z.jpg")
    post_as(client, "nodot")
    settle(sink)
    assert route.call_count == 1 and json.loads(route.calls.last.request.content)["camera"] == "Front"


ALERT = "Front.20260101_120000Z.1234.3-1.jpg"


def test_alert_key_only_for_alert_images():
    assert alert_key(ALERT) == ALERT.lower()
    assert alert_key("Drive.20260101_120000Z.17-0.jpg") is None   # no record number
    assert alert_key("Front.20260101_120007Z.jpg") is None and alert_key(None) is None
    assert alert_key("Drive.20260101_120000Z.1240.17-0.jpg")


def pred(label, conf=0.75):
    return {"confidence": conf if label != "unknown" else 0, "userid": label,
            "x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}


def test_alert_names_wait_for_a_later_image():
    async def run():
        names = AlertNames(hold_s=2)
        held = asyncio.create_task(names.share("a", [pred("unknown")]))
        await asyncio.sleep(0.05)
        own, added, waited = await names.share("a", [pred("cal")])
        assert [p["userid"] for p in own] == ["cal"] and not added and waited == 0
        preds, added, waited = await held
        assert [p["userid"] for p in preds] == ["unknown", "cal"] and added == ["cal"] and 0 < waited < 1000
        # a later image with someone else gets cal too, without waiting
        preds, added, waited = await names.share("a", [pred("bea", 0.8)])
        assert added == ["cal"] and waited == 0
        # another alert shares nothing; the hold counts from its first image, not each image
        names.hold_s = 0.2
        preds, added, waited = await names.share("b", [])
        assert preds == [] and added == [] and waited >= 190
        preds, added, waited = await names.share("b", [])
        assert preds == [] and waited == 0
    asyncio.run(run())


def test_recognize_shares_names_within_an_alert(client, monkeypatch):
    monkeypatch.setattr(server, "settings", dataclasses.replace(server.settings, alert_share=True, alert_hold_ms=0))
    server.app.state.ml = FakeML([Face((10, 20, 50, 70), 0.9, unit(0.1, 1))])
    assert post_as(client, ALERT).json()["predictions"][0]["userid"] == "cal"
    server.app.state.ml = FakeML([])
    body = post_as(client, ALERT).json()
    assert body["success"] is True and [p["userid"] for p in body["predictions"]] == ["cal"]
    # not shared with frames outside the alert, or with another alert
    assert post_as(client, "Front.20260101_120007Z.jpg").json()["success"] is False
    assert post_as(client, ALERT.replace("1234", "1235")).json()["success"] is False


def test_merge_memo_keeps_unknown_and_fits_49_chars():
    memo = "person:86%,unknown,car:83%,vehicle:73%"
    out = merge_memo(memo, {"Bea Jones": 0.734, "Cal Smith": 0.745})
    assert out == "person:86%,Cal Smith:74%,Bea Jones:73%,unknown" and len(out) <= 49
    assert merge_memo("person:90%,Cal Smith:80%", {"Cal Smith": 0.7}) == "person:90%,Cal Smith:80%"
    assert merge_memo("", {"Bea": 0.75}) == "Bea:75%"
    assert merge_memo("person:86%,unknown", {"Cal": 0.8}) == "person:86%,Cal:80%,unknown"
    # a name that doesn't fit is left out whole, lowest score first
    long = merge_memo("person:86%,car:83%", {"Cal Smithson-Jones": 0.75, "Beatrice Jones": 0.74})
    assert long == "person:86%,Cal Smithson-Jones:75%"
    # Blue Iris cut its own memo at 49: the partial last label goes
    cut = "person:94%,Bea Jones:80%,Cal Smith:71%,car:79%,ve"
    assert merge_memo(cut[:49], {"Dan": 0.7}) == "person:94%,Bea Jones:80%,Cal Smith:71%,Dan:70%"


BI = "http://bi.test/json"


def fake_blueiris(alerts):
    """A Blue Iris JSON API holding alerts ({path, file, memo}); records memo updates."""
    updates = []

    def handle(request):
        body = json.loads(request.content)
        cmd = body["cmd"]
        if cmd == "login":
            return httpx.Response(200, json={"result": "success", "session": "s1"} if "response" in body
                                  else {"result": "fail", "session": "s0"})
        if cmd == "alertlist":
            return httpx.Response(200, json={"result": "success",
                                             "data": [a for a in alerts if a["camera"] == body["camera"]]})
        if cmd == "update":
            updates.append((body["path"], body["memo"]))
            assert body["mask"] == 2 and body["flags"] == 2   # flag bit 2 kept as it was
            for a in alerts:
                if a["path"] == body["path"]:
                    a["memo"] = body["memo"]
        return httpx.Response(200, json={"result": "success"})

    respx.post(BI).mock(side_effect=handle)
    return updates


@respx.mock
def test_late_names_added_to_the_alert_memo(monkeypatch):
    monkeypatch.setattr(blueiris, "SETTLE_S", 0)
    monkeypatch.setattr(blueiris, "DEBOUNCE_S", 0)
    monkeypatch.setattr(blueiris, "RETRIES", ())
    alert_file = "Front.20260101_120000Z.1234.3-1.jpg"
    updates = fake_blueiris([{"camera": "Front", "path": "@77.bvr", "file": alert_file, "memo": "person:86%,unknown",
                                "flags": 0x14030002}])
    s = dataclasses.replace(server.settings, blueiris_url="http://bi.test", blueiris_user="u", blueiris_password="p")

    async def run():
        async with httpx.AsyncClient() as c:
            late = LateNames(s, c)
            late.observe("Front", alert_key(alert_file), alert_file, [pred("unknown")])
            late.observe("Drive", None, "Drive.20260101_120003Z.jpg", [pred("bea")])     # another camera
            late.observe("Front", None, "Front.20260101_120005Z.jpg", [pred("cal", 0.745)])
            await late.latest["front"].task
            assert updates == [("@77.bvr", "person:86%,cal:74%,unknown")]
            late.observe("Front", None, "Front.20260101_120006Z.jpg", [pred("cal", 0.8)])   # nothing new
            assert late.latest["front"].task.done() and late.status()["updated"] == 1
            late.latest["front"].first -= 3600                                             # past the window
            late.observe("Front", None, "Front.20260101_130000Z.jpg", [pred("bea")])
            assert "bea" not in late.latest["front"].names
    asyncio.run(run())


@respx.mock
def test_late_names_arriving_during_an_update_are_not_lost(monkeypatch):
    monkeypatch.setattr(blueiris, "SETTLE_S", 0)
    monkeypatch.setattr(blueiris, "DEBOUNCE_S", 0)
    monkeypatch.setattr(blueiris, "RETRIES", ())
    alert_file = "Front.20260101_120000Z.1234.3-1.jpg"
    updates = fake_blueiris([{"camera": "Front", "path": "@77.bvr", "file": alert_file, "memo": "person:86%",
                              "flags": 0x14030002}])
    s = dataclasses.replace(server.settings, blueiris_url="http://bi.test", blueiris_user="u", blueiris_password="p")

    async def run():
        async with httpx.AsyncClient() as c:
            late = LateNames(s, c)
            set_memo = late.bi.set_memo

            async def slow_set_memo(*args):   # bea is named while cal's update is in flight
                late.observe("Front", None, "Front.20260101_120006Z.jpg", [pred("bea", 0.7)])
                await set_memo(*args)
            late.bi.set_memo = slow_set_memo
            late.observe("Front", alert_key(alert_file), alert_file, [pred("cal", 0.8)])
            await late.latest["front"].task
            assert [m for _, m in updates] == ["person:86%,cal:80%", "person:86%,cal:80%,bea:70%"]
    asyncio.run(run())


@respx.mock
def test_late_names_wait_for_blue_iris_to_confirm(monkeypatch):
    monkeypatch.setattr(blueiris, "SETTLE_S", 0)
    monkeypatch.setattr(blueiris, "DEBOUNCE_S", 0)
    monkeypatch.setattr(blueiris, "RETRIES", (0.05, 0.05, 0.05))
    alert_file = "Front.20260101_120000Z.1234.3-1.jpg"
    alert = {"camera": "Front", "path": "@77.bvr", "file": alert_file, "memo": "", "flags": 0x20002}   # analysing
    cancelled = {"camera": "Drive", "path": "@78.bvr", "file": "Drive.20260101_120000Z.99.3-1.jpg",
                 "memo": "", "flags": 0x820002}
    updates = fake_blueiris([alert, cancelled])
    s = dataclasses.replace(server.settings, blueiris_url="http://bi.test", blueiris_user="u", blueiris_password="p")

    async def run():
        async with httpx.AsyncClient() as c:
            late = LateNames(s, c)
            late.observe("Front", alert_key(alert_file), alert_file, [pred("cal", 0.8)])
            late.observe("Drive", alert_key(cancelled["file"]), cancelled["file"], [pred("bea", 0.8)])
            await asyncio.sleep(0.07)
            assert updates == []                     # not while Blue Iris is still deciding
            alert.update(memo="person:90%", flags=0x14030002)   # Blue Iris confirms and writes its memo
            await late.latest["front"].task
            await late.latest["drive"].task
            assert updates == [("@77.bvr", "person:90%,cal:80%")]   # and the cancelled alert is left alone
    asyncio.run(run())


@respx.mock
def test_late_names_arriving_during_logout_are_not_lost(monkeypatch):
    monkeypatch.setattr(blueiris, "SETTLE_S", 0)
    monkeypatch.setattr(blueiris, "DEBOUNCE_S", 0)
    monkeypatch.setattr(blueiris, "RETRIES", ())
    alert_file = "Front.20260101_120000Z.1234.3-1.jpg"
    updates = fake_blueiris([{"camera": "Front", "path": "@77.bvr", "file": alert_file, "memo": "person:86%",
                              "flags": 0x14030002}])
    s = dataclasses.replace(server.settings, blueiris_url="http://bi.test", blueiris_user="u", blueiris_password="p")

    async def run():
        async with httpx.AsyncClient() as c:
            late = LateNames(s, c)
            logout = late.bi.logout

            async def slow_logout(sess):   # bea is named while cal's update is logging out
                late.observe("Front", None, "Front.20260101_120007Z.jpg", [pred("bea", 0.7)])
                await logout(sess)
            late.bi.logout = slow_logout
            late.observe("Front", alert_key(alert_file), alert_file, [pred("cal", 0.8)])
            while not (task := late.latest["front"].task).done():
                await task
            assert [m for _, m in updates] == ["person:86%,cal:80%", "person:86%,cal:80%,bea:70%"]
    asyncio.run(asyncio.wait_for(run(), 5))


@respx.mock
def test_late_names_seen_during_a_failed_update_are_retried(monkeypatch):
    monkeypatch.setattr(blueiris, "SETTLE_S", 0)
    monkeypatch.setattr(blueiris, "DEBOUNCE_S", 0)
    monkeypatch.setattr(blueiris, "RETRIES", ())
    alert_file = "Front.20260101_120000Z.1234.3-1.jpg"
    updates = fake_blueiris([{"camera": "Front", "path": "@77.bvr", "file": alert_file, "memo": "person:86%",
                              "flags": 0x14030002}])
    s = dataclasses.replace(server.settings, blueiris_url="http://bi.test", blueiris_user="u", blueiris_password="p")

    async def run():
        async with httpx.AsyncClient() as c:
            late = LateNames(s, c)
            set_memo, calls = late.bi.set_memo, []

            async def failing_once(*args):   # the first write fails while bea is being named
                calls.append(1)
                if len(calls) == 1:
                    late.observe("Front", None, "Front.20260101_120007Z.jpg", [pred("bea", 0.7)])
                    raise httpx.ConnectError("blue iris restarting")
                await set_memo(*args)
            late.bi.set_memo = failing_once
            late.observe("Front", alert_key(alert_file), alert_file, [pred("cal", 0.8)])
            while not (task := late.latest["front"].task).done():
                await task
            assert [m for _, m in updates] == ["person:86%,cal:80%,bea:70%"]   # both, in one retry
            assert late.status()["failed"] == 1 and not late.tasks
            # a failure with nothing new arriving ends the flush instead of retrying for ever
            late.bi.set_memo = lambda *a: (_ for _ in ()).throw(httpx.ConnectError("down"))
            late.observe("Front", None, "Front.20260101_120009Z.jpg", [pred("dan", 0.7)])
            await asyncio.wait_for(late.latest["front"].task, 2)
            assert late.status()["failed"] == 2
    asyncio.run(asyncio.wait_for(run(), 5))


@respx.mock
def test_late_names_retry_when_seen_again_after_a_failure(monkeypatch):
    monkeypatch.setattr(blueiris, "SETTLE_S", 0)
    monkeypatch.setattr(blueiris, "DEBOUNCE_S", 0)
    monkeypatch.setattr(blueiris, "RETRIES", ())
    monkeypatch.setattr(blueiris, "RETRY_AFTER_S", 0.2)
    alert_file = "Front.20260101_120000Z.1234.3-1.jpg"
    updates = fake_blueiris([{"camera": "Front", "path": "@77.bvr", "file": alert_file, "memo": "person:86%",
                              "flags": 0x14030002}])
    s = dataclasses.replace(server.settings, blueiris_url="http://bi.test", blueiris_user="u", blueiris_password="p")

    async def run():
        async with httpx.AsyncClient() as c:
            late = LateNames(s, c)
            set_memo, down = late.bi.set_memo, [True]

            async def flaky(*args):
                if down[0]:
                    raise httpx.ConnectError("blue iris restarting")
                await set_memo(*args)
            late.bi.set_memo = flaky
            seen = lambda: late.observe("Front", None, "Front.20260101_120005Z.jpg", [pred("cal", 0.8)])
            late.observe("Front", alert_key(alert_file), alert_file, [pred("cal", 0.8)])
            await late.latest["front"].task
            assert updates == [] and late.status()["failed"] == 1
            down[0] = False
            seen()                                   # seen again too soon: no retry yet
            assert late.latest["front"].task.done() and updates == []
            await asyncio.sleep(0.25)
            seen()                                   # after the pause: retried
            await late.latest["front"].task
            assert [m for _, m in updates] == ["person:86%,cal:80%"]
    asyncio.run(asyncio.wait_for(run(), 5))


def test_labels_keep_other_scripts_fold_accents_and_never_say_person():
    people = [Person("1", "Иван Петров"), Person("2", "王伟"), Person("3", "Zoë Ng"), Person("4", "José"),
              Person("5", "123"), Person("6", "Cal Smith")]
    labels = labels_for(people, {"Cal Smith": "cam2"}, "full", "title", " ")
    assert labels == {"1": "Иван Петров", "2": "王伟", "3": "Zoe Ng", "4": "Jose", "5": "unnamed", "6": "cam"}
    assert labels_for([Person("1", "Zoë Ng")], {})["1"] == "zoe"


def test_merge_memo_drops_only_a_cut_last_label():
    whole = "person:94%,Bea Jones:80%,Cal Smithson:71%,dog:79%"   # exactly 49, last label complete
    assert len(whole) == 49
    assert merge_memo(whole, {"Dan": 0.7}) == "person:94%,Bea Jones:80%,Cal Smithson:71%,Dan:70%"
    cut = "person:94%,Bea Jones:80%,Cal Smith:71%,unknown,xy"   # 49: Blue Iris cut "xy..."
    assert len(cut) == 49 and merge_memo(cut, {"Dan": 0.7}) == \
        "person:94%,Bea Jones:80%,Cal Smith:71%,Dan:70%"   # unknown makes way for Dan


def test_tiles_capped_for_oversized_frames():
    assert len(tile_boxes(3840, 2160, 1280)) == 8
    assert tile_boxes(10000, 10000, 1280) == []


def test_on_off_settings_are_strict(monkeypatch):
    monkeypatch.setenv("IMMICHEYE_ALERT_SHARE", "off")
    assert config.load().alert_share is False
    monkeypatch.setenv("IMMICHEYE_ALERT_SHARE", "maybe")
    with pytest.raises(ValueError):
        config.load()
    monkeypatch.setenv("IMMICHEYE_ALERT_SHARE", "0")
    monkeypatch.setenv("IMMICHEYE_CPAI_FALLBACK", "true")   # the router knows only 0 and 1
    with pytest.raises(ValueError):
        config.load()


@respx.mock
def test_immich_people_skips_blank_names():
    respx.get("http://immich.test/api/people").respond(json={"people": [
        {"id": "1", "name": "Cal"}, {"id": "2", "name": "  "}, {"id": "3", "name": ""},
        {"id": "4", "name": "Bea", "isHidden": True}], "hasNextPage": False})

    async def run():
        async with httpx.AsyncClient() as c:
            return await Immich("http://immich.test", "k", c).people()
    assert asyncio.run(run()) == [Person("1", "Cal")]


class FakeImmich:
    """Immich with a list of people, each with one photo holding one 120 px face."""

    def __init__(self, names):
        self.names = names

    async def people(self):
        return [Person(f"p{i}", n) for i, n in enumerate(self.names)]

    async def recent_assets(self, person_id, limit):
        return [f"a-{person_id}"]

    async def faces(self, asset_id):
        pid = asset_id[2:]
        return [FaceRef(f"f-{pid}", asset_id, pid, 100, 100, 220, 220, 640, 480)]

    async def preview(self, asset_id):
        buf = io.BytesIO()
        Image.new("RGB", (640, 480), "gray").save(buf, "JPEG")
        return buf.getvalue()


def build_gallery(tmp_path, immich):
    s = dataclasses.replace(server.settings, data_dir=tmp_path, lowres_px=(), people=())
    return GalleryBuilder(s, immich, FakeML([Face((0, 0, 50, 50), 0.9, unit(1, 0))]))


def test_gallery_build_keeps_previous_gallery_when_immich_returns_little(tmp_path):
    immich = FakeImmich(["Cal", "Bea", "Dan", "Eve"])
    b = build_gallery(tmp_path, immich)
    g = asyncio.run(b.build())
    assert len(g) == 4 and len(list((tmp_path / "crops").glob("*.jpg"))) == 4

    immich.names = []                                    # Immich answers with nobody
    with pytest.raises(RuntimeError, match="no usable faces"):
        asyncio.run(b.build())
    assert len(list((tmp_path / "crops").glob("*.jpg"))) == 4
    assert len(build_gallery(tmp_path, immich).from_cache()) == 4   # the saved cache is untouched

    immich.names = ["Cal"]                               # a sharp shrink: held once, then accepted
    with pytest.raises(RuntimeError, match="shrink"):
        asyncio.run(b.build())
    assert len(asyncio.run(b.build())) == 1
    assert [p.stem for p in (tmp_path / "crops").glob("*.jpg")] == ["f-p0"]
