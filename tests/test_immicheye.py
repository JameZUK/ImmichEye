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

from immicheye import server  # noqa: E402
from immicheye.gallery import Gallery, GalleryBuilder, labels_for  # noqa: E402
from immicheye.immich import Person  # noqa: E402
from immicheye.ml import Face  # noqa: E402


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
