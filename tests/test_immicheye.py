import dataclasses
import json
import io
import os

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
