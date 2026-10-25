"""CodeProject.AI-compatible face endpoints backed by Immich.

Blue Iris talks to nginx, which routes /v1/vision/face/recognize and /v1/vision/face/list
here and everything else straight to CodeProject.AI. Response shapes copy CPAI's
FaceProcessing module (intelligencelayer/face.py, 1.12.x), which is what Blue Iris parses.
"""
import asyncio
import io
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from email.utils import formatdate

import httpx
from PIL import Image
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from . import config
from .alerts import AlertNames, alert_key
from .blueiris import LateNames
from .cpai_sync import CpaiSync
from .events import EventSink, camera_of, context_box
from .gallery import Gallery, GalleryBuilder
from .immich import Immich
from .ml import ImmichML
from .tiles import detect

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("immicheye")
settings = config.load()
state = {"gallery": Gallery(), "last_error": None, "cpai_sync_error": None, "refreshing": False, "unavailable": 0, "requests": 0}
LOG_MAX_BYTES = 20 * 1024 * 1024


async def refresh(builder: GalleryBuilder, cpai: CpaiSync) -> None:
    if state["refreshing"]:
        return
    state["refreshing"] = True
    try:
        state["gallery"] = await builder.build()
        state["last_error"] = None
        try:
            await cpai.sync(builder.cache, builder.crops)
            state["cpai_sync_error"] = None
        except Exception as e:
            log.exception("CodeProject.AI sync failed")
            state["cpai_sync_error"] = f"{type(e).__name__}: {e}"
    except Exception as e:
        log.exception("gallery refresh failed; keeping the previous gallery")
        state["last_error"] = f"{type(e).__name__}: {e}"
    finally:
        state["refreshing"] = False


async def refresh_loop(builder: GalleryBuilder, cpai: CpaiSync) -> None:
    while True:
        await refresh(builder, cpai)
        await asyncio.sleep(settings.refresh_minutes * 60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = httpx.AsyncClient(limits=httpx.Limits(max_connections=32))
    app.state.client = client
    app.state.ml = ImmichML(settings.ml_url, settings.model, client)
    app.state.builder = GalleryBuilder(settings, Immich(settings.immich_url, settings.immich_api_key, client), app.state.ml)
    state["gallery"] = app.state.builder.from_cache()
    log.info("loaded %d people, %d faces from cache", len(state["gallery"]), sum(state["gallery"].faces_per_label.values()))
    app.state.cpai = CpaiSync(settings, client)
    app.state.events = EventSink(settings, client)
    app.state.alerts = AlertNames(settings.alert_hold_ms / 1000)
    app.state.late = LateNames(settings, client)
    tasks = [asyncio.create_task(refresh_loop(app.state.builder, app.state.cpai)),
             asyncio.create_task(app.state.events.run())]
    yield
    for task in tasks:
        task.cancel()
    await client.aclose()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


def cpai_envelope(body: dict, command: str, started: float) -> JSONResponse:
    ok = bool(body.get("success"))
    if ok:
        body["processMs"] = int((time.perf_counter() - started) * 1000)
    body.update({
        "moduleId": "FaceProcessing",
        "moduleName": "Face Processing",
        "code": 200 if ok else 500,
        "command": command,
        "requestId": str(uuid.uuid4()),
        "inferenceDevice": "GPU",
        "analysisRoundTripMs": int((time.perf_counter() - started) * 1000),
        "processedBy": "immicheye",
        "timestampUTC": formatdate(usegmt=True),
    })
    # CPAI answers HTTP 200 even when code is 500; Blue Iris reads the body.
    return JSONResponse(body)


async def unavailable(request: Request, body: bytes, started: float, why: str) -> Response:
    """Recognition can't run here. Report no face: a wrong name misleads, an unknown face
    doesn't, and CodeProject.AI names about half of faces wrongly even with the synced
    gallery. IMMICHEYE_CPAI_FALLBACK=1 lets CodeProject.AI answer instead."""
    state["unavailable"] += 1
    if not settings.cpai_fallback:
        return cpai_envelope({"success": False, "error": f"ImmichEye: {why}"}, "recognize", started)
    headers = {k: v for k, v in request.headers.items() if k.lower() in ("content-type", "user-agent")}
    r = await request.app.state.client.post(
        f"{settings.cpai_url}{request.url.path}", content=body, headers=headers, timeout=30)
    return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))


def log_recognition(entry: dict) -> None:
    if not settings.log_recognitions:
        return
    path = settings.data_dir / "recognitions.jsonl"
    try:
        if path.exists() and path.stat().st_size > LOG_MAX_BYTES:
            path.replace(path.with_suffix(".jsonl.1"))
        with path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        log.exception("could not write recognition log")


def save_face_crops(jpeg: bytes, entry: dict) -> None:
    """Keep each face Blue Iris sent (with context) for checking results; newest N only."""
    if not settings.save_faces or not entry["faces"]:
        return
    out = settings.data_dir / "faces"
    try:
        out.mkdir(parents=True, exist_ok=True)
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(entry["ts"])) + f"-{int(entry['ts'] * 1000) % 1000:03d}"
        for i, f in enumerate(entry["faces"]):
            x1, y1, x2, y2 = f["box"]
            crop = img.crop(context_box(f["box"], img.width, img.height))
            score = f["top"][0][1] if f["top"] else 0
            name = f"{stamp}_{i}_{f['result']}_{f['top'][0][0] if f['top'] else 'none'}-{score:.2f}_{x2 - x1}px.jpg"
            crop.save(out / name, "JPEG", quality=90)
            f["crop"] = name
        for old in sorted(out.glob("*.jpg"))[: -settings.save_faces]:
            old.unlink()
    except Exception:
        log.exception("could not save face crops")


@app.post("/v1/vision/face/recognize")
async def recognize(request: Request) -> Response:
    started = time.perf_counter()
    state["requests"] += 1
    body = await request.body()
    gallery: Gallery = state["gallery"]
    if not len(gallery.owners):
        return await unavailable(request, body, started, "gallery not built yet")

    form = await request.form()
    upload = form.get("image")
    if upload is None or isinstance(upload, str):
        return cpai_envelope({"success": False, "error": "No image provided"}, "recognize", started)
    jpeg = await upload.read()
    try:
        size = Image.open(io.BytesIO(jpeg)).size   # reads the header only
    except Exception:
        size = None
    entry = {"ts": time.time(), "file": getattr(upload, "filename", None), "size": list(size or ()) or None,
             "bytes": len(jpeg), "min_confidence": form.get("min_confidence")}

    t = time.perf_counter()
    try:
        faces, tiles = await detect(request.app.state.ml, jpeg, settings.detect_min_score, settings.tile_px, size)
    except Exception as e:
        log.warning("Immich ML failed: %s", e)
        return await unavailable(request, body, started, "Immich ML unavailable")
    inference_ms = int((time.perf_counter() - t) * 1000)
    if tiles:
        entry["tiles"] = tiles

    predictions, logged, events = [], [], []
    for f in faces:
        matches = gallery.match(f.embedding, settings.top_k)
        best = matches[0]
        runner_up = matches[1].score if len(matches) > 1 else -1.0
        known = best.score >= settings.match_threshold and best.score - runner_up >= settings.match_margin
        x1, y1, x2, y2 = (max(0, v) for v in f.box)
        predictions.append({
            # On CodeProject.AI's scale, (cosine + 1) / 2, so Blue Iris's one face
            # confidence setting means about the same for ImmichEye and the fallback.
            "confidence": round((best.score + 1) / 2, 4) if known else 0,
            "userid": best.label if known else "unknown",
            "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2,
        })
        logged.append({"box": [x1, y1, x2, y2], "det": round(f.score, 3),
                       "top": [[m.label, round(m.score, 3)] for m in matches[:3]],
                       "result": predictions[-1]["userid"]})
        events.append({**logged[-1], "score": round(best.score, 3), "embedding": f.embedding})

    entry.update(ms=inference_ms, faces=logged)
    alert = alert_key(entry["file"])
    if alert and settings.alert_share:
        predictions, shared, entry["held_ms"] = await request.app.state.alerts.share(alert, predictions)
        if shared:
            entry["shared"] = shared
        if entry["held_ms"] and await request.is_disconnected():
            entry["gave_up"] = True   # Blue Iris stopped waiting for this reply
    request.app.state.late.observe(camera_of(entry["file"]), alert, entry["file"], predictions)
    request.app.state.events.submit(jpeg, entry, events)
    save_face_crops(jpeg, entry)
    log_recognition(entry)
    if not predictions:
        return cpai_envelope({"success": False, "error": "No face found in image", "inferenceMs": inference_ms},
                             "recognize", started)
    found_known = any(p["userid"] != "unknown" for p in predictions)
    return cpai_envelope({
        "message": "A face was recognised" if found_known else "No known faces",
        "count": len(predictions),
        "predictions": predictions,
        "success": True,
        "inferenceMs": inference_ms,
    }, "recognize", started)


@app.post("/v1/vision/face/list")
async def list_faces() -> Response:
    started = time.perf_counter()
    return cpai_envelope({"success": True, "faces": state["gallery"].labels}, "list", started)


@app.post("/v1/vision/face/delete")
async def delete_face(request: Request) -> Response:
    """Acknowledged, never done. Blue Iris's faces view deletes every server face that isn't in
    its own faces folder, so passing this on would wipe CodeProject.AI's synced faces.
    Faces are managed in Immich."""
    started = time.perf_counter()
    userid = (await request.form()).get("userid")
    log.info("ignored face delete from Blue Iris: %s", userid)
    return cpai_envelope({"success": True}, "delete", started)


@app.post("/v1/vision/face/register")
async def register_face(request: Request) -> Response:
    started = time.perf_counter()
    userid = (await request.form()).get("userid")
    log.info("refused face register from Blue Iris: %s", userid)
    return cpai_envelope({"success": False, "error": "Faces are managed in Immich (ImmichEye)"}, "register", started)


@app.get("/immicheye/status")
async def status(request: Request) -> dict:
    g: Gallery = state["gallery"]
    return {
        "people": len(g),
        "faces": sum(g.faces_per_label.values()),
        "embeddings": len(g.owners),
        "faces_per_label": g.faces_per_label,
        "built_at": g.built_at,
        "refreshing": state["refreshing"],
        "last_error": state["last_error"],
        "requests": state["requests"],
        "unavailable": state["unavailable"],
        "cpai_fallback": settings.cpai_fallback,
        "cpai_sync": settings.cpai_sync,
        "cpai_sync_error": state["cpai_sync_error"],
        "match_threshold": settings.match_threshold,
        "match_margin": settings.match_margin,
        "events": request.app.state.events.status(),
        "late_names": request.app.state.late.status(),
    }


@app.post("/immicheye/explain")
async def explain(request: Request) -> dict:
    """Every face in a posted image with its five best matches, for tuning the thresholds.

    curl -F image=@frame.jpg http://<host>:8000/immicheye/explain
    """
    upload = (await request.form()).get("image")
    if upload is None or isinstance(upload, str):
        return {"error": "post the image as multipart field 'image'"}
    jpeg = await upload.read()
    size = Image.open(io.BytesIO(jpeg)).size
    faces, _ = await detect(request.app.state.ml, jpeg, settings.detect_min_score, settings.tile_px, size)
    return {"faces": [{
        "box": list(f.box),
        "det": round(f.score, 3),
        "matches": [[m.label, round(m.score, 3)] for m in state["gallery"].match(f.embedding, settings.top_k)[:5]],
    } for f in faces]}


@app.post("/immicheye/refresh")
async def trigger_refresh(request: Request) -> dict:
    asyncio.create_task(refresh(request.app.state.builder, request.app.state.cpai))
    return {"started": not state["refreshing"]}
