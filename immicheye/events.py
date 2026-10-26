"""Face events: every face ImmichEye sees, with its embedding, POSTed to another service.

Off unless IMMICHEYE_EVENTS_URL is set. Made for SmartEye's visitor database, but any
receiver of the same JSON works. Sending happens in the background and never slows down
or breaks recognition: a full queue drops the oldest event, a failed POST is not retried.
Queued events hold whole camera frames (crops are cut when sent), so the queue is capped by
size as well as count (IMMICHEYE_EVENTS_QUEUE_MB).
"""
import asyncio
import base64
import io
import json
import logging
import time
import uuid

import httpx
import numpy as np
from PIL import Image

from .config import Settings

log = logging.getLogger(__name__)

QUEUE_MAX = 200
CROP_MAX_PX = 384
WARN_EVERY_S = 60


def camera_of(filename: str | None) -> str | None:
    """Blue Iris names uploads <camera>.<timestamp>[...].jpg."""
    if not filename or "." not in filename:
        return None
    return filename.split(".", 1)[0] or None


def context_box(box, width: int, height: int) -> tuple[int, int, int, int]:
    """The face box grown by half its longer side on every edge, clamped to the frame."""
    x1, y1, x2, y2 = box
    m = max(x2 - x1, y2 - y1) // 2
    return max(0, x1 - m), max(0, y1 - m), min(width, x2 + m), min(height, y2 + m)


def _jpeg_b64(img: Image.Image, max_px: int) -> str:
    if max(img.size) > max_px:
        img = img.copy()
        img.thumbnail((max_px, max_px), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


class EventSink:
    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.s = settings
        self.client = client
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX)
        self.sent = self.failed = self.dropped = 0
        self.pending = 0   # submitted and not yet sent, failed or dropped
        self.queued_bytes = 0
        self.last_error: str | None = None
        self._warned = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.s.events_url)

    def submit(self, jpeg: bytes, entry: dict, faces: list[dict]) -> None:
        """Queue one recognition. faces: box, det, result, score, top and embedding (ndarray)."""
        if not self.enabled or not faces:
            return
        camera = camera_of(entry.get("file"))
        wanted = {c.lower() for c in self.s.events_cameras}
        if wanted and (camera or "").lower() not in wanted:
            return
        budget = self.s.events_queue_mb * 1024 * 1024
        while not self.queue.empty() and (self.queue.full() or self.queued_bytes + len(jpeg) > budget):
            self.queued_bytes -= len(self.queue.get_nowait()[0])   # drop the oldest
            self.dropped += 1
            self.pending -= 1
        self.queue.put_nowait((jpeg, entry, camera, faces))
        self.queued_bytes += len(jpeg)
        self.pending += 1

    def payload(self, jpeg: bytes, entry: dict, camera: str | None, faces: list[dict]) -> bytes:
        """The event as JSON. CPU-bound (decodes the frame, encodes crops): run in a thread."""
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        body = {
            "v": 1,
            "source": "immicheye",
            "id": str(uuid.uuid4()),
            "ts": entry["ts"],
            "camera": camera,
            "file": entry.get("file"),
            "size": entry.get("size"),
            "model": self.s.model,
            "threshold": self.s.match_threshold,
            "margin": self.s.match_margin,
            "faces": [{
                "box": f["box"],
                "det": f["det"],
                "embedding": base64.b64encode(np.asarray(f["embedding"], dtype="<f4").tobytes()).decode(),
                "result": f["result"],
                "score": f["score"],
                "top": f["top"],
                "crop": _jpeg_b64(img.crop(context_box(f["box"], img.width, img.height)), CROP_MAX_PX),
            } for f in faces],
        }
        if self.s.events_frame_px > 0:
            body["frame"] = _jpeg_b64(img, self.s.events_frame_px)
        return json.dumps(body).encode()

    async def _send(self, item) -> None:
        headers = {"Content-Type": "application/json"}
        if self.s.events_token:
            headers["Authorization"] = f"Bearer {self.s.events_token}"
        try:
            data = await asyncio.to_thread(self.payload, *item)
            r = await self.client.post(self.s.events_url, content=data, headers=headers, timeout=5)
            if r.status_code >= 300:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            self.sent += 1
        except Exception as e:
            self.failed += 1
            self.last_error = f"{type(e).__name__}: {e}"
            if time.monotonic() - self._warned > WARN_EVERY_S:
                self._warned = time.monotonic()
                log.warning("face event not delivered (%d so far): %s", self.failed, self.last_error)

    async def run(self) -> None:
        while True:
            item = await self.queue.get()
            self.queued_bytes -= len(item[0])
            try:
                await self._send(item)
            finally:
                self.pending -= 1

    def status(self) -> dict:
        return {"url_set": self.enabled, "sent": self.sent, "failed": self.failed,
                "dropped": self.dropped, "last_error": self.last_error}
