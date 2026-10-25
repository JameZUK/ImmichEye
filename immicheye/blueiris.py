"""Late names: add a name to a Blue Iris alert's memo when it turns up after the alert.

Blue Iris writes an alert's memo once, from the image it picked, when it confirms the alert.
People often face the camera only afterwards, in the frames Blue Iris keeps sending while
the camera is triggered. With IMMICHEYE_BLUEIRIS_URL, _USER and _PASSWORD set, ImmichEye
remembers each camera's latest alert and, when a confident name turns up in that alert's
images or in the camera's frames within IMMICHEYE_LATE_NAMES_S of it, and the memo lacks
it, adds it through Blue Iris's JSON API (`update` with the alert's path and a new memo).

Names go after the object labels and replace "unknown". Blue Iris keeps only the first 50
characters of a memo, so other labels are dropped from the end to make room. The push
notification Blue Iris already sent is not changed; the alert in Blue Iris and in apps that
read its memo is.
"""
import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx

from .config import Settings

log = logging.getLogger(__name__)

MEMO_MAX = 50
SETTLE_S = 6.0    # first update no sooner than this after the alert's first image: Blue Iris writes its own memo first
DEBOUNCE_S = 2.0  # gather names arriving together into one update
RETRIES = (3, 6, 12)   # seconds to wait for an alert Blue Iris hasn't listed yet
KEEP_FLAGS = 2
STAMP = re.compile(r"\.(\d{8}_\d{6})Z\.", re.I)


def alert_time(filename: str) -> float | None:
    m = STAMP.search(filename)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc).timestamp()


def merge_memo(memo: str, names: dict[str, float]) -> str:
    """memo with names (label -> confidence 0..1) added after the object labels, replacing
    "unknown", cut to MEMO_MAX by dropping other labels from the end."""
    items = [i.strip() for i in (memo or "").split(",") if i.strip()]
    have = {i.rsplit(":", 1)[0].lower() for i in items}
    new = [f"{label}:{round(conf * 100)}%" for label, conf in names.items() if label.lower() not in have]
    if not new:
        return memo
    items = [i for i in items if i.split(":")[0].strip().lower() != "unknown"]
    first, rest = items[:1], items[1:]   # Blue Iris puts the confirming label (person) first
    out = first + new + rest
    while len(",".join(out)) > MEMO_MAX and len(out) > len(first) + len(new):
        out.pop()
    while len(",".join(out)) > MEMO_MAX and len(out) > 1:
        out.pop()
    return ",".join(out)[:MEMO_MAX]


class BlueIris:
    def __init__(self, url: str, user: str, password: str, client: httpx.AsyncClient):
        self.url, self.user, self.password, self.client = url.rstrip("/") + "/json", user, password, client

    async def _call(self, body: dict) -> dict:
        r = await self.client.post(self.url, json=body, timeout=10)
        r.raise_for_status()
        return r.json()

    async def session(self) -> str:
        s = (await self._call({"cmd": "login"}))["session"]
        response = hashlib.md5(f"{self.user}:{s}:{self.password}".encode("latin-1")).hexdigest()
        r = await self._call({"cmd": "login", "session": s, "response": response})
        if r.get("result") != "success":
            raise RuntimeError(f"login failed: {r.get('data', {}).get('reason', r)}")
        return r["session"]

    async def find_alert(self, s: str, camera: str, filename: str) -> dict | None:
        t = alert_time(filename) or time.time()
        r = await self._call({"cmd": "alertlist", "camera": camera, "startdate": int(t) - 60,
                              "enddate": int(t) + 600, "session": s})
        for a in r.get("data") or []:
            if (a.get("file") or "").lower() == filename.lower():
                return a
        return None

    async def set_memo(self, s: str, alert: dict, memo: str) -> None:
        # A memo-only update clears flag bit 2 (set on nearly every alert); send it back unchanged.
        keep = int(alert.get("flags", 0)) & KEEP_FLAGS
        r = await self._call({"cmd": "update", "path": alert["path"], "memo": memo,
                              "flags": keep, "mask": KEEP_FLAGS, "session": s})
        if r.get("result") != "success":
            raise RuntimeError(f"update refused: {r}")

    async def logout(self, s: str) -> None:
        try:
            await self._call({"cmd": "logout", "session": s})
        except Exception:
            pass


@dataclass
class _Alert:
    camera: str
    file: str
    first: float                                          # monotonic time of its first image
    names: dict[str, float] = field(default_factory=dict)  # every confident name seen
    written: set[str] = field(default_factory=set)        # names already in the memo
    task: asyncio.Task | None = None


class LateNames:
    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.s = settings
        self.bi = BlueIris(settings.blueiris_url, settings.blueiris_user, settings.blueiris_password, client)
        self.latest: dict[str, _Alert] = {}   # camera -> its latest alert
        self.updated = self.failed = 0
        self.last_error: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.s.blueiris_url and self.s.blueiris_user)

    def observe(self, camera: str | None, alert: str | None, filename: str | None, predictions: list[dict]) -> None:
        """One reply: alert images open or feed their alert, other frames feed the camera's latest."""
        if not self.enabled or not camera:
            return
        key = camera.lower()
        now = time.monotonic()
        a = self.latest.get(key)
        if alert and (a is None or a.file.lower() != alert):
            a = self.latest[key] = _Alert(camera, filename, now)
        if a is None or now - a.first > self.s.late_names_s:
            return
        fresh = False
        for p in predictions:
            if p["userid"] != "unknown" and p["confidence"] > a.names.get(p["userid"], 0):
                fresh |= p["userid"] not in a.names
                a.names[p["userid"]] = p["confidence"]
        if fresh and (a.task is None or a.task.done()):
            a.task = asyncio.create_task(self._flush(a))

    async def _flush(self, a: _Alert) -> None:
        await asyncio.sleep(max(DEBOUNCE_S, a.first + SETTLE_S - time.monotonic()))
        if not set(a.names) - a.written:
            return
        s = None
        try:
            s = await self.bi.session()
            found = await self.bi.find_alert(s, a.camera, a.file)
            for wait in RETRIES:
                if found is not None:
                    break
                await asyncio.sleep(wait)
                found = await self.bi.find_alert(s, a.camera, a.file)
            if found is None:
                return   # cancelled (no alert to name), or never listed
            memo = found.get("memo") or ""
            new = merge_memo(memo, {n: c for n, c in a.names.items() if n not in a.written})
            if new != memo:
                await self.bi.set_memo(s, found, new)
                self.updated += 1
                log.info("%s alert %s memo: %r -> %r", a.camera, found["path"], memo, new)
            a.written |= set(a.names)
        except Exception as e:
            self.failed += 1
            self.last_error = f"{type(e).__name__}: {e}"
            log.warning("could not add late names to %s: %s", a.file, self.last_error)
        finally:
            if s:
                await self.bi.logout(s)

    def status(self) -> dict:
        return {"enabled": self.enabled, "updated": self.updated, "failed": self.failed, "last_error": self.last_error}
