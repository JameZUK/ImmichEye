"""Late names: add a name to a Blue Iris alert's memo when it turns up after the alert.

Blue Iris writes an alert's memo once, from the image it picked, when it confirms the alert.
People often face the camera only afterwards, in the frames Blue Iris keeps sending while
the camera is triggered. With IMMICHEYE_BLUEIRIS_URL, _USER and _PASSWORD set, ImmichEye
remembers each camera's latest alert and, when a confident name turns up in that alert's
images or in the camera's frames within IMMICHEYE_LATE_NAMES_S of it, and the memo lacks
it, adds it through Blue Iris's JSON API (`update` with the alert's path and a new memo).

Updates wait until Blue Iris has confirmed the alert, as it writes its own memo then, over
anything written earlier; cancelled alerts are left alone. Names go after the first label (person), ahead of "unknown" and other labels, with a
percentage like Blue Iris's own labels: apps that read the memo (Caesa) skip labels without
one. Blue Iris keeps only the first 49 characters of a memo, its own included, so other
labels, then "unknown", then the lowest-scoring names, then person, are left out to make room.
The push
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

MEMO_MAX = 49     # Blue Iris cuts longer memos, its own included
SETTLE_S = 6.0    # first update no sooner than this after the alert's first image: Blue Iris writes its own memo first
DEBOUNCE_S = 2.0  # gather names arriving together into one update
RETRIES = (3, 6, 12, 24)   # seconds to wait for Blue Iris to list, then confirm or cancel, the alert
KEEP_FLAGS = 2            # FLAGGED: a memo-only update clears it
CONFIRMED, CANCELLED, OCCUPIED = 1 << 28, 1 << 23, 1 << 29
PLAIN_HOLD_S = 3.0     # names from frames outside an alert wait this long to see if a new alert starts
RETRY_AFTER_S = 10.0   # after a failed or undecided update, names seen again retry no sooner than this
# Labels Blue Iris's object models give (COCO, plus its own); anything else in a memo is a face.
OBJECTS = {
    "person", "bicycle", "car", "motorcycle", "motorbike", "airplane", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "vehicle", "animal", "deer",
    "rabbit", "raccoon", "fox", "coyote", "possum", "skunk", "squirrel", "pig", "package", "parcel",
    "amazon", "fedex", "ups", "usps", "uhaul", "dhl", "plate", "dayplate", "nightplate", "unknown",
}
STAMP = re.compile(r"\.(\d{8}_\d{6})Z\.", re.I)


def alert_time(filename: str) -> float | None:
    m = STAMP.search(filename)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc).timestamp()


def merge_memo(memo: str, names: dict[str, float]) -> str:
    """memo with names (label -> confidence 0..1) after its first label and any names already
    there, highest score first. To fit MEMO_MAX, other labels go first, then "unknown", then
    the lowest-scoring names, and person only after all of those."""
    items = [i.strip() for i in (memo or "").split(",") if i.strip()]
    if len(memo or "") >= MEMO_MAX and items and not re.search(r":\d+%$", items[-1]) \
            and items[-1].lower() != "unknown":
        items.pop()   # Blue Iris cut the memo inside its last label
    have = {i.rsplit(":", 1)[0].strip().lower() for i in items}
    new = [f"{label}:{round(conf * 100)}%" for label, conf in sorted(names.items(), key=lambda n: -n[1])
           if label.lower() not in have]
    if not new:
        return memo
    def label(i):
        return i.split(":")[0].strip().lower()
    first, rest = items[:1], items[1:]   # Blue Iris puts the confirming label (person) first
    faces = [i for i in rest if label(i) not in OBJECTS]
    unknown = [i for i in rest if label(i) == "unknown"]
    others = [i for i in rest if i not in faces and i not in unknown]
    out = first + faces + new + unknown + others
    # Order kept; what goes to make room, first to last. Person outlasts the names: rules
    # and apps filter on it.
    person = [i for i in others if label(i) == "person"]
    drop = ([i for i in others if i not in person][::-1] + unknown + new[::-1] + faces[::-1] + person)
    for i in drop:
        if len(",".join(out)) <= MEMO_MAX:
            break
        out.remove(i)
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
    retry_at: float = 0.0     # monotonic time before which a failed update isn't retried
    closed: bool = False      # cancelled or occupied: nothing to name


class LateNames:
    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.s = settings
        self.bi = BlueIris(settings.blueiris_url, settings.blueiris_user, settings.blueiris_password, client)
        self.latest: dict[str, _Alert] = {}   # camera -> its latest alert
        self.tasks: set[asyncio.Task] = set()   # asyncio keeps only weak references to tasks
        self.updated = self.failed = 0
        self.last_error: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.s.blueiris_url and self.s.blueiris_user)

    def observe(self, camera: str | None, alert: str | None, filename: str | None, predictions: list[dict]) -> None:
        """One reply: alert images open or feed their alert, other frames feed the camera's latest.

        A new trigger's first frames arrive a second or two before its alert images. So names
        from frames outside an alert are credited PLAIN_HOLD_S later, to whichever alert is the
        camera's latest by then, rather than to the previous alert."""
        if not self.enabled or not camera:
            return
        key, now = camera.lower(), time.monotonic()
        if alert:
            a = self.latest.get(key)
            if a is None or a.file.lower() != alert:
                a = self.latest[key] = _Alert(camera, filename, now)
            self._credit(a, predictions, now)
        elif any(p["userid"] != "unknown" for p in predictions):
            if PLAIN_HOLD_S > 0:
                asyncio.get_running_loop().call_later(PLAIN_HOLD_S, self._credit_latest, key, predictions, now)
            else:
                self._credit_latest(key, predictions, now)

    def _credit_latest(self, key: str, predictions: list[dict], seen: float) -> None:
        a = self.latest.get(key)
        # Only frames from the alert's own trigger: no older than the alert (allowing for the
        # frames just before its images) and within the late-names window.
        if a is not None and a.first - PLAIN_HOLD_S <= seen <= a.first + self.s.late_names_s:
            self._credit(a, predictions, time.monotonic())

    def _credit(self, a: _Alert, predictions: list[dict], now: float) -> None:
        if now - a.first > self.s.late_names_s + PLAIN_HOLD_S:
            return
        fresh = False
        for p in predictions:
            if p["userid"] != "unknown" and p["confidence"] > a.names.get(p["userid"], 0):
                fresh |= p["userid"] not in a.names
                a.names[p["userid"]] = p["confidence"]
        # A new name starts an update; names already seen but not yet written (the last
        # update failed, or Blue Iris hadn't decided) retry when seen again, after a pause.
        due = bool(set(a.names) - a.written) and now >= a.retry_at
        if not a.closed and (fresh or due) and (a.task is None or a.task.done()):
            a.task = asyncio.create_task(self._flush(a))
            self.tasks.add(a.task)
            a.task.add_done_callback(self.tasks.discard)

    async def _flush(self, a: _Alert) -> None:
        await asyncio.sleep(max(DEBOUNCE_S, a.first + SETTLE_S - time.monotonic()))
        # Names can arrive at any await. Go round while there are names this flush hasn't
        # tried yet: a failed or abandoned round ends the flush only if nothing new came in
        # meanwhile, and a name observed after the last check finds the task done.
        tried: set[str] = set()
        while (pending := {n for n in a.names if n not in a.written}) - tried:
            tried = pending
            try:
                if not await self._write(a):
                    a.retry_at = time.monotonic() + RETRY_AFTER_S
            except Exception as e:
                a.retry_at = time.monotonic() + RETRY_AFTER_S
                self.failed += 1
                self.last_error = f"{type(e).__name__}: {e}"
                log.warning("could not add late names to %s: %s", a.file, self.last_error)

    async def _write(self, a: _Alert) -> bool:
        """Add the alert's names to its memo; False if the alert has nothing to name."""
        names = dict(a.names)
        s = await self.bi.session()
        try:
            # Blue Iris writes its own memo when it confirms the alert, over anything
            # written before; wait until it has decided.
            found = await self.bi.find_alert(s, a.camera, a.file)
            for wait in RETRIES:
                if found is not None and int(found.get("flags", 0)) & (CONFIRMED | CANCELLED | OCCUPIED):
                    break
                await asyncio.sleep(wait)
                found = await self.bi.find_alert(s, a.camera, a.file)
            flags = int(found.get("flags", 0)) if found else 0
            if flags & (CANCELLED | OCCUPIED):
                a.closed = True
                return False   # cancelled: nothing to name, now or later
            if not flags & CONFIRMED:
                log.info("%s alert %s not confirmed after %ds; trying again if the names are seen again",
                         a.camera, a.file, sum(RETRIES))
                return False   # still undecided, or not listed yet
            memo = found.get("memo") or ""
            new = merge_memo(memo, names)   # all of them: names already in the memo are skipped
            if new != memo:
                await self.bi.set_memo(s, found, new)
                self.updated += 1
                log.info("%s alert %s memo: %r -> %r", a.camera, found["path"], memo, new)
            a.written |= set(names)   # names that didn't fit count as done too
            return True
        finally:
            await self.bi.logout(s)

    def status(self) -> dict:
        return {"enabled": self.enabled, "updated": self.updated, "failed": self.failed, "last_error": self.last_error}
