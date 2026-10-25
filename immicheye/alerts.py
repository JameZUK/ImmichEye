"""Names shared across the images of one Blue Iris alert (IMMICHEYE_ALERT_SHARE).

Blue Iris analyses an alert by sending several images that all carry the alert's file name
(Front.20260101_120000Z.1234.3-1.jpg), then lists in the alert's memo only the faces of the
one image it settles on. If that image shows a back or a blur, a name found in the image
before or after it is lost. With sharing on, every reply in the alert includes the names
confidently matched in any of its images so far, and a reply that would carry no name waits
for another image to find one, until IMMICHEYE_ALERT_HOLD_MS after the alert's first image.

Blue Iris confirms an alert only when its replies are in. The deadline is per alert, not
per image, so an alert where nobody is named is delayed by at most what is left of the hold
when its last image arrives; Blue Iris spends about that long sending them anyway.

Names come only from confident matches in the same alert, a few seconds apart; a shared
name keeps the box of the image it was found in.
"""
import asyncio
import re
import time
from dataclasses import dataclass, field

ALERT_FILE = re.compile(r"^[^.]+\.\d{8}_\d{6}Z\.\d+\.\d+-\d+\.jpg$", re.I)
KEEP_S = 120   # forget an alert this long after its first image


def alert_key(filename: str | None) -> str | None:
    """The alert an image belongs to, or None for frames outside alert analysis."""
    return filename.lower() if filename and ALERT_FILE.match(filename) else None


@dataclass
class _Alert:
    first: float
    names: dict[str, dict] = field(default_factory=dict)   # label -> best prediction
    changed: asyncio.Condition = field(default_factory=asyncio.Condition)


class AlertNames:
    def __init__(self, hold_s: float):
        self.hold_s = hold_s
        self.alerts: dict[str, _Alert] = {}

    def _get(self, key: str) -> _Alert:
        now = time.monotonic()
        for k in [k for k, a in self.alerts.items() if now - a.first > KEEP_S]:
            del self.alerts[k]
        return self.alerts.setdefault(key, _Alert(now))

    async def share(self, key: str, predictions: list[dict]) -> tuple[list[dict], list[str], int]:
        """predictions plus the alert's other names; the labels added; ms spent waiting."""
        a = self._get(key)
        known = [p for p in predictions if p["userid"] != "unknown"]
        async with a.changed:
            for p in known:
                if p["confidence"] > a.names.get(p["userid"], {}).get("confidence", -1):
                    a.names[p["userid"]] = p
            if known:
                a.changed.notify_all()
        waited = 0
        left = a.first + self.hold_s - time.monotonic()
        if not known and left > 0:
            t = time.monotonic()
            try:
                async with a.changed:
                    await asyncio.wait_for(a.changed.wait_for(lambda: a.names), left)
            except asyncio.TimeoutError:
                pass
            waited = int((time.monotonic() - t) * 1000)
        own = {p["userid"] for p in known}
        added = [p for label, p in a.names.items() if label not in own]
        return predictions + added, [p["userid"] for p in added], waited
