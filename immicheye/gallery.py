"""The face gallery: reference embeddings for each named Immich person, and matching against it.

Each person's reference set is their largest faces among their most recent photos, so it
tracks how people look now and favours faces with enough pixels to embed well. Embeddings
are cached by Immich face id, so a refresh only embeds faces it hasn't seen.
"""
import asyncio
import io
import json
import logging
import re
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .config import Settings
from .immich import FaceRef, Immich, Person
from .ml import ImmichML

log = logging.getLogger(__name__)

CROP_MARGIN = 0.6   # context added around Immich's box (fraction of box size, each side)


def labels_for(people: list[Person], name_map: dict[str, str], style: str = "first") -> dict[str, str]:
    """Immich name -> Blue Iris label, lower case with underscores between words.

    style "first": the first name, or the full name where first names clash.
    style "full": always the full name ("Cal Smith (Dad)" -> cal_smith_dad).

    Labels never end in a digit: Blue Iris and Caesa treat trailing digits as a
    registration number ("cal1" is cal).
    """
    def slug(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_").rstrip("0123456789_") or "person"

    # Clashes are between distinct names: unmerged duplicates of one name aren't a clash.
    firsts: dict[str, set[str]] = {}
    for p in people:
        firsts.setdefault(slug(p.name.split()[0]), set()).add(slug(p.name))
    out = {}
    for p in people:
        if p.name in name_map:
            out[p.id] = name_map[p.name]
        else:
            first = slug(p.name.split()[0])
            out[p.id] = first if style == "first" and len(firsts[first]) == 1 else slug(p.name)
    return out


@dataclass
class Match:
    label: str
    score: float


@dataclass
class Gallery:
    labels: list[str] = field(default_factory=list)       # one per person
    owners: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    embeddings: np.ndarray = field(default_factory=lambda: np.zeros((0, 512), dtype=np.float32))
    built_at: float = 0.0
    faces_per_label: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.labels)

    def match(self, q: np.ndarray, top_k: int) -> list[Match]:
        """Every person scored against embedding q, best first."""
        if not len(self.owners):
            return []
        sims = self.embeddings @ q
        out = []
        for i, label in enumerate(self.labels):
            s = sims[self.owners == i]
            if len(s):
                k = min(top_k, len(s))
                out.append(Match(label, float(np.mean(np.partition(s, -k)[-k:]))))
        return sorted(out, key=lambda m: m.score, reverse=True)


class GalleryBuilder:
    def __init__(self, settings: Settings, immich: Immich, ml: ImmichML):
        self.s = settings
        self.immich = immich
        self.ml = ml
        self.cache_path = settings.data_dir / "embeddings.json"
        self.crops = settings.data_dir / "crops"
        self.cache = self._load_cache()
        self.sem = asyncio.Semaphore(8)

    def _load_cache(self) -> dict:
        try:
            c = json.loads(self.cache_path.read_text())
            if c.get("model") == self.s.model:
                return c["faces"]
            log.info("model changed (%s -> %s): re-embedding everything", c.get("model"), self.s.model)
        except FileNotFoundError:
            pass
        except Exception:
            log.exception("embedding cache unreadable; starting fresh")
        return {}

    def _save_cache(self) -> None:
        self.s.data_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"model": self.s.model, "faces": self.cache}))
        tmp.replace(self.cache_path)

    async def _faces_of(self, person: Person) -> list[FaceRef]:
        assets = await self.immich.recent_assets(person.id, self.s.assets_per_person)

        async def one(asset_id: str) -> list[FaceRef]:
            async with self.sem:
                return [f for f in await self.immich.faces(asset_id) if f.person_id == person.id]

        faces = [f for fs in await asyncio.gather(*map(one, assets)) for f in fs]
        faces = [f for f in faces if f.short_side >= self.s.min_face_px]
        faces.sort(key=lambda f: f.short_side, reverse=True)
        return faces[: self.s.faces_per_person]

    async def _embed(self, face: FaceRef) -> list[float] | None:
        async with self.sem:
            jpeg = await self.immich.preview(face.asset_id)
            img = Image.open(io.BytesIO(jpeg))
            img.load()
            sx, sy = img.width / face.image_width, img.height / face.image_height
            x1, y1, x2, y2 = face.x1 * sx, face.y1 * sy, face.x2 * sx, face.y2 * sy
            mx, my = (x2 - x1) * CROP_MARGIN, (y2 - y1) * CROP_MARGIN
            crop = img.convert("RGB").crop((
                max(0, int(x1 - mx)), max(0, int(y1 - my)),
                min(img.width, int(x2 + mx)), min(img.height, int(y2 + my)),
            ))
            buf = io.BytesIO()
            crop.save(buf, "JPEG", quality=92)
            found = await self.ml.faces(buf.getvalue(), self.s.detect_min_score)
        if not found:
            return None
        # Kept for registering with CodeProject.AI (cpai_sync.py)
        self.crops.mkdir(parents=True, exist_ok=True)
        (self.crops / f"{face.id}.jpg").write_bytes(buf.getvalue())
        # The crop is centred on one face; take the biggest in case a neighbour is in shot.
        best = max(found, key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]))
        return [round(float(x), 5) for x in best.embedding]

    async def build(self) -> Gallery:
        t0 = time.monotonic()
        people = await self.immich.people()
        if self.s.people:
            wanted = {n.lower() for n in self.s.people}
            people = [p for p in people if p.name.lower() in wanted]
        labels = labels_for(people, self.s.name_map, self.s.label_style)

        # The same label can cover several Immich people (e.g. an unmerged duplicate);
        # they pool their faces.
        seen, new = set(), 0
        for person in people:
            try:
                faces = await self._faces_of(person)
            except Exception:
                log.exception("listing faces for %s failed; keeping cached faces", person.name)
                faces = [FaceRef(fid, c["asset_id"], person.id, 0, 0, 0, 0, 1, 1)
                         for fid, c in self.cache.items() if c["person_id"] == person.id]
            todo = [f for f in faces
                    if self.cache.get(f.id, {}).get("person_id") != person.id
                    or (self.cache[f.id]["embedding"] and not (self.crops / f"{f.id}.jpg").exists())]

            async def embed(f: FaceRef):
                try:
                    return f, await self._embed(f)
                except Exception as e:
                    log.warning("embedding face %s (%s) failed: %r", f.id, person.name, e)
                    return f, "error"

            for f, emb in await asyncio.gather(*map(embed, todo)):
                if emb != "error":   # errors are retried next refresh; "no face" (None) is remembered
                    self.cache[f.id] = {"person_id": person.id, "asset_id": f.asset_id, "embedding": emb}
                    new += 1
            for f in faces:
                if f.id in self.cache:
                    self.cache[f.id]["label"] = labels[person.id]
                    seen.add(f.id)

        # Drop faces no longer selected (renamed, reassigned, deleted or pushed out by newer photos).
        self.cache = {k: v for k, v in self.cache.items() if k in seen}
        self._save_cache()
        if self.crops.exists():
            for crop in self.crops.glob("*.jpg"):
                if crop.stem not in self.cache:
                    crop.unlink()
        g = self.from_cache()
        log.info("gallery: %d people, %d faces (%d newly embedded) in %.0fs",
                 len(g), len(g.owners), new, time.monotonic() - t0)
        return g

    def from_cache(self) -> Gallery:
        """The gallery as of the last build, without touching Immich: used at startup."""
        entries = [c for c in self.cache.values() if c.get("embedding") and c.get("label")]
        labels = sorted({c["label"] for c in entries})
        index = {label: i for i, label in enumerate(labels)}
        g = Gallery(
            labels=labels,
            owners=np.asarray([index[c["label"]] for c in entries], dtype=np.int32),
            embeddings=np.asarray([c["embedding"] for c in entries], dtype=np.float32).reshape(-1, 512),
            built_at=time.time(),
        )
        for c in entries:
            g.faces_per_label[c["label"]] = g.faces_per_label.get(c["label"], 0) + 1
        return g
