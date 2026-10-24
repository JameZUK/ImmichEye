"""Measure recognition accuracy against your own library, to choose the thresholds.

Held-out test faces are each person's small faces (default 30-60 px short side), which the
gallery skips (IMMICHEYE_MIN_FACE_PX) and which look like CCTV crops. Each one is cropped
from its preview, downscaled further if asked, and matched against the gallery.

usage (same environment as the service, gallery already built):
    python scripts/evaluate.py [--per-person 15] [--min-px 30] [--max-px 60] [--scale 1.0]
"""
import argparse
import asyncio
import io
import random

import httpx
import numpy as np
from PIL import Image

from immicheye import config
from immicheye.gallery import CROP_MARGIN, GalleryBuilder, labels_for
from immicheye.immich import Immich
from immicheye.ml import ImmichML


async def main(a):
    s = config.load()
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=16)) as c:
        immich, ml = Immich(s.immich_url, s.immich_api_key, c), ImmichML(s.ml_url, s.model, c)
        builder = GalleryBuilder(s, immich, ml)
        g = builder.from_cache()
        in_gallery = set(builder.cache)
        people = await immich.people()
        if s.people:
            people = [p for p in people if p.name.lower() in {n.lower() for n in s.people}]
        labels = labels_for(people, s.name_map, s.label_style)
        sem = asyncio.Semaphore(8)
        cpai_sem = asyncio.Semaphore(2)   # a live server: don't crowd out Blue Iris
        rng = random.Random(1)

        async def samples(person):
            assets = await immich.recent_assets(person.id, s.assets_per_person)
            rng.shuffle(assets)
            out = []
            for asset in assets:
                if len(out) >= a.per_person:
                    break
                async with sem:
                    faces = await immich.faces(asset)
                out += [f for f in faces if f.person_id == person.id and f.id not in in_gallery
                        and a.min_px <= f.short_side < a.max_px]
            return person, out[: a.per_person]

        async def score(person, f):
            async with sem:
                img = Image.open(io.BytesIO(await immich.preview(f.asset_id))).convert("RGB")
                sx, sy = img.width / f.image_width, img.height / f.image_height
                w, h = (f.x2 - f.x1) * sx, (f.y2 - f.y1) * sy
                crop = img.crop((max(0, int(f.x1 * sx - w * CROP_MARGIN)), max(0, int(f.y1 * sy - h * CROP_MARGIN)),
                                 int(f.x2 * sx + w * CROP_MARGIN), int(f.y2 * sy + h * CROP_MARGIN)))
                if a.scale != 1.0:
                    crop = crop.resize((max(8, int(crop.width * a.scale)), max(8, int(crop.height * a.scale))))
                buf = io.BytesIO()
                crop.save(buf, "JPEG", quality=85)
                found = await ml.faces(buf.getvalue(), s.detect_min_score)
            cpai = None
            if a.cpai:
                try:
                    async with cpai_sem:
                        r = await c.post(f"{s.cpai_url}/v1/vision/face/recognize", data={"min_confidence": "0.4"},
                                         files={"image": ("t.jpg", buf.getvalue(), "image/jpeg")}, timeout=60)
                    preds = r.json().get("predictions") or []
                except (httpx.HTTPError, ValueError):
                    preds = []
                if preds:
                    p = max(preds, key=lambda x: (x["x_max"] - x["x_min"]) * (x["y_max"] - x["y_min"]))
                    cpai = (p["userid"].rstrip("0123456789"), p["confidence"])
            if not found:
                return labels[person.id], None, cpai
            best = max(found, key=lambda x: (x.box[2] - x.box[0]) * (x.box[3] - x.box[1]))
            return labels[person.id], g.match(best.embedding, s.top_k), cpai

        groups = await asyncio.gather(*map(samples, people))
        results = await asyncio.gather(*[score(p, f) for p, fs in groups for f in fs])

    undetected = sum(1 for _, m, _ in results if m is None)
    scored = [(truth, m) for truth, m, _ in results if m]
    print(f"{len(results)} test faces from {len(people)} people; {undetected} not detected at all")
    print(f"{'threshold':>9} {'margin':>6} {'correct':>8} {'wrong':>6} {'unknown':>8}")
    for thr in (0.30, 0.35, 0.40, 0.45, 0.50):
        for margin in (0.0, 0.05, 0.10):
            right = wrong = unknown = 0
            for truth, m in scored:
                ru = m[1].score if len(m) > 1 else -1
                if m[0].score >= thr and m[0].score - ru >= margin:
                    right += m[0].label == truth
                    wrong += m[0].label != truth
                else:
                    unknown += 1
            n = len(scored) or 1
            print(f"{thr:9.2f} {margin:6.2f} {right / n:8.1%} {wrong / n:6.1%} {unknown / n:8.1%}")
    true_scores = [next((x.score for x in m if x.label == truth), np.nan) for truth, m in scored]
    other_best = [max((x.score for x in m if x.label != truth), default=np.nan) for truth, m in scored]
    print(f"same-person score  median {np.nanmedian(true_scores):.3f}  p10 {np.nanpercentile(true_scores, 10):.3f}")
    print(f"best-other score   median {np.nanmedian(other_best):.3f}  p90 {np.nanpercentile(other_best, 90):.3f}")
    if a.cpai:
        cp = [(truth, x) for truth, _, x in results]
        print(f"\nCodeProject.AI on the same faces ({sum(x is None for _, x in cp)} not detected)")
        print(f"{'min_conf':>9} {'correct':>8} {'wrong':>6} {'unknown':>8}")
        for thr in (0.40, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
            right = sum(1 for t, x in cp if x and x[1] >= thr and x[0] == t)
            wrong = sum(1 for t, x in cp if x and x[1] >= thr and x[0] != t)
            n = len(cp) or 1
            print(f"{thr:9.2f} {right / n:8.1%} {wrong / n:6.1%} {(n - right - wrong) / n:8.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-person", type=int, default=15)
    ap.add_argument("--min-px", type=int, default=30)
    ap.add_argument("--max-px", type=int, default=60)
    ap.add_argument("--scale", type=float, default=1.0, help="downscale crops further, e.g. 0.5")
    ap.add_argument("--cpai", action="store_true", help="also score CodeProject.AI (after a sync) on the same faces")
    asyncio.run(main(ap.parse_args()))
