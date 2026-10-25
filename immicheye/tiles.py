"""Face detection on large frames by tiling.

Immich ML's detector letterboxes every image to 640 px before looking for faces, so in a
4K frame a face shrinks about six times and anything under about 80 px is lost. Large
frames are also searched in overlapping tiles of IMMICHEYE_TILE_PX, which keeps faces
down to about 32 px findable. The whole frame is searched too, for faces bigger than
the overlap, and duplicates are merged.
"""
import asyncio
import io

from PIL import Image

from .ml import Face, ImmichML

EDGE = 4   # a face box this close to a tile's inner edge is cut off; another tile has it whole
MAX_TILES = 24   # 4K takes 8 tiles of 1280; anything needing more than this is searched whole only


def tile_boxes(width: int, height: int, tile: int) -> list[tuple[int, int, int, int]]:
    """Overlapping tiles covering the frame; none when it already fits in one, or would need
    more than MAX_TILES (an oversized upload shouldn't queue dozens of Immich ML calls)."""
    if tile <= 0 or max(width, height) <= tile:
        return []
    overlap = tile // 4

    def starts(n: int) -> list[int]:
        if n <= tile:
            return [0]
        return list(range(0, n - tile, tile - overlap)) + [n - tile]

    boxes = [(x, y, min(width, x + tile), min(height, y + tile)) for y in starts(height) for x in starts(width)]
    return boxes if len(boxes) <= MAX_TILES else []


def _crops(jpeg: bytes, boxes: list[tuple[int, int, int, int]]) -> list[bytes]:
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    out = []
    for box in boxes:
        buf = io.BytesIO()
        img.crop(box).save(buf, "JPEG", quality=92)
        out.append(buf.getvalue())
    return out


def _overlap(a: tuple, b: tuple) -> float:
    """Intersection over the smaller box: 1.0 when one box sits inside the other."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h / max(smaller, 1)


def merge(whole: list[Face], tiled: list[tuple[tuple[int, int, int, int], list[Face]]],
          width: int, height: int) -> list[Face]:
    """Whole-frame faces plus tile faces moved into frame coordinates, without cut-off
    faces at inner tile edges or the same face found twice (the most confident is kept)."""
    found = list(whole)
    for (tx1, ty1, tx2, ty2), faces in tiled:
        for f in faces:
            x1, y1, x2, y2 = f.box[0] + tx1, f.box[1] + ty1, f.box[2] + tx1, f.box[3] + ty1
            if ((tx1 > 0 and x1 - tx1 < EDGE) or (ty1 > 0 and y1 - ty1 < EDGE)
                    or (tx2 < width and tx2 - x2 < EDGE) or (ty2 < height and ty2 - y2 < EDGE)):
                continue
            found.append(Face((x1, y1, x2, y2), f.score, f.embedding))
    kept: list[Face] = []
    for f in sorted(found, key=lambda f: f.score, reverse=True):
        if all(_overlap(f.box, k.box) < 0.5 for k in kept):
            kept.append(f)
    return kept


async def detect(ml: ImmichML, jpeg: bytes, min_score: float, tile: int,
                 size: tuple[int, int] | None) -> tuple[list[Face], int]:
    """Faces in the frame, and how many tiles were searched besides the whole frame."""
    boxes = tile_boxes(*size, tile) if size else []
    if not boxes:
        return await ml.faces(jpeg, min_score), 0
    crops = await asyncio.to_thread(_crops, jpeg, boxes)
    whole, *parts = await asyncio.gather(ml.faces(jpeg, min_score), *(ml.faces(c, min_score) for c in crops))
    return merge(whole, list(zip(boxes, parts)), *size), len(boxes)
