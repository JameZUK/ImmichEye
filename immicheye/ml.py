"""Client for the Immich machine-learning server's /predict endpoint.

This is the same detector + embedding model (InsightFace buffalo_l by default)
Immich runs on its own library, so embeddings are directly comparable.
"""
import json
from dataclasses import dataclass

import httpx
import numpy as np


@dataclass
class Face:
    box: tuple[int, int, int, int]   # x1, y1, x2, y2 in the input image's pixels
    score: float                     # detector confidence
    embedding: np.ndarray            # L2-normalised


class ImmichML:
    def __init__(self, url: str, model: str, client: httpx.AsyncClient):
        self.url = url
        self.model = model
        self.client = client

    async def faces(self, jpeg: bytes, min_score: float) -> list[Face]:
        entries = {
            "facial-recognition": {
                "detection": {"modelName": self.model, "options": {"minScore": min_score}},
                "recognition": {"modelName": self.model},
            }
        }
        r = await self.client.post(
            f"{self.url}/predict",
            data={"entries": json.dumps(entries)},
            files={"image": ("image.jpg", jpeg, "image/jpeg")},
            timeout=30,
        )
        r.raise_for_status()
        out = []
        for f in r.json().get("facial-recognition", []):
            emb = f["embedding"]
            if isinstance(emb, str):
                emb = json.loads(emb)
            v = np.asarray(emb, dtype=np.float32)
            v /= np.linalg.norm(v) or 1.0
            b = f["boundingBox"]
            out.append(Face((int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])), float(f["score"]), v))
        return out
