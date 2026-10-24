"""The few Immich server API calls the gallery needs (checked against Immich 3.3 /api/spec.json).

API key permissions required: person.read, asset.read, face.read, asset.view.
"""
from dataclasses import dataclass

import httpx


@dataclass
class Person:
    id: str
    name: str


@dataclass
class FaceRef:
    id: str
    asset_id: str
    person_id: str
    # Bounding box in the frame Immich detected on (imageWidth x imageHeight)
    x1: int
    y1: int
    x2: int
    y2: int
    image_width: int
    image_height: int

    @property
    def short_side(self) -> int:
        return min(self.x2 - self.x1, self.y2 - self.y1)


class Immich:
    def __init__(self, url: str, api_key: str, client: httpx.AsyncClient):
        self.url = f"{url}/api"
        self.client = client
        self.headers = {"x-api-key": api_key}

    async def _get(self, path: str, **params):
        r = await self.client.get(f"{self.url}{path}", params=params, headers=self.headers, timeout=30)
        r.raise_for_status()
        return r

    async def people(self) -> list[Person]:
        out, page = [], 1
        while True:
            body = (await self._get("/people", withHidden="false", page=page, size=500)).json()
            out += [Person(p["id"], p["name"]) for p in body["people"] if p.get("name") and not p.get("isHidden")]
            if not body.get("hasNextPage"):
                return out
            page += 1

    async def recent_assets(self, person_id: str, limit: int) -> list[str]:
        """Ids of the person's most recent photos (newest first)."""
        ids, page = [], 1
        while len(ids) < limit:
            size = min(250, limit - len(ids))
            r = await self.client.post(
                f"{self.url}/search/metadata",
                json={"personIds": [person_id], "type": "IMAGE", "order": "desc", "size": size, "page": page},
                headers=self.headers,
                timeout=30,
            )
            r.raise_for_status()
            assets = r.json()["assets"]
            ids += [a["id"] for a in assets["items"]]
            if not assets.get("nextPage") or not assets["items"]:
                break
            page = int(assets["nextPage"])
        return ids[:limit]

    async def faces(self, asset_id: str) -> list[FaceRef]:
        out = []
        for f in (await self._get("/faces", id=asset_id)).json():
            person = f.get("person")
            if not person:
                continue
            out.append(FaceRef(
                f["id"], asset_id, person["id"],
                f["boundingBoxX1"], f["boundingBoxY1"], f["boundingBoxX2"], f["boundingBoxY2"],
                f["imageWidth"], f["imageHeight"],
            ))
        return out

    async def preview(self, asset_id: str) -> bytes:
        return (await self._get(f"/assets/{asset_id}/thumbnail", size="preview")).content
