"""Keep CodeProject.AI's face database in step with the gallery, so its recognition (the
fallback when ImmichEye can't answer) knows the same people as Immich.

CodeProject.AI keeps one averaged embedding per userid, so each person is registered as a
few prototypes, label1..labelN, each the average of a cluster of similar-looking faces.
Blue Iris and Caesa read "cal3" as cal.

Modes (IMMICHEYE_CPAI_SYNC):
  off        never touch CodeProject.AI
  managed    add and update only the userids ImmichEye created; never overwrite or delete
             anyone else's (a number already taken is skipped)
  exclusive  as managed, and delete every other userid, so CodeProject.AI knows exactly
             the Immich people
"""
import hashlib
import json
import logging
import re

import httpx
import numpy as np

log = logging.getLogger(__name__)


def clusters(ids: list[str], vectors: np.ndarray, k: int, rounds: int = 20) -> list[list[str]]:
    """Deterministic spherical k-means: the same faces always give the same groups."""
    order = np.argsort(ids)
    ids, vectors = [ids[i] for i in order], vectors[order]
    k = min(k, len(ids))
    centres = [vectors[0]]
    while len(centres) < k:   # farthest-point initialisation
        sims = np.max(vectors @ np.stack(centres).T, axis=1)
        centres.append(vectors[int(np.argmin(sims))])
    centres = np.stack(centres)
    for _ in range(rounds):
        assign = np.argmax(vectors @ centres.T, axis=1)
        new = np.stack([vectors[assign == c].mean(0) if np.any(assign == c) else centres[c] for c in range(k)])
        new /= np.linalg.norm(new, axis=1, keepdims=True)
        if np.allclose(new, centres):
            break
        centres = new
    groups = [[ids[i] for i in np.flatnonzero(assign == c)] for c in range(k)]
    return sorted((g for g in groups if g), key=lambda g: (-len(g), g[0]))


class CpaiSync:
    def __init__(self, settings, client: httpx.AsyncClient):
        self.s = settings
        self.client = client
        self.state_path = settings.data_dir / "cpai_sync.json"

    def _load_state(self) -> dict[str, str]:
        try:
            return json.loads(self.state_path.read_text())
        except FileNotFoundError:
            return {}

    def _save_state(self, state: dict[str, str]) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1, sort_keys=True))
        tmp.replace(self.state_path)

    async def _call(self, command: str, **kw) -> dict:
        r = await self.client.post(f"{self.s.cpai_url}/v1/vision/face/{command}", timeout=120, **kw)
        r.raise_for_status()
        return r.json()

    def desired(self, cache: dict) -> dict[str, list[str]]:
        """label -> prototype face-id groups, from the gallery's embedding cache."""
        by_label: dict[str, list[str]] = {}
        for fid, c in cache.items():
            if c.get("embedding") and c.get("label"):
                by_label.setdefault(c["label"], []).append(fid)
        out = {}
        for label, fids in by_label.items():
            k = min(self.s.cpai_prototypes, max(1, len(fids) // 8))
            vectors = np.asarray([cache[f]["embedding"] for f in fids], dtype=np.float32)
            out[label] = clusters(fids, vectors, k)
        return out

    async def sync(self, cache: dict, crops_dir) -> None:
        if self.s.cpai_sync == "off":
            return
        managed = self._load_state()          # userid -> fingerprint of the faces registered
        existing = set((await self._call("list"))["faces"])
        others = {u.lower() for u in existing if u not in managed}

        target: dict[str, list[str]] = {}
        taken = others | {u.lower() for u in managed}
        for label, groups in sorted(self.desired(cache).items()):
            # Reuse this label's managed userids in order, then the lowest numbers nobody has.
            ours = sorted((u for u in managed if re.fullmatch(rf"{re.escape(label)}\d+", u)),
                          key=lambda u: int(u[len(label):]))
            n = 0
            for group in groups:
                if ours:
                    userid = ours.pop(0)
                else:
                    n += 1
                    while f"{label}{n}".lower() in taken:
                        n += 1
                    userid = f"{label}{n}"
                    taken.add(userid.lower())
                target[userid] = group

        added = updated = removed = 0
        for userid, group in target.items():
            fingerprint = hashlib.sha1(",".join(sorted(group)).encode()).hexdigest()
            if managed.get(userid) == fingerprint and userid in existing:
                continue
            files = [(f"image{i}", (f"{fid}.jpg", (crops_dir / f"{fid}.jpg").read_bytes(), "image/jpeg"))
                     for i, fid in enumerate(group) if (crops_dir / f"{fid}.jpg").exists()]
            if not files:
                continue
            try:
                body = await self._call("register", data={"userid": userid}, files=files)
            except Exception as e:
                log.warning("CodeProject.AI register %s failed: %s", userid, e)
                continue
            if body.get("success"):
                added += userid not in managed
                updated += userid in managed
                managed[userid] = fingerprint
            else:
                trace = (body.get("err_trace") or "").strip().splitlines()
                log.warning("CodeProject.AI register %s: %s%s", userid, body.get("error"),
                            f" ({trace[-1]})" if trace else "")

        doomed = [u for u in managed if u not in target]
        if self.s.cpai_sync == "exclusive":
            doomed += [u for u in existing if u not in managed]
        for userid in doomed:
            try:
                await self._call("delete", data={"userid": userid})
                managed.pop(userid, None)
                removed += 1
            except Exception as e:
                log.warning("CodeProject.AI delete %s failed: %s", userid, e)
        self._save_state(managed)
        log.info("CodeProject.AI sync (%s): %d added, %d updated, %d removed, %d managed",
                 self.s.cpai_sync, added, updated, removed, len(managed))
