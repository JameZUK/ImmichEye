"""Settings, read once from the environment (see .env.example)."""
import json
import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Settings:
    immich_url: str
    immich_api_key: str
    ml_url: str
    cpai_url: str
    model: str
    data_dir: Path
    # Gallery
    people: tuple[str, ...]          # Immich names to include; empty = every named person
    name_map: dict[str, str]         # Immich name -> Blue Iris label
    assets_per_person: int           # most recent photos searched per person
    faces_per_person: int            # best faces kept from those
    min_face_px: int                 # smallest face (short side, in Immich's frame) worth using
    refresh_minutes: int
    # Matching
    detect_min_score: float
    match_threshold: float           # cosine similarity a person's score must reach
    match_margin: float              # and beat the runner-up by
    top_k: int                       # person score = mean of their k best similarities
    log_recognitions: bool = True
    cpai_fallback: bool = False      # on failure, let CodeProject.AI answer (most of its names are wrong)
    cpai_sync: str = "managed"       # off | managed | exclusive: see cpai_sync.py
    cpai_prototypes: int = 4         # CodeProject.AI userids per person (label1..labelN)
    save_faces: int = 0              # keep crops of the last N recognised-or-not faces (0 = off)


def load() -> Settings:
    sync = _env("IMMICHEYE_CPAI_SYNC", "managed").lower()
    if sync not in ("off", "managed", "exclusive"):
        raise ValueError(f"IMMICHEYE_CPAI_SYNC must be off, managed or exclusive, not {sync!r}")
    people = tuple(p.strip() for p in _env("IMMICHEYE_PEOPLE", "").split(",") if p.strip())
    name_map = json.loads(_env("IMMICHEYE_NAME_MAP", "{}") or "{}")
    return Settings(
        immich_url=_env("IMMICH_URL", "http://localhost:2283").rstrip("/"),
        immich_api_key=_env("IMMICH_API_KEY", ""),
        ml_url=_env("IMMICH_ML_URL", "http://localhost:3003").rstrip("/"),
        cpai_url=_env("CPAI_URL", "http://localhost:32168").rstrip("/"),
        model=_env("IMMICHEYE_MODEL", "buffalo_l"),
        data_dir=Path(_env("IMMICHEYE_DATA", "/data")),
        people=people,
        name_map=name_map,
        assets_per_person=int(_env("IMMICHEYE_ASSETS_PER_PERSON", "250")),
        faces_per_person=int(_env("IMMICHEYE_FACES_PER_PERSON", "40")),
        min_face_px=int(_env("IMMICHEYE_MIN_FACE_PX", "60")),
        refresh_minutes=int(_env("IMMICHEYE_REFRESH_MINUTES", "360")),
        detect_min_score=float(_env("IMMICHEYE_DETECT_MIN_SCORE", "0.5")),
        match_threshold=float(_env("IMMICHEYE_MATCH_THRESHOLD", "0.40")),
        match_margin=float(_env("IMMICHEYE_MATCH_MARGIN", "0.05")),
        top_k=int(_env("IMMICHEYE_TOP_K", "3")),
        log_recognitions=_env("IMMICHEYE_LOG_RECOGNITIONS", "1") not in ("0", "false", "no"),
        cpai_fallback=_env("IMMICHEYE_CPAI_FALLBACK", "0") not in ("0", "false", "no"),
        cpai_sync=sync,
        cpai_prototypes=int(_env("IMMICHEYE_CPAI_PROTOTYPES", "4")),
        save_faces=int(_env("IMMICHEYE_SAVE_FACES", "0")),
    )
