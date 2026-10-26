"""Settings, read once from the environment (see .env.example)."""
import json
import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _flag(name: str, default: str) -> bool:
    v = _env(name, default).lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off", ""):
        return False
    raise ValueError(f"{name} must be 1 or 0, not {v!r}")


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
    label_style: str                 # "first" (cal) or "full" (cal_smith)
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
    tile_px: int = 1280              # also search frames larger than this in tiles (0 = off)
    lowres_px: tuple[int, ...] = ()  # also embed each reference face shrunk to these widths
    label_case: str = "lower"        # "lower" (cal_smith) or "title" (Cal_Smith)
    label_sep: str = "_"             # between words: "_" (cal_smith) or " " (cal smith)
    events_url: str = ""             # POST every detected face here (events.py); empty = off
    events_token: str = ""           # sent as a Bearer token
    events_cameras: tuple[str, ...] = ()   # camera short names to send; empty = all
    events_frame_px: int = 0         # also send the frame scaled to this long side (0 = off)
    events_queue_mb: int = 64        # cap on frames waiting to be sent (oldest dropped first)
    alert_share: bool = False        # replies include names found in other images of the same alert
    blueiris_url: str = ""           # Blue Iris web server, for adding late names to alert memos (blueiris.py)
    blueiris_user: str = ""          # a Blue Iris admin user: changing a memo needs admin
    blueiris_password: str = ""
    late_names_s: int = 60           # names this long after an alert's first image still get added
    alert_hold_ms: int = 3000        # nameless replies wait for a name until this long after the alert's first image


def load() -> Settings:
    sync = _env("IMMICHEYE_CPAI_SYNC", "managed").lower()
    if sync not in ("off", "managed", "exclusive"):
        raise ValueError(f"IMMICHEYE_CPAI_SYNC must be off, managed or exclusive, not {sync!r}")
    labels = _env("IMMICHEYE_LABELS", "first").lower()
    if labels not in ("first", "full"):
        raise ValueError(f"IMMICHEYE_LABELS must be first or full, not {labels!r}")
    case = _env("IMMICHEYE_LABEL_CASE", "lower").lower()
    if case not in ("lower", "title"):
        raise ValueError(f"IMMICHEYE_LABEL_CASE must be lower or title, not {case!r}")
    sep = {"_": "_", "underscore": "_", "space": " "}.get(_env("IMMICHEYE_LABEL_SEPARATOR", "_").lower())
    if sep is None:
        raise ValueError("IMMICHEYE_LABEL_SEPARATOR must be underscore or space")
    fallback = _env("IMMICHEYE_CPAI_FALLBACK", "0")
    if fallback not in ("0", "1"):   # the nginx router reads it too, and knows only 0 and 1
        raise ValueError(f"IMMICHEYE_CPAI_FALLBACK must be 0 or 1, not {fallback!r}")
    bi_password = os.environ.get("IMMICHEYE_BLUEIRIS_PASSWORD", "")
    try:
        bi_password.encode("latin-1")   # Blue Iris's login hashes the password as Latin-1
    except UnicodeEncodeError:
        raise ValueError("IMMICHEYE_BLUEIRIS_PASSWORD has characters Blue Iris's login can't use "
                         "(outside Latin-1); choose a different password") from None
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
        label_style=labels,
        assets_per_person=int(_env("IMMICHEYE_ASSETS_PER_PERSON", "250")),
        faces_per_person=int(_env("IMMICHEYE_FACES_PER_PERSON", "40")),
        min_face_px=int(_env("IMMICHEYE_MIN_FACE_PX", "60")),
        refresh_minutes=int(_env("IMMICHEYE_REFRESH_MINUTES", "360")),
        detect_min_score=float(_env("IMMICHEYE_DETECT_MIN_SCORE", "0.7")),
        match_threshold=float(_env("IMMICHEYE_MATCH_THRESHOLD", "0.40")),
        match_margin=float(_env("IMMICHEYE_MATCH_MARGIN", "0.05")),
        top_k=int(_env("IMMICHEYE_TOP_K", "3")),
        log_recognitions=_flag("IMMICHEYE_LOG_RECOGNITIONS", "1"),
        cpai_fallback=fallback == "1",
        cpai_sync=sync,
        cpai_prototypes=int(_env("IMMICHEYE_CPAI_PROTOTYPES", "4")),
        save_faces=int(_env("IMMICHEYE_SAVE_FACES", "0")),
        tile_px=int(_env("IMMICHEYE_TILE_PX", "1280")),
        label_case=case,
        label_sep=sep,
        lowres_px=tuple(int(p) for p in _env("IMMICHEYE_LOWRES_PX", "32,48").split(",") if p.strip()),
        events_url=_env("IMMICHEYE_EVENTS_URL", ""),
        events_token=_env("IMMICHEYE_EVENTS_TOKEN", ""),
        events_cameras=tuple(c.strip() for c in _env("IMMICHEYE_EVENTS_CAMERAS", "").split(",") if c.strip()),
        events_frame_px=int(_env("IMMICHEYE_EVENTS_FRAME_PX", "0")),
        events_queue_mb=int(_env("IMMICHEYE_EVENTS_QUEUE_MB", "64")),
        blueiris_url=_env("IMMICHEYE_BLUEIRIS_URL", "").rstrip("/"),
        blueiris_user=_env("IMMICHEYE_BLUEIRIS_USER", ""),
        blueiris_password=bi_password,
        late_names_s=int(_env("IMMICHEYE_LATE_NAMES_S", "60")),
        alert_share=_flag("IMMICHEYE_ALERT_SHARE", "0"),
        alert_hold_ms=int(_env("IMMICHEYE_ALERT_HOLD_MS", "3000")),
    )
