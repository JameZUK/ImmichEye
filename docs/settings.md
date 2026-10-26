# Settings

All settings go in `.env`, next to `docker-compose.yml`. `.env.example` lists every one with a
short comment; copy it to `.env` to start. After changing `.env`, apply it with:

```sh
docker compose up -d
```

On/off settings take `1` or `0` (`true`/`false`, `yes`/`no` and `on`/`off` work too). An invalid
value stops ImmichEye at startup, and `docker compose logs immicheye` says which setting.

## Connections (required)

| Setting | Example | |
|---|---|---|
| `IMMICH_URL` | `http://immich.lan:2283` | Your Immich server |
| `IMMICH_API_KEY` | | An Immich API key with `person.read`, `asset.read`, `face.read` and `asset.view` |
| `IMMICH_ML_URL` | `http://immich.lan:3003` | Immich's machine-learning container |
| `CPAI_URL` | `http://cpai.lan:32168` | Your CodeProject.AI server, for the sync and the fallback |
| `CPAI_UPSTREAM` | `cpai.lan:32168` | The same server as `host:port` without `http://`, for the router |

## Ports

| Setting | Default | |
|---|---|---|
| `IMMICHEYE_PORT` | `32170` | The router's port. Point Blue Iris's AI server port here |
| `IMMICHEYE_ADMIN_BIND` | `127.0.0.1` | Address the admin endpoints listen on. `127.0.0.1` means this host only |
| `IMMICHEYE_ADMIN_PORT` | `8000` | Port of the admin endpoints (status, refresh, explain) |

The admin endpoints have no password. Only bind them to another address on a trusted network.

## People and labels

| Setting | Default | |
|---|---|---|
| `IMMICHEYE_PEOPLE` | empty: everyone | Comma-separated Immich names to recognise, e.g. `Cal Smith,Bea Jones`. Empty means every named person who isn't hidden in Immich |
| `IMMICHEYE_LABELS` | `first` | `first`: first name (`cal`), or the full name when first names clash. `full`: always the full name (`cal_smith`) |
| `IMMICHEYE_LABEL_CASE` | `lower` | `lower` (`cal_smith`) or `title` (`Cal_Smith`) |
| `IMMICHEYE_LABEL_SEPARATOR` | `underscore` | `underscore` (`cal_smith`) or `space` (`cal smith`) |
| `IMMICHEYE_NAME_MAP` | `{}` | Labels for particular people, as JSON: `{"Bea Jones (Mum)": "mum"}` |

See [Labels](../README.md#labels) for the rules labels follow.

## Gallery

How the reference faces are chosen from Immich.

| Setting | Default | |
|---|---|---|
| `IMMICHEYE_MODEL` | `buffalo_l` | Must match the facial recognition model in Immich's settings. Changing it re-embeds every reference face |
| `IMMICHEYE_ASSETS_PER_PERSON` | `250` | How many of each person's most recent photos to look at |
| `IMMICHEYE_FACES_PER_PERSON` | `40` | How many of their largest faces to keep from those |
| `IMMICHEYE_MIN_FACE_PX` | `60` | Smallest face to use, in pixels (short side) |
| `IMMICHEYE_LOWRES_PX` | `32,48` | Also store each reference face shrunk to these widths, to match small CCTV faces. Empty turns it off |
| `IMMICHEYE_REFRESH_MINUTES` | `360` | How often to rebuild the gallery from Immich and sync CodeProject.AI |

## Recognition

| Setting | Default | |
|---|---|---|
| `IMMICHEYE_DETECT_MIN_SCORE` | `0.70` | How sure the detector must be that something is a face |
| `IMMICHEYE_MATCH_THRESHOLD` | `0.40` | How similar a face must be to a person to be named |
| `IMMICHEYE_MATCH_MARGIN` | `0.05` | How far the best person must be ahead of the runner-up |
| `IMMICHEYE_TOP_K` | `3` | A person's score is the average of their this-many closest reference faces |
| `IMMICHEYE_TILE_PX` | `1280` | Frames larger than this are also searched in overlapping tiles of this size, to find small faces. `0` turns it off |

What these mean and when to change them: [Tuning](accuracy.md#tuning).

## Logs and saved faces

| Setting | Default | |
|---|---|---|
| `IMMICHEYE_LOG_RECOGNITIONS` | `1` | Log every face request to `data/recognitions.jsonl`. The log is rotated at 20 MB, keeping one old file |
| `IMMICHEYE_SAVE_FACES` | `0` | Keep crops of the last this-many faces Blue Iris sent, in `data/faces`, to check results by eye. `0` turns it off |

## Names in alerts

Both are explained in [Names in alerts](names-in-alerts.md).

| Setting | Default | |
|---|---|---|
| `IMMICHEYE_ALERT_SHARE` | `0` | `1`: every reply in an alert also carries names found in its other images |
| `IMMICHEYE_ALERT_HOLD_MS` | `3000` | A reply with no name waits for one until this long after the alert's first image |
| `IMMICHEYE_BLUEIRIS_URL` | empty: off | Blue Iris's web server, e.g. `http://blueiris.lan:81`. Setting it turns on late names |
| `IMMICHEYE_BLUEIRIS_USER` | | A Blue Iris **admin** user: only admins can change an alert's memo |
| `IMMICHEYE_BLUEIRIS_PASSWORD` | | Its password. Blue Iris's login only handles Latin-1 characters, so avoid emoji and non-Latin scripts |
| `IMMICHEYE_LATE_NAMES_S` | `60` | Names found up to this many seconds after an alert's first image are added to its memo |

## Face events

Explained in [Face events](face-events.md).

| Setting | Default | |
|---|---|---|
| `IMMICHEYE_EVENTS_URL` | empty: off | Where to POST every detected face |
| `IMMICHEYE_EVENTS_TOKEN` | | Sent as `Authorization: Bearer <token>` |
| `IMMICHEYE_EVENTS_CAMERAS` | empty: all | Comma-separated camera short names to send, e.g. `Front,Back` |
| `IMMICHEYE_EVENTS_FRAME_PX` | `0` | Also send the whole frame, its long side scaled to at most this. `0` sends faces only |
| `IMMICHEYE_EVENTS_QUEUE_MB` | `64` | Most camera frames, in MB, waiting to be sent. The oldest are dropped first |

## CodeProject.AI

Explained in [CodeProject.AI](codeproject-ai.md).

| Setting | Default | |
|---|---|---|
| `IMMICHEYE_CPAI_SYNC` | `managed` | `off`, `managed` or `exclusive`: how much of CodeProject.AI's face database ImmichEye looks after |
| `IMMICHEYE_CPAI_PROTOTYPES` | `4` | Entries per person in CodeProject.AI (`cal1`…`cal4`) |
| `IMMICHEYE_CPAI_FALLBACK` | `0` | `1`: let CodeProject.AI answer face requests while ImmichEye or Immich ML is down. Only `0` or `1` |
