# ImmichEye

Give Blue Iris access to Immich for facial recognition.

Blue Iris's face recognition (through CodeProject.AI's Face Processing module) is weak.
Immich's is excellent, and you've already named everyone in it. ImmichEye gives Blue Iris
Immich's recognition, with Immich as the only source of faces. There's no training in Blue Iris
and no double bookkeeping. Name, merge or fix someone in Immich, and Blue Iris knows after the
next refresh.

## How it works

```
Blue Iris ──► router (nginx) ─┬─ /v1/vision/face/recognize, /face/list ──► ImmichEye ──► Immich ML /predict
                              │                                              │
                              └─ everything else ──► CodeProject.AI ◄─sync── └─ gallery built from the Immich API
```

- **Router.** Blue Iris's AI server setting points at the router instead of CodeProject.AI.
  Object detection, plates and everything else pass straight through to CodeProject.AI,
  untouched. Only the face calls are diverted. The router keeps serving object detection even
  while ImmichEye is down.
- **Gallery.** For every named person in Immich, ImmichEye takes the largest faces among their
  most recent photos, crops them from Immich's previews, and embeds them with the same model
  Immich uses (InsightFace `buffalo_l` by default) via the Immich machine-learning server.
  Embeddings are cached, so a refresh only embeds new faces, and a restart serves from the cache
  straight away.
- **Recognition.** Each frame Blue Iris sends goes through Immich ML for detection and
  embedding. Each face is scored against every person: the mean cosine similarity of their
  best few reference faces. A person is reported when they clear a threshold and beat the
  runner-up by a margin. Replies use CodeProject.AI's exact format, so Blue Iris (and
  anything reading its alert memos) sees ordinary face labels.
- **CodeProject.AI sync.** After each refresh, ImmichEye also registers the same people in
  CodeProject.AI's own face database (see [below](#codeprojectais-face-database)). That way
  CodeProject.AI knows the same people as Immich if Blue Iris is ever pointed back at it directly.

## Accuracy

Measured with `scripts/evaluate.py` on one real library (52 people, 2,178 reference faces).
The test faces are held-out Immich faces the gallery didn't use, with the settings below
(threshold 0.40, margin 0.05). Each face counts as correct, wrong, or unknown:

| Test faces | | ImmichEye | CodeProject.AI, same people synced |
|---|---|---|---|
| Small, 30–60 px (most like CCTV) | correct / **wrong** / unknown | 86% / **2.5%** / 11% | 41% / **53%** / 7% |
| Larger, 80–400 px | correct / **wrong** / unknown | 97% / **1.2%** / 2% | 52% / **45%** / 3% |

CodeProject.AI's numbers are at Blue Iris's usual minimum confidence of 0.40. Raising it cuts
wrong names and right ones together: at 0.80 it gets 20% right and 8% wrong on larger faces.
Run the script against your own library: results depend on your photos and cameras. Part of the
"wrong" share is usually the library's own duplicates, the same person under two names in Immich.

## Requirements

- Blue Iris using CodeProject.AI (tested with CPAI 2.9 and Face Processing 1.12).
- Immich with facial recognition enabled (built against Immich 3.3) and people named.
- Network access from ImmichEye to the Immich server, the Immich machine-learning container
  (port 3003, which isn't always published), and CodeProject.AI.

## Setup

1. In Immich, go to **Account Settings → API Keys** and create a key with only these permissions:
   `person.read`, `asset.read`, `face.read`, `asset.view`. ImmichEye never writes to Immich.
2. Configure and start:
   ```sh
   cp .env.example .env      # fill in the URLs and the API key
   mkdir -p data
   docker compose up -d --build
   docker compose logs -f immicheye    # wait for "gallery: N people, M faces"
   ```
3. Check: `curl http://127.0.0.1:8000/immicheye/status`
4. In Blue Iris, open **Settings → AI** and change the server port (and host, if ImmichEye
   runs on a different machine) to the router's port (`IMMICHEYE_PORT`, default 32170).
5. **Send the main stream to the AI.** In each camera's AI settings, enable using the main
   stream (high-resolution images) for analysis. Faces in a sub-stream frame are often only
   a few pixels wide. No model can recognise those.

To undo, point Blue Iris back at CodeProject.AI's own port.

## Managing faces

Immich is the only source of faces. People are managed there, never in Blue Iris.

- **Who is recognised.** By default, every named, non-hidden person. `IMMICHEYE_PEOPLE` limits
  recognition to a comma-separated list of names. Use it to recognise your household and regular
  visitors, not everyone in your photo library: fewer people means fewer false matches.
- **Labels** are the lower-case first name (`cal`), or the full name where two different
  people share a first name (`cal_smith`). Override them with `IMMICHEYE_NAME_MAP`, e.g.
  `{"Evelyn Mary Jones": "eve"}`. Unmerged duplicates with the same name pool their faces. Merge
  misspelt duplicates in Immich, or map them to one label.
- **Blue Iris can't change faces.** ImmichEye answers Blue Iris's face register and delete
  calls itself, and they never reach CodeProject.AI. Registering is refused ("Faces are managed
  in Immich"). Deleting is acknowledged and ignored. This matters because opening Blue Iris's
  faces view sends a delete for every server face that isn't in Blue Iris's own faces folder.
  Passed through, that would wipe CodeProject.AI's synced faces. Only ever open that view while
  Blue Iris points at the router: pointed straight at CodeProject.AI, it will still delete them.
  With `managed` or `exclusive` sync, the next refresh puts them back.

## CodeProject.AI's face database

CodeProject.AI keeps one averaged embedding per name, so ImmichEye registers each person as
up to `IMMICHEYE_CPAI_PROTOTYPES` (default 4) entries, `cal1`…`cal4`. Each entry averages
one cluster of similar-looking faces: glasses or not, older or newer photos. Blue Iris and
Caesa read `cal3` as cal.

`IMMICHEYE_CPAI_SYNC` chooses how much of the database ImmichEye owns:

| Mode | |
|---|---|
| `off` | Never touch CodeProject.AI. |
| `managed` (default) | Add, update and remove only the entries ImmichEye created. It never overwrites or deletes anyone else's, and skips a number that's already taken. Your existing faces stay. |
| `exclusive` | As `managed`, and delete every other entry, so CodeProject.AI knows exactly the Immich people. Use this once you've decided Immich is the only source of faces. |

ImmichEye tracks what it registered in `data/cpai_sync.json`. Before switching an existing
install to `exclusive`, back up CodeProject.AI's `faceembedding.db` (in its data directory,
`/etc/codeproject/ai` in the Docker image).

## When ImmichEye or Immich ML is down

By default (`IMMICHEYE_CPAI_FALLBACK=0`) face recognition reports **no face** until it's back.
Person and vehicle detection carry on as normal. Only face names stop.

Why not fall back to CodeProject.AI? Because a wrong name is worse than no name. "Unknown"
still gets you the person alert, and you look at the clip. A wrong name ("bea" for a stranger)
can make you ignore an alert you needed. It can also wrongly satisfy notification rules that
quieten known people, and it stays in alert histories and face filters for good. Even with the
same people synced, CodeProject.AI names about half of faces wrongly
([Accuracy](#accuracy)), so a fallback name tells you nothing you can rely on.

`IMMICHEYE_CPAI_FALLBACK=1` lets CodeProject.AI answer during outages, and also applies when the
router can't reach ImmichEye. It uses Blue Iris's face minimum confidence, so raise that if you
enable it.

Keep outages short instead: make sure the Immich ML container restarts on boot, as ImmichEye's
does (`restart: unless-stopped`).

## Blue Iris's minimum confidence

ImmichEye decides who someone is with its own `IMMICHEYE_MATCH_THRESHOLD` and margin, not with
the minimum confidence Blue Iris sends. CodeProject.AI applies that value to `(cosine+1)/2`, a
scale on which Blue Iris's usual 0.40 accepts almost any face. Confidences ImmichEye returns are
on that same scale. The default threshold of 0.40 cosine is 0.70 there, so names show in Blue
Iris as about 70–90%. If you raise Blue Iris's face minimum confidence above 0.70, it may drop
ImmichEye's weaker matches.

## Tuning

Every recognition is logged to `data/recognitions.jsonl` with each face's top three scores.

```sh
# How a particular frame scores
curl -F image=@frame.jpg http://127.0.0.1:8000/immicheye/explain

# Accuracy against your own library at a range of thresholds (run where .env is available)
set -a; . ./.env; set +a; IMMICHEYE_DATA=./data python scripts/evaluate.py [--cpai] [--min-px 30 --max-px 60]
```

Defaults: threshold 0.40 (cosine), margin 0.05. If people you know come back `unknown`, lower
the threshold a little. If you see wrong names, raise it or the margin.

## Admin endpoints

These listen on `IMMICHEYE_ADMIN_PORT` (default 8000), bound to `IMMICHEYE_ADMIN_BIND` (default 127.0.0.1). They're
unauthenticated.

| | |
|---|---|
| `GET /immicheye/status` | Gallery size, faces per label, last refresh and sync errors, unavailable count |
| `POST /immicheye/refresh` | Rebuild the gallery and sync CodeProject.AI now |
| `POST /immicheye/explain` | Score a posted image (`image` field) |

## Development

```sh
python -m venv .venv && .venv/bin/pip install -e '.[test]'
.venv/bin/pytest
```

## Privacy

ImmichEye sends Immich previews and camera frames only to your own Immich ML and
CodeProject.AI servers. It stores face crops, embeddings and a recognition log in `data/`.
Recognising people on camera may be regulated where you live. Check before you point it
at a public street.
