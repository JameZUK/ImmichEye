# ImmichEye

Give Blue Iris access to Immich for facial recognition.

Blue Iris's face recognition through CodeProject.AI's Face Processing module is weak. Immich's
is excellent, and you've probably already named everyone in it. ImmichEye gives Blue Iris
Immich's recognition, with Immich as the only place faces are managed. You don't train faces in
Blue Iris and nothing has to be kept in step by hand. Name, merge or fix someone in Immich, and
Blue Iris knows after the next refresh.

- **Better recognition.** It uses the same face model as Immich (InsightFace `buffalo_l`),
  compared against each person's best recent photos. On held-out faces it named 86–97% of people
  correctly and 1–2.5% wrongly, against CodeProject.AI's 41–52% correct and 45–53% wrong (see
  [Accuracy](#accuracy)).
- **One source of truth.** People are managed in Immich only. Blue Iris can't add or delete faces.
- **Drop-in.** Blue Iris sees an ordinary CodeProject.AI server. Object detection, plates and
  custom models carry on exactly as before.
- **Read-only to Immich.** The API key needs only four read permissions.

## Contents

- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuring Blue Iris](#configuring-blue-iris)
- [Managing people](#managing-people)
- [CodeProject.AI's face database](#codeprojectais-face-database)
- [When ImmichEye or Immich ML is down](#when-immicheye-or-immich-ml-is-down)
- [Accuracy](#accuracy)
- [Tuning](#tuning)
- [Settings reference](#settings-reference)
- [Admin endpoints](#admin-endpoints)
- [Troubleshooting](#troubleshooting)
- [Updating](#updating)
- [Development](#development)
- [Privacy](#privacy)

## How it works

```
                        ┌─ face/recognize, face/list ───► ImmichEye ──► Immich ML /predict
Blue Iris ──► router ───┼─ face/register, face/delete ──► ImmichEye (refused / ignored)
              (nginx)   └─ everything else ─────────────► CodeProject.AI
                                                              ▲
                                     ImmichEye ── sync ───────┘
                                         ▲
                                         └── gallery built from the Immich API
```

ImmichEye runs as two containers next to your existing CodeProject.AI server.

**The router** (nginx) is what Blue Iris talks to. Blue Iris sends every AI request, for objects,
plates, custom models and faces, to one server. The router passes everything to
CodeProject.AI untouched except the face calls, which go to ImmichEye. The router finds ImmichEye
on each request rather than once at startup, so object detection keeps working even while
ImmichEye is stopped.

**The gallery** is ImmichEye's set of reference faces. For each named person in Immich it:

1. finds their most recent photos (default 250);
2. picks their largest faces among those (default 40, at least 60 px), so the references show
   how people look now and have enough detail;
3. crops each face, with some surrounding context, from Immich's preview image;
4. turns each crop into an embedding with the Immich machine-learning server. An embedding is a
   list of 512 numbers describing the face, made by the same model Immich uses.

Embeddings are cached by Immich face id, so a refresh (every 6 hours by default) only processes
faces it hasn't seen before. A restart uses the cache straight away.

**Recognition.** When Blue Iris detects a person, it sends the frame to `face/recognize`.
ImmichEye passes the frame to Immich ML, which finds every face and makes an embedding for each.
Each face is then compared with every person in the gallery:

- A person's score is the average similarity of their three closest reference faces (cosine
  similarity: 1 means identical, about 0.2 is a stranger).
- The best-scoring person is named if their score reaches the threshold (0.40) **and** beats
  the runner-up by a margin (0.05). Otherwise the face is `unknown`.

The reply uses CodeProject.AI's exact format, so Blue Iris shows the name on the alert, in the
alert memo (e.g. `cal_smith:78%`) and to anything that reads it.

**Face management.** Blue Iris's own face training would otherwise go to CodeProject.AI.
ImmichEye answers those calls itself: `register` is refused and `delete` is acknowledged but
ignored. See [Managing people](#managing-people) for why the deletes matter.

**CodeProject.AI sync.** After each refresh, ImmichEye registers the same people in
CodeProject.AI's own face database. If you ever point Blue Iris straight back at CodeProject.AI,
it knows the same people, using its own weaker matching.

## Requirements

- **Blue Iris** using **CodeProject.AI** as its AI provider. Tested with Blue Iris 6, CodeProject.AI
  2.9 and Face Processing 1.12. Blue Iris 6's built-in AI isn't supported; see
  [Troubleshooting](#troubleshooting).
- **Immich** with facial recognition enabled and the people you care about named. Built against
  Immich 3.3.
- **Immich's machine-learning container reachable over the network** from wherever ImmichEye
  runs, on port 3003. A standard Immich install doesn't publish it outside Docker, so you may
  need to add a port mapping to `immich-machine-learning`. ImmichEye only uses it for face
  detection and embeddings, the same work Immich does with it.
- **Docker** with Compose on any machine that can reach Immich, Immich ML and CodeProject.AI.
  Running it on the CodeProject.AI host is simplest.

## Installation

1. **Create an Immich API key.** In Immich, open **Account Settings → API Keys** and create a key
   with only these permissions: `person.read`, `asset.read`, `face.read`, `asset.view`.
   ImmichEye never writes to Immich. Use the account that owns the photos and named people.
2. **Configure.**
   ```sh
   git clone https://github.com/JameZUK/ImmichEye.git immicheye && cd immicheye
   cp .env.example .env      # fill in the URLs and the API key
   mkdir -p data
   ```
   At minimum set `IMMICH_URL`, `IMMICH_API_KEY`, `IMMICH_ML_URL`, `CPAI_URL` and
   `CPAI_UPSTREAM`. Every setting is described in [Settings reference](#settings-reference).
3. **Start.**
   ```sh
   docker compose up -d --build
   docker compose logs -f immicheye
   ```
   Wait for `gallery: N people, M faces`. The first build embeds every reference face, which
   takes a minute or two for about 50 people on a GPU. Then wait for `CodeProject.AI sync`.
4. **Check.**
   ```sh
   curl http://127.0.0.1:8000/immicheye/status
   ```
   You should see your people under `faces_per_label`, and `last_error` should be null.
5. **Point Blue Iris at the router.** See below.

## Configuring Blue Iris

These are the settings that matter, from the Blue Iris 6 help (AI chapter) and testing. Exact
names and locations may differ slightly between Blue Iris versions.

### Global: Settings → AI

- **Provider:** CodeProject.AI, as before.
- **Server port** (and IP, if ImmichEye runs on another machine): the router's port,
  `IMMICHEYE_PORT` (default **32170**), instead of CodeProject.AI's 32168.
- **Don't train faces here, and don't import faces from the Faces folder.** Blue Iris "syncs" its
  own faces folder to the AI server, deleting any server face that isn't in that folder.
  ImmichEye blocks this, so opening the faces view is harmless. It will show your Immich people.

To undo everything, set the port back to CodeProject.AI's own.

### Per camera: camera properties → Trigger → AI

| Setting | Recommended | Why |
|---|---|---|
| **Use main stream if available** | **On** for every camera that should recognise faces | Blue Iris shrinks images to 640×640 for object detection but crops faces from the image it sends. On a sub-stream frame faces are often under 20 px, which no model can recognise. On main stream we saw faces of 90–160 px. The manual: main stream only benefits face and plate recognition. |
| **Facial recognition** | On; the models box must not contain `faces:0` | `faces:0` in the per-camera models list turns recognition off for that camera. |
| **To confirm** | `person` plus whatever else matters for that camera | Face recognition runs on detected people, so `person` is all faces need. You don't need face names here. See the example below. |
| **To cancel** | Face names only if you mean it | A face name here cancels the alert, e.g. no alerts for your own household. With a wildcard (`cal_smith*`) it also matches CodeProject.AI's synced `cal_smith1`…`4`. |
| **Min. confidence** | **70% or lower** | ImmichEye reports names at 70% and up (72–81% in testing). A camera or label minimum above 70 drops weaker true matches. |
| **Additional post-trigger images** | 5–10 at 250–500 ms | More images means more chances of a frame where the person faces the camera. Blue Iris prefers the image with the most faces for the alert. |
| **Zoom-in to objects** | Optional, off is fine | It only changes the saved alert image. It also sends extra face requests on a zoomed image, which in testing held *smaller* faces than the main-stream frames, so it doesn't help recognition. |
| **Save unknown faces** | Off | Blue Iris's own feature for its built-in AI. ImmichEye ignores that folder and can keep its own crops (`IMMICHEYE_SAVE_FACES`). |
| **Override server** | Empty | A camera with its own AI server bypasses ImmichEye completely. |
| **Primary object detection** | On, except for very close-up cameras | Faces are only looked for on detected people. If a camera is so close (a doorbell) that it rarely detects a whole person, turning primary detection off makes Blue Iris search for faces across the whole image. |

**Example "To confirm" lists:**

- Cameras that see a road or drive: `person,car,truck,bus,motorcycle,bicycle,vehicle`.
  Include `vehicle`: CodeProject.AI's custom `ipcam-*` models label vehicles as `vehicle`, not
  `car`.
- A garden camera where vehicles can't appear: `person`, plus `dog`, `cat` etc. if you want
  animal alerts. Labels a camera can't legitimately see only add false alerts. Leave out `boat`
  unless a camera actually looks at water.

**Face names in rules** use ImmichEye's labels (see [Labels](#labels)). Add a `*`, as the manual
suggests, to match both ImmichEye's `cal_smith` and CodeProject.AI's `cal_smith1`…`4`.

**Performance:** every image goes to each model in the camera's list, so trim custom models that a
camera doesn't need, such as licence plates on a back garden. With main-stream images this is
real work for CodeProject.AI. It doesn't affect faces.

## Managing people

Immich is the only source of faces. Manage people there: name them, merge duplicates, fix
wrongly assigned faces. Changes reach Blue Iris at the next refresh, or straight away with
`POST /immicheye/refresh`.

- **Who is recognised.** By default, every named, non-hidden person in Immich.
  `IMMICHEYE_PEOPLE` limits recognition to a comma-separated list of Immich names. Your household
  and regular visitors are a better choice than everyone in a photo library: fewer people means
  fewer chances of a wrong match.
- **Duplicates.** Several Immich people with the same name (unmerged clusters) pool their faces
  under one label. Misspelt duplicates ("Smith" and "Smiht") stay separate labels: merge them in
  Immich, or map both to one label with `IMMICHEYE_NAME_MAP`.
- **Quality.** The gallery uses each person's largest recent faces, so a few wrongly assigned
  faces in Immich matter little. Persistent wrong matches between two people usually mean
  some of one person's photos are assigned to the other.

### Labels

Labels are the names Blue Iris sees. They're lower case with words joined by underscores, which
Blue Iris and apps reading its alert memos handle reliably, and never end in a digit (Blue Iris
treats a trailing number as a registration number: `cal1` is `cal`).

| `IMMICHEYE_LABELS` | "Cal Smith" | "Cal Brown" | "Bea Jones (Mum)" |
|---|---|---|---|
| `first` (default) | `cal_smith` | `cal_brown` | `bea` |
| `full` | `cal_smith` | `cal_brown` | `bea_jones_mum` |

With `first`, a label is the first name unless two different people share it. Override anyone
with `IMMICHEYE_NAME_MAP`, e.g. `{"Bea Jones (Mum)": "mum"}`. Avoid labels that are also object
names (`person`, `car`, `dog`…). Blue Iris would confuse them.

### Blue Iris can't change faces

ImmichEye answers Blue Iris's face `register` and `delete` calls itself; they never reach
CodeProject.AI:

- **Register** is refused with "Faces are managed in Immich (ImmichEye)".
- **Delete** is acknowledged and ignored.

This matters because **opening Blue Iris's faces view sends a delete for every face on the server
that isn't in Blue Iris's own faces folder**. Passed through, that would wipe CodeProject.AI's
synced faces. If Blue Iris is ever pointed straight at CodeProject.AI, don't open that view; if
you do, the next sync puts the faces back.

## CodeProject.AI's face database

CodeProject.AI keeps one averaged embedding per name. ImmichEye therefore registers each person
as up to `IMMICHEYE_CPAI_PROTOTYPES` (default 4) entries, `cal_smith1`…`cal_smith4`. Each entry
averages one cluster of similar-looking faces, such as with and without glasses, or older and
newer photos.

`IMMICHEYE_CPAI_SYNC` sets how much of the database ImmichEye owns:

| Mode | |
|---|---|
| `off` | Never touch CodeProject.AI. |
| `managed` (default) | Add, update and remove only the entries ImmichEye created. It never overwrites or deletes anyone else's, and skips a number that's already taken. |
| `exclusive` | As `managed`, and delete every other entry, so CodeProject.AI knows exactly the Immich people. Use this once Immich is your only source of faces. |

ImmichEye records what it registered in `data/cpai_sync.json`. Before switching an existing
install to `exclusive`, back up CodeProject.AI's `faceembedding.db` (in its data directory,
`/etc/codeproject/ai` in the Docker image).

A sync warning like `register cal_smith4: No face detected` means CodeProject.AI's detector
found no face in any image of that cluster (usually side-on faces). That person's other entries
are unaffected.

## When ImmichEye or Immich ML is down

By default (`IMMICHEYE_CPAI_FALLBACK=0`), face recognition reports **no face** until it's back.
Person, vehicle and every other detection carry on normally. Only names stop.

ImmichEye doesn't fall back to CodeProject.AI by default because a wrong name is worse than no
name:

- An "unknown" person still gets you the person alert, and you look at the clip.
- A wrong name (`bea` for a stranger) can make you ignore an alert you needed. It can satisfy a
  rule that quietens known people, and it stays in alert histories and face filters for good.
- Even with the same people synced, CodeProject.AI names about half of faces wrongly (see
  [Accuracy](#accuracy)). A fallback name is no better than a guess.

`IMMICHEYE_CPAI_FALLBACK=1` lets CodeProject.AI answer face recognition during outages, both
from ImmichEye and from the router when ImmichEye itself is unreachable. Register and delete
never fall back. If you enable it, raise Blue Iris's face minimum confidence.

The better fix is short outages: make sure the Immich machine-learning container starts on boot
(ImmichEye's containers use `restart: unless-stopped`).

## Accuracy

### Held-out Immich faces

Measured with `scripts/evaluate.py` on one library: 52 people, 2,178 reference faces. Test faces
are each person's Immich faces that the gallery didn't use, at the default settings. Small faces
are the closest stand-in for CCTV.

| Test faces | ImmichEye correct / **wrong** / unknown | CodeProject.AI (same people synced) correct / **wrong** / unknown |
|---|---|---|
| Small, 30–60 px | 86% / **2.5%** / 11% | 41% / **53%** / 7% |
| Larger, 80–400 px | 97% / **1.2%** / 2% | 52% / **45%** / 3% |

CodeProject.AI's figures are at Blue Iris's usual minimum confidence (0.40). Raising it removes
right and wrong names together: at 0.80 it got 20% right and 8% wrong on larger faces. Part of
the "wrong" share is the library's own duplicates, the same person under two spellings.

### Real CCTV

On walks past two cameras running main stream, one person was named in about half to 70% of
the frames where a face was detected (9 of 13 on the closer camera), with scores of 0.44–0.62.
No frame was given the wrong name. Every
"unknown" frame, checked with saved crops, was the top or back of a head, a motion-blurred face,
or a face under 10 px from a sub-stream camera. Run `scripts/evaluate.py` and check
`data/recognitions.jsonl` for your own figures; results depend on your photos and cameras.

## Tuning

Two thresholds decide every result:

| Setting | Default | Effect |
|---|---|---|
| `IMMICHEYE_DETECT_MIN_SCORE` | 0.70 | How sure the detector must be that something is a face. On real CCTV, clear faces detect at about 0.8 and the tops and backs of heads at 0.5–0.6. 0.70 is Immich's own default. |
| `IMMICHEYE_MATCH_THRESHOLD` | 0.40 | How similar a face must be to a person to be named. A known person typically scores 0.45–0.65 on CCTV and strangers under 0.25. |
| `IMMICHEYE_MATCH_MARGIN` | 0.05 | How far the best person must be ahead of the runner-up. |

If people you know often come back `unknown` on clear, frontal faces, lower the threshold a
little (e.g. 0.35). If you see wrong names, raise the threshold or the margin.

**Tools:**

```sh
# Every request: frame size, faces, top three matches, result
tail -f data/recognitions.jsonl

# Keep crops of the last 500 faces Blue Iris sent, named with result, best match, score and width
#   .env: IMMICHEYE_SAVE_FACES=500   ->   data/faces/20261024-153924-424_0_cal_smith_cal_smith-0.60_93px.jpg

# Score any image
curl -F image=@frame.jpg http://127.0.0.1:8000/immicheye/explain

# Accuracy on your own library at a range of thresholds; --cpai also scores CodeProject.AI
docker compose exec immicheye python scripts/evaluate.py [--cpai] [--min-px 30 --max-px 60]
```

## Settings reference

All settings go in `.env` (see `.env.example`).

| Setting | Default | |
|---|---|---|
| `IMMICH_URL` | | Immich server, e.g. `http://immich.lan:2283` |
| `IMMICH_API_KEY` | | Key with `person.read`, `asset.read`, `face.read`, `asset.view` |
| `IMMICH_ML_URL` | | Immich machine-learning server, e.g. `http://immich.lan:3003` |
| `CPAI_URL` | | CodeProject.AI, used by ImmichEye for sync and fallback |
| `CPAI_UPSTREAM` | | CodeProject.AI as `host:port`, used by the router |
| `IMMICHEYE_PORT` | 32170 | Router port: point Blue Iris here |
| `IMMICHEYE_ADMIN_BIND` / `_PORT` | 127.0.0.1 / 8000 | Where the [admin endpoints](#admin-endpoints) listen |
| `IMMICHEYE_PEOPLE` | all named | Comma-separated Immich names to recognise |
| `IMMICHEYE_LABELS` | `first` | `first` or `full`; see [Labels](#labels) |
| `IMMICHEYE_NAME_MAP` | `{}` | JSON map of Immich name to label |
| `IMMICHEYE_MODEL` | `buffalo_l` | Must match the facial recognition model in Immich's settings |
| `IMMICHEYE_ASSETS_PER_PERSON` | 250 | Recent photos searched per person |
| `IMMICHEYE_FACES_PER_PERSON` | 40 | Reference faces kept per person |
| `IMMICHEYE_MIN_FACE_PX` | 60 | Smallest reference face (short side) |
| `IMMICHEYE_REFRESH_MINUTES` | 360 | Gallery refresh and CodeProject.AI sync interval |
| `IMMICHEYE_DETECT_MIN_SCORE` | 0.70 | See [Tuning](#tuning) |
| `IMMICHEYE_MATCH_THRESHOLD` | 0.40 | See [Tuning](#tuning) |
| `IMMICHEYE_MATCH_MARGIN` | 0.05 | See [Tuning](#tuning) |
| `IMMICHEYE_TOP_K` | 3 | Reference faces averaged into a person's score |
| `IMMICHEYE_CPAI_SYNC` | `managed` | `off`, `managed` or `exclusive`; see [CodeProject.AI's face database](#codeprojectais-face-database) |
| `IMMICHEYE_CPAI_PROTOTYPES` | 4 | CodeProject.AI entries per person |
| `IMMICHEYE_CPAI_FALLBACK` | 0 | 1 = let CodeProject.AI answer when ImmichEye can't |
| `IMMICHEYE_LOG_RECOGNITIONS` | 1 | Write `data/recognitions.jsonl` (rotated at 20 MB) |
| `IMMICHEYE_SAVE_FACES` | 0 | Keep crops of the last N faces in `data/faces` |

### What Blue Iris's minimum confidence means here

Blue Iris sends its minimum confidence with each face request. CodeProject.AI uses it as its match
threshold on a `(cosine + 1) / 2` scale, where the usual 0.40 accepts almost any face. ImmichEye
ignores it and decides with its own threshold and margin. Confidences it returns are on that
same `(cosine + 1) / 2` scale, so they look familiar to Blue Iris: the default threshold of 0.40
appears as 70%.

## Admin endpoints

These listen on `IMMICHEYE_ADMIN_BIND:IMMICHEYE_ADMIN_PORT` (default `127.0.0.1:8000`) and are
unauthenticated. Keep them off untrusted networks.

| | |
|---|---|
| `GET /immicheye/status` | Gallery size, faces per label, last refresh and sync errors, settings in use |
| `POST /immicheye/refresh` | Rebuild the gallery and sync CodeProject.AI now |
| `POST /immicheye/explain` | Score a posted image (`image` field): every face with its five best matches |

## Troubleshooting

| Symptom | Check |
|---|---|
| No names at all | `GET /immicheye/status`: `faces` should be non-zero and `last_error` null. Then `tail -f data/recognitions.jsonl` while someone walks past. No lines means Blue Iris isn't sending face requests: check the AI port, that the camera has face recognition on, and that `person` is detected. |
| Lines in the log, all `no face` | Look at `size`: under about 1280 px wide means the camera isn't using the main stream for AI. |
| Faces found, all `unknown` | Turn on `IMMICHEYE_SAVE_FACES` and look at the crops. Tops of heads and blur are expected. Clear frontal faces scoring 0.3–0.4 for the right person suggest lowering the threshold slightly. |
| Wrong names | The log shows the top three scores. A small margin between two people usually means mixed-up faces in Immich. Raise `IMMICHEYE_MATCH_MARGIN` or the threshold. |
| `unavailable` count rising | ImmichEye can't reach Immich ML. Check that it's running and that `IMMICH_ML_URL` works from the ImmichEye host: `curl $IMMICH_ML_URL/ping` should return `pong`. |
| Blue Iris's faces view shows the wrong people | It shows ImmichEye's list, i.e. your Immich people. That's expected. |
| Port 8000 already in use | Set `IMMICHEYE_ADMIN_PORT` to a free port. |
| Using Blue Iris 6's built-in AI | Not supported: ImmichEye works through the CodeProject.AI API. Switch the provider to CodeProject.AI. |

## Updating

```sh
git pull
docker compose up -d --build
```

`.env` and `data/` are kept. If you change `IMMICHEYE_MODEL` (or the model in Immich), every
reference face is re-embedded at the next refresh.

## Development

```sh
python -m venv .venv && .venv/bin/pip install -e '.[test]'
.venv/bin/pytest
```

The code is small: `immicheye/server.py` (endpoints), `gallery.py` (reference faces and
matching), `cpai_sync.py` (CodeProject.AI sync), `immich.py` and `ml.py` (API clients), and
`nginx/default.conf.template` (the router).

## Privacy

ImmichEye sends Immich previews and camera frames only to your own Immich ML and CodeProject.AI
servers, and reads from Immich with a read-only key. It stores reference face crops and
embeddings, a recognition log and, if enabled, crops of faces from your cameras in `data/`.
Recognising people on camera may be regulated where you live. Check before pointing cameras at
public spaces.
