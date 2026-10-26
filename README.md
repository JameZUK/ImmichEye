# ImmichEye

Give Blue Iris access to Immich for facial recognition.

Blue Iris's face recognition through CodeProject.AI's Face Processing module is weak. Immich's
is excellent, and you've probably already named everyone in it. ImmichEye lets Blue Iris use
Immich's recognition, so Immich is the only place you manage faces. You don't train faces in
Blue Iris and nothing has to be kept in step by hand: name, merge or fix someone in Immich, and
Blue Iris knows after the next refresh.

- **Better recognition.** It uses the same face model as Immich (InsightFace `buffalo_l`),
  compared against each person's best recent photos. On held-out faces it named 86–97% of people
  correctly and 1–2.5% wrongly, against CodeProject.AI's 41–52% correct and 45–53% wrong (see
  [Accuracy](docs/accuracy.md)).
- **One source of truth.** People are managed in Immich only. Blue Iris can't add or delete faces.
- **Drop-in.** Blue Iris sees an ordinary CodeProject.AI server. Object detection, plates and
  custom models carry on exactly as before.
- **Read-only to Immich.** The API key needs only four read permissions.

## Contents

- [What it does](#what-it-does)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuring Blue Iris](#configuring-blue-iris)
- [Managing people](#managing-people)
- [Checking it's working](#checking-its-working)
- [Troubleshooting](#troubleshooting)
- [Updating](#updating)
- [Development](#development)
- [Privacy](#privacy)

Further guides in [`docs/`](docs):

| Guide | Read it if you want to… |
|---|---|
| [Settings](docs/settings.md) | look up any setting in `.env` |
| [Names in alerts](docs/names-in-alerts.md) | get more alerts named: names shared across an alert's images, and names added after the alert |
| [CodeProject.AI](docs/codeproject-ai.md) | understand what ImmichEye does to CodeProject.AI's face database, and what happens during an outage |
| [Accuracy and tuning](docs/accuracy.md) | see test results, tune the thresholds, or read the recognition log |
| [Face events](docs/face-events.md) | send every detected face to another service, such as a visitor log |

## What it does

| Feature | Default | In short |
|---|---|---|
| Face recognition from Immich | always on | Blue Iris's face requests are answered from your Immich people. |
| Face management blocked | always on | Blue Iris can't add or delete faces; Immich is the only source. |
| Router | always on | One address for Blue Iris: face calls go to ImmichEye, everything else to CodeProject.AI. |
| Small and distant faces | on | Large frames are searched in tiles, and reference faces are also stored at CCTV size. |
| CodeProject.AI sync | on (`managed`) | CodeProject.AI is taught the same people, in case you ever point Blue Iris back at it. |
| [Names shared across an alert](docs/names-in-alerts.md#names-shared-across-an-alert) | off | A name found in any of an alert's images reaches the alert, not just names in the image Blue Iris picks. |
| [Late names](docs/names-in-alerts.md#late-names) | off | A name found shortly after the alert went out is added to its memo. |
| [Face events](docs/face-events.md) | off | Every face is also sent, with its embedding, to another service. |
| Fallback to CodeProject.AI | off | If ImmichEye or Immich ML is down, faces get no name rather than a likely wrong one ([why](docs/codeproject-ai.md#when-immicheye-or-immich-ml-is-down)). |

## How it works

```
                        ┌─ face/recognize, face/list ───► ImmichEye ──► Immich ML
Blue Iris ──► router ───┼─ face/register, face/delete ──► ImmichEye (refused / ignored)
              (nginx)   └─ everything else ─────────────► CodeProject.AI
                                                              ▲
                                     ImmichEye ── sync ───────┘
                                         ▲
                                         └── gallery built from the Immich API
```

ImmichEye runs as two containers next to your existing CodeProject.AI server.

**The router** (nginx) is what Blue Iris talks to. Blue Iris sends every AI request, for objects,
plates, custom models and faces, to one server. The router passes everything to CodeProject.AI
untouched except the face calls, which go to ImmichEye. It looks ImmichEye up on every request
rather than once at startup, so object detection keeps working even while ImmichEye is stopped.

**The gallery** is ImmichEye's set of reference faces. For each named person in Immich it:

1. finds their most recent photos (default 250);
2. picks their largest faces among those (default 40, at least 60 px), so the references show
   how people look now and have enough detail;
3. crops each face, with some surrounding context, from Immich's preview image;
4. turns each crop into an *embedding* using Immich's machine-learning server. An embedding is a
   list of 512 numbers describing a face; two photos of the same person give similar lists;
5. embeds each crop again shrunk to CCTV size (faces 32 and 48 px wide by default). A small,
   soft face from a camera matches these copies better than the sharp photo they came from.

Embeddings are cached in `data/`, so a refresh (every 6 hours by default) only processes faces it
hasn't seen before, and a restart uses the cache straight away.

**Recognition.** When Blue Iris detects a person, it sends the frame to `face/recognize`.
ImmichEye passes the frame to Immich ML, which finds every face and makes an embedding for each.
Immich ML shrinks every image to 640 px before looking for faces, which loses small faces in a
large frame, so frames over 1280 px are also searched in overlapping 1280 px tiles. Each face is
then compared with everyone in the gallery:

- A person's score is the average similarity of their three closest reference faces. Similarity
  runs from 1 (identical) down; a stranger scores about 0.2.
- The best person is named if their score reaches the **threshold** (0.40) **and** beats the
  runner-up by the **margin** (0.05). Otherwise the face is `unknown`.

The reply uses CodeProject.AI's exact format, so Blue Iris shows the name on the alert and in the
alert memo (e.g. `person:91%,cal_smith:78%`), where apps that read the memo pick it up.

## Requirements

- **Blue Iris** using **CodeProject.AI** as its AI provider. Tested with Blue Iris 6,
  CodeProject.AI 2.9 and Face Processing 1.12. Blue Iris 6's built-in AI isn't supported:
  ImmichEye works through the CodeProject.AI API.
- **Immich** with facial recognition enabled and the people you care about named. Built against
  Immich 3.3.
- **Immich's machine-learning container reachable over the network** from wherever ImmichEye
  runs, on port 3003. A standard Immich install doesn't publish it outside Docker, so you may
  need to add a port mapping to `immich-machine-learning`. ImmichEye only uses it for face
  detection and embeddings, the same work Immich does with it.
- **Docker** with Compose, on any machine that can reach Immich, Immich ML and CodeProject.AI.
  Running it on the CodeProject.AI host is simplest.

## Installation

1. **Create an Immich API key.** In Immich, open **Account Settings → API Keys** and create a key
   with only these permissions: `person.read`, `asset.read`, `face.read`, `asset.view`. Use the
   account that owns the photos and named people. ImmichEye never writes to Immich.

2. **Download and configure.**
   ```sh
   git clone https://github.com/JameZUK/ImmichEye.git immicheye && cd immicheye
   cp .env.example .env      # then edit .env
   mkdir -p data
   sudo chown 1000:1000 data # the container runs as uid 1000 and writes here
   ```
   In `.env`, set at least:

   | Setting | Example |
   |---|---|
   | `IMMICH_URL` | `http://immich.lan:2283` |
   | `IMMICH_API_KEY` | the key from step 1 |
   | `IMMICH_ML_URL` | `http://immich.lan:3003` |
   | `CPAI_URL` | `http://cpai.lan:32168` |
   | `CPAI_UPSTREAM` | `cpai.lan:32168` (the same server, without `http://`) |

   Everything else has a sensible default; see [Settings](docs/settings.md).

   The container runs as user id 1000, not root, and keeps its cache and logs in `./data`. If
   your own user id isn't 1000 (check with `id -u`), the `chown` is needed, or the gallery never
   builds and the status page shows a `PermissionError`.

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
   Your people should be listed under `faces_per_label`, and `last_error` should be `null`.

5. **Point Blue Iris at the router**, as below.

## Configuring Blue Iris

Setting names come from the Blue Iris 6 help (AI chapter) and testing; they may differ slightly
between versions.

### Global: Settings → AI

- **Provider:** CodeProject.AI, as before.
- **Server port** (and IP, if ImmichEye runs on another machine): the router's port,
  `IMMICHEYE_PORT` (default **32170**), instead of CodeProject.AI's 32168.
- **Don't train faces in Blue Iris, and don't import faces from its Faces folder.** People are
  managed in Immich. Opening Blue Iris's faces view is harmless: it lists your Immich people.

To undo everything, set the port back to CodeProject.AI's own.

### Per camera: camera properties → Trigger → AI

| Setting | Recommended | Why |
|---|---|---|
| **Use main stream if available** | **On** for every camera that should recognise faces | Blue Iris crops faces from the image it sends. On a sub-stream frame faces are often under 20 px, which no model can recognise; on main stream we saw 90–160 px. |
| **Facial recognition** | On; the models box must not contain `faces:0` | `faces:0` in the camera's models list turns recognition off for that camera. |
| **To confirm** | `person`, plus whatever else matters for that camera | Faces are only looked for on detected people, so `person` is all faces need. You don't need names here. |
| **To cancel** | Names only if you mean it | A name here cancels the alert, e.g. no alerts for your own household. A wildcard (`cal_smith*`) also matches CodeProject.AI's synced `cal_smith1`…`4`. |
| **Min. confidence** | **70% or lower** | ImmichEye reports names at 70% and up (72–81% in testing). A higher minimum drops weaker true matches. |
| **Additional post-trigger images** | About 20, at a short fixed interval | See [below](#how-many-images-to-analyse). |
| **Zoom-in to objects** | Off | It only changes the saved alert image, and the zoomed image it sends for faces held *smaller* faces than the main-stream frame in testing. |
| **Save unknown faces** | Off | A feature of Blue Iris's built-in AI. ImmichEye can keep its own crops (`IMMICHEYE_SAVE_FACES`). |
| **Override server** | Empty | A camera with its own AI server bypasses ImmichEye completely. |
| **Primary object detection** | On, except for very close-up cameras | Faces are only looked for on detected people. If a camera is so close (a doorbell) that it rarely sees a whole person, turning this off makes Blue Iris search the whole image for faces. |

**Example "To confirm" lists:**

- A camera that sees a road or drive: `person,car,truck,bus,motorcycle,bicycle,vehicle`.
  Include `vehicle`: CodeProject.AI's `ipcam-*` models label vehicles as `vehicle`, not `car`.
- A garden camera: `person`, plus `dog`, `cat` etc. if you want animal alerts. Labels a camera
  can't really see only add false alerts.

**Names in rules** use ImmichEye's labels (see [Labels](#labels)). Add a `*` to match both
ImmichEye's `cal_smith` and CodeProject.AI's `cal_smith1`…`4`.

**Performance:** every image goes to each model in the camera's list, so remove custom models a
camera doesn't need, such as licence plates on a back garden. It doesn't affect faces.

### How many images to analyse

When a camera triggers, Blue Iris analyses a burst of images (the trigger image plus the
"additional post-trigger images"), picks the best one, and only then confirms the alert and sends
notifications. Its memo lists the faces in that one image.

- **More images give more chances** of catching someone facing the camera.
- **But the alert waits for all of them.** With 20 images at a short fixed interval, alerts were
  confirmed about 5 s after the trigger. With the same 20 images at Blue Iris's progressive
  ("Prog") spacing they took about 11 s, and named no more people.
- **Names usually come early.** At a front door, the first name typically appeared within 2 s of
  the first image.

To catch faces that only turn up after the alert has gone out, use
[late names](docs/names-in-alerts.md#late-names) rather than a longer burst.

## Managing people

Immich is the only source of faces. Manage people there: name them, merge duplicates, fix wrongly
assigned faces. Changes reach Blue Iris at the next refresh (every 6 hours), or straight away
with:

```sh
curl -X POST http://127.0.0.1:8000/immicheye/refresh
```

- **Who is recognised.** By default, every named person in Immich who isn't hidden.
  `IMMICHEYE_PEOPLE` limits recognition to a comma-separated list of Immich names. Your household
  and regular visitors are a better choice than everyone in a photo library: fewer people means
  fewer chances of a wrong match.
- **Removing many people at once.** A refresh that would leave fewer than half the faces
  (narrowing `IMMICHEYE_PEOPLE`, for example) is held back once, in case Immich answered
  wrongly; the next refresh applies it. A refresh that would leave nobody is never applied.
  Either shows as `last_error` on the status page.
- **Duplicates.** Several Immich people with the same name (unmerged clusters) share one label.
  Misspelt duplicates ("Smith" and "Smiht") stay separate: merge them in Immich, or map both to
  one label with `IMMICHEYE_NAME_MAP`.
- **Wrong matches.** The gallery uses each person's largest recent faces, so a few wrongly
  assigned faces in Immich matter little. If two people are often confused, some of one
  person's photos are probably assigned to the other in Immich.

### Labels

A label is the name Blue Iris sees for a person. By default it's the first name in lower case
(`cal`), or the full name when two people share a first name.

| `IMMICHEYE_LABELS` | "Cal Smith" | "Cal Brown" | "Bea Jones (Mum)" |
|---|---|---|---|
| `first` (default) | `cal_smith` | `cal_brown` | `bea` |
| `full` | `cal_smith` | `cal_brown` | `bea_jones_mum` |

- `IMMICHEYE_LABEL_CASE=title` capitalises each word (`Cal_Smith`), keeping capitals already in
  the name (`Dan Lee (DJ)` → `Dan_Lee_DJ`).
- `IMMICHEYE_LABEL_SEPARATOR=space` joins words with spaces (`Cal Smith`). Blue Iris shows these
  intact in the memo (`person:85%,Cal Smith:62%`). They haven't been tried in Blue Iris's "To
  confirm" / "To cancel" boxes, so keep underscores if you filter alerts on names.
- `IMMICHEYE_NAME_MAP` sets a label for particular people, as JSON:
  `{"Bea Jones (Mum)": "mum"}`.
- Labels never end in a digit: Blue Iris treats a trailing number as a registration number
  (`cal1` is `cal`), so digits at the end are removed.
- Accents are dropped (`Zoë` → `zoe`). Names in other scripts keep their letters
  (`Иван Петров` → `иван_петров`), but haven't been tried with Blue Iris. A name with no letters
  or digits becomes `unnamed`, with a warning in the log.
- Avoid labels that are also object names (`person`, `car`, `dog`…); Blue Iris would confuse them.

Changing the label style re-registers everyone in CodeProject.AI under the new spelling at the
next refresh. Old alerts keep the old spelling.

### Blue Iris can't change faces

ImmichEye answers Blue Iris's face `register` and `delete` calls itself; they never reach
CodeProject.AI. Register is refused with "Faces are managed in Immich (ImmichEye)", and delete
is acknowledged and ignored.

This matters because **opening Blue Iris's faces view deletes every face on the AI server that
isn't in Blue Iris's own faces folder**. Without ImmichEye in the way, that would wipe
CodeProject.AI's synced faces. More in [CodeProject.AI](docs/codeproject-ai.md).

## Checking it's working

ImmichEye has three admin endpoints on `127.0.0.1:8000` (`IMMICHEYE_ADMIN_BIND` and
`IMMICHEYE_ADMIN_PORT`). They have no password, so keep them off untrusted networks.

| Endpoint | What it does |
|---|---|
| `GET /immicheye/status` | Gallery, request counts, errors and optional features (below) |
| `POST /immicheye/refresh` | Rebuild the gallery from Immich and sync CodeProject.AI now |
| `POST /immicheye/explain` | Score an image you post (field `image`): every face with its five best matches |

**The status page:**

| Field | Meaning |
|---|---|
| `people`, `faces`, `faces_per_label` | Who is in the gallery, and how many reference faces each has |
| `embeddings` | Reference faces including their low-res copies |
| `built_at`, `refreshing` | When the gallery was last built (epoch seconds), and whether a refresh is running |
| `last_error` | The last refresh problem, or `null`. Recognition keeps using the previous gallery |
| `requests` | Face requests since start |
| `unavailable` | Requests answered without recognition because Immich ML failed or the gallery wasn't ready |
| `cpai_sync`, `cpai_sync_error` | CodeProject.AI sync mode and its last error |
| `cpai_fallback`, `match_threshold`, `match_margin` | Settings in use |
| `events` | [Face events](docs/face-events.md): `sent`, `failed`, `dropped`, `last_error` |
| `late_names` | [Late names](docs/names-in-alerts.md#late-names): `enabled`, `updated`, `failed`, `last_error` |

**Watching recognitions:** every face request is logged to `data/recognitions.jsonl`, one line
each, with the faces found and their top three matches:

```sh
tail -f data/recognitions.jsonl
```

The fields are described in [Accuracy and tuning](docs/accuracy.md#the-recognition-log).

## Troubleshooting

| Symptom | Check |
|---|---|
| `last_error` is a `PermissionError` on `/data` | The container runs as uid 1000: `sudo chown -R 1000:1000 data`, then refresh or restart. |
| No names at all | On the status page `faces` should be above zero and `last_error` `null`. Then `tail -f data/recognitions.jsonl` while someone walks past. No new lines means Blue Iris isn't sending face requests: check the AI port, that the camera has facial recognition on, and that `person` is detected. |
| Lines in the log, but no faces | Look at `size`: under about 1280 px wide means the camera isn't using the main stream for AI. |
| Faces found, all `unknown` | Set `IMMICHEYE_SAVE_FACES=500` and look at the crops in `data/faces`. Tops of heads and blur are expected. Clear frontal faces scoring 0.30–0.40 for the right person suggest lowering the threshold a little ([Tuning](docs/accuracy.md#tuning)). |
| Wrong names | The log shows the top three scores. Two people close together usually means mixed-up faces in Immich. Raise `IMMICHEYE_MATCH_MARGIN` or the threshold. |
| Someone named in the log but not on the alert | Blue Iris lists only the faces in the one image it picked. Turn on [names shared across an alert and late names](docs/names-in-alerts.md). |
| `unavailable` keeps rising | ImmichEye can't reach Immich ML. `curl $IMMICH_ML_URL/ping` from the ImmichEye host should return `pong`. |
| `late_names.failed` rising | See `late_names.last_error`. Usually a wrong URL, user or password, or a user that isn't a Blue Iris admin. |
| Blue Iris's faces view shows your Immich people | Expected: it shows ImmichEye's list. |
| ImmichEye won't start | `docker compose logs immicheye`. A setting with an invalid value stops it at startup, and the last lines say which. |
| Port 8000 already in use | Set `IMMICHEYE_ADMIN_PORT` to a free port. |
| Using Blue Iris 6's built-in AI | Not supported. Switch the provider to CodeProject.AI. |

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

| File | |
|---|---|
| `immicheye/server.py` | The endpoints and request handling |
| `immicheye/gallery.py` | Reference faces, labels and matching |
| `immicheye/tiles.py` | Searching large frames in tiles |
| `immicheye/alerts.py` | Names shared across an alert's images |
| `immicheye/blueiris.py` | Late names, through Blue Iris's JSON API |
| `immicheye/events.py` | Face events |
| `immicheye/cpai_sync.py` | CodeProject.AI sync |
| `immicheye/immich.py`, `ml.py` | Immich and Immich ML clients |
| `immicheye/config.py` | Settings |
| `nginx/default.conf.template` | The router |
| `scripts/evaluate.py` | Accuracy test on your own library |

## Privacy

ImmichEye sends Immich previews and camera frames only to your own Immich ML and CodeProject.AI
servers. If enabled, it also writes names into your Blue Iris alert memos and sends face events
(embeddings and face crops) to `IMMICHEYE_EVENTS_URL`. It reads from Immich with a read-only key.

In `data/` it stores reference face crops and embeddings, a log of recognitions and, if enabled,
crops of faces from your cameras. Recognising people on camera may be regulated where you live;
check before pointing cameras at public spaces.

## Licence

MIT. See [LICENSE](LICENSE).
