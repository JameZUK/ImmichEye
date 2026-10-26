# Accuracy and tuning

- [Tuning](#tuning)
- [Tools](#tools)
- [The recognition log](#the-recognition-log)
- [Test results](#test-results)
- [Speed](#speed)

## Tuning

Three settings decide every result:

| Setting | Default | What it does |
|---|---|---|
| `IMMICHEYE_DETECT_MIN_SCORE` | 0.70 | How sure the detector must be that something is a face. On real CCTV, clear faces score about 0.8 and the tops and backs of heads 0.5–0.6. 0.70 is Immich's own default. |
| `IMMICHEYE_MATCH_THRESHOLD` | 0.40 | How similar a face must be to a person to be named. Known people typically score 0.45–0.65 on CCTV, strangers under 0.25. |
| `IMMICHEYE_MATCH_MARGIN` | 0.05 | How far the best person must be ahead of the runner-up. Stops a face that looks equally like two relatives being given either name. |

**Rules of thumb:**

- People you know often come back `unknown` on clear, frontal faces: lower the threshold a
  little, e.g. to 0.35.
- You see wrong names: raise the threshold or the margin.
- Strangers matter more to you than naming people at a distance: 0.45 names fewer strangers
  (see [Small faces](#small-faces)).

Change one thing at a time and watch the [recognition log](#the-recognition-log) for a few days.

## Tools

```sh
# Every request as it happens: frame size, faces, top three matches, result
tail -f data/recognitions.jsonl

# Score any image: every face with its five best matches
curl -F image=@frame.jpg http://127.0.0.1:8000/immicheye/explain

# Accuracy on your own Immich library, at a range of thresholds
docker compose exec immicheye python scripts/evaluate.py
```

**Saved faces.** `IMMICHEYE_SAVE_FACES=500` keeps crops of the last 500 faces Blue Iris sent, in
`data/faces`. Each file name holds the time, the result, the best match, its score and the face
width, e.g. `20261024-153924-424_0_cal_cal-0.60_93px.jpg`. Looking through them is the quickest
way to see why faces come back `unknown`.

**`scripts/evaluate.py`** tests faces from your Immich library that the gallery didn't use, and
reports how many are named correctly, wrongly or not at all, at several thresholds. Options:

| Option | |
|---|---|
| `--min-px 30 --max-px 60` | Size range of test faces (default 30–60 px, the closest to CCTV) |
| `--scale 0.5` | Shrink test faces further, e.g. to imitate a sub-stream |
| `--per-person 15` | Test faces per person |
| `--no-lowres` | Leave the low-res reference copies out |
| `--cpai` | Also score CodeProject.AI on the same faces (after a sync) |

## The recognition log

With `IMMICHEYE_LOG_RECOGNITIONS=1` (default), every face request adds one line to
`data/recognitions.jsonl`. At 20 MB it moves to `recognitions.jsonl.1`, replacing the previous
old file.

```json
{"ts": 1792938423.88, "file": "Front.20261025_142653Z.1234.3-1.jpg", "bytes": 412345,
 "min_confidence": "0.40", "size": [3840, 2160], "tiles": 8, "ms": 212,
 "faces": [{"box": [995, 830, 1041, 896], "det": 0.705,
            "top": [["cal", 0.512], ["bea", 0.231], ["dan", 0.164]], "result": "cal"}],
 "held_ms": 0}
```

| Field | |
|---|---|
| `ts` | When the request arrived (epoch seconds) |
| `file` | Blue Iris's file name: the camera's short name, then the time. Alert images also have a record number (`.1234.3-1`) |
| `bytes`, `size` | Image size in bytes and pixels |
| `min_confidence` | Blue Iris's minimum confidence (ImmichEye ignores it) |
| `tiles` | Tiles searched as well as the whole frame (large frames only) |
| `ms` | Time spent finding and embedding faces |
| `faces[].box`, `det` | Where the face is, and the detector's score |
| `faces[].top` | The three closest people and their scores |
| `faces[].result` | What Blue Iris was told: a label or `unknown` |
| `faces[].crop` | The saved crop's file name, with `IMMICHEYE_SAVE_FACES` |
| `held_ms`, `shared` | [Names shared across an alert](names-in-alerts.md#names-shared-across-an-alert): how long the reply waited, and names it took from other images |
| `gave_up` | Blue Iris stopped waiting for this reply |
| `error` | The upload wasn't an image |

## Test results

Results depend on your photos and cameras, so run `scripts/evaluate.py` and check the log for
your own figures.

### Held-out Immich faces

Measured with `scripts/evaluate.py` on one library: 52 people, 2,178 reference faces. Test faces
are each person's Immich faces that the gallery didn't use, at the default settings. Small faces
are the closest stand-in for CCTV.

| Test faces | ImmichEye correct / **wrong** / unknown | CodeProject.AI (same people synced) correct / **wrong** / unknown |
|---|---|---|
| Small, 30–60 px | 86% / **2.5%** / 11% | 41% / **53%** / 7% |
| Larger, 80–400 px | 97% / **1.2%** / 2% | 52% / **45%** / 3% |

These were measured before the low-res reference copies were added; see below.

CodeProject.AI's figures are at Blue Iris's usual minimum confidence (0.40). Raising it removes
right and wrong names together: at 0.80 it got 20% right and 8% wrong on larger faces. Part of
the "wrong" share is the library's own duplicates, the same person under two spellings.

### Small faces

**Low-res reference copies.** 449 held-out faces from 62 people, at threshold 0.40 and margin
0.05. Half size is closest to faces on a sub-stream or at a distance. "Stranger named" is how
often a face is given a name when its own person is taken out of the gallery: a pessimistic
stand-in for a visitor Immich doesn't know, since the closest "stranger" is often a relative.

| Test faces | Without copies: correct / **wrong** / stranger named | With copies (default) |
|---|---|---|
| 30–60 px | 90.2% / **1.2%** / 7.9% | 92.9% / **1.2%** / 10.4% |
| Half size, about 15–30 px | 73.5% / **1.0%** / 5.8% | 85.0% / **0.5%** / 8.7% |

At threshold 0.45 with the copies, small faces are named as often as at 0.40 without them
(73%) and strangers less often (4.5%).

**Tiling.** A face pasted into an empty 4K frame was only found by the detector at about 100 px
or larger. With 1280 px tiles it was found down to 32 px, the smallest size tried. A 4K frame
is searched in 8 tiles plus the whole frame, and a 1080p frame in 2 plus the whole frame. Frames
up to 1280 px aren't tiled.

### Real CCTV

On walks past two cameras running main stream, one person was named in about half to 70% of
the frames where a face was found (9 of 13 on the closer camera), with scores of 0.44–0.62. No
frame was given the wrong name. Every `unknown` frame, checked with saved crops, was the top or
back of a head, a motion-blurred face, or a face under 10 px from a sub-stream camera.

**With tiling and low-res copies.** On a 4K camera, no face had been found in 478 frames before
tiling. On one walk past afterwards, faces (56–125 px) were found in 12 of its 22 frames and the
person was named in 10 of the 13 faces, at 0.41–0.64. A 2560×1920 camera found faces of only
24–39 px, which were 6–10 px and undetectable at the whole-frame scale, and named the 34–39 px
ones. Faces under about 30 px still mostly came back `unknown` (scores 0.23–0.39), and no face
was given the wrong name.

## Speed

The same 15 frames (5 photos, each at the three sizes Blue Iris typically sends) were posted to
both servers from the host they run on, three times each after a warm-up, both on an Nvidia GPU,
without tiling. Times are the whole request, upload included: median, with the slowest in
brackets.

| Frame size | ImmichEye | CodeProject.AI |
|---|---|---|
| 640×480 (typical sub-stream) | **34 ms** (39) | 73 ms (80) |
| 1920×1080 | **53 ms** (67) | 84 ms (300) |
| 3840×2160 (4K main stream) | 141 ms (172) | 134 ms (530) |

Tiling adds detector passes to large frames: on the same host a 4K request took about 280 ms with
tiles (140 ms without) and a 1080p one 120 ms (55 ms). Blue Iris sends about 20 images per
alert at once, so they queue: in live use, median times were about 210 ms for 2560×1920 and
400 ms for 1080p and 4K frames, with the slowest under a second. Sub-stream frames are
unaffected. A request occasionally takes a few seconds, when Immich ML reloads its model after an
idle spell. Set `IMMICHEYE_TILE_PX=0` if tiling is too slow for your hardware.
