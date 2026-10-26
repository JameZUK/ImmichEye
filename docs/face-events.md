# Face events

ImmichEye can send every face it detects to another service, such as a visitor log that counts
who came to the door and groups unknown visitors by face. The receiver gets the embedding
ImmichEye has already computed, so it needs no face model of its own.

Face events are off unless `IMMICHEYE_EVENTS_URL` is set. Sending never slows down or changes
recognition.

## Turning it on

In `.env`:

```
IMMICHEYE_EVENTS_URL=http://visitors.lan:8090/api/events
IMMICHEYE_EVENTS_TOKEN=<the receiver's token, if it wants one>
IMMICHEYE_EVENTS_CAMERAS=Front,Back
IMMICHEYE_EVENTS_FRAME_PX=640
```

| Setting | Default | |
|---|---|---|
| `IMMICHEYE_EVENTS_URL` | empty: off | Where to POST events |
| `IMMICHEYE_EVENTS_TOKEN` | | Sent as `Authorization: Bearer <token>` |
| `IMMICHEYE_EVENTS_CAMERAS` | empty: all | Comma-separated camera short names to send (any case). Saves traffic; the receiver can filter too |
| `IMMICHEYE_EVENTS_FRAME_PX` | `0`: off | Also send the whole frame, its long side scaled to at most this, for context such as parcels or uniforms |
| `IMMICHEYE_EVENTS_QUEUE_MB` | `64` | Most camera frames, in MB, waiting to be sent |

Then `docker compose up -d`. The status page's `events` shows `sent`, `failed`, `dropped` and
`last_error`.

If the receiver also has another way of importing faces (for example reading
`recognitions.jsonl` and saved crops), turn that off so faces aren't counted twice.

## What is sent, and when

- **One event per face request with at least one face.** Frames with no face aren't sent, and
  nothing is sent for `/immicheye/explain`.
- **In the background.** Events are queued and sent one at a time. When 200 are waiting, or
  `IMMICHEYE_EVENTS_QUEUE_MB` of frames, the oldest is dropped and counted as `dropped`.
- **No retries.** Each POST has a 5 s timeout. A failure is counted, and logged at most once a
  minute. Losing one frame of a visit is expected; a receiver should cope.

## Event format

`POST` to `IMMICHEYE_EVENTS_URL` with `Content-Type: application/json`:

```json
{
  "v": 1,
  "source": "immicheye",
  "id": "5f0c2a5e-6d1b-4c84-9d0e-2b1f7f3a9c10",
  "ts": 1792938423.881,
  "camera": "Front",
  "file": "Front.20261025_142653Z.jpg",
  "size": [3840, 2160],
  "model": "buffalo_l",
  "threshold": 0.40,
  "margin": 0.05,
  "frame": "<base64 JPEG, only with IMMICHEYE_EVENTS_FRAME_PX>",
  "faces": [
    {
      "box": [995, 830, 1041, 896],
      "det": 0.705,
      "embedding": "<base64 of 512 little-endian float32, L2-normalised>",
      "result": "cal",
      "score": 0.512,
      "top": [["cal", 0.512], ["bea", 0.231], ["dan", 0.164]],
      "crop": "<base64 JPEG>"
    }
  ]
}
```

| Field | |
|---|---|
| `v` | Format version, currently 1 |
| `id` | A new UUID per event. Ignore repeats of an id, so retries can be added safely later |
| `ts` | When the request reached ImmichEye (epoch seconds), as in `recognitions.jsonl` |
| `camera` | Blue Iris's camera short name: the file name up to the first `.`. `null` if there's no `.` |
| `file` | Blue Iris's file name. Alert images have a record number (`Front.20261025_142653Z.1234.3-1.jpg`); frames after an alert don't |
| `size` | Frame width and height in pixels, or `null` if unreadable |
| `model`, `threshold`, `margin` | ImmichEye's model and match settings. Refuse embeddings from a different model than the ones you already hold: they can't be compared |
| `frame` | The whole frame, only with `IMMICHEYE_EVENTS_FRAME_PX` |
| `faces[].box` | The face in frame pixels: `[x1, y1, x2, y2]` |
| `faces[].det` | Detector score: how sure it is that this is a face |
| `faces[].embedding` | 512 numbers describing the face, as base64 little-endian float32, already scaled to length 1, so the dot product of two embeddings is their similarity |
| `faces[].result` | Exactly what Blue Iris was told: a label or `unknown` |
| `faces[].score` | The best person's score, named or not |
| `faces[].top` | The three closest people: `[label, score]` |
| `faces[].crop` | The face with some surroundings (the box grown by half its longer side on every edge), JPEG, at most 384 px |

Decoding an embedding in Python:

```python
import base64, numpy as np
vec = np.frombuffer(base64.b64decode(face["embedding"]), dtype="<f4")   # shape (512,)
```

Labels follow ImmichEye's label settings (`IMMICHEYE_LABELS`, `_LABEL_CASE`, `_LABEL_SEPARATOR`).
If you key known people by label, expect the style to change one day and be able to merge
spellings.

## Replies

Any 2xx reply counts as delivered; anything else counts as `failed`. For example, a receiver
might answer `202` with `{"accepted": 1}`, `200` for a repeated `id`, `401` for a wrong token and
`409` for a different model.
