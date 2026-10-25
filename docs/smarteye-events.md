# Face event sink

**Status:** implemented (`immicheye/events.py`). User-facing docs are in the README under
"Face events"; this is the design and the receiver's contract.

A visitor log (for example SmartEye) can build a database from door-camera faces: who came,
how often, when, with unknown visitors clustered into identities. ImmichEye already detects
every face Blue Iris sends and computes its embedding. The sink hands each face to such a
service, so the receiver needs no inference of its own.

The sink is optional, off by default, and must never slow down or break `face/recognize`.

## Settings

| Setting | Default | |
|---|---|---|
| `IMMICHEYE_EVENTS_URL` | empty (off) | Where to POST events, e.g. `http://visitors.example.lan:8090/api/events` |
| `IMMICHEYE_EVENTS_TOKEN` | empty | Sent as `Authorization: Bearer <token>` when set |
| `IMMICHEYE_EVENTS_CAMERAS` | empty (all) | Comma-separated camera short names to send (`Front,Back`). The receiver may filter too; this just saves traffic |
| `IMMICHEYE_EVENTS_FRAME_PX` | `0` (off) | Also send the whole frame, scaled so its long side is at most this many px (e.g. `640`), for context (parcels, uniforms) |

## When to send

From `recognize()` in `server.py`, after `predictions` are built, for every request where at
least one face was detected. Frames with no face are not sent.

- The request path only queues the JPEG and the face data; it never awaits the POST.
- Background sender: one `asyncio.Queue(maxsize=200)` drained by a single task started in
  `lifespan`. If the queue is full, the **oldest** event is dropped and counted.
- POST with a 5 s timeout through the shared `httpx.AsyncClient`. No retries: a dropped
  event loses one frame of one visit, which the receiver should tolerate. On error, log at
  most one warning per minute and count it.
- Crops and frame encoding run in `asyncio.to_thread` (PIL work is CPU-bound), inside the
  sender task, not the request.
- Nothing is sent for `/immicheye/explain`.

`/immicheye/status` reports `"events": {"url_set": true, "sent": n, "failed": n, "dropped": n, "last_error": ...}`.

## Camera name

Blue Iris names uploads `<camera short name>.<YYYYMMDD_HHMMSSZ>[...].jpg`, e.g.
`Front.20261025_142653Z.jpg` or `Drive.20261025_142653Z.3870.3-1.jpg`. The camera is
everything before the first `.`. If the filename is missing or has no `.`, `camera` is `null`.

## Payload

`POST {IMMICHEYE_EVENTS_URL}`, `Content-Type: application/json`:

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
  "frame": "<base64 JPEG, only if IMMICHEYE_EVENTS_FRAME_PX > 0>",
  "faces": [
    {
      "box": [995, 830, 1041, 896],
      "det": 0.705,
      "embedding": "<base64 of 512 little-endian float32, L2-normalised>",
      "result": "Cal Smith",
      "score": 0.512,
      "top": [["Cal Smith", 0.512], ["Bea", 0.231], ["Dan", 0.164]],
      "crop": "<base64 JPEG>"
    }
  ]
}
```

| Field | |
|---|---|
| `id` | A new UUID4 per request. A receiver should ignore repeats of an id, so a future retry is safe |
| `ts` | Epoch seconds when the request arrived (same as `recognitions.jsonl`) |
| `size` | Frame width and height, or `null` if the header couldn't be read |
| `model`, `threshold`, `margin` | `IMMICHEYE_MODEL`, `IMMICHEYE_MATCH_THRESHOLD`, `IMMICHEYE_MATCH_MARGIN`. A receiver should refuse embeddings from a different model than the one it already holds |
| `faces[].box` | Same clamped box as in `recognitions.jsonl`, in frame pixels |
| `faces[].det` | Detector score |
| `faces[].embedding` | `base64(f.embedding.astype("<f4").tobytes())`. Already L2-normalised by `ml.py` |
| `faces[].result` | Exactly the `userid` returned to Blue Iris: a label or `"unknown"` |
| `faces[].score` | Best person's score (cosine), whether or not it was named |
| `faces[].top` | Top three `[label, cosine]`, as logged |
| `faces[].crop` | The face with context, cut as `IMMICHEYE_SAVE_FACES` cuts it: box grown by half its longer side on every edge, clamped to the frame. JPEG quality 90, longer side at most 384 px |

Labels are whatever ImmichEye currently gives Blue Iris (`IMMICHEYE_LABELS`, `_LABEL_CASE`,
`_LABEL_SEPARATOR`). A receiver that keys known people by label should expect the style to
change and be able to merge spellings.

## Reply

Any 2xx is delivered; anything else is counted as a failure. SmartEye, for example, answers
`202` with `{"accepted": n_faces}`, `200` for a duplicate `id`, `401` for a bad token and
`409` for a model mismatch.

## Tests

In `tests/test_immicheye.py`, with `respx`:

1. With `IMMICHEYE_EVENTS_URL` unset, `recognize` makes no request to it.
2. With it set, a frame with one face produces one POST whose JSON has `camera` parsed from
   the filename, one face, an embedding that decodes to 512 floats with norm ≈ 1, a crop
   that PIL can open, and the same `result` as the recognition reply. A frame with no face
   produces no POST.
3. The sink failing (500, or `ConnectError`) does not change the recognition reply, and
   `status()["events"]["failed"]` goes up.
4. `IMMICHEYE_EVENTS_CAMERAS=Front` sends `Front.*` frames and skips the rest.

## Deployment

Add to ImmichEye's `.env`:

```
IMMICHEYE_EVENTS_URL=http://<receiver host>:8090/api/events
IMMICHEYE_EVENTS_TOKEN=<the receiver's ingest token>
IMMICHEYE_EVENTS_CAMERAS=Front,Back
IMMICHEYE_EVENTS_FRAME_PX=640
```

If the receiver has another way of importing faces (SmartEye can read `recognitions.jsonl`
and saved crops), turn that off so faces aren't counted twice.
