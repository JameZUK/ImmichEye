# Names in alerts

ImmichEye can recognise someone and Blue Iris still sends the alert without a name. Two optional
features fix most of these cases. Both are off by default.

| | Fixes | Turn on with |
|---|---|---|
| [Names shared across an alert](#names-shared-across-an-alert) | The person was named in one of the alert's images, but not the image Blue Iris picked | `IMMICHEYE_ALERT_SHARE=1` |
| [Late names](#late-names) | The person only faced the camera after the alert had gone out | `IMMICHEYE_BLUEIRIS_URL`, `_USER` and `_PASSWORD` |

Use both: they cover different moments of the same visit.

## What happens during an alert

1. **Trigger.** Motion triggers the camera.
2. **Analysis.** Blue Iris sends a burst of images, the trigger image plus the camera's
   "additional post-trigger images", to the AI. Each image goes to object detection, and those
   with a person also to face recognition. All images in a burst share the alert's file name,
   e.g. `Front.20260101_120000Z.1234.3-1.jpg`.
3. **Confirmation.** When every image has been answered, Blue Iris picks the best one (most
   likely the one with the highest `person` score) and confirms the alert. It writes that
   image's labels into the alert's memo, e.g. `person:91%,cal:78%`, and sends notifications.
4. **Afterwards.** While the camera stays triggered, Blue Iris keeps sending frames, about one a
   second, with plain file names like `Front.20260101_120007Z.jpg`. These come from the camera's
   object tracking (static object analysis); in testing they stopped when it was turned off.

A name can be missed at step 3, because only the picked image's faces reach the memo, or arrive
only at step 4, after the memo is written.

## Names shared across an alert

With `IMMICHEYE_ALERT_SHARE=1`, every reply ImmichEye gives during an alert's analysis (step 2)
also includes the names it found in that alert's other images. Whichever image Blue Iris picks,
its memo gets the names.

- **Shared names** are those given with confidence (not `unknown`). Each is reported with its own
  score and the box from the image it was found in.
- **Waiting for a name.** A reply with no name of its own waits for one to turn up in the alert's
  other images, until `IMMICHEYE_ALERT_HOLD_MS` (default 3000 ms) after the alert's first image.
  The deadline is per alert, not per image: Blue Iris keeps sending the alert's other images
  meanwhile, so an alert where nobody is named isn't held up by much. In testing, alerts with
  nobody named were confirmed as fast as before.
- **Only alert images take part.** Frames from step 4 are answered as usual.
- **In the log**, a reply that waited has `held_ms`, and one that received names from other
  images lists them under `shared`.

It can't help when nobody shows a usable face during the analysis. That's what late names are
for.

## Late names

People often face the camera only after the alert has gone out: they turn round at the door,
or walk towards the camera after getting out of a car. Blue Iris keeps sending frames (step 4)
and ImmichEye may name them, but Blue Iris never updates an alert's memo by itself.

With late names on, ImmichEye remembers each camera's latest alert. When it names someone in
that alert's images, or in the camera's frames within `IMMICHEYE_LATE_NAMES_S` (default 60 s) of
the alert, and the memo doesn't have them, it adds them through Blue Iris's JSON API:

```
person:86%,unknown,car:83%   →   person:86%,Cal Smith:74%,unknown,car:83%
```

**What gets updated:** the alert in Blue Iris (its memo and search), and any app that reads the
memo when it fetches alerts from Blue Iris afterwards. **What doesn't:** a
push notification or MQTT message Blue Iris already sent.

### Setting it up

1. In Blue Iris, **Settings → Users**, add a user for ImmichEye with **Administrator** access.
   Blue Iris only lets admins change a memo. Give it a password using plain Latin characters:
   Blue Iris's login can't handle emoji or other scripts.
2. Make sure Blue Iris's web server is reachable from ImmichEye (**Settings → Web server**).
3. In `.env`:
   ```
   IMMICHEYE_BLUEIRIS_URL=http://blueiris.lan:81
   IMMICHEYE_BLUEIRIS_USER=immicheye
   IMMICHEYE_BLUEIRIS_PASSWORD=...
   ```
4. `docker compose up -d`, then check that `late_names.enabled` is `true` on the status page.

`.env` now holds an admin password for Blue Iris, so keep it private.

### How names are written

- **Order.** Names go after the memo's first label and any names already there, highest score
  first, then `unknown`, then other labels.
- **Percentages.** Each name gets a percentage like Blue Iris's own labels (`Cal Smith:74%`).
  Some apps skip labels without one.
- **The 49-character limit.** Blue Iris keeps only the first 49 characters of any memo, its own
  included. To make room, ImmichEye leaves out other labels first (counting from the end), then
  `unknown`, then the lowest-scoring names. After `person:NN%` there's room for two names of up
  to 14 characters each. Keep labels short (`IMMICHEYE_LABELS=first`) if several people often
  arrive together.
- **Several people.** Each new name updates the memo again, also while an earlier update is
  still under way.
- **An alert's flag** (Blue Iris's "flagged" status) is kept.

### Timing

- **Updates wait for Blue Iris to confirm the alert.** Blue Iris writes its own memo then, over
  anything written earlier. The first update is never sooner than 6 s after the alert's first
  image, and waits up to 45 s for confirmation. With a long analysis burst, confirmation can be
  10 s or more after the trigger.
- **Cancelled alerts are left alone**, including alerts Blue Iris marks as an occupied (static)
  object.
- **A new trigger's first frames** reach ImmichEye a second or two before its alert images. So a
  name from a frame outside an alert waits 3 s and then goes to whichever alert is the camera's
  latest, rather than to the camera's previous alert.
- **If an update fails** (Blue Iris unreachable, or the alert not confirmed in time), the names
  are tried again when they're seen again, at most every 10 s.

### Limits

- Late names only reach a memo; a notification already sent keeps the original text.
- If Blue Iris rewrites the alert's memo after ImmichEye's update, the added name is lost. This
  has been seen once, on an alert where the camera seems to have re-triggered.
- Names are only found in frames Blue Iris sends. If the camera stops triggering, or its object
  tracking is off, no frames arrive after the alert.

### Checking it

- The status page shows `late_names`: `updated` (memos changed), `failed` and `last_error`.
- Each update is logged:
  ```
  INFO immicheye.blueiris: Front alert @12345.bvr memo: 'person:86%,unknown' -> 'person:86%,Cal Smith:74%,unknown'
  ```
