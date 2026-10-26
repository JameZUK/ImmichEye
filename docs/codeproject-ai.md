# CodeProject.AI

ImmichEye sits in front of your CodeProject.AI server. Everything except face requests passes
through to it untouched. This page covers the three ways ImmichEye does touch CodeProject.AI:
it protects its face database, keeps it in step with Immich, and decides what happens when
ImmichEye can't answer.

## Blue Iris can't change faces

Blue Iris's face `register` and `delete` calls are answered by ImmichEye and never reach
CodeProject.AI:

- **Register** is refused with "Faces are managed in Immich (ImmichEye)".
- **Delete** is acknowledged and ignored.

This matters because **opening Blue Iris's faces view sends a delete for every face on the AI
server that isn't in Blue Iris's own faces folder**. Passed through, that would wipe
CodeProject.AI's synced faces. If you ever point Blue Iris straight at CodeProject.AI again, avoid
that view; if you do open it, the next sync puts the faces back.

## Keeping CodeProject.AI in step with Immich

After each gallery refresh, ImmichEye registers the same people in CodeProject.AI's own face
database. If you point Blue Iris straight back at CodeProject.AI, it still knows everyone, using
its own (weaker) matching.

CodeProject.AI keeps one averaged face per name, which describes nobody well. So ImmichEye
registers each person as up to `IMMICHEYE_CPAI_PROTOTYPES` (default 4) entries, `cal1`…`cal4`.
Each entry averages one group of similar-looking faces, such as with and without glasses.

`IMMICHEYE_CPAI_SYNC` sets how much of the database ImmichEye looks after:

| Mode | |
|---|---|
| `off` | Never touch CodeProject.AI's face database. |
| `managed` (default) | Add, update and remove only the entries ImmichEye created. It never overwrites or deletes anyone else's, and skips a number that's already taken. |
| `exclusive` | As `managed`, and delete every other entry, so CodeProject.AI knows exactly your Immich people. Use this once Immich is your only source of faces. |

ImmichEye records what it registered in `data/cpai_sync.json`. Before switching an existing
install to `exclusive`, back up CodeProject.AI's `faceembedding.db` (in its data directory,
`/etc/codeproject/ai` in the Docker image).

A warning like `register cal4: No face detected` means CodeProject.AI's own detector found no
face in any photo of that group, usually side-on faces. The person's other entries are fine.
Problems show as `cpai_sync_error` on the status page.

## When ImmichEye or Immich ML is down

By default (`IMMICHEYE_CPAI_FALLBACK=0`), face requests get **no face** until things are back.
Person, vehicle and every other detection carry on normally; only names stop. The status page
counts these replies as `unavailable`.

ImmichEye doesn't fall back to CodeProject.AI by default because a wrong name is worse than no
name:

- An unnamed person still gets you the person alert, and you look at the clip.
- A wrong name can make you ignore an alert you needed, or satisfy a rule that quietens known
  people. It also stays in alert history for good.
- Even with the same people synced, CodeProject.AI named about half of test faces wrongly (see
  [Accuracy](accuracy.md)).

`IMMICHEYE_CPAI_FALLBACK=1` lets CodeProject.AI answer face recognition during outages, both from
ImmichEye (when Immich ML fails) and from the router (when ImmichEye itself is down). Register
and delete never fall back. If you turn it on, raise Blue Iris's face minimum confidence.

The better fix is keeping outages short: make sure Immich's machine-learning container starts on
boot. ImmichEye's own containers use `restart: unless-stopped`.

## What Blue Iris's minimum confidence means here

Blue Iris sends its minimum confidence with each face request. CodeProject.AI uses it as its
match threshold, on a scale where Blue Iris's usual 0.40 accepts almost any face. ImmichEye
ignores it and decides with its own threshold and margin.

The confidences ImmichEye returns are on CodeProject.AI's scale, `(similarity + 1) / 2`, so they
look familiar to Blue Iris. ImmichEye's default threshold of 0.40 shows as 70%, which is why
Blue Iris's minimum should be 70% or lower.
