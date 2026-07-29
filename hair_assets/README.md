# Hair assets

Drop transparent-background hairstyle PNGs here, one per style. The filename
(without .png) is the style name shown in the app: `bob.png` → "bob".

## Requirements for a good asset

- **PNG with a real alpha channel** — the transparency defines the hair edge.
  A white or solid background will show as a rectangle.
- **Front-facing, upright, centred** — the overlay scales it to head width and
  rotates it to head roll, but it can't change the *viewpoint*. A front-view
  asset looks right front-facing and degrades as the user turns.
- **Soft alpha at the strand edges** — hard cutouts look like a wig. Feather
  the mask edge slightly (a few px of alpha falloff) for believable blending.
- **Sized generously** — 512px+ wide so it stays sharp when scaled to a close
  webcam head.

## Where to get them

- Commission/source PNG hair cutouts (many stock sites sell transparent hair)
- Cut them from portraits with a segmentation tool + manual alpha cleanup
- The cleaner the alpha, the better the result — this is the single biggest
  quality lever, more than any blending setting

## Tuning (env vars on the pod)

- `HAIR_Y_OFFSET` (0.35) — vertical placement above the forehead
- `HAIR_SCALE` (1.15) — width multiplier relative to ear-to-ear distance
- `HAIR_SEAMLESS` (0) — set 1 for cv2.seamlessClone blending. Higher quality
  gradient/lighting match, but ~60ms/frame (halves FPS). Default off; the fast
  path already matches luminance at ~6ms.
