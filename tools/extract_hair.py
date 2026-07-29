#!/usr/bin/env python3
"""
Extract a transparent hair PNG from a portrait photo.

Uses MediaPipe's multiclass selfie segmentation to isolate the hair region,
cleans and feathers the alpha edge (the single biggest quality factor for the
overlay), crops tightly to the hair, and writes a ready-to-use asset into the
worker's hair_assets/ folder.

Usage:
    python tools/extract_hair.py PORTRAIT.jpg --name bob
    python tools/extract_hair.py PORTRAIT.jpg --name bob --out ../hair_assets
    python tools/extract_hair.py PORTRAIT.jpg --name bob --feather 6 --preview

Notes:
  * Best input: a clear, front-facing, well-lit portrait where the hair is
    fully visible against a reasonably plain background.
  * The output is a *front-view* asset. Like all 2D hair, it looks right
    front-facing in the app and degrades on big head turns — that's inherent
    to a flat cutout, not this script.
  * Segmentation isn't perfect. --preview writes a side-by-side so you can
    check the mask before trusting it; --feather softens the edge.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

import cv2
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)
# Class indices for the multiclass model.
HAIR = 1


def _load_segmenter(model_path: str):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not os.path.exists(model_path):
        print(f"Downloading segmentation model → {model_path}")
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, model_path)

    opts = vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        output_category_mask=True,
        running_mode=vision.RunningMode.IMAGE,
    )
    return mp, vision.ImageSegmenter.create_from_options(opts)


def extract_hair(
    image_path: str,
    out_dir: str,
    name: str,
    feather: int = 5,
    pad: int = 20,
    preview: bool = False,
) -> str:
    img = cv2.imread(image_path)
    if img is None:
        sys.exit(f"Could not read image: {image_path}")

    h, w = img.shape[:2]
    # Upscale small inputs so the 256px model has detail to work with.
    if max(h, w) < 512:
        s = 512 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]

    model_path = os.path.join(
        os.path.dirname(__file__), "selfie_multiclass_256x256.tflite"
    )
    mp, seg = _load_segmenter(model_path)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = seg.segment(mp_img)
    cat = result.category_mask.numpy_view()
    if cat.shape != (h, w):
        cat = cv2.resize(cat, (w, h), interpolation=cv2.INTER_NEAREST)

    hair_mask = (cat == HAIR).astype(np.uint8) * 255
    if hair_mask.sum() == 0:
        sys.exit(
            "No hair detected. Try a clearer, front-facing portrait with the "
            "hair well separated from the background."
        )

    # Clean the mask: close small holes, keep the largest blob, drop specks.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(hair_mask)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        hair_mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    # Feather the edge into a soft alpha — the difference between "hair" and
    # "cardboard wig" in the overlay.
    alpha = hair_mask.astype(np.float32)
    if feather > 0:
        k = feather * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    alpha = np.clip(alpha, 0, 255).astype(np.uint8)

    # Crop tightly to the hair (plus padding) so the asset isn't mostly empty.
    ys, xs = np.where(hair_mask > 0)
    x1 = max(0, xs.min() - pad)
    y1 = max(0, ys.min() - pad)
    x2 = min(w, xs.max() + pad)
    y2 = min(h, ys.max() + pad)

    bgr = img[y1:y2, x1:x2]
    a = alpha[y1:y2, x1:x2]
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = a

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.png")
    cv2.imwrite(out_path, bgra)

    coverage = 100.0 * (hair_mask > 0).sum() / (h * w)
    print(f"✓ Wrote {out_path}  ({x2 - x1}x{y2 - y1}px, hair was {coverage:.1f}% of photo)")

    if preview:
        # Composite the cutout over a checkerboard so the alpha is visible.
        ph, pw = a.shape
        board = np.zeros((ph, pw, 3), np.uint8)
        t = 16
        for yy in range(0, ph, t):
            for xx in range(0, pw, t):
                if ((yy // t) + (xx // t)) % 2 == 0:
                    board[yy : yy + t, xx : xx + t] = 200
                else:
                    board[yy : yy + t, xx : xx + t] = 120
        af = (a.astype(np.float32) / 255.0)[:, :, None]
        comp = (bgr.astype(np.float32) * af + board.astype(np.float32) * (1 - af)).astype(
            np.uint8
        )
        prev_path = os.path.join(out_dir, f"{name}_preview.png")
        cv2.imwrite(prev_path, comp)
        print(f"  preview (check the edges): {prev_path}")

    return out_path


def main():
    ap = argparse.ArgumentParser(description="Extract a transparent hair PNG from a portrait.")
    ap.add_argument("image", help="portrait photo (jpg/png)")
    ap.add_argument("--name", required=True, help="style name → <name>.png")
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "..", "hair_assets"),
        help="output folder (default: ../hair_assets)",
    )
    ap.add_argument("--feather", type=int, default=5, help="edge softness in px (default 5)")
    ap.add_argument("--pad", type=int, default=20, help="crop padding in px (default 20)")
    ap.add_argument("--preview", action="store_true", help="also write a checkerboard preview")
    args = ap.parse_args()

    extract_hair(
        args.image,
        os.path.abspath(args.out),
        args.name,
        feather=args.feather,
        pad=args.pad,
        preview=args.preview,
    )


if __name__ == "__main__":
    main()
