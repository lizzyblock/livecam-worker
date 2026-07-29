"""
Real-time region recolouring that runs alongside the face swap.

The insight behind this module: a full "body swap" is unsolved in real time,
but the effects people actually want from one — the shirt matching the target,
the hands matching the swapped face's skin, a clean background — are each just
*segment a region, recolour it*. Segmentation runs in a few milliseconds and
recolouring is a cheap per-pixel operation, so all three fit comfortably in the
live budget.

Three independent stages, each toggleable:

  * skin match  — shift the hue/brightness of exposed skin (hands, neck, arms)
                  toward the swapped face's skin tone, so they read as one
                  person. This is the single biggest "tell" of a face swap.
  * shirt tint  — recolour the torso/clothing region to a target colour.
  * background  — blur or replace everything that isn't the person.

MediaPipe's selfie/multiclass segmentation provides the masks. It's Apache-2.0
licensed, so unlike InsightFace it carries no commercial restriction.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("recolor")

# MediaPipe's multiclass selfie segmentation labels.
# 0 background, 1 hair, 2 body-skin, 3 face-skin, 4 clothes, 5 others(accessories)
BG, HAIR, BODY_SKIN, FACE_SKIN, CLOTHES, OTHERS = range(6)


class Recolorizer:
    """Segments the frame once, then applies whichever recolour stages are on."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self._seg = None
        self._seg_kind = None  # "multiclass" | "selfie" | None
        self._frame_no = 0

        # How often to re-run segmentation. The mask changes slowly, so
        # reusing it for a couple of frames roughly halves the cost with no
        # visible difference.
        self.seg_every = max(1, int(os.environ.get("SEG_EVERY", "2")))
        self._cached_mask: Optional[np.ndarray] = None

        # --- stage toggles (driven live from the dashboard) ---
        self.skin_match = os.environ.get("SKIN_MATCH", "0") == "1"
        self.shirt_on = os.environ.get("SHIRT_RECOLOR", "0") == "1"
        self.bg_mode = os.environ.get("BG_MODE", "off")  # off|blur|replace

        # target shirt colour as BGR; None until set
        self.shirt_bgr: Optional[Tuple[int, int, int]] = None
        # target skin tone (median Lab of the swapped face), set per frame
        self._face_skin_lab: Optional[np.ndarray] = None
        # replacement background image (BGR), resized lazily
        self._bg_img: Optional[np.ndarray] = None

        self._init_segmenter()

    # ---- setup -----------------------------------------------------------

    def _init_segmenter(self) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            path = os.path.join(self.model_dir, "selfie_multiclass.tflite")
            if not os.path.exists(path):
                self._fetch_model(path)

            opts = vision.ImageSegmenterOptions(
                base_options=mp_python.BaseOptions(model_asset_path=path),
                output_category_mask=True,
                running_mode=vision.RunningMode.VIDEO,
            )
            self._seg = vision.ImageSegmenter.create_from_options(opts)
            self._seg_kind = "multiclass"
            self._mp = mp
            logger.info("Segmentation ready (multiclass): shirt/skin/bg available")
        except Exception as e:
            logger.error(
                "Could not init MediaPipe segmentation (%s). Recolour stages "
                "will be inert.",
                e,
            )
            self._seg = None

    def _fetch_model(self, dest: str) -> None:
        import httpx

        url = (
            "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
            "selfie_multiclass_256x256/float32/latest/"
            "selfie_multiclass_256x256.tflite"
        )
        logger.info("Fetching selfie-multiclass segmentation model …")
        os.makedirs(self.model_dir, exist_ok=True)
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)

    @property
    def active(self) -> bool:
        """True if any stage is on AND the segmenter loaded."""
        return self._seg is not None and (
            self.skin_match or self.shirt_on or self.bg_mode != "off"
        )

    # ---- per-frame -------------------------------------------------------

    def _segment(self, bgr: np.ndarray, ts_ms: int) -> Optional[np.ndarray]:
        """Return a HxW uint8 category mask, re-running on the interval."""
        if self._seg is None:
            return None
        self._frame_no += 1
        if (
            self._cached_mask is not None
            and self._frame_no % self.seg_every != 1
            and self._cached_mask.shape == bgr.shape[:2]
        ):
            return self._cached_mask

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._seg.segment_for_video(mp_img, ts_ms)
        cat = result.category_mask.numpy_view()
        if cat.shape != bgr.shape[:2]:
            cat = cv2.resize(
                cat, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        self._cached_mask = cat
        return cat

    def apply(self, bgr: np.ndarray, ts_ms: int) -> np.ndarray:
        """Run the enabled stages. Returns the modified frame (in place ok)."""
        if not self.active:
            return bgr
        mask = self._segment(bgr, ts_ms)
        if mask is None:
            return bgr

        if self.shirt_on and self.shirt_bgr is not None:
            bgr = self._recolor_region(bgr, mask == CLOTHES, self.shirt_bgr)

        if self.skin_match and self._face_skin_lab is not None:
            skin = (mask == BODY_SKIN)
            bgr = self._match_skin(bgr, skin, self._face_skin_lab)

        if self.bg_mode != "off":
            person = mask != BG
            bgr = self._apply_background(bgr, person)

        return bgr

    # ---- stages ----------------------------------------------------------

    def set_face_skin(self, face_bgr: np.ndarray) -> None:
        """Record the swapped face's skin tone as a Lab reference.

        Called with the aligned swapped-face crop. The median over the centre
        (cheek/forehead area) avoids eyes, brows and lips skewing the tone.
        """
        if face_bgr is None or face_bgr.size == 0:
            return
        h, w = face_bgr.shape[:2]
        centre = face_bgr[int(h * 0.35) : int(h * 0.75), int(w * 0.3) : int(w * 0.7)]
        if centre.size == 0:
            return
        lab = cv2.cvtColor(centre, cv2.COLOR_BGR2LAB).reshape(-1, 3)
        self._face_skin_lab = np.median(lab, axis=0)

    @staticmethod
    def _feather(mask_bool: np.ndarray, px: int = 5) -> np.ndarray:
        """Bool mask → soft float alpha in [0,1] for seamless blending."""
        m = (mask_bool.astype(np.float32)) * 255.0
        k = px * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), 0) / 255.0
        return m[:, :, None]

    def _recolor_region(self, bgr, mask_bool, target_bgr) -> np.ndarray:
        """Tint a region toward a target colour while keeping its shading.

        Works in Lab: replace the a/b (colour) channels with the target's and
        keep L (luminance), so folds, shadows and highlights of the garment
        survive — a flat fill would look painted-on.
        """
        if not mask_bool.any():
            return bgr
        alpha = self._feather(mask_bool)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        tgt = cv2.cvtColor(
            np.uint8([[target_bgr]]), cv2.COLOR_BGR2LAB
        )[0, 0].astype(np.float32)
        recolored = lab.copy()
        recolored[:, :, 1] = tgt[1]
        recolored[:, :, 2] = tgt[2]
        out = lab * (1 - alpha) + recolored * alpha
        return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _match_skin(self, bgr, mask_bool, target_lab) -> np.ndarray:
        """Shift exposed skin toward the face's tone.

        Only a/b are shifted, and L only partially, so the hands keep their own
        lighting and texture but take on the face's colour — enough to stop the
        pale-face-on-dark-hands mismatch without looking like flat paint.
        """
        if not mask_bool.any():
            return bgr
        alpha = self._feather(mask_bool, px=7)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        cur = lab[mask_bool].reshape(-1, 3)
        if cur.size == 0:
            return bgr
        cur_med = np.median(cur, axis=0)

        shift = np.zeros(3, np.float32)
        shift[0] = (target_lab[0] - cur_med[0]) * 0.4  # partial luminance
        shift[1] = (target_lab[1] - cur_med[1]) * 0.9  # colour a
        shift[2] = (target_lab[2] - cur_med[2]) * 0.9  # colour b

        shifted = lab + shift
        out = lab * (1 - alpha) + shifted * alpha
        return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    def set_background(self, img_bgr: Optional[np.ndarray]) -> None:
        self._bg_img = img_bgr

    def _apply_background(self, bgr, person_bool) -> np.ndarray:
        alpha = self._feather(person_bool, px=3)
        if self.bg_mode == "blur":
            bg = cv2.GaussianBlur(bgr, (0, 0), 12)
        else:  # replace
            if self._bg_img is None:
                bg = cv2.GaussianBlur(bgr, (0, 0), 20)
            else:
                if self._bg_img.shape[:2] != bgr.shape[:2]:
                    self._bg_img = cv2.resize(
                        self._bg_img, (bgr.shape[1], bgr.shape[0])
                    )
                bg = self._bg_img
        out = bgr.astype(np.float32) * alpha + bg.astype(np.float32) * (1 - alpha)
        return out.astype(np.uint8)
