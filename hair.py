"""
Real-time hairstyle overlay.

Anchors a pre-made transparent hair PNG to the head using MediaPipe Face Mesh,
scales and rotates it to the current pose, and composites it with the room's
lighting matched onto the asset so it doesn't look like a pasted-on wig.

Honest scope, stated up front so nobody expects more than it gives:
  * This is a *hairstyle try-on*: the user picks from a wardrobe of hair PNGs.
    It is NOT extracting hair from an arbitrary uploaded photo.
  * A PNG is a single viewpoint. It tracks position/scale/roll from the face
    mesh, so it holds up well front-facing and degrades on large head turns —
    a flat asset has no data for the side of the head. That limit is inherent
    to 2D overlay, not a blending bug.

Blending strategy, chosen from measured cost on the target hardware:
  * Default: fast alpha composite + Lab luminance match (~6ms/frame). The
    luminance match is the cheap form of the "harmonise lighting" step — it
    pulls the asset's brightness toward the surrounding frame so it sits in
    the room light.
  * Optional HAIR_SEAMLESS=1: cv2.seamlessClone(MIXED_CLONE). Higher-quality
    gradient blend, but ~60ms/frame on this class of GPU pod (it's a CPU-only
    OpenCV routine whose cost scales with the hair size, not the frame), so it
    roughly halves frame-rate. Offered for users who want the look and will
    trade FPS for it.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("hair")

# Face Mesh anchor landmarks (from the plan): forehead top and the two ear
# regions. These give a stable centre, width and roll for placing the asset.
FOREHEAD = 10
LEFT_EAR = 234
RIGHT_EAR = 454
CHIN = 152


class HairOverlay:
    """Places and blends a hair asset onto the head each frame."""

    def __init__(self, model_dir: str, asset_dir: Optional[str] = None):
        self.model_dir = model_dir
        # Assets ship WITH the code (committed to the repo), so they live next
        # to this file at <app>/hair_assets — not under /models, which is a
        # mounted volume the repo doesn't populate.
        if asset_dir:
            self.asset_dir = asset_dir
        else:
            here = os.path.dirname(os.path.abspath(__file__))
            self.asset_dir = os.path.join(here, "hair_assets")
        self.enabled = False
        self.seamless = os.environ.get("HAIR_SEAMLESS", "0") == "1"
        # Vertical placement of the asset centre, as a fraction of face height
        # above the forehead. Higher = asset sits higher on the head. Negative
        # pushes it down onto the face. Tune live with set_placement.
        self.y_offset = float(os.environ.get("HAIR_Y_OFFSET", "0.15"))
        # Width multiplier vs ear-to-ear span. Hair is wider than the face and
        # wraps the sides of the head, so this is >1.
        self.scale_k = float(os.environ.get("HAIR_SCALE", "1.7"))

        self._mesh = None
        self._asset_bgr: Optional[np.ndarray] = None
        self._asset_alpha: Optional[np.ndarray] = None
        self._current_style: Optional[str] = None
        self._init_mesh()

    # ---- setup -----------------------------------------------------------

    def _init_mesh(self) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            model_path = os.path.join(self.model_dir, "face_landmarker.task")
            if not os.path.exists(model_path):
                self._fetch_landmarker(model_path)

            opts = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
            )
            self._mesh = vision.FaceLandmarker.create_from_options(opts)
            self._mp = mp
            self._ts = 0
            logger.info("Hair overlay: FaceLandmarker ready")
        except Exception as e:
            logger.error("Hair overlay unavailable (mediapipe: %s)", e)
            self._mesh = None

    def _fetch_landmarker(self, dest: str) -> None:
        import urllib.request

        url = (
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/face_landmarker.task"
        )
        logger.info("Fetching face_landmarker model …")
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        urllib.request.urlretrieve(url, dest)

    def list_styles(self) -> list[str]:
        """Available hair asset names (PNG filenames without extension)."""
        if not os.path.isdir(self.asset_dir):
            return []
        return sorted(
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(self.asset_dir, "*.png"))
        )

    def set_style(self, name: Optional[str]) -> bool:
        """Load a hair PNG by name. None disables the overlay."""
        if not name:
            self.enabled = False
            self._current_style = None
            return True
        path = os.path.join(self.asset_dir, f"{name}.png")
        if not os.path.exists(path):
            logger.warning("Hair style %r not found in %s", name, self.asset_dir)
            return False
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[2] < 4:
            logger.warning("Hair asset %r has no alpha channel", name)
            return False
        self._asset_bgr = img[:, :, :3]
        self._asset_alpha = (img[:, :, 3].astype(np.float32) / 255.0)
        self._current_style = name
        self.enabled = True
        logger.info("Hair style set to %r (%dx%d)", name, img.shape[1], img.shape[0])
        return True

    def set_placement(
        self,
        scale: Optional[float] = None,
        y_offset: Optional[float] = None,
    ) -> None:
        """Adjust fit live, without a rebuild.

        The right values depend on how the specific PNG was cropped (how much
        scalp/forehead it includes), so this lets you dial it in on the fly.
        """
        if scale is not None:
            self.scale_k = float(scale)
        if y_offset is not None:
            self.y_offset = float(y_offset)
        logger.info("Hair placement: scale=%.2f y_offset=%.2f", self.scale_k, self.y_offset)

    @property
    def active(self) -> bool:
        return self.enabled and self._mesh is not None and self._asset_bgr is not None

    # ---- per-frame -------------------------------------------------------

    def apply(self, bgr: np.ndarray) -> np.ndarray:
        if not self.active:
            return bgr
        try:
            anchors = self._anchors(bgr)
            if anchors is None:
                return bgr
            return self._place(bgr, anchors)
        except Exception as e:
            logger.debug("hair apply error: %s", e)
            return bgr

    def _anchors(self, bgr: np.ndarray):
        """Return placement for the hair asset from face landmarks.

        Returns (center_x, center_y, width, roll_deg). The asset is sized to
        cover the head (wider than the face) and positioned so its vertical
        centre sits around the crown, so it wraps the head instead of floating
        above the forehead.
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        self._ts += 33
        res = self._mesh.detect_for_video(mp_img, self._ts)
        if not res.face_landmarks:
            return None
        h, w = bgr.shape[:2]
        lm = res.face_landmarks[0]

        def px(i):
            return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)

        fore, le, re_, chin = px(FOREHEAD), px(LEFT_EAR), px(RIGHT_EAR), px(CHIN)
        ear_center = (le + re_) / 2.0
        ear_vec = re_ - le

        # Head width from ears, widened: hair covers the sides of the head, so
        # it's wider than the ear-to-ear span. scale_k defaults higher now.
        head_w = np.linalg.norm(ear_vec) * self.scale_k
        roll = np.degrees(np.arctan2(ear_vec[1], ear_vec[0]))

        # Up axis (forehead direction) and face height for scaling placement.
        up = fore - ear_center
        n = np.linalg.norm(up)
        up = up / n if n > 0 else np.array([0, -1], np.float32)
        face_h = np.linalg.norm(chin - fore)

        # Place the asset's CENTRE near the crown: start at the forehead and
        # move up by a fraction of face height. The asset is scaled so this
        # centre lands mid-hair, letting it drape down past the forehead and
        # around the sides. y_offset tunes how high it sits.
        center = fore + up * face_h * self.y_offset
        return float(center[0]), float(center[1]), float(head_w), float(roll)

    def _place(self, frame, anchors):
        cx, cy, width, roll = anchors
        asset = self._asset_bgr
        alpha = self._asset_alpha
        ah, aw = asset.shape[:2]

        # Scale asset so its width matches the head width.
        scale = width / aw if aw else 1.0
        new_w = max(2, int(aw * scale))
        new_h = max(2, int(ah * scale))
        a_bgr = cv2.resize(asset, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        a_alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Rotate to head roll.
        M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), -roll, 1.0)
        a_bgr = cv2.warpAffine(a_bgr, M, (new_w, new_h), flags=cv2.INTER_LINEAR)
        a_alpha = cv2.warpAffine(a_alpha, M, (new_w, new_h), flags=cv2.INTER_LINEAR)

        # Destination box.
        x1 = int(cx - new_w / 2)
        y1 = int(cy - new_h / 2)
        x2, y2 = x1 + new_w, y1 + new_h

        # Clip to frame.
        fx1, fy1 = max(0, x1), max(0, y1)
        fx2, fy2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if fx2 <= fx1 or fy2 <= fy1:
            return frame
        ax1, ay1 = fx1 - x1, fy1 - y1
        ax2, ay2 = ax1 + (fx2 - fx1), ay1 + (fy2 - fy1)

        a_bgr = a_bgr[ay1:ay2, ax1:ax2]
        a_alpha = a_alpha[ay1:ay2, ax1:ax2]
        region = frame[fy1:fy2, fx1:fx2]

        if self.seamless:
            blended = self._seamless(region, a_bgr, a_alpha)
        else:
            blended = self._fast_blend(region, a_bgr, a_alpha)
        frame[fy1:fy2, fx1:fx2] = blended
        return frame

    def _fast_blend(self, region, hair, alpha):
        """Alpha composite with the hair's brightness matched to the region.

        The luminance match is the cheap harmonisation step: it pulls the
        asset toward the frame's light so it doesn't glow or sit dark against
        the room. ~6ms/frame.
        """
        if alpha.max() <= 0:
            return region
        reg_L = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)[:, :, 0].mean()
        hair_L = cv2.cvtColor(hair, cv2.COLOR_BGR2LAB)[:, :, 0].mean()
        gain = np.clip(reg_L / max(1.0, hair_L), 0.6, 1.5)
        gain = gain * 0.4 + 0.6  # partial, so hair keeps its own modelling
        hair_f = np.clip(hair.astype(np.float32) * gain, 0, 255)
        a = alpha[:, :, None]
        out = hair_f * a + region.astype(np.float32) * (1 - a)
        return out.astype(np.uint8)

    def _seamless(self, region, hair, alpha):
        """cv2.seamlessClone MIXED_CLONE — higher quality, much slower."""
        mask = (alpha > 0.15).astype(np.uint8) * 255
        if mask.sum() == 0:
            return region
        ys, xs = np.where(mask > 0)
        cx = int((xs.min() + xs.max()) / 2)
        cy = int((ys.min() + ys.max()) / 2)
        try:
            return cv2.seamlessClone(
                hair, region, mask, (cx, cy), cv2.MIXED_CLONE
            )
        except Exception:
            return self._fast_blend(region, hair, alpha)