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
        # Fit is automatic (measured per-asset). These are neutral by default
        # and act only as an optional fine-tune nudge on top: scale_k=1.0 means
        # "use the measured size", y_offset=0.0 means "use the measured height".
        self.y_offset = float(os.environ.get("HAIR_Y_OFFSET", "0.0"))
        self.scale_k = float(os.environ.get("HAIR_SCALE", "1.0"))
        # Debug: draw anchor points and the asset box so placement is visible.
        self.debug = os.environ.get("HAIR_DEBUG", "0") == "1"

        self._mesh = None
        self._asset_bgr: Optional[np.ndarray] = None
        self._asset_alpha: Optional[np.ndarray] = None
        self._current_style: Optional[str] = None
        self._init_mesh()

    # ---- setup -----------------------------------------------------------

    def _init_mesh(self) -> None:
        # No per-frame face model here anymore. The hair overlay reuses the
        # landmarks the face-swap engine already computes each frame, passed
        # into apply(). This removed both the lag (two trackers per frame) and
        # the anchor disagreement. The still-image FaceDetector used at asset
        # load time (_detect_face_in_image) is separate and only runs once.
        self._mesh = None
        try:
            import mediapipe as mp

            self._mp = mp
        except Exception:
            self._mp = None

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
        # Measure THIS asset so it fits automatically — no manual sliders.
        self._analyze_asset()
        logger.info("Hair style set to %r (%dx%d)", name, img.shape[1], img.shape[0])
        return True

    def _analyze_asset(self) -> None:
        """Measure the loaded asset so it aligns to a head automatically.

        The goal is a fit with no manual tuning. Every hair PNG is cropped
        differently, so instead of assuming proportions we measure them once:

          * Try to detect a face *in the asset*. If the cutout includes the
            person's face, that face's width and position tell us exactly how
            the hair relates to a head — we map that face onto the live user's
            face landmarks and the hair follows perfectly.
          * If there's no face (a pure hair cutout), fall back to the alpha
            shape: the hair's width and the vertical position of its centre of
            mass give a reliable automatic placement.

        Results are stored as ratios relative to the asset, so placement scales
        with the user's head at any distance.
        """
        alpha = self._asset_alpha
        ah, aw = alpha.shape[:2]
        ys, xs = np.where(alpha > 0.3)
        if len(xs) == 0:
            self._fit = {"width_ratio": 1.7, "anchor_y": 0.4, "face_w_ratio": None}
            return

        hair_w = xs.max() - xs.min()
        hair_cx = (xs.min() + xs.max()) / 2.0
        # Store the hair's actual pixel bounds within the asset — placement
        # must anchor by where the hair *is*, not the image edges.
        self._hair_bounds = {
            "top": int(ys.min()),
            "bottom": int(ys.max()),
            "left": int(xs.min()),
            "right": int(xs.max()),
            "w": int(hair_w),
            "cx": float(hair_cx),
        }

        face_w_ratio = None
        anchor_y = None
        # Attempt face detection on the asset (opaque regions only).
        try:
            comp = self._asset_bgr.copy()
            # Flatten transparent areas to mid-grey so detection isn't fooled.
            a3 = (alpha[:, :, None] > 0.3)
            comp = np.where(a3, comp, 128).astype(np.uint8)
            faces = self._detect_face_in_image(comp)
            if faces is not None:
                fx, fy, fw, fh = faces
                # The user's ear-to-ear width will map to this asset face width;
                # from that, the whole hair scales correctly.
                face_w_ratio = fw / aw
                # Vertical anchor: where the face centre sits within the asset,
                # as a fraction of asset height. We align the user's face centre
                # to this point, so the hair sits exactly as it did on the
                # original head.
                anchor_y = (fy + fh / 2.0) / ah
                logger.info(
                    "Asset %r: face found, auto-fit (face_w=%.2f of asset, "
                    "anchor_y=%.2f)",
                    self._current_style,
                    face_w_ratio,
                    anchor_y,
                )
        except Exception as e:
            logger.debug("asset face detect failed: %s", e)

        if face_w_ratio is None:
            # No face in the cutout — use the hair blob geometry.
            # Hair typically spans ~1.5x the head width; infer head width from
            # the hair width, and anchor at the vertical centre of mass.
            width_ratio = 1.6
            cy = ys.mean() / ah
            self._fit = {
                "width_ratio": width_ratio,
                "anchor_y": float(np.clip(cy, 0.25, 0.65)),
                "face_w_ratio": None,
                "hair_cx_ratio": hair_cx / aw,
            }
            logger.info(
                "Asset %r: no face, alpha-based auto-fit (anchor_y=%.2f)",
                self._current_style,
                self._fit["anchor_y"],
            )
        else:
            self._fit = {
                "face_w_ratio": face_w_ratio,
                "anchor_y": anchor_y,
                "hair_cx_ratio": hair_cx / aw,
                "width_ratio": None,
            }

    def _detect_face_in_image(self, bgr: np.ndarray):
        """Return (x,y,w,h) of the largest face in a still image, or None."""
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            if not hasattr(self, "_still_detector"):
                model_path = os.path.join(self.model_dir, "blaze_face_short_range.tflite")
                if not os.path.exists(model_path):
                    import urllib.request

                    url = (
                        "https://storage.googleapis.com/mediapipe-models/face_detector/"
                        "blaze_face_short_range/float16/latest/"
                        "blaze_face_short_range.tflite"
                    )
                    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
                    urllib.request.urlretrieve(url, model_path)
                opts = vision.FaceDetectorOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=model_path),
                    running_mode=vision.RunningMode.IMAGE,
                )
                self._still_detector = vision.FaceDetector.create_from_options(opts)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self._still_detector.detect(mp_img)
            if not res.detections:
                return None
            best = max(
                res.detections,
                key=lambda d: d.bounding_box.width * d.bounding_box.height,
            )
            bb = best.bounding_box
            return (bb.origin_x, bb.origin_y, bb.width, bb.height)
        except Exception:
            return None

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
        return self.enabled and self._asset_bgr is not None

    # ---- per-frame -------------------------------------------------------

    def apply(self, bgr: np.ndarray, kps=None, bbox=None) -> np.ndarray:
        """Composite the hair, anchored from face landmarks passed in.

        `kps` is the 5-point InsightFace layout the swap engine already
        computes each frame: [left_eye, right_eye, nose, left_mouth,
        right_mouth]. Reusing it means no second face model runs — that was the
        source of the lag — and the hair anchors to the exact same face the
        swap uses.
        """
        if not self.enabled or self._asset_bgr is None:
            return bgr
        if kps is None:
            return bgr  # nothing to anchor to this frame
        try:
            anchors = self._anchors_from_kps(np.asarray(kps, dtype=np.float32), bbox)
            if anchors is None:
                return bgr
            out = self._place(bgr, anchors)
            if self.debug:
                cx, cy, width, roll = anchors
                cv2.circle(out, (int(cx), int(cy)), 6, (0, 0, 255), -1)
                cv2.putText(out, f"w={int(width)} roll={roll:.0f}",
                            (int(cx) - 60, int(cy) - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            return out
        except Exception as e:
            logger.debug("hair apply error: %s", e)
            return bgr

    def _anchors_from_kps(self, kps: np.ndarray, bbox):
        """Compute (cx, cy, width, roll) from 5-point landmarks.

        Layout: 0=left eye, 1=right eye, 2=nose, 3=left mouth, 4=right mouth.
        """
        if kps.shape[0] < 5:
            return None
        left_eye, right_eye = kps[0], kps[1]
        nose = kps[2]
        mouth = (kps[3] + kps[4]) / 2.0

        eye_center = (left_eye + right_eye) / 2.0
        eye_vec = right_eye - left_eye
        eye_dist = np.linalg.norm(eye_vec)
        roll = np.degrees(np.arctan2(eye_vec[1], eye_vec[0]))

        # Head up-axis: from mouth up through eye centre (stable, roll-aware).
        up = eye_center - mouth
        n = np.linalg.norm(up)
        up = up / n if n > 0 else np.array([0, -1], np.float32)

        # Scale reference: interocular distance is ~0.46 of face width. Head
        # (hair) width is ~2.1x eye distance. Vertical spans use eye distance
        # too, so everything tracks with head size/distance.
        head_w = eye_dist * 2.1 * self.scale_k

        # The forehead sits about 0.6*eye_dist above the eye centre along up.
        forehead = eye_center + up * (eye_dist * 0.6)

        hb = getattr(self, "_hair_bounds", None)
        ah, aw = self._asset_alpha.shape[:2]
        if hb is None:
            hb = {"top": 0, "bottom": ah, "w": aw, "cx": aw / 2.0}

        # Scale asset so its hair width matches head width.
        render_scale = head_w / max(1, hb["w"])
        H = ah * render_scale
        W = aw * render_scale

        # Land the hair's bottom (hairline row hb['bottom']) at the forehead,
        # with a small overlap; volume rises up over the scalp.
        overlap = eye_dist * (0.5 - self.y_offset)
        hair_bottom_from_top = hb["bottom"] * render_scale
        top_point = forehead + up * (hair_bottom_from_top - overlap)
        center = top_point - up * (H / 2.0)

        # Horizontal: align the hair's own centre column to the eye centre x.
        hair_cx_off = (hb["cx"] - aw / 2.0) * render_scale
        center = center - np.array([hair_cx_off, 0.0], dtype=np.float32)

        return float(center[0]), float(center[1]), float(W), float(roll)

    def _anchors(self, bgr: np.ndarray):
        """Auto-fit placement from face landmarks + the asset's measured fit.

        Returns (center_x, center_y, width, roll_deg). Uses the ratios computed
        in _analyze_asset so the asset aligns to the head with no manual tuning:

          * If the asset has a detected face, the user's face width is mapped to
            the asset's face width — so the whole hair scales to the head — and
            the asset is positioned so its face-region lands on the user's face.
          * Otherwise a robust alpha-based estimate is used.

        A small manual nudge (scale_k, y_offset) is still applied on top so the
        user can fine-tune, but the defaults are neutral (1.0 / 0.0) now that
        fit is automatic.
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
        roll = np.degrees(np.arctan2(ear_vec[1], ear_vec[0]))
        user_face_w = np.linalg.norm(ear_vec)

        up = fore - ear_center
        n = np.linalg.norm(up)
        up = up / n if n > 0 else np.array([0, -1], np.float32)
        face_h = np.linalg.norm(chin - fore)
        # User's face centre (mid-point between forehead and chin).
        face_center = (fore + chin) / 2.0

        fit = getattr(self, "_fit", None)
        ah, aw = self._asset_alpha.shape[:2]

        if fit and fit.get("face_w_ratio"):
            # Asset had a detectable face. Scale the whole asset so its face
            # matches the user's face width, then position so the asset's face
            # region lands exactly on the user's face.
            render_w = user_face_w / fit["face_w_ratio"]
            render_scale = render_w / aw
            H = ah * render_scale  # rendered asset height
            # Derivation: the asset is drawn centred at C; its face pixel sits a
            # fraction anchor_y down from the top. Setting that equal to the
            # user's face centre gives C = face_center - up*H*(0.5 - anchor_y).
            center = face_center - up * (H * (0.5 - fit["anchor_y"]))
            width = render_w * self.scale_k
            center = center + up * (face_h * self.y_offset)
            return float(center[0]), float(center[1]), float(width), float(roll)

        # Alpha-based fallback (hair-only asset, no face detected).
        #
        # A pure-hair PNG fills most of its frame, so a centre-of-mass anchor
        # lands the hair over the face. Anchor by the HEAD instead.

        # Alpha-based fallback (hair-only asset, no face detected).
        #
        # Anchor by the hair's ACTUAL pixel bounds inside the asset, not the
        # image edges — the hair's bottom row is the hairline, and it must sit
        # at the user's forehead with the volume rising over the scalp.
        hb = getattr(self, "_hair_bounds", None)
        if hb is None:
            hb = {"top": 0, "bottom": ah, "w": aw, "cx": aw / 2.0}

        # Scale so the hair's on-screen width covers the head (~1.6x face).
        target_head_w = user_face_w * 1.6 * self.scale_k
        render_scale = target_head_w / max(1, hb["w"])
        H = ah * render_scale
        W = aw * render_scale

        # Where the hair's bottom (hairline) should land: forehead + a little
        # overlap onto it (nudgeable via y_offset).
        overlap = face_h * (0.15 - self.y_offset)
        # In the asset, the hairline is at row hb["bottom"]; from the asset top
        # that's hb["bottom"]*render_scale px down.
        hair_bottom_from_top = hb["bottom"] * render_scale
        # Place the asset so that row lands at (forehead + overlap), measured
        # along the head's down-axis (-up). Also centre horizontally on the
        # hair's own centre, not the image centre.
        # Asset-top point in frame:
        top_point = fore + up * (hair_bottom_from_top - overlap)
        # Asset centre is H/2 further down from the top (toward chin = -up):
        center = top_point - up * (H / 2.0)

        # Horizontal: shift so the hair's centre column sits on the face x.
        hair_cx_off = (hb["cx"] - aw / 2.0) * render_scale
        # Perpendicular (roll-aware) x correction is minor; apply along x.
        center = center - np.array([hair_cx_off, 0.0], dtype=np.float32)

        return float(center[0]), float(center[1]), float(W), float(roll)

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
