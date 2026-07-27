"""
Real-time full body swap engine.

Uses MediaPipe for person segmentation and pose estimation,
IDW-based warping for pose-guided deformation, and the existing
face swap engine as a fallback when body warping is not possible.

The reference person's full appearance (face, hair, clothes, body)
is warped to match the streamer's current pose and blended onto
the original frame.

Pipeline per frame:
  1. Segment the person in the current frame
  2. Estimate body pose keypoints
  3. Warp the reference person to match the current pose
  4. Blend the warped reference onto the original background
  5. If body warp fails → fall back to face-only swap

Design notes:
  * The warping uses Inverse Distance Weighting (IDW) on a coarse
    grid, then cv2.remap at full resolution. This keeps the cost
    under ~5ms regardless of frame size.
  * Pose similarity is checked before warping — very different poses
    produce bad warps, so we fall back to face swap instead.
  * Hair, clothes, and body come from the warped reference image.
    No separate hair/clothes model is needed — the reference person's
    entire appearance is transferred.
  * The face swap engine (inswapper_128) is used as a fallback when
    body warping is not possible, and optionally for face refinement
    after the body warp.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("body-swap")

try:
    import mediapipe as mp

    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False


from face_swap import decode_portrait  # noqa: F401

# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────


@dataclass
class BodySwapSource:
    """Prepared reference for full body swap.

    Computed once at session start from the reference portrait.
    """

    reference_image: np.ndarray  # BGR
    reference_mask: np.ndarray  # Person segmentation (0-1 float)
    reference_pose: Optional[np.ndarray]  # (33, 3): x, y, visibility
    face_source: Optional[object]  # SwapSource for face-swap fallback
    name: str
    ref_vis_idx: np.ndarray  # Indices of visible keypoints in reference
    has_body: bool  # Whether the reference has enough body for warping


# ─────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────


class BodySwapEngine:
    """Full body swap: face + hair + clothes + body."""

    def __init__(self, model_dir: str, det_size: int = 0):
        if not _MP_AVAILABLE:
            raise RuntimeError(
                "mediapipe is not installed — pip install mediapipe"
            )

        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        det = det_size or int(os.environ.get("DET_SIZE", "320"))

        # ── MediaPipe Selfie Segmentation ──
        self._segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=1  # 1 = landscape, better quality
        )

        # ── MediaPipe Pose ──
        self._pose_estimator = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=int(os.environ.get("POSE_COMPLEXITY", "1")),
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # ── Face swap engine (fallback + face refinement) ──
        from face_swap import FaceSwapEngine

        self.face_engine = FaceSwapEngine(model_dir, det_size=det)

        # ── Caching ──
        self.detect_every = max(1, int(os.environ.get("DETECT_EVERY", "3")))
        self._cached_pose: Optional[np.ndarray] = None
        self._cached_mask: Optional[np.ndarray] = None
        self._frame_no = 0

        # ── Blending ──
        self.blend_feather = int(os.environ.get("BLEND_FEATHER", "25"))

        # ── Warp grid resolution (lower = faster) ──
        self.warp_grid = int(os.environ.get("WARP_GRID", "32"))

        # ── IDW power parameter ──
        self.idw_alpha = float(os.environ.get("IDW_ALPHA", "2.0"))

        # ── Pose similarity threshold ──
        # 0 = always warp, 1 = only warp identical poses
        self.pose_threshold = float(os.environ.get("POSE_THRESHOLD", "0.25"))

        # ── Apply face swap after body warp for face refinement ──
        self.face_refine = os.environ.get("FACE_REFINE", "0") != "0"

        # ── Minimum visible keypoints for body warping ──
        self.min_body_keypoints = int(os.environ.get("MIN_BODY_KP", "8"))

        # ── Color transfer for hair/clothes when body warp fails ──
        self.color_transfer_fallback = os.environ.get(
            "COLOR_TRANSFER_FALLBACK", "1"
        ) != "0"

        # ── Timing ──
        self.t_segment = 0.0
        self.t_pose = 0.0
        self.t_warp = 0.0
        self.t_blend = 0.0
        self.n_frames = 0
        self._hits = 0
        self._misses = 0
        self._warp_hits = 0

        logger.info(
            "Body swap engine ready (MediaPipe + inswapper_128, "
            "grid=%d, feather=%d, pose_threshold=%.2f)",
            self.warp_grid,
            self.blend_feather,
            self.pose_threshold,
        )

    # ── Person segmentation ──

    def _segment_person(self, bgr: np.ndarray) -> np.ndarray:
        """Extract person segmentation mask (0-1 float)."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = self._segmenter.process(rgb)
        if results.segmentation_mask is None:
            return np.zeros(bgr.shape[:2], dtype=np.float32)
        return results.segmentation_mask.astype(np.float32)

    # ── Pose estimation ──

    def _estimate_pose(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        """Extract body pose keypoints (33 x 3)."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = self._pose_estimator.process(rgb)
        if not results.pose_landmarks:
            return None
        landmarks = results.pose_landmarks.landmark
        h, w = bgr.shape[:2]
        pose = np.zeros((33, 3), dtype=np.float32)
        for i, lm in enumerate(landmarks):
            pose[i] = [lm.x * w, lm.y * h, lm.visibility]
        return pose

    # ── Keypoint helpers ──

    @staticmethod
    def _visible_keypoints(pose: np.ndarray, min_vis: float = 0.5) -> np.ndarray:
        """Indices of visible keypoints."""
        if pose is None:
            return np.array([], dtype=int)
        return np.where(pose[:, 2] > min_vis)[0]

    @staticmethod
    def _pose_similarity(
        pose1: np.ndarray, pose2: np.ndarray, common: np.ndarray
    ) -> float:
        """Compute pose similarity (0-1, 1 = identical shape)."""
        if len(common) < 4:
            return 0.0
        p1 = pose1[common, :2]
        p2 = pose2[common, :2]

        def _normalize(pts):
            center = pts.mean(axis=0)
            dists = np.sqrt(np.sum((pts - center) ** 2, axis=1))
            scale = dists.max() + 1e-8
            return (pts - center) / scale

        n1 = _normalize(p1)
        n2 = _normalize(p2)
        diff = np.sqrt(np.mean((n1 - n2) ** 2))
        return max(0.0, 1.0 - diff)

    # ── Source preparation ──

    def prepare_source(
        self, portrait_bgr: np.ndarray, name: str
    ) -> Optional[BodySwapSource]:
        """Analyze reference image once → cached data for body swap."""
        mask = self._segment_person(portrait_bgr)
        pose = self._estimate_pose(portrait_bgr)

        # Prepare face swap source (always needed for fallback)
        face_source = self.face_engine.prepare_source(portrait_bgr, name)

        vis_idx = (
            self._visible_keypoints(pose) if pose is not None else np.array([], dtype=int)
        )

        # Check if the reference has enough body for warping.
        # A headshot-only portrait won't have enough body keypoints.
        has_body = len(vis_idx) >= self.min_body_keypoints

        if pose is None:
            logger.warning(
                "No pose detected in reference for %s — body swap disabled, "
                "face swap only",
                name,
            )
        elif not has_body:
            logger.info(
                "Reference %s has only %d visible keypoints (need %d) — "
                "body warp limited, face swap + color transfer",
                name,
                len(vis_idx),
                self.min_body_keypoints,
            )
        else:
            logger.info(
                "Prepared body swap source %r: %d visible keypoints, %dx%d",
                name,
                len(vis_idx),
                portrait_bgr.shape[1],
                portrait_bgr.shape[0],
            )

        return BodySwapSource(
            reference_image=portrait_bgr,
            reference_mask=mask,
            reference_pose=pose,
            face_source=face_source,
            name=name,
            ref_vis_idx=vis_idx,
            has_body=has_body,
        )

    # ── Main swap ──

    def swap_frame(self, frame_bgr: np.ndarray, source: BodySwapSource) -> np.ndarray:
        """Swap full body for one frame.

        Returns the original frame unchanged when no person is detected.
        """
        self._frame_no += 1
        self.n_frames += 1

        # ── Detect person + pose ──
        if self._frame_no % self.detect_every == 1 or self._cached_pose is None:
            t0 = time.perf_counter()
            mask = self._segment_person(frame_bgr)
            self.t_segment += time.perf_counter() - t0

            t0 = time.perf_counter()
            pose = self._estimate_pose(frame_bgr)
            self.t_pose += time.perf_counter() - t0

            self._cached_mask = mask
            self._cached_pose = pose
        else:
            mask = self._cached_mask
            pose = self._cached_pose

        # ── Decide: body warp or face-only? ──
        can_warp = (
            pose is not None
            and source.reference_pose is not None
            and source.has_body
        )

        if can_warp:
            target_vis = self._visible_keypoints(pose)
            common = np.intersect1d(target_vis, source.ref_vis_idx)
            can_warp = len(common) >= self.min_body_keypoints

            if can_warp:
                similarity = self._pose_similarity(
                    pose, source.reference_pose, common
                )
                can_warp = similarity > self.pose_threshold

        if not can_warp:
            # ── Fallback: face swap + optional color transfer ──
            self._misses += 1
            if self._misses in (30, 300) or self._misses % 900 == 0:
                logger.warning(
                    "Body warp not possible for %d frames — using face swap%s",
                    self._misses,
                    " + color transfer" if self.color_transfer_fallback else "",
                )

            result = frame_bgr
            if source.face_source is not None:
                result = self.face_engine.swap_frame(result, source.face_source)

            # Apply hair/clothes color transfer as a best-effort
            if self.color_transfer_fallback and source.has_body:
                result = self._apply_color_transfer(result, source, mask)

            return result

        # ── Body warp ──
        target_vis = self._visible_keypoints(pose)
        common = np.intersect1d(target_vis, source.ref_vis_idx)

        t0 = time.perf_counter()
        warped_ref, warped_mask = self._warp_to_pose(
            source, pose, frame_bgr.shape[:2], common
        )
        self.t_warp += time.perf_counter() - t0

        # Optional face refinement after body warp
        if self.face_refine and source.face_source is not None:
            warped_ref = self.face_engine.swap_frame(warped_ref, source.face_source)

        # ── Blend ──
        t0 = time.perf_counter()
        result = self._blend(warped_ref, frame_bgr, warped_mask, mask)
        self.t_blend += time.perf_counter() - t0

        self._hits += 1
        self._warp_hits += 1
        if self._hits == 1:
            logger.info("First successful body swap (warped)")

        return result

    # ── Warping ──

    def _warp_to_pose(
        self,
        source: BodySwapSource,
        target_pose: np.ndarray,
        target_shape: Tuple[int, int],
        common_vis: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Warp reference image to match target pose using IDW.

        Returns (warped_image, warped_mask).
        """
        ref_pose = source.reference_pose
        ref_img = source.reference_image
        ref_mask = source.reference_mask

        th, tw = target_shape

        src_pts = ref_pose[common_vis, :2]
        dst_pts = target_pose[common_vis, :2]

        # Compute warp maps
        map_x, map_y = self._compute_warp_maps(src_pts, dst_pts, th, tw)

        # Apply warping
        warped = cv2.remap(
            ref_img,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Warp the mask
        warped_mask = cv2.remap(
            (ref_mask * 255).astype(np.uint8),
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.float32) / 255.0

        return warped, warped_mask

    def _compute_warp_maps(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        target_h: int,
        target_w: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute reverse warp maps for cv2.remap using IDW.

        For each target pixel, find the source pixel via inverse distance
        weighted interpolation of the keypoint correspondences.
        """
        gs = self.warp_grid

        # Regular grid in target space
        gy = np.linspace(0, target_h - 1, gs, dtype=np.float32)
        gx = np.linspace(0, target_w - 1, gs, dtype=np.float32)
        grid_y, grid_x = np.meshgrid(gy, gx, indexing="ij")

        grid_points = np.stack(
            [grid_x.ravel(), grid_y.ravel()], axis=1
        )  # (gs*gs, 2)

        # Distances from each grid point to each target keypoint
        diff = grid_points[:, None, :] - dst_pts[None, :, :]  # (N, M, 2)
        dist = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-8)  # (N, M)

        # IDW weights
        alpha = self.idw_alpha
        weights = 1.0 / (dist ** alpha)  # (N, M)
        weights /= weights.sum(axis=1, keepdims=True)

        # Source position = weighted average of source keypoints
        src_positions = weights @ src_pts  # (N, 2)

        # Reshape to grid
        map_x_coarse = src_positions[:, 0].reshape(gs, gs).astype(np.float32)
        map_y_coarse = src_positions[:, 1].reshape(gs, gs).astype(np.float32)

        # Upscale to full resolution
        map_x = cv2.resize(
            map_x_coarse, (target_w, target_h), interpolation=cv2.INTER_CUBIC
        )
        map_y = cv2.resize(
            map_y_coarse, (target_w, target_h), interpolation=cv2.INTER_CUBIC
        )

        return map_x, map_y

    # ── Blending ──

    def _blend(
        self,
        warped: np.ndarray,
        original: np.ndarray,
        warped_mask: np.ndarray,
        person_mask: np.ndarray,
    ) -> np.ndarray:
        """Blend warped reference onto original frame."""
        h, w = original.shape[:2]

        # Ensure masks are the right size
        if warped_mask.shape[:2] != (h, w):
            warped_mask = cv2.resize(warped_mask, (w, h))
        if person_mask.shape[:2] != (h, w):
            person_mask = cv2.resize(person_mask, (w, h))

        # Combine: use warped person mask where the person is detected
        combined_mask = warped_mask * person_mask

        # Feather the mask for smooth transitions
        feather = self.blend_feather
        if feather > 0:
            k = feather * 2 + 1
            combined_mask = cv2.GaussianBlur(combined_mask, (k, k), 0)

        # Ensure 3-channel mask for broadcasting
        alpha = combined_mask[:, :, None] if combined_mask.ndim == 2 else combined_mask

        result = (
            warped.astype(np.float32) * alpha
            + original.astype(np.float32) * (1.0 - alpha)
        ).astype(np.uint8)

        return result

    # ── Color transfer (fallback when body warp fails) ──

    def _apply_color_transfer(
        self,
        frame: np.ndarray,
        source: BodySwapSource,
        person_mask: np.ndarray,
    ) -> np.ndarray:
        """Transfer hair/clothes colors from reference when body warp fails.

        This is a best-effort approach: it matches the colour distribution
        of the reference person's clothing and hair regions onto the
        streamer's frame, keeping the streamer's body shape and pose.
        """
        if person_mask.max() < 0.1:
            return frame

        h, w = frame.shape[:2]
        ref = source.reference_image
        ref_mask = source.reference_mask

        # Resize reference to match frame
        ref_resized = cv2.resize(ref, (w, h))
        ref_mask_resized = cv2.resize(ref_mask, (w, h))

        # Extract person regions
        person_mask_3ch = np.stack([person_mask] * 3, axis=2)
        ref_mask_3ch = np.stack([ref_mask_resized] * 3, axis=2)

        # Compute mean and std for each channel in person regions
        result = frame.copy()

        for c in range(3):
            # Target (streamer) person pixels
            tgt_pixels = frame[:, :, c][person_mask > 0.5]
            if len(tgt_pixels) < 100:
                continue

            # Source (reference) person pixels
            src_pixels = ref_resized[:, :, c][ref_mask_resized > 0.5]
            if len(src_pixels) < 100:
                continue

            tgt_mean = tgt_pixels.mean()
            tgt_std = max(tgt_pixels.std(), 1.0)
            src_mean = src_pixels.mean()
            src_std = max(src_pixels.std(), 1.0)

            # Color transfer: match reference colour distribution
            transferred = (frame[:, :, c].astype(np.float32) - tgt_mean) * (
                src_std / tgt_std
            ) + src_mean

            # Apply only in person region
            mask_2d = person_mask
            result[:, :, c] = (
                transferred * mask_2d + frame[:, :, c].astype(np.float32) * (1 - mask_2d)
            ).clip(0, 255).astype(np.uint8)

        return result

    # ── Timing ──

    def timing_summary(self) -> str:
        """Average ms per stage."""
        n = max(1, self.n_frames)
        seg = self.t_segment / n * 1000
        pose = self.t_pose / n * 1000
        warp = self.t_warp / n * 1000
        blend = self.t_blend / n * 1000
        face = self.face_engine.timing_summary()
        return (
            f"body: seg {seg:.0f}ms pose {pose:.0f}ms warp {warp:.0f}ms "
            f"blend {blend:.0f}ms | {face}"
        )