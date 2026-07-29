"""
Real-time face swap engine.

Uses InsightFace's `buffalo_l` for face detection/landmarks and the
`inswapper_128` model for the actual identity swap. The reference portrait is
analyzed once at session start to produce a source identity embedding; each
incoming video frame is then swapped toward that identity.

Design notes for real-time use:
  * The source embedding is computed a single time and cached — per-frame work
    is just detect + swap, which is what keeps latency low.
  * If no face is found in a frame (streamer looks away, hand over face), the
    original frame is passed through untouched rather than dropped, so the
    stream never stutters.
  * Detection runs at a modest size (640) — larger barely helps for a single
    centered streamer and costs latency.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("face-swap")

# insightface is heavy; import lazily so the dispatch API can boot without a GPU
try:
    import insightface
    from insightface.app import FaceAnalysis
except Exception:  # pragma: no cover - import guarded for CPU/dev boxes
    insightface = None
    FaceAnalysis = None


# Community mirrors, tried in order. These move and disappear regularly, so
# the download is best-effort — if every mirror fails the worker tells you to
# place the file yourself rather than dying with a stack trace.
INSWAPPER_MIRRORS = [
    "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
    "https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx",
    "https://huggingface.co/datasets/OwlMaster/gg1342/resolve/main/inswapper_128.onnx",
    "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.onnx",
]

# Rough size sanity check — a 404 HTML page is a few KB, the model is ~530MB.
MIN_MODEL_BYTES = 100 * 1024 * 1024


@dataclass
class SwapSource:
    """A prepared reference identity, ready to swap onto frames.

    The field MUST be called `normed_embedding`: InsightFace's swapper reads
    `source_face.normed_embedding` directly off whatever it's handed. Any
    other name raises AttributeError on every frame — which, if the caller
    swallows it, looks exactly like a swap that quietly does nothing.
    """

    normed_embedding: np.ndarray
    name: str
    # Median Lab skin tone of the reference portrait, used to recolour the
    # swapped face toward the uploaded image instead of the subject's skin.
    skin_lab: Optional[np.ndarray] = None


class FaceSwapEngine:
    """One engine per worker process; sources are per-session.

    Two analysers, deliberately:

      * `source_analyzer` runs the full pipeline, but only once per session
        when the reference portrait is embedded. Quality matters, speed
        doesn't.
      * `analyzer` runs on every frame and is stripped to detection alone.
        The swapper needs `target_face.kps` and nothing else — landmarks,
        gender/age and recognition are pure overhead at 20+ fps, and they
        were roughly two thirds of the per-frame cost.
    """

    def __init__(self, model_dir: str, det_size: int = 0):
        if FaceAnalysis is None:
            raise RuntimeError(
                "insightface is not installed — run on a GPU image with "
                "requirements.txt installed."
            )
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        det = det_size or int(os.environ.get("DET_SIZE", "320"))

        # ONNX Runtime sizes its intra-op pool from the host's cores too.
        # Left alone it spawns far more threads than the cgroup allows and
        # they spend their time contending rather than computing.
        threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
        try:
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self.session_options = so
        except Exception:
            self.session_options = None

        try:
            cv2.setNumThreads(threads)
        except Exception:
            pass
        logger.info("Inference thread pools capped at %d", threads)

        # EXHAUSTIVE is onnxruntime's default and it re-benchmarks conv
        # algorithms whenever an input shape changes. The detector declares
        # dynamic dimensions, so that search can dominate the frame budget.
        # HEURISTIC picks a good algorithm analytically instead.
        cuda_opts = {
            "device_id": 0,
            "cudnn_conv_algo_search": os.environ.get("CUDNN_ALGO", "HEURISTIC"),
            "do_copy_in_default_stream": "1",
            "arena_extend_strategy": "kSameAsRequested",
        }
        providers = [
            ("CUDAExecutionProvider", cuda_opts),
            "CPUExecutionProvider",
        ]

        # Per-frame: detection only.
        self.analyzer = FaceAnalysis(
            name="buffalo_l",
            root=model_dir,
            providers=providers,
            allowed_modules=["detection"],
        )
        self.analyzer.prepare(ctx_id=0, det_size=(det, det))

        # Once per session: needs recognition for the identity embedding.
        self.source_analyzer = FaceAnalysis(
            name="buffalo_l",
            root=model_dir,
            providers=providers,
            allowed_modules=["detection", "recognition"],
        )
        self.source_analyzer.prepare(ctx_id=0, det_size=(640, 640))

        # Detection is the expensive step and faces barely move between
        # frames, so it runs every Nth frame and the last known landmarks
        # are reused in between.
        self.detect_every = max(1, int(os.environ.get("DETECT_EVERY", "2")))
        # Landmark smoothing factor (EMA). 0 = off (raw, jittery), 1 = frozen.
        # ~0.5 removes most wobble while staying responsive to real movement.
        self.smooth = float(os.environ.get("SMOOTH", "0.5"))
        self._prev_kps = None
        self._prev_bbox = None
        # How much bigger than the face box to composite over. Below ~1.6 the
        # blend can clip; above ~2.5 there's no benefit, just more pixels.
        self.crop_scale = float(os.environ.get("BLEND_CROP_SCALE", "2.0"))
        # Longest edge of the region the blend actually runs on. 384 keeps a
        # typical face near 190px — comfortably above the model's 128px
        # output — while cutting the kernel work roughly fourfold.
        self.work_size = int(os.environ.get("BLEND_WORK_SIZE", "384"))
        # Set FAST_BLEND=0 to fall back to InsightFace's own compositing,
        # which handles unusual angles a little more gracefully at roughly
        # 20x the cost.
        self.fast_blend = os.environ.get("FAST_BLEND", "1") != "0"
        # Unsharp strength on the swapped face. 0 disables; above ~0.8 the
        # edges start to look electric.
        self.sharpen = float(os.environ.get("FACE_SHARPEN", "0.35"))

        # Colour matching: push the swapped face toward the uploaded portrait's
        # skin tone instead of the subject's. ON by default now — it's what
        # "keep the target's colour" means. color_ab drives the hue match,
        # color_l a gentler luminance nudge.
        self.color_match = os.environ.get("COLOR_MATCH", "1") != "0"
        self.color_ab = float(os.environ.get("COLOR_AB", "0.85"))
        self.color_l = float(os.environ.get("COLOR_L", "0.5"))

        # Skin smoothing strength on the swapped face (0 off, ~0.4 natural,
        # >0.7 plasticky).
        self.smooth_face = float(os.environ.get("SMOOTH_FACE", "0.4"))

        # Optional GFPGAN face restoration. The swap model emits a soft
        # 128x128 face; GFPGAN reconstructs realistic skin/eye/mouth detail,
        # the single biggest available quality jump. It costs ~15-30ms/frame,
        # affordable now that the swap itself is ~35ms and there's a 4090
        # underneath. Off by default; FACE_RESTORE=1 enables it.
        self.restore = os.environ.get("FACE_RESTORE", "0") == "1"
        self._restorer = None
        if self.restore:
            self._init_restorer()
        self._blend_mask = self._make_blend_mask(128)
        self._cached_targets: list = []
        self._frame_no = 0
        logger.info(
            "Engine tuned: det_size=%d detect_every=%d", det, self.detect_every
        )

        # The swap model itself.
        # Same tuned CUDA options as the detector — the swapper was still
        # defaulting to EXHAUSTIVE algorithm search.
        swap_providers = [
            ("CUDAExecutionProvider", cuda_opts),
            "CPUExecutionProvider",
        ]
        swap_path = os.path.join(model_dir, "inswapper_128.onnx")
        if not os.path.exists(swap_path) or os.path.getsize(swap_path) < MIN_MODEL_BYTES:
            self._fetch_swapper(swap_path)
        self.swapper = insightface.model_zoo.get_model(
            swap_path,
            providers=swap_providers,
        )

        self._hits = 0
        self._misses = 0
        self._crop_logged = False
        self._crop_fallback_logged = False
        self._blend_logged = False
        self.last_swapped_face = None
        self.t_detect = 0.0
        self.t_swap = 0.0
        self.n_detect = 0
        self.n_swap = 0
        self.provider = self._active_provider()
        if self.provider == "CUDAExecutionProvider":
            logger.info("Face swap engine ready on GPU (models in %s)", model_dir)
        else:
            # Loud, because the swap still "works" on CPU — at ~2fps, which
            # is useless for streaming and easy to misdiagnose as a network
            # or LiveKit problem.
            logger.error(
                "=" * 62
                + "\n RUNNING ON CPU — face swap will be ~2fps, not usable live."
                + "\n Check the providers line logged just above:"
                + "\n   'AzureExecutionProvider, CPUExecutionProvider'"
                + "\n       -> the CPU-only onnxruntime package is installed"
                + "\n   CUDA listed but unused"
                + "\n       -> CUDA/cuDNN version mismatch with the base image"
                + "\n"
                + "\n onnxruntime-gpu >=1.19 needs CUDA 12 + cuDNN 9."
                + "\n Rebuild from the provided Dockerfile.\n"
                + "=" * 62
            )

    def _init_restorer(self) -> None:
        """Load GFPGAN once. Falls back to plain sharpening if unavailable."""
        try:
            import onnxruntime as ort

            path = os.path.join(self.model_dir, "gfpgan_1.4.onnx")
            if not os.path.exists(path):
                self._fetch_gfpgan(path)
            providers = [
                ("CUDAExecutionProvider", {"device_id": 0}),
                "CPUExecutionProvider",
            ]
            self._restorer = ort.InferenceSession(path, providers=providers)
            self._restore_in = self._restorer.get_inputs()[0].name
            logger.info("GFPGAN face restoration enabled")
        except Exception as e:
            logger.error(
                "Could not enable face restoration (%s) — continuing with "
                "sharpening only.",
                e,
            )
            self.restore = False
            self._restorer = None

    def _fetch_gfpgan(self, dest: str) -> None:
        import httpx

        # ONNX export of GFPGAN v1.4, community-mirrored.
        urls = [
            "https://huggingface.co/gmk123qwe/gfpgan/resolve/main/GFPGANv1.4.onnx",
            "https://huggingface.co/facefusion/models/resolve/main/gfpgan_1.4.onnx",
        ]
        for url in urls:
            try:
                logger.info("Fetching GFPGAN from %s", url.split("/")[2])
                with httpx.stream("GET", url, follow_redirects=True, timeout=180) as r:
                    r.raise_for_status()
                    with open(dest, "wb") as f:
                        for chunk in r.iter_bytes():
                            f.write(chunk)
                if os.path.getsize(dest) > 50 * 1024 * 1024:
                    return
                os.remove(dest)
            except Exception as e:
                logger.warning("GFPGAN mirror failed: %s", e)
                if os.path.exists(dest):
                    os.remove(dest)
        raise RuntimeError("no GFPGAN mirror available")

    def _restore_face(self, face_bgr: np.ndarray) -> np.ndarray:
        """Run GFPGAN on an aligned face crop → crisp 512x512 BGR.

        Returns the full 512px result, NOT resized back down. Shrinking it to
        the swap model's 128px would throw away exactly the detail GFPGAN just
        reconstructed — the restored face has to be warped to the frame at
        full resolution to be worth running at all.
        """
        if self._restorer is None:
            return face_bgr
        inp = cv2.resize(face_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = (inp - 0.5) / 0.5
        inp = inp.transpose(2, 0, 1)[None]
        out = self._restorer.run(None, {self._restore_in: inp})[0][0]
        out = np.clip((out.transpose(1, 2, 0) * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)  # 512x512

    @staticmethod
    def _make_blend_mask(size: int) -> np.ndarray:
        """Soft elliptical mask over the aligned face, built once.

        InsightFace derives its mask per frame from a pixel difference, then
        erodes, dilates and blurs it with kernels scaled to the face — tens of
        milliseconds every frame. For a talking head the mask barely changes,
        so a fixed ellipse with a feathered edge is visually equivalent and
        costs nothing after startup.
        """
        m = np.zeros((size, size), dtype=np.float32)
        cv2.ellipse(
            m,
            (size // 2, size // 2),
            (int(size * 0.42), int(size * 0.50)),
            0, 0, 360, 1.0, -1,
        )
        k = (size // 8) * 2 + 1  # odd kernel, ~16px feather at 128
        return cv2.GaussianBlur(m, (k, k), 0)

    def _finish_face(self, face: np.ndarray, source: "SwapSource") -> np.ndarray:
        """Recolour the swapped face toward the portrait, then smooth skin.

        inswapper blends the source identity with the *target* frame's colour,
        so the swapped face tends to take on the subject's skin tone. When the
        portrait's tone is known, this shifts the face's colour (a/b in Lab,
        and part of L) toward it, so the result matches the uploaded image.
        Smoothing is a light bilateral filter — evens skin while keeping edges.
        """
        out = face

        # Colour transfer toward the portrait's tone.
        if self.color_match and source is not None and source.skin_lab is not None:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
            # Current tone of the swapped face centre.
            h, w = lab.shape[:2]
            centre = lab[int(h * 0.35) : int(h * 0.7), int(w * 0.25) : int(w * 0.75)]
            cur = np.median(centre.reshape(-1, 3), axis=0)
            tgt = source.skin_lab
            shift = np.array(
                [
                    (tgt[0] - cur[0]) * self.color_l,  # luminance, partial
                    (tgt[1] - cur[1]) * self.color_ab,  # a channel
                    (tgt[2] - cur[2]) * self.color_ab,  # b channel
                ],
                dtype=np.float32,
            )
            lab += shift
            out = cv2.cvtColor(
                np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR
            )

        # Skin smoothing — bilateral keeps edges (eyes, lips) sharp while
        # evening skin. Blended at `smooth_face` strength so it doesn't turn
        # plasticky.
        if self.smooth_face > 0:
            sm = cv2.bilateralFilter(out, d=7, sigmaColor=45, sigmaSpace=45)
            a = self.smooth_face
            out = cv2.addWeighted(sm, a, out, 1 - a, 0)

        return out

    def _fast_paste(self, frame: np.ndarray, face_128: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Warp the swapped face back and composite it.

        Only the destination bounding box is touched — typically a few percent
        of the frame — and the two warps read from a 128x128 source, so the
        whole operation is a handful of milliseconds regardless of output
        resolution.
        """
        h, w = frame.shape[:2]
        IM = cv2.invertAffineTransform(M)

        # Where the aligned square lands in the frame.
        s = face_128.shape[0]
        corners = np.array(
            [[0, 0], [s, 0], [s, s], [0, s]], dtype=np.float32
        ).reshape(1, -1, 2)
        dst = cv2.transform(corners, IM).reshape(-1, 2)

        x1 = max(0, int(np.floor(dst[:, 0].min())))
        y1 = max(0, int(np.floor(dst[:, 1].min())))
        x2 = min(w, int(np.ceil(dst[:, 0].max())))
        y2 = min(h, int(np.ceil(dst[:, 1].max())))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return frame

        # Re-origin the transform so warping writes straight into the box.
        IM_local = IM.copy()
        IM_local[0, 2] -= x1
        IM_local[1, 2] -= y1
        bw, bh = x2 - x1, y2 - y1

        # Restore detail on the aligned face *before* warping it back, while
        # it's still square and centred — that's what GFPGAN expects. This
        # returns a 512x512 face; the source coordinate system is 128, so the
        # transform is scaled up 4x to sample from the high-res result rather
        # than the soft original.
        face_src = face_128
        src_size = 128
        if self.restore and self._restorer is not None:
            try:
                face_src = self._restore_face(face_128)  # 512x512
                src_size = face_src.shape[0]
            except Exception as e:
                if not getattr(self, "_restore_err_logged", False):
                    self._restore_err_logged = True
                    logger.error("Restoration failed mid-stream: %s", e)

        # Rebuild the warp to sample from the higher-res source.
        #
        # IM_local maps the 128px aligned face → destination box. The restored
        # face is `src_size` px (512), so a source pixel there corresponds to
        # (128/src_size) in the original space. Compose IM_local with that
        # scale-down: dst = IM_local · Sdown · [src_px; 1]. This is proper
        # matrix composition — scaling IM_local's columns directly does NOT
        # place the face correctly (verified numerically).
        if src_size != 128:
            IM3 = np.vstack([IM_local, [0, 0, 1]]).astype(np.float32)
            k = 128.0 / src_size
            Sdown = np.array([[k, 0, 0], [0, k, 0], [0, 0, 1]], dtype=np.float32)
            IM_use = (IM3 @ Sdown)[:2].astype(np.float32)
        else:
            IM_use = IM_local

        # Cubic sampling — with restoration the source is sharp, so this
        # preserves detail rather than inventing it.
        warped_face = cv2.warpAffine(
            face_src, IM_use, (bw, bh), flags=cv2.INTER_CUBIC, borderValue=0
        )
        warped_mask = cv2.warpAffine(
            self._blend_mask, IM_local, (bw, bh), flags=cv2.INTER_LINEAR, borderValue=0
        )

        # Unsharp mask, applied only to the face box.
        #
        # The model emits 128x128 regardless of output resolution, so a face
        # on screen is always an upscale and always slightly soft. This can't
        # invent detail, but it restores the local contrast that upscaling
        # flattens — and on a small region it costs well under a millisecond.
        if self.sharpen > 0:
            blurred = cv2.GaussianBlur(warped_face, (0, 0), 1.2)
            warped_face = cv2.addWeighted(
                warped_face, 1.0 + self.sharpen, blurred, -self.sharpen, 0
            )

        region = frame[y1:y2, x1:x2]
        alpha = warped_mask[:, :, None]
        blended = warped_face.astype(np.float32) * alpha + region.astype(
            np.float32
        ) * (1.0 - alpha)
        frame[y1:y2, x1:x2] = blended.astype(np.uint8)
        return frame

    def _swap_one(self, frame: np.ndarray, target, source: SwapSource) -> np.ndarray:
        """Swap a single face, blending only around it.

        InsightFace's `paste_back` does its compositing across the *entire*
        frame: three warpAffine passes, a wide Gaussian blur and a float32
        multiply-add, all on CPU. At 720p that costs an order of magnitude
        more than the 128x128 network it's blending — measured at ~193ms per
        frame versus ~6ms for detection.

        A face occupies a small part of the picture, so the same work is done
        here on a crop around it and the result written back. Identical
        output, a fraction of the pixels.
        """
        # Fast path: let the model produce the aligned face, then composite
        # it ourselves. No cropping or rescaling needed — only the face's
        # destination box is written to.
        if self.fast_blend:
            try:
                face_128, M = self.swapper.get(
                    frame, target, source, paste_back=False
                )
                # Expose the swapped face so the recolour stage can sample its
                # skin tone for hand/neck matching.
                self.last_swapped_face = face_128
                # Recolour toward the uploaded portrait's skin tone (overrides
                # inswapper's tendency to blend with the subject) and optionally
                # smooth the skin. Both operate on the aligned 128px face.
                face_128 = self._finish_face(face_128, source)
                if not self._blend_logged:
                    self._blend_logged = True
                    logger.info(
                        "Fast blending active: %dx%d face composited directly "
                        "(InsightFace paste_back bypassed)",
                        face_128.shape[1],
                        face_128.shape[0],
                    )
                return self._fast_paste(frame, face_128, M)
            except Exception as e:
                if not self._crop_fallback_logged:
                    self._crop_fallback_logged = True
                    logger.error(
                        "Fast blending failed (%s) — using InsightFace "
                        "compositing, which is much slower.",
                        e,
                    )

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = target.bbox

        # Generous margin: the blend feathers outwards, and cropping too
        # tightly leaves a visible seam at the edge of the mask.
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        half = max(x2 - x1, y2 - y1) * self.crop_scale / 2.0

        cx1 = max(0, int(cx - half))
        cy1 = max(0, int(cy - half))
        cx2 = min(w, int(cx + half))
        cy2 = min(h, int(cy + half))
        if cx2 - cx1 < 32 or cy2 - cy1 < 32:
            return self.swapper.get(frame, target, source, paste_back=True)

        # Confirm the fast path is live, once per process.
        if not self._crop_logged:
            self._crop_logged = True
            eff = min(self.work_size, max(cx2 - cx1, cy2 - cy1))
            logger.info(
                "Crop-scoped blending active: %dx%d region, blended at ~%dpx "
                "(was %dx%d full-frame)",
                cx2 - cx1,
                cy2 - cy1,
                eff,
                w,
                h,
            )

        crop = frame[cy1:cy2, cx1:cx2]
        ch, cw = crop.shape[:2]

        # Work at a reduced scale.
        #
        # InsightFace sizes its blending kernels from the *face*, not the
        # frame: a 350px face yields a 34x34 erosion and a 35x35 Gaussian.
        # Cropping shrinks the array but leaves those kernels untouched, so
        # it alone bought far less than expected. Halving the face size
        # quarters that work.
        #
        # Little is lost: the network emits a 128x128 face regardless, so
        # anything above that is already upscaling.
        scale = 1.0
        if max(cw, ch) > self.work_size:
            scale = self.work_size / float(max(cw, ch))
            crop = cv2.resize(
                crop,
                (max(1, int(cw * scale)), max(1, int(ch * scale))),
                interpolation=cv2.INTER_AREA,
            )

        # Re-express the face's geometry in crop coordinates.
        #
        # Copying the Face is not an option: InsightFace's Face.__getattr__
        # returns None for any missing key instead of raising AttributeError,
        # so copy.deepcopy asks for __deepcopy__, receives None, and tries to
        # call it. Shifting in place and restoring afterwards sidesteps the
        # whole problem — and all GPU work runs on one dedicated thread, so
        # there's no window for another frame to observe the shifted values.
        origin = np.array([cx1, cy1], dtype=np.float32)
        orig_kps = target.kps
        orig_bbox = target.bbox
        try:
            target.kps = (orig_kps - origin) * scale
            target.bbox = (
                orig_bbox - np.array([cx1, cy1, cx1, cy1], dtype=np.float32)
            ) * scale
            swapped = self.swapper.get(crop, target, source, paste_back=True)
            if scale != 1.0:
                swapped = cv2.resize(
                    swapped, (cw, ch), interpolation=cv2.INTER_LINEAR
                )
        except Exception as e:
            if not self._crop_fallback_logged:
                self._crop_fallback_logged = True
                logger.error(
                    "Crop-scoped blending failed (%s) — using full-frame "
                    "compositing, which costs ~10x more per frame.",
                    e,
                )
            target.kps = orig_kps
            target.bbox = orig_bbox
            return self.swapper.get(frame, target, source, paste_back=True)
        finally:
            # Restore before anything else sees them — these targets are
            # cached and reused across frames.
            target.kps = orig_kps
            target.bbox = orig_bbox

        frame[cy1:cy2, cx1:cx2] = swapped
        return frame

    def timing_summary(self) -> str:
        """Average ms per stage, so a slow frame can be attributed."""
        det = (self.t_detect / self.n_detect * 1000) if self.n_detect else 0
        swp = (self.t_swap / self.n_swap * 1000) if self.n_swap else 0
        return f"detect {det:.0f}ms/call, swap {swp:.0f}ms/frame"

    @staticmethod
    def _active_provider() -> str:
        """Which provider onnxruntime will actually use.

        Logged verbatim because the failure mode is silent: a CPU-only
        onnxruntime lists 'AzureExecutionProvider, CPUExecutionProvider',
        while the GPU build lists 'TensorrtExecutionProvider,
        CUDAExecutionProvider, CPUExecutionProvider'. Seeing the real list
        tells you immediately which package is installed.
        """
        try:
            import onnxruntime as ort

            available = ort.get_available_providers()
            logger.info(
                "onnxruntime %s providers: %s", ort.__version__, available
            )
            return (
                "CUDAExecutionProvider"
                if "CUDAExecutionProvider" in available
                else "CPUExecutionProvider"
            )
        except Exception:
            return "unknown"

    def _fetch_swapper(self, dest: str) -> None:
        """Try each mirror until one yields a plausibly-sized model."""
        for url in INSWAPPER_MIRRORS:
            try:
                logger.info("Fetching inswapper_128.onnx from %s", url.split("/")[2])
                self._download(url, dest)
                size = os.path.getsize(dest)
                if size < MIN_MODEL_BYTES:
                    logger.warning("Got %d bytes — not the model, trying next", size)
                    os.remove(dest)
                    continue
                logger.info("Downloaded inswapper_128.onnx (%d MB)", size // 1048576)
                return
            except Exception as e:
                logger.warning("Mirror failed (%s)", e)
                if os.path.exists(dest):
                    os.remove(dest)

        raise RuntimeError(
            "\n" + "=" * 62
            + "\n Could not download inswapper_128.onnx from any mirror."
            + "\n"
            + "\n Put the file at:  " + dest
            + "\n"
            + "\n On Runpod, open a shell on the pod and run:"
            + "\n   wget -O " + dest + " <url-to-inswapper_128.onnx>"
            + "\n"
            + "\n Because /models is a volume, you only need to do this once."
            + "\n" + "=" * 62
        )

    @staticmethod
    def _download(url: str, dest: str) -> None:
        import httpx

        with httpx.stream("GET", url, follow_redirects=True, timeout=300) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)

    def prepare_source(self, portrait_bgr: np.ndarray, name: str) -> Optional[SwapSource]:
        """Analyze the reference portrait once → cached identity embedding."""
        faces = self.source_analyzer.get(portrait_bgr)
        if not faces:
            logger.warning("No face detected in reference portrait for %s", name)
            return None
        # Largest face wins if the portrait has several.
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        logger.info(
            "Prepared source %r from a %dx%d face region",
            name,
            int(face.bbox[2] - face.bbox[0]),
            int(face.bbox[3] - face.bbox[1]),
        )
        # Sample the portrait's skin tone from the face region, so the swap
        # can be recoloured toward the *uploaded image* rather than inheriting
        # the subject's skin. Median over the cheek/forehead band avoids eyes,
        # brows and lips skewing it.
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        x1, y1 = max(0, x1), max(0, y1)
        crop = portrait_bgr[y1:y2, x1:x2]
        skin_lab = None
        if crop.size:
            ch, cw = crop.shape[:2]
            band = crop[int(ch * 0.35) : int(ch * 0.7), int(cw * 0.25) : int(cw * 0.75)]
            if band.size:
                lab = cv2.cvtColor(band, cv2.COLOR_BGR2LAB).reshape(-1, 3)
                skin_lab = np.median(lab, axis=0).astype(np.float32)

        return SwapSource(
            normed_embedding=face.normed_embedding, name=name, skin_lab=skin_lab
        )

    def _smooth_targets(self, targets: list) -> list:
        """Exponentially smooth the primary face's landmarks across frames.

        Only the largest face is tracked (the streamer). Its keypoints and box
        are blended toward the previous frame's, so detection noise and the
        cache's periodic snap turn into smooth motion. A large jump — a real
        fast turn, or a new person — resets the filter so it doesn't lag or
        smear across the cut.
        """
        if not targets or self.smooth <= 0:
            self._prev_kps = None
            return targets

        # Largest face = the streamer.
        face = max(
            targets,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        kps = face.kps.astype(np.float32)
        bbox = face.bbox.astype(np.float32)

        if self._prev_kps is not None and self._prev_kps.shape == kps.shape:
            # Reset if the face moved a lot (fast turn / different person),
            # measured relative to face size so it scales with distance.
            face_w = max(1.0, bbox[2] - bbox[0])
            jump = np.linalg.norm(kps - self._prev_kps, axis=1).mean() / face_w
            if jump < 0.35:
                a = self.smooth
                kps = a * self._prev_kps + (1 - a) * kps
                bbox = a * self._prev_bbox + (1 - a) * bbox

        self._prev_kps = kps
        self._prev_bbox = bbox
        face.kps = kps
        face.bbox = bbox
        return [face]

    def swap_frame(self, frame_bgr: np.ndarray, source: SwapSource) -> np.ndarray:
        """Swap the streamer's face toward the source identity.

        Returns the original frame unchanged when no face is present, so the
        output track never drops. Detection misses are counted and reported
        periodically — a swap that silently does nothing is the single most
        confusing failure mode here, so it should never be silent.
        """
        import time

        self._frame_no += 1
        # Re-detect on the interval; reuse the previous landmarks otherwise.
        if self._frame_no % self.detect_every == 1 or not self._cached_targets:
            t0 = time.perf_counter()
            targets = self.analyzer.get(frame_bgr)
            self.t_detect += time.perf_counter() - t0
            self.n_detect += 1
            if targets:
                # Temporal smoothing kills the frame-to-frame wobble.
                #
                # Detection carries small random noise, and the reuse-then-snap
                # cache pattern makes the face jump every Nth frame. Blending
                # each new detection toward the previous one with an
                # exponential filter (EMA) makes the landmarks glide instead of
                # snap — the single biggest fix for "juddery". SMOOTH=0
                # disables; higher = steadier but laggier on fast turns.
                targets = self._smooth_targets(targets)
                self._cached_targets = targets
        else:
            targets = self._cached_targets

        if not targets:
            self._cached_targets = []
            self._misses += 1
            if self._misses in (30, 300) or self._misses % 900 == 0:
                logger.warning(
                    "No face detected in %d frames. Check lighting and that "
                    "you're facing the camera.",
                    self._misses,
                )
            return frame_bgr

        out = frame_bgr
        t0 = time.perf_counter()
        for target in targets:
            out = self._swap_one(out, target, source)
        self.t_swap += time.perf_counter() - t0
        self.n_swap += 1

        self._hits += 1
        if self._hits == 1:
            logger.info("First successful swap — %d face(s) in frame", len(targets))
        return out


def decode_portrait(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode reference portrait")
    return img
