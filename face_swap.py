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
        self.detect_every = max(1, int(os.environ.get("DETECT_EVERY", "3")))
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

        warped_face = cv2.warpAffine(
            face_128, IM_local, (bw, bh), flags=cv2.INTER_LINEAR, borderValue=0
        )
        warped_mask = cv2.warpAffine(
            self._blend_mask, IM_local, (bw, bh), flags=cv2.INTER_LINEAR, borderValue=0
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
        return SwapSource(normed_embedding=face.normed_embedding, name=name)

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
