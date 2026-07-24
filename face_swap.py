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
    """A prepared reference identity, ready to swap onto frames."""

    embedding: np.ndarray
    name: str


class FaceSwapEngine:
    """One engine per worker process; sources are per-session."""

    def __init__(self, model_dir: str, det_size: int = 640):
        if FaceAnalysis is None:
            raise RuntimeError(
                "insightface is not installed — run on a GPU image with "
                "requirements.txt installed."
            )
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        # Face detection + recognition (produces the identity embeddings).
        self.analyzer = FaceAnalysis(
            name="buffalo_l",
            root=model_dir,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.analyzer.prepare(ctx_id=0, det_size=(det_size, det_size))

        # The swap model itself.
        swap_path = os.path.join(model_dir, "inswapper_128.onnx")
        if not os.path.exists(swap_path) or os.path.getsize(swap_path) < MIN_MODEL_BYTES:
            self._fetch_swapper(swap_path)
        self.swapper = insightface.model_zoo.get_model(
            swap_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

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
        faces = self.analyzer.get(portrait_bgr)
        if not faces:
            logger.warning("No face detected in reference portrait for %s", name)
            return None
        # Largest face wins if the portrait has several.
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return SwapSource(embedding=face.normed_embedding, name=name)

    def swap_frame(self, frame_bgr: np.ndarray, source: SwapSource) -> np.ndarray:
        """Swap the streamer's face toward the source identity.

        Returns the original frame unchanged when no face is present, so the
        output track never drops.
        """
        targets = self.analyzer.get(frame_bgr)
        if not targets:
            return frame_bgr

        out = frame_bgr
        for target in targets:
            # `paste_back=True` blends the swapped face back into the frame.
            out = self.swapper.get(out, target, source, paste_back=True)
        return out


def decode_portrait(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode reference portrait")
    return img
