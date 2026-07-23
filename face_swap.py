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


INSWAPPER_URL = (
    "https://github.com/facefusion/facefusion-assets/releases/"
    "download/models/inswapper_128.onnx"
)


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
        if not os.path.exists(swap_path):
            logger.info("Downloading inswapper_128.onnx …")
            self._download(INSWAPPER_URL, swap_path)
        self.swapper = insightface.model_zoo.get_model(
            swap_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        logger.info("Face swap engine ready (models in %s)", model_dir)

    @staticmethod
    def _download(url: str, dest: str) -> None:
        import httpx

        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
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
