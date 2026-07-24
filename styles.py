"""
Real-time look/style stage.

Runs after the face swap on every frame. Two classes of effect:

  * **Colour grades** (noir, cyberpunk) — pure OpenCV. Sub-millisecond, no
    model weights, genuinely real-time. These are what a colourist would do
    on a live grade, not a neural style transfer.

  * **Neural stylisation** (anime) — AnimeGANv3 as ONNX. A small, fast
    generator that holds up at 720p on a mid-range GPU. Weights download on
    first use; if they're unavailable the stage degrades to a pass-through
    rather than killing the stream.

Anything heavier (claymation, full diffusion restyle) is deliberately absent:
StreamDiffusion-class pipelines can't hold 24fps on a single mid-range card
alongside a face swap, and a preset that silently drops the stream to 6fps is
worse than no preset.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger("styles")

# Community mirrors, tried in order. Like the swap model, these move and
# disappear — the style degrades to a pass-through rather than taking the
# stream down, and the file can always be placed in MODEL_DIR by hand.
ANIMEGAN_MIRRORS = [
    "https://huggingface.co/spaces/Xenova/AnimeGANv3/resolve/main/AnimeGANv3_Hayao_36.onnx",
    "https://huggingface.co/vumichien/AnimeGANv3/resolve/main/AnimeGANv3_Hayao_36.onnx",
    "https://github.com/TachibanaYoshino/AnimeGANv3/releases/download/v1.1.0/AnimeGANv3_Hayao_36.onnx",
]

MIN_STYLE_BYTES = 1024 * 1024  # a 404 page is a few KB; the model is several MB


# ─────────────────────────────────────────────────────────────
# Colour grades — no model, no download, always available
# ─────────────────────────────────────────────────────────────


def _noir(frame: np.ndarray) -> np.ndarray:
    """Hard mono contrast with a little grain — film noir."""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # S-curve for crushed blacks and hot highlights
    graded = cv2.LUT(grey, _NOIR_CURVE)
    grain = np.random.default_rng().normal(0, 4, graded.shape).astype(np.int16)
    graded = np.clip(graded.astype(np.int16) + grain, 0, 255).astype(np.uint8)
    return cv2.cvtColor(graded, cv2.COLOR_GRAY2BGR)


def _cyberpunk(frame: np.ndarray) -> np.ndarray:
    """Teal shadows, magenta highlights, soft bloom on the hot areas."""
    f = frame.astype(np.float32)
    b, g, r = cv2.split(f)

    # Split-tone: push shadows to teal, highlights to magenta
    luma = (0.114 * b + 0.587 * g + 0.299 * r) / 255.0
    b = b + (1.0 - luma) * 28 + luma * 18
    g = g + (1.0 - luma) * 8 - luma * 14
    r = r - (1.0 - luma) * 12 + luma * 30

    out = cv2.merge([b, g, r])
    out = np.clip(out, 0, 255).astype(np.uint8)

    # Bloom: blur the brightest areas back over the frame
    bright = cv2.threshold(out, 185, 255, cv2.THRESH_TOZERO)[1]
    bloom = cv2.GaussianBlur(bright, (0, 0), 9)
    return cv2.addWeighted(out, 1.0, bloom, 0.35, 0)


def _build_noir_curve() -> np.ndarray:
    x = np.arange(256, dtype=np.float32) / 255.0
    # Steep S-curve
    y = np.clip((x - 0.5) * 1.55 + 0.5, 0, 1)
    y = np.power(y, 0.92)
    return (y * 255).astype(np.uint8)


_NOIR_CURVE = _build_noir_curve()


# ─────────────────────────────────────────────────────────────
# Neural stylisation
# ─────────────────────────────────────────────────────────────


class AnimeStyle:
    """AnimeGANv3 (ONNX). Loaded lazily so unused presets cost nothing."""

    def __init__(self, model_dir: str):
        self.path = os.path.join(model_dir, "animegan_v3.onnx")
        self.session = None
        self._failed = False
        self.model_dir = model_dir

    def _ensure(self) -> bool:
        if self.session is not None:
            return True
        if self._failed:
            return False
        try:
            import onnxruntime as ort

            if (
                not os.path.exists(self.path)
                or os.path.getsize(self.path) < MIN_STYLE_BYTES
            ):
                if not self._fetch(self.path):
                    self._failed = True
                    return False

            self.session = ort.InferenceSession(
                self.path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            logger.info("AnimeGANv3 ready")
            return True
        except Exception as e:
            # Never take the stream down for a missing look.
            logger.warning("Anime style unavailable, passing through: %s", e)
            self._failed = True
            return False

    def _fetch(self, dest: str) -> bool:
        """Try each mirror; returns False if none yield a usable model."""
        import httpx

        os.makedirs(self.model_dir, exist_ok=True)
        for url in ANIMEGAN_MIRRORS:
            try:
                logger.info("Fetching AnimeGANv3 from %s", url.split("/")[2])
                with httpx.stream("GET", url, follow_redirects=True, timeout=180) as r:
                    r.raise_for_status()
                    with open(dest, "wb") as f:
                        for chunk in r.iter_bytes():
                            f.write(chunk)
                if os.path.getsize(dest) < MIN_STYLE_BYTES:
                    os.remove(dest)
                    logger.warning("Mirror returned a stub, trying next")
                    continue
                return True
            except Exception as e:
                logger.warning("AnimeGAN mirror failed: %s", e)
                if os.path.exists(dest):
                    os.remove(dest)

        logger.error(
            "Anime style unavailable — no mirror served the weights. "
            "Place AnimeGANv3_Hayao_36.onnx at %s to enable it. "
            "The other looks are unaffected.",
            dest,
        )
        return False

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if not self._ensure():
            return frame
        h, w = frame.shape[:2]
        # The generator wants dimensions that are multiples of 8.
        tw, th = (w // 8) * 8, (h // 8) * 8
        small = cv2.resize(frame, (tw, th))
        inp = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        inp = np.expand_dims(inp, 0)

        try:
            out = self.session.run(None, {self.input_name: inp})[0]
        except Exception as e:
            logger.debug("anime inference failed: %s", e)
            return frame

        out = (np.squeeze(out) + 1.0) * 127.5
        out = np.clip(out, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return cv2.resize(out, (w, h))


# ─────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────

StyleFn = Callable[[np.ndarray], np.ndarray]


class StyleBank:
    """Resolves a preset id to a callable. Unknown ids pass through."""

    def __init__(self, model_dir: str):
        self._anime = AnimeStyle(model_dir)
        self._map: dict[str, StyleFn] = {
            "none": lambda f: f,
            "clean": lambda f: f,
            "noir": _noir,
            "cyberpunk": _cyberpunk,
            "anime": self._anime,
        }

    def get(self, preset: Optional[str]) -> StyleFn:
        if not preset:
            return self._map["none"]
        fn = self._map.get(preset.lower())
        if fn is None:
            logger.debug("Unknown preset %r — passing through", preset)
            return self._map["none"]
        return fn

    def available(self) -> list[str]:
        """Looks that will actually do something.

        Anime is listed only when its weights are on disk — advertising a
        preset that silently passes through is worse than not offering it.
        """
        names = [k for k in self._map if k != "anime"]
        if os.path.exists(self._anime.path) and (
            os.path.getsize(self._anime.path) >= MIN_STYLE_BYTES
        ):
            names.append("anime")
        return sorted(names)
