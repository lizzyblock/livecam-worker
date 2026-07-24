"""
Real-time voice conversion for the mic track.

Approach: capture the streamer's audio, cut it into short chunks on natural
pauses, run each chunk through ElevenLabs speech-to-speech, and publish the
converted audio as a new track.

**Latency is real and unavoidable here.** Speech-to-speech is not a
frame-by-frame transform like the face swap — the model needs a phrase of
context to sound natural. Expect roughly 600ms–1.2s behind the raw mic
depending on chunk length and network. That is fine for broadcast (viewers
never hear the original) but it does mean the converted audio trails the video
slightly. Two ways to handle it, both exposed here:

  * `sync_delay_ms` on the video stage — delays the picture to match, giving
    perfect lip sync at the cost of overall latency. Right for pre-recorded
    or one-way streaming.
  * Leave it at 0 — the picture stays snappy and the audio trails a little.
    Right for anything interactive, where total delay matters more than
    perfect sync.

Chunking is done on silence rather than a fixed clock: cutting mid-word is
what makes converted speech sound robotic.
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave
from typing import Optional

import httpx
import numpy as np

logger = logging.getLogger("voice")

SAMPLE_RATE = 16000
# Below this RMS we treat the frame as silence.
SILENCE_RMS = 350
# Cut a chunk after this much trailing silence (ms).
SILENCE_HANG_MS = 260
# Never let a chunk run longer than this, even mid-sentence (ms).
MAX_CHUNK_MS = 2600
# Don't bother converting anything shorter than this (ms).
MIN_CHUNK_MS = 320


class VoiceConverter:
    """Buffers mic audio, converts on phrase boundaries, emits PCM chunks."""

    def __init__(self, api_key: str, voice_id: str, sample_rate: int = SAMPLE_RATE):
        self.api_key = api_key
        self.voice_id = voice_id
        self.sample_rate = sample_rate

        self._buf: list[np.ndarray] = []
        self._samples = 0
        self._silence_samples = 0
        self._client = httpx.AsyncClient(timeout=30)
        self._inflight = 0

    @property
    def _hang_samples(self) -> int:
        return int(self.sample_rate * SILENCE_HANG_MS / 1000)

    @property
    def _max_samples(self) -> int:
        return int(self.sample_rate * MAX_CHUNK_MS / 1000)

    @property
    def _min_samples(self) -> int:
        return int(self.sample_rate * MIN_CHUNK_MS / 1000)

    def feed(self, pcm: np.ndarray) -> Optional[np.ndarray]:
        """Add a frame of int16 mono PCM.

        Returns a completed chunk when a phrase boundary is reached, else None.
        """
        self._buf.append(pcm)
        self._samples += len(pcm)

        rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2))) if len(pcm) else 0.0
        if rms < SILENCE_RMS:
            self._silence_samples += len(pcm)
        else:
            self._silence_samples = 0

        ready = (
            self._samples >= self._min_samples
            and self._silence_samples >= self._hang_samples
        ) or self._samples >= self._max_samples

        if not ready:
            return None

        chunk = np.concatenate(self._buf)
        self._buf = []
        self._samples = 0
        self._silence_samples = 0

        # All-silence chunk: nothing worth sending upstream.
        if float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2))) < SILENCE_RMS:
            return None
        return chunk

    async def convert(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        """Send one chunk to ElevenLabs speech-to-speech; return converted PCM."""
        # Cap concurrency — piling up requests makes latency worse, not better.
        if self._inflight >= 2:
            logger.debug("Dropping chunk, converter saturated")
            return None
        self._inflight += 1
        try:
            wav = _to_wav(chunk, self.sample_rate)
            files = {"audio": ("chunk.wav", wav, "audio/wav")}
            data = {
                "model_id": "eleven_english_sts_v2",
                "output_format": f"pcm_{self.sample_rate}",
                # Keep the streamer's delivery; only change the timbre.
                "voice_settings": '{"stability":0.4,"similarity_boost":0.85}',
            }
            resp = await self._client.post(
                f"https://api.elevenlabs.io/v1/speech-to-speech/{self.voice_id}/stream",
                headers={"xi-api-key": self.api_key},
                files=files,
                data=data,
            )
            if resp.status_code != 200:
                logger.warning("STS failed %s: %s", resp.status_code, resp.text[:200])
                return None

            # The response is a raw PCM stream, so a chunk boundary can land
            # mid-sample and leave an odd byte count. np.frombuffer rejects
            # that outright ("buffer size must be a multiple of element
            # size"), dropping audio that was otherwise fine — so trim the
            # stray byte rather than losing the chunk.
            data = resp.content
            if len(data) < 2:
                return None
            if len(data) % 2:
                data = data[: len(data) - 1]
            return np.frombuffer(data, dtype=np.int16)
        except Exception as e:
            logger.warning("STS error: %s", e)
            return None
        finally:
            self._inflight -= 1

    async def close(self) -> None:
        await self._client.aclose()


def _to_wav(pcm: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class FrameQueue:
    """Hands converted audio out in fixed-size frames for publishing."""

    def __init__(self, frame_samples: int):
        self.frame_samples = frame_samples
        self._pending = np.zeros(0, dtype=np.int16)
        self._lock = asyncio.Lock()

    async def push(self, pcm: np.ndarray) -> None:
        async with self._lock:
            self._pending = np.concatenate([self._pending, pcm])

    async def pop(self) -> np.ndarray:
        """Returns one frame, padding with silence when the buffer is dry."""
        async with self._lock:
            if len(self._pending) >= self.frame_samples:
                out = self._pending[: self.frame_samples]
                self._pending = self._pending[self.frame_samples :]
                return out
            out = np.zeros(self.frame_samples, dtype=np.int16)
            if len(self._pending):
                out[: len(self._pending)] = self._pending
                self._pending = np.zeros(0, dtype=np.int16)
            return out
