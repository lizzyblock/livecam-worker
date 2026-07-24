"""
LiveCam GPU worker — LiveKit agent.

Joins the streamer's room as a second participant and republishes two
transformed tracks:

  video -> `livecam-processed`   face swap, then the look/style grade
  audio -> `livecam-audio`       speech-to-speech voice conversion

The streamer's browser and the desktop virtual-camera companion subscribe to
those, and that is what reaches OBS/Zoom/Twitch.

Everything is hot-swappable mid-session over LiveKit data messages, so
changing a face, look or voice never drops the stream.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

import cv2
import httpx
import numpy as np
from livekit import rtc

import config
from face_swap import FaceSwapEngine, SwapSource, decode_portrait
from styles import StyleBank
from voice import FrameQueue, VoiceConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")

PROCESSED_VIDEO = "livecam-processed"
PROCESSED_AUDIO = "livecam-audio"

AUDIO_SAMPLE_RATE = 16000
AUDIO_FRAME_MS = 20
AUDIO_FRAME_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_FRAME_MS // 1000


class SessionAgent:
    """Handles one room: subscribe -> transform -> republish."""

    def __init__(
        self,
        engine: FaceSwapEngine,
        styles: StyleBank,
        room_name: str,
        cfg: dict,
    ):
        self.engine = engine
        self.styles = styles
        self.room_name = room_name
        self.cfg = cfg

        self.source: Optional[SwapSource] = None
        self.style_fn = styles.get(cfg.get("effectPreset"))
        self.converter: Optional[VoiceConverter] = None

        self.room = rtc.Room()
        self._video_out: Optional[rtc.VideoSource] = None
        self._out_w = 1280
        self._out_h = 720
        self._audio_out: Optional[rtc.AudioSource] = None
        self._audio_queue = FrameQueue(AUDIO_FRAME_SAMPLES)
        self._lock = asyncio.Lock()
        self._tasks: list[asyncio.Task] = []
        self._closing = False

    # -- lifecycle -------------------------------------------------

    async def start(self, token: str) -> None:
        await self._load_face(self.cfg.get("face"))
        await self._load_voice(self.cfg.get("voice"))

        self.room.on("track_subscribed", self._on_track)
        self.room.on("data_received", self._on_data)
        self.room.on(
            "participant_connected",
            lambda p: logger.info(
                "Participant %s joined %s", p.identity, self.room_name
            ),
        )
        self.room.on(
            "participant_disconnected",
            lambda p: logger.info(
                "Participant %s left %s", p.identity, self.room_name
            ),
        )
        await self.room.connect(config.LIVEKIT_URL, token)
        logger.info("Agent joined room %s", self.room_name)

        # Declared once and never changed. Every frame we publish is resized
        # to match: a VideoSource declared at one size receiving frames at
        # another is interpreted with the wrong stride, which renders as
        # rainbow smearing rather than an error.
        self._video_out = rtc.VideoSource(self._out_w, self._out_h)
        video_track = rtc.LocalVideoTrack.create_video_track(
            PROCESSED_VIDEO, self._video_out
        )
        await self.room.local_participant.publish_track(
            video_track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA),
        )

        self._audio_out = rtc.AudioSource(AUDIO_SAMPLE_RATE, 1)
        audio_track = rtc.LocalAudioTrack.create_audio_track(
            PROCESSED_AUDIO, self._audio_out
        )
        await self.room.local_participant.publish_track(
            audio_track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )

        self._tasks.append(asyncio.create_task(self._pump_audio()))

    async def stop(self) -> None:
        self._closing = True
        for t in self._tasks:
            t.cancel()
        if self.converter:
            await self.converter.close()
        await self.room.disconnect()
        logger.info("Agent left room %s", self.room_name)

    # -- configuration (initial + hot-swap) ------------------------

    async def _load_face(self, face: Optional[dict]) -> None:
        if not face or not face.get("portraitUrl"):
            async with self._lock:
                self.source = None
            logger.info("Face swap off for %s", self.room_name)
            return
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(face["portraitUrl"])
                resp.raise_for_status()
            portrait = decode_portrait(resp.content)
            prepared = self.engine.prepare_source(portrait, face.get("id", "face"))
            async with self._lock:
                self.source = prepared
            logger.info("Loaded swap source for %s", self.room_name)
        except Exception as e:
            logger.warning("Could not load face: %s", e)
            async with self._lock:
                self.source = None

    async def _load_voice(self, voice: Optional[dict]) -> None:
        old = self.converter
        if not voice or not voice.get("providerVoiceId"):
            self.converter = None
        else:
            key = config.ELEVENLABS_API_KEY
            if not key:
                logger.warning("Voice requested but ELEVENLABS_API_KEY is unset")
                self.converter = None
            else:
                self.converter = VoiceConverter(
                    key, voice["providerVoiceId"], AUDIO_SAMPLE_RATE
                )
                logger.info("Voice conversion on for %s", self.room_name)
        if old:
            await old.close()

    def _set_style(self, preset: Optional[str]) -> None:
        self.style_fn = self.styles.get(preset)
        logger.info("Look set to %r for %s", preset, self.room_name)

    def _on_data(self, data: rtc.DataPacket) -> None:
        """Mid-session control messages from the client.

        {"type":"set_face","face":{...}|null}
        {"type":"set_style","preset":"noir"}
        {"type":"set_voice","voice":{...}|null}
        """
        try:
            msg = json.loads(data.data.decode())
        except Exception:
            return

        kind = msg.get("type")
        if kind == "set_face":
            asyncio.create_task(self._load_face(msg.get("face")))
        elif kind == "set_voice":
            asyncio.create_task(self._load_voice(msg.get("voice")))
        elif kind == "set_style":
            self._set_style(msg.get("preset"))

    # -- media -----------------------------------------------------

    def _on_track(
        self,
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        logger.info(
            "track_subscribed: kind=%s name=%r from=%s",
            track.kind,
            publication.name,
            participant.identity,
        )
        # Never consume our own output.
        if publication.name in (PROCESSED_VIDEO, PROCESSED_AUDIO):
            logger.debug("Ignoring our own %s", publication.name)
            return
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            # LiveKit simulcasts several quality layers and hands subscribers
            # the lowest by default. At 320x180 a face is ~60px across —
            # under what the detector can find — so the swap silently does
            # nothing and the frame passes through unchanged. Ask for the
            # full-resolution layer explicitly.
            try:
                publication.set_video_quality(rtc.VideoQuality.HIGH)
                logger.info("Requested HIGH quality layer from %s", participant.identity)
            except Exception as e:
                logger.warning("Could not request high quality: %s", e)

            logger.info(
                "Subscribed to video from %s — starting transform",
                participant.identity,
            )
            self._tasks.append(
                asyncio.create_task(self._process_video(rtc.VideoStream(track)))
            )
        elif track.kind == rtc.TrackKind.KIND_AUDIO:
            self._tasks.append(
                asyncio.create_task(self._process_audio(rtc.AudioStream(track)))
            )

    async def _process_video(self, stream: rtc.VideoStream) -> None:
        frame_interval = 1.0 / max(1, config.TARGET_FPS)
        last = 0.0
        frames = 0
        async for event in stream:
            if self._closing:
                return
            now = asyncio.get_event_loop().time()
            if now - last < frame_interval:
                continue
            last = now

            frame = event.frame
            bgr = self._to_bgr(frame)
            if bgr is None:
                continue

            async with self._lock:
                source = self.source
            style = self.style_fn

            # Both of these are synchronous and take tens of milliseconds.
            # Run on the event loop and they block *everything* — track
            # subscription, publishing, control messages — for the duration
            # of every frame. Off-thread they don't.
            try:
                bgr = await asyncio.to_thread(self._transform, bgr, source, style)
            except Exception as e:
                logger.debug("transform error: %s", e)

            frames += 1
            if frames == 1:
                logger.info(
                    "First frame processed for %s (in %dx%d, out %dx%d)",
                    self.room_name,
                    bgr.shape[1],
                    bgr.shape[0],
                    self._out_w,
                    self._out_h,
                )
                if bgr.shape[1] < 480:
                    logger.warning(
                        "Incoming video is only %dx%d — faces are too small to "
                        "detect reliably and the swap will pass through. The "
                        "publisher should disable simulcast or raise its "
                        "encoding.",
                        bgr.shape[1],
                        bgr.shape[0],
                    )
            self._publish_video(bgr)

    def _transform(self, bgr: np.ndarray, source, style) -> np.ndarray:
        """Swap then grade. Runs in a worker thread, never on the loop."""
        if source is not None:
            try:
                bgr = self.engine.swap_frame(bgr, source)
            except Exception as e:
                logger.debug("swap error: %s", e)
        if style is not None:
            try:
                bgr = style(bgr)
            except Exception as e:
                logger.debug("style error: %s", e)
        return bgr

    async def _process_audio(self, stream: rtc.AudioStream) -> None:
        """Mic in -> phrase chunks -> speech-to-speech -> output queue.

        With no voice selected the mic is copied through untouched, so the
        published audio track is always usable.
        """
        async for event in stream:
            if self._closing:
                return
            frame = event.frame
            pcm = np.frombuffer(frame.data, dtype=np.int16)

            if frame.num_channels > 1:
                pcm = (
                    pcm.reshape(-1, frame.num_channels)
                    .mean(axis=1)
                    .astype(np.int16)
                )
            if frame.sample_rate != AUDIO_SAMPLE_RATE:
                pcm = _resample(pcm, frame.sample_rate, AUDIO_SAMPLE_RATE)

            converter = self.converter
            if converter is None:
                await self._audio_queue.push(pcm)
                continue

            chunk = converter.feed(pcm)
            if chunk is not None:
                self._tasks.append(
                    asyncio.create_task(self._convert_and_queue(chunk))
                )

    async def _convert_and_queue(self, chunk: np.ndarray) -> None:
        converter = self.converter
        if converter is None:
            return
        out = await converter.convert(chunk)
        if out is not None and len(out):
            await self._audio_queue.push(out)

    async def _pump_audio(self) -> None:
        """Publishes a steady 20ms audio frame, silence-padded when dry."""
        interval = AUDIO_FRAME_MS / 1000
        while not self._closing:
            await asyncio.sleep(interval)
            if self._audio_out is None:
                continue
            pcm = await self._audio_queue.pop()
            frame = rtc.AudioFrame(
                data=pcm.tobytes(),
                sample_rate=AUDIO_SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=len(pcm),
            )
            try:
                await self._audio_out.capture_frame(frame)
            except Exception as e:
                logger.debug("audio publish error: %s", e)

    # -- helpers ---------------------------------------------------

    @staticmethod
    def _to_bgr(frame: rtc.VideoFrame) -> Optional[np.ndarray]:
        """Convert an incoming frame to BGR.

        Dimensions come from the *converted* buffer, not the source frame —
        conversion can pad, and reshaping with the wrong width shears the
        image into diagonal colour bands.
        """
        rgba = frame.convert(rtc.VideoBufferType.RGBA)
        w = getattr(rgba, "width", frame.width)
        h = getattr(rgba, "height", frame.height)

        buf = np.frombuffer(rgba.data, dtype=np.uint8)
        expected = w * h * 4
        if buf.size < expected:
            logger.debug("Short frame buffer (%d < %d), skipping", buf.size, expected)
            return None
        arr = buf[:expected].reshape(h, w, 4)
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

    def _publish_video(self, bgr: np.ndarray) -> None:
        """Publish at the source's declared size, always."""
        if self._video_out is None:
            return
        if bgr.shape[1] != self._out_w or bgr.shape[0] != self._out_h:
            bgr = cv2.resize(bgr, (self._out_w, self._out_h))

        rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
        out = rtc.VideoFrame(
            self._out_w, self._out_h, rtc.VideoBufferType.RGBA, rgba.tobytes()
        )
        self._video_out.capture_frame(out)


def _resample(pcm: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Linear resample. Adequate for speech at these rates."""
    if src == dst or len(pcm) == 0:
        return pcm
    n = int(len(pcm) * dst / src)
    idx = np.linspace(0, len(pcm) - 1, n)
    return np.interp(idx, np.arange(len(pcm)), pcm).astype(np.int16)
