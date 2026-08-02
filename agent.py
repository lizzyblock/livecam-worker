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
import os
from concurrent.futures import ThreadPoolExecutor
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
        self.recolor = None  # lazily created when a recolor stage is enabled
        self.hair = None  # lazily created hairstyle overlay
        self.styles = styles
        self.room_name = room_name
        self.cfg = cfg
        # Audio-only mode: skip the entire video pipeline and only run the
        # mic → ElevenLabs → published-audio path. Used by the Avatar tab,
        # whose video comes from Decart, not this worker — it wants just the
        # cloned voice.
        self.audio_only = bool(cfg.get("audioOnly"))

        self.source: Optional[SwapSource] = None
        self.style_fn = styles.get(cfg.get("effectPreset"))
        self.converter: Optional[VoiceConverter] = None

        self.room = rtc.Room()
        self._video_out: Optional[rtc.VideoSource] = None
        # Output resolution. Every stage — conversion, swap paste-back,
        # encode — scales with pixel count, so this is the bluntest and most
        # effective lever when a session can't keep up. 960x540 costs about
        # 45% less work than 720p and is hard to tell apart on a webcam feed.
        self._out_w = int(os.environ.get("OUT_WIDTH", "1280"))
        self._out_h = int(os.environ.get("OUT_HEIGHT", "720"))
        self._audio_out: Optional[rtc.AudioSource] = None
        self._audio_queue = FrameQueue(AUDIO_FRAME_SAMPLES)
        self._lock = asyncio.Lock()
        self._tasks: list[asyncio.Task] = []
        self._closing = False
        self._swap_errors = 0
        self._style_errors = 0

        # A single dedicated thread for every GPU call. asyncio.to_thread
        # hands work to an arbitrary pool thread, and switching the CUDA
        # context between threads costs more than the inference itself —
        # pinning it to one thread keeps the context warm.
        self._gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")

    # -- lifecycle -------------------------------------------------

    async def start(self, token: str) -> None:
        if not self.audio_only:
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
        # Video output — skipped entirely in audio-only mode.
        if not self.audio_only:
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
        self._gpu.shutdown(wait=False)
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
        elif kind == "set_recolor":
            self._set_recolor(msg)
        elif kind == "set_hairstyle":
            self._set_hairstyle(msg.get("style"))
        elif kind == "set_hair_fit":
            if self.hair is not None:
                self.hair.set_placement(
                    scale=msg.get("scale"), y_offset=msg.get("y_offset")
                )

    def _set_hairstyle(self, style: Optional[str]) -> None:
        """Load a hairstyle PNG overlay (distinct from hair *colour*)."""
        if self.hair is None:
            import hair as _hair

            self.hair = _hair.HairOverlay(self.engine.model_dir)
        ok = self.hair.set_style(style)
        logger.info("Hairstyle set to %r (ok=%s)", style, ok)

    def _ensure_recolor(self):
        if self.recolor is None:
            import recolor as _rc

            self.recolor = _rc.Recolorizer(self.engine.model_dir)
        return self.recolor

    def _set_recolor(self, msg: dict) -> None:
        """Toggle/adjust the recolour stages live.

        {"type":"set_recolor","skin_match":true,
         "shirt": "#3366cc" | null,          # null disables shirt tint
         "background": "off"|"blur"|"replace"}
        """
        rc = self._ensure_recolor()
        if "skin_match" in msg:
            rc.skin_match = bool(msg["skin_match"])
        if "shirt" in msg:
            hexv = msg["shirt"]
            if hexv:
                h = hexv.lstrip("#")
                r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
                rc.shirt_bgr = (b, g, r)
                rc.shirt_on = True
            else:
                rc.shirt_on = False
        if "background" in msg:
            rc.bg_mode = msg["background"] or "off"
        if "hair" in msg:
            hexv = msg["hair"]
            if hexv:
                h = hexv.lstrip("#")
                r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
                rc.set_hair_color((b, g, r))
            else:
                rc.set_hair_color(None)
        if "light_match" in msg:
            rc.light_match = bool(msg["light_match"])
        logger.info(
            "Recolour: skin=%s shirt=%s hair=%s light=%s bg=%s",
            rc.skin_match,
            rc.shirt_on,
            rc.hair_on,
            rc.light_match,
            rc.bg_mode,
        )

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
        # In audio-only mode we don't touch video at all.
        if self.audio_only and track.kind == rtc.TrackKind.KIND_VIDEO:
            logger.debug("Audio-only session: ignoring incoming video")
            return
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            # Belt and braces on resolution. The publisher disables simulcast
            # so there's only one full-quality layer to receive — this just
            # covers older clients that still send several. The enum is named
            # inconsistently across SDK versions, so try each and stay quiet
            # if none apply.
            for attr in ("HIGH", "QUALITY_HIGH", "VIDEO_QUALITY_HIGH"):
                quality = getattr(rtc.VideoQuality, attr, None)
                if quality is not None:
                    try:
                        publication.set_video_quality(quality)
                        logger.debug("Requested %s layer", attr)
                    except Exception as e:
                        logger.debug("set_video_quality failed: %s", e)
                    break

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
        """Transform frames, dropping any we can't keep up with.

        Latency here is cumulative: if a frame takes longer to process than
        the gap between arrivals, waiting for it pushes every later frame
        further behind and the delay grows without bound. Skipping while busy
        keeps the output pinned to the present at the cost of frame rate,
        which is the right trade for a live camera.
        """
        frame_interval = 1.0 / max(1, config.TARGET_FPS)
        last = 0.0
        frames = 0
        busy = False
        dropped = 0
        cost_total = 0.0
        conv_total = 0.0
        pub_total = 0.0
        # Report on a timer rather than a frame count: at low frame rates a
        # frame-based interval can outlast a short test session, which is
        # exactly when the numbers are most wanted.
        report_every = 10.0
        next_report = asyncio.get_event_loop().time() + report_every

        async for event in stream:
            if self._closing:
                return

            now = asyncio.get_event_loop().time()
            if busy or now - last < frame_interval:
                dropped += 1
                continue
            last = now
            busy = True

            frame = event.frame
            t_conv = asyncio.get_event_loop().time()
            bgr = self._to_bgr(frame)
            if bgr is None:
                busy = False
                continue
            conv_total += asyncio.get_event_loop().time() - t_conv

            async with self._lock:
                source = self.source
            style = self.style_fn

            # Both of these are synchronous and take tens of milliseconds.
            # Run on the event loop and they block *everything* — track
            # subscription, publishing, control messages — for the duration
            # of every frame. Off-thread they don't.
            started = asyncio.get_event_loop().time()
            try:
                loop = asyncio.get_event_loop()
                bgr = await loop.run_in_executor(
                    self._gpu, self._transform, bgr, source, style
                )
            except Exception as e:
                logger.debug("transform error: %s", e)
            finally:
                busy = False
            cost_total += asyncio.get_event_loop().time() - started

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
            t_pub = asyncio.get_event_loop().time()
            self._publish_video(bgr)
            pub_total += asyncio.get_event_loop().time() - t_pub

            # Periodic honesty about throughput: average processing cost and
            # how many frames we're skipping to stay current.
            if asyncio.get_event_loop().time() >= next_report and frames:
                next_report = asyncio.get_event_loop().time() + report_every
                avg_ms = (cost_total / frames) * 1000
                logger.info(
                    "%s: %.0fms/frame (transform %.0f, decode %.0f, encode %.0f), "
                    "~%.0f fps, %d dropped | %s",
                    self.room_name,
                    ((cost_total + conv_total + pub_total) / frames) * 1000,
                    avg_ms,
                    (conv_total / frames) * 1000,
                    (pub_total / frames) * 1000,
                    1000 / avg_ms if avg_ms > 0 else 0,
                    dropped,
                    self.engine.timing_summary(),
                )

    def _transform(self, bgr: np.ndarray, source, style) -> np.ndarray:
        """Swap then grade. Runs in a worker thread, never on the loop.

        Errors are reported, not swallowed. A per-frame failure logged at
        debug level is invisible in production and presents as a swap that
        does nothing — so the first occurrence is logged loudly and the rest
        are throttled to keep the log readable.
        """
        if source is not None:
            try:
                bgr = self.engine.swap_frame(bgr, source)
            except Exception as e:
                self._swap_errors += 1
                if self._swap_errors == 1 or self._swap_errors % 300 == 0:
                    logger.error(
                        "Face swap failing (%d frames): %s",
                        self._swap_errors,
                        e,
                        exc_info=self._swap_errors == 1,
                    )
        if style is not None:
            try:
                bgr = style(bgr)
            except Exception as e:
                self._style_errors += 1
                if self._style_errors == 1 or self._style_errors % 300 == 0:
                    logger.error(
                        "Look failing (%d frames): %s", self._style_errors, e
                    )

        # Region recolour (shirt / skin-match / background) runs last, after
        # the face is swapped, so the skin-match stage can pull its target
        # tone from the already-swapped face.
        if self.recolor is not None and self.recolor.active:
            try:
                import time as _t

                if self.recolor.skin_match:
                    face_crop = getattr(self.engine, "last_swapped_face", None)
                    if face_crop is not None:
                        self.recolor.set_face_skin(face_crop)
                bgr = self.recolor.apply(bgr, int(_t.monotonic() * 1000))
            except Exception as e:
                self._recolor_errors = getattr(self, "_recolor_errors", 0) + 1
                if self._recolor_errors == 1 or self._recolor_errors % 300 == 0:
                    logger.error(
                        "Recolour failing (%d frames): %s",
                        self._recolor_errors,
                        e,
                    )

        # Hairstyle overlay (pre-made PNG anchored to the face mesh).
        if self.hair is not None and self.hair.active:
            try:
                bgr = self.hair.apply(
                    bgr,
                    kps=getattr(self.engine, "last_kps", None),
                    bbox=getattr(self.engine, "last_bbox", None),
                )
            except Exception as e:
                self._hair_errors = getattr(self, "_hair_errors", 0) + 1
                if self._hair_errors == 1 or self._hair_errors % 300 == 0:
                    logger.error("Hair overlay failing (%d frames): %s",
                                 self._hair_errors, e)
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
        """Convert an incoming frame to BGR as cheaply as possible.

        WebRTC delivers I420. Going I420 -> RGBA -> BGR touches every pixel
        twice and allocates a 3.7MB intermediate at 720p; converting straight
        from I420 halves that. RGBA is kept as a fallback for any frame type
        that won't convert.
        """
        try:
            i420 = frame.convert(rtc.VideoBufferType.I420)
            w = getattr(i420, "width", frame.width)
            h = getattr(i420, "height", frame.height)
            buf = np.frombuffer(i420.data, dtype=np.uint8)
            expected = w * h * 3 // 2
            if buf.size >= expected:
                yuv = buf[:expected].reshape(h * 3 // 2, w)
                return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        except Exception:
            pass  # fall through to RGBA

        rgba = frame.convert(rtc.VideoBufferType.RGBA)
        w = getattr(rgba, "width", frame.width)
        h = getattr(rgba, "height", frame.height)
        buf = np.frombuffer(rgba.data, dtype=np.uint8)
        expected = w * h * 4
        if buf.size < expected:
            logger.debug("Short frame buffer (%d < %d), skipping", buf.size, expected)
            return None
        return cv2.cvtColor(buf[:expected].reshape(h, w, 4), cv2.COLOR_RGBA2BGR)

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
