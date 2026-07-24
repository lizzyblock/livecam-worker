"""
Dispatch API for the LiveCam GPU worker.

The NestJS API calls POST /dispatch when a streamer starts a LiveCam session.
This process mints a worker-identity token for that room, spins up a
SessionAgent, and tracks it so it can be cleaned up when the room ends.

Run one of these per GPU instance; put them behind an autoscaler keyed on
active session count (Runpod/Modal both support this).
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from livekit import api
from pydantic import BaseModel

import config
from agent import SessionAgent
from face_swap import FaceSwapEngine
from styles import StyleBank

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

engine: FaceSwapEngine | None = None
styles: StyleBank | None = None
sessions: dict[str, SessionAgent] = {}
last_active: float = time.time()


def touch() -> None:
    """Mark the worker as recently used."""
    global last_active
    last_active = time.time()


async def _self_shutdown() -> None:
    """Stop our own Runpod pod so GPU billing halts."""
    if not (config.RUNPOD_API_KEY and config.RUNPOD_POD_ID):
        logger.warning("Idle, but no Runpod credentials — staying up")
        return
    logger.info("Idle timeout reached — stopping pod %s", config.RUNPOD_POD_ID)
    async with httpx.AsyncClient(timeout=30) as client:
        await client.post(
            f"https://rest.runpod.io/v1/pods/{config.RUNPOD_POD_ID}/stop",
            headers={"Authorization": f"Bearer {config.RUNPOD_API_KEY}"},
        )


async def _idle_watchdog() -> None:
    """Safety net: shut down if nothing has used the worker for a while.

    The API normally stops the pod itself when the last session ends. This
    catches the case where that call never lands (API restart, network blip)
    so an idle GPU can't quietly bill all night.
    """
    if config.IDLE_SHUTDOWN_SECONDS <= 0:
        return
    while True:
        await asyncio.sleep(30)
        if sessions:
            touch()
            continue
        if time.time() - last_active > config.IDLE_SHUTDOWN_SECONDS:
            await _self_shutdown()
            return


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine, styles
    logger.info("Loading face swap engine …")
    engine = FaceSwapEngine(config.MODEL_DIR)
    styles = StyleBank(config.MODEL_DIR)
    touch()
    watchdog = asyncio.create_task(_idle_watchdog())
    yield
    watchdog.cancel()
    for agent in list(sessions.values()):
        await agent.stop()


app = FastAPI(title="LiveCam Worker", lifespan=lifespan)


class Face(BaseModel):
    id: str
    portraitUrl: str


class Voice(BaseModel):
    provider: str | None = None
    providerVoiceId: str | None = None


class DispatchBody(BaseModel):
    room: str
    effectPreset: str | None = None
    face: Face | None = None
    voice: Voice | None = None


def _worker_token(room: str) -> str:
    """Token for the worker to join the room, publish, and subscribe.

    NOTE: `hidden` must stay off. A hidden participant is invisible to the
    others *along with its tracks* — the transformed video would be published
    into a room where nobody can subscribe to it. The streamer seeing a
    "LiveCam" participant in the list is a small cosmetic cost for output
    that actually reaches them.
    """
    grant = api.VideoGrants(
        room=room,
        room_join=True,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
    )
    return (
        api.AccessToken(config.LIVEKIT_API_KEY, config.LIVEKIT_API_SECRET)
        .with_identity(f"livecam-worker-{room}")
        .with_name("LiveCam")
        .with_grants(grant)
        .to_jwt()
    )


@app.get("/")
async def root():
    """Friendly landing so hitting the bare host isn't a bare 404."""
    return {
        "service": "LiveCam GPU worker",
        "ready": engine is not None,
        "endpoints": {
            "health": "/healthz",
            "dispatch": "POST /dispatch",
            "stop": "POST /stop",
        },
    }


@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "activeSessions": len(sessions),
        "engine": engine is not None,
        "provider": getattr(engine, "provider", None),
        "gpu": getattr(engine, "provider", "") == "CUDAExecutionProvider",
        "styles": styles.available() if styles else [],
        "idleSeconds": int(time.time() - last_active) if not sessions else 0,
    }


@app.post("/dispatch")
async def dispatch(body: DispatchBody):
    touch()
    if engine is None:
        raise HTTPException(503, "Engine not ready")
    if body.room in sessions:
        return {"status": "already_running", "room": body.room}

    agent = SessionAgent(engine, styles, body.room, body.model_dump())
    try:
        await agent.start(_worker_token(body.room))
    except Exception as e:
        logger.exception("Failed to start agent")
        raise HTTPException(500, f"Agent start failed: {e}")

    sessions[body.room] = agent
    touch()
    asyncio.create_task(_watch(body.room, agent))
    return {"status": "started", "room": body.room}


@app.post("/stop")
async def stop(body: DispatchBody):
    agent = sessions.pop(body.room, None)
    if agent:
        await agent.stop()
    touch()  # start the idle clock from now
    return {"status": "stopped", "room": body.room}


async def _watch(room: str, agent: SessionAgent) -> None:
    """Reap the session once the streamer has left.

    Two rules keep this from cutting a session short:

    1. A join grace period. The API dispatches us *before* it hands the
       browser its token, so we are always in the room first. Reaping on an
       empty room immediately would kill every session before it starts.
    2. Only reap after someone has actually been seen. A room that never had
       a participant is one where the streamer is still connecting — not one
       they have left.
    """
    join_grace_seconds = 90
    empty_checks_before_reap = 3  # ~45s of confirmed emptiness
    waited = 0
    seen_participant = False
    empty_streak = 0

    while room in sessions:
        await asyncio.sleep(15)
        waited += 15

        participants = len(agent.room.remote_participants)
        if participants > 0:
            seen_participant = True
            empty_streak = 0
            continue

        if not seen_participant:
            if waited < join_grace_seconds:
                logger.debug(
                    "Room %s still empty after %ds — within join grace",
                    room,
                    waited,
                )
                continue
            logger.warning(
                "Room %s had no participant within %ds — the browser never "
                "joined. Reaping.",
                room,
                join_grace_seconds,
            )
            break

        empty_streak += 1
        if empty_streak >= empty_checks_before_reap:
            logger.info("Streamer left room %s — reaping", room)
            break

    if room in sessions:
        await agent.stop()
        sessions.pop(room, None)
        touch()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=config.PORT)
