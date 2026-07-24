"""
Worker configuration.

Every setting comes from the environment. Required values are validated at
import time and reported together with a readable message — a missing key
should tell you which one, not raise a bare KeyError from deep in os.py.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

REQUIRED = {
    "LIVEKIT_URL": "LiveKit server URL, e.g. wss://your-project.livekit.cloud",
    "LIVEKIT_API_KEY": "LiveKit API key (must match the backend's)",
    "LIVEKIT_API_SECRET": "LiveKit API secret (must match the backend's)",
}


def _require() -> dict[str, str]:
    missing = {k: why for k, why in REQUIRED.items() if not os.environ.get(k)}
    if missing:
        lines = [
            "",
            "=" * 62,
            " LiveCam worker cannot start — missing configuration",
            "=" * 62,
            "",
            "Set these environment variables on the pod:",
            "",
        ]
        lines += [f"  {k:<22} {why}" for k, why in missing.items()]
        lines += [
            "",
            "In Runpod: pod menu -> Edit Pod -> Environment Variables.",
            "The LiveKit values must match your backend exactly, or the",
            "worker's token will be rejected and it will never join rooms.",
            "",
            "=" * 62,
            "",
        ]
        print("\n".join(lines), file=sys.stderr, flush=True)
        raise SystemExit(1)
    return {k: os.environ[k] for k in REQUIRED}


_cfg = _require()

LIVEKIT_URL = _cfg["LIVEKIT_URL"]
LIVEKIT_API_KEY = _cfg["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = _cfg["LIVEKIT_API_SECRET"]

def _fingerprint(secret: str) -> str:
    """Short, non-reversible identifier for a credential.

    Printed by both the worker and the API so the two can be compared at a
    glance. A mismatch here is the cause of every "invalid token" 401, and
    without this you're left comparing invisible strings across two dashboards.
    """
    import hashlib

    return hashlib.sha256(secret.encode()).hexdigest()[:8]


print(
    "LiveKit config | url=%s | key=%s | secret_fp=%s"
    % (LIVEKIT_URL, LIVEKIT_API_KEY, _fingerprint(LIVEKIT_API_SECRET)),
    flush=True,
)

# Whitespace from copy-paste produces an invalid token with no clue why.
if LIVEKIT_API_KEY != LIVEKIT_API_KEY.strip() or (
    LIVEKIT_API_SECRET != LIVEKIT_API_SECRET.strip()
):
    print(
        "WARNING: LiveKit credentials have leading/trailing whitespace. "
        "Stripping them — but fix the env var.",
        file=sys.stderr,
        flush=True,
    )
    LIVEKIT_API_KEY = LIVEKIT_API_KEY.strip()
    LIVEKIT_API_SECRET = LIVEKIT_API_SECRET.strip()

# Warn about a common paste error rather than failing mysteriously later.
if LIVEKIT_URL.startswith("http"):
    print(
        f"WARNING: LIVEKIT_URL is {LIVEKIT_URL!r} — LiveKit expects a "
        "wss:// URL, not http(s)://. Connection will likely fail.",
        file=sys.stderr,
        flush=True,
    )

MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
PORT = int(os.environ.get("PORT", "8080"))
TARGET_FPS = int(os.environ.get("TARGET_FPS", "24"))

# Self-shutdown safety net. If the worker sits with zero sessions for this
# long it stops its own Runpod pod, so a missed /stop call from the API can
# never leave a GPU billing overnight. Set IDLE_SHUTDOWN_SECONDS=0 to disable.
IDLE_SHUTDOWN_SECONDS = int(os.environ.get("IDLE_SHUTDOWN_SECONDS", "900"))
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_POD_ID = os.environ.get("RUNPOD_POD_ID")

# Optional — only needed for real-time voice conversion. Absence disables
# the voice stage rather than stopping the worker.
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")

if not ELEVENLABS_API_KEY:
    print(
        "NOTE: ELEVENLABS_API_KEY not set — real-time voice conversion is off. "
        "Face swap and looks are unaffected.",
        flush=True,
    )
if not (RUNPOD_API_KEY and RUNPOD_POD_ID):
    print(
        "NOTE: RUNPOD_API_KEY/RUNPOD_POD_ID not set — idle self-shutdown is "
        "off. The GPU will keep billing when unused.",
        flush=True,
    )
