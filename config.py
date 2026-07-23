import os
from dotenv import load_dotenv

load_dotenv()

LIVEKIT_URL = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]

MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
PORT = int(os.environ.get("PORT", "8080"))
TARGET_FPS = int(os.environ.get("TARGET_FPS", "24"))

# Self-shutdown safety net. If the worker sits with zero sessions for this
# long it stops its own Runpod pod, so a missed /stop call from the API can
# never leave a GPU billing overnight. Set IDLE_SHUTDOWN_SECONDS=0 to disable.
IDLE_SHUTDOWN_SECONDS = int(os.environ.get("IDLE_SHUTDOWN_SECONDS", "900"))
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_POD_ID = os.environ.get("RUNPOD_POD_ID")

# Needed only for real-time voice conversion.
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
