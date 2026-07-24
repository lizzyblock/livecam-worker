# LiveCam Worker — real-time face swap (GPU)

The service that makes FUA LiveCam actually transform a stream. It joins a
LiveKit room as a hidden participant, swaps the streamer's face onto a chosen
reference identity frame-by-frame, and republishes the result as the
`livecam-processed` track that OBS/Zoom/Twitch pick up.

## How a session flows

```
Streamer browser ──cam track──► LiveKit room ◄──joins── Worker (this service)
                                     │                        │ swap each frame
   OBS / Zoom  ◄──livecam-processed──┘◄───republished─────────┘
   (via virtual-cam companion)
```

1. NestJS API mints a room token and `POST`s `/dispatch` here with the room
   name + face config (a short-lived signed portrait URL).
2. Worker analyzes the portrait **once** → cached identity embedding.
3. Each incoming frame: detect face → swap toward the identity → publish.
4. Mid-session, the client can send a LiveKit data message
   (`{"type":"set_face","face":{...}}`) to hot-swap or disable the face.

## The three stages

Every frame and every audio chunk passes through this pipeline:

| Stage | What it does | Cost |
|---|---|---|
| **Face swap** | InsightFace `inswapper_128`. Source embedding cached once. | ~15-25ms/frame |
| **Look** | `styles.py` — noir and cyberpunk are OpenCV colour grades (sub-ms); anime is AnimeGANv3 ONNX | 0-20ms/frame |
| **Voice** | `voice.py` — ElevenLabs speech-to-speech on phrase boundaries | see below |

### A note on voice latency

Speech-to-speech is **not** frame-by-frame like the face swap — the model needs
a phrase of context to sound natural. Audio is cut on silence (never mid-word)
and converted in chunks, which puts the converted voice roughly **600ms-1.2s**
behind the raw mic.

For one-way streaming that's fine: viewers only ever hear the converted track.
For anything interactive, leave voice off — a second of delay ruins a
conversation. There's no way around this with current speech-to-speech models;
anyone claiming zero-latency voice conversion is either not converting or not
telling the truth.

### On style presets

Only looks the GPU can hold at frame rate are exposed. Diffusion-based
restyling (claymation, full scene restyle) runs at single-digit FPS alongside a
face swap on a mid-range card, so it's deliberately absent rather than shipped
broken.

## ⚠️ Model licensing — read before charging customers

`inswapper_128` is an InsightFace research model. **The authors restrict it to
non-commercial use**, and pulled it from official distribution — which is why
it now only exists on community mirrors and why those mirrors keep breaking.

Every open face-swap project (roop, facefusion, and their forks) uses this same
model and inherits the same restriction. Running it inside a paid subscription
product is a real legal exposure, not a technicality.

Options, honestly:

1. **Contact InsightFace for a commercial licence.** The correct route, and
   worth doing before you take payments.
2. **Use a commercially-licensed face-swap API** and swap out the engine —
   `face_swap.py` is isolated behind a small interface for exactly this reason.
   Higher per-minute cost, clean licensing.
3. **Launch the other modules first.** Images, video, voice and campaigns all
   run on properly licensed commercial APIs. Face swap can follow once the
   licence question is settled.

Nothing else in this stack has this problem — it's specific to the swap model.

## Models

- **buffalo_l** (InsightFace) — detection, landmarks, recognition embeddings.
- **inswapper_128** — the swap itself. Auto-downloaded to `MODEL_DIR` on first
  boot.

## Run it

```bash
pip install -r requirements.txt      # GPU box: keeps onnxruntime-gpu
cp .env.example .env                 # fill LiveKit creds
python server.py                     # dispatch API on :8080
```

Point the API at it:

```dotenv
# in the NestJS backend .env
LIVECAM_WORKER_URL=http://your-worker-host:8080
```

## Build times & gotchas

First build: **10–25 minutes.** Most of it is the ~2.5GB CUDA base plus the
`onnxruntime-gpu` and `opencv` wheels. Rebuilds are 1–2 minutes once Docker has
the layers cached — and because `requirements.txt` is copied before the source,
changing your Python code doesn't re-run pip at all.

Three things that trip up this image specifically:

- **onnxruntime silently falls back to CPU.** The nastiest failure here,
  because nothing errors: models load, the worker reports healthy, and face
  swap runs at ~2fps. Two independent causes, both handled:

  1. *Package shadowing.* insightface depends on the CPU-only `onnxruntime`.
     Both packages install into the same directory, so whichever pip touches
     last wins. The Dockerfile removes every variant and installs
     `onnxruntime-gpu` as the final step.
  2. *CUDA version mismatch.* PyPI's default `onnxruntime-gpu` wheel is built
     for **CUDA 11.8 up to 1.18**, and **CUDA 12 + cuDNN 9 from 1.19**. Pair
     1.18 with a CUDA 12 image and it loads but can't see the GPU. The base
     image is therefore `12.4.1-cudnn-` (cuDNN 9) with `onnxruntime-gpu>=1.19`.

  **Diagnosing it:** the startup log prints the real provider list.
  `AzureExecutionProvider, CPUExecutionProvider` means the CPU package is
  installed. `TensorrtExecutionProvider, CUDAExecutionProvider, ...` means the
  GPU build is live. **Check `"gpu": true` on `/healthz`, not just
  `"engine": true`.**

- **insightface compiles from source.** It needs `build-essential` and
  `python3-dev`, and its `setup.py` imports numpy and Cython at build time —
  so those are installed in a separate, earlier layer. Installing them
  alongside insightface fails.
- **numpy must stay on 1.x.** insightface and onnxruntime are built against the
  numpy 1.x ABI. With numpy 2 installed the build succeeds and then crashes at
  import, which looks like a model-loading failure and sends you hunting in the
  wrong place.

If the build stalls with no output for several minutes during
`Building wheel for insightface`, that's normal — it's compiling.

## Building in the cloud (recommended)

Pushing a ~6GB image from a home connection is usually slower than the build
itself. `.github/workflows/build.yml` builds on GitHub's runners and pushes to
GHCR, which sits on the same backbone as the registry.

```bash
git init && git add . && git commit -m "worker"
git remote add origin git@github.com:YOU/livecam-worker.git
git push -u origin main
```

The workflow runs on push. When it finishes, make the package public:
**repo → Packages → livecam-worker → Package settings → Change visibility →
Public** — otherwise Runpod can't pull it without credentials.

Your image is then `ghcr.io/YOU/livecam-worker:latest`.

Notes on the workflow:
- It frees ~20GB on the runner first; CUDA images fill the default 14GB.
- `cache-from/to: type=gha` means pip only re-runs when `requirements.txt`
  changes — later builds land in 2–3 minutes.
- It pins `linux/amd64`, which is what Runpod GPUs are. Building on an Apple
  Silicon Mac without this produces an arm64 image that won't start.

To rebuild without changing code, use **Actions → Build & push worker image →
Run workflow**.

## Deploying

Build the image and run one container per GPU (A10 is plenty for 720p at
24fps; scale horizontally by active session count):

```bash
docker build -t livecam-worker .
docker run --gpus all -p 8080:8080 -v livecam-models:/models --env-file .env livecam-worker
```

Runpod and Modal both autoscale on a queue/HTTP signal — key the scaler on
`/healthz`'s `activeSessions`.

## Latency & quality knobs

- `TARGET_FPS` (default 24) — the worker throttles processing to this; lower
  it to buy latency headroom on busier GPUs.
- Detection size is 640 in `face_swap.py` — fine for a single centered
  streamer; raising it rarely helps and costs milliseconds.
- Frames with no detected face pass through untouched, so looking away or
  covering your face never freezes the stream.

## Responsible use

Face swap enrollment in the API is consent-gated: a face can't be enrolled
without an explicit attestation that the uploader owns the likeness or has
permission to use it, and that timestamp is stored. Keep that gate — it's what
separates a creator tool from a deepfake tool. Consider also watermarking or
disclosure for published content depending on your jurisdiction.
