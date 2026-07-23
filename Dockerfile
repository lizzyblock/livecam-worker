# CUDA runtime base for GPU inference. cuDNN 8 here pairs with the pinned
# onnxruntime-gpu 1.18.1 in requirements.txt — change one and you must
# change the other, or inference silently drops to CPU.
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Runtime libs + the toolchain insightface needs to compile its Cython
# extensions. The toolchain is removed again below to keep the image small.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3-pip python3-dev \
      build-essential cmake \
      libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# insightface's setup.py imports numpy and Cython at build time, so these must
# land first — installing them in the same pass as insightface fails.
RUN pip3 install --upgrade pip setuptools wheel \
    && pip3 install "numpy<2" cython

COPY requirements.txt .
RUN pip3 install -r requirements.txt

# insightface depends on the CPU-only `onnxruntime` package. If both it and
# onnxruntime-gpu are installed, the CPU build wins and every model runs on
# CPUExecutionProvider at a few frames per second. Remove it and reinstall
# the GPU build to be certain which one is live.
RUN pip3 uninstall -y onnxruntime || true \
    && pip3 install --force-reinstall --no-deps onnxruntime-gpu==1.18.1

# Log which providers the build produced. This is informational only —
# CI runners have no GPU, so a hard assert here would fail every cloud build.
# The real check happens at runtime and is reported on /healthz as "gpu".
RUN python3 -c "\
import onnxruntime as ort; \
print('onnxruntime build providers:', ort.get_available_providers())" || true

# Drop the compiler once the wheels are built (~300MB back).
RUN apt-get purge -y build-essential cmake python3-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Model weights live on a mounted volume so restarts don't re-download them.
ENV MODEL_DIR=/models
VOLUME ["/models"]

EXPOSE 8080
CMD ["python3", "server.py"]
