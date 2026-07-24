# ─────────────────────────────────────────────────────────────
# GPU base. THIS TAG AND THE onnxruntime-gpu VERSION ARE A PAIR.
#
#   onnxruntime-gpu >= 1.19  ->  CUDA 12 + cuDNN 9   (PyPI default wheel)
#   onnxruntime-gpu <= 1.18  ->  CUDA 11.8           (PyPI default wheel)
#
# The 1.18 CUDA-12 wheels only exist on a separate Microsoft index, which is
# why pinning 1.18 here quietly produced a CPU-only runtime. Note the tag is
# `-cudnn-` (cuDNN 9), not `-cudnn8-`.
# ─────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

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

# insightface depends on the CPU-only `onnxruntime`. Both packages install
# into the same `onnxruntime/` directory, so whichever lands last wins and
# the loser's binaries are overwritten. Remove every variant, then install
# only the GPU build — this must be the final pip step.
RUN pip3 uninstall -y onnxruntime onnxruntime-gpu onnxruntime-openvino || true \
    && pip3 install "onnxruntime-gpu>=1.19,<2"

# Informational: CI runners have no GPU, so this can't be a hard assert.
# The authoritative check is at runtime, reported as "gpu" on /healthz.
RUN python3 -c "\
import onnxruntime as ort; \
print('onnxruntime', ort.__version__, 'providers:', ort.get_available_providers())" || true

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
