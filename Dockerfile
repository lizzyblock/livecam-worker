# CUDA runtime base for GPU inference.
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
