# syntax=docker/dockerfile:1

# ------------------------------------------------------------
# FlashAttention-2.8 Benchmark Dockerfile
# CUDA 12.6 | Ubuntu 22.04 | Python 3.11 | PyTorch 2.7.1+cu126
# ------------------------------------------------------------

FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install Python3 and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-distutils python3-pip git wget ca-certificates && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install dependencies
RUN pip install \
    torch==2.7.1+cu126 torchvision --extra-index-url https://download.pytorch.org/whl/cu126 \
    flash-attn==2.8.0.post2 \
    transformers accelerate bitsandbytes

# Copy benchmark script and entrypoint
COPY benchmark.py ./

# Default command runs the benchmark
ENTRYPOINT ["python", "benchmark.py"]
