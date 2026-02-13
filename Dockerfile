# PersonaPlex GPU Server for Koyeb
# Based on NVIDIA CUDA runtime with PersonaPlex (Moshi) model
ARG BASE_IMAGE="nvcr.io/nvidia/cuda"
ARG BASE_IMAGE_TAG="12.4.1-runtime-ubuntu22.04"
FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG}

# Install system dependencies
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libopus-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy PersonaPlex server code
WORKDIR /app/moshi/
COPY moshi/ /app/moshi/

# Create virtual environment and install dependencies
RUN uv venv /app/moshi/.venv --python 3.12
RUN uv sync

# Create cache directory for model weights
RUN mkdir -p /root/.cache

# Expose the WebSocket port
EXPOSE 8998

# Run without SSL (Koyeb handles TLS termination)
# Bind to 0.0.0.0 so Koyeb can reach the container
# Use PORT env var if set by Koyeb, otherwise default to 8998
ENV NO_TORCH_COMPILE=1

ENTRYPOINT []
CMD ["/app/moshi/.venv/bin/python", "-m", "moshi.server", "--host", "0.0.0.0", "--port", "8998"]
