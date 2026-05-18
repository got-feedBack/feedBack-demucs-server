# ============================================================
# Slopsmith Demucs Server — Docker Image
# ============================================================
# Build:
#   docker build -t slopsmith-demucs-server .
#
# Run (CPU):
#   docker run -p 7865:7865 slopsmith-demucs-server
#
# Run (GPU — requires nvidia-container-toolkit):
#   docker run --gpus all -p 7865:7865 slopsmith-demucs-server
# ============================================================

# ---- Base: Python slim ----
FROM python:3.11-slim AS base

LABEL org.opencontainers.image.title="Slopsmith Demucs Server"
LABEL org.opencontainers.image.description="AI source separation, lyrics alignment, and pitch extraction service for Slopsmith"
LABEL org.opencontainers.image.source="https://github.com/byrongamatos/slopsmith-demucs-server"

# Prevent Python from writing .pyc files & buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---- Application directory ----
WORKDIR /app

# ---- COPY requirements first (leverage Docker layer cache) ----
COPY requirements.txt .

# ---- Install main Python dependencies ----
# whisperx pins torch~=2.8.0 + torchaudio~=2.8.0 — this satisfies torchcrepe too.
RUN pip install --no-cache-dir -r requirements.txt

# ---- Install demucs SEPARATELY to avoid torchaudio version conflict ----
# demucs 4.0.1 (latest PyPI) requires torchaudio<2.1 which conflicts with
# whisperx (torchaudio~=2.8.0). Installing with --no-deps skips the bad pin.
# Runtime deps that are compatible with modern torch are added manually.
RUN pip install --no-cache-dir \
        demucs>=4.0.0 \
        --no-deps \
    && pip install --no-cache-dir \
        einops \
        julius \
        lameenc \
        openunmix \
        pyyaml \
        tqdm \
        dora-search

# ---- COPY application code ----
COPY . .

# ---- Least privilege runtime user ----
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app


# ---- Server port ----
EXPOSE 7865

# ---- Default environment ----
ENV PORT=7865 \
    HOST=0.0.0.0 \
    AUTO_UPDATE=false \
    UPDATE_TIME=04:00 \
    UPDATE_CHECK_INTERVAL=3600 \
    SLOPSMITH_DEMUCS_CACHE=/app/cache \
    # Redirect HuggingFace and PyTorch caches to the persistent volume
    HF_HOME=/app/cache/huggingface \
    TORCH_HOME=/app/cache/torch \
    HUGGINGFACE_HUB_CACHE=/app/cache/huggingface/hub \
    MPLLOCALEDIR=/app/cache/locale
USER appuser


# ---- Volume for model cache (persist across restarts) ----
VOLUME /app/cache

# ---- Entrypoint ----
ENTRYPOINT ["./docker-entrypoint.sh"]
