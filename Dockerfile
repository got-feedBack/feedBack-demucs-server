# ============================================================
# feedBack Demucs Server — Docker Image
# ============================================================
# Build:
#   docker build -t feedback-demucs-server .
#
# Run (CPU):
#   docker run -p 7865:7865 feedback-demucs-server
#
# Run (GPU — requires nvidia-container-toolkit):
#   docker run --gpus all -p 7865:7865 feedback-demucs-server
# ============================================================

# ---- Base: Python slim ----
FROM python:3.11-slim AS base

LABEL org.opencontainers.image.title="feedBack Demucs Server"
LABEL org.opencontainers.image.description="AI source separation, lyrics alignment, and pitch extraction service for feedBack"
# image.source is what links the published package to this repo on GHCR - it must
# point at the repo that actually builds it, or the package is orphaned.
LABEL org.opencontainers.image.source="https://github.com/got-feedBack/feedBack-demucs-server"
LABEL org.opencontainers.image.licenses="AGPL-3.0-only"

# Prevent Python from writing .pyc files & buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
ffmpeg \
git \
libgomp1 \
&& rm -rf /var/lib/apt/lists/*

# ---- Application directory ----
WORKDIR /app

# ---- COPY requirements first (leverage Docker layer cache) ----
COPY requirements.txt .

# ---- Install main Python dependencies ----
# whisperx pins torch~=2.8.0 + torchaudio~=2.8.0 — this satisfies torchcrepe too.
# requirements.txt deliberately does NOT list audio-separator or demucs; both are
# installed --no-deps below. See the ⚠️ note at the top of that file.
RUN pip install --no-cache-dir -r requirements.txt

# ---- Install audio-separator SEPARATELY, with --no-deps ----
# Its metadata requires `diffq` on non-Windows, a C-extension whose newest wheels
# stop at cp310. On this python:3.11 base pip finds no wheel, falls back to the
# sdist and tries to compile it — and this image has no gcc (deliberately: adding a
# toolchain to ship one optional quantizer is the wrong trade). That compile is what
# broke every image build. Its real deps are in requirements.txt instead.
RUN pip install --no-cache-dir --no-deps "audio-separator>=0.44.0"

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
        dora-search \
        sphn

# ---- diffq: BINARY-ONLY, and optional ----
# Needed only to load QUANTIZED demucs checkpoints. `demucs` guards its import and
# audio-separator only reaches for it from its bundled demucs architecture — neither
# is on the bs_roformer_sw path, which is what the app splits with.
#
# --only-binary=:all: is the point: it makes pip FAIL rather than silently fall back
# to the sdist, so a missing wheel can never reintroduce a source build here. On this
# base (glibc >= 2.28, cp311) `diffq-fixed` has a manylinux_2_28 wheel; it installs
# under the module name `diffq`, so it satisfies the import just like the original.
#
# The tolerated failure is gated on the SPECIFIC "no wheel exists" signature. A bare
# `|| true` would also swallow a transient network/index error and quietly ship an image
# without diffq while CI stayed green — a silent downgrade is worse than a failed build.
RUN out="$(pip install --no-cache-dir --no-deps --only-binary=:all: 'diffq-fixed>=0.2' 2>&1)"; rc=$?; \
    echo "$out"; \
    if [ "$rc" -ne 0 ]; then \
        case "$out" in \
            *"No matching distribution found"*|*"Could not find a version that satisfies"*) \
                echo "WARNING: no diffq wheel for this interpreter - quantized demucs checkpoints will not load (bs_roformer_sw is unaffected)" ;; \
            *) \
                echo "ERROR: diffq install failed for a reason OTHER than a missing wheel (network? index?). Failing the build rather than silently shipping without it." >&2; \
                exit "$rc" ;; \
        esac; \
    fi

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
    CACHE_TTL=24h \
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
