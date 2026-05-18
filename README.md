# Slopsmith Demucs Server

A lightweight GPU-accelerated service providing AI source separation, lyrics alignment, and per-syllable pitch extraction for [Slopsmith](https://github.com/byrongamatos/slopsmith). Designed to run on a desktop with a CUDA GPU while Slopsmith runs on a NAS or Docker host.

[![Docker Build](https://github.com/byrongamatos/slopsmith-demucs-server/actions/workflows/docker-build.yml/badge.svg)](https://github.com/byrongamatos/slopsmith-demucs-server/actions/workflows/docker-build.yml)

## Features

- **Source Separation** (`POST /separate`) — Split audio into stems using Demucs (`htdemucs_ft` default, also `htdemucs_6s`, `mdx_extra`)
- **Lyrics Alignment** (`POST /align`) — Forced alignment of plain text lyrics against audio via WhisperX + wav2vec2 (line/word/syllable/phoneme granularity)
- **Per-syllable Pitch Extraction** (`POST /pitch`) — MIDI note estimation per syllable using CREPE/torchcrepe

## Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU fallback
- FFmpeg (`apt install ffmpeg` / `brew install ffmpeg`)

### Install (Native)

```bash
git clone https://github.com/byrongamatos/slopsmith-demucs-server.git
cd slopsmith-demucs-server
python -m venv .venv
source .venv/bin/activate

# Step 1: Install main dependencies (fastapi, whisperx, torchcrepe, etc.)
# whisperx pins torch~=2.8.0 + torchaudio~=2.8.0
pip install -r requirements.txt

# Step 2: Install demucs SEPARATELY (torchaudio version conflict workaround)
# demucs requires torchaudio<2.1, which conflicts with whisperx.
# Installing with --no-deps bypasses the bad pin.
# dora-search is demucs's logging lib (imported as `import dora`).
pip install demucs --no-deps
pip install einops julius lameenc openunmix pyyaml tqdm dora-search
```

> ⚠️ **Why two install steps?** `demucs` (PyPI 4.0.1) pins `torchaudio<2.1` while `whisperx` needs `torchaudio~=2.8.0`. These are incompatible. Installing demucs with `--no-deps` avoids the conflict. Demucs works fine with modern torchaudio — only the `save_audio` function had issues, and that's patched in `run_demucs.py` to use `soundfile` instead.

### Run

```bash
python server.py --port 7865
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 7865 | Port to listen on |
| `--host` | 0.0.0.0 | Host to bind to |
| `--model` | htdemucs_ft | Demucs model (htdemucs_ft, htdemucs_6s, mdx_extra) |
| `--device` | auto | Force cpu or cuda |
| `--api-key` | — | API key for authentication |
| `--skip-warmup` | — | Skip startup model-weight prefetch |

Environment variables override CLI defaults: `SLOPSMITH_DEMUCS_MODEL`, `SLOPSMITH_DEMUCS_DEVICE`, `SLOPSMITH_API_KEY`.

### First-start model weight download

On first start the server pre-downloads model weights (~1.5 GB for all three endpoints: htdemucs_ft, Whisper medium, CREPE full, English wav2vec2). Subsequent restarts use cached weights.

The download runs in a background thread, so `/health` is queryable immediately. Each library prints its own `tqdm` progress bar.

`/health` reports per-model status:

```json
{
  "status": "ok",
  "warmup": {
    "demucs": "ready",
    "whisperx": "downloading",
    "crepe": "pending",
    "whisperx_aligners": { "en": "ready" }
  }
}
```

States: `pending` → `downloading` → `ready` | `failed: <reason>` | `skipped` | `evicted`.

Pass `--skip-warmup` for environments without internet access.

### Run as a systemd service

1. Copy and edit the service file:
```bash
cp slopsmith-demucs.service ~/.config/systemd/user/
# Edit ~/.config/systemd/user/slopsmith-demucs.service
# Set User, ExecStart paths to match your setup
nano ~/.config/systemd/user/slopsmith-demucs.service
```

2. Enable and start:
```bash
systemctl --user daemon-reload
systemctl --user enable slopsmith-demucs
systemctl --user start slopsmith-demucs
```

3. Monitor:
```bash
journalctl --user -u slopsmith-demucs --follow
```

## Docker

### Build

```bash
docker build -t slopsmith-demucs-server .
```

### Run (CPU)

```bash
docker run -p 7865:7865 slopsmith-demucs-server
```

### Run (GPU)

Requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html):

```bash
docker run --gpus all -p 7865:7865 slopsmith-demucs-server
```

### Docker Compose

```bash
# Pull from GHCR and run (CPU)
docker compose up -d

# GPU mode: uncomment runtime: nvidia + NVIDIA_* env vars in compose file
docker compose up -d
```

### Persistent model cache

Model weights are stored in `/app/cache` inside the container. The compose file maps this to a persistent volume so weights survive restarts:

```bash
docker compose down    # cache preserved
docker compose down -v # cache deleted (if using named volume)
```

To use a custom host path instead of a named volume (e.g. for Portainer or to save space on a specific drive), replace the volume in `docker-compose.yml`:

```yaml
volumes:
  - /home/AI/slopsmith-demucs-cache:/app/cache
```

Then copy the existing cache to the new location:
```bash
# Find old volume path
docker volume inspect slopsmith-demucs-server_demucs-cache
# Copy to new location
sudo cp -a /var/lib/docker/volumes/slopsmith-demucs-server_demucs-cache/_data/. /home/AI/slopsmith-demucs-cache/
```

**Cache environment variables** (all redirect to `/app/cache` to prevent container root disk exhaustion):

| Variable | Purpose |
|----------|---------|
| `SLOPSMITH_DEMUCS_CACHE` | Server cache root |
| `HF_HOME` | HuggingFace model cache |
| `TORCH_HOME` | PyTorch hub cache |
| `HUGGINGFACE_HUB_CACHE` | HuggingFace hub downloads |

### Auto-update

The container can automatically check for repository updates and restart. **Disabled by default** (safe for Portainer/deployments without `.git` access).

**To enable:**
1. Uncomment the `.git` bind mount in `docker-compose.yml`
2. Set `AUTO_UPDATE=true` in environment
3. Redeploy

**How it works:**
1. A background daemon runs inside the container
2. Every `UPDATE_CHECK_INTERVAL` seconds (default: 3600 = 1 hour), it checks if the current time matches `UPDATE_TIME` (default: 04:00)
3. At the configured time, it runs `git fetch origin` and compares `HEAD` with `@{upstream}`
4. If changes are detected, it pulls the new code, reinstalls dependencies, and gracefully restarts the server

**Configuration via environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTO_UPDATE` | `false` | Enable/disable auto-update |
| `UPDATE_TIME` | `04:00` | Time of day to check (HH:MM, 24h) |
| `UPDATE_CHECK_INTERVAL` | `3600` | Seconds between time checks (3600 = 1 hour) |
| `SKIP_WARMUP` | `false` | Skip model weight download on startup |
| `SLOPSMITH_DEMUCS_MODEL` | — | Override default Demucs model |
| `SLOPSMITH_API_KEY` | — | API authentication key |

**Disable auto-update** (default — safe for Portainer):
```bash
docker run -e AUTO_UPDATE=false -p 7865:7865 slopsmith-demucs-server
```

### GitHub Container Registry (CI)

The CI workflow (`.github/workflows/docker-build.yml`) automatically builds the Docker image, pushes it to GHCR, generates an SBOM, and runs a grype vulnerability scan on every push to `main`.

**To enable on your fork:**
1. Go to your fork on GitHub → **Actions** tab
2. Click **"I understand my workflows, go ahead and enable them"**
3. Push to `main` — the CI builds and scans automatically

**Pull the latest image:**
```bash
docker pull ghcr.io/YOUR_GITHUB_USER/slopsmith-demucs-server:latest
```

**Or from the upstream repo (once PR is merged):**
```bash
docker pull ghcr.io/byrongamatos/slopsmith-demucs-server:latest
```

**Build directly from git (no clone needed):**
```bash
# From upstream main
docker build -t slopsmith-demucs-server https://github.com/byrongamatos/slopsmith-demucs-server.git#main

# From your fork
docker build -t slopsmith-demucs-server https://github.com/YOUR_USER/slopsmith-demucs-server.git#main

# Run it
docker run --gpus all -p 7865:7865 slopsmith-demucs-server
```

**Run via Docker Compose with git build:**
```yaml
services:
  slopsmith-demucs:
    build: https://github.com/byrongamatos/slopsmith-demucs-server.git#main
    ports:
      - "7865:7865"
```


## API

### `GET /health`

Returns server status, model, GPU availability, cache directory, and per-model warmup state (see [First-start model weight download](#first-start-model-weight-download)).

### `POST /separate`

Separate audio into stems.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | Upload | Audio file |
| `stems` | Query | Comma-separated stem names (default: `drums,bass,vocals,other`) |
| `model` | Query | Override model (optional) |

### `POST /align`

Forced-align lyrics against audio using WhisperX (faster-whisper transcription + wav2vec2 forced aligner).

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | Form (file) | Audio file (vocals stem) |
| `text` | Form | Plain text lyrics |
| `language` | Form | ISO 639-1/2 language code hint, e.g. `en`, `es`, `pt` (optional, auto-detected). Must be 2–8 lowercase letters; subtags like `en-US` are not supported. |
| `granularity` | Form | `line` (default), `word`, `syllable`, or `phoneme` |

Granularity behaviour:

- `line` — segment-level boundaries.
- `word` — wav2vec2-aligned word timestamps. The first entry in each line carries `new_line: true`.
- `syllable` — `word` output split via pyphen; carries `new_line` on the first syllable of each line.
- `phoneme` — character-level CTC token timestamps from the aligner. Each entry carries `phoneme: true`. With wav2vec2 character models these are letter-aligned; with phoneme-trained models they're true phonemes.

Returns: `{"segments": [...], "language": "en"}` where each segment is `{start, end, text, ...}`.

### `POST /pitch`

Per-syllable pitch extraction using CREPE.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | Form (file) | Vocals stem (any format librosa can read) |
| `lyrics` | Form | JSON array of `{"t": float, "d": float}` — token start / duration in seconds |

Returns: `{"notes": [{"t": 12.34, "d": 0.5, "midi": 64}, ...]}`. Tokens for which no pitch could be estimated (even after neighbour-borrow) are omitted.

### `GET /download/{job_id}/{stem}`

Download a separated stem by job ID.

### `GET /jobs` / `GET /jobs/{job_id}`

List or inspect separation jobs.

### `WS /ws/jobs/{job_id}`

WebSocket for real-time separation progress updates.

### Configure in Slopsmith

Set the Demucs Server URL to `http://<your-server-ip>:7865` in Slopsmith settings.
