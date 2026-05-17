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
pip install demucs --no-deps
pip install einops julius lameenc openunmix pyyaml tqdm
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
# CPU mode
docker compose up -d

# GPU mode (uncomment deploy.resources in docker-compose.yml first)
docker compose up -d
```

The compose file mounts `.git` as a bind volume to enable auto-update (see below).

### Persistent model cache

The Docker image stores downloaded model weights in `/app/cache`. A named volume `demucs-cache` is defined in `docker-compose.yml` to persist weights across restarts:

```bash
docker compose down    # cache preserved
docker compose down -v # cache deleted
```

### Auto-update

When running in Docker, the server can automatically check for repository updates and restart with the new code.

**How it works:**
1. A background daemon runs inside the container
2. Every `UPDATE_CHECK_INTERVAL` seconds (default: 3600 = 1 hour), it checks if the current time matches `UPDATE_TIME` (default: 04:00)
3. At the configured time, it runs `git fetch origin` and compares `HEAD` with `@{upstream}`
4. If changes are detected, it pulls the new code, reinstalls dependencies, and gracefully restarts the server

**Configuration via environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTO_UPDATE` | `true` | Enable/disable auto-update |
| `UPDATE_TIME` | `04:00` | Time of day to check (HH:MM, 24h) |
| `UPDATE_CHECK_INTERVAL` | `3600` | Seconds between time checks (3600 = 1 hour) |
| `SKIP_WARMUP` | `false` | Skip model weight download on startup |
| `SLOPSMITH_DEMUCS_MODEL` | — | Override default Demucs model |
| `SLOPSMITH_API_KEY` | — | API authentication key |

**Important:** Auto-update requires the `.git` directory to be available inside the container. The provided `docker-compose.yml` binds `.git` from the host. If running with plain `docker run`, mount it manually:

```bash
docker run -v $(pwd)/.git:/app/.git -p 7865:7865 slopsmith-demucs-server
```

**Disable auto-update:**

```bash
docker run -e AUTO_UPDATE=false -p 7865:7865 slopsmith-demucs-server
```

### GitHub Container Registry

The CI workflow (`.github/workflows/docker-build.yml`) automatically builds and pushes the image to `ghcr.io` on every push to `main`.

To pull the pre-built image:

```bash
docker pull ghcr.io/byrongamatos/slopsmith-demucs-server:latest
docker run -p 7865:7865 ghcr.io/byrongamatos/slopsmith-demucs-server:latest
```

## API

### `GET /health`

Returns server status, model, GPU availability, cache directory, and per-model warmup state.

### `POST /separate`

Separate audio into stems.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | Upload | Audio file |
| `stems` | Query | Comma-separated stem names (default: `drums,bass,vocals,other`) |
| `model` | Query | Override model (optional) |

### `POST /align`

Forced-align lyrics against audio.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | Form (file) | Audio file |
| `text` | Form | Plain text lyrics |
| `language` | Form | ISO 639-1/2 language code hint (optional, auto-detected) |
| `granularity` | Form | `line` (default), `word`, `syllable`, or `phoneme` |

### `POST /pitch`

Per-syllable pitch extraction using CREPE.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | Form (file) | Vocals stem |
| `lyrics` | Form | JSON array of `{"t": float, "d": float}` — token start/duration |

### `GET /download/{job_id}/{stem}`

Download a separated stem by job ID.

### `GET /jobs` / `GET /jobs/{job_id}`

List or inspect separation jobs.

### `WS /ws/jobs/{job_id}`

WebSocket for real-time separation progress updates.

### Configure in Slopsmith

Set the Demucs Server URL to `http://<your-server-ip>:7865` in Slopsmith settings.
