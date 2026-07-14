# Slopsmith Demucs Server

A lightweight GPU-accelerated service providing AI source separation, lyrics alignment, and per-syllable pitch extraction for [Slopsmith](https://github.com/byrongamatos/slopsmith). Designed to run on a desktop with a CUDA GPU while Slopsmith runs on a NAS or Docker host.

[![Docker Build](https://github.com/byrongamatos/slopsmith-demucs-server/actions/workflows/docker-build.yml/badge.svg)](https://github.com/byrongamatos/slopsmith-demucs-server/actions/workflows/docker-build.yml)

## Features

### Source Separation (`POST /separate`)

Splits audio into individual stems using [Demucs](https://github.com/facebookresearch/demucs):

- Default model: **`htdemucs_ft`** (4-stem fine-tuned: drums, bass, vocals, other)
- Other models selectable per-request: `htdemucs_6s` (6-stem incl. guitar/piano), `mdx_extra` (lighter)
- **`bs_roformer_sw`** — BS-Roformer-SW (6-stem: vocals/drums/bass/guitar/piano/other), via
  [audio-separator](https://github.com/nomadkaraoke/python-audio-separator). Higher SDR than
  Demucs (notably bass/guitar) with far less cross-stem bleed; checkpoint (~700 MB) lazy-downloads
  on first use to `<cache>/_roformer-models/`. Stems returned as lossless FLAC.
- File upload or URL input
- Per-stem caching, keyed by audio **and model** (avoids re-processing; same song under two models caches separately)
- WebSocket progress updates

### Lyrics Transcription (`POST /transcribe`)

Transcribe sung audio to **word-level timed lyrics** — no lyrics supplied. Whisper hears
the words; wav2vec2 then force-aligns Whisper's own transcript, giving timestamps far tighter
than Whisper's segment boundaries (a chart built from raw Whisper segments sits visibly late).

- Answers the question `/align` can't: *what are the lyrics, and when is each word sung?*
- Automatic language detection (or manual hint)
- An instrumental track transcribes to `{"segments": []}` — that's an answer, not an error

### Lyrics Alignment (`POST /align`)

Forced alignment of plain text lyrics against an audio file using
[WhisperX](https://github.com/m-bain/whisperX) — Whisper transcription
plus a wav2vec2 forced aligner for tighter sub-word timestamps:

- **Line, word, syllable, or phoneme granularity**
- Phoneme/character-level CTC alignment via wav2vec2 (per-language model)
- Syllable splitting layered on word output via pyphen hyphenation (CJK character support)
- Automatic language detection (or manual language hint)
- Used by the [Lyrics Sync plugin](https://github.com/got-feedback/feedback-plugin-lyrics-sync)
  and the [Lyrics Karaoke plugin](https://github.com/got-feedback/feedback-plugin-lyrics-karaoke)

### Per-syllable Pitch Extraction (`POST /pitch`)

Estimates one MIDI note per syllable from a vocals stem using
[CREPE](https://github.com/marl/crepe) via
[torchcrepe](https://github.com/maxrmorrison/torchcrepe). Powers the
karaoke pitch chart in the
[Lyrics Karaoke plugin](https://github.com/got-feedback/feedback-plugin-lyrics-karaoke):

- CREPE neural pitch tracker — order-of-magnitude fewer octave errors than pYIN
- Confidence-weighted mode-of-semitone aggregation per syllable
- Song-wide range narrowing (clamps each syllable to ±12 semitones around the median)
- Octave-error correction against the song-wide median
- Neighbour-borrowed pitch for tokens CREPE can't lock (so whispered phrases still get bars)

## Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU fallback
- FFmpeg (`apt install ffmpeg` / `brew install ffmpeg`)

### Install (Native)

```bash
git clone https://github.com/got-feedback/feedback-demucs-server.git
cd feedback-demucs-server
python -m venv .venv
source .venv/bin/activate

# Step 1: Install main dependencies (fastapi, whisperx, torchcrepe, etc.)
# whisperx pins torch~=2.8.0 + torchaudio~=2.8.0
pip install -r requirements.txt

# Step 2: Install audio-separator SEPARATELY (diffq source-build workaround)
# Its deps pull in `diffq`, which has no wheel for Python 3.11+ and would try to
# compile from source. Its real deps are already in requirements.txt.
pip install "audio-separator>=0.44.0" --no-deps

# Step 3: Install demucs SEPARATELY (torchaudio version conflict workaround)
# demucs requires torchaudio<2.1, which conflicts with whisperx.
# Installing with --no-deps bypasses the bad pin.
# dora-search is demucs's logging lib (imported as `import dora`).
pip install demucs --no-deps
pip install einops julius lameenc openunmix pyyaml tqdm dora-search sphn

# Step 4 (optional): diffq, for QUANTIZED demucs checkpoints only.
# --only-binary=:all: makes pip fail rather than fall back to the sdist, so this can
# never start a compile.
#
# No `|| true` here on purpose: that would hide a network/index/permissions failure too,
# and you would think diffq was installed when it isn't. Let it fail loudly, and skip it
# ONLY if pip says "No matching distribution found" / "Could not find a version that
# satisfies" — that means no wheel exists for your Python (macOS on 3.11+, Linux on 3.13),
# which is safe: the bs_roformer_sw model feedBack splits with does not use diffq.
pip install "diffq-fixed>=0.2" --no-deps --only-binary=:all:
```

> ⚠️ **Why the separate install steps?**
>
> **`demucs`** (PyPI 4.0.1) pins `torchaudio<2.1` while `whisperx` needs `torchaudio~=2.8.0`. These are incompatible. Installing demucs with `--no-deps` avoids the conflict. Demucs works fine with modern torchaudio — only the `save_audio` function had issues, and that's patched in `run_demucs.py` to use `soundfile` instead.
>
> **`audio-separator`** declares `diffq (>=0.2); sys_platform != "win32"`. `diffq` is a C-extension whose newest wheels stop at **cp310** — so on any Python 3.11+ pip falls back to its sdist and needs a C compiler. That is why this is not in `requirements.txt`: `pip install -r` resolves that file first, so the compile would fail before the `--no-deps` step could run. (Windows escapes this by accident: it resolves to `diffq-fixed`, which does ship modern wheels.) `audio-separator`'s real runtime deps are listed in `requirements.txt` instead, and it is installed on top with `--no-deps`.

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

# GPU mode: uncomment `gpus: all` in the compose file (needs nvidia-container-toolkit;
# Linux or Windows/WSL2 only — macOS cannot pass a GPU through at all)
docker compose up -d
```

> #### ⚠️ Ran an image from before 2026-07-12? Delete the cache volume.
>
> Early images created their model-cache volume **owned by root**, while the server runs as
> an unprivileged user (uid 10001). The container would start and then immediately die with
>
> ```text
> PermissionError: [Errno 13] Permission denied: '/app/cache/_roformer-models'
> ```
>
> ...which `restart: unless-stopped` turns into a crash-loop.
>
> The image is fixed — but **pulling the new image is not enough on its own**. Docker sets a
> volume's ownership only when it *first creates* it, so a volume made by an older image stays
> root-owned forever and will keep crash-looping on a perfectly good image. Remove it:
>
> ```bash
> docker compose down
> docker volume rm feedback-demucs-cache   # or: slopsmith-demucs-server_demucs-cache
> docker compose up -d
> ```
>
> The volume only holds cached model weights — deleting it costs you a re-download, nothing else.

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
| `CACHE_TTL` | `24h` | Cache cleanup TTL (`1h`, `12h`, `24h`, or `NEVER` to disable auto-cleanup) |

**Disable auto-update** (default — safe for Portainer):
```bash
docker run -e AUTO_UPDATE=false -p 7865:7865 slopsmith-demucs-server
```

### Cache cleanup

The server automatically deletes old stem cache directories to prevent disk growth. A background thread runs every 10 minutes, checks each stem cache directory under `SLOPSMITH_DEMUCS_CACHE`, and removes directories older than `CACHE_TTL`.

Model weight caches (`torch/`, `huggingface/`, `locale/`) are **never** deleted — only the stem output cache is cleaned.

| Variable | Default | Description |
|----------|---------|-------------|
| `CACHE_TTL` | `24h` | Maximum age of cache entries (`1h`, `12h`, `24h`, or `NEVER` to disable) |

**Disable auto-cleanup:**
```bash
docker run -e CACHE_TTL=NEVER -p 7865:7865 slopsmith-demucs-server
```

**Set custom TTL (e.g. 12 hours):**
```bash
docker run -e CACHE_TTL=12h -p 7865:7865 slopsmith-demucs-server
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

### `POST /transcribe`

Transcribe sung audio to word-level timed lyrics using WhisperX (faster-whisper ASR + wav2vec2 forced alignment of its own transcript). Use this when you do **not** have the lyrics; use `/align` when you do.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | Form (file) | Audio file (vocals stem) |
| `language` | Form | ISO 639-1/2 language code hint, e.g. `en`, `es`, `pt` (optional, auto-detected). Case-insensitive — the value is trimmed and lowercased before validation, so `EN` and `en` both work. Must then be 2–8 letters; subtags like `en-US` are rejected (400) because the hyphen is not a letter. |

Returns native WhisperX alignment output:

```json
{"segments": [{"start": 12.3, "end": 15.1, "text": "hello world",
               "words": [{"word": "hello", "start": 12.3, "end": 12.8, "score": 0.94}]}],
 "language": "en"}
```

A stem with no singing in it returns `{"segments": [], "language": "en"}` with a 200 — an instrumental is a valid answer, not a failed request.

### `POST /align`

Forced-align lyrics against audio using WhisperX (faster-whisper transcription + wav2vec2 forced aligner).

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | Form (file) | Audio file (vocals stem) |
| `text` | Form | Plain text lyrics |
| `language` | Form | ISO 639-1/2 language code hint, e.g. `en`, `es`, `pt` (optional, auto-detected). Case-insensitive — the value is trimmed and lowercased before validation, so `EN` and `en` both work. Must then be 2–8 letters; subtags like `en-US` are rejected (400) because the hyphen is not a letter. |
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

## License

**AGPL-3.0-only** — the same license as the [feedBack](https://github.com/got-feedBack/feedBack)
app this server exists to serve, and as the plugins that call it. See [LICENSE](LICENSE).
