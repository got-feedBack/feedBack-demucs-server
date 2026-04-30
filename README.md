# Slopsmith Demucs Server

A lightweight GPU-accelerated service that provides AI source separation and lyrics alignment for [Slopsmith](https://github.com/byrongamatos/slopsmith). Designed to run on a desktop with a CUDA GPU while Slopsmith runs on a NAS or Docker host.

## Features

### Source Separation (`POST /separate`)

Splits audio into individual stems using [Demucs](https://github.com/facebookresearch/demucs):

- Default model: **`htdemucs_ft`** (4-stem fine-tuned: drums, bass, vocals, other)
- Other models selectable per-request: `htdemucs_6s` (6-stem incl. guitar/piano), `mdx_extra` (lighter)
- File upload or URL input
- Per-stem caching (avoids re-processing)
- WebSocket progress updates

### Lyrics Alignment (`POST /align`)

Forced alignment of plain text lyrics against an audio file using
[WhisperX](https://github.com/m-bain/whisperX) — Whisper transcription
plus a wav2vec2 forced aligner for tighter sub-word timestamps:

- **Line, word, syllable, or phoneme granularity**
- Phoneme/character-level CTC alignment via wav2vec2 (per-language model)
- Syllable splitting layered on word output via pyphen hyphenation (CJK character support)
- Automatic language detection (or manual language hint)
- Used by the [Lyrics Sync plugin](https://github.com/byrongamatos/slopsmith-plugin-lyrics-sync)
  and the [Lyrics Karaoke plugin](https://github.com/byrongamatos/slopsmith-plugin-lyrics-karaoke)

### Per-syllable Pitch Extraction (`POST /pitch`)

Estimates one MIDI note per syllable from a vocals stem using
[CREPE](https://github.com/marl/crepe) via
[torchcrepe](https://github.com/maxrmorrison/torchcrepe). Powers the
karaoke pitch chart in the
[Lyrics Karaoke plugin](https://github.com/byrongamatos/slopsmith-plugin-lyrics-karaoke):

- CREPE neural pitch tracker — order-of-magnitude fewer octave errors than pYIN
- Confidence-weighted mode-of-semitone aggregation per syllable
- Song-wide range narrowing (clamps each syllable to ±12 semitones around the median)
- Octave-error correction against the song-wide median
- Neighbour-borrowed pitch for tokens CREPE can't lock (so whispered phrases still get bars)

## Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU fallback
- FFmpeg

### Install

```bash
git clone https://github.com/byrongamatos/slopsmith-demucs-server.git
cd slopsmith-demucs-server
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
python server.py --port 7865
```

Options:
- `--port` — port to listen on (default: 7865)
- `--host` — host to bind to (default: 0.0.0.0)
- `--model` — Demucs model (default: htdemucs_ft)
- `--device` — force cpu or cuda (auto-detected by default)
- `--api-key` — optional API key for authentication
- `--skip-warmup` — skip the startup model-weight prefetch (see below)

### First-start model weight download

On first start the server pre-downloads the model weights for all
three endpoints so the first user-facing request doesn't stall on a
CDN fetch. Total download is **~1.5 GB** (htdemucs_ft, Whisper medium,
CREPE full, wav2vec2-base for the active language).

The download runs in a background thread started from the FastAPI
startup hook, so it fires only after the server has bound the port —
meaning `/health` is queryable from the very first moment.  Each
library prints its own `tqdm` progress bar to stderr — the operator
sees real byte-level progress in the terminal or in
`journalctl -u slopsmith-demucs --follow`.

`/health` reports per-model status under `warmup`:

```json
{
  "status": "ok",
  "warmup": {
    "demucs": "ready",
    "whisperx": "downloading",
    "crepe": "pending",
    "whisperx_aligners": {
      "en": "ready",
      "es": "downloading"
    }
  }
}
```

Values:
- `pending` → `downloading` → `ready` — normal warmup progression.
- `failed: <reason>` — download or model-load error; endpoint still works, just lazy-downloads on first request.
- `skipped` — `--skip-warmup` was passed; per-endpoint calls lazy-download on demand.
- `evicted` — an LRU aligner was evicted to free memory (appears in `whisperx_aligners`; also appears at the top-level `whisperx` field if the English aligner is evicted, since that breaks the warmup contract).

Subsequent restarts use the cached weights and reach `ready` within
a couple of seconds. Pass `--skip-warmup` if you need to start the
server in an environment without internet access; per-endpoint calls
will lazy-download on demand instead.

The top-level `whisperx` field reflects the warmup contract — ASR
model + English wav2vec2 aligner. Other languages aren't pre-warmed
(we don't know which ones a client will use) and download on the
first `/align` request in that language. The `whisperx_aligners` map
exposes per-language aligner state so multilingual clients can poll
for their language's readiness before issuing a real `/align` call.

### Run as a systemd service

```bash
cp slopsmith-demucs.service ~/.config/systemd/user/
systemctl --user enable slopsmith-demucs
systemctl --user start slopsmith-demucs
```

Edit the service file to adjust the path to your clone and desired model.

### Configure in Slopsmith

In Slopsmith settings, set the Demucs Server URL to `http://<your-desktop-ip>:7865`.

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

Returns: `{"notes": [{"t": 12.34, "d": 0.5, "midi": 64}, ...]}`. Tokens for which no
pitch could be estimated (even after neighbour-borrow) are omitted.

### `GET /download/{job_id}/{stem}`

Download a separated stem by job ID.

### `GET /jobs` / `GET /jobs/{job_id}`

List or inspect separation jobs.

### `WS /ws/jobs/{job_id}`

WebSocket for real-time separation progress updates.
