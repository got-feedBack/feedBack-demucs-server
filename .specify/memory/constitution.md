# feedBack Demucs Server Constitution

## Core Principles

### I. GPU-Co-Located, NAS-Friendly

The server is designed to run on a workstation that has the GPU/RAM, while
feedBack itself can live anywhere (NAS, Docker host, laptop). This split
is the entire reason the project exists. The server MUST stay HTTP-callable
from any host, MUST not require shared filesystems, and MUST tolerate
clients that come and go (jobs and per-stem downloads survive client
disconnects).

### II. Three Endpoints, One Process

`/separate`, `/align`, `/pitch` are the user-visible surface. They share the
same FastAPI process so a single `python server.py --port 7865` brings up
the whole feature set, and so heavy ML weights (Demucs, Whisper, wav2vec2,
CREPE) can share the same CUDA allocator. New ML capabilities for
feedBack plugins SHOULD be added as new endpoints in this server before
spawning a separate service.

### III. First-Run Friendliness

Total weight download is ~1.5 GB across four models. The server MUST
pre-warm weights from a background thread launched after the port is
bound, so `/health` is queryable from t=0 and operators can watch progress
via `tqdm`/`journalctl`. `--skip-warmup` MUST exist for offline boots, and
endpoints MUST lazy-download on demand if a model is not yet ready.

### IV. Cache-First, Idempotent

Separation jobs are keyed by hash of the input audio plus stem set.
Repeated requests for the same audio MUST return the cached stems
instead of re-running Demucs. Per-stem files MUST be downloadable
individually so clients can fetch only what changed.

### V. CUDA Memory Hygiene (NON-NEGOTIABLE)

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` MUST be set before any
`import torch` so the allocator can resize blocks rather than leaking
fragments. Without this, `/pitch` reliably OOMs on ≤10 GB GPUs after a
Demucs run. WhisperX language aligners MUST live behind an LRU so a
multilingual session does not pin every wav2vec2 model in VRAM forever.

### VI. Trust the Checkpoint Source

PyTorch ≥2.6 default `weights_only=True` breaks legacy torchaudio /
demucs checkpoints. The server MUST patch `torch.load` to force
`weights_only=False` because every checkpoint loaded comes from a known
trusted source (torchaudio CDN, Demucs hub, HuggingFace). The patch MUST
land before any model library is imported.

### VII. Progress Is a First-Class Output

Source separation is slow. The server MUST expose progress via WebSocket
(`/ws/jobs/{job_id}`) and MUST keep a queryable job table (`/jobs`,
`/jobs/{job_id}`) so the feedBack UI can show real progress instead of
spinning.

## Operational Constraints

- Python 3.10+, CUDA-capable GPU (with CPU fallback), FFmpeg.
- Config via env vars / CLI flags only (no config files): `--port`,
  `--host`, `--model`, `--device`, `--api-key`, `--skip-warmup`,
  `FEEDBACK_DEMUCS_MODEL`, `FEEDBACK_DEMUCS_DEVICE`,
  `FEEDBACK_API_KEY`, `FEEDBACK_DEMUCS_CACHE`.
- Default cache: `~/.cache/feedBack-demucs/`.
- Default model: `htdemucs_ft` (4-stem fine-tuned). `htdemucs_6s` and
  `mdx_extra` selectable per-request.
- Max concurrent jobs: 2 (`MAX_CONCURRENT` in `server.py`).
- Authentication: optional bearer-style API key shared across all three
  endpoints.

## Development Workflow

- Single source file: `server.py` (≈1.7k LOC). New endpoints land here
  unless they grow large enough to justify a module split.
- `run_demucs.py` is the standalone separator subprocess driver.
- Run as a user-level systemd service via `feedBack-demucs.service`.
- New ML dependencies MUST be pinned in `requirements.txt` with a lower
  bound that matches the import patterns in `server.py`.

## Governance

This server is one of several repos in the feedBack ecosystem
(`feedBack`, `feedBack-desktop`, `feedBack-demo`, `feedBack-ignition`,
plus per-feature plugin repos). The shared workspace at
`~/Repositories/feedBack-workspace/` coordinates them. Changes to the
HTTP contract here are breaking changes for the feedBack plugin repos
(`feedBack-plugin-lyrics-sync`, `feedBack-plugin-lyrics-karaoke`) and
MUST be rolled out compatibly: add new fields, never silently change
existing ones; bump server version when removing.

**Version**: 1.0.0 | **Ratified**: 2026-05-09 | **Last Amended**: 2026-05-09
