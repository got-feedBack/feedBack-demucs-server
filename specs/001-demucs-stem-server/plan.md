# Implementation Plan: Demucs Stem Server

**Branch**: `001-demucs-stem-server` (retrospective) | **Date**: 2026-05-09
**Spec**: [spec.md](./spec.md)

## Summary

A single FastAPI process exposing three ML endpoints (`/separate`,
`/align`, `/pitch`) plus job/health/download support. The whole server
lives in `server.py` (~1.7k LOC). It is meant to be co-located with a
CUDA GPU and called over HTTP from a feedback instance running anywhere.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: FastAPI, uvicorn, demucs, torch, whisperx,
torchcrepe, librosa, soundfile, pyphen, python-multipart (see
`requirements.txt`).
**Storage**: Filesystem cache at `FEEDBACK_DEMUCS_CACHE`
(default `~/.cache/feedback-demucs/`). No DB.
**Testing**: [NEEDS CLARIFICATION: no test suite present in repo.]
**Target Platform**: Linux desktop with NVIDIA GPU (CPU fallback).
Deployed as a user-level systemd service via
`feedback-demucs.service`.
**Project Type**: Single-service backend (HTTP).
**Performance Goals**: 4-minute song separates in under 30 s on a
modern CUDA GPU once warm; `/health` responds within 50 ms.
**Constraints**: ≤10 GB VRAM target; `expandable_segments` mandatory.
Total weight footprint ~1.5 GB across four models. `MAX_CONCURRENT=2`
heavy jobs.
**Scale/Scope**: Single-tenant, single-host. feedback ecosystem
dictates request volume — typically a handful of separations per song
ingestion, plus on-the-fly `/align` and `/pitch` calls per song.

## Constitution Check

| Principle | Where it shows up |
|---|---|
| I. GPU-co-located, NAS-friendly | HTTP-only contract; `--host 0.0.0.0` default; CORS wildcard for local nets. |
| II. Three endpoints, one process | All endpoints in `server.py`; shared CUDA allocator; one systemd unit. |
| III. First-run friendliness | Background warmup thread launched from FastAPI startup hook (`server.py`); `/health.warmup` per-model state; `--skip-warmup`. |
| IV. Cache-first, idempotent | `CACHE_DIR` keyed by audio hash + stem set; per-stem `/download/{job_id}/{stem}`. |
| V. CUDA memory hygiene | `os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", ...)` before `import torch` (`server.py`); WhisperX aligner LRU. |
| VI. Trust the checkpoint source | `torch.load` monkey-patch forcing `weights_only=False` (`server.py`). |
| VII. Progress is a first-class output | `WS /ws/jobs/{job_id}`; `GET /jobs[/id]`; tqdm bars on stderr for warmup. |

No deviations.

## Project Structure

```
feedback-demucs-server/
├── server.py                    # The whole server. ~1.7k LOC.
├── run_demucs.py                # Standalone separator subprocess driver. ~70 LOC.
├── requirements.txt
├── feedback-demucs.service # systemd user-unit
├── README.md
└── CLAUDE.md
```

No `tests/`, no `src/` layout. Future test additions land in
`tests/` at repo root.

## Architecture & Data Flow

### `/separate`

```
Client ──► POST /separate (multipart file)
            │
            ▼
        FastAPI handler
            │  hash audio + stems → cache key
            │  if hit: return job pointing at cached files
            │  else: spawn job, queue under MAX_CONCURRENT=2
            ▼
        run_demucs.py subprocess (per job)
            │  writes per-stem files under CACHE_DIR/<job_id>/
            │  posts progress to internal queue
            ▼
        Job table updates ──► WS /ws/jobs/{id} subscribers
            │
            ▼
        Client polls GET /jobs/{id} or downloads via /download/{id}/{stem}
```

### `/align`

```
multipart {file=vocals, text, language?, granularity?}
       │
       ▼
WhisperX faster-whisper transcription
       │  → ASR segments
       ▼
wav2vec2 forced aligner (per-language, LRU-cached)
       │  → word-level timestamps
       ▼
Granularity post-processor:
   - line: keep ASR segments
   - word: tag first word per line with new_line=true
   - syllable: pyphen split + tag first syllable per line
   - phoneme: emit raw CTC tokens with phoneme=true
       │
       ▼
{segments: [...], language: "en"}
```

### `/pitch`

```
multipart {file=vocals, lyrics=[{t,d}, ...]}
       │
       ▼
torchcrepe (CREPE full) → frame-level pitch + confidence
       │
       ▼
Per-syllable aggregation:
   - confidence-weighted mode of semitone bin within [t, t+d]
   - song-wide median
   - clamp each note into ±12 semitones around median
   - octave-error correction against median
   - neighbour-borrow for tokens without detection
       │
       ▼
{notes: [{t, d, midi}, ...]}  (omits truly silent tokens)
```

## Design Decisions

### One file, three models

`server.py` is intentionally monolithic — one process so heavy weights
share the CUDA allocator, one systemd unit to manage, one health
endpoint to query. Splitting into per-endpoint services would force
duplicated weight loading and three-way LRU coordination.

### LRU on wav2vec2 aligners, not on Demucs / Whisper / CREPE

Demucs / Whisper / CREPE are each one model. wav2vec2 aligners are
**per-language**, so a multilingual user could in principle pin a
dozen models in VRAM. The LRU only exists where multiplicity exists.

### Background warmup thread, not blocking startup

If warmup blocked `app.on_event("startup")`, `/health` would not be
queryable until ~1.5 GB of weights downloaded. The thread is launched
from the startup hook, but only after uvicorn has bound the port, so
operators can poll `/health.warmup` and watch tqdm in `journalctl`.

### `expandable_segments` set via `os.environ.setdefault` BEFORE `import torch`

Constitution Principle V — non-negotiable. `setdefault` lets a
caller override via systemd `Environment=` without source edits.

### `torch.load` monkey-patch BEFORE any model library import

PyTorch 2.6 default `weights_only=True` rejects pickled checkpoints
that demucs and torchaudio still ship. The wrapper preserves the
positional/keyword signature so libraries that explicitly pass
`weights_only=True` still get the safe-mode override.

## Constraints Worth Restating

- No request quotas, no per-user isolation. The optional API key is the
  only auth.
- No model versioning in cache keys yet. [NEEDS CLARIFICATION: does
  changing `--model` invalidate prior cache entries, or do they coexist?]
- No persistent job table — restarting the server loses the in-memory
  job dict. Cache-on-disk survives, so re-issuing `/separate` simply
  hits the cache and creates a fresh job row.

## Slopsmith Ecosystem Integration

- **feedback server**: configured via feedback Settings →
  "Demucs Server URL". When unset, stem-aware features degrade to
  `.sloppak`-baked stems only.
- **Lyrics Sync plugin**: calls `/align` for line/word timestamps.
- **Lyrics Karaoke plugin**: calls `/align` for syllables + `/pitch`
  for MIDI bars.
- **feedback Demo**: deliberately ships *without* this server —
  demo uses pre-baked `lyrics.json` + `vocal_pitch.json` inside the
  bundled `.sloppak`. See `slopsmith-demo/README.md` "What's blocked".
