# Tasks: Demucs Stem Server

**Input**: Retrospective documentation of the existing implementation.
**Organization**: Tasks grouped by user story. Each task is marked
**DONE** with a file pointer when present in the codebase, or **OPEN**
when it is a real gap.

## Phase 1: Setup (Shared Infrastructure)

- [x] **DONE** T001 FastAPI app + CORS + lifespan hooks — `server.py`
- [x] **DONE** T002 Pin all ML deps with lower bounds — `requirements.txt`
- [x] **DONE** T003 systemd user-unit for boot — `feedback-demucs.service`
- [x] **DONE** T004 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  set before `import torch` — `server.py` top
- [x] **DONE** T005 `torch.load` monkey-patch forcing
  `weights_only=False` before any model lib imports — `server.py`

## Phase 2: Foundational

- [x] **DONE** T006 Cache dir bootstrap (`SLOPSMITH_DEMUCS_CACHE`,
  default `~/.cache/feedback-demucs/`) — `server.py`
- [x] **DONE** T007 Job table (`OrderedDict`) for `/jobs[/id]` and the
  WebSocket — `server.py`
- [x] **DONE** T008 [P] `MAX_CONCURRENT=2` queue/gate — `server.py`
- [x] **DONE** T009 [P] Background warmup thread launched from FastAPI
  startup hook — `server.py`
- [x] **DONE** T010 [P] WhisperX wav2vec2 aligner LRU — `server.py`
- [ ] **OPEN** T011 [P] Test harness (`tests/`) — none present.
  Without this, regressions in pitch aggregation / align granularity
  are caught only by Slopsmith-side smoke runs.

## Phase 3: User Story 1 — Stem separation (P1)

- [x] **DONE** T020 `POST /separate` handler — `server.py`
- [x] **DONE** T021 `run_demucs.py` subprocess driver —
  `run_demucs.py`
- [x] **DONE** T022 Cache-key on audio hash + stem set — `server.py`
- [x] **DONE** T023 `WS /ws/jobs/{job_id}` progress stream — `server.py`
- [x] **DONE** T024 `GET /download/{job_id}/{stem}` — `server.py`
- [x] **DONE** T025 [P] `htdemucs_6s` and `mdx_extra` selectable per
  request — `server.py`
- [ ] **OPEN** T026 Cache invalidation on `--model` change.
  [NEEDS CLARIFICATION] — see clarify.md.
- [ ] **OPEN** T027 Persistent job table across restarts.
  Currently in-memory only.

**Checkpoint**: P1 ships. Cache + WS make this independently usable
for any stem consumer.

## Phase 4: User Story 2 — Lyrics alignment (P2)

- [x] **DONE** T030 `POST /align` (line / word / syllable / phoneme)
  — `server.py`
- [x] **DONE** T031 WhisperX faster-whisper transcription path —
  `server.py`
- [x] **DONE** T032 wav2vec2 forced aligner per language —
  `server.py`
- [x] **DONE** T033 [P] pyphen syllable splitter, CJK char support
  — `server.py`
- [x] **DONE** T034 [P] `new_line: true` tag on first sub-token of
  each line — `server.py`
- [x] **DONE** T035 [P] `phoneme: true` tag on phoneme rows —
  `server.py`
- [x] **DONE** T036 Auto-detect language; accept `language` form
  field; validate 2–8 lowercase letters — `server.py`
- [x] **DONE** T037 Lazy-download non-warmed-up language aligners on
  first use — `server.py`
- [ ] **OPEN** T038 [P] Document subtag policy explicitly (we reject
  `en-US`). Update README example block if needed.

**Checkpoint**: P2 ships. Lyrics-Sync and Lyrics-Karaoke plugins are
unblocked.

## Phase 5: User Story 3 — Per-syllable pitch (P3)

- [x] **DONE** T040 `POST /pitch` accepting `{t, d}` token list —
  `server.py`
- [x] **DONE** T041 torchcrepe (CREPE full) frame-level pitch —
  `server.py`
- [x] **DONE** T042 [P] Confidence-weighted mode-of-semitone
  aggregation per syllable — `server.py`
- [x] **DONE** T043 [P] Song-wide range narrowing (clamp ±12 semitones
  around median) — `server.py`
- [x] **DONE** T044 [P] Octave-error correction against median —
  `server.py`
- [x] **DONE** T045 [P] Neighbour-borrow for low-confidence tokens —
  `server.py`
- [ ] **OPEN** T046 Optional: expose CREPE confidence in response so
  the karaoke renderer can fade low-confidence bars.

**Checkpoint**: P3 ships. Karaoke pitch bars work end-to-end.

## Phase 6: Cross-cutting / Polish

- [x] **DONE** T050 `/health` reports `status, model, gpu, cache_dir,
  warmup{demucs, whisperx, crepe, whisperx_aligners{lang}}` —
  `server.py`
- [x] **DONE** T051 `--skip-warmup` flag — `server.py`
- [x] **DONE** T052 Optional `--api-key` auth — `server.py`
- [ ] **OPEN** T053 OpenAPI/Swagger pass — FastAPI gives us
  `/docs` for free; verify request/response models are typed.
- [ ] **OPEN** T054 Disk-cache eviction policy. Currently unbounded.
- [ ] **OPEN** T055 Metrics endpoint (jobs run, cache hit-rate, GPU
  memory).

## Parallel-Safe Sets

- T010, T033 – T035, T042 – T045 are independent post-processors;
  safe to refactor in parallel.
- T011 (tests) is independent of all feature tasks.
