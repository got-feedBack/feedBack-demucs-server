# Feature Specification: Demucs Stem Server

**Feature Branch**: `001-demucs-stem-server` (retrospective)
**Created**: 2026-05-09
**Status**: Implemented (documented after the fact)
**Input**: A GPU-hosting service that gives feedBack on-demand source
separation, lyrics alignment, and per-syllable pitch over HTTP.

## User Scenarios & Testing

### User Story 1 — Separate a song into stems on demand (Priority: P1)

A feedBack user has a song with no pre-baked stems. They want to play
along with just the bass, or strip the vocals for karaoke. The feedBack
server forwards the audio to this Demucs server, which returns
per-stem files; the response is cached so re-requests are instant.

**Why this priority**: Stem separation is the foundational capability
the whole server was built for. Without it the other two endpoints
have no reason to exist on a separate host.

**Independent Test**: `POST /separate` with a 4-minute audio file,
subscribe to `/ws/jobs/{job_id}`, watch progress reach 100, then
download each stem from `/download/{job_id}/{stem}`. Re-request the
same audio and confirm the second job reports a cache hit instantly.

**Acceptance Scenarios**:

1. **Given** the server is warm, **When** a client POSTs unique audio
   to `/separate`, **Then** a job is created, queued (≤
   `MAX_CONCURRENT` running), and progress is published over WebSocket.
2. **Given** the server has previously separated this audio, **When**
   the same audio is POSTed again, **Then** the response references
   the cached stems without invoking Demucs.
3. **Given** a `model=htdemucs_6s` query, **When** the request is
   accepted, **Then** the 6-stem variant runs (drums, bass, vocals,
   guitar, piano, other).
4. **Given** the GPU is unavailable, **When** the server boots,
   **Then** it falls back to CPU and `/health.gpu == false`.

---

### User Story 2 — Force-align lyrics for karaoke (Priority: P2)

A user has plain-text lyrics and a vocals stem. They want syllable-level
timestamps so the Lyrics-Karaoke plugin can highlight syllables as they
are sung.

**Why this priority**: Karaoke and lyrics-sync features depend on
this. Without `/align` the karaoke plugin shipped by feedBack is
ornamental.

**Independent Test**: `POST /align` with a vocals stem,
`text="line one\nline two\n..."`, `granularity=syllable`. Confirm
returned segments split per syllable, with `new_line: true` on the
first syllable of each line.

**Acceptance Scenarios**:

1. **Given** `granularity=line`, **When** alignment completes, **Then**
   each segment matches one line of the input text.
2. **Given** `granularity=word`, **When** alignment completes, **Then**
   the first word of each line carries `new_line: true`.
3. **Given** `granularity=syllable`, **When** alignment completes,
   **Then** words are split via pyphen and the first syllable of each
   line carries `new_line: true`.
4. **Given** `granularity=phoneme`, **When** alignment completes,
   **Then** each row carries `phoneme: true`.
5. **Given** `language=es` is supplied but the Spanish wav2vec2
   aligner has not been pre-warmed, **When** the request runs,
   **Then** it lazy-downloads the aligner and succeeds (slower first
   call, fast subsequent ones).
6. **Given** an invalid language tag like `en-US`, **When** the
   request is received, **Then** the server rejects it (must be 2–8
   lowercase letters).

---

### User Story 3 — Per-syllable pitch for karaoke bars (Priority: P3)

The Lyrics-Karaoke plugin needs one MIDI note per syllable so it can
draw pitch bars in a SingStar-style highway.

**Why this priority**: Polish on top of US2 — karaoke works without
pitch bars, just less spectacularly.

**Independent Test**: `POST /pitch` with a vocals stem and the
`{t, d}` token list returned by US2's `/align`. Verify each returned
note is within ±12 semitones of the song-wide median, that whispered
or quiet tokens still receive a neighbour-borrowed MIDI note where
possible, and that genuinely unvoiced tokens are omitted.

**Acceptance Scenarios**:

1. **Given** a clean vocals stem, **When** pitch extraction runs,
   **Then** each token's MIDI note matches a confidence-weighted mode
   of the semitone bin within `[t, t+d]`.
2. **Given** an octave-error candidate (CREPE returns +12 semitones
   from the song median), **When** the post-processor runs, **Then**
   it is corrected back into the median ±12 range.
3. **Given** a low-confidence token between two confident ones,
   **When** post-processing runs, **Then** the low-confidence token
   borrows the nearer neighbour's MIDI note.

---

### Edge Cases

- Network drop mid-WebSocket: client reconnects with same `job_id`
  and the server re-streams current state.
- 80 MB+ audio file: [NEEDS CLARIFICATION: is there an upload size cap?
  None explicit in `server.py`.]
- Two clients call `/separate` with the same audio simultaneously:
  one runs Demucs; both should receive the cached result. [NEEDS
  CLARIFICATION: is there an in-flight de-dup, or do both run?]
- WhisperX aligner LRU evicts the English model under multilingual
  load: `/health.warmup.whisperx == "evicted"` is reported, and the
  next `/align en` request lazy-reloads.
- `--skip-warmup` set in an offline environment: per-endpoint calls
  succeed only for already-cached weights; otherwise they fail loudly.

## Requirements

### Functional Requirements

- **FR-001**: System MUST expose `GET /health` reporting `status`,
  `model`, `gpu`, `cache_dir`, and `warmup` per-model state.
- **FR-002**: System MUST expose `POST /separate` accepting multipart
  `file` and optional `stems`, `model`.
- **FR-003**: System MUST cache separation outputs keyed by audio
  hash + stem set; cache hits MUST skip Demucs.
- **FR-004**: System MUST stream job progress over
  `WS /ws/jobs/{job_id}`.
- **FR-005**: System MUST expose `GET /download/{job_id}/{stem}` for
  per-stem retrieval.
- **FR-006**: System MUST expose `GET /jobs` and `GET /jobs/{job_id}`.
- **FR-007**: System MUST expose `POST /align` accepting multipart
  `file`, `text`, optional `language`, optional `granularity` ∈
  {`line`, `word`, `syllable`, `phoneme`}.
- **FR-008**: System MUST tag the first sub-token of each input line
  with `new_line: true` for `word` / `syllable` granularity, and tag
  every token with `phoneme: true` for `phoneme` granularity.
- **FR-009**: System MUST expose `POST /pitch` accepting multipart
  `file` and a JSON token list `[{t, d}, ...]`.
- **FR-010**: System MUST clamp every detected note into ±12
  semitones around the song-wide median and apply octave-error
  correction.
- **FR-011**: System MUST accept an optional API key
  (`FEEDBACK_API_KEY` / `--api-key`) shared across all endpoints.
  [NEEDS CLARIFICATION: bearer header? Custom header? README does not
  pin this down.]
- **FR-012**: System MUST cap concurrent heavy jobs at
  `MAX_CONCURRENT = 2`.
- **FR-013**: System MUST set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before importing
  torch.
- **FR-014**: System MUST patch `torch.load` to force
  `weights_only=False` before importing any model library.
- **FR-015**: System MUST run weight pre-warming in a background
  thread launched after the port is bound; `/health` MUST be
  queryable during warmup.
- **FR-016**: System MUST support `--skip-warmup` for offline starts;
  per-endpoint calls MUST then lazy-download on demand.
- **FR-017**: System MUST keep wav2vec2 language aligners behind an
  LRU so VRAM does not grow unbounded under multilingual load.

### Key Entities

- **Job**: `{job_id, status, model, stems, progress, started_at,
  finished_at, output_files}` — held in memory in `OrderedDict`.
- **Cache entry**: `<cache_dir>/<audio_hash>/<stem>.wav` plus a small
  metadata sidecar.
- **Warmup state**: `{demucs, whisperx, crepe, whisperx_aligners{lang}}`,
  each ∈ `pending | downloading | ready | failed:<reason> | skipped |
  evicted`.
- **Align segment**: `{start, end, text, new_line?, phoneme?, ...}`.
- **Pitch note**: `{t, d, midi}`.

## Success Criteria

- **SC-001**: A 4-minute song separates into 4 stems on a modern
  CUDA GPU in under 30 seconds once the server is warm. [NEEDS
  CLARIFICATION: pin to a reference GPU.]
- **SC-002**: Cache hit on a previously separated file returns within
  500 ms.
- **SC-003**: `/health` is queryable within 50 ms from process start
  (because warmup runs in a background thread).
- **SC-004**: Switching from English to Spanish alignment, without
  prior Spanish warmup, succeeds within the time it takes to download
  the Spanish wav2vec2 aligner once.
- **SC-005**: Karaoke pitch bars produced by `/pitch` cover ≥ 90 %
  of voiced syllables in a typical pop vocals stem (the rest being
  whispered or unvoiced sections).

## Assumptions

- Operator owns the GPU host and has accepted the trust assumption
  on torchaudio / demucs / HuggingFace checkpoints.
- Disk under `FEEDBACK_DEMUCS_CACHE` is large enough to hold cached
  stems for the songs the user actually plays. No automatic eviction.
- Audio uploaded by feedBack is something `librosa` / `ffmpeg` can
  read.
- feedBack trusts the network path (no TLS in-process; reverse
  proxy if you need it).
