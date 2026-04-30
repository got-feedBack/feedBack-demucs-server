"""Slopsmith Demucs Separation Service.

Lightweight HTTP server that runs demucs source separation.
Designed to run on a desktop with GPU/RAM while Slopsmith runs on a NAS.

Usage:
    python server.py --port 7865
    python server.py --port 7865 --device cuda
    python server.py --port 7865 --model mdx_extra --api-key mysecret
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# Heavy imports happen at module top so a missing dep crashes startup
# loudly with ModuleNotFoundError instead of silently disabling features
# at request time. Everything below is required (see requirements.txt).
import torch
import torchcrepe
import librosa
import whisperx

# ── Configuration ───────────────────────────────────────────────────────

DEMUCS_MODEL = os.environ.get("SLOPSMITH_DEMUCS_MODEL", "htdemucs_ft")
DEMUCS_DEVICE = os.environ.get("SLOPSMITH_DEMUCS_DEVICE", "")
API_KEY = os.environ.get("SLOPSMITH_API_KEY", "")
CACHE_DIR = Path(os.environ.get(
    "SLOPSMITH_DEMUCS_CACHE",
    Path.home() / ".cache" / "slopsmith-demucs",
))
MAX_CONCURRENT = 2

# ── State ───────────────────────────────────────────────────────────────

app = FastAPI(title="Slopsmith Demucs Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Default: run warmup. main() overrides to True when --skip-warmup is passed.
app.state.skip_warmup = False

# Jobs: job_id -> {status, progress, stems, error, created_at, audio_hash, model}
jobs: OrderedDict[str, dict] = OrderedDict()
jobs_lock = threading.Lock()
active_count = 0
active_lock = threading.Lock()

# WebSocket subscribers: job_id -> set of WebSocket
ws_subscribers: dict[str, set] = {}

# Resolved config (set at startup)
_model = DEMUCS_MODEL
_device = ""
_gpu_available = False

# ── Warmup state ────────────────────────────────────────────────────────
#
# On first start, demucs / whisperx / torchcrepe each pull their model
# weights from a CDN — together ~1.5 GB. The warmup thread does these
# downloads up front so the first user-facing /separate /align /pitch
# call doesn't hang on a CDN fetch. Each library prints its own tqdm
# progress bar to stderr so the admin sees actual download progress
# in the terminal / journalctl. Clients can also poll /health to see
# the status of each model.

# warmup_state[name] = "pending" | "downloading" | "ready" | "failed: <reason>"
#                    | "skipped" (--skip-warmup) | "evicted" (LRU aligner evicted)
warmup_state: dict[str, str] = {
    "demucs": "pending",
    "whisperx": "pending",
    "crepe": "pending",
}
warmup_state_lock = threading.Lock()

# Per-language wav2vec2 aligner state. The top-level "whisperx" entry
# above tracks the warmup contract (ASR + en aligner). This dict tells
# clients which other-language aligners are currently loaded so non-
# English /align callers can poll for their own language's readiness.
# The first /align in a language flips its entry from "downloading"
# (transient) to "ready"; failures land as "failed: <reason>".
warmup_aligners: dict[str, str] = {}
warmup_aligners_lock = threading.Lock()


def _set_warmup_state(name: str, value: str) -> None:
    with warmup_state_lock:
        warmup_state[name] = value
    # Print on every transition so the systemd journal carries a
    # readable trace alongside the per-library tqdm bars.
    print(f"[warmup] {name}: {value}", flush=True)


def _set_aligner_state(language: str, value: str) -> None:
    with warmup_aligners_lock:
        warmup_aligners[language] = value
    print(f"[warmup] whisperx aligner ({language}): {value}", flush=True)


# ── Auth middleware ─────────────────────────────────────────────────────

@app.middleware("http")
async def check_api_key(request, call_next):
    if API_KEY and request.url.path not in ("/health", "/docs", "/openapi.json"):
        key = request.headers.get("X-API-Key", request.query_params.get("api_key", ""))
        if key != API_KEY:
            return JSONResponse({"error": "Unauthorized"}, 401)
    return await call_next(request)


# ── Startup hook ─────────────────────────────────────────────────────────
#
# The warmup is registered here so it runs AFTER uvicorn has bound the
# port — meaning /health is already queryable the moment the warmup
# thread starts. Starting the thread before uvicorn.run() would give no
# such guarantee.

@app.on_event("startup")
async def _startup_event():
    if app.state.skip_warmup:
        for k in warmup_state:
            _set_warmup_state(k, "skipped")
    else:
        threading.Thread(target=_run_warmup, daemon=True).start()


# ── Health ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    with warmup_state_lock:
        warmup = dict(warmup_state)
    with warmup_aligners_lock:
        aligners = dict(warmup_aligners)
    # Surface per-language aligner state alongside the top-level
    # whisperx field. The top-level value reflects the warmup contract
    # (ASR + en aligner ready); the aligners dict shows which other
    # languages have been loaded so non-English /align callers can
    # poll for their own language before issuing a real request.
    warmup["whisperx_aligners"] = aligners
    return {
        "status": "ok",
        "demucs_model": _model,
        "gpu": _gpu_available,
        "device": _device,
        "cache_dir": str(CACHE_DIR),
        # Per-model warmup status. Values: pending | downloading | ready |
        # failed: <reason>. Clients (lyrics_karaoke etc.) can poll this
        # to wait for `crepe == "ready"` before relying on /pitch
        # latency, or to surface a user-visible "server warming up"
        # progress indicator.
        "warmup": warmup,
    }


# ── Separation via file upload ──────────────────────────────────────────

@app.post("/separate")
async def separate_upload(
    file: UploadFile = File(...),
    stems: str = Query("drums,bass,vocals,other"),
    model: str = Query(""),
):
    use_model = model or _model
    stem_list = [s.strip() for s in stems.split(",") if s.strip()]

    # Save upload to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "audio.mp3").suffix)
    content = await file.read()
    tmp.write(content)
    tmp.close()

    # Hash for cache key
    audio_hash = hashlib.sha256(content).hexdigest()[:16]
    job_id = audio_hash

    # Check cache
    cached = _check_cache(job_id, stem_list, use_model)
    if cached:
        os.unlink(tmp.name)
        return {"job_id": job_id, "stems": cached, "cached": True}

    # Queue the job
    result = _enqueue_job(job_id, tmp.name, stem_list, use_model)
    if result.get("error"):
        return JSONResponse(result, 503)
    return result


# ── Separation via URL ──────────────────────────────────────────────────

@app.post("/separate-url")
async def separate_url(
    data: dict,
    stems: str = Query("drums,bass,vocals,other"),
    model: str = Query(""),
):
    url = data.get("url", "").strip()
    if not url:
        return JSONResponse({"error": "url required"}, 400)

    use_model = model or _model
    stem_list = [s.strip() for s in stems.split(",") if s.strip()]

    # Hash the URL for cache key
    audio_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    job_id = audio_hash

    # Check cache
    cached = _check_cache(job_id, stem_list, use_model)
    if cached:
        return {"job_id": job_id, "stems": cached, "cached": True}

    # Download the file first
    import urllib.request
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    try:
        urllib.request.urlretrieve(url, tmp.name)
    except Exception as e:
        os.unlink(tmp.name)
        return JSONResponse({"error": f"Failed to download audio: {e}"}, 400)

    result = _enqueue_job(job_id, tmp.name, stem_list, use_model)
    if result.get("error"):
        return JSONResponse(result, 503)
    return result


# ── WhisperX forced alignment ──────────────────────────────────────────

# Lazy-loaded WhisperX models (shared across requests). WhisperX runs
# faster-whisper for transcription, then a wav2vec2 forced aligner for
# tighter word/character boundaries than stable-ts produced. The aligner
# is language-specific so we cache one per language.
_whisperx_model = None
_whisperx_model_lock = threading.Lock()
_whisperx_model_name = "medium"
# OrderedDict for LRU eviction semantics — a multilingual server
# accumulating dozens of ~1 GB wav2vec2 aligners would OOM. Bounded
# by MAX_WHISPERX_ALIGNERS; the least-recently-used aligner is evicted
# when the cap is exceeded.
MAX_WHISPERX_ALIGNERS = max(1, int(os.environ.get("SLOPSMITH_MAX_WHISPERX_ALIGNERS", "4")))
_whisperx_aligners: OrderedDict[str, tuple] = OrderedDict()
_whisperx_aligners_lock = threading.Lock()
# Per-language locks so a /align call for language A doesn't block a
# concurrent /align for language B, while two concurrent calls for the
# SAME language serialise on a single download/load. Without this,
# the previous "release the cache lock during load_align_model" pattern
# allowed both threads to slip past the cache miss and double-download
# the same wav2vec2 weights (latency spike, possible GPU OOM).
_whisperx_aligner_locks: dict[str, threading.Lock] = {}
_whisperx_aligner_locks_guard = threading.Lock()


def _get_aligner_load_lock(lang: str) -> threading.Lock:
    """Return the lock that serialises load_align_model calls for one
    language. Created lazily; never removed (per-language locks are
    cheap and the language set is bounded)."""
    with _whisperx_aligner_locks_guard:
        lock = _whisperx_aligner_locks.get(lang)
        if lock is None:
            lock = threading.Lock()
            _whisperx_aligner_locks[lang] = lock
    return lock


def _whisperx_device() -> str:
    return _device or ("cuda" if _gpu_available else "cpu")


def _whisperx_compute_type() -> str:
    # faster-whisper / CTranslate2 picks compute_type per-device. CUDA
    # benefits from float16; CPU only supports int8/float32 reliably.
    # Key off the effective runtime device (which may be forced to "cpu"
    # via --device on a CUDA-capable host) — keying off _gpu_available
    # would pick float16 on CPU and crash faster-whisper at load time.
    return "float16" if _whisperx_device() == "cuda" else "int8"


def _mark_lazy_loaded(name: str) -> None:
    """Update warmup_state to ``ready`` after a successful lazy load.
    Called from the lazy-load helpers so /health reflects truth even
    when the user passed --skip-warmup or warmup itself failed and
    the endpoint subsequently lazy-loaded its model on first use.

    Skip the transition while warmup is mid-flight (state ==
    "downloading") — the warmup thread loads sub-components in order
    (e.g. whisperx model first, then the aligner), and a premature
    "ready" here would make /health claim readiness before the
    aligner finishes downloading. The warmup thread sets the final
    "ready" state itself."""
    with warmup_state_lock:
        current = warmup_state.get(name)
    if current in ("ready", "downloading"):
        return
    _set_warmup_state(name, "ready")


def _get_whisperx_model():
    global _whisperx_model
    if _whisperx_model is None:
        with _whisperx_model_lock:
            if _whisperx_model is None:
                _whisperx_model = whisperx.load_model(
                    _whisperx_model_name,
                    device=_whisperx_device(),
                    compute_type=_whisperx_compute_type(),
                )
    # Intentionally do NOT mark warmup ready here. /align needs both the
    # ASR model AND a wav2vec2 aligner to function — if the aligner load
    # fails (unsupported language, transient network), /align would still
    # 500 while /health.warmup.whisperx claimed "ready". The aligner
    # helper marks ready only after a successful aligner load, which is
    # the true gate for subsystem readiness.
    return _whisperx_model


def _get_whisperx_aligner(language: str):
    """Load (or fetch from cache) the wav2vec2 aligner for a language.
    Returns ``(aligner_model, metadata)`` per the whisperx contract.

    Per-language locking ensures two concurrent /align calls for the
    same fresh language serialise on one download (no double-instantiate
    of the wav2vec2 weights, no GPU OOM spike), while concurrent calls
    for *different* languages still parallelise.
    """
    import re
    lang = (language or "en").lower()

    # Validate the language code before creating any per-language state.
    # Arbitrary strings would cause unbounded growth in
    # _whisperx_aligner_locks / warmup_aligners — reject codes that
    # aren't simple ISO 639-1/2 tags (2–8 lowercase ASCII alpha).
    # whisperx.load_align_model() accepts only these short codes; full
    # BCP-47 tags with hyphens (e.g. 'zh-Hans-CN') are not supported.
    if not re.fullmatch(r'[a-z]{2,8}', lang):
        raise ValueError(f"Invalid language code: {lang!r}")

    # Fast path: already cached.
    with _whisperx_aligners_lock:
        cached = _whisperx_aligners.get(lang)
        if cached is not None:
            # LRU touch — move to end so eviction picks the
            # least-recently-used language first.
            _whisperx_aligners.move_to_end(lang)
    if cached is not None:
        # Top-level "whisperx" warmup contract is "ASR + en aligner".
        # Lazy-loading a non-English aligner shouldn't satisfy it; only
        # an "en" load promotes the top-level state.
        if lang == "en":
            _mark_lazy_loaded("whisperx")
        return cached

    # Slow path: serialise on the per-language lock so only one thread
    # actually runs load_align_model for a given language at a time.
    load_lock = _get_aligner_load_lock(lang)
    with load_lock:
        # Re-check under the load lock — a sibling thread may have
        # populated the cache while we were waiting.
        with _whisperx_aligners_lock:
            cached = _whisperx_aligners.get(lang)
            if cached is not None:
                _whisperx_aligners.move_to_end(lang)
        if cached is not None:
            if lang == "en":
                _mark_lazy_loaded("whisperx")
            return cached
        # Surface per-language download progress on /health so non-
        # English /align callers can poll for their language's
        # readiness instead of waiting blind for the first request to
        # stall on a CDN fetch.
        _set_aligner_state(lang, "downloading")
        try:
            pair = whisperx.load_align_model(
                language_code=lang,
                device=_whisperx_device(),
            )
        except Exception as exc:  # noqa: BLE001
            _set_aligner_state(lang, f"failed: {exc}")
            raise
        evicted_langs: list[str] = []
        with _whisperx_aligners_lock:
            _whisperx_aligners[lang] = pair
            _whisperx_aligners.move_to_end(lang)
            # LRU evict to bound RAM/VRAM. Each wav2vec2 aligner is
            # ~1 GB; without a cap a multilingual server accumulates
            # forever and eventually OOMs /align or /separate.
            while len(_whisperx_aligners) > MAX_WHISPERX_ALIGNERS:
                old_lang, _old_pair = _whisperx_aligners.popitem(last=False)
                evicted_langs.append(old_lang)
        _set_aligner_state(lang, "ready")
        for el in evicted_langs:
            _set_aligner_state(el, "evicted")
            # Top-level whisperx state tracks "ASR + en aligner". If
            # the en aligner is evicted, that contract no longer holds
            # — clients polling /health.warmup.whisperx must see this
            # to avoid blasting requests at a server that'll re-stall
            # on en's CDN fetch on next /align.
            if el == "en":
                _set_warmup_state("whisperx", "evicted")
        # Best-effort GPU memory release after eviction. Safe to call
        # on CPU (no-op).
        if evicted_langs:
            try:
                if _whisperx_device() == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass

    if lang == "en":
        _mark_lazy_loaded("whisperx")
    return pair


def _get_hyphenator(lang_code: str):
    """Get a pyphen hyphenator for the given language, with fallback to English."""
    import pyphen
    # Map common Whisper language codes to pyphen locale codes
    lang_map = {
        "en": "en_US", "es": "es_ES", "fr": "fr_FR", "de": "de_DE",
        "it": "it_IT", "pt": "pt_PT", "nl": "nl_NL", "ru": "ru_RU",
        "ja": "en_US", "ko": "en_US", "zh": "en_US",  # CJK: no hyphenation, 1 char = 1 syllable
        "sv": "sv_SE", "da": "da_DK", "nb": "nb_NO", "fi": "fi_FI",
        "pl": "pl_PL", "cs": "cs_CZ", "hu": "hu_HU", "ro": "ro_RO",
    }
    locale = lang_map.get(lang_code, "")
    if not locale:
        # Try constructing a locale from the code
        locale = f"{lang_code}_{lang_code.upper()}" if lang_code else "en_US"
    try:
        return pyphen.Pyphen(lang=locale)
    except Exception:
        return pyphen.Pyphen(lang="en_US")


def _syllabify(word: str, hyphenator) -> list[str]:
    """Split a word into syllables using hyphenation. Falls back to the whole word."""
    if not word:
        return [word]
    # For CJK characters, each character is roughly one syllable
    if any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff'
           or '\uac00' <= c <= '\ud7af' for c in word):
        return list(word)
    parts = hyphenator.inserted(word).split('-')
    return parts if parts else [word]


def _split_word_into_syllables(word_seg: dict, hyphenator) -> list[dict]:
    """Split a word segment into syllable segments with proportional timing."""
    syllables = _syllabify(word_seg["text"], hyphenator)
    if len(syllables) <= 1:
        return [word_seg]
    total_chars = sum(len(s) for s in syllables)
    if total_chars == 0:
        return [word_seg]
    duration = word_seg["end"] - word_seg["start"]
    result = []
    t = word_seg["start"]
    for syl in syllables:
        s_dur = duration * (len(syl) / total_chars)
        result.append({
            "start": round(t, 3),
            "end": round(t + s_dur, 3),
            "text": syl,
        })
        t += s_dur
    return result


@app.post("/align")
async def align_lyrics(
    file: UploadFile = File(...),
    text: str = Form(...),
    language: str = Form(""),
    granularity: str = Form("line"),
):
    """Forced-align plain text lyrics against an audio file using WhisperX.

    Granularities:
        line     — segment-level boundaries (default).
        word     — per-word timestamps from the wav2vec2 aligner.
        syllable — words split via pyphen hyphenation.
        phoneme  — per-character (CTC token) timestamps from the aligner.
                   For wav2vec2 character models these are tighter than
                   syllables; for phoneme-trained models they're true
                   phonemes. Both shapes are returned with a
                   ``phoneme: true`` flag so clients can disambiguate.

    Returns ``{"segments": [{start, end, text, ...}, ...]}``.
    """
    # Save upload to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "audio.ogg").suffix)
    content = await file.read()
    tmp.write(content)
    tmp.close()

    want_word = granularity in ("word", "syllable")
    want_phoneme = granularity == "phoneme"

    def _do_align():
        try:
            clean_text = (text or "").strip()
            if not clean_text:
                return {"error": "lyrics text is empty", "_http_status": 400}

            # WhisperX expects a numpy float32 mono 16k array for both
            # transcription and alignment. load_audio handles the
            # resample / mono conversion identically to its internals.
            audio = whisperx.load_audio(tmp.name)
            audio_duration = float(len(audio)) / 16000.0
            if audio_duration <= 0:
                return {"error": "audio is empty", "_http_status": 400}

            lines = [ln.strip() for ln in clean_text.splitlines() if ln.strip()]
            if not lines:
                return {"error": "lyrics text is empty", "_http_status": 400}

            # Build a flat word list with a parallel word→line index
            # array so we can recover new_line markers after alignment.
            flat_words: list[str] = []
            word_to_line: list[int] = []
            for line_idx, ln in enumerate(lines):
                for w in ln.split():
                    flat_words.append(w)
                    word_to_line.append(line_idx)
            if not flat_words:
                return {"error": "lyrics text contains no words", "_http_status": 400}

            # Run Whisper transcription. We use it for two purposes:
            #   1. Language detection (free byproduct of the call).
            #   2. VAD-anchored alignment windows — Whisper skips long
            #      instrumental sections, so the segments it returns
            #      correspond to actual sung audio. The transcribed
            #      text is intentionally discarded (forced alignment
            #      uses caller-supplied lyrics); we keep only the
            #      time boundaries.
            #
            # Why not single-segment global align? wav2vec2 forced
            # alignment scales with input length; multi-minute songs
            # work fine but very long inputs can OOM. Why not per-line
            # proportional windows? Long intros / instrumental breaks
            # can land a line's window over silence. VAD chunking
            # threads both needles: each window is anchored to actual
            # speech (no silence misalignment) and short (no OOM).
            asr_model = _get_whisperx_model()
            transcribe_kwargs: dict = {"batch_size": 16}
            if language:
                transcribe_kwargs["language"] = language.lower()
            transcribed = asr_model.transcribe(audio, **transcribe_kwargs)
            # Caller's explicit hint takes precedence — Whisper's auto-
            # detection can mis-classify on short clips, instrumental
            # intros, or non-English vocals, which would then load the
            # wrong wav2vec2 aligner.
            detected_lang = (language or transcribed.get("language") or "en").lower()
            raw_speech_segments = transcribed.get("segments", []) or []

            # Drop tiny / zero-duration segments — wav2vec2 align tends
            # to produce empty alignments for sub-frame windows.
            speech_segments = [
                s for s in raw_speech_segments
                if float(s.get("end", 0.0)) - float(s.get("start", 0.0)) > 0.2
            ]

            # Distribute user-text words across speech segments
            # proportionally to each segment's duration.
            custom_segments: list[dict] = []

            if not speech_segments:
                # Fallback for very short audio or VAD-empty results:
                # one segment covering the whole audio. wav2vec2 will
                # align everything in one pass — fine for short clips.
                custom_segments.append({
                    "start": 0.0,
                    "end": float(audio_duration),
                    "text": " ".join(flat_words),
                })
            else:
                # Distribute user-text words across speech segments by
                # local tempo. Whisper's per-segment transcribed word
                # count is the best proxy when available — a long held
                # note shows up as one transcribed word, a fast run
                # shows up as many. But Whisper sometimes returns
                # segments that have timing without transcribed text
                # (noisy stems, humming, unintelligible vocals); those
                # segments still contain singing, so user lyrics need
                # to land there too. Building a hybrid per-segment
                # weight: transcribed word count when present, else
                # duration × an assumed tempo of ~3 words/sec so the
                # magnitudes are comparable to the word-count weights.
                #
                # This avoids the round-8 finding's failure mode: an
                # all-zero-words segment in the middle of a song
                # causing all subsequent user text to shift into later
                # segments.
                ASSUMED_WPS = 3.0  # rough singing tempo for fallback
                weights: list[float] = []
                for s in speech_segments:
                    n = len((s.get("text") or "").split())
                    if n > 0:
                        weights.append(float(n))
                    else:
                        seg_dur = float(s["end"]) - float(s["start"])
                        weights.append(max(0.1, seg_dur * ASSUMED_WPS))

                total_w = sum(weights) or 1.0
                cumulative: list[int] = []
                running = 0.0
                for w in weights:
                    running += (w / total_w) * len(flat_words)
                    cumulative.append(int(round(running)))
                # Pin the last entry to len(flat_words) so rounding
                # error doesn't drop trailing words.
                cumulative[-1] = len(flat_words)

                cursor = 0
                for _, (s, end_word_idx) in enumerate(zip(speech_segments, cumulative)):
                    end_word_idx = max(end_word_idx, cursor)
                    if end_word_idx <= cursor:
                        # No words allocated to this speech segment —
                        # skip it. Common for very short utterances.
                        continue
                    chunk_words = flat_words[cursor:end_word_idx]
                    custom_segments.append({
                        "start": round(float(s["start"]), 3),
                        "end": round(float(s["end"]), 3),
                        "text": " ".join(chunk_words),
                    })
                    cursor = end_word_idx

                # Trailing words (rounding can leave a few unassigned)
                # → fold into the last segment.
                if cursor < len(flat_words) and custom_segments:
                    trailing = flat_words[cursor:]
                    custom_segments[-1]["text"] += " " + " ".join(trailing)

            if not custom_segments:
                return {"error": "no speech segments found to align against", "_http_status": 400}

            aligner_model, aligner_meta = _get_whisperx_aligner(detected_lang)
            aligned = whisperx.align(
                custom_segments,
                aligner_model,
                aligner_meta,
                audio,
                _whisperx_device(),
                return_char_alignments=want_phoneme,
            )

            segments_out: list[dict] = []
            aligned_segments = aligned.get("segments", [])

            if want_phoneme:
                # Flatten char alignments. WhisperX puts them under each
                # segment as `chars: [{char, start, end, score}, ...]`.
                # `start`/`end` may be missing on whitespace tokens.
                for seg in aligned_segments:
                    for ch in seg.get("chars", []) or []:
                        cs = ch.get("start")
                        ce = ch.get("end")
                        ct = ch.get("char", "")
                        if cs is None or ce is None or not ct.strip():
                            continue
                        segments_out.append({
                            "start": round(float(cs), 3),
                            "end": round(float(ce), 3),
                            "text": ct,
                            "phoneme": True,
                        })
            elif want_word:
                # Single-segment alignment, so collect aligned words
                # across whatever segments WhisperX produced internally
                # (it may chunk long inputs). Match each aligned word
                # back to its source line via the parallel word_to_line
                # array so we can emit new_line markers at line
                # boundaries.
                aligned_words: list[dict] = []
                for seg in aligned_segments:
                    for w in seg.get("words", []) or []:
                        if w.get("start") is None or w.get("end") is None:
                            continue
                        wt = (w.get("word") or "").strip()
                        if not wt:
                            continue
                        aligned_words.append(w)

                # If the aligned word count matches the input flat_words
                # count we can map positionally with confidence. If they
                # diverge (whisperx may merge/split occasionally), fall
                # back to a textual walk: advance the input cursor each
                # time the output text matches the next input word.
                # If even that fails, omit new_line markers rather than
                # placing them incorrectly.
                line_for_aligned: list[int | None] = [None] * len(aligned_words)
                if len(aligned_words) == len(word_to_line):
                    line_for_aligned = list(word_to_line)
                else:
                    cursor_in = 0
                    for i, w in enumerate(aligned_words):
                        if cursor_in >= len(flat_words):
                            break
                        wt = (w.get("word") or "").strip().lower()
                        target = flat_words[cursor_in].lower()
                        # Strip trailing punctuation on either side so
                        # "love," and "love" still match.
                        wt_norm = wt.rstrip(".,!?;:\"'-")
                        target_norm = target.rstrip(".,!?;:\"'-")
                        if wt_norm == target_norm or target_norm.startswith(wt_norm) or wt_norm.startswith(target_norm):
                            line_for_aligned[i] = word_to_line[cursor_in]
                            cursor_in += 1

                prev_line = -1
                for i, w in enumerate(aligned_words):
                    entry = {
                        "start": round(float(w["start"]), 3),
                        "end": round(float(w["end"]), 3),
                        "text": (w.get("word") or "").strip(),
                    }
                    line_idx = line_for_aligned[i]
                    if line_idx is not None and line_idx != prev_line:
                        entry["new_line"] = True
                        prev_line = line_idx
                    segments_out.append(entry)

                if granularity == "syllable":
                    lang_code = detected_lang
                    hyphenator = _get_hyphenator(lang_code)
                    syllable_segs = []
                    for ws in segments_out:
                        syls = _split_word_into_syllables(ws, hyphenator)
                        if ws.get("new_line") and syls:
                            syls[0]["new_line"] = True
                        syllable_segs.extend(syls)
                    segments_out = syllable_segs
            else:
                # Line granularity: aggregate aligned words back into
                # their source user-text lines so each output segment
                # is one input line with refined start/end timestamps.
                aligned_words = []
                for seg in aligned_segments:
                    for w in seg.get("words", []) or []:
                        if w.get("start") is None or w.get("end") is None:
                            continue
                        if not (w.get("word") or "").strip():
                            continue
                        aligned_words.append(w)

                # Same positional vs textual fallback as the word path.
                line_for_aligned = [None] * len(aligned_words)
                if len(aligned_words) == len(word_to_line):
                    line_for_aligned = list(word_to_line)
                else:
                    cursor_in = 0
                    for i, w in enumerate(aligned_words):
                        if cursor_in >= len(flat_words):
                            break
                        wt = (w.get("word") or "").strip().lower().rstrip(".,!?;:\"'-")
                        target = flat_words[cursor_in].lower().rstrip(".,!?;:\"'-")
                        if wt == target or target.startswith(wt) or wt.startswith(target):
                            line_for_aligned[i] = word_to_line[cursor_in]
                            cursor_in += 1

                line_buckets: dict[int, list] = {}
                for i, w in enumerate(aligned_words):
                    li = line_for_aligned[i]
                    if li is None:
                        continue
                    line_buckets.setdefault(li, []).append(w)

                # First pass: collect (start, end) for lines that have
                # aligned words. Then fill in gaps with neighbour-based
                # estimates so EVERY input line produces an output
                # segment. Dropping lines silently breaks index-based
                # clients (line N in the response no longer matches
                # line N in the input) and de-syncs the rest of the
                # song display. A best-effort estimate is strictly
                # better than a missing line.
                line_times: list[tuple[float, float] | None] = [None] * len(lines)
                for line_idx in range(len(lines)):
                    bucket = line_buckets.get(line_idx, [])
                    if bucket:
                        line_times[line_idx] = (
                            float(bucket[0]["start"]),
                            float(bucket[-1]["end"]),
                        )

                # If no line got mapped via word_to_line but we DID get
                # aligned words from wav2vec2, the failure is line-
                # mapping (tokenization mismatch — CJK/Thai without
                # spaces, contractions split or merged differently
                # than ln.split() produces). Fall back to distributing
                # the aligned-word timestamps across user lines by
                # word-count ratio so we still produce a usable chart.
                # Only error out when wav2vec2 itself returned no
                # aligned words at all (genuinely unreadable / silent
                # / language-mismatched).
                if not any(lt is not None for lt in line_times):
                    if not aligned_words:
                        return {
                            "error": (
                                "wav2vec2 alignment produced no word "
                                "timestamps — vocals stem may be silent, "
                                "language mismatched, or audio unreadable"
                            ),
                            "_http_status": 400,
                        }
                    # Word-mapping fell through; spread the available
                    # aligned word timestamps across user lines by
                    # index ratio. Each line i gets words in the slice
                    # [i * N / L, (i+1) * N / L) where N is aligned
                    # word count and L is line count.
                    n_aw = len(aligned_words)
                    n_lines = len(lines)
                    for li in range(n_lines):
                        i0 = (li * n_aw) // n_lines
                        i1 = ((li + 1) * n_aw) // n_lines
                        if i1 <= i0:
                            continue
                        bucket_words = aligned_words[i0:i1]
                        line_times[li] = (
                            float(bucket_words[0]["start"]),
                            float(bucket_words[-1]["end"]),
                        )

                # Fill missing lines by interpolating between known
                # neighbours. Runs of consecutive missing lines split
                # the gap evenly so timestamps stay monotonic — a naive
                # per-line walk would assign each missing line the
                # FULL prev_end..next_start gap, producing overlapping
                # or out-of-order entries when two or more adjacent
                # lines failed to align.
                i = 0
                while i < len(lines):
                    if line_times[i] is not None:
                        i += 1
                        continue
                    # Find the run of consecutive missing lines [i, j).
                    j = i
                    while j < len(lines) and line_times[j] is None:
                        j += 1
                    # Anchor times: last real timing before i, first
                    # real timing at-or-after j.
                    prev_end = 0.0
                    for k in range(i - 1, -1, -1):
                        if line_times[k] is not None:
                            prev_end = line_times[k][1]
                            break
                    next_start = audio_duration
                    for k in range(j, len(lines)):
                        if line_times[k] is not None:
                            next_start = line_times[k][0]
                            break
                    n_missing = j - i
                    # Guarantee at least a small forward slot per line
                    # so timestamps stay monotonic even if anchors are
                    # collapsed (e.g. all lines missing → fall back to
                    # 0 → audio_duration shared).
                    if next_start <= prev_end:
                        next_start = min(audio_duration, prev_end + 0.5 * n_missing)
                    slice_dur = (next_start - prev_end) / n_missing
                    for k in range(i, j):
                        slot_start = prev_end + slice_dur * (k - i)
                        slot_end = prev_end + slice_dur * (k - i + 1)
                        line_times[k] = (
                            slot_start,
                            min(slot_end, audio_duration),
                        )
                    i = j

                for line_idx, ln in enumerate(lines):
                    seg_start, seg_end = line_times[line_idx]  # type: ignore[misc]
                    segments_out.append({
                        "start": round(seg_start, 3),
                        "end": round(seg_end, 3),
                        "text": ln,
                    })

            return {"segments": segments_out, "language": detected_lang}
        except ValueError as e:
            # ValueError signals client input problems (e.g. invalid
            # language code) — expose as 400 so clients can distinguish
            # their own mistakes from server faults.
            return {"error": str(e), "_http_status": 400}
        except Exception as e:
            return {"error": str(e)}
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    import asyncio
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _do_align)

    if "error" in result:
        status = result.pop("_http_status", 500)
        return JSONResponse(result, status)
    return result


# ── Per-syllable pitch extraction (CREPE) ───────────────────────────────
#
# /pitch returns one MIDI note per supplied syllable timing. The
# extractor runs CREPE (a neural pitch tracker) over the vocals stem,
# then applies four quality steps that meaningfully improve the
# resulting karaoke chart:
#
#   1. CREPE itself — vastly fewer octave errors than pYIN, with a
#      per-frame confidence that we use as the aggregation weight.
#   2. Confidence-weighted mode-of-semitone per syllable. For each
#      token, we round each frame's f0 to the nearest semitone and
#      pick the semitone with the highest summed confidence. More
#      robust than median-Hz to a few wrong frames mid-syllable.
#   3. Range narrowing. Compute the song-wide median midi from
#      confident frames and, for each syllable, discard frames that
#      fall outside ±12 semitones of that median — but only when
#      ≥50 % of the syllable's confidence weight is already in range
#      (so a legitimate high/low note is never discarded). Catches
#      the long-tail of octave doublings that CREPE still
#      occasionally produces on breathy / quiet notes.
#   4. Octave-error correction against the song-wide median. If
#      shifting a token by ±12 brings its midi closer to the song's
#      median than the raw value, prefer the shifted one.
#
# Tokens with no confident frames inside their window get their midi
# borrowed from the nearest confident neighbour so whole phrases that
# CREPE can't lock (whispered / spoken bridges) still produce bars.

def _crepe_device() -> str:
    """Use the same device choice as demucs/whisper so a CUDA box runs
    everything on the GPU and a CPU box stays on the CPU."""
    if _device:
        return _device
    if _gpu_available:
        return "cuda"
    return "cpu"


def _extract_pitch_with_crepe(audio_path: Path, lyrics: list[dict]) -> list[dict]:
    """Run CREPE on the vocals stem and return one ``{t, d, midi}`` per
    token. See module-level comment above for the four quality steps.
    """
    import numpy as np

    sr = 16000  # CREPE is trained at 16 kHz
    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    if y.size == 0:
        return []

    audio = torch.from_numpy(y).unsqueeze(0).float()
    hop_length = 160  # 10 ms frames at 16 kHz — matches CREPE's design
    fmin = float(librosa.note_to_hz("C2"))
    fmax = float(librosa.note_to_hz("C6"))
    device = _crepe_device()

    # CREPE's `full` model is the most accurate; `tiny` is the fastest.
    # `full` is fine on CPU at 16 kHz for ~5 min songs (~1-2× realtime).
    f0, periodicity = torchcrepe.predict(
        audio,
        sr,
        hop_length,
        fmin,
        fmax,
        model="full",
        batch_size=2048,
        device=device,
        return_periodicity=True,
    )
    # First successful run implies the CREPE weights are loaded and
    # the model is functional — flip /health.warmup.crepe to ready
    # for the lazy-load (--skip-warmup or post-failure) path.
    _mark_lazy_loaded("crepe")
    # torchcrepe applies a Viterbi-like decoder when batched; periodicity
    # serves the role pYIN's voiced_prob played in tier 1 — a 0..1
    # confidence per frame.
    f0_np = f0.squeeze(0).cpu().numpy().astype(float)
    conf_np = periodicity.squeeze(0).cpu().numpy().astype(float)

    # CREPE returns 0 Hz where it failed to estimate; mask those out so
    # log2 doesn't see them.
    valid = (f0_np > 0) & np.isfinite(f0_np)
    times = np.arange(f0_np.size) * (hop_length / sr)
    n_frames = len(times)

    # Pre-compute song-wide median midi from confident frames so range
    # narrowing has a stable reference.
    confident_mask = valid & (conf_np > 0.5)
    if int(confident_mask.sum()) >= 32:
        midis_all = 69 + 12 * np.log2(f0_np[confident_mask] / 440.0)
        song_median = float(np.median(midis_all))
        clamp_low = song_median - 12
        clamp_high = song_median + 12
    else:
        song_median = None
        clamp_low = clamp_high = None

    raw: list[dict] = []
    for tok in lyrics:
        t0 = float(tok["t"])
        t1 = t0 + float(tok["d"])
        i0 = int(np.searchsorted(times, t0, side="left"))
        i1 = int(np.searchsorted(times, t1, side="right"))
        midi: int | None = None
        if i1 > i0 and i0 < n_frames:
            seg_hz = f0_np[i0:i1]
            seg_w = conf_np[i0:i1]
            mask = (seg_hz > 0) & np.isfinite(seg_hz) & (seg_w > 0.2)
            if mask.any():
                hz = seg_hz[mask]
                w = seg_w[mask]
                semitones = np.rint(69 + 12 * np.log2(hz / 440.0)).astype(int)

                # Range narrowing: drop frames outside ±12 semitones of
                # the song median ONLY when the in-range frames carry
                # enough of the syllable's confidence weight that they
                # represent the syllable's actual pitch. A syllable
                # hitting a legitimate high/low note will have most of
                # its weight outside the clamp range — narrowing then
                # would snap the bar onto a few noisy in-range frames
                # instead of the real note. Require ≥50% of the
                # syllable's voicing confidence to be in-range before
                # discarding the rest.
                if clamp_low is not None and clamp_high is not None:
                    in_range = (semitones >= clamp_low) & (semitones <= clamp_high)
                    total_w = float(w.sum())
                    in_range_w = float(w[in_range].sum()) if in_range.any() else 0.0
                    if total_w > 0 and in_range_w / total_w >= 0.5:
                        semitones = semitones[in_range]
                        w = w[in_range]

                # Confidence-weighted mode of semitones.
                unique = np.unique(semitones)
                weights = np.array(
                    [float(w[semitones == u].sum()) for u in unique],
                    dtype=float,
                )
                midi = int(unique[int(np.argmax(weights))])
        raw.append({"t": t0, "d": float(tok["d"]), "midi": midi})

    # Octave-error correction against the song-wide median. Even with
    # CREPE's lower error rate, a quiet syllable can land an octave off
    # the surrounding melody — snap if shifting brings it closer.
    if song_median is not None:
        for r in raw:
            if r["midi"] is None:
                continue
            base = int(r["midi"])
            best = base
            best_dist = abs(base - song_median)
            for shift in (-12, 12):
                cand = base + shift
                d = abs(cand - song_median)
                if d < best_dist:
                    best, best_dist = cand, d
            r["midi"] = best

    # Neighbour-borrow for tokens that still have no midi (consonant
    # syllables, whispered phrases CREPE couldn't lock). Skip if the
    # whole song produced nothing — there's nothing to borrow.
    indexed_confident = [(i, r["midi"]) for i, r in enumerate(raw) if r["midi"] is not None]
    if indexed_confident:
        for i, r in enumerate(raw):
            if r["midi"] is not None:
                continue
            nearest = min(indexed_confident, key=lambda c: abs(c[0] - i))
            r["midi"] = nearest[1]

    return [r for r in raw if r["midi"] is not None]


@app.post("/pitch")
async def pitch_extract(
    file: UploadFile = File(...),
    lyrics: str = Form(...),
):
    """Run CREPE on a vocals stem and return one MIDI note per syllable.

    Body:
      - ``file``    — vocals audio (any format librosa can read)
      - ``lyrics``  — JSON array of ``{"t": float, "d": float}`` token
                      timings (start / duration in seconds)

    Returns ``{"notes": [{"t", "d", "midi"}, ...]}``. Tokens for which
    no pitch could be estimated (even after neighbour-borrow) are
    omitted from the output.
    """
    import json

    try:
        token_list = json.loads(lyrics)
        if not isinstance(token_list, list):
            raise ValueError("lyrics must be a JSON array")
        for entry in token_list:
            if not isinstance(entry, dict) or "t" not in entry or "d" not in entry:
                raise ValueError("each lyric entry needs 't' and 'd'")
            # Numeric type-check at the boundary so a malformed payload
            # (e.g. {"t": "abc"}) returns a clean 400 instead of a 500
            # from the worker thread when float() blows up.
            try:
                float(entry["t"])
                float(entry["d"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"'t' and 'd' must be numeric ({exc})")
    except (json.JSONDecodeError, ValueError) as exc:
        return JSONResponse({"error": f"invalid lyrics payload: {exc}"}, 400)

    if not token_list:
        return {"notes": []}

    suffix = Path(file.filename or "audio.ogg").suffix or ".ogg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = await file.read()
    tmp.write(content)
    tmp.close()

    def _do_extract():
        try:
            return {"notes": _extract_pitch_with_crepe(Path(tmp.name), token_list)}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    import asyncio
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _do_extract)
    if "error" in result:
        return JSONResponse(result, 500)
    return result


# ── Download stems ──────────────────────────────────────────────────────

@app.get("/download/{job_id}/{stem}")
def download_stem(job_id: str, stem: str):
    # stem can be "drums.mp3", "drums.wav", or just "drums"
    stem_name = Path(stem).stem

    # Try multiple extensions
    for ext in (".mp3", ".wav", ".flac"):
        path = CACHE_DIR / job_id / f"{stem_name}{ext}"
        if path.exists():
            media = {"mp3": "audio/mpeg", "wav": "audio/wav", "flac": "audio/flac"}
            return FileResponse(str(path), media_type=media.get(ext[1:], "application/octet-stream"))

    return JSONResponse({"error": "Stem not found"}, 404)


# ── Jobs list ───────────────────────────────────────────────────────────

@app.get("/jobs")
def list_jobs():
    with jobs_lock:
        return list(jobs.values())[-50:]  # last 50


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, 404)
    return job


# ── Cache management ────────────────────────────────────────────────────

@app.delete("/cache/{job_id}")
def delete_cache(job_id: str):
    cache_path = CACHE_DIR / job_id
    if cache_path.exists():
        shutil.rmtree(cache_path, ignore_errors=True)
    with jobs_lock:
        jobs.pop(job_id, None)
    return {"ok": True}


# ── WebSocket for job progress ──────────────────────────────────────────

@app.websocket("/ws/jobs/{job_id}")
async def ws_job_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()
    if job_id not in ws_subscribers:
        ws_subscribers[job_id] = set()
    ws_subscribers[job_id].add(websocket)
    try:
        # Send current state immediately
        with jobs_lock:
            job = jobs.get(job_id)
        if job:
            await websocket.send_json(job)
        # Keep connection open until client disconnects
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        ws_subscribers.get(job_id, set()).discard(websocket)


# ── Internal helpers ────────────────────────────────────────────────────

def _check_cache(job_id, stem_list, model):
    """Return stem download URLs if all requested stems are cached."""
    cache_path = CACHE_DIR / job_id
    if not cache_path.exists():
        return None

    stems_found = {}
    for stem_name in stem_list:
        for ext in (".mp3", ".wav", ".flac"):
            p = cache_path / f"{stem_name}{ext}"
            if p.exists():
                stems_found[stem_name] = f"/download/{job_id}/{stem_name}{ext}"
                break

    if len(stems_found) == len(stem_list):
        return stems_found
    return None


def _enqueue_job(job_id, audio_path, stem_list, model):
    """Create a job and start processing in background."""
    global active_count

    with jobs_lock:
        # If job already exists and is processing/complete, return it
        existing = jobs.get(job_id)
        if existing and existing["status"] in ("processing", "complete"):
            if existing["status"] == "complete":
                return {"job_id": job_id, "stems": existing["stems"], "cached": True}
            return {"job_id": job_id, "status": "processing"}

    with active_lock:
        if active_count >= MAX_CONCURRENT:
            return {"error": "Server busy — max concurrent separations reached", "job_id": job_id}

    job = {
        "job_id": job_id,
        "status": "processing",
        "progress": 0,
        "stems": {},
        "error": None,
        "model": model,
        "created_at": time.time(),
    }
    with jobs_lock:
        jobs[job_id] = job
        # Trim old jobs
        while len(jobs) > 200:
            jobs.popitem(last=False)

    thread = threading.Thread(
        target=_run_demucs,
        args=(job_id, audio_path, stem_list, model),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "processing"}


def _run_demucs(job_id, audio_path, stem_list, model):
    """Run demucs separation in a background thread."""
    global active_count

    with active_lock:
        active_count += 1

    tmp_out = tempfile.mkdtemp(prefix="demucs_out_")
    try:
        _update_job(job_id, status="processing", progress=10)

        # Build demucs command
        run_demucs = str(Path(__file__).parent / "run_demucs.py")
        cmd = [sys.executable, run_demucs, "--shifts", "2"]
        if model:
            cmd.extend(["-n", model])
        if _device:
            cmd.extend(["-d", _device])
        cmd.extend(["-o", tmp_out, audio_path])

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

        # Read stderr for progress (demucs outputs progress there)
        _update_job(job_id, progress=20)
        _, stderr = proc.communicate(timeout=600)

        if proc.returncode != 0:
            # Strip tqdm progress bars — real errors are at the end
            err_lines = [l for l in stderr.splitlines() if l and '%|' not in l and 'B/s]' not in l]
            err_msg = '\n'.join(err_lines[-20:]) if err_lines else stderr[-1000:]
            _update_job(job_id, status="failed", error=err_msg[:1000])
            return

        # First successful demucs run implies the model weights are
        # loaded — flip /health.warmup.demucs to ready for the
        # lazy-load (--skip-warmup or post-warmup-failure) path.
        _mark_lazy_loaded("demucs")
        _update_job(job_id, progress=80)

        # Find output stems
        # Demucs outputs to: {out_dir}/{model}/{track_name}/{stem}.wav
        audio_stem = Path(audio_path).stem
        out_model_dir = Path(tmp_out) / model
        if not out_model_dir.exists():
            # Try finding any model directory
            subdirs = list(Path(tmp_out).iterdir())
            out_model_dir = subdirs[0] if subdirs else Path(tmp_out)

        out_track_dir = out_model_dir / audio_stem
        if not out_track_dir.exists():
            # Try finding any track directory
            subdirs = list(out_model_dir.iterdir())
            out_track_dir = subdirs[0] if subdirs else out_model_dir

        # Copy stems to cache — keep as lossless WAV for quality
        cache_path = CACHE_DIR / job_id
        cache_path.mkdir(parents=True, exist_ok=True)

        stems_result = {}
        for stem_name in stem_list:
            src = out_track_dir / f"{stem_name}.wav"
            if not src.exists():
                continue

            wav_dest = cache_path / f"{stem_name}.wav"
            shutil.copy2(src, wav_dest)
            stems_result[stem_name] = f"/download/{job_id}/{stem_name}.wav"

        _update_job(job_id, status="complete", progress=100, stems=stems_result)

    except subprocess.TimeoutExpired:
        proc.kill()
        _update_job(job_id, status="failed", error="Separation timed out (10 min limit)")
    except Exception as e:
        _update_job(job_id, status="failed", error=str(e))
    finally:
        with active_lock:
            active_count -= 1
        # Cleanup
        shutil.rmtree(tmp_out, ignore_errors=True)
        try:
            os.unlink(audio_path)
        except OSError:
            pass


def _update_job(job_id, **kwargs):
    """Update job state and notify WebSocket subscribers."""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        job.update(kwargs)

    # Notify WebSocket subscribers
    subs = ws_subscribers.get(job_id, set()).copy()
    for ws in subs:
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(ws.send_json(job))
        except Exception:
            ws_subscribers.get(job_id, set()).discard(ws)


# ── GPU detection ───────────────────────────────────────────────────────

def _detect_gpu():
    """Check if CUDA GPU is available."""
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


# ── Model weight warmup ─────────────────────────────────────────────────
#
# On first start the three model families (demucs, whisperx, crepe)
# pull weights from CDNs. This is ~1.5 GB total. Without warmup, the
# first user-facing /separate /align /pitch call hangs on download
# without surfacing progress, the request likely times out, and the
# operator has no idea what's happening.
#
# Warmup runs all three downloads sequentially in a daemon thread that
# is spawned from the FastAPI startup event (after uvicorn has bound
# the port). Each library's own tqdm progress bar is left untouched so
# the operator sees real byte-level progress in the terminal / journal.
# /health additionally reports a per-model state dict so client UIs
# (the lyrics_karaoke plugin, etc.) can poll for "warming up" status
# and surface progress.

def _warmup_demucs() -> None:
    """Pre-download the configured demucs separation model. Invokes
    run_demucs.py with --download-only so the soundfile patching path
    matches the real /separate flow."""
    _set_warmup_state("demucs", "downloading")
    run_demucs = str(Path(__file__).parent / "run_demucs.py")
    cmd = [sys.executable, run_demucs, "--download-only"]
    if _model:
        cmd.extend(["-n", _model])
    # Stream stderr/stdout straight through so demucs' tqdm is visible.
    proc = subprocess.run(cmd)
    if proc.returncode == 0:
        _set_warmup_state("demucs", "ready")
    else:
        _set_warmup_state("demucs", f"failed: exit {proc.returncode}")


def _warmup_whisperx() -> None:
    """Pre-download the WhisperX ASR model and the English aligner.
    Other languages still lazy-load on first /align in that language."""
    _set_warmup_state("whisperx", "downloading")
    try:
        _get_whisperx_model()
        _get_whisperx_aligner("en")
        _set_warmup_state("whisperx", "ready")
    except Exception as exc:  # noqa: BLE001
        _set_warmup_state("whisperx", f"failed: {exc}")


def _warmup_crepe() -> None:
    """Pre-download the CREPE pitch model. torchcrepe.load.model handles
    both the download and putting the network on the chosen device."""
    _set_warmup_state("crepe", "downloading")
    try:
        torchcrepe.load.model(device=_crepe_device(), capacity="full")
        _set_warmup_state("crepe", "ready")
    except Exception as exc:  # noqa: BLE001
        _set_warmup_state("crepe", f"failed: {exc}")


def _run_warmup() -> None:
    """Run all three warmups sequentially. Called from a daemon thread
    after the server binds so /health is queryable while downloads
    progress."""
    print("[warmup] starting model weight prefetch — first run can take ~5 min", flush=True)
    _warmup_demucs()
    _warmup_whisperx()
    _warmup_crepe()
    with warmup_state_lock:
        ready = all(s == "ready" for s in warmup_state.values())
    if ready:
        print("[warmup] all models ready", flush=True)
    else:
        print("[warmup] finished with failures — see /health for per-model state", flush=True)


# ── CLI entry point ─────────────────────────────────────────────────────

def main():
    global _model, _device, _gpu_available, API_KEY

    parser = argparse.ArgumentParser(description="Slopsmith Demucs Separation Service")
    parser.add_argument("--port", type=int, default=7865, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--model", default="", help="Demucs model (htdemucs, mdx_extra)")
    parser.add_argument("--device", default="", help="Device (cpu, cuda)")
    parser.add_argument("--api-key", default="", help="API key for auth")
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip the startup model-weight prefetch. Per-endpoint calls "
        "will lazy-download instead. Useful for restricted CI environments.",
    )
    args = parser.parse_args()

    if args.model:
        _model = args.model
    if args.device:
        _device = args.device
    if args.api_key:
        API_KEY = args.api_key

    _gpu_available = _detect_gpu()
    if not _device:
        _device = "cuda" if _gpu_available else "cpu"

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Slopsmith Demucs Server starting on {args.host}:{args.port}")
    print(f"  Model: {_model}")
    print(f"  Device: {_device} (GPU: {_gpu_available})")
    print(f"  Cache: {CACHE_DIR}")
    if API_KEY:
        print("  API key: enabled")

    # Store the --skip-warmup flag on app.state so the startup hook can
    # read it without a module-level global. The startup hook fires only
    # after uvicorn has bound the port (making /health queryable from the
    # very first moment the warmup thread starts).
    app.state.skip_warmup = args.skip_warmup

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
