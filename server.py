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
warmup_state: dict[str, str] = {
    "demucs": "pending",
    "whisperx": "pending",
    "crepe": "pending",
}
warmup_state_lock = threading.Lock()


def _set_warmup_state(name: str, value: str) -> None:
    with warmup_state_lock:
        warmup_state[name] = value
    # Print on every transition so the systemd journal carries a
    # readable trace alongside the per-library tqdm bars.
    print(f"[warmup] {name}: {value}", flush=True)


# ── Auth middleware ─────────────────────────────────────────────────────

@app.middleware("http")
async def check_api_key(request, call_next):
    if API_KEY and request.url.path not in ("/health", "/docs", "/openapi.json"):
        key = request.headers.get("X-API-Key", request.query_params.get("api_key", ""))
        if key != API_KEY:
            return JSONResponse({"error": "Unauthorized"}, 401)
    return await call_next(request)


# ── Health ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    with warmup_state_lock:
        warmup = dict(warmup_state)
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
_whisperx_aligners: dict[str, tuple] = {}
_whisperx_aligners_lock = threading.Lock()


def _whisperx_device() -> str:
    return _device or ("cuda" if _gpu_available else "cpu")


def _whisperx_compute_type() -> str:
    # faster-whisper / CTranslate2 picks compute_type per-device. CUDA
    # benefits from float16; CPU only supports int8/float32 reliably.
    # Key off the effective runtime device (which may be forced to "cpu"
    # via --device on a CUDA-capable host) — keying off _gpu_available
    # would pick float16 on CPU and crash faster-whisper at load time.
    return "float16" if _whisperx_device() == "cuda" else "int8"


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
    return _whisperx_model


def _get_whisperx_aligner(language: str):
    """Load (or fetch from cache) the wav2vec2 aligner for a language.
    Returns ``(aligner_model, metadata)`` per the whisperx contract."""
    lang = (language or "en").lower()
    with _whisperx_aligners_lock:
        cached = _whisperx_aligners.get(lang)
        if cached is not None:
            return cached
    # load_align_model can be slow on first call (downloads wav2vec2
    # weights). Release the lock during the actual load so a concurrent
    # request for a *different* language doesn't have to wait — the
    # double-check inside the lock prevents a duplicate download for the
    # same language.
    pair = whisperx.load_align_model(
        language_code=lang,
        device=_whisperx_device(),
    )
    with _whisperx_aligners_lock:
        existing = _whisperx_aligners.get(lang)
        if existing is not None:
            return existing
        _whisperx_aligners[lang] = pair
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
                return {"error": "lyrics text is empty"}

            # WhisperX expects a numpy float32 mono 16k array for both
            # transcription and alignment. load_audio handles the
            # resample / mono conversion identically to its internals.
            audio = whisperx.load_audio(tmp.name)
            audio_duration = float(len(audio)) / 16000.0
            if audio_duration <= 0:
                return {"error": "audio is empty"}

            lines = [ln.strip() for ln in clean_text.splitlines() if ln.strip()]
            if not lines:
                return {"error": "lyrics text is empty"}

            # Build a flat word list with a parallel word→line index
            # array so we can recover new_line markers after alignment.
            flat_words: list[str] = []
            word_to_line: list[int] = []
            for line_idx, ln in enumerate(lines):
                for w in ln.split():
                    flat_words.append(w)
                    word_to_line.append(line_idx)
            if not flat_words:
                return {"error": "lyrics text contains no words"}

            # Resolve language — used only to pick the wav2vec2 aligner.
            # Whisper transcription text is intentionally discarded; the
            # contract is forced alignment of caller-supplied text. We
            # sample the middle 60s of the audio rather than the first
            # 30s so songs with long instrumental intros / late-starting
            # vocals still detect reliably.
            detected_lang = (language or "").lower().strip()
            if not detected_lang:
                try:
                    asr_model = _get_whisperx_model()
                    window_s = min(60.0, audio_duration)
                    mid = audio_duration / 2.0
                    start_s = max(0.0, mid - window_s / 2.0)
                    end_s = min(audio_duration, mid + window_s / 2.0)
                    s_idx = int(start_s * 16000)
                    e_idx = int(end_s * 16000)
                    sample = audio[s_idx:e_idx] if e_idx > s_idx else audio
                    detected = asr_model.transcribe(sample, batch_size=16)
                    detected_lang = (detected.get("language") or "en").lower()
                except Exception:
                    detected_lang = "en"

            # Forced alignment with a single segment spanning the whole
            # audio. wav2vec2 Viterbi-decodes the optimal char-to-frame
            # mapping globally, which is robust to long instrumental
            # intros, mid-song breaks, and uneven line pacing — none of
            # those constrain where individual words can land. (Per-line
            # windows would force each line into a hard time slice and
            # break exactly those cases.)
            #
            # new_line markers are recovered after alignment by matching
            # aligned word output to the parallel word_to_line array
            # built above.
            custom_segments = [{
                "start": 0.0,
                "end": float(audio_duration),
                "text": " ".join(flat_words),
            }]

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

                for line_idx, ln in enumerate(lines):
                    bucket = line_buckets.get(line_idx, [])
                    if not bucket:
                        # No words landed in this line — emit nothing
                        # rather than a zero-duration entry.
                        continue
                    seg_start = float(bucket[0]["start"])
                    seg_end = float(bucket[-1]["end"])
                    segments_out.append({
                        "start": round(seg_start, 3),
                        "end": round(seg_end, 3),
                        "text": ln,
                    })

            return {"segments": segments_out, "language": detected_lang}
        except Exception as e:
            return {"error": str(e)}
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    import asyncio
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _do_align)

    if "error" in result:
        return JSONResponse(result, 500)
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
#      confident frames and clamp each per-syllable estimate to ±12
#      semitones around that median (preferring the candidate octave
#      that lies in range). Catches the long-tail of pYIN-style
#      octave doublings that CREPE still occasionally produces on
#      breathy / quiet notes.
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
                # the song median when we have a stable reference. Keep
                # the unclamped version too in case clamping kills every
                # frame (legitimately out-of-range high notes).
                if clamp_low is not None and clamp_high is not None:
                    in_range = (semitones >= clamp_low) & (semitones <= clamp_high)
                    if in_range.any():
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
    loop = asyncio.get_event_loop()
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
# is spawned right before uvicorn binds the port. Each library's own
# tqdm progress bar is left untouched so the operator sees real
# byte-level progress in the terminal / journal. /health additionally
# reports a per-model state dict so client UIs (the lyrics_karaoke
# plugin, etc.) can poll for "warming up" status and surface progress.

def _warmup_demucs() -> None:
    """Pre-download the configured demucs separation model. Invokes
    run_demucs.py with --download-only so the soundfile patching path
    matches the real /separate flow."""
    _set_warmup_state("demucs", "downloading")
    run_demucs = str(Path(__file__).parent / "run_demucs.py")
    cmd = [sys.executable, run_demucs, "--download-only"]
    if _model:
        cmd.extend(["-n", _model])
    if _device:
        cmd.extend(["-d", _device])
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

    if args.skip_warmup:
        # Mark all warmup steps as "skipped" so /health reflects the
        # operator's choice — subsequent endpoint calls still work,
        # they just lazy-download on demand.
        for k in list(warmup_state.keys()):
            _set_warmup_state(k, "skipped")
    else:
        threading.Thread(target=_run_warmup, daemon=True).start()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
