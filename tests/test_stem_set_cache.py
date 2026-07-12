"""The cache must honour the requested STEM SET, not just the audio and the model. (#10)

`job_id` is `(audio_hash, model)` and deliberately does not include the stem set. That's
fine for the on-disk cache — `_check_cache` requires every requested stem to be present —
but `_enqueue_job` short-circuited on the in-memory jobs table and returned a completed
job's stems *regardless of what had been asked for*:

    POST /separate?model=bs_roformer_sw                         -> drums, bass, vocals, other
    POST /separate?model=bs_roformer_sw&stems=...,guitar,piano  -> drums, bass, vocals, other

...in 0 ms, with no error. The caller asked for guitar and piano, got neither, and the
response *looked* like a fast success. Silent and fast is the worst combination: nothing
tells you the answer is stale rather than authoritative.

Extracted via AST, like test_cache_cleanup, to avoid server.py's torch/whisperx import
chain — which is exactly why this can run in CI.
"""
import ast
import collections
import threading
import time
from pathlib import Path

import pytest

SERVER_PY = Path(__file__).parent.parent / "server.py"


def _load_enqueue_job(jobs, max_concurrent=2):
    """Extract _enqueue_job with a namespace standing in for server.py's module globals."""
    tree = ast.parse(SERVER_PY.read_text(encoding="utf-8"))
    node = next(n for n in ast.iter_child_nodes(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_enqueue_job")
    mod = ast.Module(body=[node], type_ignores=[])
    ast.copy_location(mod, node)

    started = []

    class _FakeThread:
        def __init__(self, target=None, args=(), daemon=None):
            self._args = args

        def start(self):
            started.append(self._args)      # record; never actually separate anything

    ns = {
        "jobs": jobs,
        "jobs_lock": threading.Lock(),
        "active_lock": threading.Lock(),
        "active_count": 0,
        "MAX_CONCURRENT": max_concurrent,
        "threading": type("T", (), {"Thread": _FakeThread}),
        "time": time,
        "_run_roformer": lambda *a: None,
        "_run_demucs": lambda *a: None,
        "_is_roformer_model": lambda m: "roformer" in m,
    }
    exec(compile(ast.unparse(mod), "<test>", "exec"), ns)
    return ns["_enqueue_job"], started


JOB_ID = "deadbeef-bs-roformer-sw"
FOUR = {"drums": "/download/x/drums.flac", "bass": "/download/x/bass.flac",
        "vocals": "/download/x/vocals.flac", "other": "/download/x/other.flac"}
SIX = dict(FOUR, guitar="/download/x/guitar.flac", piano="/download/x/piano.flac")
SIX_NAMES = ["drums", "bass", "vocals", "other", "guitar", "piano"]


def _completed(stems_all, stem_list=None):
    return collections.OrderedDict({JOB_ID: {
        "job_id": JOB_ID, "status": "complete", "progress": 100,
        "stems": dict(stems_all), "stems_all": dict(stems_all),
        "stem_list": list(stem_list or stems_all), "missing": [],
        "error": None, "model": "bs_roformer_sw", "created_at": time.time(),
    }})


def _in_flight(stem_list):
    return collections.OrderedDict({JOB_ID: {
        "job_id": JOB_ID, "status": "processing", "progress": 40,
        "stems": {}, "stems_all": {}, "stem_list": list(stem_list), "missing": [],
        "error": None, "model": "bs_roformer_sw", "created_at": time.time(),
    }})


def test_superset_request_is_not_served_from_a_smaller_completed_job():
    """THE bug: a 6-stem request answered with the cached 4, instantly, with no error."""
    jobs = _completed(FOUR)
    enqueue, started = _load_enqueue_job(jobs)
    result = enqueue(JOB_ID, "/tmp/a.ogg", SIX_NAMES, "bs_roformer_sw")

    assert result.get("cached") is not True, (
        "serving the cached 4-stem result for a 6-stem request is silent data loss — the "
        "caller asked for guitar and piano and got neither, with no error"
    )
    assert started, "it must actually re-separate rather than return a short result"


def test_exact_match_is_served_from_cache():
    jobs = _completed(SIX)
    enqueue, started = _load_enqueue_job(jobs)
    result = enqueue(JOB_ID, "/tmp/a.ogg", SIX_NAMES, "bs_roformer_sw")
    assert result["cached"] is True
    assert set(result["stems"]) == set(SIX)
    assert not started


def test_subset_request_is_served_from_a_larger_completed_job():
    """Fewer stems than were computed is a legitimate hit — and returns only what was asked
    for, not everything we happen to have lying around."""
    jobs = _completed(SIX)
    enqueue, started = _load_enqueue_job(jobs)
    result = enqueue(JOB_ID, "/tmp/a.ogg", ["vocals", "drums"], "bs_roformer_sw")
    assert result["cached"] is True
    assert set(result["stems"]) == {"vocals", "drums"}
    assert not started


def test_case_and_whitespace_do_not_defeat_the_coverage_check():
    jobs = _completed(SIX)
    enqueue, _ = _load_enqueue_job(jobs)
    assert enqueue(JOB_ID, "/tmp/a.ogg", [" Vocals ", "DRUMS"], "bs_roformer_sw")["cached"]


def test_a_job_from_before_this_fix_still_serves_its_stems():
    # Old jobs carry only `stems` (no `stems_all`). Coverage must fall back to it, or every
    # pre-existing entry would be needlessly re-separated.
    jobs = collections.OrderedDict({JOB_ID: {
        "job_id": JOB_ID, "status": "complete", "progress": 100,
        "stems": dict(SIX), "error": None, "model": "bs_roformer_sw",
        "created_at": time.time(),
    }})
    enqueue, started = _load_enqueue_job(jobs)
    result = enqueue(JOB_ID, "/tmp/a.ogg", ["vocals", "guitar"], "bs_roformer_sw")
    assert result["cached"] is True
    assert not started


def test_in_flight_job_with_a_smaller_set_is_not_silently_joined():
    """Riding along on a running 4-stem job completes without guitar/piano — the same silent
    loss, merely delayed."""
    jobs = _in_flight(["drums", "bass", "vocals", "other"])
    enqueue, _ = _load_enqueue_job(jobs)
    result = enqueue(JOB_ID, "/tmp/a.ogg", SIX_NAMES, "bs_roformer_sw")
    assert "error" in result
    assert result.get("status") != "processing"


def test_in_flight_job_that_covers_us_is_joined():
    jobs = _in_flight(SIX_NAMES)
    enqueue, started = _load_enqueue_job(jobs)
    result = enqueue(JOB_ID, "/tmp/a.ogg", ["vocals", "guitar"], "bs_roformer_sw")
    assert result["status"] == "processing"
    assert not started, "must attach to the running job, not start a second separation"


def test_pre_fix_job_with_original_cased_keys_returns_urls_not_an_empty_dict():
    """Coverage was checked case-insensitively while the lookup used the lowercased name.

    A job from before this fix stores `stems` keyed by the caller's ORIGINAL casing, so the
    check passed and the lookup then found nothing — `cached: True` with an EMPTY stems
    dict. A confident, instant, empty answer is worse than the bug this PR set out to fix.
    """
    jobs = collections.OrderedDict({JOB_ID: {
        "job_id": JOB_ID, "status": "complete", "progress": 100,
        "stems": {"Vocals": "/download/x/Vocals.flac", "Drums": "/download/x/Drums.flac"},
        "error": None, "model": "bs_roformer_sw", "created_at": time.time(),
    }})
    enqueue, started = _load_enqueue_job(jobs)
    result = enqueue(JOB_ID, "/tmp/a.ogg", ["vocals", "drums"], "bs_roformer_sw")

    assert result["cached"] is True
    assert set(result["stems"]) == {"vocals", "drums"}, "keys echo what the caller asked for"
    assert all(result["stems"].values()), "and every one must carry a real URL, not None"
    assert not started


def test_caller_casing_is_echoed_back_with_normalized_urls():
    jobs = _completed(SIX)
    enqueue, _ = _load_enqueue_job(jobs)
    result = enqueue(JOB_ID, "/tmp/a.ogg", ["Vocals", "GUITAR"], "bs_roformer_sw")
    assert result["cached"] is True
    assert set(result["stems"]) == {"Vocals", "GUITAR"}
    assert result["stems"]["GUITAR"] == SIX["guitar"]


def _load_check_cache(cache_dir):
    """Extract _check_cache, pointing it at a temp cache dir."""
    tree = ast.parse(SERVER_PY.read_text(encoding="utf-8"))
    node = next(n for n in ast.iter_child_nodes(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_check_cache")
    mod = ast.Module(body=[node], type_ignores=[])
    ast.copy_location(mod, node)
    ns = {
        "_cache_entry_path": lambda job_id: Path(cache_dir),
        "_remember_cache_entry": lambda job_id: None,
    }
    exec(compile(ast.unparse(mod), "<test>", "exec"), ns)
    return ns["_check_cache"]


def test_on_disk_cache_is_found_for_a_mixed_case_request(tmp_path):
    """The workers now write LOWERCASE filenames. Probing only the caller's own spelling
    would miss a cache entry that exists — and after a restart the in-memory jobs table is
    empty, so this is the ONLY path that can find it. A mixed-case request would silently
    re-separate something already on disk."""
    for name in ("vocals", "drums"):
        (tmp_path / f"{name}.flac").write_bytes(b"x")

    check = _load_check_cache(tmp_path)
    found = check(JOB_ID, ["Vocals", "DRUMS"], "bs_roformer_sw")

    assert found is not None, "a mixed-case request must still hit the lowercase cache files"
    assert set(found) == {"Vocals", "DRUMS"}, "keys echo the caller's spelling"
    assert found["Vocals"].endswith("vocals.flac"), "URL points at the file that exists"


def test_on_disk_cache_still_finds_pre_fix_original_cased_files(tmp_path):
    """Entries written before this change carry the caller's casing. They must still be found.

    Deliberately case-INSENSITIVE about the URL. Windows' filesystem is case-insensitive, so
    the lowercase probe matches `Vocals.flac` and we emit `vocals.flac`; Linux's is
    case-sensitive, so the lowercase probe misses and we emit `Vocals.flac`. Both resolve to
    the same file on their own platform. Asserting the exact spelling would encode the
    developer's OS into the test — which is precisely the class of bug that had these tests
    passing in CI while broken on Windows.
    """
    (tmp_path / "Vocals.flac").write_bytes(b"x")
    check = _load_check_cache(tmp_path)
    found = check(JOB_ID, ["Vocals"], "bs_roformer_sw")
    assert found is not None
    assert found["Vocals"].lower().endswith("vocals.flac")


def test_on_disk_cache_misses_when_a_requested_stem_is_absent(tmp_path):
    (tmp_path / "vocals.flac").write_bytes(b"x")
    check = _load_check_cache(tmp_path)
    assert check(JOB_ID, ["vocals", "guitar"], "bs_roformer_sw") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
