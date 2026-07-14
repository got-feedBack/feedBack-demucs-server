"""`POST /transcribe` — the endpoint feedBack's transcribe path always meant to call.

There wasn't one. The client posted the vocals stem to `/align`, whose `text` form field is
REQUIRED, so FastAPI rejected the request with a 422 before the handler ever ran — every time,
for everyone. Remote transcription had never worked
(got-feedBack/feedBack-plugin-stem-splitter#17).

The bug is a *contract* bug, so the test has to exercise the contract: a real FastAPI app, a real
request, no lyrics in it. A unit test calling `transcribe_audio(...)` directly would pass while
the endpoint 422s, because the rejection happens in FastAPI's validation layer, above the handler
— i.e. it would reproduce exactly the blind spot that let this ship.

Heavy ML is stubbed (no GPU, no multi-GB downloads); FastAPI is real, which is the whole point.
Runs in a subprocess so the sys.modules stubs can't leak into the rest of the suite.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent

_HARNESS = r"""
import sys, types, io
from unittest.mock import MagicMock

def _fake_torch_load(f, map_location=None, pickle_module=None, *, weights_only=False, **kw):
    return None

_torch = types.ModuleType("torch")
_torch.load = _fake_torch_load
_torch.cuda = MagicMock()
_torch.from_numpy = MagicMock()
sys.modules["torch"] = _torch

for _name in ("torchcrepe", "librosa", "numpy"):
    sys.modules[_name] = MagicMock()

# whisperx is stubbed, but load_audio must return something with a real length: the handler
# computes a duration from it, and len(MagicMock()) raises.
_whisperx = types.ModuleType("whisperx")
_whisperx.load_audio = lambda path: [0.0] * (16000 * 10)      # 10 seconds
_whisperx.align = lambda segments, model, meta, audio, device, return_char_alignments=False: {
    "segments": [{
        "start": 1.0, "end": 2.0, "text": "hello world",
        "words": [
            {"word": "hello", "start": 1.0, "end": 1.4, "score": 0.9},
            {"word": "world", "start": 1.5, "end": 2.0, "score": 0.8},
        ],
    }]
}
sys.modules["whisperx"] = _whisperx

# fastapi is REAL. If it were stubbed, the 422 this bug is made of could not happen.
import importlib
server = importlib.import_module("server")
from fastapi.testclient import TestClient

class _ASR:
    def __init__(self, segments, language="en"):
        self._segments, self._language = segments, language
    def transcribe(self, audio, **kw):
        return {"segments": self._segments, "language": self._language}

def _install(segments, language="en"):
    server._get_whisperx_model = lambda: _ASR(segments, language)
    server._get_whisperx_aligner = lambda lang: (MagicMock(), MagicMock())
    server._whisperx_device = lambda: "cpu"

client = TestClient(server.app)
AUDIO = {"file": ("vocals.ogg", io.BytesIO(b"not really ogg"), "audio/ogg")}

def fail(msg):
    print("HARNESS_FAIL:" + msg)
    sys.exit(3)

# 1. THE regression. A transcribe request carries no lyrics. That must not be a 422.
_install([{"start": 1.0, "end": 2.0, "text": "hello world"}])
r = client.post("/transcribe", files=AUDIO)
if r.status_code == 422:
    fail("/transcribe 422s on a request with no lyrics - this IS the bug (#17)")
if r.status_code != 200:
    fail("/transcribe returned %s: %s" % (r.status_code, r.text[:300]))

body = r.json()
if not body.get("segments"):
    fail("no segments returned")
words = body["segments"][0].get("words")
if not words or words[0].get("word") != "hello":
    fail("response is not native whisperx.align() shape (the client's mapper reads .words)")
if body.get("language") != "en":
    fail("language missing from response")

# 2. The language hint is honoured, and a bad one is the CALLER's fault (400, not 500).
_install([{"start": 0.0, "end": 1.0, "text": "hola"}], language="es")
r = client.post("/transcribe", files=AUDIO, data={"language": "es"})
if r.status_code != 200 or r.json().get("language") != "es":
    fail("explicit language hint not honoured: %s %s" % (r.status_code, r.text[:200]))

r = client.post("/transcribe", files=AUDIO, data={"language": "en-US!!"})
if r.status_code != 400:
    fail("a malformed language code must be a 400, got %s" % r.status_code)

# 3. An instrumental is an ANSWER, not an error: no vocals -> no words, 200.
_install([])
r = client.post("/transcribe", files=AUDIO)
if r.status_code != 200:
    fail("an instrumental must not be an error, got %s" % r.status_code)
if r.json().get("segments") != []:
    fail("an instrumental must return an empty segment list")

# 4. /align still REQUIRES text — that difference is the whole reason /transcribe exists, and
#    if /align ever stopped 422ing here, the original bug would have become invisible instead
#    of fixed.
r = client.post("/align", files=AUDIO)
if r.status_code != 422:
    fail("/align without text should still be a 422 (it is forced alignment), got %s" % r.status_code)

print("TRANSCRIBE_OK")
"""


def test_transcribe_accepts_audio_with_no_lyrics():
    """The contract #17 broke: transcription takes audio and nothing else."""
    result = subprocess.run(
        [sys.executable, "-c", _HARNESS],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=180,
    )
    combined = result.stdout + "\n" + result.stderr

    if "TRANSCRIBE_OK" in result.stdout:
        return
    if "HARNESS_FAIL:" in result.stdout:
        pytest.fail(combined)
    if "ModuleNotFoundError" in combined:
        pytest.skip("a dependency we don't stub is missing:\n" + combined)
    pytest.fail("harness did not complete:\n" + combined)
