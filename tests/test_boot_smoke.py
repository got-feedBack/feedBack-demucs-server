"""Boot smoke test: actually import server.py and exercise the startup path.

The rest of the suite AST-extracts individual functions and never imports the
module, so it cannot catch a module that fails to boot — e.g. the model-idle
regression where ``_model_idle_timeout`` / ``_model_last_used`` /
``_model_idle_lock`` were referenced (in ``_startup_event``, ``_touch_model``
and ``_model_idle_loop``) but never defined, aborting startup with a NameError.

This test imports the real server.py in a subprocess with the heavy ML deps
stubbed (so it runs in CI without a GPU / multi-GB downloads), then calls
``_touch_model`` — the cheapest function that touches all three model-idle
globals — so a dropped definitions block resurfaces as a hard failure here.

It skips gracefully only when the import fails for a reason UNRELATED to the
server's own code (a genuinely missing/uninstallable dependency we didn't
stub); a NameError or a missing model-idle global is always a failure.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent

# Runs in a fresh interpreter: stub the heavy third-party modules, import the
# real server.py, then exercise the model-idle globals. Prints one sentinel.
_HARNESS = r"""
import sys, types
from unittest.mock import MagicMock

# torch needs a REAL load() so server.py's `inspect.signature(torch.load)`
# weights_only shim at import time doesn't choke on a MagicMock.
def _fake_torch_load(f, map_location=None, pickle_module=None, *,
                     weights_only=False, **kw):
    return None

_torch = types.ModuleType("torch")
_torch.load = _fake_torch_load
_torch.cuda = MagicMock()
_torch.from_numpy = MagicMock()
sys.modules["torch"] = _torch

for _name in ("torchcrepe", "librosa", "whisperx", "uvicorn", "numpy"):
    sys.modules[_name] = MagicMock()

# fastapi is imported as `from fastapi import FastAPI, ...` plus a couple of
# submodules; MagicMock modules satisfy all of those attribute/call sites,
# including `app = FastAPI(...)`, `app.add_middleware(...)`, `app.state.*`.
for _name in ("fastapi", "fastapi.middleware", "fastapi.middleware.cors",
              "fastapi.responses"):
    sys.modules[_name] = MagicMock()

import importlib
server = importlib.import_module("server")

# The three globals the dropped block defined. Missing any of these means the
# init block is gone again -> server can't start.
for _attr in ("_model_idle_timeout", "_model_last_used", "_model_idle_lock"):
    if not hasattr(server, _attr):
        print("SMOKE_MISSING_ATTR:" + _attr)
        sys.exit(3)

# Exercise the exact code path that used to raise NameError: _touch_model reads
# _model_idle_timeout and _model_idle_lock and writes _model_last_used.
server._touch_model("smoke-test")

print("SMOKE_OK")
"""


def test_server_module_boots():
    """server.py imports cleanly and its model-idle globals are defined."""
    result = subprocess.run(
        [sys.executable, "-c", _HARNESS],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=120,
    )
    combined = result.stdout + "\n" + result.stderr

    if "SMOKE_OK" in result.stdout:
        return  # server imported and the startup path ran without NameError

    if "SMOKE_MISSING_ATTR" in result.stdout:
        pytest.fail(
            "server.py is missing a model-idle global (dropped init block "
            "regression): " + combined
        )

    # A NameError at import/startup is exactly the blocker this test guards.
    if "NameError" in combined:
        pytest.fail("server.py failed to boot with a NameError:\n" + combined)

    # Only excuse a failure that is clearly an unrelated missing dependency
    # (something we didn't stub and isn't installed). Anything else is a real
    # boot failure and must fail the test.
    if "ModuleNotFoundError" in combined or "ImportError" in combined:
        pytest.skip(
            "Heavy dependency unavailable and not stubbed; cannot run boot "
            "smoke test in this environment:\n" + combined
        )

    pytest.fail(
        f"server.py boot smoke test failed (exit {result.returncode}):\n"
        + combined
    )
