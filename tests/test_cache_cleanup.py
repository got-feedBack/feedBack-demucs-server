"""TDD: Test cache cleanup TTL parser (_parse_ttl).

Extracts the function from server.py using AST to avoid the heavy
torch/whisperx import chain that requires GPU dependencies.
"""

import ast
import re
from pathlib import Path

import pytest

SERVER_PY = Path(__file__).parent.parent / "server.py"


def _extract_function(func_name: str):
    """Extract a function from server.py source code using AST."""
    source = SERVER_PY.read_text()
    tree = ast.parse(source)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            # Collect any decorators/comments before the function too
            mod = ast.Module(body=[node], type_ignores=[])
            ast.copy_location(mod, node)
            code = compile(ast.unparse(mod), "<test>", "exec")
            namespace = {"re": re}
            exec(code, namespace)
            return namespace[func_name]
    raise ValueError(f"Function {func_name} not found in {SERVER_PY}")


# Extract the pure function once at module load
_parse_ttl = _extract_function("_parse_ttl")


class TestParseTtl:
    """Test suite for _parse_ttl — a pure function with no side effects."""

    def test_one_hour(self):
        assert _parse_ttl("1h") == 3600

    def test_twelve_hours(self):
        assert _parse_ttl("12h") == 43200

    def test_twentyfour_hours(self):
        assert _parse_ttl("24h") == 86400

    def test_never_disables(self):
        assert _parse_ttl("NEVER") is None

    def test_never_case_insensitive(self):
        assert _parse_ttl("never") is None
        assert _parse_ttl("Never") is None

    def test_whitespace_trimmed(self):
        assert _parse_ttl(" 1h ") == 3600
        assert _parse_ttl("  12h  ") == 43200

    def test_minutes_unit(self):
        assert _parse_ttl("60m") == 3600
        assert _parse_ttl("30m") == 1800

    def test_days_unit(self):
        assert _parse_ttl("1d") == 86400
        assert _parse_ttl("7d") == 604800

    def test_invalid_input_defaults_to_one_hour(self):
        assert _parse_ttl("garbage") == 3600
        assert _parse_ttl("") == 3600
        assert _parse_ttl("1x") == 3600
        assert _parse_ttl("abc") == 3600

    def test_zero_value(self):
        assert _parse_ttl("0h") == 0

    def test_large_value(self):
        # 999 hours should be fine
        assert _parse_ttl("999h") == 999 * 3600


# ── Requirements pin security ───────────────────────────────────────────

def _parse_requirement_pin(line: str):
    """Parse a requirements.txt line like 'fastapi>=0.109.1'.
    
    Returns (package_name, min_version_str) or None for non-pin lines.
    """
    line = line.strip()
    if not line or line.startswith('#') or line.startswith('-'):
        return None
    import re
    m = re.match(r'^([a-zA-Z0-9_.-]+)\s*>=\s*(\d+\.\d+\.\d+.*)$', line)
    if m:
        return m.group(1), m.group(2)
    # Also match unbounded pins like 'fastapi>=0.100.0'
    m = re.match(r'^([a-zA-Z0-9_.-]+)\s*>=\s*(\d[\d.]*)$', line)
    if m:
        return m.group(1), m.group(2)
    return None


def _read_requirements():
    """Read requirements.txt and return dict of package -> min_version."""
    req_path = Path(__file__).parent.parent / "requirements.txt"
    pins = {}
    with open(req_path) as f:
        for line in f:
            parsed = _parse_requirement_pin(line)
            if parsed:
                pins[parsed[0]] = parsed[1]
    return pins


class TestRequirementsPins:
    """Minimum version pins must be high enough to avoid known CVEs."""

    REQUIRED_PINS = {
        "fastapi": "0.109.1",
        "python-multipart": "0.0.7",
    }

    def test_requirements_file_exists(self):
        req_path = Path(__file__).parent.parent / "requirements.txt"
        assert req_path.exists(), "requirements.txt not found"

    def test_parse_fastapi_pin(self):
        pins = _read_requirements()
        assert "fastapi" in pins, "fastapi pin missing from requirements.txt"
        actual = pins["fastapi"]
        required = self.REQUIRED_PINS["fastapi"]
        assert actual >= required, (
            f"fastapi pin {actual} is below minimum {required} "
            f"(vuln < {required})"
        )

    def test_parse_multipart_pin(self):
        pins = _read_requirements()
        assert "python-multipart" in pins, "python-multipart pin missing"
        actual = pins["python-multipart"]
        required = self.REQUIRED_PINS["python-multipart"]
        assert actual >= required, (
            f"python-multipart pin {actual} is below minimum {required} "
            f"(vuln < {required})"
        )


# ── feedBack-demucs.service systemd config ────────────────────────────

class TestServiceFile:
    """Systemd service file must have correct network ordering."""

    SERVICE_PATH = Path(__file__).parent.parent / "feedBack-demucs.service"

    def test_service_file_exists(self):
        assert self.SERVICE_PATH.exists(), "feedBack-demucs.service not found"

    def test_after_includes_network_online(self):
        """After= must reference both network.target AND network-online.target
        to prevent the service from starting before the network stack is
        fully ready (Wants= alone doesn't enforce ordering)."""
        content = self.SERVICE_PATH.read_text()
        assert "After=network.target network-online.target" in content, (
            "After= should be 'network.target network-online.target', "
            "not just 'network.target'"
        )


# ── Stale jobs cleanup on cache expiry ─────────────────────────────────

class TestStaleJobsCleanup:
    """When a cache directory is deleted by the cleanup sweep, the
    corresponding job entry in the `jobs` dict must also be removed
    to prevent returning 'cached: true' with dead download links."""

    def test_cleanup_removes_stale_jobs(self):
        """Extract _cache_cleanup_loop from server.py and verify that
        shutil.rmtree is immediately followed by jobs.pop with the
        same entry name, wrapped in the jobs_lock."""
        source = SERVER_PY.read_text()
        tree = ast.parse(source)
        
        # Find _cache_cleanup_loop function
        func_node = None
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_cache_cleanup_loop":
                func_node = node
                break
        
        assert func_node is not None, "_cache_cleanup_loop not found in server.py"
        
        # Unparse the function to get its source text
        func_text = ast.unparse(func_node)
        
        # Check that shutil.rmtree is called
        assert "shutil.rmtree" in func_text, (
            "Cache cleanup must call shutil.rmtree to delete expired dirs"
        )
        
        # Check that jobs.pop is called after shutil.rmtree
        # The pattern should be: shutil.rmtree(...); with jobs_lock: jobs.pop(...)
        rmtree_idx = func_text.index("shutil.rmtree")
        after_rmtree = func_text[rmtree_idx:]
        assert "jobs_lock" in after_rmtree, (
            "jobs_lock must be acquired after deleting cache dir, "
            "found: " + after_rmtree[:200]
        )
        assert "jobs.pop" in after_rmtree, (
            "jobs.pop must be called after rmtree to remove stale job entry, "
            "found: " + after_rmtree[:200]
        )


# ── First cache sweep should not be delayed ────────────────────────────

class TestFirstSweepImmediate:
    """The first cache sweep should run immediately at startup, not
    sleep for CHECK_INTERVAL (10 min) before the first sweep."""

    def test_sleep_at_end_of_loop_not_beginning(self):
        """Extract _cache_cleanup_loop and verify time.sleep is at the
        END of the while body, not the beginning. This ensures the
        first sweep runs immediately on startup."""
        source = SERVER_PY.read_text()
        tree = ast.parse(source)
        
        func_node = None
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_cache_cleanup_loop":
                func_node = node
                break
        
        assert func_node is not None, "_cache_cleanup_loop not found"
        
        # Find the while loop inside the function
        while_node = None
        for node in ast.walk(func_node):
            if isinstance(node, ast.While):
                while_node = node
                break
        
        assert while_node is not None, "while True loop not found in _cache_cleanup_loop"
        
        # Get the body of the while loop
        body = while_node.body
        
        # The time.sleep should be the LAST statement in the while body
        # (not the first), so the first sweep runs immediately on startup
        last_stmt = body[-1] if body else None
        last_stmt_text = ast.unparse(last_stmt) if last_stmt else ""
        
        first_stmt = body[0] if body else None
        first_stmt_text = ast.unparse(first_stmt) if first_stmt else ""
        
        # Check that the first statement is NOT a time.sleep
        is_first_sleep = "time.sleep" in first_stmt_text
        
        print(f"  First statement in while body: {first_stmt_text[:100]}")
        print(f"  Last statement in while body: {last_stmt_text[:100]}")
        
        assert not is_first_sleep, (
            f"time.sleep is the FIRST statement in the while loop "
            f"({first_stmt_text[:80]}...). "
            f"It should be the LAST statement so the first sweep "
            f"runs immediately on startup."
        )
        
        # Also verify sleep IS somewhere (it should be last)
        assert "time.sleep" in last_stmt_text, (
            f"time.sleep should be the LAST statement in the while loop, "
            f"but last statement is: {last_stmt_text[:100]}"
        )
