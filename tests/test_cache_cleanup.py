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
