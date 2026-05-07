"""Smoke tests for ``app.services.source_fetch``.

The W1.D testing-gap audit flagged this as the highest-risk untested
file project-wide. Pure function over filesystem state — easy to test
with ``tmp_path`` + a fake LadybugDB connection.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.services import source_fetch


# ---------------------------------------------------------------------------
# fetch_source — pure file-IO over a line range
# ---------------------------------------------------------------------------


def test_fetch_source_reads_full_range(tmp_path: Path) -> None:
    f = tmp_path / "demo.py"
    f.write_text("a\nb\nc\nd\ne\n")
    assert source_fetch.fetch_source(str(f), 2, 4) == "b\nc\nd"


def test_fetch_source_default_line_start(tmp_path: Path) -> None:
    f = tmp_path / "demo.py"
    f.write_text("first\nsecond\nthird\n")
    # line_start=None → default 1; line_end=2 (inclusive 1..2)
    assert source_fetch.fetch_source(str(f), None, 2) == "first\nsecond"


def test_fetch_source_default_line_end_returns_single_line(tmp_path: Path) -> None:
    f = tmp_path / "demo.py"
    f.write_text("only\nlines\nhere\n")
    # line_end=None → start..start+1; with start=2 (1-indexed) returns "lines"
    assert source_fetch.fetch_source(str(f), 2, None) == "lines"


def test_fetch_source_missing_file_returns_empty() -> None:
    assert source_fetch.fetch_source("/nonexistent/path/foo.py", 1, 5) == ""


def test_fetch_source_empty_path_returns_empty() -> None:
    assert source_fetch.fetch_source("", 1, 5) == ""


def test_fetch_source_handles_non_utf8_bytes(tmp_path: Path) -> None:
    """Files with invalid UTF-8 should not crash; errors='replace' degrades gracefully."""
    f = tmp_path / "binary.py"
    f.write_bytes(b"hello \xff\xfe world\n")
    # Should return whatever utf-8-replace gives, NOT raise
    out = source_fetch.fetch_source(str(f), 1, 1)
    assert "hello" in out
    assert "world" in out


def test_fetch_source_clamps_line_end_past_eof(tmp_path: Path) -> None:
    f = tmp_path / "demo.py"
    f.write_text("a\nb\nc\n")
    # end past EOF → Python's slice clamps; should not raise
    assert source_fetch.fetch_source(str(f), 2, 999) == "b\nc"


# ---------------------------------------------------------------------------
# fetch_sources_for_symbols — joins LadybugDB locations to source slices
# ---------------------------------------------------------------------------


class _FakeResult:
    """Minimal mock of a LadybugDB query result iterator."""

    def __init__(self, rows: list[tuple[Any, ...]], cols: tuple[str, ...]) -> None:
        self._rows = rows
        self._cols = cols
        self._idx = 0

    def get_column_names(self) -> tuple[str, ...]:
        return self._cols

    def has_next(self) -> bool:
        return self._idx < len(self._rows)

    def get_next(self) -> tuple[Any, ...]:
        r = self._rows[self._idx]
        self._idx += 1
        return r


class _FakeConn:
    """Minimal mock of a LadybugDB connection.

    `executions` collects (query, params) tuples; `result_factory` is a
    callable returning a `_FakeResult` for each execution. Defaults to
    returning empty results (i.e. no rows match) so tests can opt-in.
    """

    def __init__(self, result_factory: Any = None) -> None:
        self.executions: list[tuple[str, dict]] = []
        self._result_factory = result_factory or (lambda _q, _p: _FakeResult([], ()))

    def execute(self, query: str, params: dict) -> _FakeResult:
        self.executions.append((query, params))
        return self._result_factory(query, params)


def test_fetch_sources_for_symbols_returns_empty_for_unknown_symbols() -> None:
    """Unknown qualified_names map to empty string, not missing key."""
    conn = _FakeConn()
    out = source_fetch.fetch_sources_for_symbols(conn, ["foo.bar.baz"])
    assert "foo.bar.baz" in out
    assert out["foo.bar.baz"] == ""


def test_fetch_sources_for_symbols_reads_real_file(tmp_path: Path) -> None:
    """When a row resolves to a real on-disk file, we read the source slice."""
    src = tmp_path / "module.py"
    src.write_text("def alpha():\n    return 1\n\ndef beta():\n    return 2\n")

    def factory(_q: str, params: dict) -> _FakeResult:
        if params.get("node_id") == "module.beta":
            # absolute path; root_path empty so we don't need to join
            return _FakeResult(
                [(str(src), "", 4, 5)],
                ("path", "root_path", "start_line", "end_line"),
            )
        return _FakeResult([], ())

    conn = _FakeConn(result_factory=factory)
    out = source_fetch.fetch_sources_for_symbols(conn, ["module.beta", "module.unknown"])
    assert "def beta()" in out["module.beta"]
    assert out["module.unknown"] == ""


def test_fetch_sources_for_symbols_resolves_relative_path_via_root_path(tmp_path: Path) -> None:
    """When `path` is relative, prepend `root_path` to find the file."""
    repo_root = tmp_path / "fake_repo"
    repo_root.mkdir()
    src = repo_root / "subdir" / "f.py"
    src.parent.mkdir()
    src.write_text("x\ny\nz\n")

    def factory(_q: str, _p: dict) -> _FakeResult:
        # `path` is relative; `root_path` is absolute
        return _FakeResult(
            [("subdir/f.py", str(repo_root), 2, 3)],
            ("path", "root_path", "start_line", "end_line"),
        )

    conn = _FakeConn(result_factory=factory)
    out = source_fetch.fetch_sources_for_symbols(conn, ["m.f"])
    assert out["m.f"] == "y\nz"


def test_fetch_sources_for_symbols_swallows_per_symbol_exceptions() -> None:
    """If the conn.execute raises mid-iteration, the bad symbol gets ''."""

    def factory(_q: str, params: dict) -> _FakeResult:
        if params.get("node_id") == "good.symbol":
            return _FakeResult([], ())
        raise RuntimeError("simulated DB hiccup on this symbol only")

    conn = _FakeConn(result_factory=factory)
    out = source_fetch.fetch_sources_for_symbols(conn, ["good.symbol", "bad.symbol"])
    assert out == {"good.symbol": "", "bad.symbol": ""}


def test_fetch_sources_for_symbols_preserves_input_order_in_dict() -> None:
    """Output dict iteration order should match input list order (Python 3.7+ guaranteed)."""
    conn = _FakeConn()
    inputs = ["a", "b", "c", "d"]
    out = source_fetch.fetch_sources_for_symbols(conn, inputs)
    assert list(out.keys()) == inputs


def test_fetch_sources_for_symbols_empty_input_returns_empty_dict() -> None:
    conn = _FakeConn()
    assert source_fetch.fetch_sources_for_symbols(conn, []) == {}


def test_fetch_sources_for_symbols_logs_one_query_per_symbol() -> None:
    """Each requested symbol issues exactly one DB query (no batching)."""
    conn = _FakeConn()
    source_fetch.fetch_sources_for_symbols(conn, ["one", "two", "three"])
    assert len(conn.executions) == 3
    seen_ids = {params["node_id"] for _q, params in conn.executions}
    assert seen_ids == {"one", "two", "three"}
