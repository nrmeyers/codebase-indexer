"""Tests for the bounded LadybugDB buffer-pool sizing helper.

These tests import ONLY ``app.services.ladybug_buffer_pool`` (a leaf module
with no model / embedder / FastAPI dependencies) plus ``ladybug`` for
the live-open smoke test, so they are safe to run under tight host memory
pressure without booting the service or pulling in torch.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.services.ladybug_buffer_pool import (
    DEFAULT_BUFFER_POOL_SIZE,
    ENV_VAR,
    resolve_buffer_pool_size,
)


# ---------------------------------------------------------------------------
# (a) the env var is parsed and bounds the value
# ---------------------------------------------------------------------------


def test_resolves_explicit_positive_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid positive integer env value is returned verbatim."""
    monkeypatch.setenv(ENV_VAR, str(512 * 1024 * 1024))  # 512 MiB
    assert resolve_buffer_pool_size() == 512 * 1024 * 1024


def test_strips_surrounding_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leading/trailing whitespace around the integer is tolerated."""
    monkeypatch.setenv(ENV_VAR, "  1073741824  ")  # 1 GiB
    assert resolve_buffer_pool_size() == 1073741824


def test_unset_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the env var is unset, the bounded default is used."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert resolve_buffer_pool_size() == DEFAULT_BUFFER_POOL_SIZE


def test_default_is_two_gib() -> None:
    """The default is a fixed 2 GiB cap (not a fraction of RAM)."""
    assert DEFAULT_BUFFER_POOL_SIZE == 2 * 1024 * 1024 * 1024
    assert DEFAULT_BUFFER_POOL_SIZE == 2_147_483_648


# ---------------------------------------------------------------------------
# (b) invalid values fall back to the default
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_value",
    [
        "",          # empty
        "   ",       # whitespace only
        "not-a-num", # non-numeric
        "1.5",       # float string (int() rejects)
        "0",         # zero == Kùzu auto-size sentinel (the bug we guard against)
        "-1",        # negative
        "-2147483648",
    ],
)
def test_invalid_values_fall_back_to_default(
    monkeypatch: pytest.MonkeyPatch, bad_value: str
) -> None:
    """Empty / non-numeric / zero / negative all fall back to the default."""
    monkeypatch.setenv(ENV_VAR, bad_value)
    assert resolve_buffer_pool_size() == DEFAULT_BUFFER_POOL_SIZE


def test_result_is_always_strictly_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    """The helper never returns 0 (Kùzu's unbounded auto-size sentinel)."""
    for value in ("", "0", "-5", "garbage", None):
        if value is None:
            monkeypatch.delenv(ENV_VAR, raising=False)
        else:
            monkeypatch.setenv(ENV_VAR, value)
        assert resolve_buffer_pool_size() > 0


# ---------------------------------------------------------------------------
# (c) a LadybugDB (Kùzu fork) db opens successfully with the bounded size
# ---------------------------------------------------------------------------


def test_database_opens_with_bounded_buffer_pool() -> None:
    """A real db opens + serves a query with the bounded buffer pool.

    Uses a small explicit cap (256 MiB) on a throwaway temp path so the
    open is light even under host memory pressure. This is the regression
    test for the ``Mmap for size ... failed`` hard-fail: a bounded pool
    must open where the unbounded default could not.
    """
    lb = pytest.importorskip("ladybug")

    bounded = 256 * 1024 * 1024  # 256 MiB — comfortably small

    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "bufpool_test.db")
        db = lb.Database(db_path, buffer_pool_size=bounded)
        try:
            conn = lb.Connection(db)
            res = conn.execute("RETURN 1 AS one")
            assert res.has_next()
            assert int(res.get_next()[0]) == 1
        finally:
            del db


def test_resolved_size_opens_a_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """The helper's resolved value is a legal ``buffer_pool_size`` for open."""
    lb = pytest.importorskip("ladybug")

    # Pin a small bounded value via the env var so the open stays light.
    monkeypatch.setenv(ENV_VAR, str(256 * 1024 * 1024))
    resolved = resolve_buffer_pool_size()

    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "resolved_test.db")
        db = lb.Database(db_path, buffer_pool_size=resolved)
        try:
            conn = lb.Connection(db)
            assert conn.execute("RETURN 42 AS answer").get_next()[0] == 42
        finally:
            del db
