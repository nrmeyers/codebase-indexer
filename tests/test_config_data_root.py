"""Tests for the user-scoped data-root resolver in :mod:`app.config`.

The resolver decides where the per-repo datastore lives when no explicit
override is given. Precedence (highest first):

1. explicit env (``CGR_DATA_DIR`` / ``LADYBUG_DB_DIR`` / ``JOBS_DB_PATH``)
2. an existing ``./.cgr`` in the working directory (legacy / in-place deploy)
3. the XDG user data dir (``${XDG_DATA_HOME:-~/.local/share}/codebase-indexer``)

These guard the "pipx install behaves like an installed tool" contract: a
fresh install run from any directory must NOT scatter ``.cgr`` into the cwd,
while existing in-place installs (and TheForge, which ``cd``s into its
service checkout) keep their current location with no migration.
"""
from __future__ import annotations

import pytest

from app.config import Settings


def _clear_data_env(mp: pytest.MonkeyPatch) -> None:
    for var in ("CGR_DATA_DIR", "LADYBUG_DB_DIR", "JOBS_DB_PATH", "XDG_DATA_HOME"):
        mp.delenv(var, raising=False)


def test_uses_legacy_cgr_when_present_in_cwd(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing ``./.cgr`` wins so in-place deploys are unchanged."""
    _clear_data_env(monkeypatch)
    (tmp_path / ".cgr").mkdir()
    monkeypatch.chdir(tmp_path)
    s = Settings()
    assert s.CGR_DATA_DIR == str(tmp_path / ".cgr")
    assert s.LADYBUG_DB_DIR == str(tmp_path / ".cgr" / "repos")
    assert s.JOBS_DB_PATH == str(tmp_path / ".cgr" / "jobs.sqlite")


def test_uses_xdg_data_home_when_no_legacy_cgr(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``./.cgr`` → user-scoped XDG dir, not the caller's cwd."""
    _clear_data_env(monkeypatch)
    xdg = tmp_path / "xdg-data"
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg))
    monkeypatch.chdir(tmp_path)  # deliberately no ./.cgr here
    s = Settings()
    assert s.CGR_DATA_DIR == str(xdg / "codebase-indexer")
    assert s.LADYBUG_DB_DIR == str(xdg / "codebase-indexer" / "repos")


def test_explicit_cgr_data_dir_reroots_subpaths(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single ``CGR_DATA_DIR`` knob moves the whole datastore coherently."""
    _clear_data_env(monkeypatch)
    monkeypatch.setenv("CGR_DATA_DIR", str(tmp_path / "srv"))
    monkeypatch.chdir(tmp_path)
    s = Settings()
    assert s.LADYBUG_DB_DIR == str(tmp_path / "srv" / "repos")
    assert s.JOBS_DB_PATH == str(tmp_path / "srv" / "jobs.sqlite")


def test_explicit_subpath_override_still_wins(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit ``LADYBUG_DB_DIR`` is not clobbered by the derive step."""
    _clear_data_env(monkeypatch)
    monkeypatch.setenv("CGR_DATA_DIR", str(tmp_path / "srv"))
    monkeypatch.setenv("LADYBUG_DB_DIR", "/custom/repos")
    monkeypatch.chdir(tmp_path)
    s = Settings()
    assert s.LADYBUG_DB_DIR == "/custom/repos"
    assert s.JOBS_DB_PATH == str(tmp_path / "srv" / "jobs.sqlite")
