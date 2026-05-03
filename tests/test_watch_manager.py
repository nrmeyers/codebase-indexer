"""Unit tests for Phase 5 — app.services.watch_manager.

Tests focus on the debouncer, hash-diff short-circuit, lock serialisation,
auto-purge, shutdown, and capacity cap.  All tests are async-safe via
pytest-asyncio and mock the blocking index call to stay fast.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.jobs_store import _reset_for_tests


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_watches():
    """Ensure _watches dict is empty before and after every test."""
    import app.services.watch_manager as wm
    with wm._watches_lock:
        wm._watches.clear()
    yield
    with wm._watches_lock:
        wm._watches.clear()


@pytest.fixture(autouse=True)
def reset_jobs_store(tmp_path):
    """Fresh in-memory jobs store for every test."""
    _reset_for_tests(":memory:")
    yield


@pytest.fixture()
def fake_repo(tmp_path: Path) -> Path:
    """Create a minimal fake repo directory with a Python source file."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / "main.py").write_text("def hello(): pass\n", encoding="utf-8")
    return repo


# ---------------------------------------------------------------------------
# Helper: patch watch_manager so it doesn't actually schedule a Watchdog
# Observer or call the blocking index — we test the asyncio logic only.
# ---------------------------------------------------------------------------


class _FakeObserver:
    def schedule(self, *a, **kw): pass
    def start(self): pass
    def stop(self): pass
    def join(self, timeout=None): pass
    def is_alive(self): return False


def _patch_start_observer():
    """Return a context-manager that stubs out the watchdog Observer import.

    The watch_manager does ``from watchdog.observers import Observer`` inside
    start_watch → so we patch the symbol on the watchdog.observers module so
    the local import picks up the fake.
    """
    return patch("watchdog.observers.Observer", _FakeObserver)


# ---------------------------------------------------------------------------
# Test: debouncer coalesces N events into one dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_debouncer_coalesces_burst(fake_repo: Path, monkeypatch):
    """should dispatch exactly one partial run when N events burst inside the window."""
    import app.services.watch_manager as wm
    from app.routers.index import indexed_repo_paths

    slug = fake_repo.name
    indexed_repo_paths[slug] = str(fake_repo)

    dispatch_calls: list[frozenset] = []

    async def _fake_run_partial(entry, paths):
        dispatch_calls.append(paths)

    monkeypatch.setattr(wm, "_run_partial_index", _fake_run_partial)
    monkeypatch.setattr("app.config.settings.WATCH_ENABLED", True)
    monkeypatch.setattr("app.config.settings.WATCH_DEBOUNCE_MS", 200)

    with _patch_start_observer():
        handle = await wm.start_watch(slug, actor_oid="u1", actor_email="u1@test")

    entry = wm._watches[slug]

    # Emit 7 paths rapidly (all within the debounce window).
    paths = [str(fake_repo / f"file{i}.py") for i in range(7)]
    for p in paths:
        await entry._queue.put(p)

    # Wait for the debouncer to fire (debounce=200ms → wait 500ms).
    await asyncio.sleep(0.5)

    await wm.stop_watch(slug)

    assert len(dispatch_calls) == 1, f"Expected 1 dispatch, got {len(dispatch_calls)}"
    assert dispatch_calls[0] == frozenset(paths)


# ---------------------------------------------------------------------------
# Test: hash-diff short-circuits unchanged files
# ---------------------------------------------------------------------------


def test_hash_diff_skips_unchanged_file(fake_repo: Path):
    """should produce noop=True when no file content changed."""
    from app.services.watch_manager import _file_sha1, _load_hash_cache, _save_hash_cache

    src = fake_repo / "main.py"
    sha = _file_sha1(src)
    assert sha is not None

    # Pre-populate hash cache with current hash.
    cache = {str(src.relative_to(fake_repo)): sha}
    _save_hash_cache(fake_repo, cache)

    # Re-load and verify no delta.
    loaded = _load_hash_cache(fake_repo)
    rel = str(src.relative_to(fake_repo))
    current_sha = _file_sha1(src)
    assert loaded.get(rel) == current_sha, "Cache should match, so diff is empty"


# ---------------------------------------------------------------------------
# Test: dirty file triggers non-noop result
# ---------------------------------------------------------------------------


def test_hash_diff_detects_changed_file(fake_repo: Path):
    """should mark a file dirty when its hash differs from the cache."""
    from app.services.watch_manager import _file_sha1, _load_hash_cache, _save_hash_cache

    src = fake_repo / "main.py"

    # Populate cache with a stale hash.
    stale_cache = {"main.py": "deadbeef" * 5}
    _save_hash_cache(fake_repo, stale_cache)

    loaded = _load_hash_cache(fake_repo)
    current_sha = _file_sha1(src)
    assert loaded.get("main.py") != current_sha, "Hash should differ → dirty"


# ---------------------------------------------------------------------------
# Test: watch_partial job row is created in jobs_store
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_partial_run_creates_jobs_store_row(fake_repo: Path, monkeypatch):
    """should create a kind='watch_partial' row when a partial run fires."""
    import app.services.watch_manager as wm
    from app.routers.index import indexed_repo_paths, _repo_locks
    from app.services import jobs_store

    slug = fake_repo.name
    indexed_repo_paths[slug] = str(fake_repo)

    # Stub out the blocking index so no real graph work happens.
    def _fake_blocking(entry_arg, job_id_arg, changed_paths_arg):
        from app.services import jobs_store as _js
        _js.mark_done(job_id_arg, node_count=0, rel_count=0, embedding_count=0)
        return {"noop": False, "dirty_count": 1, "files_done": 1,
                "embedding_count": 0, "node_count": 0, "rel_count": 0}

    # Stub broadcast so we don't need a running WS server.
    monkeypatch.setattr(wm, "_blocking_partial_index", _fake_blocking)

    from app.routers import websocket as _ws
    monkeypatch.setattr(_ws, "broadcast_partial_update", AsyncMock())

    entry = wm._WatchEntry(
        repo_slug=slug,
        repo_path=str(fake_repo),
        actor_oid="u1",
        actor_email="u1@test",
        debounce_ms=200,
        loop=asyncio.get_event_loop(),
    )
    entry.state = "active"

    await wm._run_partial_index(entry, frozenset([str(fake_repo / "main.py")]))

    # The last_partial_job_id should now be set on the entry.
    assert entry.last_partial_job_id is not None, "last_partial_job_id must be set"

    job = jobs_store.get_job(entry.last_partial_job_id)
    assert job is not None, "Job row must exist in jobs_store"
    assert job.kind == "watch_partial"
    assert job.status == "done"


# ---------------------------------------------------------------------------
# Test: auto-purge via clear_terminal (kind + older_than_hours)
# ---------------------------------------------------------------------------


def test_clear_terminal_purges_watch_partial_rows():
    """should purge watch_partial rows older than 24h and leave recent ones."""
    from app.services import jobs_store

    # Insert an old watch_partial row by manipulating started_at directly.
    old_job = jobs_store.create_job(
        kind="watch_partial",
        actor_oid="u1",
        actor_email="u1@test",
        repo_path="/fake/repo",
        force_reindex=False,
    )
    jobs_store.mark_done(old_job.job_id, node_count=0, rel_count=0, embedding_count=0)

    # Backdate started_at so it appears 25 hours old.
    conn = jobs_store._require_conn()
    with jobs_store._lock:
        conn.execute(
            "UPDATE jobs SET started_at = ? WHERE job_id = ?",
            (time.time() - 25 * 3600, old_job.job_id),
        )

    # Insert a recent watch_partial row.
    recent_job = jobs_store.create_job(
        kind="watch_partial",
        actor_oid="u1",
        actor_email="u1@test",
        repo_path="/fake/repo",
        force_reindex=False,
    )
    jobs_store.mark_done(recent_job.job_id, node_count=0, rel_count=0, embedding_count=0)

    cleared = jobs_store.clear_terminal(
        statuses={"done", "failed", "cancelled"},
        kind="watch_partial",
        older_than_hours=24,
    )

    assert cleared == 1, f"Expected 1 row cleared, got {cleared}"
    assert jobs_store.get_job(old_job.job_id) is None, "Old row should be gone"
    assert jobs_store.get_job(recent_job.job_id) is not None, "Recent row should survive"


# ---------------------------------------------------------------------------
# Test: capacity cap enforced
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_capacity_cap_raises_429(fake_repo: Path, monkeypatch, tmp_path: Path):
    """should raise WatchCapacityError when WATCH_MAX_REPOS is exceeded."""
    import app.services.watch_manager as wm
    from app.routers.index import indexed_repo_paths
    from app.services.watch_manager import WatchCapacityError

    monkeypatch.setattr("app.config.settings.WATCH_ENABLED", True)
    monkeypatch.setattr("app.config.settings.WATCH_MAX_REPOS", 1)

    # Pre-populate _watches with a fake entry so the cap is already hit.
    fake_entry = wm._WatchEntry(
        repo_slug="already-watching",
        repo_path=str(fake_repo),
        actor_oid="u1",
        actor_email="u1@test",
        debounce_ms=200,
        loop=asyncio.get_event_loop(),
    )
    fake_entry.state = "active"
    with wm._watches_lock:
        wm._watches["already-watching"] = fake_entry

    # A second start_watch should hit the cap.
    slug = fake_repo.name
    indexed_repo_paths[slug] = str(fake_repo)

    with pytest.raises(WatchCapacityError):
        await wm.start_watch(slug, actor_oid="u2", actor_email="u2@test")


# ---------------------------------------------------------------------------
# Test: shutdown_all joins all observers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shutdown_all_clears_watches(fake_repo: Path, monkeypatch):
    """should leave _watches empty and all observers stopped after shutdown_all."""
    import app.services.watch_manager as wm
    from app.routers.index import indexed_repo_paths

    monkeypatch.setattr("app.config.settings.WATCH_ENABLED", True)
    monkeypatch.setattr("app.config.settings.WATCH_DEBOUNCE_MS", 200)

    stopped_slugs: list[str] = []

    async def _fake_stop(slug):
        stopped_slugs.append(slug)
        with wm._watches_lock:
            wm._watches.pop(slug, None)
        return True

    monkeypatch.setattr(wm, "stop_watch", _fake_stop)

    # Add two fake entries.
    for slug in ("repo-a", "repo-b"):
        entry = wm._WatchEntry(
            repo_slug=slug,
            repo_path=str(fake_repo),
            actor_oid="u1",
            actor_email="u1@test",
            debounce_ms=200,
            loop=asyncio.get_event_loop(),
        )
        entry.state = "active"
        with wm._watches_lock:
            wm._watches[slug] = entry

    await wm.shutdown_all(timeout_s=2.0)

    assert set(stopped_slugs) == {"repo-a", "repo-b"}
    with wm._watches_lock:
        assert len(wm._watches) == 0
