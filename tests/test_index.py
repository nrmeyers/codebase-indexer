"""Tests for POST /index and GET /index/{job_id}/status."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers.index import _jobs
from app.services import jobs_store

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_jobs() -> None:
    """Isolate tests by clearing the in-memory job store and resetting jobs_store."""
    _jobs.clear()
    # Reset and re-init jobs_store against in-memory SQLite for each test.
    jobs_store._reset_for_tests(":memory:")
    yield
    _jobs.clear()


def test_post_index_accepts_valid_repo(tmp_path: Path) -> None:
    with patch("app.routers.index._run_ingestion", new_callable=AsyncMock):
        resp = client.post("/index", json={"repo_path": str(tmp_path)})
    assert resp.status_code == 202
    body = resp.json()
    assert "job_id" in body
    assert body["message"] == "Indexing job accepted"


def test_post_index_rejects_missing_path() -> None:
    resp = client.post(
        "/index", json={"repo_path": "/this/path/does/not/exist/ever"}
    )
    assert resp.status_code == 422
    assert "does not exist" in resp.json()["detail"]


def test_get_status_running(tmp_path: Path) -> None:
    with patch("app.routers.index._run_ingestion", new_callable=AsyncMock):
        post = client.post("/index", json={"repo_path": str(tmp_path)})
    job_id = post.json()["job_id"]

    resp = client.get(f"/index/{job_id}/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == job_id
    assert body["status"] in ("running", "done", "failed")


def test_get_status_not_found() -> None:
    resp = client.get("/index/nonexistent-job-id/status")
    assert resp.status_code == 404


def test_get_status_done(tmp_path: Path) -> None:
    """Simulate a completed job by patching _blocking_index."""

    def _fake_blocking(job, force_reindex):  # type: ignore[override]
        job.node_count = 42
        job.rel_count = 10
        job.progress_pct = 100.0
        job.status = "done"

    with patch("app.routers.index._blocking_index", side_effect=_fake_blocking):
        post = client.post("/index", json={"repo_path": str(tmp_path)})

    job_id = post.json()["job_id"]

    # Give the background task time to run.
    import time

    deadline = time.time() + 5
    while time.time() < deadline:
        resp = client.get(f"/index/{job_id}/status")
        if resp.json()["status"] == "done":
            break
        time.sleep(0.05)

    assert resp.json()["status"] == "done"
    assert resp.json()["node_count"] == 42
    assert resp.json()["rel_count"] == 10


def test_get_status_failed(tmp_path: Path) -> None:
    def _fail(job, force_reindex):  # type: ignore[override]
        raise RuntimeError("ingestion exploded")

    with patch("app.routers.index._blocking_index", side_effect=_fail):
        post = client.post("/index", json={"repo_path": str(tmp_path)})

    job_id = post.json()["job_id"]

    import time

    deadline = time.time() + 5
    while time.time() < deadline:
        resp = client.get(f"/index/{job_id}/status")
        if resp.json()["status"] == "failed":
            break
        time.sleep(0.05)

    body = resp.json()
    assert body["status"] == "failed"
    assert "ingestion exploded" in (body["error"] or "")


def test_post_index_accepts_force_reindex_flag(tmp_path: Path) -> None:
    """force_reindex=true should be accepted and passed through to the ingestion job."""
    with patch("app.routers.index._run_ingestion", new_callable=AsyncMock):
        resp = client.post("/index", json={"repo_path": str(tmp_path), "force_reindex": True})
    assert resp.status_code == 202
    body = resp.json()
    assert "job_id" in body
    assert body["message"] == "Indexing job accepted"


def test_post_index_duplicate_same_repo_returns_409(tmp_path: Path) -> None:
    """Two POSTs with the same repo_path: the second returns 409 Conflict.

    LadybugDB is a single-writer database, so the service enforces one
    indexing job per repo at a time via an asyncio.Lock. A second concurrent
    POST for the same repo must fail fast with 409 rather than queue or
    run concurrently (which would corrupt the DB).
    """
    with patch("app.routers.index._run_ingestion", new_callable=AsyncMock):
        resp1 = client.post("/index", json={"repo_path": str(tmp_path)})
        resp2 = client.post("/index", json={"repo_path": str(tmp_path)})

    # First request accepted, second rejected while the first is still running
    # (the mocked _run_ingestion never completes, so the lock is still held).
    assert resp1.status_code == 202
    assert resp2.status_code == 409
    job_id1 = resp1.json()["job_id"]
    # The first job should still be retrievable.
    assert client.get(f"/index/{job_id1}/status").status_code == 200


# ---------------------------------------------------------------------------
# Phase 2 integration tests — jobs_store persistence
# ---------------------------------------------------------------------------


def test_restart_recovery_marks_jobs_interrupted(tmp_path: Path) -> None:
    """Should mark running jobs as 'interrupted' when swept by a different worker token.

    Simulates a service restart by writing a running job into jobs_store with
    worker_token='old-worker', then calling sweep_interrupted with a new token.
    The job should transition to 'interrupted' — visible via GET /index/{job_id}/status
    once the in-memory _jobs dict is cleared (simulating the restart).
    """
    # Write a running job directly to the persistent store.
    old_token = "old-worker-token"
    persisted = jobs_store.create_job(
        kind="index",
        actor_oid="",
        actor_email="",
        repo_path=str(tmp_path),
        force_reindex=False,
        exclude_paths=frozenset(),
        worker_token=old_token,
        initial_status="running",
        initial_phase="parsing",
    )

    # Simulate restart: sweep with a new token.
    swept = jobs_store.sweep_interrupted("new-worker-token")
    assert swept == 1, f"Expected 1 swept job, got {swept}"

    # The _jobs dict is empty (restart cleared it).
    assert persisted.job_id not in _jobs

    # GET /index/{job_id}/status should now return 'interrupted' from the store.
    resp = client.get(f"/index/{persisted.job_id}/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "interrupted", f"Expected 'interrupted', got {body['status']!r}"
    assert body["job_id"] == persisted.job_id


def test_concurrent_post_same_repo_returns_409_via_store(tmp_path: Path) -> None:
    """Should return 409 when jobs_store shows an active job for the same repo.

    Writes an active job directly into jobs_store (simulating the state after
    a POST /index that is still running). A new POST /index for the same repo
    path must be rejected with 409 via the store check, before the in-memory
    _jobs scan even runs.
    """
    # Simulate an already-running job in the persistent store.
    jobs_store.create_job(
        kind="index",
        actor_oid="",
        actor_email="",
        repo_path=str(tmp_path),
        force_reindex=False,
        exclude_paths=frozenset(),
        worker_token="some-worker-token",
        initial_status="running",
        initial_phase="parsing",
    )

    # POST /index for the same repo — should be rejected by store check.
    with patch("app.routers.index._run_ingestion", new_callable=AsyncMock):
        resp = client.post("/index", json={"repo_path": str(tmp_path)})

    assert resp.status_code == 409
    assert "already running" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Embed-pass lock-conflict regression
# ---------------------------------------------------------------------------


def test_blocking_embed_opens_ladybug_read_only(tmp_path: Path) -> None:
    """The embed subprocess driver MUST open LadybugDB with ``read_only=True``.

    Without this flag the subprocess takes a write lock on the same .db
    file the live indexer is already holding open, causing the embed pass
    to die with ``IO exception: Could not set lock on file: ...`` almost
    immediately after start.  This test pins the contract so a future
    refactor can't silently regress it.

    Post-BUC-1601, the driver is invoked via ``python -m
    app.scripts.embed_driver`` rather than an inline ``python -c "<body>"``
    f-string.  This test therefore pins two invariants:

      1. ``_blocking_embed`` invokes the driver as a module
         (``-m app.scripts.embed_driver``), never as an inline ``-c`` body.
      2. The driver module's source opens every LadybugDB it touches with
         ``read_only=True``.  Asserting on the source file is what makes
         this resistant to subprocess-flag refactors: if a future change
         drops the kwarg from the driver itself, this test fails before
         the embed pass ever wedges on a write lock in production.
    """
    import inspect
    import re
    import subprocess
    import sys
    from unittest.mock import MagicMock, patch

    import app.scripts.embed_driver as embed_driver_mod
    from app.routers import index as index_mod
    from app.routers.index import _EmbedJob, _blocking_embed

    # Make the configured per-repo .db live under tmp_path so the
    # FileNotFoundError early-exit doesn't trip.
    db_dir = tmp_path
    repo_name = "fakerepo"
    # Redirect LADYBUG_DB_DIR to tmp_path; db_path_for_repo recomputes
    # from this attribute on every call so this is enough.
    orig_dir = index_mod.settings.LADYBUG_DB_DIR
    object.__setattr__(index_mod.settings, "LADYBUG_DB_DIR", str(db_dir))
    try:
        fake_db = Path(index_mod.settings.db_path_for_repo(repo_name))
        fake_db.parent.mkdir(parents=True, exist_ok=True)
        fake_db.write_bytes(b"\x00" * 8)

        captured: dict[str, list[str]] = {}

        def fake_run(cmd, *args, **kwargs):  # noqa: ARG001
            captured["argv"] = list(cmd)
            result = MagicMock()
            result.returncode = 0
            return result

        job = _EmbedJob(
            job_id="t1", repo_name=repo_name, repo_path=str(tmp_path),
        )

        with patch.object(subprocess, "run", side_effect=fake_run):
            _blocking_embed(job)
    finally:
        object.__setattr__(index_mod.settings, "LADYBUG_DB_DIR", orig_dir)

    # --- Invariant 1: subprocess shells out to the driver as a *module*. ---
    argv = captured["argv"]
    assert argv[0] == sys.executable, f"driver must run under the same interpreter: {argv!r}"
    assert argv[1] == "-m", (
        f"driver must be invoked via ``python -m`` (BUC-1601), got argv[1]={argv[1]!r}. "
        "Inline ``-c`` bodies are forbidden — they bypass the unit-tested driver module."
    )
    assert argv[2] == "app.scripts.embed_driver", (
        f"driver module path must be ``app.scripts.embed_driver``, got {argv[2]!r}"
    )
    # Required driver flags — these are the contract _blocking_embed owes
    # the driver module.  If a future refactor drops one of them, the
    # driver will fail at argparse time and embed jobs will never start.
    assert "--repo-db-path" in argv, "driver must receive --repo-db-path"
    assert "--vec-db-path" in argv, "driver must receive --vec-db-path"
    assert "--repo-path" in argv, "driver must receive --repo-path"

    # --- Invariant 2: every LadybugDB open in the driver is read_only=True. ---
    # We grep the driver source directly rather than relying on the cmd
    # body (which post-BUC-1601 no longer contains the open call).  This
    # is strictly more specific than the old check: it pins the *driver
    # source* against silently losing the read_only kwarg on any
    # lb.Database() open, not just the first one.
    driver_src = inspect.getsource(embed_driver_mod)
    db_opens = re.findall(r"lb\.Database\([^)]*\)", driver_src)
    assert db_opens, (
        "Expected at least one ``lb.Database(...)`` open in "
        "app/scripts/embed_driver.py — driver source may have moved."
    )
    for open_call in db_opens:
        assert "read_only=True" in open_call, (
            f"Embed driver opens LadybugDB without read_only=True: {open_call!r}. "
            "This will cause IO lock conflicts with the live FastAPI process."
        )


def test_parse_embed_progress_reads_live_counters(tmp_path: Path) -> None:
    """Verify _parse_embed_progress tails the embed log and extracts live counters.

    This is the core fix for BUC-1539: the frontend should see live progress
    counts (embedded, skipped, filtered) WHILE an embed job is running,
    not just at completion.
    """
    from app.routers.index import _parse_embed_progress

    job_id = "test-job-123"
    log_file = tmp_path / f"cis_embed_{job_id}-embed.log"

    # Simulate an active embed pass with multiple PROGRESS lines.
    log_file.write_text(
        "existing content_hashes: 0\n"
        "PROGRESS embedded=100 skipped=0 filtered=1\n"
        "PROGRESS embedded=200 skipped=0 filtered=2\n"
        "PROGRESS embedded=300 skipped=5 filtered=3\n"
    )

    # Mock the log path to use tmp_path
    with patch("app.routers.index.Path") as mock_path:
        mock_path.return_value = log_file
        result = _parse_embed_progress(job_id)

    # Should return the LAST PROGRESS line
    assert result == (300, 5, 3), f"Expected (300, 5, 3), got {result}"


def test_parse_embed_progress_missing_file(tmp_path: Path) -> None:
    """When embed log doesn't exist, return None (job not in embedding phase yet)."""
    from app.routers.index import _parse_embed_progress

    job_id = "nonexistent-job"
    # Don't create a log file, it should return None

    with patch("app.routers.index.Path") as mock_path:
        mock_log = tmp_path / f"cis_embed_{job_id}-embed.log"
        mock_path.return_value = mock_log
        result = _parse_embed_progress(job_id)

    assert result is None


def test_parse_embed_progress_empty_file(tmp_path: Path) -> None:
    """When embed log exists but has no PROGRESS lines, return None."""
    from app.routers.index import _parse_embed_progress

    job_id = "test-job-empty"
    log_file = tmp_path / f"cis_embed_{job_id}-embed.log"
    log_file.write_text("some debug output\nno progress yet\n")

    with patch("app.routers.index.Path") as mock_path:
        mock_path.return_value = log_file
        result = _parse_embed_progress(job_id)

    assert result is None


def test_parse_embed_progress_malformed_line(tmp_path: Path) -> None:
    """When PROGRESS line is malformed, return None instead of crashing."""
    from app.routers.index import _parse_embed_progress

    job_id = "test-job-malformed"
    log_file = tmp_path / f"cis_embed_{job_id}-embed.log"
    log_file.write_text(
        "PROGRESS embedded=100 skipped=0 filtered=1\n"
        "PROGRESS invalid_format_here\n"  # Malformed
    )

    with patch("app.routers.index.Path") as mock_path:
        mock_path.return_value = log_file
        result = _parse_embed_progress(job_id)

    # Should gracefully handle and return None
    assert result is None


def test_get_status_with_live_embed_progress(tmp_path: Path) -> None:
    """Integration: GET /index/{job_id}/status reflects live embed counts from log.

    This tests the complete path where an active embedding job publishes
    live progress to the log file, and the status endpoint reflects it
    WITHOUT updating the database.
    """
    job_id = "test-live-embed-id"

    # Create a real embed log file with live progress
    log_file = tmp_path / f"cis_embed_{job_id}-embed.log"
    log_file.write_text(
        "PROGRESS embedded=1000 skipped=10 filtered=50\n"
        "PROGRESS embedded=1500 skipped=10 filtered=75\n"
    )

    # Mock just the Path constructor for _parse_embed_progress to return our log
    from app.routers.index import _parse_embed_progress

    # Test _parse_embed_progress directly with a temporary override
    original_path = None
    try:
        import app.routers.index as index_module

        # Store original Path and replace temporarily
        original_path = index_module.Path

        # Create a custom Path that intercepts embed log requests
        class TestPath:
            def __init__(self, path_str: str) -> None:
                if "-embed.log" in str(path_str) and job_id in str(path_str):
                    # Return our test log file
                    self.path = str(log_file)
                else:
                    self.path = str(path_str)

            def exists(self) -> bool:
                return Path(self.path).exists()

            def open(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                return Path(self.path).open(*args, **kwargs)

            def __str__(self) -> str:
                return self.path

        index_module.Path = TestPath  # type: ignore[misc]
        result = _parse_embed_progress(job_id)
        assert result is not None, "Should have found PROGRESS lines in log"
        assert result == (1500, 10, 75), f"Expected (1500, 10, 75), got {result}"
    finally:
        if original_path:
            index_module.Path = original_path  # type: ignore[misc]
