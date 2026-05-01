"""Tests for POST /index and GET /index/{job_id}/status."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers.index import _jobs

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_jobs() -> None:
    """Isolate tests by clearing the in-memory job store."""
    _jobs.clear()
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
# Embed-pass lock-conflict regression
# ---------------------------------------------------------------------------


def test_blocking_embed_opens_ladybug_read_only(tmp_path: Path) -> None:
    """The embed subprocess driver MUST open LadybugDB with ``read_only=True``.

    Without this flag the subprocess takes a write lock on the same .db
    file the live indexer is already holding open, causing the embed pass
    to die with ``IO exception: Could not set lock on file: ...`` almost
    immediately after start.  This test pins the contract so a future
    refactor of the driver string can't silently regress it.
    """
    import subprocess
    from unittest.mock import MagicMock, patch

    from app.routers.index import _EmbedJob, _blocking_embed

    from app.routers import index as index_mod

    # Make the configured per-repo .db live under tmp_path so the
    # FileNotFoundError early-exit doesn't trip.
    db_dir = tmp_path
    repo_name = "fakerepo"
    fake_db = Path(index_mod.settings.db_path_for_repo(repo_name))
    # Redirect LADYBUG_DB_DIR to tmp_path; db_path_for_repo recomputes
    # from this attribute on every call so this is enough.
    orig_dir = index_mod.settings.LADYBUG_DB_DIR
    object.__setattr__(index_mod.settings, "LADYBUG_DB_DIR", str(db_dir))
    try:
        fake_db = Path(index_mod.settings.db_path_for_repo(repo_name))
        fake_db.parent.mkdir(parents=True, exist_ok=True)
        fake_db.write_bytes(b"\x00" * 8)

        captured: dict[str, str] = {}

        def fake_run(cmd, *args, **kwargs):  # noqa: ARG001
            captured["driver"] = cmd[2]
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

    driver = captured["driver"]
    # The exact call site the production fix targets.
    assert "lb.Database(" in driver
    assert "read_only=True" in driver, (
        "Embed subprocess must open LadybugDB read-only to avoid lock "
        "conflicts with the live FastAPI process."
    )
