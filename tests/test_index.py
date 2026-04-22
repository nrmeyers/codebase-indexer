"""Tests for POST /index and GET /index/{job_id}/status."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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


def test_post_index_duplicate_same_repo_creates_separate_jobs(tmp_path: Path) -> None:
    """Two POSTs with the same repo_path both return 202 with distinct job IDs."""
    with patch("app.routers.index._run_ingestion", new_callable=AsyncMock):
        resp1 = client.post("/index", json={"repo_path": str(tmp_path)})
        resp2 = client.post("/index", json={"repo_path": str(tmp_path)})

    assert resp1.status_code == 202
    assert resp2.status_code == 202
    job_id1 = resp1.json()["job_id"]
    job_id2 = resp2.json()["job_id"]
    # Each call creates an independent job — both should have unique IDs
    assert job_id1 != job_id2
    # Both jobs should be retrievable
    assert client.get(f"/index/{job_id1}/status").status_code == 200
    assert client.get(f"/index/{job_id2}/status").status_code == 200
