"""Tests for GET /index/{job_id}/diff_metrics — Phase 1.4 (BUC-1574)."""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers.index import _Job, _jobs
from app.services import jobs_store

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_jobs() -> None:
    """Isolate tests by resetting both job stores."""
    _jobs.clear()
    jobs_store._reset_for_tests(":memory:")
    yield
    _jobs.clear()


def test_diff_metrics_returns_404_for_unknown_job() -> None:
    """An unknown job_id should produce a clean 404, not a 500."""
    resp = client.get("/index/this-job-does-not-exist/diff_metrics")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_diff_metrics_shape_for_completed_job() -> None:
    """A completed job exposes the persisted final totals + hash_match_rate."""
    job = _Job(job_id="job-completed", repo_path="/tmp/fixture")
    job.status = "done"
    job.phase = "done"
    job.embedded_count = 10
    job.embeddings_skipped_unchanged = 90
    job.embeddings_filtered_out = 5
    job.embed_started_at = time.time() - 12.5
    job.embed_finished_at = job.embed_started_at + 11.0
    _jobs[job.job_id] = job

    resp = client.get(f"/index/{job.job_id}/diff_metrics")
    assert resp.status_code == 200
    body = resp.json()

    assert body["total_symbols"] == 105  # 10 + 90 + 5
    assert body["embedded"] == 10
    assert body["skipped_unchanged"] == 90
    assert body["skipped_filtered"] == 5
    # 90 / (10 + 90) = 0.9
    assert body["hash_match_rate"] == pytest.approx(0.9, abs=1e-4)
    assert body["wall_clock_seconds"] == pytest.approx(11.0, abs=0.1)


def test_diff_metrics_shape_for_running_job() -> None:
    """A running job reports the current PROGRESS-line totals (no log → in-memory)."""
    job = _Job(job_id="job-running", repo_path="/tmp/fixture")
    job.status = "running"
    job.phase = "embedding"
    job.embedded_count = 3
    job.embeddings_skipped_unchanged = 7
    job.embeddings_filtered_out = 1
    job.embed_started_at = time.time() - 2.0
    job.embed_finished_at = None  # still in flight
    _jobs[job.job_id] = job

    resp = client.get(f"/index/{job.job_id}/diff_metrics")
    assert resp.status_code == 200
    body = resp.json()

    assert body["total_symbols"] == 11
    assert body["embedded"] == 3
    assert body["skipped_unchanged"] == 7
    assert body["skipped_filtered"] == 1
    # 7 / (3 + 7) = 0.7
    assert body["hash_match_rate"] == pytest.approx(0.7, abs=1e-4)
    # Running job: wall clock is now - embed_started_at, so > 0
    assert body["wall_clock_seconds"] > 0.0
