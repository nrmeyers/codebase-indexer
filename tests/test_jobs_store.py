"""Smoke tests for ``app.services.jobs_store``.

Covers the round-trip cases called out in
``.planning/phase-plans/PHASE_2_PERSISTENT_JOBS.md`` §6. The full
restart-recovery integration test lives in ``test_index.py`` and lands
once the router refactor wires the store into the lifespan.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.services import jobs_store


@pytest.fixture
def store_db(tmp_path: Path) -> str:
    """Fresh on-disk DB per test; ``init`` is idempotent so back-to-back calls are fine."""
    db_path = str(tmp_path / "jobs.sqlite")
    jobs_store.init(db_path)
    yield db_path


def test_create_and_get_round_trip(store_db: str) -> None:
    job = jobs_store.create_job(
        kind="index",
        actor_oid="user-a",
        actor_email="a@example.com",
        repo_path="/tmp/repo-a",
        force_reindex=False,
        exclude_paths=frozenset({".venv", "__pycache__"}),
    )
    assert job.status == "running"
    assert job.kind == "index"
    assert job.actor_oid == "user-a"
    assert job.repo_slug == "repo-a"

    got = jobs_store.get_job(job.job_id)
    assert got is not None
    assert got.actor_oid == "user-a"
    assert got.exclude_paths == frozenset({".venv", "__pycache__"})


def test_update_progress_is_partial_update(store_db: str) -> None:
    job = jobs_store.create_job(
        kind="index",
        actor_oid="u",
        actor_email="u@example.com",
        repo_path="/tmp/r",
        force_reindex=False,
        exclude_paths=frozenset(),
    )
    jobs_store.update_progress(job.job_id, phase="parsing", progress_pct=25.0, files_total=10, files_done=2)
    after_first = jobs_store.get_job(job.job_id)
    assert after_first is not None
    assert after_first.phase == "parsing"
    assert after_first.files_total == 10

    # Subsequent partial update should not clobber unset fields.
    jobs_store.update_progress(job.job_id, files_done=5)
    after_second = jobs_store.get_job(job.job_id)
    assert after_second is not None
    assert after_second.phase == "parsing"  # preserved
    assert after_second.files_total == 10  # preserved
    assert after_second.files_done == 5  # updated


def test_mark_done_is_terminal(store_db: str) -> None:
    job = jobs_store.create_job(
        kind="index",
        actor_oid="u",
        actor_email="u@x",
        repo_path="/tmp/r",
        force_reindex=False,
        exclude_paths=frozenset(),
    )
    jobs_store.mark_done(job.job_id, node_count=100, rel_count=200, embedding_count=50)
    got = jobs_store.get_job(job.job_id)
    assert got is not None
    assert got.status == "done"
    assert got.node_count == 100
    assert got.embedding_count == 50
    assert got.finished_at is not None


def test_find_active_for_repo_ignores_terminal(store_db: str) -> None:
    j1 = jobs_store.create_job(
        kind="index", actor_oid="u", actor_email="u@x",
        repo_path="/tmp/foo", force_reindex=False, exclude_paths=frozenset(),
    )
    jobs_store.mark_done(j1.job_id, node_count=1, rel_count=1, embedding_count=1)
    # No active job for this repo any more.
    assert jobs_store.find_active_for_repo("foo") is None

    j2 = jobs_store.create_job(
        kind="index", actor_oid="u", actor_email="u@x",
        repo_path="/tmp/foo", force_reindex=False, exclude_paths=frozenset(),
    )
    active = jobs_store.find_active_for_repo("foo")
    assert active is not None
    assert active.job_id == j2.job_id


def test_cross_user_isolation(store_db: str) -> None:
    a = jobs_store.create_job(
        kind="index", actor_oid="user-a", actor_email="a@x",
        repo_path="/tmp/repo1", force_reindex=False, exclude_paths=frozenset(),
    )
    b = jobs_store.create_job(
        kind="index", actor_oid="user-b", actor_email="b@x",
        repo_path="/tmp/repo2", force_reindex=False, exclude_paths=frozenset(),
    )
    a_jobs = jobs_store.list_jobs(actor_oid="user-a")
    b_jobs = jobs_store.list_jobs(actor_oid="user-b")
    a_ids = {j.job_id for j in a_jobs}
    b_ids = {j.job_id for j in b_jobs}
    assert a.job_id in a_ids
    assert b.job_id not in a_ids
    assert b.job_id in b_ids
    assert a.job_id not in b_ids


def test_sweep_interrupted_only_touches_other_workers(store_db: str) -> None:
    # Job created by "this" worker (current WORKER_TOKEN).
    own = jobs_store.create_job(
        kind="index", actor_oid="u", actor_email="u@x",
        repo_path="/tmp/own", force_reindex=False, exclude_paths=frozenset(),
    )
    # Sweep with a fresh token — simulates a restart where this same row
    # was orphaned by a now-dead worker. The "fresh" token is OUR token
    # (since jobs_store.WORKER_TOKEN is module-level). The behaviour we
    # want: rows whose worker_token != caller's get flipped to interrupted.
    interrupted_count = jobs_store.sweep_interrupted("a-different-token")
    assert interrupted_count >= 1  # the `own` row had the module token

    after = jobs_store.get_job(own.job_id)
    assert after is not None
    assert after.status == "interrupted"
    assert after.error == "service restart"


def test_cancel_request_round_trip(store_db: str) -> None:
    job = jobs_store.create_job(
        kind="index", actor_oid="u", actor_email="u@x",
        repo_path="/tmp/r", force_reindex=False, exclude_paths=frozenset(),
    )
    assert jobs_store.is_cancel_requested(job.job_id) is False
    requested = jobs_store.request_cancel(job.job_id)
    assert requested is True
    assert jobs_store.is_cancel_requested(job.job_id) is True
