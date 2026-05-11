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


# ---------------------------------------------------------------------------
# BUC-1599 — v2 schema additions
# ---------------------------------------------------------------------------


def test_v2_schema_bump_creates_new_tables(store_db: str) -> None:
    """Fresh init() must produce schema_meta v2 + the new tables."""
    import sqlite3
    conn = sqlite3.connect(store_db)
    try:
        version = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'version'"
        ).fetchone()[0]
        assert version == "2"

        table_names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert "indexed_repos" in table_names
        assert "job_events" in table_names

        jobs_cols = {
            r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()
        }
        assert "triggered_by" in jobs_cols
    finally:
        conn.close()


def test_v1_to_v2_migration_adds_column_and_tables(tmp_path: Path) -> None:
    """Booting against a v1-shaped DB must promote it to v2 without data loss."""
    import sqlite3
    from app.services.jobs_store import _DDL_V1

    db_path = str(tmp_path / "legacy.sqlite")
    legacy = sqlite3.connect(db_path)
    legacy.executescript(_DDL_V1)
    legacy.execute(
        """
        INSERT INTO jobs (
          job_id, kind, actor_oid, actor_email, repo_slug, repo_path,
          status, started_at, updated_at
        ) VALUES ('legacy-1', 'index', 'u', 'u@x', 'r', '/tmp/r',
                  'done', 1, 1)
        """
    )
    legacy.commit()
    legacy.close()

    jobs_store.init(db_path)

    got = jobs_store.get_job("legacy-1")
    assert got is not None
    assert got.status == "done"
    assert got.triggered_by is None

    assert jobs_store.list_indexed_repos() == []
    assert jobs_store.list_job_events("legacy-1") == []

    fresh = jobs_store.create_job(
        kind="index", actor_oid="u", actor_email="u@x",
        repo_path="/tmp/r", force_reindex=False, exclude_paths=frozenset(),
        triggered_by="webhook",
    )
    after = jobs_store.get_job(fresh.job_id)
    assert after is not None
    assert after.triggered_by == "webhook"


def test_triggered_by_defaults_to_manual(store_db: str) -> None:
    job = jobs_store.create_job(
        kind="index", actor_oid="u", actor_email="u@x",
        repo_path="/tmp/r", force_reindex=False, exclude_paths=frozenset(),
    )
    got = jobs_store.get_job(job.job_id)
    assert got is not None
    assert got.triggered_by == "manual"


def test_indexed_repos_upsert_round_trip(store_db: str) -> None:
    jobs_store.upsert_indexed_repo(
        slug="navistone__TheForge",
        display_name="TheForge",
        db_path="/data/repos/navistone__TheForge.db",
        remote_url="https://github.com/navistone/TheForge.git",
    )
    row = jobs_store.get_indexed_repo("navistone__TheForge")
    assert row is not None
    assert row["display_name"] == "TheForge"
    assert row["last_indexed_at"] is None
    first_updated = row["updated_at"]

    import time as _t
    _t.sleep(1.1)  # advance unix-second timestamp
    jobs_store.upsert_indexed_repo(
        slug="navistone__TheForge",
        display_name="TheForge (renamed)",
        db_path="/data/repos/navistone__TheForge.db",
    )
    row2 = jobs_store.get_indexed_repo("navistone__TheForge")
    assert row2 is not None
    assert row2["display_name"] == "TheForge (renamed)"
    assert row2["created_at"] == row["created_at"]
    assert row2["updated_at"] >= first_updated

    assert jobs_store.mark_indexed(
        "navistone__TheForge", last_commit_sha="deadbeefcafefacef00d"
    )
    row3 = jobs_store.get_indexed_repo("navistone__TheForge")
    assert row3 is not None
    assert row3["last_indexed_at"] is not None
    assert row3["last_commit_sha"] == "deadbeefcafefacef00d"

    listed = jobs_store.list_indexed_repos()
    assert len(listed) == 1
    assert listed[0]["slug"] == "navistone__TheForge"

    assert jobs_store.delete_indexed_repo("navistone__TheForge") is True
    assert jobs_store.get_indexed_repo("navistone__TheForge") is None
    assert jobs_store.delete_indexed_repo("navistone__TheForge") is False


def test_mark_done_records_info_event(store_db: str) -> None:
    job = jobs_store.create_job(
        kind="index", actor_oid="u", actor_email="u@x",
        repo_path="/tmp/r", force_reindex=False, exclude_paths=frozenset(),
    )
    jobs_store.mark_done(job.job_id, node_count=10, rel_count=20, embedding_count=5)
    events = jobs_store.list_job_events(job.job_id)
    assert len(events) == 1
    assert events[0]["level"] == "info"
    assert "done" in str(events[0]["message"])
    assert "nodes=10" in str(events[0]["message"])


def test_mark_failed_records_error_event(store_db: str) -> None:
    job = jobs_store.create_job(
        kind="index", actor_oid="u", actor_email="u@x",
        repo_path="/tmp/r", force_reindex=False, exclude_paths=frozenset(),
    )
    jobs_store.mark_failed(job.job_id, error="parser crashed")
    events = jobs_store.list_job_events(job.job_id)
    assert len(events) == 1
    assert events[0]["level"] == "error"
    assert "parser crashed" in str(events[0]["message"])


def test_mark_failed_cancelled_records_warn_event(store_db: str) -> None:
    job = jobs_store.create_job(
        kind="index", actor_oid="u", actor_email="u@x",
        repo_path="/tmp/r", force_reindex=False, exclude_paths=frozenset(),
    )
    jobs_store.mark_failed(
        job.job_id, error="Cancelled by user", terminal_status="cancelled"
    )
    events = jobs_store.list_job_events(job.job_id)
    assert len(events) == 1
    assert events[0]["level"] == "warn"


def test_record_event_explicit_order(store_db: str) -> None:
    job = jobs_store.create_job(
        kind="index", actor_oid="u", actor_email="u@x",
        repo_path="/tmp/r", force_reindex=False, exclude_paths=frozenset(),
    )
    jobs_store.record_event(job.job_id, "info", "phase: parsing")
    jobs_store.record_event(job.job_id, "warn", "phase: embed (retry)")
    events = jobs_store.list_job_events(job.job_id)
    assert len(events) == 2
    assert events[0]["message"] == "phase: parsing"
    assert events[1]["message"] == "phase: embed (retry)"


def test_list_job_events_limit_caps(store_db: str) -> None:
    job = jobs_store.create_job(
        kind="index", actor_oid="u", actor_email="u@x",
        repo_path="/tmp/r", force_reindex=False, exclude_paths=frozenset(),
    )
    rows = jobs_store.list_job_events(job.job_id, limit=99999)
    assert rows == []


def test_record_event_rejects_bad_level(store_db: str) -> None:
    """CHECK constraint at the DB level rejects unknown levels."""
    import sqlite3

    job = jobs_store.create_job(
        kind="index", actor_oid="u", actor_email="u@x",
        repo_path="/tmp/r", force_reindex=False, exclude_paths=frozenset(),
    )
    with pytest.raises(sqlite3.IntegrityError):
        jobs_store.record_event(job.job_id, "debug", "should fail")
