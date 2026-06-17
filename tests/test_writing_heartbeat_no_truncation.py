"""Regression: the writing-phase liveness heartbeat must NOT truncate the graph write.

Root-cause guard for the "reindex reports status=done but the graph holds only
a few hundred nodes for a repo with thousands of symbols" bug. The writing-phase
heartbeat (``app.routers.index._writing_phase_heartbeat``) runs a daemon thread
concurrently with the blocking Kùzu bulk write. Kùzu connections are NOT
thread-safe, so the heartbeat must touch ONLY the separate jobs_store sqlite
connection (``touch_heartbeat``) and never the graph connection / ingestor.

These tests ingest N nodes WHILE the real heartbeat thread ticks against a real
jobs_store, then assert all N persist — i.e. the liveness mechanism does not race
or truncate the write.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest


@pytest.fixture()
def migrated_db(tmp_path: Path) -> str:
    from app.services.ladybug_schema import migrate

    db_path = str(tmp_path / "graph.db")
    migrate(db_path)
    return db_path


@pytest.fixture()
def jobs_store(tmp_path: Path):
    """Initialise the durable jobs_store against an isolated sqlite file."""
    from app.services import jobs_store as js

    js.init(str(tmp_path / "jobs.db"))
    try:
        yield js
    finally:
        js._reset_for_tests()


def _make_running_writing_job(js, job_id: str, repo_slug: str):
    """Create a durable job and flip it into the 'writing' running phase."""
    js.create_job(
        job_id=job_id,
        actor_oid="test-actor",
        actor_email="test@local",
        repo_path=f"/tmp/{repo_slug}",
        force_reindex=True,
    )
    js.update_progress(job_id, phase="writing", progress_pct=50.0)


def _count_nodes(db_path: str) -> int:
    import ladybug as lb

    db = lb.Database(db_path)
    conn = lb.Connection(db)
    result = conn.execute("MATCH (n) RETURN count(n)")
    rows = []
    while result.has_next():
        rows.append(result.get_next())
    conn.close()
    return int(rows[0][0]) if rows else 0


def test_concurrent_heartbeat_does_not_truncate_node_write(
    migrated_db: str, jobs_store
) -> None:
    """N Function nodes must all persist while the heartbeat thread ticks.

    Reproduces the prime hypothesis: a daemon liveness thread racing the bulk
    Kùzu write. The heartbeat must operate only on the jobs_store sqlite
    connection, leaving the graph write intact.
    """
    from app.routers.index import _Job, _writing_phase_heartbeat
    from app.services.ladybug_ingestor import LadybugIngestor

    n_nodes = 5000
    job_id = "hb-trunc-test"
    _make_running_writing_job(jobs_store, job_id, "trunc-repo")

    job = _Job(job_id=job_id, repo_path="/tmp/trunc-repo")
    job.status = "running"
    job.phase = "writing"
    job.last_progress_at = time.time()

    # Tick aggressively (every 1ms) to maximise contention against the write.
    with _writing_phase_heartbeat(job, interval_seconds=0.001):
        with LadybugIngestor(migrated_db, batch_size=500) as ingestor:
            for i in range(n_nodes):
                ingestor.ensure_node_batch(
                    "Function",
                    {
                        "qualified_name": f"pkg.mod.func_{i}",
                        "name": f"func_{i}",
                    },
                )
            ingestor.flush_all()
            assert ingestor.node_count == n_nodes, (
                f"ingestor counted {ingestor.node_count}, expected {n_nodes}"
            )

    persisted = _count_nodes(migrated_db)
    assert persisted == n_nodes, (
        f"graph truncated: {persisted} nodes persisted, expected {n_nodes} "
        f"(heartbeat raced the Kùzu write)"
    )


def test_touch_heartbeat_only_bumps_jobs_store_not_graph(jobs_store) -> None:
    """touch_heartbeat must mutate only the jobs_store updated_at, on its own conn."""
    job_id = "hb-isolation"
    _make_running_writing_job(jobs_store, job_id, "iso-repo")

    before = jobs_store.get_job(job_id)
    time.sleep(0.01)
    jobs_store.touch_heartbeat(job_id)
    after = jobs_store.get_job(job_id)

    # updated_at advanced; phase + progress untouched (owned by real callback).
    assert after.updated_at >= before.updated_at
    assert after.phase == "writing"
    assert after.progress_pct == 50.0


def test_touch_heartbeat_noop_on_terminal_job(jobs_store) -> None:
    """A done/failed job has no liveness clock to advance — touch is a no-op."""
    job_id = "hb-terminal"
    jobs_store.create_job(
        job_id=job_id,
        actor_oid="a",
        actor_email="a@local",
        repo_path="/tmp/term-repo",
        force_reindex=True,
    )
    jobs_store.mark_done(job_id, node_count=1, rel_count=0, embedding_count=0)
    before = jobs_store.get_job(job_id)
    jobs_store.touch_heartbeat(job_id)  # must not raise, must not resurrect
    after = jobs_store.get_job(job_id)
    assert after.status == before.status == "done"


def test_many_heartbeat_threads_dont_corrupt_write(
    migrated_db: str, jobs_store
) -> None:
    """Stress: several concurrent liveness threads against one ongoing write."""
    from app.routers.index import _Job, _writing_phase_heartbeat
    from app.services.ladybug_ingestor import LadybugIngestor

    n_nodes = 3000
    job_id = "hb-stress"
    _make_running_writing_job(jobs_store, job_id, "stress-repo")
    job = _Job(job_id=job_id, repo_path="/tmp/stress-repo")
    job.status = "running"
    job.phase = "writing"
    job.last_progress_at = time.time()

    stop = threading.Event()

    def hammer() -> None:
        while not stop.is_set():
            jobs_store.touch_heartbeat(job_id)
            time.sleep(0.0005)

    hammers = [threading.Thread(target=hammer, daemon=True) for _ in range(4)]
    for h in hammers:
        h.start()
    try:
        with _writing_phase_heartbeat(job, interval_seconds=0.001):
            with LadybugIngestor(migrated_db, batch_size=400) as ingestor:
                for i in range(n_nodes):
                    ingestor.ensure_node_batch(
                        "Function",
                        {"qualified_name": f"s.func_{i}", "name": f"func_{i}"},
                    )
                ingestor.flush_all()
    finally:
        stop.set()
        for h in hammers:
            h.join(timeout=2.0)

    assert _count_nodes(migrated_db) == n_nodes
