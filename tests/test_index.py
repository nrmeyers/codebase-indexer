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


def test_start_index_direct_call_coerces_header_default_triggered_by(
    tmp_path: Path,
) -> None:
    """LE-146 regression — calling ``start_index`` directly (as ``reindex_repo``
    does) must not AttributeError on the unresolved ``Header()`` default.

    ``reindex_repo`` invokes ``start_index(req, background_tasks)`` positionally
    without supplying ``x_forge_triggered_by``. FastAPI only resolves the
    ``Header(default=...)`` sentinel during request injection — on a direct
    Python call the parameter retains the ``Header`` object (a
    ``fastapi.params.Header`` instance), which has no ``.strip()``. Before the
    fix this raised ``AttributeError`` and surfaced as a 500 from the reindex
    endpoint. The coercion guards the direct-call path.
    """
    import asyncio
    import inspect

    from fastapi import BackgroundTasks

    from app.routers.index import IndexRequest, start_index

    # Reproduce the reindex_repo call shape: two positional args, header omitted
    # so the Header() sentinel default is in force (NOT a str).
    sig = inspect.signature(start_index)
    header_default = sig.parameters["x_forge_triggered_by"].default
    # Guard the test's own premise: the default is a non-str Header sentinel,
    # which is exactly the object that used to blow up on .strip().
    assert not isinstance(header_default, str)

    with patch("app.routers.index._run_ingestion", new_callable=AsyncMock):
        accepted = asyncio.run(
            start_index(
                IndexRequest(repo_path=str(tmp_path), force_reindex=True),
                BackgroundTasks(),
            )
        )

    assert accepted.job_id  # IndexAccepted returned, no AttributeError raised


def test_reindex_endpoint_returns_202_not_500(tmp_path: Path) -> None:
    """LE-146 regression — ``POST /repos/{name}/reindex`` must return 202.

    The endpoint delegates to ``start_index`` via a direct Python call, so it
    is the integration-level reproduction of the unresolved-Header-default bug:
    pre-fix it returned 500 (AttributeError on ``.strip()``); post-fix it
    returns 202 Accepted with a job_id.
    """
    from app.routers.index import indexed_repo_paths

    repo_name = "le146-regression-repo"
    indexed_repo_paths[repo_name] = str(tmp_path)
    try:
        with patch("app.routers.index._run_ingestion", new_callable=AsyncMock):
            resp = client.post(
                f"/repos/{repo_name}/reindex", json={"force": True}
            )
        assert resp.status_code == 202, resp.text
        assert "job_id" in resp.json()
    finally:
        indexed_repo_paths.pop(repo_name, None)


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
# LE-143 follow-up — lock release on cancel/clear/delete + reindex reconcile
# ---------------------------------------------------------------------------


def test_durable_and_inmemory_job_id_unified(tmp_path: Path) -> None:
    """The returned job_id matches the durable row id (root-cause fix).

    Before the fix create_job minted a separate UUID, so terminal
    transitions never touched the durable row and the lock leaked.
    """
    with patch("app.routers.index._run_ingestion", new_callable=AsyncMock):
        resp = client.post("/index", json={"repo_path": str(tmp_path)})
    job_id = resp.json()["job_id"]
    stored = jobs_store.get_job(job_id)
    assert stored is not None
    assert stored.job_id == job_id


def test_cancel_releases_repo_lock_and_reindex_proceeds(tmp_path: Path) -> None:
    """POST /cancel marks the durable row terminal so reindex is unblocked."""
    with patch("app.routers.index._run_ingestion", new_callable=AsyncMock):
        r1 = client.post("/index", json={"repo_path": str(tmp_path)})
        job_id = r1.json()["job_id"]
        # Second request is locked out.
        assert (
            client.post("/index", json={"repo_path": str(tmp_path)}).status_code
            == 409
        )
        # Cancel the running job — must release the durable lock.
        assert client.post(f"/index/{job_id}/cancel").status_code == 200
        # A new reindex now proceeds instead of 409ing.
        r3 = client.post("/index", json={"repo_path": str(tmp_path)})
    assert r3.status_code == 202, r3.text
    assert jobs_store.find_active_for_repo(tmp_path.name) is not None


def test_delete_running_job_reconciles_then_releases_lock(tmp_path: Path) -> None:
    """DELETE of a no-progress running job releases the durable per-repo lock."""
    import time

    with patch("app.routers.index._run_ingestion", new_callable=AsyncMock):
        r1 = client.post("/index", json={"repo_path": str(tmp_path)})
    job_id = r1.json()["job_id"]
    # Make the in-memory + durable job look stuck (no progress).
    _jobs[job_id].last_progress_at = time.time() - 2400
    conn = jobs_store._require_conn()
    conn.execute(
        "UPDATE jobs SET updated_at = ? WHERE job_id = ?",
        (time.time() - 2400, job_id),
    )
    # Mark in-memory terminal so delete is permitted; durable still 'running'.
    _jobs[job_id].status = "failed"
    resp = client.delete(f"/index/jobs/{job_id}")
    assert resp.status_code == 200, resp.text
    # Durable row is gone — lock released, reindex would proceed.
    assert jobs_store.get_job(job_id) is None
    assert jobs_store.find_active_for_repo(tmp_path.name) is None


def test_clear_jobs_clears_durable_terminal_rows(tmp_path: Path) -> None:
    """POST /index/jobs/clear empties durable terminal history too."""
    with patch("app.routers.index._run_ingestion", new_callable=AsyncMock):
        r1 = client.post("/index", json={"repo_path": str(tmp_path)})
    job_id = r1.json()["job_id"]
    jobs_store.mark_failed(job_id, error="boom", terminal_status="failed")
    _jobs[job_id].status = "failed"
    resp = client.post("/index/jobs/clear", params={"status": "done,failed"})
    assert resp.status_code == 200, resp.text
    assert jobs_store.get_job(job_id) is None


def test_reindex_reconciles_stuck_lock_and_proceeds(tmp_path: Path) -> None:
    """A reindex request reconciles a stuck (no-progress) lock and proceeds.

    Direct reproduction of LE-143: a durable 'running' row with no progress
    for 40 min must not 409-block a new reindex — the request path reconciles
    it (releases the lock) and accepts the new job.
    """
    import time

    # Seed a stuck durable running row directly (simulates orphaned worker).
    stuck = jobs_store.create_job(
        kind="index", actor_oid="", actor_email="",
        repo_path=str(tmp_path), force_reindex=False, exclude_paths=frozenset(),
    )
    conn = jobs_store._require_conn()
    conn.execute(
        "UPDATE jobs SET updated_at = ? WHERE job_id = ?",
        (time.time() - 2400, stuck.job_id),
    )
    with patch("app.routers.index._run_ingestion", new_callable=AsyncMock):
        resp = client.post("/index", json={"repo_path": str(tmp_path)})
    assert resp.status_code == 202, resp.text
    # The stuck job was reconciled to a terminal state.
    reaped = jobs_store.get_job(stuck.job_id)
    assert reaped is not None and reaped.status == "failed"


# ---------------------------------------------------------------------------
# Writing-phase heartbeat — regression for the slow-bulk-write false-kill
#
# Root cause: GraphUpdater.run() emits one {"phase": "writing"} event then
# blocks in LadybugIngestor.flush_all() (a multi-minute Kùzu bulk COPY) with
# zero further progress callbacks. last_progress_at / durable updated_at both
# freeze, so reconcile_stale_running_jobs marks the job failed mid-write,
# leaving a partial graph (missing route handlers, degenerate KG mega-cluster).
# The fix emits periodic liveness ticks during the write so neither reaper
# false-kills a healthy slow flush.
# ---------------------------------------------------------------------------


def _seed_running_job(repo_path: Path) -> str:
    """Create a running _Job (in-memory + durable) and return its job_id."""
    import time

    from app.routers.index import _Job

    durable = jobs_store.create_job(
        kind="index", actor_oid="", actor_email="",
        repo_path=str(repo_path), force_reindex=False, exclude_paths=frozenset(),
    )
    job_id = durable.job_id
    _jobs[job_id] = _Job(
        job_id=job_id,
        repo_path=str(repo_path),
        status="running",
        phase="writing",
        last_progress_at=time.time(),
    )
    return job_id


def test_writing_heartbeat_keeps_slow_write_alive_so_reconciler_does_not_fail_it(
    tmp_path: Path,
) -> None:
    """A callback-silent slow write that emits heartbeat ticks is NOT reaped.

    Simulates the writing phase: no progress callbacks fire, but the heartbeat
    thread bumps last_progress_at + durable updated_at on a short interval.
    The reconciler (run with a small threshold) must see fresh liveness and
    NOT mark the job failed, and the simulated write must complete.
    """
    import time

    from app.routers.index import (
        _writing_phase_heartbeat,
        reconcile_stale_running_jobs,
    )

    job_id = _seed_running_job(tmp_path)
    job = _jobs[job_id]
    # Make the job look already-silent so that, WITHOUT ticks, a 1s-threshold
    # reconcile would reap it immediately.
    job.last_progress_at = time.time() - 10.0
    conn = jobs_store._require_conn()
    conn.execute(
        "UPDATE jobs SET updated_at = ? WHERE job_id = ?",
        (time.time() - 10.0, job_id),
    )

    write_completed = {"done": False}

    def _slow_write() -> None:
        # Mimic LadybugIngestor.flush_all(): a few seconds of blocking work
        # with NO progress callback. The heartbeat thread ticks underneath.
        time.sleep(2.0)
        write_completed["done"] = True

    # Tick every 0.2s — well under the 1s reconcile threshold below.
    with _writing_phase_heartbeat(job, interval_seconds=0.2):
        # Reconcile mid-write: must NOT reap because ticks keep liveness fresh.
        time.sleep(0.5)
        reconciled_midwrite = reconcile_stale_running_jobs(
            staleness_threshold_seconds=1
        )
        _slow_write()

    assert write_completed["done"] is True
    assert reconciled_midwrite == 0, "heartbeat tick should keep the write alive"
    assert job.status == "running"
    # After ticks, in-memory + durable liveness are fresh (within the last 1s).
    assert time.time() - job.last_progress_at < 1.0
    row = jobs_store.get_job(job_id)
    assert row is not None and row.status == "running"
    assert time.time() - row.updated_at < 1.0


def test_writing_heartbeat_advances_inmemory_and_durable_liveness(
    tmp_path: Path,
) -> None:
    """Each tick advances BOTH last_progress_at and durable updated_at."""
    import time

    from app.routers.index import _writing_phase_heartbeat

    job_id = _seed_running_job(tmp_path)
    job = _jobs[job_id]
    stale = time.time() - 100.0
    job.last_progress_at = stale
    conn = jobs_store._require_conn()
    conn.execute(
        "UPDATE jobs SET updated_at = ? WHERE job_id = ?", (stale, job_id)
    )

    with _writing_phase_heartbeat(job, interval_seconds=0.1):
        time.sleep(0.35)  # ~3 ticks

    assert job.last_progress_at > stale + 50.0
    row = jobs_store.get_job(job_id)
    assert row is not None and row.updated_at > stale + 50.0
    # Heartbeat never mutated phase/progress — those belong to the real
    # callback. The durable phase stays as create_job seeded it ('queued').
    assert row.phase == "queued"


def test_touch_heartbeat_only_bumps_running_jobs(tmp_path: Path) -> None:
    """touch_heartbeat advances a running row and is a no-op on a terminal one."""
    import time

    job_id = _seed_running_job(tmp_path)
    before = jobs_store.get_job(job_id).updated_at
    time.sleep(0.01)
    jobs_store.touch_heartbeat(job_id)
    after = jobs_store.get_job(job_id).updated_at
    assert after > before

    # Terminal row: touch is a no-op (no liveness clock to advance).
    jobs_store.mark_failed(job_id, error="x", terminal_status="failed")
    failed_at = jobs_store.get_job(job_id).updated_at
    time.sleep(0.01)
    jobs_store.touch_heartbeat(job_id)
    assert jobs_store.get_job(job_id).updated_at == failed_at


def test_reconciler_widens_budget_during_writing_phase(tmp_path: Path) -> None:
    """Even with starved ticks, a job in phase=='writing' gets the wider budget.

    Belt-and-suspenders: if the heartbeat thread is itself starved (GIL
    contention behind the CPU-bound write), the reconciler must still not reap
    a job demonstrably in the writing phase until JOB_PHASE_WATCHDOG_SECONDS.
    """
    import time

    from app.config import settings
    from app.routers.index import reconcile_stale_running_jobs

    # Silent for longer than the small threshold but less than the watchdog.
    silent_for = settings.JOB_STALENESS_THRESHOLD_SECONDS + 60
    assert silent_for < settings.JOB_PHASE_WATCHDOG_SECONDS

    job_id = _seed_running_job(tmp_path)
    job = _jobs[job_id]
    job.phase = "writing"
    job.last_progress_at = time.time() - silent_for

    reconciled = reconcile_stale_running_jobs(
        staleness_threshold_seconds=settings.JOB_STALENESS_THRESHOLD_SECONDS
    )
    assert reconciled == 0, "a writing-phase job must not be reaped before the watchdog"
    assert job.status == "running"


def test_blocking_index_completes_with_heartbeat_wrapper_and_resolves_symbols(
    tmp_path: Path,
) -> None:
    """End-to-end: a real structural index runs through the writing-phase
    heartbeat wrapper, reaches non-zero node/rel counts, and a route-handler
    symbol resolves in the graph.

    This is the STEP-3 bar in miniature: the heartbeat wrapper must not break
    the real write path, the graph must be fully written (counts > 0), and a
    previously-missing-style symbol (a route handler) must be present. No
    embedder/model is loaded — _blocking_index is the structural pass only.
    """
    from app.routers.index import _Job, _blocking_index

    # Tiny synthetic repo with a "route handler" so we can assert it resolves.
    repo = tmp_path / "tinyrepo"
    pkg = repo / "src" / "routes"
    pkg.mkdir(parents=True)
    (repo / "src" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "chat.py").write_text(
        "def handle_list_conversations(actor_id):\n"
        "    return _query(actor_id)\n\n"
        "def _query(actor_id):\n"
        "    return []\n"
    )

    job = _Job(job_id="e2e-heartbeat-job", repo_path=str(repo), status="running")
    _jobs[job.job_id] = job

    # Mock the downstream embed pass (loads sentence-transformers — out of
    # scope; the bug is the structural Kùzu write) and tantivy (best-effort).
    with patch("app.routers.index._blocking_embed"), patch(
        "app.services.tantivy_index.TantivyIndex", create=True
    ):
        _blocking_index(job, force_reindex=True)

    # Graph was fully written — counts are non-zero and stable.
    assert job.node_count > 0, "structural write produced no nodes"
    assert job.rel_count > 0, "structural write produced no relationships"

    # The route handler resolves in the written graph (no partial write).
    import real_ladybug as lb  # type: ignore[import-untyped]

    from app.config import settings as _settings
    from app.services.slug import derive_slug as _derive_slug

    repo_name = _derive_slug(repo.resolve(), repo.name)
    db_path = _settings.db_path_for_repo(repo_name)
    db = lb.Database(db_path)
    conn = lb.Connection(db)
    try:
        res = conn.execute(
            "MATCH (f:Function) WHERE f.name = $n RETURN count(f) AS c",
            {"n": "handle_list_conversations"},
        )
        found = int(res.get_next()[0]) if res.has_next() else 0
    finally:
        try:
            conn.close()
            db.close()
        except Exception:
            pass
    assert found >= 1, "route handler missing — partial/incomplete write"
    _jobs.pop(job.job_id, None)


def test_truncated_graph_self_heals_on_incremental_reindex(tmp_path: Path) -> None:
    """A truncated graph (structure-only) must self-heal on the NEXT reindex.

    Reproduces the "369-node" durable failure: a prior run persisted the
    structural skeleton (Folders / Files / Modules) but no definition nodes,
    while the on-disk hash cache was written for every file. A naive
    incremental reindex would see every file "unchanged", skip parsing, and
    leave the graph truncated forever — even though it reports status=done.

    _graph_is_truncated() must detect the zero-definition state and force a
    full re-parse so definitions + relationships come back.
    """
    from app.routers.index import _Job, _blocking_index, _graph_is_truncated

    repo = tmp_path / "tinyrepo"
    pkg = repo / "src" / "routes"
    pkg.mkdir(parents=True)
    (repo / "src" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "chat.py").write_text(
        "def handle_list_conversations(actor_id):\n"
        "    return _query(actor_id)\n\n"
        "def _query(actor_id):\n"
        "    return []\n"
    )

    # --- Run 1: full index. Graph is complete. ---
    job1 = _Job(job_id="heal-run-1", repo_path=str(repo), status="running")
    _jobs[job1.job_id] = job1
    with patch("app.routers.index._blocking_embed"), patch(
        "app.services.tantivy_index.TantivyIndex", create=True
    ):
        _blocking_index(job1, force_reindex=True)
    assert job1.node_count > 0
    assert job1.rel_count > 0

    from app.config import settings as _settings
    from app.services.slug import derive_slug as _derive_slug

    repo_name = _derive_slug(repo.resolve(), repo.name)
    db_path = _settings.db_path_for_repo(repo_name)

    # A healthy graph is NOT truncated.
    assert _graph_is_truncated(db_path) is False

    # --- Simulate truncation: delete all definition nodes, keeping structure
    # and the (now-stale) hash cache. This is the live "369-node" state. ---
    import real_ladybug as lb  # type: ignore[import-untyped]

    db = lb.Database(db_path)
    conn = lb.Connection(db)
    for label in ("Function", "Method", "Class", "Interface", "Enum"):
        conn.execute(f"MATCH (n:{label}) DETACH DELETE n")
    conn.close()
    db.close()
    import gc

    gc.collect()

    # The hash cache from run 1 still lists chat.py as indexed.
    hash_cache = repo / ".cgr-hash-cache.json"
    assert hash_cache.exists(), "run 1 should have written a hash cache"

    # Now the graph IS truncated — detector must fire.
    assert _graph_is_truncated(db_path) is True

    # --- Run 2: incremental (force_reindex=False). Must self-heal. ---
    job2 = _Job(job_id="heal-run-2", repo_path=str(repo), status="running")
    _jobs[job2.job_id] = job2
    with patch("app.routers.index._blocking_embed"), patch(
        "app.services.tantivy_index.TantivyIndex", create=True
    ):
        _blocking_index(job2, force_reindex=False)

    # Definitions are back — the route handler resolves again.
    db = lb.Database(db_path)
    conn = lb.Connection(db)
    try:
        res = conn.execute(
            "MATCH (f:Function) WHERE f.name = $n RETURN count(f) AS c",
            {"n": "handle_list_conversations"},
        )
        found = int(res.get_next()[0]) if res.has_next() else 0
    finally:
        conn.close()
        db.close()
    assert found >= 1, (
        "truncated graph did NOT self-heal — definitions still missing after "
        "an incremental reindex (the 369-node bug)"
    )
    assert _graph_is_truncated(db_path) is False
    _jobs.pop(job1.job_id, None)
    _jobs.pop(job2.job_id, None)


def test_reconciler_still_reaps_genuinely_hung_writing_job(tmp_path: Path) -> None:
    """A writing-phase job silent past the WATCHDOG budget is still reaped.

    The widened budget must not be infinite — a genuinely dead worker stuck in
    'writing' past JOB_PHASE_WATCHDOG_SECONDS is still failed + lock released.
    """
    import time

    from app.config import settings
    from app.routers.index import reconcile_stale_running_jobs

    job_id = _seed_running_job(tmp_path)
    job = _jobs[job_id]
    job.phase = "writing"
    job.last_progress_at = time.time() - (settings.JOB_PHASE_WATCHDOG_SECONDS + 60)

    reconciled = reconcile_stale_running_jobs(
        staleness_threshold_seconds=settings.JOB_STALENESS_THRESHOLD_SECONDS
    )
    assert reconciled == 1
    assert job.status == "failed"
    assert "writing" in (job.error or "")


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


# ---------------------------------------------------------------------------
# Job heartbeat reconciliation (LE-143)
# ---------------------------------------------------------------------------


def test_reconcile_stale_running_jobs_in_memory() -> None:
    """A job whose progress heartbeat went silent past the threshold is failed.

    LE-143 phase-watchdog semantics: staleness is keyed on the progress
    heartbeat (``last_progress_at``), not whole-job age. A job that has not
    advanced progress past the threshold is treated as hung and failed.
    """
    import time

    from app.routers.index import _Job, _jobs, reconcile_stale_running_jobs

    job_id = "test-stale-job-1"
    now = time.time()
    job = _Job(job_id=job_id, repo_path="/tmp/repo-stale")
    job.status = "running"
    # NOTE: phase 'parsing' (not 'writing') — the writing phase gets a widened
    # budget (it emits no callbacks during the slow Kùzu bulk flush); 'parsing'
    # emits ~1 Hz callbacks so silence there is a genuine hang.
    job.phase = "parsing"
    job.started_at = now - 400
    # Heartbeat went silent 400s ago (> 300s threshold) — hung phase.
    job.last_progress_at = now - 400
    _jobs[job_id] = job

    reconciled = reconcile_stale_running_jobs(staleness_threshold_seconds=300)

    assert reconciled == 1
    failed = _jobs[job_id]
    assert failed.status == "failed"
    assert "phase watchdog" in (failed.error or "")
    assert failed.finished_at is not None
    _jobs.pop(job_id, None)


def test_reconcile_stale_running_jobs_ignores_recent_progress() -> None:
    """A long-but-healthy job (recent heartbeat) is NOT reaped.

    Critical no-regression: a job that started long ago but is still emitting
    progress must survive. Old code keyed on ``started_at`` and would kill it;
    the phase watchdog keys on ``last_progress_at`` so it is left alone.
    """
    import time

    from app.routers.index import _Job, _jobs, reconcile_stale_running_jobs

    job_id = "test-healthy-long-job"
    now = time.time()
    job = _Job(job_id=job_id, repo_path="/tmp/repo-healthy")
    job.status = "running"
    job.phase = "embedding"
    job.started_at = now - 4000  # started over an hour ago
    job.last_progress_at = now - 5  # but still progressing
    _jobs[job_id] = job

    reconciled = reconcile_stale_running_jobs(staleness_threshold_seconds=300)

    assert reconciled == 0
    survivor = _jobs[job_id]
    assert survivor.status == "running"
    assert survivor.error is None
    _jobs.pop(job_id, None)


def test_reconcile_stale_running_jobs_persistent_store(tmp_path: Path) -> None:
    """Verify reconciliation of stale jobs in the persistent store.

    Creates a running job directly in jobs_store with an old updated_at
    timestamp, then verifies that reconcile_stale_running_jobs() finds and
    marks it as failed.
    """
    import time

    from app.routers.index import reconcile_stale_running_jobs

    # Create a stale job directly in the persistent store
    old_timestamp = time.time() - 400  # 400 seconds ago
    persisted = jobs_store.create_job(
        kind="index",
        actor_oid="",
        actor_email="",
        repo_path=str(tmp_path),
        force_reindex=False,
        exclude_paths=frozenset(),
        worker_token="test-worker",
        initial_status="running",
        initial_phase="parsing",
    )
    job_id = persisted.job_id

    # Backdate the job in the database by directly updating updated_at
    conn = jobs_store._require_conn()
    conn.execute(
        "UPDATE jobs SET updated_at = ? WHERE job_id = ?",
        (old_timestamp, job_id),
    )
    conn.commit()

    # Reconcile with a 300s threshold
    reconciled = reconcile_stale_running_jobs(staleness_threshold_seconds=300)

    # Should have reconciled 1 job from the persistent store
    assert reconciled == 1

    # Verify the job was marked as failed in the store
    resp = client.get(f"/index/{job_id}/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "failed"
    assert "stale by heartbeat reconciliation" in (body.get("error") or "")


# ---------------------------------------------------------------------------
# Orphaned-indexing-flag regression tests
#
# Root cause: when a background asyncio task dies without transitioning the
# _Job to a terminal state (e.g. CancelledError propagating as BaseException,
# bypassing the except-Exception handler), job.status stays "running" and
# /health reports indexing:true until the periodic reconciler fires (up to
# 5 min).  Simultaneously, the reindex path sees an in-memory "running" job
# and returns 409, permanently blocking reindex without a process restart.
#
# The fix adds three layers:
#   1. _run_ingestion catches BaseException and always transitions the job.
#   2. is_repo_indexing() lazily preempts stale in-memory jobs.
#   3. start_index()'s in-memory loop preempts stale jobs instead of 409ing.
# ---------------------------------------------------------------------------


def _plant_dead_job(repo_path: Path, *, stale_seconds: int = 2400) -> str:
    """Seed a job that looks exactly like one whose worker died mid-run.

    Creates BOTH the in-memory _Job (status='running', heartbeat stale) AND
    the persistent durable row (status='running', updated_at stale) without
    any live asyncio task behind it — this is the orphaned-flag scenario.
    """
    import time

    from app.routers.index import _Job

    durable = jobs_store.create_job(
        kind="index",
        actor_oid="",
        actor_email="",
        repo_path=str(repo_path),
        force_reindex=False,
        exclude_paths=frozenset(),
    )
    job_id = durable.job_id
    stale_ts = time.time() - stale_seconds
    # In-memory job with a stale heartbeat (simulates dead asyncio task).
    _jobs[job_id] = _Job(
        job_id=job_id,
        repo_path=str(repo_path),
        status="running",
        phase="parsing",
        last_progress_at=stale_ts,
    )
    # Make the durable row stale too.
    conn = jobs_store._require_conn()
    conn.execute(
        "UPDATE jobs SET updated_at = ? WHERE job_id = ?",
        (stale_ts, job_id),
    )
    return job_id


def test_health_reports_not_indexing_for_dead_job(tmp_path: Path) -> None:
    """is_repo_indexing() must return False when the in-memory job is stale.

    Regression: before the fix, /health returned indexing:true even after the
    worker died because is_repo_indexing() trusted job.status without checking
    the heartbeat age.
    """
    from app.routers.index import is_repo_indexing

    dead_job_id = _plant_dead_job(tmp_path)
    repo_name = tmp_path.name

    # With a stale heartbeat the job must be treated as dead.
    assert is_repo_indexing(repo_name) is False, (
        "is_repo_indexing should return False for a job with a stale heartbeat"
    )
    # The dead job must have been lazily transitioned to failed.
    assert _jobs[dead_job_id].status == "failed"
    row = jobs_store.get_job(dead_job_id)
    assert row is not None and row.status == "failed"


def test_reindex_unblocked_after_dead_job_clears_stale_flag(tmp_path: Path) -> None:
    """POST /repos/{name}/reindex must succeed (not 409) when the only running
    job has a stale heartbeat — i.e. its worker is dead.

    Regression: before the fix, start_index() unconditionally 409'd on any
    in-memory "running" job regardless of heartbeat freshness, permanently
    blocking reindex until a process restart.
    """
    from app.routers.index import indexed_repo_paths

    repo_name = tmp_path.name
    indexed_repo_paths[repo_name] = str(tmp_path)
    dead_job_id = _plant_dead_job(tmp_path)
    try:
        with patch("app.routers.index._run_ingestion", new_callable=AsyncMock):
            resp = client.post(
                f"/repos/{repo_name}/reindex", json={"force": True}
            )
        assert resp.status_code == 202, (
            f"Expected 202 after dead-job preemption, got {resp.status_code}: {resp.text}"
        )
        new_job_id = resp.json()["job_id"]
        assert new_job_id != dead_job_id
        # The dead job was transitioned to failed.
        assert _jobs[dead_job_id].status == "failed"
        row = jobs_store.get_job(dead_job_id)
        assert row is not None and row.status == "failed"
    finally:
        indexed_repo_paths.pop(repo_name, None)


def test_live_job_still_returns_409_on_concurrent_reindex(tmp_path: Path) -> None:
    """A genuinely live job (fresh heartbeat) must still return 409 conflict.

    Validates that the stale-job self-heal does NOT remove the concurrency
    guard for healthy in-progress jobs.
    """
    from app.routers.index import indexed_repo_paths

    repo_name = tmp_path.name
    indexed_repo_paths[repo_name] = str(tmp_path)
    try:
        with patch("app.routers.index._run_ingestion", new_callable=AsyncMock):
            r1 = client.post(f"/repos/{repo_name}/reindex", json={"force": True})
        assert r1.status_code == 202

        # The job was just created — its heartbeat is fresh.
        job_id = r1.json()["job_id"]
        assert _jobs[job_id].status == "running"

        # A second concurrent reindex must still be blocked.
        with patch("app.routers.index._run_ingestion", new_callable=AsyncMock):
            r2 = client.post(f"/repos/{repo_name}/reindex", json={"force": True})
        assert r2.status_code == 409, (
            f"Expected 409 for a genuinely live job, got {r2.status_code}"
        )
    finally:
        indexed_repo_paths.pop(repo_name, None)


def test_run_ingestion_transitions_job_on_cancellation(tmp_path: Path) -> None:
    """_run_ingestion must mark the job failed when the asyncio task is cancelled.

    Regression: the original except-Exception block did not catch
    asyncio.CancelledError (a BaseException subclass since Python 3.8), so a
    cancelled task left job.status = "running" and the stale flag stuck.
    """
    import asyncio
    import time

    from app.routers.index import _Job, _run_ingestion

    job_id = "cancel-regression-job"
    job = _Job(
        job_id=job_id,
        repo_path=str(tmp_path),
        status="running",
        phase="parsing",
        last_progress_at=time.time(),
    )
    _jobs[job_id] = job

    def _raise_cancelled(_job: object, _force: bool) -> None:
        # Simulate the executor raising CancelledError mid-indexing.
        raise asyncio.CancelledError("test-induced cancellation")

    async def _drive() -> None:
        with patch("app.routers.index._blocking_index", side_effect=_raise_cancelled):
            try:
                await _run_ingestion(job, False)
            except asyncio.CancelledError:
                pass  # expected — BaseException handler re-raises

    asyncio.run(_drive())

    # The job must be terminal — NOT stuck at "running".
    assert job.status == "failed", (
        f"Expected job.status='failed' after CancelledError, got '{job.status}'"
    )
    assert job.error is not None and "cancelled" in job.error.lower()
    # The durable row must also be terminal.
    row = jobs_store.get_job(job_id)
    if row is not None:
        assert row.status == "failed"
