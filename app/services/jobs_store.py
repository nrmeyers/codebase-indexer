"""Persistent job state for the Code Indexer Service.

Thin DAO around stdlib ``sqlite3``. WAL-mode SQLite at ``JOBS_DB_PATH``
backs the in-memory job tracker that previously lived in
``app/routers/index.py``. Survives restart, supports cross-user filtering,
and lets the service fail-fast on duplicate concurrent requests for the
same repo.

Design decisions:
    * Single long-lived connection (``check_same_thread=False``) guarded by a
      module-level ``threading.Lock``. Single writer plus WAL gives readers
      lock-free access at no cost — workload is far below the few-writes/sec
      threshold that would justify a connection pool or a full Postgres swap.
    * No ORM. All SQL is hand-written so the schema is auditable and the
      hot path stays a few ``execute()`` calls.
    * ``Job`` is a frozen dataclass — callers should treat results as
      immutable snapshots; mutate state through the DAO functions only.
    * ``worker_token`` set per-process at create-time. ``sweep_interrupted``
      uses the mismatch to flag rows owned by a previous (now-dead) process
      so a UI can render them with a "service restarted" message.

This module is internal to the service. The surface is intentionally small:
``init`` once at startup, then per-job functions for create / update /
terminal-transition / query.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL CHECK (kind IN ('index','embed','watch_partial')),
  actor_oid TEXT NOT NULL,
  actor_email TEXT NOT NULL,
  repo_slug TEXT NOT NULL,
  repo_path TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('queued','running','done','failed','cancelled','interrupted')),
  phase TEXT,
  progress_pct REAL NOT NULL DEFAULT 0.0,
  files_total INTEGER NOT NULL DEFAULT 0,
  files_done INTEGER NOT NULL DEFAULT 0,
  current_file TEXT,
  node_count INTEGER NOT NULL DEFAULT 0,
  rel_count INTEGER NOT NULL DEFAULT 0,
  embedding_count INTEGER NOT NULL DEFAULT 0,
  force_reindex INTEGER NOT NULL DEFAULT 0,
  exclude_paths TEXT,
  error TEXT,
  cancel_requested INTEGER NOT NULL DEFAULT 0,
  pid INTEGER,
  worker_token TEXT,
  started_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  finished_at REAL,
  schema_version INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_jobs_actor       ON jobs(actor_oid);
CREATE INDEX IF NOT EXISTS idx_jobs_repo        ON jobs(repo_slug);
CREATE INDEX IF NOT EXISTS idx_jobs_status      ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_started_at  ON jobs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_repo_active
   ON jobs(repo_slug) WHERE status IN ('queued','running');

CREATE TABLE IF NOT EXISTS schema_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
INSERT OR IGNORE INTO schema_meta(key,value) VALUES ('version','1');
"""


_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"done", "failed", "cancelled", "interrupted"}
)
_ACTIVE_STATUSES: frozenset[str] = frozenset({"queued", "running"})


# ---------------------------------------------------------------------------
# Job dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    """Immutable snapshot of one persisted row.

    Attributes mirror the ``jobs`` table 1:1 with two convenience
    transformations:
        * ``exclude_paths`` is exposed as a frozenset (stored as JSON).
        * ``force_reindex`` / ``cancel_requested`` are bools (stored as 0/1).
    """

    job_id: str
    kind: str
    actor_oid: str
    actor_email: str
    repo_slug: str
    repo_path: str
    status: str
    phase: str | None
    progress_pct: float
    files_total: int
    files_done: int
    current_file: str | None
    node_count: int
    rel_count: int
    embedding_count: int
    force_reindex: bool
    exclude_paths: frozenset[str]
    error: str | None
    cancel_requested: bool
    pid: int | None
    worker_token: str | None
    started_at: float
    updated_at: float
    finished_at: float | None


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------


_conn: sqlite3.Connection | None = None
_lock = threading.Lock()
_db_path: str = ""


def _row_to_job(row: sqlite3.Row) -> Job:
    """Materialise a row dict into a frozen ``Job``."""
    raw_excludes = row["exclude_paths"] or "[]"
    try:
        excludes = frozenset(json.loads(raw_excludes))
    except (TypeError, ValueError, json.JSONDecodeError):
        excludes = frozenset()
    return Job(
        job_id=row["job_id"],
        kind=row["kind"],
        actor_oid=row["actor_oid"],
        actor_email=row["actor_email"],
        repo_slug=row["repo_slug"],
        repo_path=row["repo_path"],
        status=row["status"],
        phase=row["phase"],
        progress_pct=float(row["progress_pct"]),
        files_total=int(row["files_total"]),
        files_done=int(row["files_done"]),
        current_file=row["current_file"],
        node_count=int(row["node_count"]),
        rel_count=int(row["rel_count"]),
        embedding_count=int(row["embedding_count"]),
        force_reindex=bool(row["force_reindex"]),
        exclude_paths=excludes,
        error=row["error"],
        cancel_requested=bool(row["cancel_requested"]),
        pid=row["pid"],
        worker_token=row["worker_token"],
        started_at=float(row["started_at"]),
        updated_at=float(row["updated_at"]),
        finished_at=(
            float(row["finished_at"]) if row["finished_at"] is not None else None
        ),
    )


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


def init(db_path: str) -> None:
    """Open (or reopen) the persistent jobs database at ``db_path``.

    Creates the parent directory if missing, applies WAL/PRAGMAs, and
    runs the idempotent ``CREATE TABLE IF NOT EXISTS`` DDL. Safe to call
    multiple times — subsequent calls swap the underlying connection so
    tests can re-init against a fresh ``:memory:`` or ``tmp_path`` DB.
    """
    global _conn, _db_path
    with _lock:
        if _conn is not None:
            try:
                _conn.close()
            except Exception:
                pass
            _conn = None
        if db_path != ":memory:":
            parent = Path(db_path).parent
            if str(parent):
                parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            db_path, check_same_thread=False, isolation_level=None
        )
        conn.row_factory = sqlite3.Row
        conn.executescript(_DDL)
        _conn = conn
        _db_path = db_path
    logger.info("jobs_store initialised at %s", db_path)


def _require_conn() -> sqlite3.Connection:
    if _conn is None:
        raise RuntimeError(
            "jobs_store.init() must be called before any other DAO function"
        )
    return _conn


# ---------------------------------------------------------------------------
# create / update
# ---------------------------------------------------------------------------


def create_job(
    *,
    kind: str = "index",
    actor_oid: str,
    actor_email: str,
    repo_path: str,
    force_reindex: bool,
    exclude_paths: Iterable[str] = (),
    worker_token: str | None = None,
    initial_status: str = "running",
    initial_phase: str = "queued",
) -> Job:
    """Insert a new ``running`` job row and return the snapshot.

    Args:
        kind: One of ``index`` | ``embed`` | ``watch_partial``.
        actor_oid: M365 ``oid`` claim (Phase 1 dep). Falls back to
            ``"anon"`` upstream when auth is disabled.
        actor_email: M365 ``email`` claim (Phase 1 dep). Display only.
        repo_path: Absolute or relative path; ``Path(...).name`` is used
            as ``repo_slug``.
        force_reindex: Whether the caller asked for a clean re-index.
        exclude_paths: Iterable of path prefixes the worker should skip.
        worker_token: Process-scoped token; rows whose token differs at
            startup are flipped to ``interrupted`` by ``sweep_interrupted``.
        initial_status: Defaults to ``"running"`` to match today's
            in-memory behavior. Phase 5 may pass ``"queued"`` for the
            watcher path.
        initial_phase: Defaults to ``"queued"``.

    Returns:
        Job: Frozen snapshot of the inserted row.
    """
    conn = _require_conn()
    job_id = str(uuid.uuid4())
    now = time.time()
    repo_slug = Path(repo_path).name or "repo"
    excludes_json = json.dumps(sorted({str(p) for p in exclude_paths}))
    pid = os.getpid()
    with _lock:
        conn.execute(
            """
            INSERT INTO jobs (
              job_id, kind, actor_oid, actor_email, repo_slug, repo_path,
              status, phase, progress_pct, files_total, files_done,
              current_file, node_count, rel_count, embedding_count,
              force_reindex, exclude_paths, error, cancel_requested,
              pid, worker_token, started_at, updated_at, finished_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0.0, 0, 0,
                      NULL, 0, 0, 0,
                      ?, ?, NULL, 0,
                      ?, ?, ?, ?, NULL)
            """,
            (
                job_id,
                kind,
                actor_oid,
                actor_email,
                repo_slug,
                repo_path,
                initial_status,
                initial_phase,
                1 if force_reindex else 0,
                excludes_json,
                pid,
                worker_token,
                now,
                now,
            ),
        )
        row = conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
    assert row is not None  # we just inserted it
    return _row_to_job(row)


def update_progress(
    job_id: str,
    *,
    phase: str | None = None,
    progress_pct: float | None = None,
    files_total: int | None = None,
    files_done: int | None = None,
    current_file: str | None = None,
    node_count: int | None = None,
    rel_count: int | None = None,
    embedding_count: int | None = None,
) -> None:
    """Partial-update progress fields. Unset args are left unchanged.

    Implemented with an explicit set-clause builder rather than COALESCE
    so a caller can deliberately blank ``current_file`` (pass empty
    string then null in two calls).
    """
    fields: list[str] = []
    params: list[object] = []
    if phase is not None:
        fields.append("phase = ?")
        params.append(phase)
    if progress_pct is not None:
        fields.append("progress_pct = ?")
        params.append(float(progress_pct))
    if files_total is not None:
        fields.append("files_total = ?")
        params.append(int(files_total))
    if files_done is not None:
        fields.append("files_done = ?")
        params.append(int(files_done))
    if current_file is not None:
        fields.append("current_file = ?")
        params.append(current_file)
    if node_count is not None:
        fields.append("node_count = ?")
        params.append(int(node_count))
    if rel_count is not None:
        fields.append("rel_count = ?")
        params.append(int(rel_count))
    if embedding_count is not None:
        fields.append("embedding_count = ?")
        params.append(int(embedding_count))
    if not fields:
        return
    fields.append("updated_at = ?")
    params.append(time.time())
    params.append(job_id)
    sql = f"UPDATE jobs SET {', '.join(fields)} WHERE job_id = ?"
    conn = _require_conn()
    with _lock:
        conn.execute(sql, params)


def mark_done(
    job_id: str,
    *,
    node_count: int,
    rel_count: int,
    embedding_count: int,
) -> None:
    """Idempotent transition to ``status='done'``, ``progress_pct=100``."""
    conn = _require_conn()
    now = time.time()
    with _lock:
        conn.execute(
            """
            UPDATE jobs SET
              status = 'done',
              phase = 'done',
              progress_pct = 100.0,
              node_count = ?,
              rel_count = ?,
              embedding_count = ?,
              error = NULL,
              updated_at = ?,
              finished_at = COALESCE(finished_at, ?)
            WHERE job_id = ?
            """,
            (
                int(node_count),
                int(rel_count),
                int(embedding_count),
                now,
                now,
                job_id,
            ),
        )


def mark_failed(
    job_id: str,
    *,
    error: str,
    terminal_status: str = "failed",
) -> None:
    """Idempotent transition to a terminal failure status.

    ``terminal_status`` accepts ``failed`` | ``cancelled`` | ``interrupted``.
    Any other value is rejected by the CHECK constraint at the DB level.
    """
    conn = _require_conn()
    now = time.time()
    with _lock:
        conn.execute(
            """
            UPDATE jobs SET
              status = ?,
              error = ?,
              updated_at = ?,
              finished_at = COALESCE(finished_at, ?)
            WHERE job_id = ?
            """,
            (terminal_status, error, now, now, job_id),
        )


def request_cancel(job_id: str) -> bool:
    """Set ``cancel_requested=1``. Returns True iff the row exists and was
    in an active (non-terminal) state."""
    conn = _require_conn()
    now = time.time()
    with _lock:
        cur = conn.execute(
            """
            UPDATE jobs SET cancel_requested = 1, updated_at = ?
            WHERE job_id = ? AND status IN ('queued','running')
            """,
            (now, job_id),
        )
        return cur.rowcount > 0


def is_cancel_requested(job_id: str) -> bool:
    """Lock-free read used by the worker between phases."""
    conn = _require_conn()
    row = conn.execute(
        "SELECT cancel_requested FROM jobs WHERE job_id = ?", (job_id,)
    ).fetchone()
    return bool(row[0]) if row is not None else False


# ---------------------------------------------------------------------------
# queries
# ---------------------------------------------------------------------------


def get_job(job_id: str) -> Job | None:
    """Return the job snapshot or None if no row matches."""
    conn = _require_conn()
    row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    return _row_to_job(row) if row is not None else None


# Wall-clock ceiling for an "active" row.  Anything claiming to be
# queued/running but whose started_at is older than this is almost
# certainly a phantom — the worker died (SIGKILL, OOM, container
# restart) without flushing terminal state.  4h matches the embed
# subprocess timeout in routers/index.py; nothing legitimate runs
# longer than that.
_PHANTOM_AGE_SEC = 4 * 60 * 60


def find_active_for_repo(repo_slug: str) -> Job | None:
    """Return the most recent active (queued|running) job for a slug.

    Auto-expires phantom rows whose started_at is older than the wall-clock
    ceiling — these are jobs whose worker died without flushing terminal
    state, so reporting them as "still running" makes /index POST 409 on
    every retry forever.  We mark them ``failed`` with a clear error so
    they appear in the job history but no longer block new work.
    """
    conn = _require_conn()
    row = conn.execute(
        """
        SELECT * FROM jobs
        WHERE repo_slug = ? AND status IN ('queued','running')
        ORDER BY started_at DESC
        LIMIT 1
        """,
        (repo_slug,),
    ).fetchone()
    if row is None:
        return None

    job = _row_to_job(row)
    age_sec = time.time() - (job.started_at or 0)
    if age_sec > _PHANTOM_AGE_SEC:
        # Auto-expire.  Use the same UPDATE shape as mark_failed but
        # inline so we can run it during a read path without callers
        # caring.  The next call to find_active_for_repo for this slug
        # will return None.
        with _lock:
            conn.execute(
                """
                UPDATE jobs SET
                  status = 'failed',
                  error = COALESCE(error, 'phantom — worker died without flushing'),
                  updated_at = ?,
                  finished_at = COALESCE(finished_at, ?)
                WHERE job_id = ? AND status IN ('queued','running')
                """,
                (time.time(), time.time(), job.job_id),
            )
        return None
    return job


def list_jobs(
    *,
    actor_oid: str | None = None,
    repo_slug: str | None = None,
    status: set[str] | None = None,
    kind: set[str] | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Job]:
    """Newest-first paged history.

    Filters compose with AND. Pass ``status`` to constrain to a subset
    (e.g. ``{'running','done'}``); ``kind`` defaults to including every
    kind so callers can flip the watcher-noise filter on at the call
    site (Phase 5).
    """
    conn = _require_conn()
    where: list[str] = []
    params: list[object] = []
    if actor_oid is not None:
        where.append("actor_oid = ?")
        params.append(actor_oid)
    if repo_slug is not None:
        where.append("repo_slug = ?")
        params.append(repo_slug)
    if status:
        placeholders = ",".join("?" for _ in status)
        where.append(f"status IN ({placeholders})")
        params.extend(sorted(status))
    if kind:
        placeholders = ",".join("?" for _ in kind)
        where.append(f"kind IN ({placeholders})")
        params.extend(sorted(kind))
    sql = "SELECT * FROM jobs"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY started_at DESC LIMIT ? OFFSET ?"
    params.extend([max(1, min(int(limit), 500)), max(0, int(offset))])
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_job(r) for r in rows]


def count_active(kind: str | None = None) -> int:
    """Return the count of active (queued|running) rows. Used by /health."""
    conn = _require_conn()
    sql = "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')"
    params: tuple[object, ...] = ()
    if kind is not None:
        sql += " AND kind = ?"
        params = (kind,)
    row = conn.execute(sql, params).fetchone()
    return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# admin / housekeeping
# ---------------------------------------------------------------------------


def clear_terminal(
    actor_oid: str | None = None,
    statuses: set[str] | None = None,
) -> int:
    """Delete rows in terminal states. Returns count removed.

    By default clears done / failed / cancelled — ``interrupted`` rows
    are kept by default so a UI can show "we restarted while you were
    indexing" until the user explicitly dismisses them.
    """
    if statuses is None:
        statuses = {"done", "failed", "cancelled"}
    statuses = statuses & _TERMINAL_STATUSES
    if not statuses:
        return 0
    placeholders = ",".join("?" for _ in statuses)
    sql = f"DELETE FROM jobs WHERE status IN ({placeholders})"
    params: list[object] = sorted(statuses)
    if actor_oid is not None:
        sql += " AND actor_oid = ?"
        params.append(actor_oid)
    conn = _require_conn()
    with _lock:
        cur = conn.execute(sql, params)
        return int(cur.rowcount)


def sweep_interrupted(worker_token: str) -> int:
    """Flag any active rows from a previous process as ``interrupted``.

    Called from ``app/main.py``'s ``lifespan`` startup. A fresh
    ``WORKER_TOKEN`` is generated per-process; rows whose stored
    ``worker_token`` differs (or is NULL — pre-Phase-2 rows) are owned
    by a now-dead process and must be retired before any new work
    starts.
    """
    conn = _require_conn()
    now = time.time()
    with _lock:
        cur = conn.execute(
            """
            UPDATE jobs SET
              status = 'interrupted',
              error = COALESCE(error, 'service restart'),
              updated_at = ?,
              finished_at = COALESCE(finished_at, ?)
            WHERE status IN ('queued','running')
              AND (worker_token IS NULL OR worker_token != ?)
            """,
            (now, now, worker_token),
        )
        return int(cur.rowcount)


def delete_job(job_id: str) -> bool:
    """Drop a single row. Returns True iff something was deleted."""
    conn = _require_conn()
    with _lock:
        cur = conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def journal_mode() -> str:
    """Return the current journal mode (used by tests to verify WAL)."""
    conn = _require_conn()
    row = conn.execute("PRAGMA journal_mode").fetchone()
    return str(row[0]) if row else ""


def db_path() -> str:
    """Return the path the store was initialised against."""
    return _db_path


# ---------------------------------------------------------------------------
# Test helpers (never call from production code)
# ---------------------------------------------------------------------------


def _reset_for_tests(new_db_path: str = ":memory:") -> None:
    """Tear down the module-level connection and reinitialise against ``new_db_path``.

    Only for use in test fixtures — never call in production code. Allows each
    test to get a clean isolated store without restarting the process.

    Args:
        new_db_path: SQLite path for the fresh store.  Defaults to ``:memory:``.
    """
    global _conn, _db_path  # noqa: PLW0603
    with _lock:
        if _conn is not None:
            try:
                _conn.close()
            except Exception:
                pass
        _conn = None
        _db_path = ""
    init(new_db_path)
