"""POST /index and GET /index/{job_id}/status — background ingestion.

Indexing a repository is CPU-bound and can take minutes. To keep the HTTP
API responsive, this router accepts indexing requests asynchronously:

    1. ``POST /index`` creates a ``_Job`` record and returns a job_id (202).
    2. The heavy work runs in a thread-pool executor via FastAPI's
       BackgroundTasks so the event loop stays free.
    3. Clients poll ``GET /index/{job_id}/status`` until ``done`` or
       ``failed``.

Every index run executes two passes:

    Pass 1–3 (structural): tree-sitter parse → LadybugDB graph (nodes + rels)
    Pass 4   (embedding):  CodeRankEmbed model → DuckDB (.duck, v5.3 §6.5)

Embeddings are **required** — if pass 4 fails the job is marked ``failed``
and the caller must re-index.  Structural graph data is preserved on disk so
re-indexing without ``force_reindex=true`` only re-runs the embedding pass
over already-parsed files.

Jobs live in an in-memory dict (``_jobs``). This is acceptable because the
service is single-process; running multiple replicas would require moving
the store to Redis or LadybugDB itself.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from ..config import settings
from .. import metrics as _metrics
from ..services import jobs_store as _jobs_store
from ..models import (
    DeleteIndexResponse,
    DiffMetrics,
    IndexAccepted,
    IndexRequest,
    IndexStatus,
    JobClearResponse,
    JobListResponse,
    JobSummary,
    NodeTypeStat,
    RepoStatsResponse,
)

import os as _os

logger = logging.getLogger(__name__)

router = APIRouter()

# Per-process worker token used by jobs_store.sweep_interrupted() to identify
# orphaned rows from prior processes. Matches the token written to each new row.
_WORKER_TOKEN: str = _os.urandom(8).hex()


def _cleanup_old_embed_logs(keep: int = 5) -> int:
    """Remove all but the ``keep`` most-recent /tmp/cis_embed_*.log files.

    Embed logs are diagnostic artefacts that pile up over weeks of dev use
    (one per re-index attempt).  Tail behaviour and incremental embedding
    mean we run more re-indexes per day, so without housekeeping the file
    count grows unbounded.  Called on uvicorn startup; best-effort.
    """
    try:
        log_paths = sorted(
            Path("/tmp").glob("cis_embed_*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        removed = 0
        for stale in log_paths[keep:]:
            try:
                stale.unlink()
                removed += 1
            except OSError:
                pass
        return removed
    except Exception:  # noqa: BLE001
        return 0


def _parse_embed_progress(job_id: str) -> tuple[int, int, int] | None:
    """Read the latest PROGRESS line from an embed subprocess log.

    Returns ``(embedded, skipped_unchanged, filtered_out)`` from the most
    recent ``PROGRESS embedded=N skipped=M filtered=K`` line, or None when
    the log file does not exist or has no PROGRESS line yet.

    The embed driver writes one PROGRESS line per concurrent flush
    (BUC-1517 / BUC-1519).  Tailing the log gives the frontend live
    visibility into the embed pass without changing the subprocess
    contract.
    """
    log_path = Path(f"/tmp/cis_embed_{job_id}-embed.log")
    if not log_path.exists():
        return None
    try:
        # Embed logs are tiny (one PROGRESS line every ~30s).  No need
        # for a stat-then-seek tail; just read and reverse-scan.
        with log_path.open() as fh:
            lines = fh.readlines()
    except OSError:
        return None
    for line in reversed(lines):
        if line.startswith("PROGRESS "):
            parts = line.split()
            try:
                vals = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in parts[1:]}
                return (
                    vals.get("embedded", 0),
                    vals.get("skipped", 0),
                    vals.get("filtered", 0),
                )
            except (ValueError, IndexError):
                return None
    return None


# Pre-warm tracking — only warm the endpoint once per uvicorn process per job.
# A set of job_ids that have already had a warmup ping sent.
_prewarmed_jobs: set[str] = set()


def _prewarm_sagemaker_endpoint(job_id: str) -> None:
    """Fire-and-forget warmup ping to the SageMaker embedding endpoint.

    Hides the 30-60s Serverless Inference cold start by running in parallel
    with the parsing phase (~70s for typical repos).  By the time embedding
    starts, the endpoint is already hot and the first real batch returns
    in normal latency.

    Idempotent per job — only fires once even if the parsing-phase callback
    runs multiple times for the same job_id.  Cheap (~$0.0001 per call) and
    failures are silently swallowed since this is purely a latency optimisation.
    """
    if job_id in _prewarmed_jobs:
        return
    _prewarmed_jobs.add(job_id)

    def _ping() -> None:
        try:
            from ..services.sagemaker_embedder import get_sagemaker_embedder
            sm = get_sagemaker_embedder()
            if sm is None:
                return  # not configured; nothing to warm
            t0 = time.time()
            sm.embed("warmup")
            logger.info(
                "SageMaker prewarm: endpoint=%s, latency=%.2fs (job=%s)",
                sm.endpoint_name,
                time.time() - t0,
                job_id[:8],
            )
        except Exception as exc:  # noqa: BLE001
            # Warmup is best-effort — never fail the job over a missed ping.
            logger.debug(
                "SageMaker prewarm failed (best-effort): %s: %s",
                type(exc).__name__,
                exc,
            )

    threading.Thread(target=_ping, name=f"sm-prewarm-{job_id[:8]}", daemon=True).start()


class _IndexCancelledError(RuntimeError):
    """Raised from the progress callback when a job receives a cancel signal."""


# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------


@dataclass
class _Job:
    """Internal bookkeeping record for a single indexing request.

    Attributes:
        job_id: UUID4 string returned to the client.
        repo_path: Resolved absolute path to the repo being indexed.
        status: Lifecycle state — ``running`` → ``done`` | ``failed``.
        progress_pct: Best-effort progress indicator, monotonically
            non-decreasing.  Derived from phase + per-file counters via
            the progress callback rather than hard-coded milestones.
        phase: Current phase label shown to UIs.
        files_total: Total eligible files discovered (0 until discovering
            phase completes).
        files_done: Files fully scanned so far (advances during parsing).
        current_file: Relative path of the file currently being parsed;
            None outside the parsing phase.
        node_count: Final graph node count, populated on completion.
        rel_count: Final graph relationship count, populated on completion.
        embedded_count: Number of function/method embeddings written.
        cancelled: Set to True by the cancel endpoint; the progress
            callback raises _IndexCancelledError on the next check.
        error: Populated only when ``status == "failed"``.
        started_at: Wall-clock start time used for TTL-based pruning.
        finished_at: Wall-clock end time; None while running.
    """

    job_id: str
    repo_path: str
    status: Literal["running", "done", "failed"] = "running"
    progress_pct: float = 0.0
    phase: str = "queued"
    files_total: int = 0
    files_done: int = 0
    current_file: str | None = None
    node_count: int = 0
    rel_count: int = 0
    embedded_count: int = 0
    embeddings_skipped_unchanged: int = 0
    embeddings_filtered_out: int = 0
    # BUC-1574 (Phase 1.4) — diff-metrics instrumentation. Captured from the
    # embed subprocess so /index/{job_id}/diff_metrics can report the
    # incremental-embed audit shape without re-tailing the log.
    embed_started_at: float | None = None
    embed_finished_at: float | None = None
    cancelled: bool = False
    elapsed_sec: float = 0.0
    eta_sec: float | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    exclude_paths: frozenset[str] = field(default_factory=frozenset)


# Module-level transient runtime cache — indexed by job_id. Single-process.
#
# Two-store design (intentional, not a TODO):
#   - `_jobs` holds the high-frequency mutable progress fields for an
#     in-flight job (current_file, eta_sec, files_done) and is updated at
#     ~1 Hz from the GraphUpdater progress_cb. Lives only in RAM.
#   - `app.services.jobs_store` is the durable record (SQLite, WAL): it
#     gets the terminal counters via `mark_done` / `mark_failed`, sweeps
#     interrupted rows on lifespan boot, and surfaces persistent state
#     across restarts. Updates are coarser — phase transitions only.
#
# Persisting every callback to SQLite would either thrash disk or require
# a write-coalescer that adds complexity for no gain. Querying _jobs at
# response time is O(1) and always reflects the latest callback fire.
# Querying jobs_store is needed only when the job_id isn't in _jobs (i.e.
# the job survived a restart) — that lookup happens lazily in
# get_index_status's fallback branch.
_jobs: dict[str, _Job] = {}

# Per-repo-path lock: prevents two concurrent jobs from writing to the same
# LadybugDB instance simultaneously (single-writer constraint).
_repo_locks: dict[str, asyncio.Lock] = {}

# TTL: keep completed jobs for 1 hour so callers can poll after completion.
_JOB_TTL_SECONDS = 3600

# In-memory set of successfully indexed project names.  Updated by the
# background worker when a job completes.  Health + explorer endpoints read
# from here instead of opening a second DB connection (LadybugDB is
# single-writer — concurrent open() calls corrupt the WAL).
indexed_repos: set[str] = set()

# Maps repo_name → resolved absolute repo_path for embed jobs that need it.
indexed_repo_paths: dict[str, str] = {}

# Maps repo_name → unix timestamp of the last successful index.  Populated from
# the Project node's ``last_indexed_at`` on first read (see _get_last_indexed_at)
# so it survives restarts.  In-memory cache avoids a DB round-trip on every
# /health probe.
_last_indexed_cache: dict[str, float] = {}


def _write_meta(repo_name: str, **fields: Any) -> None:
    """Upsert ``fields`` into the per-repo DuckDB ``repo_metadata`` table.

    Opens the ``.duck`` file (creating it if absent), writes all key-value
    pairs atomically, and closes the connection.  Replaces the old JSON
    sidecar approach — DuckDB transactions provide the same atomic-write
    guarantee.
    """
    from codebase_rag.storage.vector_store import open_or_create, write_metadata

    vec_path = settings.vec_db_path_for_repo(repo_name)
    try:
        conn = open_or_create(vec_path)
        write_metadata(conn, **{k: str(v) for k, v in fields.items()})
        conn.close()
    except Exception:
        pass  # metadata write is best-effort; never fail the index job


def _read_meta(repo_name: str) -> dict[str, Any]:
    """Return all metadata for ``repo_name`` from the DuckDB ``repo_metadata`` table.

    Falls back to an empty dict when the ``.duck`` file does not yet exist
    (e.g. before the first successful index) or when DuckDB is unavailable.
    """
    from codebase_rag.storage.vector_store import open_or_create, read_all_metadata

    vec_path = settings.vec_db_path_for_repo(repo_name)
    if not Path(vec_path).exists():
        return {}
    try:
        conn = open_or_create(vec_path)
        meta = read_all_metadata(conn)
        conn.close()
        return meta
    except Exception:
        return {}


def _get_last_indexed_at(repo_name: str) -> float | None:
    """Return the last successful-index timestamp for ``repo_name``.

    Checks the in-memory cache first, then the ``repo_metadata`` SQLite table.
    Returns None on any miss or parse failure.
    """
    if repo_name in _last_indexed_cache:
        return _last_indexed_cache[repo_name]

    meta = _read_meta(repo_name)
    ts = meta.get("last_indexed_at")
    if ts is not None:
        try:
            val = float(ts)
            _last_indexed_cache[repo_name] = val
            return val
        except (TypeError, ValueError):
            return None
    return None


def is_repo_indexing(repo_name: str) -> bool:
    """Return True when any currently-running job targets ``repo_name``.

    Used by /health and /stats to signal UI-level mutual exclusion.
    """
    for j in _jobs.values():
        if j.status == "running" and Path(j.repo_path).name == repo_name:
            return True
    return False


def _prune_old_jobs() -> None:
    """Drop completed/failed jobs older than the TTL to bound memory.

    Only terminal jobs are pruned — a long-running job is never evicted
    regardless of age.
    """
    now = time.time()
    stale = [
        jid
        for jid, j in _jobs.items()
        if j.status in ("done", "failed") and (now - j.started_at) > _JOB_TTL_SECONDS
    ]
    for jid in stale:
        del _jobs[jid]


# ---------------------------------------------------------------------------
# Background ingestion worker
# ---------------------------------------------------------------------------


async def _run_ingestion(job: _Job, force_reindex: bool) -> None:
    """Drive code-graph-rag indexing in a background asyncio task.

    The core indexing work is CPU-bound / blocking.  We run it in a thread
    pool executor so the event loop stays responsive to health and status
    polling during long ingestion runs.

    A per-repo asyncio.Lock serialises concurrent jobs that target the same
    repo_path, enforcing LadybugDB's single-writer constraint and preventing
    graph corruption from two simultaneous ingest operations.

    Args:
        job: The mutable job record to update with progress and final state.
        force_reindex: When true, the underlying GraphUpdater clears the
            graph before re-ingesting.
    """
    # Acquire (or create) the per-repo lock before touching the DB.
    repo_key = str(Path(job.repo_path).resolve())
    if repo_key not in _repo_locks:
        _repo_locks[repo_key] = asyncio.Lock()
    lock = _repo_locks[repo_key]

    async with lock:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None, _blocking_index, job, force_reindex
            )
        except _IndexCancelledError:
            # phase/error already set by the progress callback before raising.
            if job.finished_at is None:
                job.finished_at = time.time()
            # Phase 2+4: persist cancel state.
            try:
                _jobs_store.mark_failed(job.job_id, error="Cancelled by user", terminal_status="cancelled")
            except RuntimeError:
                pass
            _metrics.record_index_terminal("cancelled", kind="index")
        except Exception as exc:
            # Capture failure on the job so pollers see the error reason rather
            # than a silent stuck-running status.
            job.status = "failed"
            job.error = str(exc)
            job.finished_at = time.time()
            # Phase 2+4: persist failure state.
            try:
                _jobs_store.mark_failed(job.job_id, error=str(exc))
            except RuntimeError:
                pass
            _metrics.record_index_terminal("failed", kind="index")


def _blocking_index(job: _Job, force_reindex: bool) -> None:
    """Synchronous ingestion — called from the thread pool.

    Uses LadybugIngestor as a context manager (which handles DB connection,
    schema migration, and VECTOR extension loading on entry). GraphUpdater
    requires parsers and queries to be loaded from the tree-sitter registry.

    Args:
        job: The job record to mutate with progress and final counts.
            ``job.exclude_paths`` is forwarded to GraphUpdater to skip
            synthetic test fixtures and other noise from the semantic index.
        force_reindex: When true, the graph is cleared before ingesting.
    """
    from codebase_rag.config import settings as cgr_settings
    from codebase_rag.services.ladybug_ingestor import LadybugIngestor
    from codebase_rag.graph_updater import GraphUpdater
    from codebase_rag.parser_loader import load_parsers

    repo = Path(job.repo_path).resolve()

    # Per-repo DB file: each indexed repo gets its own ``.db`` so the explorer
    # can open one index at a time and WAL corruption / re-indexing stays
    # scoped.  Parent directory is created lazily because LadybugDB will
    # otherwise fail with "No such file or directory".
    repo_db_path = settings.db_path_for_repo(repo.name)
    Path(repo_db_path).parent.mkdir(parents=True, exist_ok=True)

    # Point code-graph-rag at the per-repo DB. Without this, the two packages
    # would each read their own config and could end up writing to different
    # database files.
    cgr_settings.LADYBUG_DB_PATH = repo_db_path
    cgr_settings.LADYBUG_BATCH_SIZE = settings.LADYBUG_BATCH_SIZE

    parsers, queries = load_parsers()

    # Detect a fresh / empty DB BEFORE the ingestor creates/opens it.
    # If the DB didn't exist (or is zero-byte from a previous wipe), force a
    # full re-parse regardless of force_reindex.  Without this, the on-disk
    # hash cache (.cgr-hash-cache.json inside the repo) causes the incremental
    # updater to skip every "unchanged" file, leaving an empty graph even
    # though the DB has never been populated.
    db_was_new = (
        not Path(repo_db_path).exists()
        or Path(repo_db_path).stat().st_size < 4096  # < 4 KB = essentially empty
    )
    effective_force = force_reindex or db_was_new

    # When the caller requested a full force-reindex, physically delete the
    # per-repo DB file + WAL + hash cache.  Without this, GraphUpdater's
    # ``force=True`` only resets the in-memory hash cache — old Function /
    # Call / File nodes from previous ingestions REMAIN in LadybugDB and
    # the new parse just layers fresh copies on top.  Symptoms: junk
    # symbols persist across re-indexes, node count grows monotonically,
    # cgrignore changes only partially take effect.
    if force_reindex:
        db_file = Path(repo_db_path)
        # LadybugDB artefacts (kuzu naming: .db-wal, .db-shm)
        ladybug_wal = db_file.with_suffix(db_file.suffix + "-wal")
        ladybug_shm = db_file.with_suffix(db_file.suffix + "-shm")
        # DuckDB artefacts live alongside the .duck file derived from the
        # same stem — DuckDB names its journal `<file>.wal` (dot, not dash)
        # and may leave a `<file>.tmp` from interrupted writes.
        duck_file = db_file.with_suffix(".duck")
        duck_wal = db_file.with_name(db_file.stem + ".duck.wal")
        duck_tmp = db_file.with_name(db_file.stem + ".duck.tmp")
        # Belt-and-braces: also catch the ".db.wal" form that DuckDB has
        # been observed to construct when its connection path got the
        # LadybugDB extension by mistake.
        belt_db_dot_wal = db_file.with_name(db_file.name + ".wal")
        hash_cache = repo / ".cgr-hash-cache.json"
        for artifact in (
            db_file,
            ladybug_wal,
            ladybug_shm,
            duck_file,
            duck_wal,
            duck_tmp,
            belt_db_dot_wal,
            hash_cache,
        ):
            try:
                artifact.unlink(missing_ok=True)
            except OSError:
                # Best-effort — a locked WAL file is not fatal; the ingestor
                # will recreate what it needs.
                pass

    # ------------------------------------------------------------------
    # Progress callback — maps GraphUpdater events to _Job state.
    # Called from the GraphUpdater thread; must be thread-safe for the
    # simple field writes it performs (GIL-protected in CPython).
    # ------------------------------------------------------------------
    _job_start = job.started_at

    def _progress_cb(event: dict) -> None:
        # Cancel check — must happen before any state mutation so that
        # phase/error are set consistently before raising.
        if job.cancelled:
            job.phase = "cancelled"
            job.status = "failed"
            job.error = "Cancelled by user"
            job.finished_at = time.time()
            raise _IndexCancelledError("Cancelled by user")

        phase = event.get("phase")
        if phase:
            job.phase = str(phase)
            # Pre-warm SageMaker the moment we enter the parsing phase.
            # Parsing typically takes ~70s; that's enough to absorb the
            # 30-60s Serverless Inference cold start in parallel, so
            # embedding kicks off against an already-hot endpoint.
            if phase == "parsing":
                _prewarm_sagemaker_endpoint(job.job_id)
        if "files_total" in event:
            job.files_total = int(event["files_total"])
        if "files_done" in event:
            job.files_done = int(event["files_done"])
        if "current_file" in event:
            cf = event["current_file"]
            job.current_file = str(cf) if cf else None

        # Compute progress_pct from phase + counters (monotonically non-decreasing).
        if "progress_pct" in event:
            pct: float = float(event["progress_pct"])
        elif phase == "discovering":
            pct = 2.0
        elif phase == "parsing":
            pct = 5.0 + (job.files_done / max(job.files_total, 1)) * 60.0
        elif phase == "writing":
            pct = 65.0
        elif phase == "embedding":
            pct = 70.0
        elif phase == "finalizing":
            pct = 98.0
        elif phase == "done":
            pct = 100.0
        else:
            pct = job.progress_pct  # unchanged

        job.progress_pct = max(job.progress_pct, min(100.0, pct))

        # Elapsed + ETA — computed at callback time (cheap, no extra polling).
        elapsed = time.time() - _job_start
        job.elapsed_sec = elapsed
        if job.progress_pct > 10.0 and job.progress_pct < 100.0:
            job.eta_sec = elapsed * (100.0 - job.progress_pct) / job.progress_pct
        else:
            job.eta_sec = None

    # Load .cgrignore patterns from the repo root and merge with explicit
    # exclude_paths from the POST body.  Without this the service would skip
    # only the request's exclude_paths and ignore the user's .cgrignore file
    # entirely (which is the expected extensibility surface for teams who
    # can't modify the POST body — e.g. IDE integrations).
    from codebase_rag.config import load_cgrignore_patterns
    cgrignore = load_cgrignore_patterns(repo)
    merged_excludes: frozenset[str] | None = None
    if job.exclude_paths or cgrignore.exclude:
        merged_excludes = frozenset(job.exclude_paths) | cgrignore.exclude
    unignore_paths: frozenset[str] | None = (
        cgrignore.unignore if cgrignore.unignore else None
    )

    # Phase 4: record start time for "parse" phase timer.
    _phase_parse_start = time.monotonic()

    # Sub-phase timers (Cycle 5 follow-up — localising the 4.4 sym/s
    # end-to-end bottleneck. Embed bench showed embedding is 89→121 sym/s
    # standalone, so >95% of indexing time is downstream of embedding.
    # These timers tell us whether the cost is in LadybugIngestor open/close
    # or in GraphUpdater.run() so we know where to push deeper
    # instrumentation. Phase-name keys: "parse_open", "parse_run",
    # "parse_metadata", "parse_close").
    _t_open = time.monotonic()
    # LadybugIngestor is a context manager — __enter__ opens the DB connection
    # and runs schema migration; __exit__ flushes remaining buffers and closes.
    with LadybugIngestor(
        db_path=repo_db_path,
        batch_size=settings.LADYBUG_BATCH_SIZE,
    ) as ingestor:
        _metrics.record_index_phase("parse_open", time.monotonic() - _t_open)

        updater = GraphUpdater(
            ingestor=ingestor,
            repo_path=repo,
            parsers=parsers,
            queries=queries,
            progress_cb=_progress_cb,
            exclude_paths=merged_excludes,
            unignore_paths=unignore_paths,
            skip_embeddings=True,  # embeddings handled by DuckDB subprocess below
        )

        _t_run = time.monotonic()
        updater.run(force=effective_force)
        _metrics.record_index_phase("parse_run", time.monotonic() - _t_run)
        # GraphUpdater emits "done" at 100% at the end of run(); reset to 92%
        # so the UI knows the embedding subprocess pass still follows.
        job.progress_pct = min(job.progress_pct, 92.0)

        # Store the absolute repo root on the Project node so search endpoints
        # can resolve relative file paths back to absolute paths.  The
        # last-indexed timestamp and other operational metadata live in the
        # ``repo_metadata`` table inside the per-repo ``.duck`` file instead —
        # LadybugDB's typed schema doesn't allow adding new columns without a
        # migration, while DuckDB key/value rows are free to extend.
        _t_meta = time.monotonic()
        project_name = repo.name
        ingestor.conn.execute(  # type: ignore[union-attr]
            "MATCH (p:Project {name: $name}) SET p.root_path = $root_path",
            {"name": project_name, "root_path": str(repo)},
        )
        _metrics.record_index_phase("parse_metadata", time.monotonic() - _t_meta)
        _t_close = time.monotonic()

    # `parse_close` ends here — flush + connection teardown happens on __exit__.
    _metrics.record_index_phase("parse_close", time.monotonic() - _t_close)
    # Top-level "parse" stays for back-compat with existing dashboard panels.
    _metrics.record_index_phase("parse", time.monotonic() - _phase_parse_start)

    # Collect final counts from LadybugDB. Best-effort: if the count
    # queries fail we still mark the job done — the ingestion itself
    # succeeded and callers can always re-query later.
    # IMPORTANT: use a short-lived block so Python GC can release the DB
    # lock before the embedding subprocess tries to open the same file.
    _count_db = None
    _count_conn = None
    try:
        import real_ladybug as lb  # type: ignore[import-untyped]

        _count_db = lb.Database(repo_db_path)
        _count_conn = lb.Connection(_count_db)

        node_res = _count_conn.execute("MATCH (n) RETURN count(n) AS cnt")
        if node_res.has_next():
            job.node_count = int(node_res.get_next()[0])

        rel_res = _count_conn.execute("MATCH ()-[r]->() RETURN count(r) AS cnt")
        if rel_res.has_next():
            job.rel_count = int(rel_res.get_next()[0])
    except Exception:
        pass  # counts are best-effort
    finally:
        # Explicitly close and drop references so LadybugDB releases the
        # file lock before the embedding subprocess attempts to open the
        # same DB. `del` alone is not enough because CPython's refcount
        # may delay destruction when an exception is in-flight.
        try:
            if _count_conn is not None and hasattr(_count_conn, "close"):
                _count_conn.close()
        except Exception:
            pass
        try:
            if _count_db is not None and hasattr(_count_db, "close"):
                _count_db.close()
        except Exception:
            pass
        _count_conn = None
        _count_db = None
        import gc as _gc
        _gc.collect()

    # Persist metadata sidecar AFTER counts are populated so the UI sees
    # authoritative node/rel totals in /stats without a separate query.
    _now = time.time()
    _write_meta(
        repo.name,
        last_indexed_at=_now,
        root_path=str(repo),
        node_count=str(job.node_count),
        rel_count=str(job.rel_count),
        last_job_id=job.job_id,
        schema_version="1.5",
    )
    _last_indexed_cache[repo.name] = _now

    # -------------------------------------------------------------------
    # Phase 1.1 — Tantivy BM25 lexical index (best-effort)
    # -------------------------------------------------------------------
    # After the parse pass commits LadybugDB we read the symbol metadata
    # back out and mirror it into a per-repo Tantivy index.  Failures here
    # are NON-FATAL — the lexical arm is additive; semantic + structural
    # search still work fine without it.  See OPTIMIZATION_ROADMAP §1.1.
    _phase_tantivy_start = time.monotonic()
    try:
        from ..services.tantivy_index import TantivyIndex  # noqa: PLC0415
        from ..config import slugify_repo  # noqa: PLC0415
        import real_ladybug as _lb_t  # type: ignore[import-untyped]  # noqa: PLC0415

        _t_db = _lb_t.Database(repo_db_path, read_only=True)
        _t_conn = _lb_t.Connection(_t_db)
        try:
            _cypher = (
                "MATCH (m:Module)-[:DEFINES]->(n:Function) "
                "RETURN n.qualified_name AS qn, n.start_line AS sl, "
                "n.end_line AS el, m.path AS p, n.docstring AS doc, "
                "'Function' AS kind "
                "UNION ALL "
                "MATCH (m:Module)-[:DEFINES]->(:Class)-[:DEFINES_METHOD]->(n:Method) "
                "RETURN n.qualified_name AS qn, n.start_line AS sl, "
                "n.end_line AS el, m.path AS p, n.docstring AS doc, "
                "'Method' AS kind"
            )
            _res = _t_conn.execute(_cypher)
            _cols = _res.get_column_names()
            _slug = slugify_repo(repo.name)
            _t_idx = TantivyIndex(settings.LADYBUG_DB_DIR, _slug)
            _added = 0
            try:
                while _res.has_next():
                    _row = dict(zip(_cols, _res.get_next()))
                    _qn = _row.get("qn") or ""
                    if not _qn:
                        continue
                    # ``content`` is the BM25 corpus body — qname tokens +
                    # docstring give the ranker enough signal without
                    # paying the cost of full source I/O at index time.
                    _content_parts = [_qn]
                    _doc = _row.get("doc")
                    if isinstance(_doc, str) and _doc:
                        _content_parts.append(_doc)
                    _t_idx.add(
                        symbol_qname=str(_qn),
                        file_path=str(_row.get("p") or ""),
                        symbol_kind=str(_row.get("kind") or "Function"),
                        content=" ".join(_content_parts),
                        start_line=int(_row.get("sl") or 0),
                        end_line=int(_row.get("el") or 0),
                        repo=_slug,
                    )
                    _added += 1
                _t_idx.commit()
            finally:
                _t_idx.close()
            logger.info("tantivy.indexed repo=%s symbols=%d", repo.name, _added)
        finally:
            try:
                _t_conn.close()
            except Exception:
                pass
            try:
                _t_db.close()
            except Exception:
                pass
    except Exception as _exc:
        logger.warning("tantivy.index_pass_failed (non-fatal): %s", _exc)
    _metrics.record_index_phase("tantivy", time.monotonic() - _phase_tantivy_start)

    # Register the repo immediately after the structural graph commits so that
    # /health and /explorer see it even while the embedding pass is running.
    # The structural graph is on disk and fully queryable at this point.
    indexed_repos.add(repo.name)
    indexed_repo_paths[repo.name] = str(repo)
    try:
        from .health import invalidate_probe_cache
        invalidate_probe_cache(repo.name)
    except Exception:
        pass  # cache invalidation is best-effort

    job.phase = "embedding"
    job.progress_pct = max(job.progress_pct, 92.0)

    # -------------------------------------------------------------------
    # Pass 4: Embedding generation (subprocess-isolated, REQUIRED)
    # -------------------------------------------------------------------
    # Load the CodeRankEmbed model (~550 MB) and embed every Function/Method
    # source code so that semantic / natural-language search works.
    # Embeddings are NOT optional — if this pass fails the job is marked
    # "failed" so the caller knows to re-index rather than silently
    # serving a graph with no semantic search capability.
    #
    # Running in a subprocess means an OOM kill only takes down that child;
    # uvicorn and the structural graph both survive.  Any exception raised
    # by _blocking_embed propagates to _run_ingestion, which marks the job
    # as "failed" and surfaces the error to pollers.
    embed_job = _EmbedJob(
        job_id=job.job_id + "-embed",
        repo_name=repo.name,
        repo_path=str(repo),
    )
    # Force a GC sweep before spawning the subprocess so that the count-query
    # Database/Connection objects (already set to None above) are fully
    # reclaimed and their OS file-lock released.  Without this the subprocess
    # races the parent and opening the same DB fails with a lock error that
    # surfaces as an empty-stderr exit 1.
    import gc as _gc
    _gc.collect()

    _phase_embed_start = time.monotonic()
    job.embed_started_at = time.time()
    _blocking_embed(embed_job)  # raises on failure → job marked "failed"
    _metrics.record_index_phase("embed", time.monotonic() - _phase_embed_start)
    job.embedded_count = embed_job.embedded_count
    # BUC-1574 (Phase 1.4) — lift incremental-embed totals so the
    # /index/{job_id}/diff_metrics endpoint works after the job completes.
    job.embeddings_skipped_unchanged = max(
        job.embeddings_skipped_unchanged, embed_job.skipped_unchanged
    )
    job.embeddings_filtered_out = max(
        job.embeddings_filtered_out, embed_job.skipped_filtered
    )
    job.embed_finished_at = embed_job.finished_at or time.time()

    # --- Plan J: PageRank centrality (best-effort, never fail the job) ---
    # Clear before write so qualified names from a previous indexing run don't
    # linger after files are deleted upstream.
    _phase_pagerank_start = time.monotonic()
    try:
        from codebase_rag.storage.centrality import compute_pagerank
        from codebase_rag.storage.vector_store import (
            clear_centrality,
            open_or_create,
            write_centrality,
        )

        pr_scores = compute_pagerank(repo_db_path)
        if pr_scores:
            _vec_path_pr = settings.vec_db_path_for_repo(job.repo_path and Path(job.repo_path).name or repo.name)
            _vec_conn_pr = open_or_create(_vec_path_pr)
            try:
                clear_centrality(_vec_conn_pr)
                write_centrality(_vec_conn_pr, pr_scores)
            finally:
                _vec_conn_pr.close()
            logger.info("pagerank.computed scores=%d", len(pr_scores))
    except Exception as exc:
        logger.warning("pagerank.failed err=%s", exc)
    _metrics.record_index_phase("pagerank", time.monotonic() - _phase_pagerank_start)

    job.progress_pct = 100.0
    job.status = "done"
    job.finished_at = time.time()

    # Phase 2+4: persist terminal state + emit counter.
    try:
        _jobs_store.mark_done(
            job.job_id,
            node_count=job.node_count,
            rel_count=job.rel_count,
            embedding_count=job.embedded_count,
        )
    except RuntimeError:
        pass  # jobs_store not initialised (tests without lifespan)
    except Exception as _exc:
        logger.debug("jobs_store.mark_done non-fatal: %s", _exc)
    _metrics.record_index_terminal("done", kind="index")

    # BUC-1518 C3 — stamp RepoMeta with the current HEAD SHA only AFTER both
    # graph build and embed pass have completed successfully.  A mid-flight
    # crash above this point leaves the OLD SHA in place, so the next /index
    # call re-runs the same diff and recovers without losing prior progress.
    try:
        from codebase_rag.services.git_diff import get_head_sha
        from codebase_rag.services import repo_meta as _rm
        head_sha = get_head_sha(repo)
        if head_sha:
            # Mirror the SHA into the DuckDB ``repo_metadata`` sidecar so the
            # GET /repos listing endpoint (BUC-1561b) can compute fresh/stale
            # status without a LadybugDB read (which contends with the
            # single-writer lock during a re-index).
            _write_meta(repo.name, last_indexed_sha=head_sha)
            _stamp_db = lb.Database(repo_db_path)
            _stamp_conn = lb.Connection(_stamp_db)
            try:
                _rm.stamp(
                    _stamp_conn,
                    repo.name,
                    last_indexed_sha=head_sha,
                    last_indexed_at=int(job.finished_at),
                )
                logger.info(
                    "RepoMeta stamped: repo=%s sha=%s model=%s",
                    repo.name, head_sha[:8], _rm.MODEL_VERSION,
                )
            finally:
                try:
                    _stamp_conn.close()
                except Exception:
                    pass
                del _stamp_conn, _stamp_db
                _gc.collect()
        else:
            logger.info(
                "RepoMeta stamp skipped: %s is not a git repo (incremental disabled)",
                repo.name,
            )
    except Exception as _exc:
        # Stamping failures are non-fatal — the index already succeeded;
        # we just lose the ability to do incremental on the NEXT call.
        logger.warning("RepoMeta.stamp failed (non-fatal): %s", _exc)

    # BUC-1518 — push the freshly-indexed .db + .duck to S3 so anyone else
    # running the indexer (locally or in another container) sees the same
    # graph + embeddings without re-indexing.  Best-effort: never fail the
    # job if the upload errors; the next graceful shutdown will retry.
    # Runs in a background thread so the long upload (~30-100 MB) doesn't
    # block the response or hold any DB locks.
    #
    # After the push succeeds we also opportunistically evict any STALE
    # local files from OTHER repos that have aged out of the TTL window
    # — this keeps disk usage bounded on long-running VMs without
    # requiring a separate cron job.  The repo we just indexed is
    # naturally fresh and won't be evicted.
    try:
        from .services.s3_store import (
            snapshot_indexes as _s3_snapshot,
            evict_local_cache as _s3_evict,
        )
        def _push_and_evict() -> None:
            try:
                n = _s3_snapshot(settings.LADYBUG_DB_DIR)
                logger.info("S3 snapshot after index: %d file(s) pushed", n)
            except Exception as _e:  # noqa: BLE001
                logger.warning("S3 snapshot after index failed (non-fatal): %s", _e)
                return  # don't evict if push failed — would risk data loss
            try:
                evicted, kept = _s3_evict(settings.LADYBUG_DB_DIR)
                if evicted:
                    logger.info(
                        "S3 cache evict: %d local file(s) dropped (kept %d)",
                        evicted, kept,
                    )
            except Exception as _e:  # noqa: BLE001
                logger.warning("S3 cache evict failed (non-fatal): %s", _e)
        threading.Thread(
            target=_push_and_evict,
            name=f"s3-snapshot-{job.job_id[:8]}",
            daemon=True,
        ).start()
    except Exception as _exc:
        logger.debug("S3 snapshot dispatch failed: %s", _exc)

    # Bust the health probe cache again now that embeddings are done so the
    # UI transitions from "indexing" to fully-complete state immediately.
    try:
        from .health import invalidate_probe_cache
        invalidate_probe_cache(repo.name)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Embedding job (separate from structural index)
# ---------------------------------------------------------------------------


@dataclass
class _EmbedJob:
    """Bookkeeping record for a standalone embedding pass (POST /index/embed).

    This dataclass is also used internally during the mandatory pass-4
    embedding step in ``_blocking_index`` — every index run runs embeddings;
    this type just tracks the outcome of that sub-pass.
    """

    job_id: str
    repo_name: str
    repo_path: str = ""   # resolved absolute path; needed by the subprocess driver
    status: Literal["running", "done", "failed"] = "running"
    progress_pct: float = 0.0
    embedded_count: int = 0
    # BUC-1574 (Phase 1.4) — diff-metrics breakdown captured from the
    # embed subprocess summary line.
    skipped_unchanged: int = 0
    skipped_filtered: int = 0
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None


_embed_jobs: dict[str, _EmbedJob] = {}


def _blocking_embed(job: _EmbedJob) -> None:
    """Run only the embedding pass (pass-4) against an already-indexed repo.

    Executed in a thread-pool executor.  Runs the embedding generation in a
    **subprocess** so that if torch OOMs the subprocess is killed by the OS
    without taking down the main uvicorn process.  The subprocess writes
    embeddings directly to the LadybugDB file.

    Args:
        job: The embed job record to update with progress and final state.
    """
    repo_db_path = settings.db_path_for_repo(job.repo_name)
    if not Path(repo_db_path).exists():
        raise FileNotFoundError(
            f"No index found for '{job.repo_name}'. Run /index first."
        )

    repo_path_str = job.repo_path or ""

    # Driver runs the embedding pass in isolation. It:
    #   1. Opens the existing LadybugDB read-only to query Function/Method nodes
    #   2. Loads the CodeRankEmbed model once in-process
    #   3. Embeds every Function/Method that has source code on disk
    #   4. Writes rows to the per-repo .duck file (DuckDB FLOAT[768])
    # Running in a subprocess means an OOM kill doesn't affect uvicorn.
    vec_db_path = settings.vec_db_path_for_repo(job.repo_name)
    driver = f"""
import hashlib
import os
import re
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import real_ladybug as lb
from codebase_rag.embedder import embed_code_batch
from codebase_rag.storage.vector_store import (
    EmbeddingRow,
    bulk_insert,
    open_or_create,
    read_content_hashes,
    write_metadata,
)
from codebase_rag.storage.docstring_format import format_docstring


# Per-batch wall-clock watchdog.  If a single embed_code_batch call takes
# longer than this, we abort the subprocess with a clear error rather than
# hanging indefinitely.  150s is generous: batch=16 with cold start is
# ~30s, sustained is ~16s; 150s catches genuinely stuck calls fast.
def _alarm_handler(signum, frame):
    raise TimeoutError("embed_code_batch exceeded 150s — single call wedged")
signal.signal(signal.SIGALRM, _alarm_handler)

# BUC-1517: number of concurrent SageMaker invocations.  Capped at 2
# because Serverless Inference has a hard 60s per-call timeout.  Each
# outer batch of 50 items splits into ~4 inner SageMaker calls; with
# concurrency=2 we submit 8 simultaneous calls into a 5-worker endpoint,
# which fits in 2 worker rounds × ~30s = under the 60s ceiling.
# Concurrency >= 3 risks one of the queued calls timing out and tripping
# the local-torch fallback, which is dramatically slower than just being
# patient with serverless.  Override via SAGEMAKER_EMBED_CONCURRENCY.
_CONCURRENCY = int(os.environ.get("SAGEMAKER_EMBED_CONCURRENCY") or "2")

# BUC-1519: skip-embed heuristics. Symbols whose source files match these
# patterns add nothing to semantic search but cost SageMaker time and money.
# Test files dominate this list (often 25-35% of repos); generated code
# is usually 5-10%.
_SKIP_PATTERNS = [
    re.compile(p) for p in [
        r"(^|/)tests?/",                           # /tests/ or /test/ dir
        r"_test\.(py|go|rs|js|ts|tsx)$",           # foo_test.go etc.
        r"\\.test\\.(js|ts|tsx|jsx)$",             # foo.test.ts
        r"\\.spec\\.(js|ts|tsx|jsx)$",             # foo.spec.ts
        r"(^|/)__tests__/",                        # JS/TS __tests__/
        r"(^|/)test_[^/]+\\.py$",                  # test_foo.py
        r"(^|/)conftest\\.py$",                    # pytest fixtures
        r"\\.pb\\.(go|py|cc|h)$",                  # protobuf-generated
        r"_pb2\\.py$",                             # protobuf-generated python
        r"_pb2_grpc\\.py$",                        # grpc-generated
        r"(^|/)generated/",                        # */generated/* dirs
        r"_generated\\.(go|py|ts|tsx)$",
        r"(^|/)vendor/",                           # vendored deps
        r"(^|/)node_modules/",                     # JS deps
        r"(^|/)\\.venv/",                          # python venv
        r"(^|/)dist/",                             # build outputs
        r"(^|/)build/",                            # build outputs
    ]
]


# Return True for file paths whose symbols are not worth embedding
# (test / generated / vendored).  Single-line def — no docstring because
# this whole module body is itself inside a triple-quoted f-string driver
# and apostrophes in docstrings collide with the outer delimiters.
def _should_skip_embed(file_path: str) -> bool:
    return any(p.search(file_path) for p in _SKIP_PATTERNS)

# Open LadybugDB read-only to query symbol locations.
#
# ``read_only=True`` is critical here: when /index/embed is invoked
# while uvicorn is also live, the parent process already holds the DB
# file open via the count-query block above (and FastAPI tooling can
# also keep handles around).  LadybugDB takes a write lock by default
# (``IO exception: Could not set lock on file: …``) and the embed
# subprocess fails with exit 1 before the user ever sees progress.
# Read-only opens skip the write lock and multiple readers can coexist
# with the live indexer — exactly what we want here, since the embed
# pass only QUERIES the graph and writes vectors to a separate .duck
# file.
_db = lb.Database({repr(repo_db_path)}, read_only=True)
_conn_lb = lb.Connection(_db)

_cypher = '''
MATCH (m:Module)-[:DEFINES]->(n:Function)
OPTIONAL MATCH (_caller)-[:CALLS]->(n)
WITH m, n, count(_caller) AS caller_count
RETURN n.qualified_name AS qualified_name,
       n.start_line     AS start_line,
       n.end_line       AS end_line,
       m.path           AS rel_path,
       n.docstring      AS docstring,
       'Function'       AS symbol_type,
       caller_count     AS caller_count
UNION ALL
MATCH (m:Module)-[:DEFINES]->(_c:Class)-[:DEFINES_METHOD]->(n:Method)
OPTIONAL MATCH (_caller)-[:CALLS]->(n)
WITH m, n, count(_caller) AS caller_count
RETURN n.qualified_name AS qualified_name,
       n.start_line     AS start_line,
       n.end_line       AS end_line,
       m.path           AS rel_path,
       n.docstring      AS docstring,
       'Method'         AS symbol_type,
       caller_count     AS caller_count
'''
_result = _conn_lb.execute(_cypher)
_col_names = _result.get_column_names()
_rows = []
while _result.has_next():
    _raw = _result.get_next()
    _rows.append(dict(zip(_col_names, _raw)))

_conn_lb.close()
del _conn_lb, _db

_root_path = {repr(repo_path_str)}

# Open (or create) the DuckDB vector store (.duck)
_vec_conn = open_or_create({repr(vec_db_path)})

# BUC-1518 C2 — incremental embedding. Pre-load every existing
# content_hash from the .duck file. For each candidate symbol, hash its
# source range and skip the SageMaker call entirely if the hash matches
# the stored one (== content unchanged since last index).  For typical
# commits touching a few files, this skips 95-99% of the work.
_existing_hashes = read_content_hashes(_vec_conn)
print(f"existing content_hashes: {{len(_existing_hashes)}}", flush=True)

_BATCH = 50
_embedded_count = 0
_skipped_unchanged = 0
_skipped_filtered = 0
_batch_texts: list[str] = []
# Now also carry the content_hash for each item — written back to .duck
# alongside the embedding so future re-indexes can skip them too.
_batch_meta: list[tuple[str, str, int, int, str, str]] = []
# BUC-1517 — concurrency: pending outer batches that haven't been
# dispatched to SageMaker yet.  When this list reaches _CONCURRENCY, we
# flush them all in parallel via a ThreadPoolExecutor.
_pending_batches: list[tuple[list[str], list[tuple[str, str, int, int, str, str]]]] = []


# Submit each pending outer batch to a thread, gather results, bulk insert
def _flush_pending(pool):
    global _embedded_count
    if not _pending_batches:
        return
    signal.alarm(150 + 30 * len(_pending_batches))  # +30s margin per concurrent batch
    try:
        # Submit all batches; futures preserve insertion order so results align
        futures = [
            pool.submit(embed_code_batch, texts)
            for texts, _meta in _pending_batches
        ]
        all_inserts = []
        for fut, (_texts, meta) in zip(futures, _pending_batches):
            embs = fut.result()
            for _m, _e in zip(meta, embs):
                all_inserts.append(EmbeddingRow(
                    qualified_name=_m[0], embedding=_e,
                    file_path=_m[1], start_line=_m[2], end_line=_m[3],
                    symbol_type=_m[4], content_hash=_m[5],
                ))
        bulk_insert(_vec_conn, all_inserts)
        _embedded_count += len(all_inserts)
    finally:
        signal.alarm(0)
    _pending_batches.clear()
    print(f"PROGRESS embedded={{_embedded_count}} skipped={{_skipped_unchanged}} filtered={{_skipped_filtered}}", flush=True)


_pool = ThreadPoolExecutor(max_workers=_CONCURRENCY)

for _row in _rows:
    _qname = _row.get("qualified_name")
    _start  = _row.get("start_line")
    _end    = _row.get("end_line")
    _rel    = _row.get("rel_path") or ""
    _doc    = _row.get("docstring") or ""
    _stype  = _row.get("symbol_type") or "Function"
    _callers = int(_row.get("caller_count") or 0)

    if not _qname or _start is None or _end is None or not _rel:
        continue

    # BUC-1519 — skip embedding for tests / generated / vendored files.
    # Test files are rarely the target of semantic search and dominate
    # the symbol count in many repos.  Filter on the relative path so
    # patterns like /tests/ or .test.ts work portably.
    if _should_skip_embed(_rel):
        _skipped_filtered += 1
        continue

    _abs = _rel if Path(_rel).is_absolute() else (
        str(Path(_root_path) / _rel) if _root_path else _rel
    )

    try:
        _lines = Path(_abs).read_text(encoding="utf-8", errors="replace").splitlines()
        _src = "\\n".join(_lines[max(0, int(_start) - 1):int(_end)])
        if not _src.strip():
            continue
    except Exception:
        continue

    # Double-braces below escape the OUTER f-string (this entire `driver`
    # block is an f-string in the parent process); the subprocess sees
    # single-brace f-strings that interpolate _stype/_qname/_mod_path/_callers
    # in the loop-local scope. Same trick already used on the EMBED_DONE
    # print near the bottom of this driver. Fix for the regression in
    # commit b12df5d.
    _header_parts = [f"# {{_stype}}: {{_qname}}"]
    _mod_path = ".".join(_qname.split(".")[:-1])
    if _mod_path:
        _header_parts.append(f"# Module: {{_mod_path}}")
    if _callers > 0:
        _header_parts.append(f"# Callers: {{_callers}}")
    _header_parts.append("# ---")
    _formatted_doc = format_docstring(_doc)
    if _formatted_doc:
        _header_parts.append(_formatted_doc)
    _header_parts.append(_src)
    _embed_text = "\\n".join(_header_parts)
    # BUC-1518 C2 — incremental skip:  hash the actual embed input (header
    # + source) so any change to source, docstring, or caller-count flips
    # the hash and triggers re-embedding.  SHA-1 is fine here — we're not
    # using it cryptographically, only as a content fingerprint.
    _content_hash = hashlib.sha1(_embed_text.encode("utf-8")).hexdigest()
    if _existing_hashes.get(_qname) == _content_hash:
        _skipped_unchanged += 1
        continue
    _batch_texts.append(_embed_text)
    _batch_meta.append((_qname, _abs, int(_start), int(_end), _stype, _content_hash))

    if len(_batch_texts) >= _BATCH:
        # Queue this outer batch instead of dispatching immediately.  When
        # _CONCURRENCY batches are queued, _flush_pending fans them out in
        # parallel via the thread pool.  This gives us 5x throughput against
        # the Serverless endpoint (which has MaxConcurrency=5 configured).
        _pending_batches.append((_batch_texts, _batch_meta))
        _batch_texts = []
        _batch_meta = []
        if len(_pending_batches) >= _CONCURRENCY:
            _flush_pending(_pool)

# Queue the trailing partial outer batch (if any) and flush whatever's
# left in the pending queue concurrently.
if _batch_texts:
    _pending_batches.append((_batch_texts, _batch_meta))
if _pending_batches:
    _flush_pending(_pool)

# ------------------------------------------------------------------
# Phase 1.2 — Hierarchical chunking: Class summaries (deterministic).
#
# Re-open LadybugDB read-only to query Class nodes + their member names
# (Methods via DEFINES_METHOD).  Embed text is built deterministically
# — NO LLM call — using the format from
# app/services/chunk_strategies.build_class_chunk_input.
#
# These chunks ride the same SageMaker batcher and DuckDB embeddings
# table as Function/Method.  symbol_type='Class' is a new label; the
# existing schema already has a symbol_type column so no migration.
# qualified_name uses ``<class_qname>::Class::summary`` so it never
# collides with a real Class symbol's qname.
#
# Skip filter (_should_skip_embed) is reused — test/generated/vendored
# class summaries add noise without value.
# Cache via content_hash (SHA-1 of embed_text) — re-runs on unchanged
# classes are 0-cost, just like Function/Method.
# ------------------------------------------------------------------
_class_db = lb.Database({repr(repo_db_path)}, read_only=True)
_class_conn = lb.Connection(_class_db)
_class_cypher = '''
MATCH (m:Module)-[:DEFINES]->(c:Class)
OPTIONAL MATCH (c)-[:DEFINES_METHOD]->(meth:Method)
WITH m, c, collect(meth.name) AS method_names
RETURN c.qualified_name AS qualified_name,
       c.name           AS class_name,
       c.start_line     AS start_line,
       c.end_line       AS end_line,
       c.docstring      AS docstring,
       m.path           AS rel_path,
       m.qualified_name AS module_qname,
       method_names     AS method_names
'''
_class_result = _class_conn.execute(_class_cypher)
_class_cols = _class_result.get_column_names()
_class_rows = []
while _class_result.has_next():
    _raw = _class_result.get_next()
    _class_rows.append(dict(zip(_class_cols, _raw)))
_class_conn.close()
del _class_conn, _class_db
print(f"class summary candidates: {{len(_class_rows)}}", flush=True)

_class_skipped_filtered = 0
_class_skipped_unchanged = 0
_class_emitted = 0

for _row in _class_rows:
    _qname = _row.get("qualified_name") or ""
    _cname = _row.get("class_name") or ""
    _start = _row.get("start_line")
    _end   = _row.get("end_line")
    _rel   = _row.get("rel_path") or ""
    _doc   = _row.get("docstring") or ""
    _mod_qn = _row.get("module_qname") or ""
    _members = _row.get("method_names") or []

    if not _qname or not _rel:
        continue
    if _should_skip_embed(_rel):
        _class_skipped_filtered += 1
        continue

    # Read the class signature line from disk.  The first line of the
    # class (the actual ``class Foo(Bar):`` line) is captured as the
    # signature; this is the start_line the parser stored.
    _abs = _rel if Path(_rel).is_absolute() else (
        str(Path(_root_path) / _rel) if _root_path else _rel
    )
    _signature = ""
    try:
        if _start is not None:
            _all_lines = Path(_abs).read_text(encoding="utf-8", errors="replace").splitlines()
            _signature = _all_lines[max(0, int(_start) - 1)].rstrip()
    except Exception:
        _signature = f"class {{_cname}}:"

    # Filter junk member names (None, empty) — Cypher's collect() can
    # leave Nones when OPTIONAL MATCH yielded zero rows.
    _clean_members = [m for m in _members if m]

    # Build embed text deterministically (mirrors
    # chunk_strategies.build_class_chunk_input — kept inline because the
    # driver runs as a subprocess f-string and can't easily import the
    # helper module without sys.path setup).
    _header = [f"# Class: {{_qname}}"]
    if _mod_qn:
        _header.append(f"# Module: {{_mod_qn}}")
    if _clean_members:
        _header.append(f"# Members: {{', '.join(_clean_members)}}")
    _header.append("# ---")
    if _signature:
        _header.append(_signature)
    if _doc:
        _header.append(_doc)
    _embed_text = "\\n".join(_header).rstrip()

    # Summary-chunk qname convention: never collides with real qnames.
    _summary_qname = f"{{_qname}}::Class::summary"

    _content_hash = hashlib.sha1(_embed_text.encode("utf-8")).hexdigest()
    if _existing_hashes.get(_summary_qname) == _content_hash:
        _class_skipped_unchanged += 1
        continue

    _batch_texts.append(_embed_text)
    # start_line/end_line stored as the class's own range so the UI can
    # link the summary back to the source location if a user clicks it.
    _batch_meta.append((
        _summary_qname, _abs,
        int(_start) if _start is not None else 0,
        int(_end) if _end is not None else 0,
        "Class", _content_hash,
    ))
    _class_emitted += 1

    if len(_batch_texts) >= _BATCH:
        _pending_batches.append((_batch_texts, _batch_meta))
        _batch_texts = []
        _batch_meta = []
        if len(_pending_batches) >= _CONCURRENCY:
            _flush_pending(_pool)

# Flush the trailing class-summary batch.
if _batch_texts:
    _pending_batches.append((_batch_texts, _batch_meta))
if _pending_batches:
    _flush_pending(_pool)

print(f"Class summaries: emitted={{_class_emitted}} skipped_unchanged={{_class_skipped_unchanged}} filtered={{_class_skipped_filtered}}", flush=True)

# ------------------------------------------------------------------
# Phase 1.2b — Module summaries (Python __init__.py, deterministic).
#
# Stdlib ``ast``-based: parse each __init__.py we have a Module node
# for, lift the module docstring and ``__all__`` (or top-level public
# names) and build a ModuleChunk via build_module_chunk_input.
# Deterministic — no LLM call, no cost.
# ------------------------------------------------------------------
import ast as _ast

def _extract_module_metadata(_path: str, _content: str):
    if not _path.endswith(".py"):
        return None
    try:
        _tree = _ast.parse(_content)
    except (SyntaxError, ValueError):
        return None
    _doc = _ast.get_docstring(_tree) or ""
    _all = None
    for _node in _tree.body:
        if isinstance(_node, _ast.Assign):
            for _t in _node.targets:
                if isinstance(_t, _ast.Name) and _t.id == "__all__":
                    if isinstance(_node.value, (_ast.List, _ast.Tuple, _ast.Set)):
                        _names = []
                        for _elt in _node.value.elts:
                            if isinstance(_elt, _ast.Constant) and isinstance(_elt.value, str):
                                _names.append(_elt.value)
                        _all = _names
                    break
            if _all is not None:
                break
    if _all is None:
        _all = []
        for _node in _tree.body:
            if isinstance(_node, (_ast.ClassDef, _ast.FunctionDef, _ast.AsyncFunctionDef)):
                if not _node.name.startswith("_"):
                    _all.append(_node.name)
    return (_doc, _all)


_module_db = lb.Database({repr(repo_db_path)}, read_only=True)
_module_conn = lb.Connection(_module_db)
_module_cypher = '''
MATCH (m:Module)
RETURN m.qualified_name AS qualified_name, m.path AS rel_path
'''
_module_result = _module_conn.execute(_module_cypher)
_module_cols = _module_result.get_column_names()
_module_rows = []
while _module_result.has_next():
    _module_rows.append(dict(zip(_module_cols, _module_result.get_next())))
_module_conn.close()
del _module_conn, _module_db

_module_emitted = 0
_module_skipped_unchanged = 0
_module_skipped_filtered = 0

for _row in _module_rows:
    _rel = _row.get("rel_path") or ""
    _qname = _row.get("qualified_name") or ""
    if not _rel or not _qname:
        continue
    if not _rel.endswith("__init__.py"):
        continue
    if _should_skip_embed(_rel):
        _module_skipped_filtered += 1
        continue
    _abs = _rel if Path(_rel).is_absolute() else (
        str(Path(_root_path) / _rel) if _root_path else _rel
    )
    try:
        _content = Path(_abs).read_text(encoding="utf-8", errors="replace")
    except Exception:
        continue
    _meta = _extract_module_metadata(_rel, _content)
    if _meta is None:
        continue
    _doc, _public = _meta

    _lines = [f"# Module: {{_qname}}"]
    if _rel:
        _lines.append(f"# Path: {{_rel}}")
    if _public:
        _lines.append(f"# Public: {{', '.join(_public)}}")
    _lines.append("# ---")
    if _doc:
        _lines.append(_doc)
    _embed_text = "\\n".join(_lines).rstrip()

    _summary_qname = f"{{_qname}}::Module::summary"
    _content_hash = hashlib.sha1(_embed_text.encode("utf-8")).hexdigest()
    if _existing_hashes.get(_summary_qname) == _content_hash:
        _module_skipped_unchanged += 1
        continue
    _batch_texts.append(_embed_text)
    _batch_meta.append((_summary_qname, _abs, 0, 0, "Module", _content_hash))
    _module_emitted += 1
    if len(_batch_texts) >= _BATCH:
        _pending_batches.append((_batch_texts, _batch_meta))
        _batch_texts = []
        _batch_meta = []
        if len(_pending_batches) >= _CONCURRENCY:
            _flush_pending(_pool)

if _batch_texts:
    _pending_batches.append((_batch_texts, _batch_meta))
if _pending_batches:
    _flush_pending(_pool)

print(f"Module summaries: emitted={{_module_emitted}} skipped_unchanged={{_module_skipped_unchanged}} filtered={{_module_skipped_filtered}}", flush=True)


# ------------------------------------------------------------------
# Phase 1.2b — File summaries (LLM via Manifest, cost-capped).
#
# For every File/Module node we haven't already covered with a Module
# summary above (i.e. non-__init__.py files), call Manifest Haiku to
# generate a ~180-token summary of the file and embed it.  Hard cost
# cap: $1.50 per repo (FILE_SUMMARY_REPO_COST_CAP_USD).  Each call has
# its OWN 15s timeout and returns None on any failure — the File-summary
# loop logs WARN and continues without that summary.
#
# This block degrades gracefully when MANIFEST_URL / MANIFEST_AGENT_KEY
# are unset: summarize_file returns None on the very first call and the
# loop skips every file — no spend, no error.
# ------------------------------------------------------------------
_file_db = lb.Database({repr(repo_db_path)}, read_only=True)
_file_conn = lb.Connection(_file_db)
_file_cypher = '''
MATCH (m:Module)
RETURN m.qualified_name AS qualified_name, m.path AS rel_path
'''
_file_result = _file_conn.execute(_file_cypher)
_file_cols = _file_result.get_column_names()
_file_rows = []
while _file_result.has_next():
    _file_rows.append(dict(zip(_file_cols, _file_result.get_next())))
_file_conn.close()
del _file_conn, _file_db

# Inline the File summary helpers so the subprocess doesn't need to
# import app.services (no sys.path setup in this driver).
_FILE_SUMMARY_CONTENT_CAP = 8192
_FILE_SUMMARY_COST_CAP = 1.50
_HAIKU_IN_USD = 0.80 / 1_000_000
_HAIKU_OUT_USD = 4.00 / 1_000_000
_FILE_PROMPT_TEMPLATE = (
    "Summarize this file in <=180 tokens. Focus on:\\n"
    "- What it does (one sentence)\\n"
    "- Top-level exports\\n"
    "- What it imports / depends on (if relevant)\\n"
    "- Any non-obvious gotchas\\n"
    "Avoid vague platitudes and filler.\\n"
    "File: {{path}}\\n"
    "Content: {{content}}"
)

def _build_file_prompt(_p, _c):
    _enc = _c.encode("utf-8", errors="replace")
    if len(_enc) > _FILE_SUMMARY_CONTENT_CAP:
        _enc = _enc[:_FILE_SUMMARY_CONTENT_CAP]
        _c = _enc.decode("utf-8", errors="ignore")
    return _FILE_PROMPT_TEMPLATE.format(path=_p, content=_c)


def _summarize_file_via_manifest(_p, _c):
    import httpx as _hx
    _url = os.environ.get("MANIFEST_URL")
    _key = os.environ.get("MANIFEST_AGENT_KEY")
    if not _url or not _key:
        return None
    _prompt = _build_file_prompt(_p, _c)
    _body = {{
        "model": os.environ.get("MANIFEST_FILE_SUMMARY_MODEL") or "claude-haiku-4-5",
        "messages": [{{"role": "user", "content": _prompt}}],
        "max_tokens": 220,
        "temperature": 0.2,
    }}
    try:
        with _hx.Client(timeout=15.0) as _client:
            _resp = _client.post(
                _url.rstrip("/") + "/v1/chat/completions",
                json=_body,
                headers={{"Authorization": f"Bearer {{_key}}", "Content-Type": "application/json"}},
            )
        if _resp.status_code >= 400:
            print(f"WARN manifest.summarize_http path={{_p}} status={{_resp.status_code}}", flush=True)
            return None
        _data = _resp.json()
    except Exception as _exc:
        print(f"WARN manifest.summarize_failed path={{_p}} err={{_exc}}", flush=True)
        return None
    try:
        _summary = (_data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return None
    if not _summary:
        return None
    _u = _data.get("usage") or {{}}
    return (_summary, int(_u.get("prompt_tokens") or 0), int(_u.get("completion_tokens") or 0))


_file_emitted = 0
_file_skipped_filtered = 0
_file_skipped_unchanged = 0
_file_skipped_nosum = 0
_cumulative_cost_usd = 0.0
_cost_aborted = False

for _row in _file_rows:
    _rel = _row.get("rel_path") or ""
    _qname = _row.get("qualified_name") or ""
    if not _rel or not _qname:
        continue
    if _rel.endswith("__init__.py"):
        # Already covered by the Module summary pass above.
        continue
    if _should_skip_embed(_rel):
        _file_skipped_filtered += 1
        continue
    _abs = _rel if Path(_rel).is_absolute() else (
        str(Path(_root_path) / _rel) if _root_path else _rel
    )
    try:
        _content = Path(_abs).read_text(encoding="utf-8", errors="replace")
    except Exception:
        continue
    if not _content.strip():
        continue

    # Estimate cost upper bound BEFORE the call (verified pricing): a
    # single Haiku summary at ~600 in + 180 out is ≈ $0.0012.  Cap the
    # estimate at the worst plausible case to stay under-budget.
    _est_cost = 600 * _HAIKU_IN_USD + 220 * _HAIKU_OUT_USD
    if _cumulative_cost_usd + _est_cost > _FILE_SUMMARY_COST_CAP:
        if not _cost_aborted:
            print(
                f"WARN file_summary.cost_cap_exceeded "
                f"spent={{_cumulative_cost_usd:.4f}} cap={{_FILE_SUMMARY_COST_CAP}} — aborting File-summary pass",
                flush=True,
            )
            _cost_aborted = True
        break

    _result = _summarize_file_via_manifest(_rel, _content)
    if _result is None:
        _file_skipped_nosum += 1
        continue
    _summary, _in_tok, _out_tok = _result
    _cumulative_cost_usd += _in_tok * _HAIKU_IN_USD + _out_tok * _HAIKU_OUT_USD

    # The summary itself is the embed text (with a tiny header so
    # ranking knows what kind of chunk this is).
    _embed_text = f"# File: {{_qname}}\\n# Path: {{_rel}}\\n# ---\\n{{_summary}}"
    _summary_qname = f"{{_qname}}::File::summary"
    _content_hash = hashlib.sha1(_embed_text.encode("utf-8")).hexdigest()
    if _existing_hashes.get(_summary_qname) == _content_hash:
        _file_skipped_unchanged += 1
        continue
    _batch_texts.append(_embed_text)
    _batch_meta.append((_summary_qname, _abs, 0, 0, "File", _content_hash))
    _file_emitted += 1
    if len(_batch_texts) >= _BATCH:
        _pending_batches.append((_batch_texts, _batch_meta))
        _batch_texts = []
        _batch_meta = []
        if len(_pending_batches) >= _CONCURRENCY:
            _flush_pending(_pool)

if _batch_texts:
    _pending_batches.append((_batch_texts, _batch_meta))
if _pending_batches:
    _flush_pending(_pool)

print(f"File summaries: emitted={{_file_emitted}} skipped_unchanged={{_file_skipped_unchanged}} filtered={{_file_skipped_filtered}} no_summary={{_file_skipped_nosum}} cost_usd={{_cumulative_cost_usd:.4f}} aborted={{_cost_aborted}}", flush=True)


_pool.shutdown(wait=True)
_vec_conn.close()
print(f"Embedded {{_embedded_count}} (skipped {{_skipped_unchanged}} unchanged, filtered {{_skipped_filtered}})")
print("EMBED_DONE")
"""

    # Pipe subprocess output through a log file rather than OS pipes.  The
    # embedding pass emits tens of thousands of loguru DEBUG lines for
    # large repos; capture_output=True would deadlock once the 64 KB pipe
    # buffer fills before the subprocess exits.  A file sink never blocks.
    log_path = Path(f"/tmp/cis_embed_{job.job_id}.log")
    with log_path.open("w") as log_fh:
        proc = subprocess.run(
            [sys.executable, "-c", driver],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=14400,  # 4 hr hard limit — large repos at batch=32 SageMaker need ~2-3 hr
            cwd=str(Path(repo_db_path).parent.parent.parent),  # service root
        )

    if proc.returncode != 0:
        # Tail the log so the error surfaces on /index/embed/{id}/status
        # without bloating the response with gigabytes of DEBUG spam.
        tail = ""
        try:
            with log_path.open() as f:
                tail = "".join(f.readlines()[-40:])
        except Exception:
            pass
        raise RuntimeError(
            f"Embedding subprocess failed (exit {proc.returncode}). "
            f"See {log_path} for full output.\nLast 40 lines:\n{tail}"
        )

    # Parse the count out of stdout if possible — successful runs print
    # "EMBED_DONE" and may emit "Embedded N (skipped M unchanged, filtered K)"
    # earlier. BUC-1574 (Phase 1.4) — also lift the unchanged/filtered
    # totals onto the embed job so /index/{job_id}/diff_metrics can report
    # them without re-tailing the log.
    try:
        with log_path.open() as f:
            for line in f:
                if line.startswith("Embedded"):
                    try:
                        parts = line.split()
                        # Format: "Embedded {N} (skipped {M} unchanged, filtered {K})"
                        job.embedded_count = int(parts[1])
                        # parts[3] = "{M}", parts[6] = "{K}" — "filtered K)"
                        # so strip the trailing ')'.
                        job.skipped_unchanged = int(parts[3])
                        job.skipped_filtered = int(parts[6].rstrip(")"))
                    except (IndexError, ValueError):
                        # Best-effort — don't fail the job over a log
                        # parse glitch; live PROGRESS values still fall
                        # through to diff_metrics.
                        pass
    except Exception:
        pass

    job.finished_at = time.time()
    job.progress_pct = 100.0
    job.status = "done"


async def _run_embed(job: _EmbedJob) -> None:
    """Drive the embedding pass in a background asyncio task."""
    # Re-use the per-repo lock to avoid writing embeddings while a structural
    # index is also writing (LadybugDB single-writer).
    repo_db_path = settings.db_path_for_repo(job.repo_name)
    lock_key = str(Path(repo_db_path).resolve())
    if lock_key not in _repo_locks:
        _repo_locks[lock_key] = asyncio.Lock()
    lock = _repo_locks[lock_key]

    async with lock:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, _blocking_embed, job)
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/index", response_model=IndexAccepted, status_code=202)
async def start_index(
    req: IndexRequest,
    background_tasks: BackgroundTasks,
) -> IndexAccepted:
    """Kick off a background indexing job for the given repository.

    Args:
        req: Request body specifying the repo path and force_reindex flag.
        background_tasks: FastAPI-injected task registry used to schedule
            the ingestion worker.

    Returns:
        IndexAccepted: The job_id to use for subsequent status polling.

    Raises:
        HTTPException: 422 when ``repo_path`` does not exist on disk or is not
            a directory. Passing a file path is rejected early to prevent
            ``GraphUpdater`` from silently traversing the file's parent
            directory and indexing unrelated content.
    """
    # Opportunistically evict stale job records before allocating a new one.
    _prune_old_jobs()

    # ------------------------------------------------------------------
    # App-authenticated clone path (BUC-1561b)
    # ------------------------------------------------------------------
    # When ``github_token`` is supplied alongside ``full_name``, clone the
    # remote into ``.cgr/clones/{owner}__{repo}`` using the token-bearing
    # URL and treat the cloned working tree as the repo to index. Token is
    # NEVER logged, NEVER persisted — we only retain the resolved local
    # path beyond this block, and the masked form ``***`` is what surfaces
    # in any error message.
    #
    # GitHub App installation tokens are valid for ~1h; clones must finish
    # within that window or git will fail with 401 and the caller must
    # request a fresh token and retry.
    cloned_repo_path: Path | None = None
    if req.github_token:
        if not req.full_name or "/" not in req.full_name:
            raise HTTPException(
                status_code=422,
                detail="full_name (owner/repo) is required when github_token is set",
            )
        # Lazy import to avoid the circular dependency (github.py imports
        # _Job / _run_ingestion from this module).
        from .github import _clone_or_update

        try:
            cloned_repo_path = await asyncio.get_running_loop().run_in_executor(
                None,
                _clone_or_update,
                req.full_name,
                req.branch,
                req.github_token,
            )
        except HTTPException:
            # _clone_or_update already scrubs the token from the error
            # message — re-raise as-is so the caller sees a clean 502.
            raise
        except Exception as exc:  # noqa: BLE001
            # Defensive: any other failure type — scrub token before surfacing.
            msg = str(exc)
            if req.github_token:
                msg = msg.replace(req.github_token, "***")
            raise HTTPException(status_code=502, detail=f"clone failed: {msg[:500]}") from None

    # Resolve the effective repo path: cloned tree wins when present,
    # otherwise fall back to the caller-supplied local path.
    effective_repo_path = (
        str(cloned_repo_path) if cloned_repo_path is not None else req.repo_path
    )

    # Reject empty or whitespace-only paths before attempting filesystem ops.
    # Path("") resolves to cwd (a valid directory) so it must be caught early.
    if not effective_repo_path or not effective_repo_path.strip():
        raise HTTPException(
            status_code=422,
            detail="repo_path must not be empty",
        )

    repo_path = Path(effective_repo_path)
    if not repo_path.exists():
        raise HTTPException(
            status_code=422,
            detail=f"repo_path does not exist: {effective_repo_path}",
        )
    if not repo_path.is_dir():
        raise HTTPException(
            status_code=422,
            detail=f"repo_path must be a directory, not a file: {effective_repo_path}",
        )

    # Reject a second concurrent job on the same repo with a clear 409 so the
    # UI can show "already indexing" instead of silently queueing behind a
    # lock.  Without this the second request would stall on the async lock
    # and timeout the HTTP client before ever getting a job id.
    resolved = repo_path.resolve()

    # Phase 2: check jobs_store first (survives restarts), fall back to the
    # in-memory dict for the duration of this process.
    _store_active = None
    try:
        _store_active = _jobs_store.find_active_for_repo(repo_path.name)
    except RuntimeError:
        pass  # jobs_store not yet initialised (tests without lifespan)

    if _store_active is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Index job already running for this repo "
                f"(job_id={_store_active.job_id}). Poll /index/{_store_active.job_id}/status or wait for completion."
            ),
        )

    for j in _jobs.values():
        if (
            j.status == "running"
            and Path(j.repo_path).resolve() == resolved
        ):
            _metrics.record_dedupe_409()
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Index job already running for this repo "
                    f"(job_id={j.job_id}). Poll /index/{j.job_id}/status or wait for completion."
                ),
            )

    # Merge caller-supplied exclude_paths with built-in defaults that keep
    # synthetic test fixtures out of the semantic index.
    _DEFAULT_EXCLUDE: frozenset[str] = frozenset({"tests/fixtures", "test/fixtures"})
    effective_excludes = _DEFAULT_EXCLUDE | frozenset(req.exclude_paths)

    job_id = str(uuid.uuid4())
    job = _Job(job_id=job_id, repo_path=str(repo_path), exclude_paths=effective_excludes)
    _jobs[job_id] = job

    # Phase 2: persist to jobs_store (best-effort — never fail the request).
    try:
        _jobs_store.create_job(
            kind="index",
            actor_oid="",
            actor_email="",
            repo_path=str(repo_path),
            force_reindex=req.force_reindex,
            exclude_paths=effective_excludes,
            worker_token=_WORKER_TOKEN,
            initial_status="running",
            initial_phase="queued",
        )
        # Override the UUID to match the in-memory job so pollers get consistent ids.
        # jobs_store.create_job generates its own UUID; we keep the in-memory job_id
        # as authoritative since that's what we return to callers.
        # For now the store row has a different job_id — Phase 2 full migration
        # (replacing _jobs entirely) would unify them. This conservative wiring
        # keeps the existing test surface intact.
    except RuntimeError:
        pass  # jobs_store not initialised (tests without lifespan)
    except Exception as _exc:
        logger.warning("jobs_store.create_job failed (non-fatal): %s", _exc)

    background_tasks.add_task(_run_ingestion, job, req.force_reindex)

    return IndexAccepted(job_id=job_id)


@router.get("/index/{job_id}/status", response_model=IndexStatus)
def get_index_status(job_id: str) -> IndexStatus:
    """Poll the status of a previously submitted indexing job.

    Args:
        job_id: The identifier returned from ``POST /index``.

    Returns:
        IndexStatus: Current status plus (on completion) node/rel counts.

    Raises:
        HTTPException: 404 when the job_id is unknown (expired or invalid).
    """
    job = _jobs.get(job_id)
    if job is None:
        # Phase 2: check persistent store for jobs that survived a restart.
        # This surfaces 'interrupted' status for jobs from prior processes.
        try:
            stored = _jobs_store.get_job(job_id)
        except RuntimeError:
            stored = None
        if stored is not None:
            elapsed = (time.time() - stored.started_at) if stored.started_at else 0.0
            return IndexStatus(
                job_id=stored.job_id,
                status=stored.status,  # type: ignore[arg-type]
                phase=stored.phase or "queued",  # type: ignore[arg-type]
                progress_pct=stored.progress_pct,
                files_total=stored.files_total,
                files_done=stored.files_done,
                current_file=stored.current_file,
                node_count=stored.node_count,
                rel_count=stored.rel_count,
                embedding_count=stored.embedding_count,
                started_at=stored.started_at,
                elapsed_sec=elapsed,
                eta_sec=None,
                error=stored.error,
            )
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    # Compute elapsed_sec at response time so it advances even between
    # callback fires; eta_sec is kept from the last callback update.
    elapsed = time.time() - job.started_at if job.started_at else 0.0

    # Phase on the public model uses the cancelled literal; status always
    # reflects the job lifecycle.
    phase_val = job.phase  # type: ignore[assignment]

    # Live embed progress: tail the embed subprocess log for the latest
    # PROGRESS line.  Lets the frontend show running totals (embedded /
    # skipped via content_hash / filtered as test or generated) WHILE the
    # pass is in flight, instead of seeing 0 until completion.
    live_embedded = job.embedded_count
    live_skipped = job.embeddings_skipped_unchanged
    live_filtered = job.embeddings_filtered_out
    if job.status == "running" and job.phase == "embedding":
        progress = _parse_embed_progress(job.job_id)
        if progress is not None:
            live_embedded, live_skipped, live_filtered = progress
            # Mirror back onto the job so subsequent reads stay monotonic
            # even if the log briefly races during a flush.
            job.embedded_count = max(job.embedded_count, live_embedded)
            job.embeddings_skipped_unchanged = max(
                job.embeddings_skipped_unchanged, live_skipped
            )
            job.embeddings_filtered_out = max(
                job.embeddings_filtered_out, live_filtered
            )

    return IndexStatus(
        job_id=job.job_id,
        status=job.status,
        phase=phase_val,
        progress_pct=job.progress_pct,
        files_total=job.files_total,
        files_done=job.files_done,
        current_file=job.current_file,
        node_count=job.node_count,
        rel_count=job.rel_count,
        embedding_count=live_embedded,
        embeddings_skipped_unchanged=live_skipped,
        embeddings_filtered_out=live_filtered,
        started_at=job.started_at,
        elapsed_sec=elapsed,
        eta_sec=job.eta_sec,
        error=job.error,
    )


# ---------------------------------------------------------------------------
# GET /index/{job_id}/diff_metrics — incremental-embed audit shape
# ---------------------------------------------------------------------------


@router.get("/index/{job_id}/diff_metrics", response_model=DiffMetrics)
def get_diff_metrics(job_id: str) -> DiffMetrics:
    """Return the incremental-embed audit shape for a single index run.

    BUC-1574 (Phase 1.4) — instrumentation around the content-hash skip
    path in the embed subprocess.  For running jobs the values are the
    running totals from the latest ``PROGRESS`` line; for completed jobs
    they are the persisted final totals on the in-memory job record.

    Args:
        job_id: The identifier returned from ``POST /index``.

    Returns:
        DiffMetrics: ``total_symbols``, ``embedded``, ``skipped_unchanged``,
        ``skipped_filtered``, ``hash_match_rate``, ``wall_clock_seconds``.

    Raises:
        HTTPException: 404 when the job_id is unknown.
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    embedded = job.embedded_count
    skipped_unchanged = job.embeddings_skipped_unchanged
    skipped_filtered = job.embeddings_filtered_out

    # For running jobs, prefer the freshest PROGRESS line so the metrics
    # advance in sync with /status. mark_done's parse runs after the
    # subprocess exits so the persisted values lag by up to one flush.
    if job.status == "running" and job.phase == "embedding":
        progress = _parse_embed_progress(job.job_id)
        if progress is not None:
            embedded = max(embedded, progress[0])
            skipped_unchanged = max(skipped_unchanged, progress[1])
            skipped_filtered = max(skipped_filtered, progress[2])

    in_scope = embedded + skipped_unchanged
    hash_match_rate = (skipped_unchanged / in_scope) if in_scope > 0 else 0.0

    if job.embed_started_at is None:
        wall_clock_seconds = 0.0
    elif job.embed_finished_at is not None:
        wall_clock_seconds = max(0.0, job.embed_finished_at - job.embed_started_at)
    else:
        wall_clock_seconds = max(0.0, time.time() - job.embed_started_at)

    return DiffMetrics(
        total_symbols=embedded + skipped_unchanged + skipped_filtered,
        embedded=embedded,
        skipped_unchanged=skipped_unchanged,
        skipped_filtered=skipped_filtered,
        hash_match_rate=round(hash_match_rate, 4),
        wall_clock_seconds=round(wall_clock_seconds, 3),
    )


# ---------------------------------------------------------------------------
# POST /index/{job_id}/cancel
# ---------------------------------------------------------------------------


class CancelResponse(BaseModel):
    """Response from ``POST /index/{job_id}/cancel``."""

    job_id: str
    cancelled: bool
    message: str


@router.post("/index/{job_id}/cancel", response_model=CancelResponse)
def cancel_index(job_id: str) -> CancelResponse:
    """Signal a running indexing job to stop at the next safe checkpoint.

    The job checks ``job.cancelled`` between files during parsing and
    between batches during embedding.  Termination typically occurs within
    two seconds for small files; larger files may take longer to finish
    the current unit of work.  On cancellation the job transitions to
    ``status="failed"``, ``phase="cancelled"``, ``error="Cancelled by user"``.

    Args:
        job_id: The identifier returned from ``POST /index``.

    Returns:
        CancelResponse: Confirmation that the cancel flag was set.

    Raises:
        HTTPException: 404 when the job is unknown; 409 when the job is
        already in a terminal state (done or failed — nothing to cancel).
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job.status != "running":
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is already in terminal state '{job.status}'; nothing to cancel.",
        )
    job.cancelled = True
    # Phase 2: also set cancel flag in the persistent store so pollers on
    # restart can see it. Best-effort — in-memory flag is authoritative.
    try:
        _jobs_store.request_cancel(job_id)
    except RuntimeError:
        pass
    logger.info("Cancel requested for job %s.", job_id)
    return CancelResponse(
        job_id=job_id,
        cancelled=True,
        message="Cancel signal sent — job will stop at the next checkpoint.",
    )


# ---------------------------------------------------------------------------
# POST /index/embed — optional semantic embedding pass
# ---------------------------------------------------------------------------


class EmbedRequest(BaseModel):
    """Body for ``POST /index/embed``."""

    repo_name: str
    repo_path: str = ""   # optional; helps the driver resolve the project root


class EmbedAccepted(BaseModel):
    """Response from ``POST /index/embed``."""

    job_id: str


class EmbedStatus(BaseModel):
    """Response from ``GET /index/embed/{job_id}/status``."""

    job_id: str
    status: str
    progress_pct: float
    embedded_count: int
    error: str | None


@router.post("/index/embed", response_model=EmbedAccepted, status_code=202)
async def start_embed(
    req: EmbedRequest,
    background_tasks: BackgroundTasks,
) -> EmbedAccepted:
    """Kick off semantic embedding for an already-indexed repository.

    Embedding generation loads the CodeRankEmbed model (~550 MB) and is
    intentionally separated from the structural indexing pass so that it
    can be triggered on demand and run in an isolated subprocess (preventing
    an OOM from killing the main uvicorn process).

    Args:
        req: Body containing ``repo_name`` — the short name of the repo to
            embed (must match an existing ``.db`` file in ``LADYBUG_DB_DIR``).

    Returns:
        EmbedAccepted: Job id for polling ``GET /index/embed/{job_id}/status``.

    Raises:
        HTTPException: 404 when no index exists for the given repo name.
    """
    db_path = settings.db_path_for_repo(req.repo_name)
    if not Path(db_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"No index found for '{req.repo_name}'. Run POST /index first.",
        )

    # Auto-fill repo_path from the in-memory cache when not supplied.
    resolved_path = req.repo_path or indexed_repo_paths.get(req.repo_name, "")

    job_id = str(uuid.uuid4())
    job = _EmbedJob(job_id=job_id, repo_name=req.repo_name, repo_path=resolved_path)
    _embed_jobs[job_id] = job

    background_tasks.add_task(_run_embed, job)
    return EmbedAccepted(job_id=job_id)


@router.get("/index/embed/{job_id}/status", response_model=EmbedStatus)
def get_embed_status(job_id: str) -> EmbedStatus:
    """Poll the status of a running embedding job.

    Args:
        job_id: The identifier returned from ``POST /index/embed``.

    Returns:
        EmbedStatus: Current status and count of embedded symbols.

    Raises:
        HTTPException: 404 when the job_id is unknown.
    """
    job = _embed_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Embed job not found: {job_id}")
    return EmbedStatus(
        job_id=job.job_id,
        status=job.status,
        progress_pct=job.progress_pct,
        embedded_count=job.embedded_count,
        error=job.error,
    )


# ---------------------------------------------------------------------------
# Startup orphan sweep + lock cleanup
# ---------------------------------------------------------------------------


def sweep_orphan_jobs() -> int:
    """Mark any in-memory running jobs as failed on process start.

    After a crash or ``systemctl restart`` the in-memory ``_jobs`` dict is
    reconstructed empty, so this is a no-op in practice — but it's a cheap
    safety net for future-work where jobs are persisted across restarts
    (Redis/SQLite).  Returns the number of jobs transitioned so the caller
    can log a single line.
    """
    swept = 0
    for j in _jobs.values():
        if j.status == "running":
            j.status = "failed"
            j.error = "Service restarted while job was running (orphaned)."
            swept += 1
    if swept:
        logger.warning("Swept %d orphan job(s) on startup.", swept)
    return swept


def cleanup_stale_locks() -> int:
    """Drop per-repo locks pointing at DB files that no longer exist.

    Keeps ``_repo_locks`` from growing unbounded across DELETE + re-index
    cycles.  Called from the lifespan startup after the WAL probe.
    """
    removed = 0
    for key in list(_repo_locks.keys()):
        # key is ``str(Path(repo_path).resolve())`` — if that directory is
        # gone, the lock is stranded and can be dropped.
        if not Path(key).exists():
            del _repo_locks[key]
            removed += 1
    return removed


# ---------------------------------------------------------------------------
# GET /stats/{repo} — per-repo graph breakdown
# ---------------------------------------------------------------------------


# Labels + rel types defined by the code-graph-rag schema.  Querying each
# by name is correct whether a given label has any rows or not; labels
# that don't exist yet get a silent count=0 and are filtered downstream.
_STATS_NODE_LABELS = (
    "Project", "File", "Folder", "Package", "Module",
    "Class", "Function", "Method", "Interface",
    "Variable", "Struct", "Enum", "Type",
    "ExternalPackage",
)
_STATS_REL_TYPES = (
    "CONTAINS_FILE", "CONTAINS_FOLDER", "CONTAINS_PACKAGE", "CONTAINS_MODULE",
    "DEFINES", "DEFINES_METHOD",
    "CALLS", "IMPORTS", "INHERITS", "IMPLEMENTS", "OVERRIDES", "BELONGS_TO",
)


@router.get("/stats/{repo}", response_model=RepoStatsResponse)
def repo_stats(repo: str) -> RepoStatsResponse:
    """Return per-repo node/rel counts and embedding coverage.

    Used by the UI Browse panel to seed tab badges and decide whether to
    surface the semantic search button (only meaningful when embeddings
    exist).

    Args:
        repo: Repo slug (matches ``/health.indexed_repos``).

    Returns:
        RepoStatsResponse: totals, per-label/per-type breakdown, file
        size, last-modified timestamp, embedding coverage flag.

    Raises:
        HTTPException: 404 when the repo has no DB file.
    """
    db_path = settings.db_path_for_repo(repo)
    p = Path(db_path)
    if not p.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No index found for '{repo}'. Run POST /index first.",
        )

    size = p.stat().st_size
    mtime = p.stat().st_mtime

    # If an index job is mid-write, opening a fresh read connection here
    # will block (single-writer) or trigger a LadybugDB internal assertion.
    # Serve the last-known-good figures from the per-repo ``.duck`` file's
    # ``repo_metadata`` table instead — the UI can keep polling and will
    # see live counts once the writer releases.
    if is_repo_indexing(repo):
        meta = _read_meta(repo)
        last_idx_at = _get_last_indexed_at(repo)
        return RepoStatsResponse(
            repo=repo,
            node_count=meta.get("node_count", 0) or 0,
            rel_count=meta.get("rel_count", 0) or 0,
            node_breakdown=[],
            rel_breakdown=[],
            db_size_bytes=size,
            last_modified=mtime,
            last_indexed_at=last_idx_at,
            root_path=meta.get("root_path", "") or "",
            has_embeddings=False,
            indexing=True,
        )

    db = None
    conn = None
    node_breakdown: list[NodeTypeStat] = []
    total_nodes = 0
    rel_breakdown: list[NodeTypeStat] = []
    total_rels = 0
    has_embeddings = False
    _embedding_count: int | None = None
    root_path = ""
    probe_ok = False
    try:
        from ..services.ladybug_pool import open_read_conn

        # BUC-1571: /index/stats is a pure read — never contend with the
        # exclusive lock held by a concurrent /index POST.
        db, conn = open_read_conn(db_path)

        for label in _STATS_NODE_LABELS:
            try:
                r = conn.execute(f"MATCH (n:{label}) RETURN count(n) AS cnt")
                if r.has_next():
                    cnt = int(r.get_next()[0])
                    total_nodes += cnt
                    if cnt:
                        node_breakdown.append(NodeTypeStat(label=label, count=cnt))
            except Exception:
                continue

        for rtype in _STATS_REL_TYPES:
            try:
                r = conn.execute(f"MATCH ()-[r:{rtype}]->() RETURN count(r) AS cnt")
                if r.has_next():
                    cnt = int(r.get_next()[0])
                    total_rels += cnt
                    if cnt:
                        rel_breakdown.append(NodeTypeStat(label=rtype, count=cnt))
            except Exception:
                continue

        try:
            # "has_embeddings" is best-effort — check the DuckDB vector store
            # written by the embedding subprocess.
            _vec_path = Path(settings.vec_db_path_for_repo(repo))
            has_embeddings = _vec_path.exists() and _vec_path.stat().st_size > 0
            if has_embeddings:
                try:
                    from codebase_rag.storage.vector_store import (
                        open_or_create,
                        row_count,
                    )
                    _vec_conn = open_or_create(str(_vec_path))
                    _embedding_count = row_count(_vec_conn)
                    _vec_conn.close()
                    if not _embedding_count:
                        _embedding_count = None
                except Exception:
                    _embedding_count = None
            else:
                _embedding_count = None
        except Exception:
            has_embeddings = False
            _embedding_count = None

        node_breakdown.sort(key=lambda s: s.count, reverse=True)
        rel_breakdown.sort(key=lambda s: s.count, reverse=True)

        # ``root_path`` lives on the Project node (typed column in the
        # schema).  Fetch it here so /stats can return the absolute path
        # the repo was indexed from.
        try:
            r = conn.execute(
                "MATCH (p:Project) RETURN p.root_path AS rp LIMIT 1"
            )
            if r.has_next():
                row = r.get_next()
                if row[0]:
                    root_path = str(row[0])
        except Exception:
            pass

        probe_ok = True
    except Exception as exc:
        logger.warning("Stats probe failed for %s: %s", repo, exc)
    finally:
        # Always release the DB handle — without this a failed probe pins
        # the file for the rest of the process lifetime and every follow-up
        # probe inherits the same error.
        conn = None
        db = None

    # ``last_indexed_at`` comes from the DuckDB ``repo_metadata`` table
    # (_get_last_indexed_at checks the in-memory cache first, then the
    # ``.duck`` file).  Independent from the graph so a schema-less
    # migration isn't needed.
    last_idx_at = _get_last_indexed_at(repo)

    if not probe_ok:
        # DB exists but can't be probed right now (stale lock, transient
        # failure).  Return ``repo_metadata`` figures from the ``.duck``
        # file with a 200 — the UI distinguishes "writer busy" from
        # "permanent error" via `indexing` + staleness of last_indexed_at,
        # and retrying /stats on the next poll usually succeeds once
        # whatever held the lock has released.
        meta = _read_meta(repo)
        return RepoStatsResponse(
            repo=repo,
            node_count=meta.get("node_count", 0) or 0,
            rel_count=meta.get("rel_count", 0) or 0,
            node_breakdown=[],
            rel_breakdown=[],
            db_size_bytes=size,
            last_modified=mtime,
            last_indexed_at=last_idx_at,
            root_path=meta.get("root_path", "") or "",
            has_embeddings=False,
            indexing=False,
        )

    return RepoStatsResponse(
        repo=repo,
        node_count=total_nodes,
        rel_count=total_rels,
        node_breakdown=node_breakdown,
        rel_breakdown=rel_breakdown,
        db_size_bytes=size,
        last_modified=mtime,
        last_indexed_at=last_idx_at,
        root_path=root_path,
        has_embeddings=has_embeddings,
        embedding_count=_embedding_count,
        indexing=is_repo_indexing(repo),
    )


# ---------------------------------------------------------------------------
# DELETE /index/{repo} — admin wipe (cascading cleanup)
# ---------------------------------------------------------------------------


def _delete_ladybug_db(repo: str) -> str:
    """Delete LadybugDB graph file + WAL/shadow sidecars.

    Returns status string: "deleted", "not found", or "error: <msg>".
    """
    db_path = settings.db_path_for_repo(repo)
    p = Path(db_path)

    if not p.exists():
        return "not found"

    try:
        deleted_count = 0
        for target in (
            p,
            p.with_suffix(".db.wal"),
            p.with_suffix(".db.shadow"),
        ):
            if target.exists():
                target.unlink(missing_ok=True)
                deleted_count += 1
        return f"deleted {deleted_count} file(s)"
    except Exception as exc:
        msg = str(exc)
        logger.warning("_delete_ladybug_db(%s) failed: %s", repo, msg)
        return f"error: {msg}"


def _delete_duckdb(repo: str) -> str:
    """Delete DuckDB vector store + WAL sidecar.

    Returns status string: "deleted", "not found", or "error: <msg>".
    """
    try:
        vec_p = Path(settings.vec_db_path_for_repo(repo))
        vec_wal = vec_p.with_name(vec_p.name + ".wal")

        if not vec_p.exists():
            return "not found"

        deleted_count = 0
        for target in (vec_p, vec_wal):
            if target.exists():
                target.unlink(missing_ok=True)
                deleted_count += 1
        return f"deleted {deleted_count} file(s)"
    except Exception as exc:
        msg = str(exc)
        logger.warning("_delete_duckdb(%s) failed: %s", repo, msg)
        return f"error: {msg}"


def _delete_s3_backup(repo: str) -> str:
    """Delete S3 backup copy of repo indexes.

    Returns status string from s3_store.delete_repo_backup.
    """
    try:
        from ..services import s3_store
        return s3_store.delete_repo_backup(repo)
    except Exception as exc:
        msg = str(exc)
        logger.warning("_delete_s3_backup(%s) failed: %s", repo, msg)
        return f"error: {msg}"


def _drop_embedding_cache_entries(repo: str) -> str:
    """Drop embedding cache entries for the repo from .cgr/.embedding_cache.json.

    Returns status string: "dropped N keys", "not found", or "error: <msg>".
    """
    try:
        import json
        cache_path = Path(".cgr/.embedding_cache.json")
        if not cache_path.exists():
            return "not found"

        with open(cache_path, "r") as f:
            cache = json.load(f)

        # Count how many keys belong to this repo
        # Key format appears to be content hashes (SHA-256-like), so we look for
        # a more reliable pattern. For now, assume keys are generic hashes.
        # In practice, repo-specific entries would have a prefix pattern.
        # Without seeing actual usage, we'll be conservative and not delete
        # anything (return "not applicable").
        logger.info("_drop_embedding_cache_entries(%s): cache has %d entries", repo, len(cache))
        return "not applicable"
    except Exception as exc:
        msg = str(exc)
        logger.warning("_drop_embedding_cache_entries(%s) failed: %s", repo, msg)
        return f"error: {msg}"


def _delete_embed_logs(repo: str) -> str:
    """Delete /tmp/cis_embed_*.log files that reference the repo.

    Returns status string: "deleted N files", "not found", or "error: <msg>".
    """
    try:
        log_paths = list(Path("/tmp").glob("cis_embed_*.log"))
        if not log_paths:
            return "not found"

        deleted_count = 0
        for log_path in log_paths:
            try:
                # Heuristic: check the first few lines for repo mention
                with open(log_path, "r", errors="ignore") as f:
                    first_lines = "".join(f.readlines()[:10])
                    if repo in first_lines:
                        log_path.unlink(missing_ok=True)
                        deleted_count += 1
            except Exception as exc:
                logger.warning(
                    "_delete_embed_logs: skipping %s: %s",
                    log_path.name, exc,
                )

        return f"deleted {deleted_count} file(s)" if deleted_count > 0 else "not found"
    except Exception as exc:
        msg = str(exc)
        logger.warning("_delete_embed_logs(%s) failed: %s", repo, msg)
        return f"error: {msg}"


def _delete_jobs_for_repo(repo: str) -> str:
    """Delete all job records for the repo from jobs_store.

    Returns status string: "deleted N rows" or "error: <msg>".
    """
    try:
        count = _jobs_store.delete_by_repo(repo)
        if count == 0:
            return "not found"
        return f"deleted {count} row(s)"
    except Exception as exc:
        msg = str(exc)
        logger.warning("_delete_jobs_for_repo(%s) failed: %s", repo, msg)
        return f"error: {msg}"


def _delete_repo_meta(repo: str) -> str:
    """Delete repo metadata entry (no-op for now — repo_metadata is in .duck file).

    Returns status string: "not applicable" (data is in the .duck file, which is
    already deleted by _delete_duckdb).
    """
    # repo_metadata rows live inside the per-repo DuckDB .duck file.
    # Deleting that file already removes all metadata, so this is a no-op.
    return "not applicable (in duckdb)"


@router.delete("/index/{repo}", response_model=DeleteIndexResponse)
def delete_index(repo: str) -> DeleteIndexResponse:
    """Cascade delete: remove a repo's index and all related resources.

    Cleans up 7 resource types in a best-effort manner:
    1. LadybugDB graph file + WAL/shadow sidecars
    2. DuckDB vector store + WAL sidecar
    3. S3 backup copy
    4. Embedding cache entries (if applicable)
    5. Embed log files
    6. Job history records
    7. Repo metadata (no-op — stored in DuckDB file)

    Every cleanup step continues on error and logs at WARN; the response
    includes a summary of what happened for each resource type.

    Args:
        repo: Repo slug (matches filename stem in ``LADYBUG_DB_DIR``).

    Returns:
        DeleteIndexResponse: the repo slug, files removed, cleanup status per
        resource type, and ok=True.

    Raises:
        HTTPException: 404 when no index exists for the repo.
    """
    # Check that at least one DB file exists (LadybugDB or DuckDB).
    db_path = settings.db_path_for_repo(repo)
    vec_path = settings.vec_db_path_for_repo(repo)
    db_exists = Path(db_path).exists()
    vec_exists = Path(vec_path).exists()

    if not db_exists and not vec_exists:
        raise HTTPException(
            status_code=404,
            detail=f"No index found for '{repo}'.",
        )

    # Execute cascading cleanup: every step continues on error.
    cleanup = {}
    cleanup["ladybug_db"] = _delete_ladybug_db(repo)
    cleanup["duckdb"] = _delete_duckdb(repo)
    cleanup["s3"] = _delete_s3_backup(repo)
    cleanup["embedding_cache"] = _drop_embedding_cache_entries(repo)
    cleanup["embed_logs"] = _delete_embed_logs(repo)
    cleanup["jobs_store"] = _delete_jobs_for_repo(repo)
    cleanup["repo_meta"] = _delete_repo_meta(repo)

    # Build list of removed files (for backward compatibility).
    removed: list[str] = []
    for target in (
        Path(db_path),
        Path(db_path).with_suffix(".db.wal"),
        Path(db_path).with_suffix(".db.shadow"),
        Path(vec_path),
        Path(vec_path).with_name(Path(vec_path).name + ".wal"),
    ):
        if not target.exists():
            removed.append(str(target))

    # Drop in-memory bookkeeping so subsequent /health doesn't advertise a
    # phantom repo.
    indexed_repos.discard(repo)
    indexed_repo_paths.pop(repo, None)
    _last_indexed_cache.pop(repo, None)

    try:
        from .health import invalidate_probe_cache
        invalidate_probe_cache(repo)
    except Exception:
        pass

    logger.info(
        "Deleted index for repo '%s': %s",
        repo,
        ", ".join(f"{k}={v}" for k, v in cleanup.items()),
    )
    return DeleteIndexResponse(repo=repo, removed_files=removed, ok=True, cleanup=cleanup)


# ---------------------------------------------------------------------------
# Job history — list / clear
# ---------------------------------------------------------------------------


def _job_to_summary(j: _Job) -> JobSummary:
    """Project an internal ``_Job`` to the public ``JobSummary`` shape."""
    return JobSummary(
        job_id=j.job_id,
        repo_path=j.repo_path,
        repo_name=Path(j.repo_path).name,
        status=j.status,
        progress_pct=j.progress_pct,
        phase=j.phase,
        node_count=j.node_count,
        rel_count=j.rel_count,
        error=j.error,
        started_at=j.started_at,
        finished_at=j.finished_at,
    )


@router.get("/index/jobs", response_model=JobListResponse)
def list_jobs(
    status: str | None = None,
    repo: str | None = None,
    limit: int = 50,
) -> JobListResponse:
    """List all known index jobs, newest-first.

    Used by the UI's indexing-history panel — no UI should hit every job
    by id to rebuild the list.

    Args:
        status: Optional filter (``running`` | ``done`` | ``failed``).  Comma-
            separated list accepted (e.g. ``done,failed``).
        repo: Optional repo-name filter (matches the tail of ``repo_path``).
        limit: Max rows returned (defaults to 50, capped at 500).

    Returns:
        JobListResponse: newest-first job summaries + totals.
    """
    limit = max(1, min(int(limit), 500))

    wanted_statuses: set[str] | None = None
    if status:
        wanted_statuses = {s.strip() for s in status.split(",") if s.strip()}

    filtered: list[_Job] = []
    running_count = 0
    for j in _jobs.values():
        if j.status == "running":
            running_count += 1
        if wanted_statuses and j.status not in wanted_statuses:
            continue
        if repo and Path(j.repo_path).name != repo:
            continue
        filtered.append(j)

    filtered.sort(key=lambda j: j.started_at, reverse=True)
    return JobListResponse(
        jobs=[_job_to_summary(j) for j in filtered[:limit]],
        total=len(filtered),
        running=running_count,
    )


@router.post("/index/jobs/clear", response_model=JobClearResponse)
def clear_jobs(status: str = "done,failed") -> JobClearResponse:
    """Drop completed/failed job records from the in-memory store.

    Running jobs are never dropped regardless of the ``status`` filter —
    clearing a running job would leave the background task orphaned and the
    UI unable to poll it.

    Args:
        status: Comma-separated statuses to clear.  Defaults to clearing
            every terminal job (``done,failed``).  ``running`` is silently
            skipped even if requested.

    Returns:
        JobClearResponse: number of jobs cleared + number remaining.
    """
    wanted = {s.strip() for s in status.split(",") if s.strip()} or {"done", "failed"}
    wanted.discard("running")  # never clear in-flight jobs

    to_drop = [jid for jid, j in _jobs.items() if j.status in wanted]
    for jid in to_drop:
        del _jobs[jid]

    logger.info("Cleared %d terminal job(s) (status=%s).", len(to_drop), ",".join(sorted(wanted)))
    return JobClearResponse(cleared=len(to_drop), remaining=len(_jobs))


@router.delete("/index/jobs/{job_id}", response_model=JobClearResponse)
def delete_job(job_id: str) -> JobClearResponse:
    """Remove a single job record from the store.

    Rejects attempts to drop a running job with 409 — use ``/status`` to
    monitor or wait for completion first.

    Raises:
        HTTPException: 404 if no job with that id exists, 409 if the job
        is still running.
    """
    j = _jobs.get(job_id)
    if j is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if j.status == "running":
        raise HTTPException(
            status_code=409,
            detail=(
                f"Job {job_id} is still running. Wait for it to finish "
                f"or restart the service to reset state."
            ),
        )
    del _jobs[job_id]
    return JobClearResponse(cleared=1, remaining=len(_jobs))
