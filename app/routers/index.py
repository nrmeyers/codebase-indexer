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
    Pass 4   (embedding):  UniXcoder model → numpy .npy vector store

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
import json
import logging
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException

from ..config import settings
from ..models import (
    DeleteIndexResponse,
    IndexAccepted,
    IndexRequest,
    IndexStatus,
    JobClearResponse,
    JobListResponse,
    JobSummary,
    NodeTypeStat,
    RepoStatsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


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
        progress_pct: Best-effort progress indicator (milestone-based).
        phase: Current phase label shown to UIs (``parsing``, ``embedding``).
        node_count: Final graph node count, populated on completion.
        rel_count: Final graph relationship count, populated on completion.
        embedded_count: Number of function/method embeddings written.
            Zero on incremental runs when nothing new was embedded (all
            functions were already cached from a prior index).
        error: Populated only when ``status == "failed"``.
        started_at: Wall-clock start time used for TTL-based pruning.
        finished_at: Wall-clock end time; None while running.
    """

    job_id: str
    repo_path: str
    status: Literal["running", "done", "failed"] = "running"
    progress_pct: float = 0.0
    phase: str = "parsing"
    node_count: int = 0
    rel_count: int = 0
    embedded_count: int = 0
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None


# Module-level store — indexed by job_id. Single-process only.
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


def _meta_path_for_repo(repo_name: str) -> Path:
    """Return the path to the sidecar metadata JSON for ``repo_name``.

    The sidecar lives next to the DB file (``.cgr/repos/{slug}.meta.json``).
    Using a sidecar instead of writing the timestamp onto the Project node
    sidesteps LadybugDB's typed schema — adding a new column would require
    a migration, while JSON on disk is cheap, observable, and survives
    restarts.
    """
    db_path = Path(settings.db_path_for_repo(repo_name))
    return db_path.with_suffix(".meta.json")


def _write_meta(repo_name: str, **fields: Any) -> None:
    """Merge ``fields`` into the per-repo sidecar JSON.

    Atomic write via tempfile + rename so a mid-write SIGKILL leaves the
    old sidecar intact.  Missing / corrupt sidecars are treated as empty.
    """
    meta_path = _meta_path_for_repo(repo_name)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    current: dict[str, Any] = {}
    if meta_path.exists():
        try:
            current = json.loads(meta_path.read_text())
        except Exception:
            current = {}

    current.update(fields)
    tmp = meta_path.with_suffix(".meta.json.tmp")
    tmp.write_text(json.dumps(current, indent=2))
    tmp.replace(meta_path)  # atomic rename on the same filesystem


def _read_meta(repo_name: str) -> dict[str, Any]:
    """Return the sidecar JSON dict for ``repo_name`` (empty on miss/error)."""
    meta_path = _meta_path_for_repo(repo_name)
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return {}


def _get_last_indexed_at(repo_name: str) -> float | None:
    """Return the last successful-index timestamp for ``repo_name``.

    Checks in-memory cache first, then the sidecar JSON file.  Any error
    (missing file, unreadable JSON) returns None — callers fall back to
    the DB file's mtime.
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
            await loop.run_in_executor(None, _blocking_index, job, force_reindex)
        except Exception as exc:
            # Capture failure on the job so pollers see the error reason rather
            # than a silent stuck-running status.
            job.status = "failed"
            job.error = str(exc)
            job.finished_at = time.time()


def _blocking_index(job: _Job, force_reindex: bool) -> None:
    """Synchronous ingestion — called from the thread pool.

    Uses LadybugIngestor as a context manager (which handles DB connection,
    schema migration, and VECTOR extension loading on entry). GraphUpdater
    requires parsers and queries to be loaded from the tree-sitter registry.

    Args:
        job: The job record to mutate with progress and final counts.
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

    # LadybugIngestor is a context manager — __enter__ opens the DB connection
    # and runs schema migration; __exit__ flushes remaining buffers and closes.
    with LadybugIngestor(
        db_path=repo_db_path,
        batch_size=settings.LADYBUG_BATCH_SIZE,
    ) as ingestor:
        # Progress updates are milestone-based rather than per-file so the
        # status endpoint never hot-loops on a mutex.
        job.progress_pct = 5.0

        updater = GraphUpdater(
            ingestor=ingestor,
            repo_path=repo,
            parsers=parsers,
            queries=queries,
        )

        job.progress_pct = 10.0
        updater.run(force=effective_force)
        job.progress_pct = 90.0

        # Store the absolute repo root on the Project node so search endpoints
        # can resolve relative file paths back to absolute paths.  The
        # last-indexed timestamp and other operational metadata live in a
        # sidecar JSON file instead — LadybugDB's typed schema doesn't allow
        # adding new columns without a migration, but sidecars are free.
        project_name = repo.name
        ingestor.conn.execute(  # type: ignore[union-attr]
            "MATCH (p:Project {name: $name}) SET p.root_path = $root_path",
            {"name": project_name, "root_path": str(repo)},
        )

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
        node_count=job.node_count,
        rel_count=job.rel_count,
        job_id=job.job_id,
        force_reindex=force_reindex,
        effective_force=effective_force,
    )
    _last_indexed_cache[repo.name] = _now

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

    job.progress_pct = 92.0
    job.phase = "embedding"

    # -------------------------------------------------------------------
    # Pass 4: Embedding generation (subprocess-isolated, REQUIRED)
    # -------------------------------------------------------------------
    # Load the UniXcoder model (~400 MB) and embed every Function/Method
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

    _blocking_embed(embed_job)  # raises on failure → job marked "failed"
    job.embedded_count = embed_job.embedded_count

    job.progress_pct = 100.0
    job.status = "done"
    job.finished_at = time.time()

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
    error: str | None = None
    started_at: float = field(default_factory=time.time)


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
    #   1. Opens the existing DB (no structural re-parse)
    #   2. Loads the UniXcoder model once
    #   3. Embeds every Function/Method that has source_code
    #   4. Writes Embedding nodes to the DB
    # Running in a subprocess means an OOM kill doesn't affect uvicorn.
    driver = f"""
import os, sys
from pathlib import Path

os.environ["LADYBUG_DB_PATH"] = {repr(repo_db_path)}
os.environ["SKIP_EMBEDDINGS"] = "false"

from codebase_rag.config import settings as s
s.LADYBUG_DB_PATH = {repr(repo_db_path)}
s.SKIP_EMBEDDINGS = False

from codebase_rag.services.ladybug_ingestor import LadybugIngestor
from codebase_rag.graph_updater import GraphUpdater
from codebase_rag.parser_loader import load_parsers
from codebase_rag.vector_store import flush_embeddings

parsers, queries = load_parsers()
repo_path = Path({repr(repo_path_str)}) if {repr(repo_path_str)} else Path.cwd()

with LadybugIngestor(db_path={repr(repo_db_path)}, batch_size=50) as ingestor:
    updater = GraphUpdater(
        ingestor=ingestor,
        repo_path=repo_path,
        parsers=parsers,
        queries=queries,
    )
    updater._generate_semantic_embeddings()

# flush_embeddings is also called inside _generate_semantic_embeddings, but
# calling it again here is a no-op when _pending is already clear — acts as
# a safety net in case the ingestor path differs from the settings default.
flush_embeddings(db_path={repr(repo_db_path)})

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
            timeout=1800,  # 30 min hard limit
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
    # "EMBED_DONE" and may emit "Embedded N" earlier.
    try:
        with log_path.open() as f:
            for line in f:
                if line.startswith("Embedded"):
                    try:
                        job.embedded_count = int(line.split()[1])
                    except (IndexError, ValueError):
                        pass
    except Exception:
        pass

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

    # Reject empty or whitespace-only paths before attempting filesystem ops.
    # Path("") resolves to cwd (a valid directory) so it must be caught early.
    if not req.repo_path or not req.repo_path.strip():
        raise HTTPException(
            status_code=422,
            detail="repo_path must not be empty",
        )

    repo_path = Path(req.repo_path)
    if not repo_path.exists():
        raise HTTPException(
            status_code=422,
            detail=f"repo_path does not exist: {req.repo_path}",
        )
    if not repo_path.is_dir():
        raise HTTPException(
            status_code=422,
            detail=f"repo_path must be a directory, not a file: {req.repo_path}",
        )

    # Reject a second concurrent job on the same repo with a clear 409 so the
    # UI can show "already indexing" instead of silently queueing behind a
    # lock.  Without this the second request would stall on the async lock
    # and timeout the HTTP client before ever getting a job id.
    resolved = repo_path.resolve()
    for j in _jobs.values():
        if (
            j.status == "running"
            and Path(j.repo_path).resolve() == resolved
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Index job already running for this repo "
                    f"(job_id={j.job_id}). Poll /index/{j.job_id}/status or wait for completion."
                ),
            )

    job_id = str(uuid.uuid4())
    job = _Job(job_id=job_id, repo_path=str(repo_path))
    _jobs[job_id] = job

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
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return IndexStatus(
        job_id=job.job_id,
        status=job.status,
        progress_pct=job.progress_pct,
        phase=job.phase,
        node_count=job.node_count,
        rel_count=job.rel_count,
        error=job.error,
    )


# ---------------------------------------------------------------------------
# POST /index/embed — optional semantic embedding pass
# ---------------------------------------------------------------------------


from pydantic import BaseModel


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

    Embedding generation loads the UniXcoder model (~400 MB) and is
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
)
_STATS_REL_TYPES = (
    "CONTAINS_FILE", "DEFINES", "DEFINES_METHOD",
    "CALLS", "IMPORTS", "INHERITS",
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
    # Serve the last-known-good figures from the sidecar instead — the UI
    # can keep polling and will see live counts once the writer releases.
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
    root_path = ""
    probe_ok = False
    try:
        import real_ladybug as lb  # type: ignore[import-untyped]

        db = lb.Database(db_path)
        conn = lb.Connection(db)

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
            # "has_embeddings" is best-effort — check for the numpy sidecar
            # written by the embedding pass (avoids any LadybugDB VECTOR dep).
            _npy = Path(db_path).with_suffix(".embeddings.npy")
            has_embeddings = _npy.exists() and _npy.stat().st_size > 0
        except Exception:
            has_embeddings = False

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

    # ``last_indexed_at`` comes from the sidecar JSON (_get_last_indexed_at
    # checks the in-memory cache first, then the file).  Independent from
    # the graph so a schema-less migration isn't needed.
    last_idx_at = _get_last_indexed_at(repo)

    if not probe_ok:
        # DB exists but can't be probed right now (stale lock, transient
        # failure).  Return sidecar figures with a 200 — the UI distinguishes
        # "writer busy" from "permanent error" via `indexing` + staleness of
        # last_indexed_at, and retrying /stats on the next poll usually
        # succeeds once whatever held the lock has released.
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
        indexing=is_repo_indexing(repo),
    )


# ---------------------------------------------------------------------------
# DELETE /index/{repo} — admin wipe
# ---------------------------------------------------------------------------


@router.delete("/index/{repo}", response_model=DeleteIndexResponse)
def delete_index(repo: str) -> DeleteIndexResponse:
    """Remove a repo's DB + any WAL/shadow sidecars.

    Used to reset a corrupt or stale index from the UI without shelling
    into the server.  Removing the file itself is enough — the next
    ``POST /index`` call for that repo recreates schema from scratch.

    Args:
        repo: Repo slug (matches filename stem in ``LADYBUG_DB_DIR``).

    Returns:
        DeleteIndexResponse: the repo slug and every file removed.

    Raises:
        HTTPException: 404 when no DB exists for the repo, 503 on unlink
        failure (permission / IO).
    """
    db_path = settings.db_path_for_repo(repo)
    p = Path(db_path)
    if not p.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No index found for '{repo}'.",
        )

    removed: list[str] = []
    try:
        # Remove primary DB file + WAL/shadow sidecars + metadata JSON.
        # ``missing_ok`` so partial cleanup (only .db present, no .wal) still
        # succeeds rather than leaving orphan files behind on retry.
        for ext in (".db", ".db.wal", ".db.shadow", ".meta.json", ".embeddings.npy", ".embeddings_idx.json"):
            sidecar = p.with_suffix(ext) if ext != ".db" else p
            if sidecar.exists():
                sidecar.unlink(missing_ok=True)
                removed.append(str(sidecar))
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to remove index files: {exc}",
        ) from exc

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

    logger.info("Deleted index for repo '%s' (%d file(s)).", repo, len(removed))
    return DeleteIndexResponse(repo=repo, removed_files=removed, ok=True)


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
