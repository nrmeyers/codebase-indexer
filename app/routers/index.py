"""POST /index and GET /index/{job_id}/status — background ingestion.

Indexing a repository is CPU-bound and can take minutes. To keep the HTTP
API responsive, this router accepts indexing requests asynchronously:

    1. ``POST /index`` creates a ``_Job`` record and returns a job_id (202).
    2. The heavy work runs in a thread-pool executor via FastAPI's
       BackgroundTasks so the event loop stays free.
    3. Clients poll ``GET /index/{job_id}/status`` until ``done`` or
       ``failed``.

Jobs live in an in-memory dict (``_jobs``). This is acceptable because the
service is single-process; running multiple replicas would require moving
the store to Redis or LadybugDB itself.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException

from ..config import settings
from ..models import IndexAccepted, IndexRequest, IndexStatus

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
        node_count: Final graph node count, populated on completion.
        rel_count: Final graph relationship count, populated on completion.
        error: Populated only when ``status == "failed"``.
        started_at: Wall-clock start time used for TTL-based pruning.
    """

    job_id: str
    repo_path: str
    status: Literal["running", "done", "failed"] = "running"
    progress_pct: float = 0.0
    node_count: int = 0
    rel_count: int = 0
    error: str | None = None
    started_at: float = field(default_factory=time.time)


# Module-level store — indexed by job_id. Single-process only.
_jobs: dict[str, _Job] = {}

# TTL: keep completed jobs for 1 hour so callers can poll after completion.
_JOB_TTL_SECONDS = 3600


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

    Args:
        job: The mutable job record to update with progress and final state.
        force_reindex: When true, the underlying GraphUpdater clears the
            graph before re-ingesting.
    """
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, _blocking_index, job, force_reindex)
    except Exception as exc:
        # Capture failure on the job so pollers see the error reason rather
        # than a silent stuck-running status.
        job.status = "failed"
        job.error = str(exc)


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

    # Point code-graph-rag at the same DB this service uses. Without this,
    # the two packages would each read their own config and could end up
    # writing to different database files.
    cgr_settings.LADYBUG_DB_PATH = settings.LADYBUG_DB_PATH
    cgr_settings.LADYBUG_BATCH_SIZE = settings.LADYBUG_BATCH_SIZE

    repo = Path(job.repo_path).resolve()
    parsers, queries = load_parsers()

    # LadybugIngestor is a context manager — __enter__ opens the DB connection
    # and runs schema migration; __exit__ flushes remaining buffers and closes.
    with LadybugIngestor(
        db_path=settings.LADYBUG_DB_PATH,
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
        updater.run(force=force_reindex)
        job.progress_pct = 90.0

        # Store the absolute repo root on the Project node so search endpoints
        # can resolve relative file paths back to absolute paths for source
        # extraction. The ingestor stores all paths relative to the repo root
        # for portability, so without this, symbol source reads silently fail.
        project_name = repo.name
        ingestor.conn.execute(  # type: ignore[union-attr]
            "MATCH (p:Project {name: $name}) SET p.root_path = $root_path",
            {"name": project_name, "root_path": str(repo)},
        )

    # Collect final counts from LadybugDB. Best-effort: if the count
    # queries fail we still mark the job done — the ingestion itself
    # succeeded and callers can always re-query later.
    try:
        import real_ladybug as lb  # type: ignore[import-untyped]

        db = lb.Database(settings.LADYBUG_DB_PATH)
        conn = lb.Connection(db)

        node_res = conn.execute(
            "MATCH (n) RETURN count(n) AS cnt"
        )
        if node_res.has_next():
            job.node_count = int(node_res.get_next()[0])

        rel_res = conn.execute(
            "MATCH ()-[r]->() RETURN count(r) AS cnt"
        )
        if rel_res.has_next():
            job.rel_count = int(rel_res.get_next()[0])
    except Exception:
        pass  # counts are best-effort

    job.progress_pct = 100.0
    job.status = "done"


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
        node_count=job.node_count,
        rel_count=job.rel_count,
        error=job.error,
    )
