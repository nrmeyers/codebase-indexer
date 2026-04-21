"""POST /index and GET /index/{job_id}/status — background ingestion."""
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
    job_id: str
    repo_path: str
    status: Literal["running", "done", "failed"] = "running"
    progress_pct: float = 0.0
    node_count: int = 0
    rel_count: int = 0
    error: str | None = None
    started_at: float = field(default_factory=time.time)


_jobs: dict[str, _Job] = {}

# TTL: keep completed jobs for 1 hour so callers can poll after completion.
_JOB_TTL_SECONDS = 3600


def _prune_old_jobs() -> None:
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
    pool executor so the event loop stays responsive.
    """
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, _blocking_index, job, force_reindex)
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)


def _blocking_index(job: _Job, force_reindex: bool) -> None:
    """Synchronous ingestion — called from the thread pool."""
    from codebase_rag.config import settings as cgr_settings
    from codebase_rag.services.graph_service import MemgraphIngestor
    from codebase_rag.graph_updater import GraphUpdater

    # Point code-graph-rag at the same DB this service uses.
    cgr_settings.LADYBUG_DB_PATH = settings.LADYBUG_DB_PATH
    cgr_settings.LADYBUG_BATCH_SIZE = settings.LADYBUG_BATCH_SIZE

    repo = Path(job.repo_path).resolve()

    ingestor = MemgraphIngestor(
        db_path=settings.LADYBUG_DB_PATH,
        batch_size=settings.LADYBUG_BATCH_SIZE,
    )
    ingestor.connect()

    job.progress_pct = 5.0

    updater = GraphUpdater(
        ingestor=ingestor,
        repo_path=str(repo),
        clean=force_reindex,
    )

    job.progress_pct = 10.0
    updater.update()
    job.progress_pct = 90.0

    # Collect final counts from LadybugDB.
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
    """Kick off a background indexing job for the given repository."""
    _prune_old_jobs()

    repo_path = Path(req.repo_path)
    if not repo_path.exists():
        raise HTTPException(
            status_code=422,
            detail=f"repo_path does not exist: {req.repo_path}",
        )

    job_id = str(uuid.uuid4())
    job = _Job(job_id=job_id, repo_path=str(repo_path))
    _jobs[job_id] = job

    background_tasks.add_task(_run_ingestion, job, req.force_reindex)

    return IndexAccepted(job_id=job_id)


@router.get("/index/{job_id}/status", response_model=IndexStatus)
def get_index_status(job_id: str) -> IndexStatus:
    """Poll the status of a previously submitted indexing job."""
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
