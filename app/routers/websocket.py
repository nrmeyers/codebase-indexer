"""WebSocket /ws — index progress event stream (BACKEND_HANDOVER §2.3).

The frontend's IndexRunDashboard subscribes to ``/ws`` and filters
events by name (``index.progress`` / ``index.complete`` / ``index.failed``).
Each connection runs an independent polling loop so a slow client cannot
back up state for other subscribers.

Implementation choice (per BACKEND_HANDOVER open question #2): we poll
the in-process ``_jobs`` table at 1 Hz and synthesise events from state
diffs. This keeps the indexer code path itself untouched — no need to
plumb a publish call through every progress callback. The cost is a
~1s latency floor on event delivery, which the FE renders smoothly via
its own progress-bar interpolation.

Event shapes:

    {
      "event": "index.progress",
      "data": {
        "repo": "TheForge",
        "pass": 4,                    // 1 | 2 | 3 | 4
        "current": 412,
        "total": 547,
        "throughput": 18.5            // optional, fragments/sec for pass 4
      }
    }

    { "event": "index.complete", "data": { "repo": "TheForge" } }

    { "event": "index.failed", "data": { "repo": "TheForge", "reason": "..." } }
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["websocket"])
logger = logging.getLogger(__name__)


# Phase → pass-number mapping mirrors BACKEND_HANDOVER §2.3:
#   "Pass 1-3 are structural (tree-sitter parse + LadybugDB write),
#    pass 4 is the embedder."
# We map fine-grained phases onto the 4-bucket pass numbering the FE
# renders. Phases not in this map (e.g. "queued") emit pass=1 so the
# very first progress event still lights up the bar.
_PHASE_TO_PASS: dict[str, int] = {
    "queued": 1,
    "discovering": 1,
    "parsing": 1,
    "writing": 2,
    "finalizing": 3,
    "embedding": 4,
    "done": 4,
    "cancelled": 1,
}


def _repo_slug_from_path(repo_path: str) -> str:
    """Compute the repo slug that ``/health`` reports for a given path.

    Mirrors how the indexer registers repos: the slug is the basename of
    the absolute path. Centralised here so the WS event ``repo`` field
    matches what the FE filters on.
    """
    if not repo_path:
        return ""
    return Path(repo_path).name


def _job_snapshot(job: Any) -> dict[str, Any]:
    """Return a hashable-ish dict snapshot of a job for change detection.

    We compare these snapshots between polls; any difference triggers a
    new ``index.progress`` event.
    """
    return {
        "status": job.status,
        "phase": job.phase,
        "progress_pct": round(float(job.progress_pct or 0.0), 2),
        "files_done": int(job.files_done or 0),
        "files_total": int(job.files_total or 0),
        "embedded_count": int(getattr(job, "embedded_count", 0) or 0),
    }


def _build_progress_event(job: Any, prev_embedded: int, prev_ts: float) -> dict[str, Any]:
    """Construct an ``index.progress`` payload from a job and prior state.

    Args:
        job: Current ``_Job`` instance.
        prev_embedded: ``embedded_count`` from the prior poll, used to
            compute pass-4 throughput.
        prev_ts: Wall-clock time of the prior poll.

    Returns:
        dict[str, Any]: Payload matching BACKEND_HANDOVER §2.3 shape.
    """
    pass_no = _PHASE_TO_PASS.get(str(job.phase), 1)

    # ``current``/``total`` are stage-relative. For passes 1–3 we use the
    # file counters; for pass 4 (embedding) the absolute counter is the
    # embedded_count and the total is files_total (a usable proxy until
    # the embedder reports its own total).
    if pass_no == 4:
        current = int(job.embedded_count or 0)
        total = int(job.files_total or 0) or current
    else:
        current = int(job.files_done or 0)
        total = int(job.files_total or 0) or current

    payload: dict[str, Any] = {
        "repo": _repo_slug_from_path(job.repo_path),
        "pass": pass_no,
        "current": current,
        "total": total,
    }

    # Throughput only makes sense for pass 4 and only when we can compare
    # against the prior poll. Avoid divide-by-zero and skip on first event.
    if pass_no == 4 and prev_ts:
        delta = float(time.time() - prev_ts)
        if delta > 0 and current > prev_embedded:
            payload["throughput"] = round((current - prev_embedded) / delta, 2)

    return payload


@router.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    """Stream index progress + completion events to a single subscriber.

    Polls ``_jobs`` at 1 Hz, emits a ``index.progress`` event for any
    job whose snapshot changed since the last poll, and emits a single
    ``index.complete`` or ``index.failed`` event the moment a job's
    status flips out of ``running``. Connections are independent — a
    new subscriber sees one initial progress event per running job
    (so re-connect after a disconnect is seamless).

    The handler runs forever until the client disconnects; cancellation
    propagates cleanly through ``WebSocketDisconnect``.
    """
    # Local import to avoid circular dependency at module load time.
    from .index import _jobs

    await websocket.accept()
    logger.info("WS subscriber connected: %s", websocket.client)

    last_snapshots: dict[str, dict[str, Any]] = {}
    last_emit_ts: dict[str, float] = {}
    last_embedded: dict[str, int] = {}
    finished_emitted: set[str] = set()

    try:
        while True:
            # Snapshot the dict before iterating so concurrent insertions
            # from `_run_ingestion` don't blow up the loop.
            current_jobs = list(_jobs.values())

            for job in current_jobs:
                snap = _job_snapshot(job)
                prev = last_snapshots.get(job.job_id)

                # Progress event when state changed (or first sighting of
                # a running job — gives reconnects an immediate paint).
                if job.status == "running" and snap != prev:
                    payload = _build_progress_event(
                        job,
                        last_embedded.get(job.job_id, 0),
                        last_emit_ts.get(job.job_id, 0.0),
                    )
                    await websocket.send_text(
                        json.dumps({"event": "index.progress", "data": payload})
                    )
                    last_emit_ts[job.job_id] = time.time()
                    last_embedded[job.job_id] = int(job.embedded_count or 0)

                # Terminal events fire exactly once per job.
                if (
                    job.status in ("done", "failed")
                    and job.job_id not in finished_emitted
                ):
                    repo = _repo_slug_from_path(job.repo_path)
                    if job.status == "done":
                        await websocket.send_text(
                            json.dumps({
                                "event": "index.complete",
                                "data": {"repo": repo},
                            })
                        )
                    else:
                        await websocket.send_text(
                            json.dumps({
                                "event": "index.failed",
                                "data": {
                                    "repo": repo,
                                    "reason": job.error or "Unknown error",
                                },
                            })
                        )
                    finished_emitted.add(job.job_id)

                last_snapshots[job.job_id] = snap

            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        logger.info("WS subscriber disconnected: %s", websocket.client)
    except Exception as exc:
        logger.warning("WS handler error: %s", exc)
        try:
            await websocket.close()
        except Exception:
            pass
