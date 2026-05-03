"""Code Indexer Service — FastAPI application factory.

This module owns the top-level FastAPI app construction for the Code Indexer
Service. It wires routers for health checks, repository indexing, structural
and semantic search, and the context-bundle endpoint used by TheForge's
dev-agent.

Key design decisions:
    * A single ``create_app`` factory is exposed so the service can be
      instantiated under tests with a fresh state and so ASGI servers can
      import ``app`` directly (``app = create_app()`` at module scope).
    * The application lifespan eagerly warms the LadybugDB schema so the first
      ``/index`` call does not pay the migration cost.
    * A generic exception handler converts any uncaught ``Exception`` into a
      structured 500 JSON response so the service never leaks an HTML error
      page.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Bridge .env → os.environ so modules that read os.environ directly (notably
# ``app.services.lm_studio``, which is intentionally settings-stack-free)
# pick up local overrides.  pydantic-settings populates the ``Settings``
# instance from .env but does NOT push the values back into os.environ —
# without this call, ``LM_STUDIO_*`` would silently default to "" / "CodeRankLLM"
# under uvicorn even with a fully-populated .env.  Idempotent and side-effect
# free when no .env exists.
load_dotenv()

from .config import settings  # noqa: E402  -- must run AFTER load_dotenv()
from .routers import (  # noqa: E402  -- must run AFTER load_dotenv()
    context_bundle,
    disk,
    explorer,
    github,
    health,
    index,
    repos,
    search,
    symbols,
    websocket,
)

# Basic structured logging — without this, our logger.info/warning calls stay
# silent under uvicorn's default handler.  Format matches uvicorn's access
# log so mixed output stays readable.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan hook — runs startup tasks then yields control.

    Runs the LadybugDB schema migration on process start so that the first
    ``/index`` request does not have to pay for DDL. Schema migration is
    idempotent (``IF NOT EXISTS`` guards), so repeat startups are safe.

    Args:
        app: The FastAPI application being started. Unused, but required by
            the lifespan contract.

    Yields:
        None: Control is yielded to FastAPI once startup finishes.
    """
    # Ensure the per-repo DB directory exists so the first /index call isn't
    # blocked by a "No such file or directory" from LadybugDB on a clean
    # install.  Individual per-repo DB files are migrated by the ingestor
    # on first use, so there's no eager schema warm-up here.
    from pathlib import Path as _Path
    import logging as _logging
    import real_ladybug as _lb

    db_dir = _Path(settings.LADYBUG_DB_DIR)
    db_dir.mkdir(parents=True, exist_ok=True)

    # Probe each existing DB file at startup.  A corrupt WAL or shadow file
    # (from a SIGKILL mid-write) causes every subsequent open() call to
    # raise "Corrupted wal file" — wipe and start fresh rather than block
    # every index job forever.
    _log = _logging.getLogger(__name__)
    _healed = 0
    _probed = 0
    for db_file in sorted(db_dir.glob("*.db")):
        _probed += 1
        try:
            _conn = _lb.Connection(_lb.Database(str(db_file)))
            del _conn
        except Exception as _exc:
            _msg = str(_exc)
            # A lock-acquisition failure means *someone else* holds the file
            # — typically a leftover subprocess from a prior crash that
            # hasn't exited yet, or another uvicorn on the same port.  That
            # is NOT corruption and the DB must not be deleted.  Only treat
            # WAL / shadow / schema-level failures as heal-worthy.
            _is_lock_failure = (
                "Could not set lock on file" in _msg
                or "already in use" in _msg.lower()
            )
            if _is_lock_failure:
                _log.warning(
                    "DB %s is locked by another process (%s) — skipping self-heal.",
                    db_file.name, _exc,
                )
                continue
            _log.warning(
                "DB %s appears corrupt (%s) — removing stale files for clean restart.",
                db_file.name, _exc,
            )
            _healed += 1
            for _ext in ("", ".wal", ".shadow"):
                _stale = db_file.with_suffix(".db" + _ext)
                if _stale.exists():
                    _stale.unlink(missing_ok=True)

    _log.info(
        "Startup DB probe: %d repo(s) checked, %d self-healed.", _probed, _healed,
    )

    # --- Phase 2: Persistent job store ---
    # Generate a fresh per-process worker token so sweep_interrupted() can
    # distinguish rows owned by *this* process from orphans of a prior one.
    from .services import jobs_store as _jobs_store
    _worker_token = os.urandom(8).hex()
    _jobs_store.init(settings.JOBS_DB_PATH)
    _interrupted = _jobs_store.sweep_interrupted(_worker_token)
    if _interrupted:
        _log.warning(
            "jobs_store: swept %d interrupted job(s) from prior worker", _interrupted
        )

    # Orphan-job sweep + stale lock cleanup so a prior crash doesn't leave
    # the in-memory state wedged.
    from .routers.index import sweep_orphan_jobs, cleanup_stale_locks
    swept = sweep_orphan_jobs()
    stale = cleanup_stale_locks()
    if swept or stale:
        _log.info("Reaped %d orphan job(s) and %d stale lock(s).", swept, stale)

    # Rehydrate the in-memory `indexed_repo_paths` map from the per-repo
    # ``.duck`` ``repo_metadata`` rows left behind by past index jobs.
    # Without this, every restart loses the repo→abs-path mapping and
    # callers (the orchestrator's chat flow, /context-bundle validation)
    # can no longer resolve a repo slug back to a path until a fresh
    # index runs.
    from .routers.index import indexed_repo_paths, indexed_repos, _read_meta
    for db_file in sorted(db_dir.glob("*.db")):
        _slug = db_file.stem  # "TheForge.db" → "TheForge"
        try:
            _meta = _read_meta(_slug)
        except Exception:
            _meta = {}
        _root = _meta.get("root_path")
        if _root and _Path(_root).exists():
            indexed_repo_paths[_slug] = _root
            indexed_repos.add(_slug)
    _log.info(
        "Rehydrated %d repo path(s) from DuckDB repo_metadata.",
        len(indexed_repo_paths),
    )

    # --- Phase 4: Prometheus metrics ---
    # setup_metrics mounts /metrics and wires HTTP auto-instrumentation.
    # start_background_collectors polls LM Studio health + disk usage every 30 s.
    _metrics_task: asyncio.Task | None = None
    if settings.METRICS_ENABLED:
        from . import metrics as _metrics
        from .services.lm_studio import is_available as _lm_available, can_rerank as _lm_can_rerank

        def _lm_health() -> tuple[bool, bool]:
            return _lm_available(), _lm_can_rerank()

        _metrics.setup_metrics(app)
        _metrics_task = asyncio.create_task(
            _metrics.start_background_collectors(
                lm_studio_health_fn=_lm_health,
                cgr_data_dir=settings.CGR_DATA_DIR,
            )
        )
        _log.info("metrics: background collectors started")

    # --- Phase 5: Sweep stale watch_partial rows (high-volume housekeeping) ---
    if settings.WATCH_ENABLED:
        swept_wp = _jobs_store.clear_terminal(
            statuses={"done", "failed", "cancelled"},
            kind="watch_partial",
            older_than_hours=settings.WATCH_PARTIAL_RETENTION_HOURS,
        )
        if swept_wp:
            _log.info(
                "jobs_store: swept %d stale watch_partial row(s) older than %dh",
                swept_wp,
                settings.WATCH_PARTIAL_RETENTION_HOURS,
            )

    yield

    # Shutdown: stop all active file-watchers before the event loop exits.
    if settings.WATCH_ENABLED:
        try:
            from .services.watch_manager import shutdown_all as _watch_shutdown_all
            await _watch_shutdown_all()
        except Exception as _exc:
            _log.warning("watch_manager: shutdown_all error: %s", _exc)

    # Shutdown: cancel the metrics background task gracefully.
    if _metrics_task is not None:
        _metrics_task.cancel()
        try:
            await _metrics_task
        except asyncio.CancelledError:
            pass


def create_app() -> FastAPI:
    """Construct and return a fully-wired FastAPI application.

    Returns:
        FastAPI: An app with all routers registered and a catch-all exception
        handler installed.
    """
    app = FastAPI(
        title="Code Indexer Service",
        description=(
            "HTTP gateway for code-graph-rag — indexes repositories into LadybugDB "
            "and exposes structural + semantic search to TheForge."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(health.router, tags=["health"])
    app.include_router(index.router, tags=["index"])
    app.include_router(search.router, tags=["search"])
    app.include_router(context_bundle.router, tags=["context"])
    app.include_router(explorer.router, tags=["explorer"])
    app.include_router(github.router, tags=["github"])
    # Frontend-shape endpoints (BACKEND_HANDOVER doc):
    #   /repos/{name}/stats, /repos/{name}/reindex     (§2.1, §2.2)
    #   /symbols/{fqn}, /symbols/{fqn}/{callers,callees} (§2.7, §2.9)
    #   /disk-usage                                     (§2.11)
    #   /ws (WebSocket index.{progress,complete,failed}) (§2.3)
    app.include_router(repos.router)
    app.include_router(symbols.router)
    app.include_router(disk.router)
    app.include_router(websocket.router)

    @app.exception_handler(Exception)
    async def _generic_error(request, exc):  # type: ignore[override]
        # Catch-all fallback so any unhandled error surfaces as JSON rather
        # than a default HTML error page. Specific handlers/HTTPException
        # cases are still honored by FastAPI's own exception pipeline.
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
        )

    return app


# Module-level app instance used by ASGI servers (e.g. `uvicorn app.main:app`).
app = create_app()
