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
    admin,
    context_bundle,
    disk,
    embed,
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
    from .services.ladybug_buffer_pool import resolve_buffer_pool_size as _resolve_bps
    from .services.s3_store import restore_indexes as _s3_restore, snapshot_indexes as _s3_snapshot

    db_dir = _Path(settings.LADYBUG_DB_DIR)
    db_dir.mkdir(parents=True, exist_ok=True)

    # --- S3 restore: pull any absent or stale index files from S3 before
    # probing / rehydrating.  Non-fatal — a warning is logged and startup
    # continues if S3 is unreachable or S3_INDEX_BUCKET is unset.
    _s3_restored = _s3_restore(db_dir)
    if _s3_restored:
        _logging.getLogger(__name__).info(
            "Startup S3 restore: pulled %d index file(s) from S3.", _s3_restored
        )

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
            _conn = _lb.Connection(
                _lb.Database(str(db_file), buffer_pool_size=_resolve_bps())
            )
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

    # Rehydrate the in-memory `indexed_repo_paths` map. Two stores
    # cooperate here (BUC-1599):
    #   1. ``jobs_store.indexed_repos`` — authoritative table created in
    #      v2, keyed by canonical slug. Tells us which repos we've ever
    #      indexed and where their DB lives.
    #   2. per-repo DuckDB ``repo_metadata`` — written by every index
    #      job's ``_write_meta`` call. Still the source-of-truth for the
    #      absolute filesystem path of the repo working tree.
    #
    # First boot after the v2 migration, ``indexed_repos`` is empty even
    # on a system that has plenty of ``.cgr/repos/*.db`` files on disk.
    # The reconcile pass below glob-discovers any orphans and back-
    # populates the table from their DuckDB ``repo_metadata`` rows.
    from .routers.index import indexed_repo_paths, indexed_repos, _read_meta

    def _reconcile_indexed_repos() -> int:
        """Back-populate ``indexed_repos`` from on-disk artefacts.

        Walks ``LADYBUG_DB_DIR/*.db``; for every DB whose slug isn't
        already in ``indexed_repos`` we look up its DuckDB
        ``repo_metadata`` and upsert a row. Returns the number of rows
        added so the lifespan can log it.
        """
        existing = {r["slug"] for r in _jobs_store.list_indexed_repos()}
        added = 0
        for db_file in sorted(db_dir.glob("*.db")):
            _slug = db_file.stem
            if _slug in existing:
                continue
            try:
                _meta = _read_meta(_slug)
            except Exception:
                _meta = {}
            try:
                _jobs_store.upsert_indexed_repo(
                    slug=_slug,
                    display_name=_slug,
                    db_path=str(db_file),
                    last_commit_sha=_meta.get("last_indexed_sha") or None,
                )
                _li = _meta.get("last_indexed_at")
                if _li is not None:
                    try:
                        _ = float(_li)
                        _jobs_store.mark_indexed(
                            _slug,
                            last_commit_sha=_meta.get("last_indexed_sha") or None,
                        )
                    except (TypeError, ValueError):
                        pass
                added += 1
            except Exception as _exc:  # noqa: BLE001
                _log.warning(
                    "reconcile_indexed_repos: upsert %s failed (non-fatal): %s",
                    _slug, _exc,
                )
        return added

    _reconciled = _reconcile_indexed_repos()
    if _reconciled:
        _log.info(
            "reconcile_indexed_repos: back-populated %d row(s) from on-disk artefacts.",
            _reconciled,
        )

    for _row in _jobs_store.list_indexed_repos():
        _slug = str(_row.get("slug") or "")
        if not _slug:
            continue
        try:
            _meta = _read_meta(_slug)
        except Exception:
            _meta = {}
        _root = _meta.get("root_path")
        if _root and _Path(_root).exists():
            indexed_repo_paths[_slug] = _root
            indexed_repos.add(_slug)
        else:
            # No DuckDB metadata, but the indexed_repos row exists: surface
            # the slug to /health anyway so callers don't see the repo
            # vanish across a restart.
            indexed_repos.add(_slug)
    _log.info(
        "Rehydrated %d repo path(s) from indexed_repos + DuckDB repo_metadata.",
        len(indexed_repo_paths),
    )

    # --- Phase 4: Prometheus metrics ---
    # setup_metrics mounts /metrics and wires HTTP auto-instrumentation.
    # start_background_collectors polls disk usage every 30 s.
    # LM Studio health probing was removed (LM Studio retired in PR #168; see BUC-1545).
    _metrics_task: asyncio.Task | None = None
    if settings.METRICS_ENABLED:
        from . import metrics as _metrics

        _metrics.setup_metrics(app)
        _metrics_task = asyncio.create_task(
            _metrics.start_background_collectors(
                lm_studio_health_fn=None,
                cgr_data_dir=settings.CGR_DATA_DIR,
            )
        )
        _log.info("metrics: background collectors started")

    # --- Embedder availability probe ---
    # Synchronous, fail-soft. Populates the cached status read by /health
    # and prints the operator-visible banner when no backend is reachable.
    # The probe itself does NOT block startup — a missing
    # sentence-transformers install logs a warning and continues so that
    # structural search / re-index / /health all still work.
    try:
        from .embedders.availability import emit_startup_warning, probe_embedder

        _embedder_status = probe_embedder()
        emit_startup_warning(_embedder_status)
        _log.info(
            "embedder probe: backend=%s available=%s dim=%s fallback_lm_studio=%s",
            _embedder_status.get("backend"),
            _embedder_status.get("available"),
            _embedder_status.get("dim"),
            _embedder_status.get("fallback_lm_studio"),
        )
    except Exception as _exc:  # noqa: BLE001 — never crash startup on the probe
        _log.warning("embedder availability probe failed (non-fatal): %s", _exc)

    # --- Phase 5: Embedder pre-warm (BUC-1518 D1, generalised in BUC-1605) ---
    # Fire a fire-and-forget warmup invocation to absorb the SageMaker
    # Serverless Inference cold start (~4-5s observed) in the background.
    # Without this, the first user-facing call after restart (semantic
    # search, /index parse-phase prewarm not yet fired, etc.) pays the
    # cold-start latency.  Cost: ~$0.0001 per call when the backend is
    # SageMaker; effectively free for local / TEI backends.  Safe to skip
    # if no backend is configured; failures are silently swallowed.
    def _startup_prewarm() -> None:
        try:
            from .embedders.sync_bridge import embed_text_sync, get_embedder_or_none

            backend = get_embedder_or_none()
            if backend is None:
                _log.info("Embedder startup prewarm: no backend configured (EMBEDDER_BACKEND unset or unconfigured)")
                return  # not configured
            import time as _t
            t0 = _t.time()
            vec = embed_text_sync("warmup")
            if vec is None:
                # embed_text_sync swallows errors and returns None — the model
                # likely failed to load (e.g. EMBEDDER_BACKEND=local with a
                # broken sentence-transformers install).  Log at WARNING so
                # operators see this immediately rather than discovering it on
                # the first semantic search request.
                _log.warning(
                    "Embedder startup prewarm: backend=%s returned None — "
                    "embedder may be misconfigured or missing required packages. "
                    "Semantic search will fail until this is resolved. "
                    "For EMBEDDER_BACKEND=local: run 'uv sync'.",
                    backend.name,
                )
            else:
                _log.info(
                    "Embedder startup prewarm: backend=%s latency=%.2fs OK",
                    backend.name, _t.time() - t0,
                )
        except Exception as _exc:  # noqa: BLE001
            _log.warning(
                "Embedder startup prewarm failed (best-effort): %s — "
                "semantic search may be unavailable.",
                _exc,
            )

    import threading as _threading
    _threading.Thread(target=_startup_prewarm, name="sm-startup-prewarm", daemon=True).start()

    # --- Phase 6: housekeeping — drop stale /tmp/cis_embed_*.log files ---
    # Each re-index creates one diagnostic log file.  Without cleanup these
    # accumulate forever (saw 23 lying around after a couple of dev days).
    # Keep the 5 most recent for post-mortem use; everything older is noise.
    try:
        from .routers.index import _cleanup_old_embed_logs
        removed = _cleanup_old_embed_logs(keep=5)
        if removed:
            _log.info("startup: removed %d stale embed log file(s)", removed)
    except Exception as _exc:  # noqa: BLE001
        _log.debug("embed-log cleanup failed (non-fatal): %s", _exc)

    # --- Phase 7: Sweep stale watch_partial rows (high-volume housekeeping) ---
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

    # --- Phase 8: Periodic S3 snapshot (BUC-1555) ---
    # Background task that periodically pushes changed index files to S3.
    # Complements the lifespan shutdown snapshot with a more reliable mechanism
    # (in-dev, kill -9 bypasses graceful shutdown).  Non-fatal — S3 unavailability
    # does not break the indexer.
    _S3_SNAPSHOT_INTERVAL_SEC = 600  # 10 minutes
    _s3_snapshot_task: asyncio.Task | None = None

    async def _periodic_s3_snapshot() -> None:
        """Background task: periodically snapshot indexes to S3."""
        while True:
            try:
                await asyncio.sleep(_S3_SNAPSHOT_INTERVAL_SEC)
                count = _s3_snapshot(_Path(settings.LADYBUG_DB_DIR))
                if count > 0:
                    _log.info("s3: periodic snapshot pushed %d files", count)
            except asyncio.CancelledError:
                break
            except Exception as _exc:  # noqa: BLE001
                _log.warning("s3: periodic snapshot failed (will retry in %ds): %s",
                            _S3_SNAPSHOT_INTERVAL_SEC, _exc)

    if settings.LADYBUG_DB_DIR:  # only start if DB dir is configured
        _s3_snapshot_task = asyncio.create_task(_periodic_s3_snapshot())
        _log.info("s3: periodic snapshot task started (interval=%ds)", _S3_SNAPSHOT_INTERVAL_SEC)

    # --- Phase 9: Tantivy segment warm-up ---
    # On the first real search call, tantivy's MmapDirectory causes OS page
    # faults to load segment files from disk.  For large repos that can add
    # 200-800 ms to P99 latency.  Warming all existing .tantivy/ directories
    # during startup amortises that cost to zero for the first user request.
    # Non-fatal — missing tantivy, corrupt segment files, or an empty index
    # directory are all handled gracefully.
    def _warm_tantivy_indexes() -> None:
        try:
            from .services.tantivy_index import TantivyIndex

            _tantivy_db_dir = _Path(settings.LADYBUG_DB_DIR)
            if not _tantivy_db_dir.exists():
                return
            warmed = 0
            for _tantivy_dir in sorted(_tantivy_db_dir.glob("*.tantivy")):
                _slug = _tantivy_dir.stem
                try:
                    idx = TantivyIndex(str(_tantivy_db_dir), _slug)
                    if not idx._unavailable and idx._index is not None:
                        idx._index.reload()
                        warmed += 1
                except Exception as _exc:  # noqa: BLE001
                    _log.debug(
                        "tantivy warmup: skipped %s (%s)", _tantivy_dir.name, _exc
                    )
            if warmed:
                _log.info("tantivy startup warm-up: paged in %d index/es", warmed)
        except Exception as _exc:  # noqa: BLE001
            _log.debug("tantivy startup warm-up failed (non-fatal): %s", _exc)

    _threading.Thread(
        target=_warm_tantivy_indexes,
        name="tantivy-startup-warmup",
        daemon=True,
    ).start()

    # --- Phase 10: Job heartbeat reconciliation (LE-143) ---
    # Periodic background task that marks any running job that hasn't updated
    # in N seconds as stale/failed. Prevents phantom jobs from showing in the
    # queue forever when a worker crashes mid-run.
    _job_heartbeat_task: asyncio.Task | None = None

    # Validate and use env-var settings for interval and staleness threshold
    _heartbeat_interval = settings.JOB_HEARTBEAT_INTERVAL_SECONDS
    _staleness_threshold = settings.JOB_STALENESS_THRESHOLD_SECONDS

    # Sanity bounds: interval must be at least 10s; threshold at least interval
    if _heartbeat_interval < 10:
        _log.warning(
            "JOB_HEARTBEAT_INTERVAL_SECONDS=%d is too small (min 10s), "
            "using default 60s", _heartbeat_interval
        )
        _heartbeat_interval = 60
    if _staleness_threshold < _heartbeat_interval:
        _log.warning(
            "JOB_STALENESS_THRESHOLD_SECONDS=%d is less than interval %ds, "
            "using default 300s", _staleness_threshold, _heartbeat_interval
        )
        _staleness_threshold = 300

    async def _periodic_job_heartbeat() -> None:
        """Background task: periodically reconcile stale running jobs."""
        from .routers.index import reconcile_stale_running_jobs

        while True:
            try:
                await asyncio.sleep(_heartbeat_interval)
                reconciled = reconcile_stale_running_jobs(
                    staleness_threshold_seconds=_staleness_threshold
                )
                # Log only if reconciliation happened (avoid spam)
                if reconciled > 0:
                    _log.info(
                        "job-heartbeat: reconciled %d stale job(s)", reconciled
                    )
            except asyncio.CancelledError:
                break
            except Exception as _exc:  # noqa: BLE001
                _log.warning(
                    "job-heartbeat: reconciliation failed (will retry in %ds): %s",
                    _heartbeat_interval, _exc
                )

    _job_heartbeat_task = asyncio.create_task(_periodic_job_heartbeat())
    _log.info(
        "job-heartbeat: periodic reconciliation task started "
        "(interval=%ds, staleness_threshold=%ds)",
        _heartbeat_interval, _staleness_threshold
    )

    yield

    # Shutdown: cancel the job heartbeat task.
    if _job_heartbeat_task is not None:
        _job_heartbeat_task.cancel()
        try:
            await _job_heartbeat_task
        except asyncio.CancelledError:
            pass

    # Shutdown: cancel the periodic S3 snapshot task.
    if _s3_snapshot_task is not None:
        _s3_snapshot_task.cancel()
        try:
            await _s3_snapshot_task
        except asyncio.CancelledError:
            pass

    # Shutdown: push changed index files to S3 so the next container inherits them.
    _s3_snapshot(settings.LADYBUG_DB_DIR)

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
    # BUC-1592 — query-embedding surface for TheForge's cross-repo affinity
    # weighting. Single-string input → 768-dim vector (sagemaker:
    # jina-code-v2; local/tei: e5-base-v2). LE-129 swapped SageMaker 2026-05-26.
    app.include_router(embed.router, tags=["embed"])
    app.include_router(explorer.router, tags=["explorer"])
    app.include_router(github.router, tags=["github"])
    app.include_router(admin.router, tags=["admin"])
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
