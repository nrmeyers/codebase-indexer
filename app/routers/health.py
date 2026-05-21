"""GET /health — liveness + per-repo probe + indexed-repo summary.

Per-repo probes are cached for a short TTL so UIs polling at 1 Hz don't
hammer LadybugDB with a new connection per request (the probe's main cost).
Cache keys are invalidated any time a job writes to a repo (see
``invalidate_probe_cache``) so freshly-completed indexes always see new
counts on the next /health call.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import APIRouter

from ..config import settings
from ..embedders import availability as embedder_availability
from ..models import EmbedderStatus, HealthResponse, LmStudioStatus, RepoHealth
from ..services import lm_studio

router = APIRouter()
logger = logging.getLogger(__name__)

# Probe TTL — 3s is long enough to absorb 1–3 Hz UI polling without becoming
# user-visibly stale.  Invalidated explicitly when an index job commits so
# the next /health call after ingestion always reports fresh data.
_PROBE_TTL_SECONDS = 3.0
_probe_cache: dict[str, tuple[float, RepoHealth]] = {}


def invalidate_probe_cache(repo_name: str | None = None) -> None:
    """Drop the cached probe(s) so the next ``/health`` call re-probes.

    Args:
        repo_name: When given, only that repo's cache entry is dropped.
            When None (the default), the entire cache is flushed — use
            after a bulk operation like ``cleanup_stale_locks()``.
    """
    if repo_name is None:
        _probe_cache.clear()
        return
    _probe_cache.pop(repo_name, None)


def _get_indexed_repos() -> list[str]:
    """Return a deduplicated list of project names represented on disk.

    Lists ``*.db`` files in ``LADYBUG_DB_DIR``.  Falls back to the in-memory
    set populated by the index router for pre-DB-dir installs or when the
    directory doesn't yet exist.

    Returns:
        list[str]: Sorted, deduplicated project slugs corresponding to
        ``.cgr/repos/*.db`` files.  Empty list when no repos are indexed.
    """
    from .index import indexed_repos  # local import to avoid circular deps

    names: set[str] = set(indexed_repos)
    db_dir = Path(settings.LADYBUG_DB_DIR)
    if db_dir.is_dir():
        for p in db_dir.glob("*.db"):
            names.add(p.stem)
    return sorted(names)


def _probe_repo(name: str) -> RepoHealth:
    """Open a per-repo DB with a short-lived connection and return its state.

    TTL-cached: probes hit LadybugDB at most once per ``_PROBE_TTL_SECONDS``
    per repo, so a 1 Hz /health poll costs one DB open per 3s (not per call).

    Args:
        name: Project slug (filename stem of ``{slug}.db``).

    Returns:
        RepoHealth: readability + size + approximate node count +
        last_indexed_at + in-flight flag.
    """
    from .index import _get_last_indexed_at, _read_meta, is_repo_indexing, indexed_repo_paths

    now = time.time()
    cached = _probe_cache.get(name)
    if cached and (now - cached[0]) < _PROBE_TTL_SECONDS:
        # Refresh the live-only fields (indexing flag changes mid-window)
        # but keep the expensive count/size readings from the cache.
        stale = cached[1]
        return RepoHealth(
            name=stale.name,
            db_path=stale.db_path,
            size_bytes=stale.size_bytes,
            node_count=stale.node_count,
            readable=stale.readable,
            last_indexed_at=_get_last_indexed_at(name) or stale.last_indexed_at,
            indexing=is_repo_indexing(name),
            repo_path=indexed_repo_paths.get(name) or stale.repo_path,
        )

    db_path = settings.db_path_for_repo(name)
    p = Path(db_path)
    size = p.stat().st_size if p.exists() else 0
    last_idx = _get_last_indexed_at(name)
    indexing = is_repo_indexing(name)

    if not p.exists():
        rh = RepoHealth(
            name=name,
            db_path=db_path,
            size_bytes=0,
            node_count=None,
            readable=False,
            last_indexed_at=last_idx,
            indexing=indexing,
            repo_path=indexed_repo_paths.get(name),
        )
        _probe_cache[name] = (now, rh)
        return rh

    # Skip the live DB probe while an index job is actively writing to this
    # repo — LadybugDB is single-writer, so opening a probe connection here
    # would either block or trigger an internal assertion.  The sidecar
    # (last_indexed_at, node_count) provides stale-but-safe data for the UI
    # until the writer releases the lock.
    if indexing:
        meta = _read_meta(name)
        rh = RepoHealth(
            name=name,
            db_path=db_path,
            size_bytes=size,
            node_count=meta.get("node_count"),
            readable=True,
            last_indexed_at=last_idx,
            indexing=True,
            repo_path=indexed_repo_paths.get(name),
        )
        _probe_cache[name] = (now, rh)
        return rh

    db = None
    conn = None
    count: int | None = None
    readable = False
    try:
        from ..services.ladybug_pool import open_read_conn

        # BUC-1571: read-only probe — never block on the indexer's
        # exclusive lock. A locked DB simply reports unreadable and
        # falls back to the sidecar count below.
        db, conn = open_read_conn(db_path)
        res = conn.execute("MATCH (n) RETURN count(n) AS cnt")
        if res.has_next():
            count = int(res.get_next()[0])
        readable = True
    except Exception as exc:
        logger.warning("Probe failed for %s: %s", name, exc)
    finally:
        # Always release the DB handle, even on UNREACHABLE_CODE / lock-
        # acquisition failure.  Without this, a failed probe pins the DB
        # file for the remainder of the process lifetime and every
        # subsequent probe inherits the same error.
        conn = None
        db = None

    if not readable:
        # Fall back to sidecar node_count (last known good) so the UI still
        # shows something meaningful while the DB is temporarily unreadable.
        meta = _read_meta(name)
        count = meta.get("node_count")

    rh = RepoHealth(
        name=name,
        db_path=db_path,
        size_bytes=size,
        node_count=count,
        readable=readable,
        last_indexed_at=last_idx,
        indexing=indexing,
        repo_path=indexed_repo_paths.get(name),
    )
    _probe_cache[name] = (now, rh)
    return rh


def _probe_lm_studio() -> LmStudioStatus:
    """Build the ``lm_studio`` block for the health response.

    Short-circuits to an all-False/None payload when LM Studio is not
    configured (no ``LM_STUDIO_URL``), avoiding any network call. All
    adapter probes are wrapped in a broad ``except`` so a misbehaving
    LM Studio process can never break /health.
    """
    try:
        url = lm_studio.base_url()
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("LM Studio base_url() failed: %s", exc)
        return LmStudioStatus()

    if not url:
        return LmStudioStatus()

    try:
        reachable = lm_studio.is_available()
        embed_model = lm_studio.resolve_model(lm_studio.embed_model_hint())
        rerank_model = lm_studio.resolve_model(lm_studio.rerank_model_hint())
        can_embed = lm_studio.can_embed()
        can_rerank = lm_studio.can_rerank()
    except Exception as exc:
        logger.warning("LM Studio probe failed in /health: %s", exc)
        return LmStudioStatus(configured=True)

    return LmStudioStatus(
        configured=True,
        reachable=reachable,
        embed_model=embed_model,
        rerank_model=rerank_model,
        can_embed=can_embed,
        can_rerank=can_rerank,
    )


def _probe_embedder() -> EmbedderStatus:
    """Build the ``embedder`` block for the health response.

    Reads the cached startup probe result populated by
    :func:`app.embedders.availability.probe_embedder` so /health calls are
    O(1) and never trigger a backend construction (which can be expensive
    for SageMaker cold starts or block a long time waiting for HTTP).

    Backend construction or dep-validation failures (e.g.
    ``EMBEDDER_BACKEND=local`` with the ``[local-embed]`` extras group
    missing) are surfaced as ``available=false`` / ``configured=false``
    with the captured ``last_error``. /health stays 200 because the
    service can still serve cached searches and structural queries that
    don't need the embedder.
    """
    try:
        return EmbedderStatus(**embedder_availability.current_status())
    except Exception as exc:  # noqa: BLE001 — never let a payload-build error
        # break /health. Surface a minimal placeholder so the UI can still
        # render a "status unknown" badge.
        logger.warning("embedder status payload build failed: %s", exc)
        return EmbedderStatus(
            backend="unknown",
            available=False,
            last_error=f"{type(exc).__name__}: {exc}",
        )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Readiness probe — always returns 200, flips ``status`` to ``degraded``
    when any per-repo DB fails to open.

    Returns:
        HealthResponse: ``status``, DB directory, indexed project names, a
        probe row per repo (size, node count, readable, last_indexed_at,
        indexing), a count of currently-running index jobs, and an
        ``lm_studio`` backend-status block.
    """
    from .index import _jobs  # local import to avoid circular deps

    indexed = _get_indexed_repos()
    probes = [_probe_repo(n) for n in indexed]
    status = "ok" if all(p.readable for p in probes) else "degraded"
    running = sum(1 for j in _jobs.values() if j.status == "running")

    # S3 sync state is best-effort — if the import or call fails for any
    # reason, fall back to "disabled" so /health stays alive.
    try:
        from ..services.s3_store import get_sync_state
        from ..models import S3SyncStatus
        s3_sync = S3SyncStatus(**get_sync_state())
    except Exception as exc:
        logger.debug("S3 sync state unavailable: %s", exc)
        from ..models import S3SyncStatus
        s3_sync = S3SyncStatus(enabled=False)

    return HealthResponse(
        status=status,
        db_path=settings.LADYBUG_DB_DIR,
        indexed_repos=indexed,
        repos=probes,
        running_jobs=running,
        lm_studio=_probe_lm_studio(),
        s3_sync=s3_sync,
        embedder=_probe_embedder(),
    )
