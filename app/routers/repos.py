"""Frontend-shape per-repo endpoints (BACKEND_HANDOVER §2.1, §2.2).

These endpoints are the surface the TheForge frontend calls — see
``web/src/components/code-indexer/IndexRunDashboard.tsx`` and
``web/src/components/code-indexer/api.ts``. They wrap existing internal
machinery (``_blocking_index``, ``_read_meta``, the per-repo ``.db`` /
``.duck`` files) into the field shapes the dashboard renders.

Routes mounted at ``/repos/...`` so the TheForge proxy can forward
``/api/code-indexer/repos/...`` straight through.

Endpoints:
    GET  /repos/{name}/stats      — sidebar facts (db size, fragment count,
                                    edge count, per-label node counts, last
                                    indexed timestamp, indexed commit SHA)
    POST /repos/{name}/reindex    — force re-index. Wipes the LadybugDB +
                                    DuckDB files, then schedules all 4
                                    indexing passes.
    POST /repos/{slug}/watch      — start file-watcher (Phase 5)
    GET  /repos/{slug}/watch      — watcher status (Phase 5)
    DELETE /repos/{slug}/watch    — stop file-watcher (Phase 5)
"""
from __future__ import annotations

import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException

from ..config import settings
from ..models import (
    IndexRequest,
    PartialIndexEvent,
    ReindexAccepted,
    ReindexRequest,
    RepoIndexStatsResponse,
    WatchAccepted,
    WatchStatus,
)

router = APIRouter(prefix="/repos", tags=["repos"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Labels we report in ``node_count_by_label``. Mirrors the breakdown the
# IndexRunDashboard sidebar renders. We only include labels that have a
# non-zero count so the FE doesn't render padded zero rows.
_REPORTED_LABELS = (
    "Project", "File", "Folder", "Package", "Module",
    "Class", "Function", "Method", "Interface",
    "Variable", "Struct", "Enum", "Type",
    "ExternalPackage",
)
_REPORTED_REL_TYPES = (
    "CONTAINS_FILE", "CONTAINS_FOLDER", "CONTAINS_PACKAGE", "CONTAINS_MODULE",
    "DEFINES", "DEFINES_METHOD",
    "CALLS", "IMPORTS", "INHERITS", "IMPLEMENTS", "OVERRIDES", "BELONGS_TO",
)


def _to_iso(ts: float | None) -> str | None:
    """Convert a unix epoch (seconds) into an ISO 8601 UTC string.

    The frontend's ``RepoIndexStats.last_indexed_at`` is typed as a string
    so callers can ``new Date(stats.last_indexed_at)`` without epoch math.
    """
    if ts is None:
        return None
    try:
        return (
            datetime.fromtimestamp(float(ts), tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    except (ValueError, TypeError, OSError):
        return None


def _git_sha(repo_path: str) -> str | None:
    """Return the indexed repo's HEAD SHA, or None when not a git checkout.

    Best-effort — failures (missing git binary, detached worktree, dirty
    permissions) all return None. The frontend renders ``—`` in that case.
    """
    if not repo_path:
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if result.returncode == 0:
            sha = result.stdout.strip()
            return sha or None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _node_count_by_label(repo_name: str, db_path: str) -> tuple[dict[str, int], int]:
    """Return per-label node counts and the total relationship count.

    Opens the LadybugDB read-only and runs one ``count`` query per label /
    rel type. Returns ``({}, 0)`` on any failure so the endpoint can still
    serve ``repo_metadata``-derived fields from the ``.duck`` file.
    """
    counts: dict[str, int] = {}
    total_rels = 0
    db = None
    conn = None
    try:
        import real_ladybug as lb  # type: ignore[import-untyped]

        db = lb.Database(db_path)
        conn = lb.Connection(db)
        for label in _REPORTED_LABELS:
            try:
                r = conn.execute(f"MATCH (n:{label}) RETURN count(n) AS cnt")
                if r.has_next():
                    cnt = int(r.get_next()[0])
                    if cnt > 0:
                        counts[label] = cnt
            except Exception:
                # Label might not exist for this schema version; skip.
                continue
        for rtype in _REPORTED_REL_TYPES:
            try:
                r = conn.execute(f"MATCH ()-[r:{rtype}]->() RETURN count(r) AS cnt")
                if r.has_next():
                    total_rels += int(r.get_next()[0])
            except Exception:
                continue
    except Exception as exc:
        logger.warning("Stats probe failed for %s: %s", repo_name, exc)
    finally:
        # Always release the DB handle so a probe failure doesn't pin the
        # file for the rest of the process lifetime.
        conn = None
        db = None
    return counts, total_rels


# ---------------------------------------------------------------------------
# GET /repos/{name}/stats
# ---------------------------------------------------------------------------


@router.get("/{name}/stats", response_model=RepoIndexStatsResponse)
def repo_index_stats(name: str) -> RepoIndexStatsResponse:
    """FE-shape per-repo index stats — powers IndexRunDashboard sidebar.

    All fields are nullable. The frontend renders ``—`` for any missing
    field, so we prefer a partial response over a 5xx when one source is
    unavailable (e.g. ``.duck`` exists but is empty → ``fragment_count: 0``,
    not null; LadybugDB locked by an active writer → ``node_count_by_label``
    returns the cached totals from ``repo_metadata``, not a 503).

    Args:
        name: Repo slug — same key used in ``/health.indexed_repos``.

    Returns:
        RepoIndexStatsResponse: All facts the dashboard needs. Returns 404
        only when neither the ``.db`` nor the ``.duck`` file exists.
    """
    from .index import _read_meta, indexed_repo_paths, is_repo_indexing

    db_path = Path(settings.db_path_for_repo(name))
    duck_path = Path(settings.vec_db_path_for_repo(name))

    if not db_path.exists() and not duck_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No index found for '{name}'. POST /repos/{name}/reindex first.",
        )

    db_size = db_path.stat().st_size if db_path.exists() else None
    duck_size = duck_path.stat().st_size if duck_path.exists() else None

    # Fragment count = embedding row count in the DuckDB vector store.
    fragment_count: int | None = None
    if duck_path.exists():
        try:
            from codebase_rag.storage.vector_store import open_or_create, row_count

            conn = open_or_create(str(duck_path))
            fragment_count = row_count(conn)
            conn.close()
        except Exception as exc:
            logger.warning("DuckDB fragment count failed for %s: %s", name, exc)
            fragment_count = None

    # Per-label node counts — skip the live DB probe while a writer is
    # active to avoid the single-writer assertion.  ``repo_metadata`` from
    # the ``.duck`` file gives the FE last-known-good numbers in the
    # meantime.
    if is_repo_indexing(name) or not db_path.exists():
        node_count_by_label: dict[str, int] = {}
        edge_count: int | None = None
        meta = _read_meta(name)
        if isinstance(meta.get("node_count"), (int, float)) and meta.get("node_count"):
            # ``repo_metadata`` stores totals only; we can't break it down
            # by label while a writer holds the DB. Surface a single bucket
            # so the sidebar still renders something useful.
            node_count_by_label = {"_total": int(meta["node_count"])}
        if isinstance(meta.get("rel_count"), (int, float)):
            edge_count = int(meta["rel_count"])
    else:
        node_count_by_label, edge_count = _node_count_by_label(name, str(db_path))
        if edge_count == 0:
            edge_count = None  # sentinel: no rels = "—" in the UI

    # last_indexed_at — DuckDB ``repo_metadata`` row, then convert to ISO.
    meta = _read_meta(name)
    last_indexed_at_raw = meta.get("last_indexed_at")
    last_indexed_at_iso = _to_iso(
        float(last_indexed_at_raw) if last_indexed_at_raw is not None else None
    )

    # Indexed commit SHA — git rev-parse against the source repo path
    # captured at index time. None when we never recorded a path or the
    # repo isn't a git checkout.
    repo_root = indexed_repo_paths.get(name) or meta.get("root_path") or ""
    indexed_commit_sha = _git_sha(str(repo_root)) if repo_root else None

    return RepoIndexStatsResponse(
        db_size_bytes=db_size,
        duck_size_bytes=duck_size,
        last_indexed_at=last_indexed_at_iso,
        indexed_commit_sha=indexed_commit_sha,
        fragment_count=fragment_count,
        edge_count=edge_count,
        node_count_by_label=node_count_by_label,
    )


# ---------------------------------------------------------------------------
# POST /repos/{name}/reindex
# ---------------------------------------------------------------------------


@router.post("/{name}/reindex", response_model=ReindexAccepted, status_code=202)
async def reindex_repo(
    name: str,
    req: ReindexRequest,
    background_tasks: BackgroundTasks,
) -> ReindexAccepted:
    """Force re-index a repo — convenience wrapper around ``POST /index``.

    Resolves the repo's source path from the in-memory map populated at
    index time (or rehydrated from ``.duck`` metadata at startup), then
    delegates to the existing ``start_index`` machinery with
    ``force_reindex=True``. Keeping the wrapper thin means re-index always
    follows the same code path as a fresh index — locks, idempotency,
    progress reporting all reused.

    Args:
        name: Repo slug.
        req: Reindex request body. Currently ``force=true`` is the only
            supported mode; ``force=false`` is reserved for future
            incremental re-index.
        background_tasks: FastAPI-injected scheduler.

    Returns:
        ReindexAccepted: ``{ "job_id": "<uuid>" }``.

    Raises:
        HTTPException: 404 when we have no path on file for the repo
            (never indexed in this service / on this host).
        HTTPException: 409 when an index job is already running for this
            repo — surfaced from ``start_index``.
    """
    from .index import indexed_repo_paths, start_index, _read_meta

    repo_path = indexed_repo_paths.get(name)
    if not repo_path:
        # Last-ditch: try ``repo_metadata`` — we may have rehydrated it
        # but not the in-memory dict yet (defence in depth, shouldn't
        # normally hit).
        meta = _read_meta(name)
        candidate = meta.get("root_path")
        if candidate and Path(candidate).is_dir():
            repo_path = candidate

    if not repo_path:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Repo '{name}' has no recorded source path. POST /index with "
                "the absolute repo_path first; subsequent re-indexes can use "
                f"POST /repos/{name}/reindex."
            ),
        )

    accepted = await start_index(
        IndexRequest(repo_path=repo_path, force_reindex=bool(req.force)),
        background_tasks,
    )
    return ReindexAccepted(job_id=accepted.job_id)


# ---------------------------------------------------------------------------
# Phase 5 — file-watcher lifecycle endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/{slug}/watch",
    response_model=WatchAccepted,
    status_code=202,
    summary="Start file-watcher for a repo (Phase 5)",
)
async def start_watch(slug: str) -> WatchAccepted:
    """Start a per-repo file-watcher.

    The watcher debounces FS events and triggers partial re-indexes so
    edits are visible in ``/search/semantic`` within ~5 s of save.

    Returns:
        WatchAccepted (202): ``{ watcher_id, started_at, debounce_ms }``.

    Raises:
        503: ``WATCH_ENABLED=false`` — feature flag is off.
        409: Watcher already active for this slug.
        404: Repo is not indexed — run ``POST /index`` first.
        429: Too many active watchers (``WATCH_MAX_REPOS`` limit).
    """
    from ..services.watch_manager import (
        start_watch as _start_watch,
        WatchDisabledError,
        WatchAlreadyActiveError,
        WatchCapacityError,
        get_watch,
    )

    try:
        handle = await _start_watch(
            slug,
            actor_oid="anon",
            actor_email="anon@local",
        )
    except WatchDisabledError:
        raise HTTPException(
            status_code=503,
            detail={"code": "watch_disabled", "message": "WATCH_ENABLED is false"},
        )
    except WatchAlreadyActiveError:
        existing = get_watch(slug)
        raise HTTPException(
            status_code=409,
            detail={
                "code": "watch_already_active",
                "watcher_id": slug,
                "started_at": existing.started_at if existing else 0.0,
            },
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "repo_not_indexed",
                "message": f"Repo '{slug}' has no index. POST /index first.",
            },
        )
    except WatchCapacityError:
        raise HTTPException(
            status_code=429,
            detail={
                "code": "watch_capacity_exceeded",
                "max": settings.WATCH_MAX_REPOS,
            },
        )
    except OSError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "watch_inotify_exhausted",
                "message": (
                    f"Observer.schedule failed: {exc}. "
                    "Try: echo fs.inotify.max_user_watches=524288 >> /etc/sysctl.conf"
                ),
            },
        )

    return WatchAccepted(
        watcher_id=handle.repo_slug,
        started_at=handle.started_at,
        debounce_ms=handle.debounce_ms,
    )


@router.get(
    "/{slug}/watch",
    response_model=WatchStatus,
    summary="Get file-watcher status for a repo (Phase 5)",
)
def get_watch_status(slug: str) -> WatchStatus:
    """Return the current state of the watcher for ``slug``.

    Returns:
        WatchStatus (200): Current watcher snapshot.

    Raises:
        404: No active watcher for this slug.
    """
    from ..services.watch_manager import get_watch as _get_watch

    handle = _get_watch(slug)
    if handle is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "watch_not_active", "message": f"No watcher for '{slug}'"},
        )
    return WatchStatus(
        repo_slug=handle.repo_slug,
        repo_path=handle.repo_path,
        actor_oid=handle.actor_oid,
        actor_email=handle.actor_email,
        started_at=handle.started_at,
        last_event_at=handle.last_event_at,
        last_partial_job_id=handle.last_partial_job_id,
        debounce_ms=handle.debounce_ms,
        pending_paths_count=handle.pending_paths_count,
        state=handle.state,
    )


@router.delete(
    "/{slug}/watch",
    summary="Stop file-watcher for a repo (Phase 5)",
)
async def stop_watch(slug: str) -> dict:
    """Stop the file-watcher for ``slug``.

    Returns:
        dict: ``{ stopped_at, last_partial_job_id }``.

    Raises:
        404: No active watcher for this slug.
    """
    from ..services.watch_manager import (
        stop_watch as _stop_watch,
        get_watch as _get_watch,
        WatchNotActiveError,
    )

    handle = _get_watch(slug)
    if handle is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "watch_not_active", "message": f"No watcher for '{slug}'"},
        )

    last_partial_job_id = handle.last_partial_job_id

    try:
        await _stop_watch(slug)
    except WatchNotActiveError:
        raise HTTPException(
            status_code=404,
            detail={"code": "watch_not_active", "message": f"No watcher for '{slug}'"},
        )

    return {
        "stopped_at": time.time(),
        "last_partial_job_id": last_partial_job_id,
    }
