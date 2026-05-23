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

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

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
# GET /repos — listing endpoint (BUC-1561b)
# ---------------------------------------------------------------------------


class RepoListItem(BaseModel):
    """One row in the ``GET /repos`` response.

    The shape complements TheForge's ``GET /api/code-indexer/repos/available``
    endpoint: this one is "what does the indexer know about", that one is
    "what does GitHub say is available". The frontend joins on ``slug`` /
    ``full_name``.
    """

    slug: str = Field(description="Repo slug — the key used by /repos/{slug}/stats")
    full_name: str | None = Field(
        default=None,
        description="``owner/repo`` if the slug encodes one (e.g. 'navistone__TheForge'), else None.",
    )
    default_branch: str | None = Field(
        default=None,
        description="Best-effort default branch from local git config; None when unknown.",
    )
    last_indexed_at: str | None = Field(
        default=None,
        description="ISO-8601 UTC timestamp of last successful index, or None.",
    )
    last_indexed_sha: str | None = Field(
        default=None,
        description="Git SHA recorded at index time; None when never indexed or unknown.",
    )
    repo_path: str | None = Field(
        default=None,
        description=(
            "Absolute path on the indexer host to the source tree captured at "
            "index time (the ``root_path`` row in the per-repo ``repo_metadata`` "
            "table). TheForge's ``defaultProbeLocalDrift`` uses this to compute "
            "drift between the indexed SHA and the current local HEAD. None when "
            "the repo has not been indexed yet or the path was not recorded."
        ),
    )
    indexed: bool = Field(description="True when at least one successful index has run.")
    status: str = Field(
        description=(
            "Freshness verdict: ``unindexed`` | ``fresh`` | ``stale``. "
            "``fresh`` means last_indexed_sha matches the current local HEAD; "
            "``stale`` means the index ran but the SHAs differ; "
            "``unindexed`` means no successful index job has ever completed. "
            "Drift detection against the GitHub remote is TheForge's "
            "responsibility — this endpoint only sees what's locally known."
        ),
    )


class RepoListResponse(BaseModel):
    """Envelope for ``GET /repos``."""

    repos: list[RepoListItem]


def _slug_to_full_name(slug: str) -> str | None:
    """Recover ``owner/repo`` from a slug when the clone helper encoded it.

    ``_clone_or_update`` writes clones to ``.cgr/clones/{owner}__{name}``,
    and ``slugify_repo`` flattens that to ``{owner}__{name}``. We invert
    the convention here to give TheForge a join key against its own
    ``GET /api/code-indexer/repos/available`` listing. Slugs that don't
    follow the convention return None — the frontend renders ``—`` then.
    """
    if "__" in slug:
        owner, _, name = slug.partition("__")
        if owner and name:
            return f"{owner}/{name}"
    return None


@router.get("", response_model=RepoListResponse)
def list_repos() -> RepoListResponse:
    """List every repo this indexer knows about.

    Reads from the in-memory ``indexed_repos`` set + per-repo DuckDB
    ``repo_metadata`` table (rehydrated on startup). Does NOT call GitHub
    — that's TheForge's job via the GitHub App. The caller can intersect
    this list with TheForge's ``/api/code-indexer/repos/available`` to
    decide which repos need (re)indexing.

    Status semantics (BUC-1561b):
        * ``unindexed`` — no successful index job has ever completed.
          Should not normally appear in this list (the list is built from
          repos that *have* metadata) but included as a defence-in-depth
          fallback when ``last_indexed_at`` is missing.
        * ``fresh`` — ``last_indexed_sha`` matches the local working
          tree's current ``git rev-parse HEAD``. Drift detection against
          the *remote* is TheForge's responsibility.
        * ``stale`` — index ran, but the local HEAD has moved on.

    Returns an empty list on a fresh database.
    """
    from datetime import datetime, timezone
    from .index import _read_meta, indexed_repo_paths, indexed_repos

    items: list[RepoListItem] = []

    # Union of in-memory registry + on-disk DB files so we still report
    # repos whose ``indexed_repos`` set entry was lost (e.g. the rehydrate
    # at startup couldn't resolve ``root_path``).
    db_dir = Path(settings.LADYBUG_DB_DIR)
    on_disk_slugs: set[str] = set()
    if db_dir.exists():
        for db_file in sorted(db_dir.glob("*.db")):
            on_disk_slugs.add(db_file.stem)

    for slug in sorted(indexed_repos | on_disk_slugs):
        meta = _read_meta(slug) if slug else {}

        last_indexed_at_raw = meta.get("last_indexed_at")
        last_indexed_at_iso: str | None = None
        if last_indexed_at_raw is not None:
            try:
                last_indexed_at_iso = (
                    datetime.fromtimestamp(float(last_indexed_at_raw), tz=timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z")
                )
            except (ValueError, TypeError, OSError):
                last_indexed_at_iso = None

        last_indexed_sha = meta.get("last_indexed_sha") or None
        if isinstance(last_indexed_sha, str):
            last_indexed_sha = last_indexed_sha.strip() or None

        # ``indexed`` reflects whether a real timestamp was recorded — a
        # row in repo_metadata without ``last_indexed_at`` indicates an
        # in-flight or interrupted job.
        indexed = bool(last_indexed_at_iso)

        # Status verdict — see docstring for semantics.
        if not indexed:
            status_verdict = "unindexed"
        else:
            repo_root = indexed_repo_paths.get(slug) or meta.get("root_path") or ""
            current_head = _git_sha(str(repo_root)) if repo_root else None
            if last_indexed_sha and current_head and last_indexed_sha == current_head:
                status_verdict = "fresh"
            elif last_indexed_sha and current_head and last_indexed_sha != current_head:
                status_verdict = "stale"
            else:
                # No SHA on either side → can't prove drift; assume stale
                # so TheForge re-checks against the GitHub App rather than
                # trusting an unverifiable "fresh" answer.
                status_verdict = "stale"

        # Best-effort default branch from local git config.
        default_branch: str | None = None
        repo_root = indexed_repo_paths.get(slug) or meta.get("root_path") or ""
        if repo_root:
            try:
                result = subprocess.run(
                    ["git", "-C", str(repo_root), "symbolic-ref", "--short", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                    check=False,
                )
                if result.returncode == 0:
                    default_branch = result.stdout.strip() or None
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                default_branch = None

        # LE-111: surface root_path as ``repo_path`` so TheForge's
        # ``defaultProbeLocalDrift`` has the source-tree path needed for
        # SHA-drift visualization. Falls back to the indexed_repo_paths
        # in-memory cache when the meta row was written before the
        # root_path column was populated.
        repo_path_meta = meta.get("root_path") or indexed_repo_paths.get(slug) or None
        if isinstance(repo_path_meta, str):
            repo_path_meta = repo_path_meta.strip() or None
        else:
            repo_path_meta = None

        items.append(
            RepoListItem(
                slug=slug,
                full_name=_slug_to_full_name(slug),
                default_branch=default_branch,
                last_indexed_at=last_indexed_at_iso,
                last_indexed_sha=last_indexed_sha,
                repo_path=repo_path_meta,
                indexed=indexed,
                status=status_verdict,
            )
        )

    return RepoListResponse(repos=items)


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
        from ..services.ladybug_pool import open_read_conn

        # BUC-1571: /repos counts are pure reads — open read-only so
        # listing the repos doesn't fight the indexer's exclusive lock.
        db, conn = open_read_conn(db_path)
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


# ---------------------------------------------------------------------------
# GET /repos/{name}/centrality — Phase 1.5 PageRank top-N (debug surface)
# ---------------------------------------------------------------------------


class CentralitySymbol(BaseModel):
    """One row in ``GET /repos/{name}/centrality``."""

    qname: str = Field(description="Qualified name of the callable.")
    centrality: float = Field(
        description="PageRank score from the per-repo centrality table. "
        "Plan J persists divide-by-max scores; downstream fusion re-normalises."
    )


class CentralityListResponse(BaseModel):
    """Envelope for ``GET /repos/{name}/centrality``."""

    symbols: list[CentralitySymbol] = Field(default_factory=list)
    last_computed_at: str | None = Field(
        default=None,
        description=(
            "ISO-8601 UTC timestamp of when PageRank was last computed for this "
            "repo (derived from the ``updated_at`` epoch in the centrality table). "
            "``null`` when the table has never been populated. TheForge uses this "
            "to determine whether the cached centrality boost scores are still "
            "fresh relative to the last index run."
        ),
    )


@router.get(
    "/{name}/centrality",
    response_model=CentralityListResponse,
    summary="Top-N most-central symbols by PageRank (Phase 1.5 debug)",
)
def repo_centrality_top_n(
    name: str,
    limit: int = Query(default=20, ge=1, le=200),
) -> CentralityListResponse:
    """Return the ``limit`` most-central symbols for ``name``.

    Companion endpoint to ``GET /search/centrality`` — same data source
    (the per-repo ``.duck`` ``centrality`` table populated by Plan J at the
    end of every full ingest pass), but a simpler shape that TheForge can
    use to inspect which symbols would be boosted once the
    ``mergeAndRank`` integration lands. The location-enriched
    ``/search/centrality`` route stays the FE-facing contract.

    Reads are sub-millisecond: PageRank is computed once at index-finish
    time (``_blocking_index`` Plan J block) and persisted in the ``.duck``
    DuckDB file; this endpoint is a pure SELECT with no on-demand compute.

    Args:
        name: Repo slug — the same key used by ``/repos/{name}/stats``.
        limit: Max rows to return (1–200; default 20 per the brief).

    Returns:
        CentralityListResponse: ``{ symbols: [{ qname, centrality }],
        last_computed_at }``.
        Returns an empty array — never 404 — when the centrality table is
        empty (PageRank not yet computed for this repo). Graceful
        degradation lets TheForge poll without special-casing freshly-
        indexed repos.
    """
    duck_path = Path(settings.vec_db_path_for_repo(name))
    if not duck_path.exists():
        return CentralityListResponse(symbols=[])

    try:
        from codebase_rag.storage.vector_store import open_or_create  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("centrality.no_sibling_pkg repo=%s", name)
        return CentralityListResponse(symbols=[])

    rows: list[tuple[str, float]] = []
    last_computed_at: str | None = None
    try:
        conn = open_or_create(str(duck_path))
        try:
            res = conn.execute(
                "SELECT qualified_name, pagerank, updated_at FROM centrality "
                "ORDER BY pagerank DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
            rows = [(r[0], float(r[1])) for r in res]
            # Derive last_computed_at from the max updated_at across all
            # returned rows.  The ``updated_at`` column stores a Unix epoch
            # (integer seconds) written by ``write_centrality`` at the end
            # of every index run — a reliable proxy for "when was PageRank
            # last computed".  We read it from the already-fetched result
            # set to avoid a second query.
            if res:
                max_epoch: int | None = None
                for row in res:
                    try:
                        epoch = int(row[2])
                        if max_epoch is None or epoch > max_epoch:
                            max_epoch = epoch
                    except (TypeError, ValueError, IndexError):
                        pass
                if max_epoch is not None:
                    last_computed_at = (
                        datetime.fromtimestamp(max_epoch, tz=timezone.utc)
                        .isoformat()
                        .replace("+00:00", "Z")
                    )
        finally:
            conn.close()
    except Exception as exc:
        # Table may not exist yet (PageRank skipped on this repo) or the
        # `.duck` file was created by a stage that doesn't run the
        # centrality DDL. Either way, return empty rather than 5xx.
        logger.warning("centrality.read_failed repo=%s err=%s", name, exc)
        return CentralityListResponse(symbols=[])

    return CentralityListResponse(
        symbols=[CentralitySymbol(qname=qn, centrality=score) for qn, score in rows],
        last_computed_at=last_computed_at,
    )


# ---------------------------------------------------------------------------
# POST /repos/{name}/recompute-centrality — LE-32 manual trigger
# ---------------------------------------------------------------------------


class RecomputeCentralityResponse(BaseModel):
    """Result of ``POST /repos/{name}/recompute-centrality``."""

    repo: str = Field(description="Repo slug the computation ran against.")
    scores_written: int = Field(
        description="Number of PageRank rows written to the centrality table. "
        "0 when the graph has no CALLS edges (single-file repos are valid).",
    )
    message: str = Field(description="Human-readable status summary.")


@router.post(
    "/{name}/recompute-centrality",
    response_model=RecomputeCentralityResponse,
    summary="Recompute PageRank centrality for an existing index (LE-32)",
)
def recompute_centrality(name: str) -> RecomputeCentralityResponse:
    """Recompute and persist PageRank centrality for repo ``name``.

    Runs the same Plan J pipeline that ``POST /index`` executes at the end
    of a full ingest — useful when an index was created before the centrality
    bug fix (LE-32) landed, so callers can back-fill without a full reindex.

    Args:
        name: Repo slug (must already have a ``.db`` + ``.duck`` file on
            disk — i.e. the repo must have been indexed at least once).

    Returns:
        RecomputeCentralityResponse: Number of rows written and a status message.

    Raises:
        HTTPException: 404 when no ``.db`` file exists for ``name``.
        HTTPException: 503 when ``codebase_rag`` is not installed.
        HTTPException: 500 when the PageRank computation itself fails.
    """
    from pathlib import Path as _Path

    db_path = settings.db_path_for_repo(name)
    if not _Path(db_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"No LadybugDB index found for repo '{name}'. Run POST /index first.",
        )

    duck_path = settings.vec_db_path_for_repo(name)

    try:
        from codebase_rag.storage.centrality import compute_pagerank  # type: ignore[import-untyped]
        from codebase_rag.storage.vector_store import (  # type: ignore[import-untyped]
            clear_centrality,
            open_or_create,
            write_centrality,
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"codebase_rag package not available: {exc}",
        ) from exc

    try:
        pr_scores = compute_pagerank(db_path)
    except Exception as exc:
        logger.error(
            "recompute_centrality: compute_pagerank failed repo=%s err=%s",
            name, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"PageRank computation failed: {exc}",
        ) from exc

    if not pr_scores:
        return RecomputeCentralityResponse(
            repo=name,
            scores_written=0,
            message=(
                "PageRank returned 0 scores — the graph has no CALLS edges "
                "(single-file repos and pure-data repos are expected here)."
            ),
        )

    try:
        conn = open_or_create(duck_path)
        try:
            clear_centrality(conn)
            written = write_centrality(conn, pr_scores)
        finally:
            conn.close()
    except Exception as exc:
        logger.error(
            "recompute_centrality: write failed repo=%s err=%s",
            name, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to persist centrality scores: {exc}",
        ) from exc

    logger.info(
        "recompute_centrality: repo=%s scores_written=%d",
        name, written,
    )
    return RecomputeCentralityResponse(
        repo=name,
        scores_written=written,
        message=f"Wrote {written} PageRank rows to centrality table.",
    )


# ---------------------------------------------------------------------------
# GET /repos/{name}/centroid — Phase 2.5 v1 topic centroid (BUC-1581)
# ---------------------------------------------------------------------------


class RepoCentroidResponse(BaseModel):
    """Per-repo topic centroid response (BUC-1581).

    Unblocks TheForge's ``getRepoAffinity`` (BUC-1575). Returns a single
    768-dim vector — the mean of the top-k centrality symbols' embeddings
    — that callers can compare against a query embedding via cosine
    similarity to compute repo-vs-query affinity.

    The vector is **not** L2-normalised at compute time. Callers that want
    cosine arithmetic should normalise client-side.
    """

    repo: str = Field(description="Repo slug — same key used by /repos/{name}/stats")
    centroid: list[float] = Field(
        description="768-dim mean-pooled embedding vector. NOT L2-normalised."
    )
    computed_at: str = Field(
        description="ISO-8601 UTC timestamp when the centroid was computed. "
        "On a cache hit this reflects the cached entry's compute time, not "
        "the request time, so callers can detect a stale-but-served centroid."
    )
    cache_age_seconds: int = Field(
        description="Wall-clock seconds since ``computed_at``. 0 on a fresh "
        "compute; up to ~3600 on a cache hit (TTL is 1h)."
    )
    k: int = Field(
        description="Effective k used for the mean-pool. May be smaller than "
        "the requested ``k`` query-param when the centrality table has "
        "fewer rows than asked."
    )


@router.get(
    "/{name}/centroid",
    response_model=RepoCentroidResponse,
    summary="Per-repo topic centroid — mean of top-k centrality embeddings (BUC-1581)",
)
def repo_topic_centroid(
    name: str,
    k: int = Query(
        default=20,
        ge=1,
        le=200,
        description="Number of top-centrality symbols to mean-pool.",
    ),
) -> RepoCentroidResponse:
    """Compute (or serve from cache) the topic centroid for ``name``.

    Mean-pools the embedding vectors of the top-``k`` PageRank-central
    symbols into a single 768-dim vector. TheForge's affinity router uses
    this as the per-repo "topic signature" against which it scores incoming
    queries — replacing the v0 stub that returned uniform weights.

    Caching:
        Per-repo, per-k, in-process LRU with a 1h soft TTL. Re-indexing
        bumps the .duck mtime which automatically invalidates the entry.

    Args:
        name: Repo slug.
        k: Top-k centrality symbols to pool (1–200; default 20).

    Returns:
        RepoCentroidResponse: ``{ repo, centroid[768], computed_at,
        cache_age_seconds, k }``.

    Raises:
        HTTPException 404: Repo has not been indexed at all (no .duck file).
        HTTPException 503: Repo indexed but the centrality table or
            embeddings are not yet populated. Callers can poll — TheForge's
            affinity module already falls back to uniform weights on 503.
    """
    from ..services.centroid import (
        CentroidNotFoundError,
        CentroidUnavailableError,
        compute_repo_centroid,
    )

    duck_path = Path(settings.vec_db_path_for_repo(name))

    try:
        result = compute_repo_centroid(repo=name, duck_path=duck_path, k=k)
    except CentroidNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "repo_not_indexed",
                "message": str(exc),
            },
        )
    except CentroidUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "centroid_unavailable",
                "message": str(exc),
            },
        )

    return RepoCentroidResponse(
        repo=result.repo,
        centroid=result.centroid,
        computed_at=result.computed_at.isoformat().replace("+00:00", "Z"),
        cache_age_seconds=result.cache_age_seconds,
        k=result.k,
    )
