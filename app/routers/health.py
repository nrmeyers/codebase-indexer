"""GET /health — liveness + indexed-repo summary.

This router exists so orchestrators (TheForge API, k8s, developers) can both
confirm the Code Indexer Service is reachable and discover which repositories
are currently represented in LadybugDB without having to run a search.
"""
from __future__ import annotations

from fastapi import APIRouter

from ..config import settings
from ..models import HealthResponse

router = APIRouter()


def _get_indexed_repos() -> list[str]:
    """Return a deduplicated list of project names stored in LadybugDB.

    We query the ``Project`` node table introduced by CI-3 (ladybug_schema.py).
    If the DB file doesn't exist yet or the query fails we return an empty
    list — the service should still report healthy so that a fresh deployment
    without any indexed repos is not considered degraded.

    Returns:
        list[str]: Sorted, deduplicated project names. Empty list on any
        error (missing DB file, missing extension, malformed schema).
    """
    try:
        import real_ladybug as lb  # type: ignore[import-untyped]

        db = lb.Database(settings.LADYBUG_DB_PATH)
        conn = lb.Connection(db)
        # Match every Project node and project just the name column.
        result = conn.execute("MATCH (p:Project) RETURN p.name AS name")
        repos: list[str] = []
        while result.has_next():
            row = result.get_next()
            repos.append(str(row[0]))
        return sorted(set(repos))
    except Exception:
        # Swallow: /health must never raise. Empty list tells the caller the
        # DB is not yet populated or temporarily unreachable.
        return []


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Readiness probe — always returns 200 with a status indicator.

    Returns:
        HealthResponse: ``status="ok"`` plus the resolved DB path and the
        set of currently-indexed project names.
    """
    indexed = _get_indexed_repos()
    return HealthResponse(
        status="ok",
        db_path=settings.LADYBUG_DB_PATH,
        indexed_repos=indexed,
    )
