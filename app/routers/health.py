"""GET /health — liveness + indexed-repo summary."""
from __future__ import annotations

from fastapi import APIRouter

from ..config import settings
from ..models import HealthResponse

router = APIRouter()


def _get_indexed_repos() -> list[str]:
    """Return a deduplicated list of project names stored in LadybugDB.

    We query the ``Project`` node table introduced by CI-3 (ladybug_schema.py).
    If the DB file doesn't exist yet or the query fails we return an empty list
    — the service should still be healthy.
    """
    try:
        import real_ladybug as lb  # type: ignore[import-untyped]

        db = lb.Database(settings.LADYBUG_DB_PATH)
        conn = lb.Connection(db)
        result = conn.execute("MATCH (p:Project) RETURN p.name AS name")
        repos: list[str] = []
        while result.has_next():
            row = result.get_next()
            repos.append(str(row[0]))
        return sorted(set(repos))
    except Exception:
        return []


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    indexed = _get_indexed_repos()
    return HealthResponse(
        status="ok",
        db_path=settings.LADYBUG_DB_PATH,
        indexed_repos=indexed,
    )
