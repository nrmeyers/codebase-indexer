"""Symbol-detail endpoints (BACKEND_HANDOVER §2.7, §2.9).

Powers the HotspotsPanel symbol-detail view (3-column source / callers /
callees) and the SearchPlayground "Symbol" tab.

These are path-parameter analogues to the existing query-string
``/search/symbol`` endpoint; we expose both shapes so direct-link URLs
like ``/api/code-indexer/symbols/app.foo.bar`` work without
URL-encoding the FQN twice through the proxy.

Endpoints:
    GET /symbols/{fqn}/callers  — symbols that CALL this one  (declared first)
    GET /symbols/{fqn}/callees  — symbols this one CALLS      (declared first)
    GET /symbols/{fqn}          — exact-name lookup with source

Route order matters: FastAPI matches in declaration order, and ``path``
converters greedily consume the rest of the URL. The catch-all
``/{fqn:path}`` route therefore MUST be declared after the
``/callers`` and ``/callees`` variants — otherwise a request to
``/symbols/foo.bar/callers`` would match the lookup route with
``fqn = "foo.bar/callers"`` and the callers handler would be unreachable.
"""
from __future__ import annotations

import logging
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Query

from ..models import (
    CallSiteResponse,
    CallSiteResult,
    SymbolResponse,
)
from .search import _get_conn, _resolve_db_path, _result_to_rows, symbol_lookup

router = APIRouter(prefix="/symbols", tags=["symbols"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers — caller/callee queries
# ---------------------------------------------------------------------------


# Two-pass UNION ALL because LadybugDB's CALLS edge connects to either a
# Function (defined directly by Module) or a Method (defined transitively
# via Class-[:DEFINES_METHOD]). A single MATCH can't handle both file-path
# resolutions, so we UNION ALL.
_CALLERS_CYPHER = """
MATCH (caller:Function)-[:CALLS]->(target)
WHERE target.qualified_name = $fqn
MATCH (m:Module)-[:DEFINES]->(caller)
RETURN caller.qualified_name AS qualified_name,
       m.path                AS file_path,
       caller.start_line     AS line_number
UNION ALL
MATCH (caller:Method)-[:CALLS]->(target)
WHERE target.qualified_name = $fqn
MATCH (m:Module)-[:DEFINES]->(:Class)-[:DEFINES_METHOD]->(caller)
RETURN caller.qualified_name AS qualified_name,
       m.path                AS file_path,
       caller.start_line     AS line_number
"""

_CALLEES_CYPHER = """
MATCH (source)-[:CALLS]->(callee:Function)
WHERE source.qualified_name = $fqn
MATCH (m:Module)-[:DEFINES]->(callee)
RETURN callee.qualified_name AS qualified_name,
       m.path                AS file_path,
       callee.start_line     AS line_number
UNION ALL
MATCH (source)-[:CALLS]->(callee:Method)
WHERE source.qualified_name = $fqn
MATCH (m:Module)-[:DEFINES]->(:Class)-[:DEFINES_METHOD]->(callee)
RETURN callee.qualified_name AS qualified_name,
       m.path                AS file_path,
       callee.start_line     AS line_number
"""


def _run_relation_query(
    cypher: str, fqn: str, repo: str | None
) -> list[CallSiteResult]:
    """Run a callers-or-callees query and convert rows to ``CallSiteResult``s.

    Returns an empty list (not an error) when the symbol has no
    callers/callees so the FE can render its empty-state copy. The DB
    resolution still raises 404 when ``repo`` is supplied but its index
    file is missing — the FE distinguishes that from "no callers" via the
    HTTP status code.
    """
    # _resolve_db_path raises 404 when repo is supplied but missing — let
    # that surface, the FE handles 404.
    _resolve_db_path(repo)

    conn = _get_conn(repo)
    try:
        rows = _result_to_rows(conn.execute(cypher, {"fqn": fqn}))  # type: ignore[attr-defined]
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Caller/callee query failed for %s: %s", fqn, exc)
        return []

    results: list[CallSiteResult] = []
    seen: set[str] = set()
    for r in rows:
        qn = r.get("qualified_name")
        if not qn or qn in seen:
            continue
        seen.add(qn)
        ln = r.get("line_number")
        results.append(
            CallSiteResult(
                qualified_name=qn,
                file_path=r.get("file_path") or "",
                line_number=int(ln) if isinstance(ln, (int, float)) else 0,
            )
        )
    return results


# ---------------------------------------------------------------------------
# GET /symbols/{fqn}/callers  — DECLARE FIRST so it wins over the catch-all
# ---------------------------------------------------------------------------


@router.get("/{fqn:path}/callers", response_model=CallSiteResponse)
def callers_for_symbol(
    fqn: str,
    repo: str | None = Query(default=None),
) -> CallSiteResponse:
    """Return the symbols that ``CALL`` ``fqn``.

    Args:
        fqn: Fully-qualified name of the target symbol.
        repo: Optional repo slug to scope the query to.

    Returns:
        CallSiteResponse: Empty list when the symbol has no callers;
        deduplicated by qualified_name otherwise.
    """
    decoded = unquote(fqn)
    return CallSiteResponse(
        results=_run_relation_query(_CALLERS_CYPHER, decoded, repo),
    )


# ---------------------------------------------------------------------------
# GET /symbols/{fqn}/callees
# ---------------------------------------------------------------------------


@router.get("/{fqn:path}/callees", response_model=CallSiteResponse)
def callees_for_symbol(
    fqn: str,
    repo: str | None = Query(default=None),
) -> CallSiteResponse:
    """Return the symbols that ``fqn`` calls.

    Args:
        fqn: Fully-qualified name of the source symbol.
        repo: Optional repo slug to scope the query to.

    Returns:
        CallSiteResponse: Empty list when the symbol calls nothing;
        deduplicated by qualified_name otherwise.
    """
    decoded = unquote(fqn)
    return CallSiteResponse(
        results=_run_relation_query(_CALLEES_CYPHER, decoded, repo),
    )


# ---------------------------------------------------------------------------
# GET /symbols/{fqn}  — declared LAST so /callers and /callees match first
# ---------------------------------------------------------------------------


@router.get("/{fqn:path}", response_model=SymbolResponse)
def symbol_by_path(
    fqn: str,
    repo: str | None = Query(
        default=None,
        description="Repo slug to scope the lookup to. Omit for first indexed DB.",
    ),
) -> SymbolResponse:
    """Path-parameter wrapper around ``/search/symbol``.

    ``fqn`` may contain dots, slashes, or ``::`` separators — FastAPI's
    ``path`` converter passes the whole tail through. We URL-decode it so
    ``/symbols/app%2Efoo`` and ``/symbols/app.foo`` are equivalent.
    """
    decoded = unquote(fqn)
    return symbol_lookup(fqn=decoded, repo=repo)


__all__ = ["router"]
