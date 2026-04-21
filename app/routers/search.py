"""GET /search/structural, /search/semantic, /search/symbol."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ..config import settings
from ..models import (
    SemanticResult,
    SemanticSearchResponse,
    StructuralSearchResponse,
    SymbolResponse,
)

router = APIRouter(prefix="/search")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_conn():  # type: ignore[override]  # returns lb.Connection
    import real_ladybug as lb  # type: ignore[import-untyped]

    db = lb.Database(settings.LADYBUG_DB_PATH)
    conn = lb.Connection(db)
    try:
        conn.execute("LOAD EXTENSION VECTOR")
    except Exception:
        pass
    return conn


def _result_to_rows(result: object) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    col_names = result.get_column_names()  # type: ignore[attr-defined]
    while result.has_next():  # type: ignore[attr-defined]
        raw = result.get_next()  # type: ignore[attr-defined]
        rows.append(dict(zip(col_names, raw)))
    return rows


def _is_node(v: Any) -> bool:
    return isinstance(v, dict) and "_LABEL" in v


def _is_rel(v: Any) -> bool:
    return isinstance(v, dict) and "_SRC" in v


def _clean(v: Any) -> Any:
    """Convert LadybugDB internal dicts to plain JSON-serialisable values."""
    if isinstance(v, dict):
        if "_LABEL" in v:
            # Node: strip internal keys
            return {k: _clean(val) for k, val in v.items() if not k.startswith("_")}
        if "_SRC" in v:
            return {k: _clean(val) for k, val in v.items() if not k.startswith("_")}
        return {k: _clean(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_clean(i) for i in v]
    return v


# ---------------------------------------------------------------------------
# GET /search/structural
# ---------------------------------------------------------------------------


@router.get("/structural", response_model=StructuralSearchResponse)
def structural_search(
    q: str = Query(description="Cypher query to execute against the graph"),
    limit: int = Query(default=20, ge=1, le=500),
) -> StructuralSearchResponse:
    """Execute a raw Cypher query and return matching nodes and relationships."""
    # Append LIMIT to guard against runaway queries (only if not already present).
    cypher = q.strip()
    if "LIMIT" not in cypher.upper():
        cypher = f"{cypher}\nLIMIT {limit}"

    try:
        conn = _get_conn()
        rows = _result_to_rows(conn.execute(cypher))  # type: ignore[attr-defined]
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cypher error: {exc}") from exc

    nodes: list[dict[str, Any]] = []
    rels: list[dict[str, Any]] = []

    for row in rows:
        for v in row.values():
            cleaned = _clean(v)
            if _is_node(v):
                nodes.append(cleaned)
            elif _is_rel(v):
                rels.append(cleaned)
            # scalar columns are discarded — use them in the query's RETURN directly

    # Also add non-node/rel scalars as plain row dicts when no structural data found.
    if not nodes and not rels:
        nodes = [_clean(row) for row in rows]

    return StructuralSearchResponse(
        nodes=nodes,
        relationships=rels,
        row_count=len(rows),
    )


# ---------------------------------------------------------------------------
# GET /search/semantic
# ---------------------------------------------------------------------------


@router.get("/semantic", response_model=SemanticSearchResponse)
def semantic_search(
    q: str = Query(description="Natural-language description of the code to find"),
    k: int = Query(default=10, ge=1, le=100),
) -> SemanticSearchResponse:
    """Find the top-k most semantically similar functions/methods."""
    try:
        from codebase_rag.tools.semantic_search import semantic_code_search
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Semantic search unavailable (missing deps): {exc}",
        ) from exc

    raw = semantic_code_search(q, top_k=k)
    return SemanticSearchResponse(
        results=[
            SemanticResult(
                symbol=r["qualified_name"],
                score=r["score"],
                type=r.get("type", ""),
            )
            for r in raw
        ]
    )


# ---------------------------------------------------------------------------
# GET /search/symbol
# ---------------------------------------------------------------------------


@router.get("/symbol", response_model=SymbolResponse)
def symbol_lookup(
    fqn: str = Query(description="Fully-qualified name of the function or method"),
) -> SymbolResponse:
    """Return source code and file location for a qualified symbol name."""
    from codebase_rag.cypher_queries import CYPHER_GET_FUNCTION_SOURCE_LOCATION

    try:
        conn = _get_conn()
        rows = _result_to_rows(
            conn.execute(  # type: ignore[attr-defined]
                CYPHER_GET_FUNCTION_SOURCE_LOCATION, {"node_id": fqn}
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"DB error: {exc}") from exc

    if not rows:
        raise HTTPException(status_code=404, detail=f"Symbol not found: {fqn}")

    row = rows[0]
    file_path: str = row.get("path") or ""
    line_start: int | None = row.get("start_line")
    line_end: int | None = row.get("end_line")

    source = ""
    if file_path and Path(file_path).exists() and line_start is not None:
        try:
            lines = Path(file_path).read_text(encoding="utf-8", errors="replace").splitlines()
            start = max(0, line_start - 1)
            end = line_end if line_end is not None else line_start
            source = "\n".join(lines[start:end])
        except Exception:
            pass

    return SymbolResponse(
        qualified_name=fqn,
        file=file_path,
        line_start=line_start,
        line_end=line_end,
        source=source,
    )
