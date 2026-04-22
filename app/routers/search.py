"""GET /search/structural, /search/semantic, /search/symbol.

Three complementary search surfaces against LadybugDB:

* ``/search/structural`` — raw Cypher passthrough for graph traversals.
* ``/search/semantic``  — vector-similarity search over function/method
  embeddings.
* ``/search/symbol``    — exact-name lookup returning source + location.

All three share a single ``_get_conn`` helper that lazily loads the VECTOR
extension so semantic search works even against a cold DB file.
"""
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
# Semantic search import cache
# ---------------------------------------------------------------------------
# The semantic search function lives inside codebase_rag, which may require
# torch/transformers. We cache the result of the first import attempt so that:
#   - A successful import is reused on every call (avoids repeated module init).
#   - A failed import short-circuits immediately on subsequent calls instead of
#     re-attempting the import each time (saves ~500ms per failed call on
#     deployments without ML deps).
_semantic_fn: Any = None          # cached callable when import succeeds
_semantic_unavailable: bool = False  # True once import fails; never retried


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_conn():  # type: ignore[override]  # returns lb.Connection
    """Open a fresh LadybugDB connection with the VECTOR extension loaded.

    Returns:
        lb.Connection: A connection usable for Cypher queries. The VECTOR
        extension is silently skipped if it's already loaded or unavailable —
        semantic search will return an error at call-time in that case.
    """
    import real_ladybug as lb  # type: ignore[import-untyped]

    db = lb.Database(settings.LADYBUG_DB_PATH)
    conn = lb.Connection(db)
    try:
        conn.execute("LOAD EXTENSION VECTOR")
    except Exception:
        # Already loaded or unavailable — both are non-fatal here.
        pass
    return conn


def _result_to_rows(result: object) -> list[dict[str, Any]]:
    """Consume a LadybugDB result iterator into a list of column-keyed dicts.

    Args:
        result: A LadybugDB query result with ``get_column_names``,
            ``has_next``, and ``get_next`` methods.

    Returns:
        list[dict[str, Any]]: One dict per row keyed by column name.
    """
    rows: list[dict[str, Any]] = []
    col_names = result.get_column_names()  # type: ignore[attr-defined]
    while result.has_next():  # type: ignore[attr-defined]
        raw = result.get_next()  # type: ignore[attr-defined]
        rows.append(dict(zip(col_names, raw)))
    return rows


def _is_node(v: Any) -> bool:
    """Return True if ``v`` is a LadybugDB node dict (identified by ``_LABEL``)."""
    return isinstance(v, dict) and "_LABEL" in v


def _is_rel(v: Any) -> bool:
    """Return True if ``v`` is a LadybugDB relationship dict (identified by ``_SRC``)."""
    return isinstance(v, dict) and "_SRC" in v


def _clean(v: Any) -> Any:
    """Convert LadybugDB internal dicts to plain JSON-serialisable values.

    Strips private keys (``_LABEL``, ``_SRC``, etc.) that LadybugDB uses to
    mark node/relationship metadata — these are not safe to expose to HTTP
    clients as-is and are recovered via ``_is_node`` / ``_is_rel`` first.

    Args:
        v: Any value from a query result — scalar, list, node dict, rel dict.

    Returns:
        The same value with internal-only keys removed and nested dicts/lists
        recursively cleaned.
    """
    if isinstance(v, dict):
        if "_LABEL" in v:
            # Node: strip internal keys so only user-defined properties escape.
            return {k: _clean(val) for k, val in v.items() if not k.startswith("_")}
        if "_SRC" in v:
            # Relationship: same strip rule as nodes.
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
    """Execute a raw Cypher query and return matching nodes and relationships.

    Args:
        q: Arbitrary Cypher query. If the query does not already contain a
            LIMIT clause, one is appended using the ``limit`` parameter.
        limit: Maximum rows to return (1–500). Only applied if ``q`` does
            not already include a LIMIT clause.

    Returns:
        StructuralSearchResponse: Nodes, relationships, and row count.

    Raises:
        HTTPException: 422 when the Cypher query is malformed.
    """
    # Append LIMIT to guard against runaway queries (only if not already
    # present — clients that need pagination can specify their own).
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

    # Split results by type so clients can render graphs directly. Scalar
    # columns (counts, strings) are discarded unless no structural data came
    # back (see fallback below).
    for row in rows:
        for v in row.values():
            cleaned = _clean(v)
            if _is_node(v):
                nodes.append(cleaned)
            elif _is_rel(v):
                rels.append(cleaned)
            # scalar columns are discarded — use them in the query's RETURN directly

    # Fallback: if the user issued a pure-scalar query (e.g. aggregates),
    # expose the rows so they aren't lost to the caller.
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
    """Find the top-k most semantically similar functions/methods.

    Args:
        q: Natural-language description (e.g. "function that retries HTTP
            requests with exponential backoff"). Embedded via the same model
            used at ingestion time and compared against the Embedding node
            table.
        k: Number of results to return (1–100).

    Returns:
        SemanticSearchResponse: Ranked list of qualified names with scores.

    Raises:
        HTTPException: 503 when the semantic search dependency
            (``codebase_rag.tools.semantic_search``) is not importable —
            typically because the embedding model or VECTOR extension is
            missing from the deployment. Subsequent calls after a failed
            import return 503 immediately without re-attempting the import
            (fast-fail via ``_semantic_unavailable`` flag).
    """
    global _semantic_fn, _semantic_unavailable  # noqa: PLW0603

    # Fast-fail path: import previously failed — don't retry.
    if _semantic_unavailable:
        raise HTTPException(
            status_code=503,
            detail="Semantic search unavailable (missing deps; import failed on first attempt)",
        )

    # Lazy-load path: first call (or after a successful warm-up).
    if _semantic_fn is None:
        try:
            from codebase_rag.tools.semantic_search import semantic_code_search  # type: ignore[import-untyped]
            _semantic_fn = semantic_code_search
        except ImportError as exc:
            _semantic_unavailable = True
            raise HTTPException(
                status_code=503,
                detail=f"Semantic search unavailable (missing deps): {exc}",
            ) from exc

    raw = _semantic_fn(q, top_k=k)
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
    """Return source code and file location for a qualified symbol name.

    Args:
        fqn: Fully-qualified symbol name (e.g. ``myapp.utils.retry``).

    Returns:
        SymbolResponse: Location metadata plus the source snippet read from
        disk. Source is empty when the file cannot be read (e.g. repo moved).

    Raises:
        HTTPException: 404 when no node with that qualified name exists,
            500 on unexpected DB errors.
    """
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
    root_path: str = row.get("root_path") or ""
    line_start: int | None = row.get("start_line")
    line_end: int | None = row.get("end_line")

    # Paths stored in LadybugDB are relative to the repo root (for portability).
    # Resolve to absolute using the root_path stored on the Project node; fall
    # back to treating the path as-is when root_path is unavailable (e.g. DBs
    # indexed before this field was added).
    if file_path and root_path and not Path(file_path).is_absolute():
        file_path = str(Path(root_path) / file_path)

    # Read the source directly from disk rather than storing it in the DB —
    # keeps the graph compact and guarantees freshness if the file changed
    # between ingestion and query.
    source = ""
    if file_path and Path(file_path).exists() and line_start is not None:
        try:
            lines = Path(file_path).read_text(encoding="utf-8", errors="replace").splitlines()
            # Cypher stores 1-indexed lines; Python slicing is 0-indexed and
            # end-exclusive. Using line_end directly (when set) keeps the
            # last line inclusive as users expect.
            start = max(0, line_start - 1)
            end = line_end if line_end is not None else line_start
            source = "\n".join(lines[start:end])
        except Exception:
            # File may have been moved/deleted since ingestion — swallow so
            # the metadata response is still useful.
            pass

    return SymbolResponse(
        qualified_name=fqn,
        file=file_path,
        line_start=line_start,
        line_end=line_end,
        source=source,
    )
