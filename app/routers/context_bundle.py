"""POST /context-bundle — build a grounded code context for the TheForge dev-agent."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..config import settings

router = APIRouter()

# Rough token estimate: 4 chars ≈ 1 token
_CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# Models (local to this router — not shared in models.py)
# ---------------------------------------------------------------------------


class ContextBundleRequest(BaseModel):
    repo_path: str = Field(description="Absolute or relative path to the indexed repo")
    task_description: str = Field(description="Natural-language description of the dev task")
    k: int = Field(default=10, ge=1, le=50, description="Number of seed symbols from semantic search")
    depth: int = Field(default=2, ge=0, le=4, description="Call-graph hop depth")


class ContextBundleResponse(BaseModel):
    symbols: list[str]
    source_snippets: dict[str, str]
    call_graph: dict[str, list[str]]
    total_tokens: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_conn():  # type: ignore[override]
    import real_ladybug as lb  # type: ignore[import-untyped]

    db = lb.Database(settings.LADYBUG_DB_PATH)
    conn = lb.Connection(db)
    try:
        conn.execute("LOAD EXTENSION VECTOR")
    except Exception:
        pass
    return conn


def _result_to_rows(result: object) -> list[dict]:
    rows = []
    col_names = result.get_column_names()  # type: ignore[attr-defined]
    while result.has_next():  # type: ignore[attr-defined]
        raw = result.get_next()  # type: ignore[attr-defined]
        rows.append(dict(zip(col_names, raw)))
    return rows


def _fetch_source(file_path: str, line_start: int | None, line_end: int | None) -> str:
    if not file_path or not Path(file_path).exists():
        return ""
    try:
        lines = Path(file_path).read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(0, (line_start or 1) - 1)
        end = line_end or (start + 1)
        return "\n".join(lines[start:end])
    except Exception:
        return ""


def _fetch_source_for_symbols(
    conn: object, qualified_names: list[str]
) -> dict[str, str]:
    """Return { qualified_name → source_snippet } for a list of symbols."""
    from codebase_rag.cypher_queries import CYPHER_GET_FUNCTION_SOURCE_LOCATION

    snippets: dict[str, str] = {}
    for qn in qualified_names:
        try:
            rows = _result_to_rows(
                conn.execute(CYPHER_GET_FUNCTION_SOURCE_LOCATION, {"node_id": qn})  # type: ignore[attr-defined]
            )
            if rows:
                r = rows[0]
                snippets[qn] = _fetch_source(r.get("path", ""), r.get("start_line"), r.get("end_line"))
        except Exception:
            snippets[qn] = ""
    return snippets


def _expand_call_graph(
    conn: object, seed_symbols: list[str], depth: int
) -> tuple[set[str], dict[str, list[str]]]:
    """BFS over the CALLS graph up to ``depth`` hops from the seed symbols.

    Returns:
        all_symbols: every symbol encountered (seeds + reachable)
        call_graph:  { caller → [callee, ...] }
    """
    call_graph: dict[str, list[str]] = {}
    all_symbols: set[str] = set(seed_symbols)
    frontier: set[str] = set(seed_symbols)

    for _ in range(depth):
        if not frontier:
            break
        next_frontier: set[str] = set()
        for sym in frontier:
            try:
                rows = _result_to_rows(
                    conn.execute(  # type: ignore[attr-defined]
                        "MATCH (n {qualified_name: $qn})-[:CALLS]->(m) "
                        "RETURN m.qualified_name AS callee",
                        {"qn": sym},
                    )
                )
            except Exception:
                continue
            callees = [r["callee"] for r in rows if r.get("callee")]
            if callees:
                call_graph[sym] = callees
            for c in callees:
                if c not in all_symbols:
                    all_symbols.add(c)
                    next_frontier.add(c)
        frontier = next_frontier

    return all_symbols, call_graph


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/context-bundle", response_model=ContextBundleResponse)
def build_context_bundle(req: ContextBundleRequest) -> ContextBundleResponse:
    """Build a grounded code context bundle for a dev-agent task prompt.

    Steps:
    1. Semantic search: find top-k functions/methods most relevant to the task.
    2. Expand via CALLS graph up to ``depth`` hops.
    3. Fetch source snippets for every symbol in the expanded set.
    4. Return { symbols, source_snippets, call_graph, total_tokens }.
    """
    # 1. Semantic seed
    try:
        from codebase_rag.tools.semantic_search import semantic_code_search
        seed_results = semantic_code_search(req.task_description, top_k=req.k)
        seed_symbols = [r["qualified_name"] for r in seed_results]
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Semantic search unavailable: {exc}",
        ) from exc

    if not seed_symbols:
        return ContextBundleResponse(
            symbols=[],
            source_snippets={},
            call_graph={},
            total_tokens=0,
        )

    # 2. Expand call graph
    conn = _get_conn()
    all_symbols, call_graph = _expand_call_graph(conn, seed_symbols, req.depth)

    # 3. Fetch source snippets
    source_snippets = _fetch_source_for_symbols(conn, sorted(all_symbols))

    # 4. Token estimate
    total_chars = sum(len(s) for s in source_snippets.values())
    total_tokens = total_chars // _CHARS_PER_TOKEN

    return ContextBundleResponse(
        symbols=sorted(all_symbols),
        source_snippets=source_snippets,
        call_graph=call_graph,
        total_tokens=total_tokens,
    )
