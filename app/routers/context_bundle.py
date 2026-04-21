"""POST /context-bundle — build a grounded code context for the dev-agent.

The dev-agent (TheForge) needs a small, high-signal slice of the repository
to feed into an LLM prompt when implementing a task. This endpoint assembles
that slice by:

    1. Semantic search over the task description to find relevant seed
       functions/methods.
    2. Expansion through the CALLS graph up to ``depth`` hops so callees are
       included (so the LLM sees what the seed symbols actually do).
    3. Source snippet retrieval for every reached symbol.
    4. A rough token estimate so the caller can budget the prompt window.

Models are defined locally (not in ``app/models.py``) because they are
specific to this router and not reused elsewhere.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..config import settings

router = APIRouter()

# Rough token estimate: ~4 characters per token for typical English/code
# mixes. Good enough for prompt-window budgeting; exact tokenization varies
# per model and is not worth pulling in a tokenizer dependency for.
_CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# Models (local to this router — not shared in models.py)
# ---------------------------------------------------------------------------


class ContextBundleRequest(BaseModel):
    """Request body for ``POST /context-bundle``."""

    repo_path: str = Field(description="Absolute or relative path to the indexed repo")
    task_description: str = Field(description="Natural-language description of the dev task")
    k: int = Field(default=10, ge=1, le=50, description="Number of seed symbols from semantic search")
    depth: int = Field(default=2, ge=0, le=4, description="Call-graph hop depth")


class ContextBundleResponse(BaseModel):
    """Response body for ``POST /context-bundle``.

    Attributes:
        symbols: Every qualified name in the bundle (seeds + expansion).
        source_snippets: Map of qualified name → source code. Empty string
            when the symbol's file could not be read.
        call_graph: Adjacency list ``caller → [callees]`` limited to edges
            discovered during BFS expansion.
        total_tokens: Rough estimate of token cost if every snippet were
            concatenated into a prompt.
    """

    symbols: list[str]
    source_snippets: dict[str, str]
    call_graph: dict[str, list[str]]
    total_tokens: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_conn():  # type: ignore[override]
    """Open a LadybugDB connection with the VECTOR extension lazily loaded.

    Returns:
        lb.Connection: A connection usable for Cypher queries.
    """
    import real_ladybug as lb  # type: ignore[import-untyped]

    db = lb.Database(settings.LADYBUG_DB_PATH)
    conn = lb.Connection(db)
    try:
        conn.execute("LOAD EXTENSION VECTOR")
    except Exception:
        # Already loaded or unavailable — both are non-fatal.
        pass
    return conn


def _result_to_rows(result: object) -> list[dict]:
    """Consume a LadybugDB result iterator into a list of column-keyed dicts."""
    rows = []
    col_names = result.get_column_names()  # type: ignore[attr-defined]
    while result.has_next():  # type: ignore[attr-defined]
        raw = result.get_next()  # type: ignore[attr-defined]
        rows.append(dict(zip(col_names, raw)))
    return rows


def _fetch_source(file_path: str, line_start: int | None, line_end: int | None) -> str:
    """Read a source slice from disk between 1-indexed start/end lines.

    Args:
        file_path: Absolute path to the file.
        line_start: 1-indexed start line; defaults to 1 when ``None``.
        line_end: 1-indexed inclusive end line; defaults to one line past
            ``line_start`` when ``None``.

    Returns:
        str: The joined source lines, or empty string on any read failure.
    """
    if not file_path or not Path(file_path).exists():
        return ""
    try:
        lines = Path(file_path).read_text(encoding="utf-8", errors="replace").splitlines()
        # 1-indexed → 0-indexed slice start; fall back to line 1 when unset.
        start = max(0, (line_start or 1) - 1)
        end = line_end or (start + 1)
        return "\n".join(lines[start:end])
    except Exception:
        # Swallow — the bundle is still useful without one file's source.
        return ""


def _fetch_source_for_symbols(
    conn: object, qualified_names: list[str]
) -> dict[str, str]:
    """Return ``{qualified_name → source_snippet}`` for a list of symbols.

    Args:
        conn: An open LadybugDB connection.
        qualified_names: The symbols whose source should be read.

    Returns:
        dict[str, str]: Per-symbol source snippets. Missing or unreadable
        symbols map to an empty string rather than being omitted.
    """
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
            # Record an empty string so the caller can see which symbols
            # failed to resolve rather than silently dropping them.
            snippets[qn] = ""
    return snippets


def _expand_call_graph(
    conn: object, seed_symbols: list[str], depth: int
) -> tuple[set[str], dict[str, list[str]]]:
    """BFS over the CALLS graph up to ``depth`` hops from the seed symbols.

    Args:
        conn: An open LadybugDB connection.
        seed_symbols: Qualified names to start BFS from (seed set).
        depth: Maximum number of hops to traverse. 0 returns only seeds.

    Returns:
        tuple:
            * all_symbols: every symbol encountered (seeds + reachable).
            * call_graph: ``{caller → [callee, ...]}`` for every edge
              traversed during BFS.
    """
    call_graph: dict[str, list[str]] = {}
    all_symbols: set[str] = set(seed_symbols)
    frontier: set[str] = set(seed_symbols)

    # Standard BFS: expand one hop per iteration, tracking only newly-reached
    # symbols in next_frontier to avoid revisiting.
    for _ in range(depth):
        if not frontier:
            break
        next_frontier: set[str] = set()
        for sym in frontier:
            try:
                # Parameterised Cypher: look up outgoing CALLS edges from the
                # current symbol and return callee qualified names.
                rows = _result_to_rows(
                    conn.execute(  # type: ignore[attr-defined]
                        "MATCH (n {qualified_name: $qn})-[:CALLS]->(m) "
                        "RETURN m.qualified_name AS callee",
                        {"qn": sym},
                    )
                )
            except Exception:
                # A broken symbol node should not abort the whole expansion.
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
        1. Semantic search: find top-k functions/methods most relevant to
           the task.
        2. Expand via CALLS graph up to ``depth`` hops.
        3. Fetch source snippets for every symbol in the expanded set.
        4. Return ``{symbols, source_snippets, call_graph, total_tokens}``.

    Args:
        req: Validated request body with repo path, task description, k,
            and depth parameters.

    Returns:
        ContextBundleResponse: The assembled bundle; empty fields when the
        seed search returns no matches.

    Raises:
        HTTPException: 503 when semantic search is unavailable.
    """
    # 1. Semantic seed — find the most task-relevant functions/methods.
    try:
        from codebase_rag.tools.semantic_search import semantic_code_search
        seed_results = semantic_code_search(req.task_description, top_k=req.k)
        seed_symbols = [r["qualified_name"] for r in seed_results]
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Semantic search unavailable: {exc}",
        ) from exc

    # No seed matches → return an empty bundle rather than an error; the
    # caller can decide whether to widen the task description.
    if not seed_symbols:
        return ContextBundleResponse(
            symbols=[],
            source_snippets={},
            call_graph={},
            total_tokens=0,
        )

    # 2. Expand call graph to pick up callees the LLM will need to reason
    #    about (default depth=2 balances breadth vs. prompt size).
    conn = _get_conn()
    all_symbols, call_graph = _expand_call_graph(conn, seed_symbols, req.depth)

    # 3. Fetch source snippets — sorted for deterministic output.
    source_snippets = _fetch_source_for_symbols(conn, sorted(all_symbols))

    # 4. Token estimate — char-count / 4 is accurate within ~20% for mixed
    #    code/English content and avoids pulling in a tokenizer.
    total_chars = sum(len(s) for s in source_snippets.values())
    total_tokens = total_chars // _CHARS_PER_TOKEN

    return ContextBundleResponse(
        symbols=sorted(all_symbols),
        source_snippets=source_snippets,
        call_graph=call_graph,
        total_tokens=total_tokens,
    )
