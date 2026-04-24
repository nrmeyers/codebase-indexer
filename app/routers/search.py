"""GET /search/structural, /search/semantic, /search/symbol.

Three complementary search surfaces against LadybugDB:

* ``/search/structural`` — raw Cypher passthrough for graph traversals.
* ``/search/semantic``  — numpy cosine-similarity search over function/method
  embeddings stored in per-repo ``.embeddings.npy`` files.
* ``/search/symbol``    — exact-name lookup returning source + location.

Semantic search does NOT require the LadybugDB VECTOR extension — embeddings
are stored in numpy files alongside the ``.db`` file and searched in-process.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ..config import settings
from ..models import (
    FileEntry,
    FileListResponse,
    GraphEdge,
    GraphNode,
    GraphOverviewResponse,
    NodeTypeStat,
    NodeTypesResponse,
    SemanticResult,
    SemanticSearchResponse,
    StructuralSearchResponse,
    SymbolResponse,
)

router = APIRouter(prefix="/search")

# Cypher keywords that mutate graph state.  The structural endpoint is a
# read-only query surface; any of these in a client-supplied query is
# rejected early so a typo can't accidentally wipe or corrupt a repo's
# graph.  Word-boundary matching means these are only blocked as top-level
# clauses, not as substrings inside a string literal (e.g. WHERE n.name =
# "DELETE me" stays legal).
_WRITE_KEYWORDS = (
    "CREATE",
    "MERGE",
    "DELETE",
    "DETACH",
    "SET",
    "REMOVE",
    "DROP",
    "COPY",
    "CALL",   # conservative: CALL procedures can mutate — use semantic endpoint instead
    "LOAD",
)

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


def _resolve_db_path(repo: str | None) -> str:
    """Pick the LadybugDB file a query should run against.

    Args:
        repo: Optional repo slug.  When given, resolves to that repo's
            per-repo DB file.  When omitted, falls back to the first indexed
            repo on disk, or the legacy combined ``LADYBUG_DB_PATH``.

    Returns:
        str: Filesystem path to the DB the caller should open.

    Raises:
        HTTPException: 404 when ``repo`` is supplied but no matching DB
            exists — signalling the caller to index that repo first.
    """
    if repo:
        path = settings.db_path_for_repo(repo)
        if not Path(path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"No index found for repo '{repo}'. Run /index first.",
            )
        return path

    # No repo specified — try the first indexed DB, else legacy combined file.
    db_dir = Path(settings.LADYBUG_DB_DIR)
    if db_dir.is_dir():
        dbs = sorted(db_dir.glob("*.db"))
        if dbs:
            return str(dbs[0])
    return settings.LADYBUG_DB_PATH


def _get_conn(repo: str | None = None):  # type: ignore[override]  # returns lb.Connection
    """Open a fresh LadybugDB connection for structural/symbol queries.

    Args:
        repo: Optional repo slug.  Routes the connection to that repo's
            per-repo DB file.  When omitted, falls back to the first indexed
            DB or the legacy combined path.

    Returns:
        lb.Connection: A connection usable for Cypher queries.
    """
    import real_ladybug as lb  # type: ignore[import-untyped]

    db_path = _resolve_db_path(repo)
    db = lb.Database(db_path)
    conn = lb.Connection(db)
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
    repo: str | None = Query(
        default=None,
        description="Repo slug to scope the query to. Omit for first indexed DB.",
    ),
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
    if not cypher:
        raise HTTPException(status_code=422, detail="Query must not be empty")

    # Strip string literals before scanning for write keywords — keywords
    # inside quotes (e.g. n.name CONTAINS "DELETE") are not mutations.
    import re as _re
    scan_target = _re.sub(r"'[^']*'|\"[^\"]*\"", "''", cypher)
    for kw in _WRITE_KEYWORDS:
        if _re.search(rf"\b{kw}\b", scan_target, _re.IGNORECASE):
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Write operations are not permitted through /search/structural "
                    f"(keyword '{kw}' detected). Use POST /index to mutate the graph."
                ),
            )

    # Detect a top-level LIMIT clause using a word-boundary regex so we don't
    # match the word "limit" inside a string literal (e.g. WHERE n.name = "limit")
    # or a sub-query. A simple .upper() substring check would match literals.
    if not _re.search(r'\bLIMIT\b', cypher, _re.IGNORECASE):
        cypher = f"{cypher}\nLIMIT {limit}"

    try:
        conn = _get_conn(repo)
        rows = _result_to_rows(conn.execute(cypher))  # type: ignore[attr-defined]
    except HTTPException:
        raise  # e.g. 404 from _resolve_db_path — preserve status code
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
    repo: str | None = Query(
        default=None,
        description="Repo slug to scope the search to. Omit for first indexed DB.",
    ),
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
            typically because the embedding model (torch/transformers) is
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

    # Point code-graph-rag at the right per-repo DB before the search runs —
    # the semantic helper reads ``cgr_settings.LADYBUG_DB_PATH`` internally.
    try:
        from codebase_rag.config import settings as _cgr_settings  # type: ignore[import-untyped]
        _cgr_settings.LADYBUG_DB_PATH = _resolve_db_path(repo)
    except HTTPException:
        raise
    except Exception:
        # Non-fatal — if we can't swap the path, fall back to cgr's default.
        pass

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
    repo: str | None = Query(
        default=None,
        description="Repo slug to scope the lookup to. Omit for first indexed DB.",
    ),
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
        conn = _get_conn(repo)
        rows = _result_to_rows(
            conn.execute(  # type: ignore[attr-defined]
                CYPHER_GET_FUNCTION_SOURCE_LOCATION, {"node_id": fqn}
            )
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB error: {exc}") from exc

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
            # line_end is 1-indexed inclusive; Python slice end is exclusive so
            # line_end passes through directly. When line_end is absent (None),
            # fall back to start+1 so we still return the single start line
            # rather than an empty slice (lines[start:start] = []).
            end = line_end if line_end is not None else line_start + 1
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


# ---------------------------------------------------------------------------
# GET /search/files
# ---------------------------------------------------------------------------


@router.get("/files", response_model=FileListResponse)
def list_files(
    repo: str | None = Query(
        default=None,
        description="Repo slug to list files for. Omit for first indexed DB.",
    ),
    filter: str = Query(
        default="",
        description="Case-insensitive substring filter applied to the relative path.",
    ),
    extension: str = Query(
        default="",
        description="Optional extension filter (e.g. '.ts', 'py'). Leading dot optional.",
    ),
    limit: int = Query(default=500, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
) -> FileListResponse:
    """List indexed files with optional substring / extension filtering.

    This is a dedicated surface so the UI doesn't have to hand-write Cypher
    to render a file tree or search-as-you-type box.  Filtering is done in
    Cypher when possible to avoid shipping thousands of rows over the wire.

    Args:
        repo: Repo slug to scope the listing to.
        filter: Case-insensitive substring applied against the path column.
        extension: Optional file extension filter.
        limit: Max rows to return (1–5000).
        offset: Skip N matching rows before returning results (cursor paging).

    Returns:
        FileListResponse: matching files + total match count (post-filter).
    """
    # Normalise extension — accept both '.ts' and 'ts'.
    ext = extension.strip().lower()
    if ext and not ext.startswith("."):
        ext = "." + ext

    where_parts: list[str] = []
    params: dict[str, Any] = {}
    if filter.strip():
        where_parts.append("toLower(f.path) CONTAINS toLower($needle)")
        params["needle"] = filter.strip()
    if ext:
        where_parts.append("toLower(f.extension) = $ext")
        params["ext"] = ext

    where_clause = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
    count_cypher = f"MATCH (f:File){where_clause} RETURN count(f) AS cnt"
    list_cypher = (
        f"MATCH (f:File){where_clause} "
        f"RETURN f.path AS path, f.name AS name, f.extension AS extension "
        f"ORDER BY f.path SKIP {int(offset)} LIMIT {int(limit)}"
    )

    try:
        conn = _get_conn(repo)
        total = 0
        cnt_res = conn.execute(count_cypher, params) if params else conn.execute(count_cypher)
        if cnt_res.has_next():  # type: ignore[attr-defined]
            total = int(cnt_res.get_next()[0])  # type: ignore[attr-defined]

        rows_res = conn.execute(list_cypher, params) if params else conn.execute(list_cypher)
        rows = _result_to_rows(rows_res)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB error: {exc}") from exc

    return FileListResponse(
        files=[
            FileEntry(
                path=str(r.get("path") or ""),
                name=str(r.get("name") or ""),
                extension=str(r.get("extension") or ""),
            )
            for r in rows
        ],
        total=total,
    )


# ---------------------------------------------------------------------------
# GET /search/types
# ---------------------------------------------------------------------------

# Node labels the schema defines.  Iterating these is far cheaper than a
# MATCH (n) and a CASE label switch, and lets the endpoint stay correct
# even when a label has zero rows.
_KNOWN_LABELS = (
    "Project",
    "File",
    "Folder",
    "Package",
    "Module",
    "Class",
    "Function",
    "Method",
    "Interface",
    "Variable",
    "Struct",
    "Enum",
    "Type",
)


@router.get("/types", response_model=NodeTypesResponse)
def list_node_types(
    repo: str | None = Query(
        default=None,
        description="Repo slug to scope the summary to. Omit for first indexed DB.",
    ),
    non_zero_only: bool = Query(
        default=True,
        description="Drop labels with zero rows from the response.",
    ),
) -> NodeTypesResponse:
    """Return the node labels present in the graph with per-label counts.

    Useful for UIs that build Browse tabs dynamically — no need to hardcode
    which labels exist (which would miss new node types added to the schema).

    Args:
        repo: Repo slug to scope to.
        non_zero_only: When true (default), only labels with at least one
            row are returned.  Set false to probe the full schema.

    Returns:
        NodeTypesResponse: label + count pairs, sorted by count descending.
    """
    try:
        conn = _get_conn(repo)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB error: {exc}") from exc

    stats: list[NodeTypeStat] = []
    for label in _KNOWN_LABELS:
        try:
            res = conn.execute(f"MATCH (n:{label}) RETURN count(n) AS cnt")
            cnt = 0
            if res.has_next():  # type: ignore[attr-defined]
                cnt = int(res.get_next()[0])  # type: ignore[attr-defined]
            if cnt or not non_zero_only:
                stats.append(NodeTypeStat(label=label, count=cnt))
        except Exception:
            # Label may not exist in this DB (schema evolves over time) —
            # skip it rather than 500 the whole response.
            continue

    stats.sort(key=lambda s: s.count, reverse=True)
    return NodeTypesResponse(types=stats)


# ---------------------------------------------------------------------------
# GET /search/graph/overview
# ---------------------------------------------------------------------------


def _node_stable_id(v: dict[str, Any]) -> str:
    """Derive a stable string ID for a raw LadybugDB node dict."""
    return str(v.get("qname") or v.get("path") or v.get("name") or id(v))


@router.get("/graph/overview", response_model=GraphOverviewResponse)
def graph_overview(
    repo: str | None = Query(
        default=None,
        description="Repo slug to scope the graph to. Omit for first indexed DB.",
    ),
    max_nodes: int = Query(
        default=300,
        ge=1,
        le=2000,
        description="Maximum number of nodes to return.",
    ),
) -> GraphOverviewResponse:
    """Return a compact graph (nodes + edges) for repo-wide canvas rendering.

    Fetches relationships from the graph, derives stable node IDs from each
    node's ``qname`` or ``path`` property, and caps output at ``max_nodes``.
    Nodes with no relationships are not included (overview emphasises
    connectivity rather than exhaustive enumeration).

    Args:
        repo: Repo slug to scope to.
        max_nodes: Cap on unique nodes included in the response.

    Returns:
        GraphOverviewResponse: Nodes, edges, and counts.

    Raises:
        HTTPException: 503 when the DB cannot be opened.
    """
    try:
        conn = _get_conn(repo)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB error: {exc}") from exc

    # Fetch relationship triples.  We pull more than max_nodes worth so we
    # can include nodes that appear only as targets.
    edge_limit = max_nodes * 5
    try:
        cypher = f"MATCH (a)-[r]->(b) RETURN a, r, b LIMIT {edge_limit}"
        rows = _result_to_rows(conn.execute(cypher))  # type: ignore[attr-defined]
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Graph query error: {exc}") from exc

    nodes: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []

    for row in rows:
        vals = list(row.values())
        if len(vals) < 3:
            continue
        a_raw, r_raw, b_raw = vals[0], vals[1], vals[2]
        if not (_is_node(a_raw) and _is_rel(r_raw) and _is_node(b_raw)):
            continue

        src_id = _node_stable_id(a_raw)
        dst_id = _node_stable_id(b_raw)

        if src_id not in nodes and len(nodes) < max_nodes:
            nodes[src_id] = GraphNode(
                id=src_id,
                label=a_raw.get("_LABEL", "Node"),
                name=str(a_raw.get("name") or a_raw.get("path") or src_id),
                qname=a_raw.get("qname"),
                path=a_raw.get("path"),
            )
        if dst_id not in nodes and len(nodes) < max_nodes:
            nodes[dst_id] = GraphNode(
                id=dst_id,
                label=b_raw.get("_LABEL", "Node"),
                name=str(b_raw.get("name") or b_raw.get("path") or dst_id),
                qname=b_raw.get("qname"),
                path=b_raw.get("path"),
            )

        # Only include edges where both endpoints are in our node set.
        if src_id in nodes and dst_id in nodes:
            rel_type = str(r_raw.get("_TYPE", "RELATES_TO"))
            edges.append(GraphEdge(source=src_id, target=dst_id, type=rel_type))

    node_list = list(nodes.values())
    return GraphOverviewResponse(
        nodes=node_list,
        edges=edges,
        node_count=len(node_list),
        edge_count=len(edges),
    )
