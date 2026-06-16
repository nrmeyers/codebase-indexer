"""GET /search/structural, /search/semantic, /search/symbol.

Three complementary search surfaces against LadybugDB:

* ``/search/structural`` — raw Cypher passthrough for graph traversals.
* ``/search/semantic``   — DuckDB ``array_cosine_distance`` similarity
  search over function/method embeddings stored in per-repo ``.duck``
  files (v5.3 §6.5 + §8.4).
* ``/search/symbol``     — exact-name lookup returning source + location.

Semantic search does NOT require the LadybugDB VECTOR extension — embeddings
live in the per-repo DuckDB file (``.duck``) alongside the structural
``.db`` file.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from .. import metrics as _metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query rewriter (descriptive natural-language → tighter token set)
# ---------------------------------------------------------------------------
#
# Bi-encoder embeddings score concept-matches against function source +
# docstrings. Function names use verbs (`createErrorEnvelope`, `validateBearer`),
# but descriptive queries phrase intent nominally ("error envelope construction",
# "JWT validation against AAD JWKS"). The English filler words around the
# content tokens drag the query embedding toward generic prose, costing top-K
# precision. Stripping them is a 30-line, zero-LLM lift on descriptive queries.
#
# Empirically (TheForge/docs/CODE_INDEXER_EVAL_RESULTS.md Iter 5):
#
#   raw 25-query benchmark    descriptive P@1 = 50%
#   rewriter active           descriptive P@1 = 60%   (+10pp, no model cost)
#
# Behaviour:
#   - 4+ token queries: drop stop-words, send the rest verbatim
#   - <4 token queries: pass through (too short to safely strip)
#   - Dotted / snake_case tokens (module.path.fn, setup_test_env): pass
#     through (explicit symbol-name signal)
#   - CamelCase common nouns (WebSocket, MSAL, JavaScript): NOT a
#     short-circuit — these are English words bi-encoder handles fine,
#     and dropping noise around them still helps
_QUERY_STOP_WORDS = frozenset({
    # Articles
    "a", "an", "the", "this", "that", "these", "those",
    # Prepositions
    "in", "on", "at", "to", "from", "of", "for", "with", "without",
    "into", "onto", "upon", "via", "by", "as", "about", "against",
    "between", "across", "through", "during", "before", "after",
    "above", "below",
    # Conjunctions
    "and", "or", "but", "nor", "so", "yet",
    # Aux verbs / question shaping
    "is", "are", "was", "were", "be", "been", "being", "do", "does",
    "did", "has", "have", "had", "can", "could", "should", "would",
    "will", "shall", "may", "might", "must",
    "how", "what", "where", "when", "why", "which", "who", "whom",
    # Pronouns
    "i", "me", "my", "you", "your", "we", "us", "our", "it", "its",
    "they", "them", "their",
    # Generic search verbs
    "show", "find", "list", "get",
    # Code-search nominalisations — verb-form names (`create…`, `register…`)
    # are what we want to match against. Stripping these forces remaining
    # tokens onto the verb-form via embedding-model compositional structure.
    "construction", "configuration", "implementation", "registration",
    "initialization", "initialisation", "destruction", "creation",
})


def _rewrite_descriptive_query(raw: str) -> tuple[str, str]:
    """Return ``(rewritten_query, outcome_label)``.

    ``outcome_label`` ∈ {"applied", "skip-short", "skip-symbol-like",
    "skip-overstrip"} — used by the Prometheus metric so we can A/B the
    rewriter's hit-rate in production without changing call-site code.

    The rewriter never raises; it falls through to the original on every
    edge case. See module docstring for behavioural details.
    """
    text = raw.rstrip("?").strip()
    tokens = text.split()
    if len(tokens) < 4:
        return raw, "skip-short"
    # Dotted / snake_case tokens are an explicit symbol-name signal —
    # don't second-guess the caller's already-tight query.
    for t in tokens:
        if any(ch in t for ch in "._-"):
            return raw, "skip-symbol-like"
    kept = [t for t in tokens if t.lower() not in _QUERY_STOP_WORDS]
    if len(kept) < 2:
        return raw, "skip-overstrip"
    return " ".join(kept), "applied"

from ..config import settings, slugify_repo
from ..models import (
    CentralityResponse,
    CentralityResult,
    FileEntry,
    FileListResponse,
    GraphEdge,
    GraphNode,
    GraphOverviewResponse,
    LexicalHit,
    LexicalSearchResponse,
    NodeTypeStat,
    NodeTypesResponse,
    SemanticResult,
    SemanticSearchResponse,
    StructuralSearchResponse,
    SymbolResponse,
)
from ..services.tantivy_index import TantivyIndex

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
# Semantic search — lazy import cache
# ---------------------------------------------------------------------------
# embed_query lives in codebase_rag and requires torch/transformers.  Cache
# the import result so subsequent calls avoid re-importing a 400 MB library.
_embed_fn: Any = None            # cached embed_query callable
_embed_unavailable: bool = False  # True once import fails; never retried


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
    from ..services.ladybug_pool import open_read_conn

    db_path = _resolve_db_path(repo)
    # BUC-1571: read-only mode so /search/* never contends with the
    # exclusive write-lock held by an active /index job.
    _db, conn = open_read_conn(db_path)
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
    """Return True iff ``v`` is a LadybugDB node dict.

    Both nodes and relationships carry ``_LABEL``, so we must additionally
    require the absence of ``_SRC`` (which only rels have) to avoid
    classifying relationships as nodes.
    """
    return isinstance(v, dict) and "_LABEL" in v and "_SRC" not in v


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


# Pattern for the explicit ``... AS <alias>`` projection columns that the KG
# generator (and every well-formed paged caller) uses. We deliberately only
# recognise EXPLICIT aliases: an aliased projection is guaranteed to be a
# scalar column (string / number / label()) that the engine can ORDER BY, so
# injecting ``ORDER BY`` over them is always legal Cypher. Bare projections
# (e.g. ``RETURN n``) bind a node/relationship variable that is NOT orderable
# in kuzu, so we skip the injection for those and preserve the legacy
# (unordered) behaviour rather than risk a parser error.
_RETURN_ALIAS_RE = re.compile(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)


def _extract_return_aliases(cypher: str) -> list[str]:
    """Return the explicit ``AS <alias>`` names from a Cypher RETURN clause.

    Used to synthesise a deterministic ``ORDER BY`` for service-applied paging
    so a paged complete-scan over an otherwise-unordered query visits every
    matching row exactly once across repeated calls.

    We scan only the text AFTER the final top-level ``RETURN`` keyword so an
    alias bound earlier in a ``WITH ... AS x`` projection is not mistaken for a
    final-projection column.

    Args:
        cypher: The (literal-stripped or raw) Cypher query text.

    Returns:
        Ordered, de-duplicated list of alias identifiers. Empty when the query
        has no explicit aliases (e.g. ``RETURN n``) — the caller then leaves
        ordering untouched.
    """
    # Anchor on the LAST RETURN so a WITH-clause alias doesn't leak in.
    return_matches = list(re.finditer(r"\bRETURN\b", cypher, re.IGNORECASE))
    if not return_matches:
        return []
    tail = cypher[return_matches[-1].end():]
    seen: set[str] = set()
    aliases: list[str] = []
    for m in _RETURN_ALIAS_RE.finditer(tail):
        alias = m.group(1)
        # ``ASC``/``ASCENDING``/``AS`` keyword false-positives can't occur here
        # because the regex requires ``AS`` followed by whitespace + identifier;
        # but guard against a stray ORDER-BY direction token just in case.
        if alias.upper() in {"ASC", "DESC", "ASCENDING", "DESCENDING"}:
            continue
        if alias not in seen:
            seen.add(alias)
            aliases.append(alias)
    return aliases


# ---------------------------------------------------------------------------
# GET /search/structural
# ---------------------------------------------------------------------------


@router.get("/structural", response_model=StructuralSearchResponse)
def structural_search(
    q: str = Query(description="Cypher query to execute against the graph"),
    limit: int = Query(default=500, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
    repo: str | None = Query(
        default=None,
        description="Repo slug to scope the query to. Omit for first indexed DB.",
    ),
) -> StructuralSearchResponse:
    """Execute a raw Cypher query and return matching nodes and relationships.

    Args:
        q: Arbitrary Cypher query. If the query does not already contain a
            LIMIT clause, a bounded ``SKIP {offset} LIMIT {limit}`` is appended
            so large-graph navigation can page through results without
            shipping the whole graph in one response.
        limit: Maximum rows to return (1–5000). Only applied if ``q`` does
            not already include a LIMIT clause. Raised from the historical
            cap of 500 to support large-repo graph navigation (LE-169a).
        offset: Skip N matching rows before returning results (cursor paging).
            Only applied if ``q`` does not already include a LIMIT clause —
            clients that hand-write their own SKIP/LIMIT keep full control.
            The engine-side fetch stays bounded at ``offset + limit`` (≤ 5000).

            LE-181b — DETERMINISTIC COMPLETE-SCAN PAGING. When the service owns
            paging (no caller LIMIT) and the query has explicit ``... AS alias``
            RETURN columns but no ``ORDER BY``, a deterministic ``ORDER BY`` over
            the full alias tuple is injected before SKIP/LIMIT. kuzu's
            multi-label node pattern ``(n:A|B|C)`` is a UNION SCAN whose
            cross-table order is NOT stable across executions; without an
            ``ORDER BY`` a paged complete scan (offset += page_len) can skip and
            double-count rows, leaving a handful of first-party files (web/*,
            src/services/routes/*, src/adapters/identity/*) intermittently
            absent from a consumer's complete-scan aggregate. Ordering on the
            full projected tuple makes pages disjoint and the union complete and
            byte-identical across repeated identical requests, WITHOUT any
            caller-side ``ORDER BY`` (which front-loads the multiplied tests.*
            rows and truncates the alphabetically-last layer). Bare projections
            (``RETURN n``) and caller-supplied ``ORDER BY`` are left untouched.

    Returns:
        StructuralSearchResponse: Nodes, relationships, row count, plus
            additive paging metadata — ``offset`` and ``limit`` echo the
            effective page window, and ``has_more`` is True when at least one
            matching row exists beyond this page (so consumers can page the
            full graph without a separate count query). ``has_more`` is only
            meaningful for service-applied paging; when the caller hand-writes
            their own LIMIT it is always False.

    Raises:
        HTTPException: 422 when the Cypher query is malformed.
    """
    _t0 = time.monotonic()
    _status_code = 200
    try:
        return _structural_search_impl(q=q, limit=limit, offset=offset, repo=repo)
    except HTTPException as _e:
        _status_code = _e.status_code
        raise
    finally:
        _metrics.record_search("structural", reranked=False, duration_seconds=time.monotonic() - _t0, status_code=_status_code)


def _structural_search_impl(
    *,
    q: str,
    limit: int,
    offset: int = 0,
    repo: str | None,
) -> StructuralSearchResponse:
    """Inner implementation for structural_search (extracted for metrics wrapping)."""
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
    #
    # When the caller hasn't written their own LIMIT, append a bounded
    # SKIP/LIMIT so the engine never fetches an unbounded result set. The
    # engine fetches at most ``offset + limit`` rows (≤ 5000 — both params are
    # Query-capped) then skips ``offset``, so at most ``limit`` rows reach the
    # caller. Clients that hand-write their own LIMIT keep full pagination
    # control (the historical contract — see the docstring) and bypass this.
    # ``has_more`` is only meaningful for service-applied paging (the caller
    # didn't hand-write a LIMIT). Default values cover the caller-supplied case.
    service_paged = not _re.search(r"\bLIMIT\b", cypher, _re.IGNORECASE)
    safe_limit = max(1, min(int(limit), 5000))
    safe_offset = max(0, min(int(offset), 5000))
    # Number of rows actually requested from the engine. When the service owns
    # paging we fetch ONE extra row (``limit + 1``, still hard-capped at 5000):
    # if the engine returns more than ``safe_limit`` rows we know another page
    # exists. This avoids a second COUNT query over an arbitrary Cypher graph
    # while keeping the engine-side fetch bounded.
    fetch_limit = safe_limit
    if service_paged:
        # LE-181b — deterministic complete-scan paging.
        #
        # ROOT CAUSE: kuzu's multi-label node pattern `(n:A|B|C)` is executed
        # as a UNION SCAN across the per-label tables. The scan order across
        # those tables is an incidental engine-internal detail, NOT stable
        # across separate query executions. When the service appends a bare
        # `SKIP/LIMIT` to such a query with NO `ORDER BY`, consecutive pages
        # are cut out of an unstable ordering, so a paged complete-scan
        # (offset += page_len) can skip rows that shifted between pages and
        # double-count others — non-deterministic, incomplete results.
        # Consumers (TheForge's KG generator) saw a residual handful of
        # first-party files (web/* frontend, src/services/routes/*,
        # src/adapters/identity/*) intermittently get zero nodes.
        #
        # The generator can NOT fix this caller-side: its symbol projection
        # multiplies each distinct symbol into ~4 rows and duplicates the
        # `tests.*` subtree so heavily that a caller-side `ORDER BY qname`
        # front-loads tens of thousands of test rows and pushes the
        # alphabetically-last layer (`web.*`) past any sane truncation budget,
        # dropping the whole frontend.
        #
        # FIX: inject a deterministic `ORDER BY` over the query's explicit
        # RETURN aliases (the full projected tuple) BEFORE the SKIP/LIMIT,
        # but ONLY when the caller did not already specify an ORDER BY. This
        # makes the paged scan a stable, gapless, duplicate-free total order:
        # every matching row is visited exactly once across pages, regardless
        # of the engine's incidental union-scan order, and the result is
        # byte-identical across repeated identical requests. It does NOT
        # truncate any layer — full-scan paging walks the COMPLETE result set
        # (the layer-starvation only ever arose from caller-side ORDER BY +
        # truncation, which this replaces). Row multiplicity is harmless under
        # a complete scan: dedup happens downstream in the consumer.
        #
        # We order over the full alias tuple (not just the first column) so
        # the order is TOTAL even when many rows share a leading column value
        # (the multiplied rows), eliminating ties that the engine could break
        # differently between calls.
        has_order_by = bool(re.search(r"\bORDER\s+BY\b", cypher, re.IGNORECASE))
        if not has_order_by:
            aliases = _extract_return_aliases(scan_target)
            if aliases:
                order_cols = ", ".join(aliases)
                cypher = f"{cypher}\nORDER BY {order_cols}"
        # Defensive clamp: the Query() validators already bound these, but the
        # impl is also called directly (tests / internal callers) where the
        # bounds aren't enforced. Never let an unbounded fetch through.
        fetch_limit = min(safe_limit + 1, 5000)
        if safe_offset:
            cypher = f"{cypher}\nSKIP {safe_offset} LIMIT {fetch_limit}"
        else:
            cypher = f"{cypher}\nLIMIT {fetch_limit}"

    try:
        conn = _get_conn(repo)
        rows = _result_to_rows(conn.execute(cypher))  # type: ignore[attr-defined]
    except HTTPException:
        raise  # e.g. 404 from _resolve_db_path — preserve status code
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cypher error: {exc}") from exc

    # Derive paging metadata, then trim the probe row so the caller still sees
    # exactly ``safe_limit`` rows (the +1 was only a "is there more?" sentinel).
    has_more = False
    if service_paged:
        has_more = len(rows) > safe_limit
        if has_more:
            rows = rows[:safe_limit]

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
        offset=safe_offset if service_paged else 0,
        limit=safe_limit,
        has_more=has_more,
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
    rerank: bool = Query(
        default=False,
        description=(
            "When true, widen the bi-encoder fetch and rerank with "
            "CodeRankLLM via LM Studio (two-stage retrieval). Silently "
            "no-ops when LM Studio is unavailable — the bi-encoder order "
            "is returned unchanged so callers see no behaviour change "
            "beyond a small fetch-time cost."
        ),
    ),
) -> SemanticSearchResponse:
    """Find the top-k most semantically similar functions/methods.

    Args:
        q: Natural-language description (e.g. "function that retries HTTP
            requests with exponential backoff"). Embedded with CodeRankEmbed and
            compared against the per-repo DuckDB vector store.
        k: Number of results to return (1–100).
        rerank: Opt-in two-stage retrieval. Stage 1 is the standard
            DuckDB ``array_cosine_distance`` bi-encoder; stage 2 runs the
            top-50 through ``nomic-ai/CodeRankLLM`` (listwise generative
            reranker) via LM Studio. Best-effort — falls back to the
            bi-encoder ordering when LM Studio isn't running.

    Returns:
        SemanticSearchResponse: Ranked list of qualified names with scores.

    Raises:
        HTTPException: 503 when torch/transformers are unavailable (first
            import failed; fast-fail thereafter), or when no .duck file
            exists for the requested repo.
    """
    _t0 = time.monotonic()
    _status_code = 200
    try:
        return _semantic_search_impl(q=q, k=k, repo=repo, rerank=rerank)
    except HTTPException as _e:
        _status_code = _e.status_code
        raise
    finally:
        _metrics.record_search(
            "semantic",
            reranked=bool(rerank),
            duration_seconds=time.monotonic() - _t0,
            status_code=_status_code,
        )


from ..services.symbol_cards import (
    SYMBOL_CARD_MARKER as _SYMBOL_CARD_MARKER,
    fold_card_qname as _fold_card_qname,
)


def _semantic_search_impl(
    *,
    q: str,
    k: int,
    repo: str | None,
    rerank: bool,
) -> SemanticSearchResponse:
    """Inner implementation for semantic_search (extracted for metrics wrapping)."""
    import re as _re

    global _embed_fn, _embed_unavailable  # noqa: PLW0603

    # Provider priority: configured embedder backend (prod = SageMaker) →
    # LM Studio (dev) → in-process torch.
    from ..embedders.sync_bridge import (  # noqa: PLC0415
        embed_text_sync,
        get_embedder_or_none,
    )
    from ..services import lm_studio          # local import keeps cold-start cheap

    def _embed_query(text: str) -> list[float]:
        # Configured embedder primary. ``role="query"`` lets the local
        # backend prepend the model's query prefix (e5 ``query: ``,
        # CodeRankEmbed instruction) — symmetric with the ``document`` prefix
        # the index pass applies; see app.embedders.prefixes. No-op for prod
        # backends and symmetric models. ``embed_text_sync`` returns the
        # *full* 768-dim vector (it unwraps the single-element async batch
        # internally), so no ``[0]`` indexing here — regression guard from
        # BUC-1570, where ``[0]`` sliced one float out of the 768-dim vector
        # and crashed downstream in ``_l2_normalise``.
        vec = embed_text_sync(text, role="query")
        if vec:
            return vec

        # LM Studio dev fallback — uses asymmetric "search_query: " prefix.
        if vec := lm_studio.embed(text, prefix="search_query: "):
            return vec

        # In-process torch last resort.  ``_embed_fn`` may have been loaded
        # already (when both ``_sm_available`` and ``_lm_available`` were
        # False at route entry), or it may be None because ``_sm_available``
        # was True (configured backend constructed but its model failed to
        # load at embed time — e.g. EMBEDDER_BACKEND=local without the
        # sentence-transformers missing).  Attempt a lazy load here before giving up
        # so that a partially-available configured backend doesn't
        # permanently suppress the torch fallback.
        global _embed_fn, _embed_unavailable  # noqa: PLW0603
        if _embed_fn is None and not _embed_unavailable:
            try:
                from codebase_rag.embedder import embed_query as _eq  # type: ignore[import-untyped]
                _embed_fn = _eq
            except ImportError:
                _embed_unavailable = True

        if _embed_fn is not None:
            return _embed_fn(text)

        # All embedding providers failed or are unavailable.  Surface a
        # clear, actionable 503 rather than a bare RuntimeError so callers
        # see "503 Semantic search unavailable" instead of a generic 500.
        # Common cause: EMBEDDER_BACKEND=local without sentence-transformers
        # extra installed.  Install with:
        #   uv sync
        raise HTTPException(
            status_code=503,
            detail=(
                "Semantic search unavailable: no embedding provider succeeded. "
                "For local installs run: uv sync "
                "(installs sentence-transformers for EMBEDDER_BACKEND=local). "
                "Check server logs for the exact initialisation error."
            ),
        )

    _sm_available = get_embedder_or_none() is not None
    _lm_available = lm_studio.can_embed()

    if _embed_unavailable and not _sm_available and not _lm_available:
        raise HTTPException(
            status_code=503,
            detail="Semantic search unavailable (missing deps; import failed on first attempt)",
        )

    if _embed_fn is None and not _sm_available and not _lm_available:
        try:
            from codebase_rag.embedder import embed_query  # type: ignore[import-untyped]
            _embed_fn = embed_query
        except ImportError as exc:
            _embed_unavailable = True
            raise HTTPException(
                status_code=503,
                detail=f"Semantic search unavailable (missing deps): {exc}",
            ) from exc

    # Resolve the .duck path for the requested repo.
    if repo:
        vec_path = settings.vec_db_path_for_repo(repo)
    else:
        # No repo specified — find the first .duck on disk.
        db_dir = Path(settings.LADYBUG_DB_DIR)
        vec_path = ""
        if db_dir.is_dir():
            for f in sorted(db_dir.glob("*.duck")):
                vec_path = str(f)
                break
        if not vec_path:
            raise HTTPException(
                status_code=404,
                detail="No embedding store found. Run POST /index first.",
            )

    if not Path(vec_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"No embedding store found for repo '{repo}'. Run POST /index first.",
        )

    try:
        from codebase_rag.storage.vector_store import open_or_create, search_similar  # type: ignore[import-untyped]
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"DuckDB vector store unavailable: {exc}",
        ) from exc

    # Over-fetch to push past degenerate anonymous/fixture embeddings.
    # When rerank=true, we additionally guarantee at least 50 post-noise
    # candidates reach the reranker (Nomic's eval sweet spot for
    # CodeRankLLM — beyond ~30 the listwise prompt grows linearly and
    # accuracy plateaus).
    fetch_k = max(k * 50, 500)
    if rerank:
        fetch_k = max(fetch_k, 500)  # guarantee post-noise headroom for top-50 rerank

    _FIXTURE_SEGMENTS = {"fixtures", "large-file", "__fixtures__"}
    _ANON_RE = _re.compile(r"^anonymous_\d+_\d+$")

    def _is_noise(sym: str) -> bool:
        parts = sym.split(".")
        if any(seg in _FIXTURE_SEGMENTS for seg in parts):
            return True
        if any(_ANON_RE.match(seg) for seg in parts):
            return True
        if len(parts) >= 2 and parts[-1] == parts[-2]:
            return True
        return False

    _BARE_FQN_RE = _re.compile(r"^[\w][\w.]*[\w]$")

    # Single vec_conn spans both cosine search and PageRank centrality read —
    # opening one DuckDB connection per query (was three) is cheaper and avoids
    # races when the .duck is being concurrently written by an indexer job.
    _pr_scores: dict[str, float] = {}
    search_intent: str = "semantic"
    try:
        # Strip natural-language scaffolding from descriptive queries
        # before embedding. No-op on symbol-like or short queries —
        # outcome label feeds the rewriter A/B observability counter so
        # we can measure live hit-rate from /metrics.
        embed_query_text, _rewrite_outcome = _rewrite_descriptive_query(q)
        _metrics.record_query_rewriter("semantic", _rewrite_outcome)
        query_embedding = _embed_query(embed_query_text)
        vec_conn = open_or_create(vec_path)
        try:
            raw = search_similar(vec_conn, query_embedding, k=fetch_k)

            filtered = [r for r in raw if not _is_noise(r.qualified_name)]

            # Fold {qn}::Symbol::card proxies into their parent symbol BEFORE
            # any downstream stage (FQN pinning, PageRank fusion, RRF/BM25
            # fusion, rerank) so a single canonical row per symbol carries the
            # max(parent_cosine, card_cosine) score forward and the card qname
            # never reaches consumers.
            _by_parent: dict[str, Any] = {}
            for _r in filtered:
                _pqn = _fold_card_qname(_r.qualified_name)
                _prev = _by_parent.get(_pqn)
                if _prev is None or _r.score > _prev.score:
                    _r.qualified_name = _pqn
                    _by_parent[_pqn] = _r
            filtered = sorted(_by_parent.values(), key=lambda r: r.score, reverse=True)

            # Intent routing: if query looks like a bare qualified name (e.g.
            # "myapp.utils.retry"), pin exact / prefix matches to the top.
            _q_stripped = q.strip()
            if _BARE_FQN_RE.match(_q_stripped) and "." in _q_stripped:
                _exact = [r for r in filtered if r.qualified_name == _q_stripped
                          or r.qualified_name.endswith("." + _q_stripped)
                          or r.qualified_name.startswith(_q_stripped + ".")]
                _rest  = [r for r in filtered if r not in _exact]
                filtered = _exact + _rest
                search_intent = "fqn"

            # --- Plan J: PageRank fusion (read-side) ---
            # Reuse the cosine connection for the centrality read. Best-effort:
            # any failure here is swallowed so PageRank cannot break search.
            try:
                from codebase_rag.storage.vector_store import read_centrality  # type: ignore[import-untyped]
                _pr_scores = read_centrality(
                    vec_conn, [r.qualified_name for r in filtered]
                )
            except Exception:
                _pr_scores = {}
        finally:
            vec_conn.close()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Semantic search failed: {exc}",
        ) from exc

    # --- Plan J: PageRank fusion (apply scores) ---
    # final = 0.7 * cosine + 0.3 * normalised_pagerank. Outside the connection
    # block — only needs the scores dict.
    if _pr_scores:
        for r in filtered:
            pr = _pr_scores.get(r.qualified_name, 0.0)
            r.score = 0.7 * r.score + 0.3 * pr
        filtered.sort(key=lambda r: r.score, reverse=True)

    # --- Plan E: Reciprocal Rank Fusion with BM25 lexical retrieval ---
    # Runs AFTER FQN intent pinning and AFTER PageRank fusion so exact-symbol
    # matches stay anchored at the top while RRF blends semantic ranks
    # (post-PageRank) with BM25 lexical ranks across the rest of the pool.
    # K_RRF=60 is the canonical RRF constant from Cormack et al. (2009) —
    # empirically robust across query types and the value used by Vespa,
    # Elasticsearch's RRF retriever, and most published RAG fusion baselines.
    try:
        from ..services.bm25_index import bm25_service
        bm25_results = bm25_service.search(vec_path, q, k=max(fetch_k, 100))
        if bm25_results:
            K_RRF = 60
            fused: dict[str, float] = {}

            # Semantic ranks — `filtered` is already in (post-PageRank) order.
            for rank, r in enumerate(filtered, start=1):
                fused[r.qualified_name] = (
                    fused.get(r.qualified_name, 0.0) + 1.0 / (K_RRF + rank)
                )

            # BM25 ranks.
            for rank, (qn, _score) in enumerate(bm25_results, start=1):
                fused[qn] = fused.get(qn, 0.0) + 1.0 / (K_RRF + rank)

            # Reorder `filtered` by fused score; only items already in the
            # semantic candidate set are surfaced (BM25-only hits are absorbed
            # via tie-breaking on shared symbols, not introduced as new rows).
            order = sorted(
                range(len(filtered)),
                key=lambda i: fused.get(filtered[i].qualified_name, 0.0),
                reverse=True,
            )
            filtered = [filtered[i] for i in order]
    except Exception:
        # Best-effort fusion: never fail a search because BM25 misbehaved.
        pass

    # --- Stage 2: optional listwise rerank via CodeRankLLM (disabled by default) ---
    # When RERANK_ENABLED=true, runs AFTER all stage-1 fusion (PageRank + RRF/BM25)
    # so the bi-encoder's best-fused order is what the LLM rescores. We hand the
    # reranker the top-50 candidates *enriched with their source snippets* (joined
    # from LadybugDB Module nodes), get back its permutation, and slice to k.
    # Best-effort: any failure (reranker offline, parse error, timeout, source-fetch
    # error) leaves ``filtered`` untouched. LM Studio rerank is a local-only opt-in
    # via ``?rerank=true``; hosted deploys (where LM Studio is unreachable) gracefully
    # degrade to the un-reranked bi-encoder order with a structured warning logged
    # by ``reranker.rerank()`` (BUC-1651). Future backends may slot in alongside;
    # see BUC-1545 for the cross-encoder / TEI sidecar exploration.
    if settings.RERANK_ENABLED and rerank:
        try:
            from ..services import reranker, source_fetch  # noqa: WPS433 — runtime-optional

            head = filtered[: 50]
            tail = filtered[50:]

            # Resolve source snippets for the head candidates so the LLM
            # ranks against actual code body, not just identifier names.
            # Empirically (Nomic CodeRankLLM eval, Qwen3 internal):
            # snippet-grounded rerank beats FQN-only by ~12-20 nDCG@10
            # points on code-search benchmarks.  Best-effort: any DB
            # failure leaves snippet="" and the LLM falls back to
            # ranking on FQN alone.
            snippets: dict[str, str] = {}
            try:
                snip_conn = _get_conn(repo)
                try:
                    snippets = source_fetch.fetch_sources_for_symbols(
                        snip_conn, [r.qualified_name for r in head]
                    )
                finally:
                    snip_conn.close()  # type: ignore[attr-defined]
            except Exception:
                snippets = {}

            cand_dicts = [
                {
                    "qualified_name": r.qualified_name,
                    "score": r.score,
                    # Reranker reads "source" first, then "snippet" —
                    # passing both lets us swap parsers later without
                    # breaking the candidate schema.
                    "source": snippets.get(r.qualified_name, ""),
                }
                for r in head
            ]
            reordered = reranker.rerank(q, cand_dicts)
            if reordered and len(reordered) == len(cand_dicts):
                # Map back to the original SemanticResult-shaped objects
                # so we don't lose score/type metadata.
                by_qn = {r.qualified_name: r for r in head}
                filtered = [by_qn[c["qualified_name"]] for c in reordered if c["qualified_name"] in by_qn] + tail
        except Exception:
            # Reranker failures must never break search — keep stage-1 order.
            pass
    # search_intent surfaces the internal routing label ("fqn" when a bare
    # qualified-name was detected and pinned, "semantic" otherwise) so HTTP
    # callers can see why a particular ranking was produced without
    # reverse-engineering the query string.
    _results = [
        SemanticResult(symbol=r.qualified_name, score=round(r.score, 4), type="")
        for r in filtered[:k]
    ]
    return SemanticSearchResponse(results=_results, search_intent=search_intent)


# ---------------------------------------------------------------------------
# GET /search/centrality — top-N PageRank scores (BACKEND_HANDOVER §2.8)
# ---------------------------------------------------------------------------


def _resolve_vec_path(repo: str | None) -> str:
    """Pick the ``.duck`` file (vector + centrality store) for ``repo``.

    Mirrors ``_resolve_db_path`` but for the DuckDB store. Used by
    ``/search/centrality`` and any future endpoint that reads the
    ``centrality`` or ``embeddings`` tables directly.

    Args:
        repo: Optional repo slug.

    Returns:
        str: Filesystem path to the ``.duck`` file. Empty string when no
        ``.duck`` files are present anywhere — callers should treat empty
        as "no embeddings yet" and return an empty result rather than 404.
    """
    if repo:
        return settings.vec_db_path_for_repo(repo)
    db_dir = Path(settings.LADYBUG_DB_DIR)
    if db_dir.is_dir():
        ducks = sorted(db_dir.glob("*.duck"))
        if ducks:
            return str(ducks[0])
    return ""


def _enrich_centrality_locations(
    repo: str | None, qnames: list[str]
) -> dict[str, tuple[str, int, int]]:
    """Look up ``(file_path, start_line, end_line)`` for each FQN.

    Best-effort: any LadybugDB failure returns an empty dict so centrality
    rows still ship without location metadata. The frontend renders ``—``
    for missing line ranges.

    Args:
        repo: Repo slug for DB resolution.
        qnames: Symbol qualified names to look up.

    Returns:
        dict[str, tuple[str, int, int]]: ``qualified_name`` →
        ``(file_path, start_line, end_line)``. Missing keys default to
        ``("", 0, 0)`` at the call site.
    """
    if not qnames:
        return {}
    try:
        # Same UNION ALL pattern as /symbols/{fqn}/callers — Functions reach
        # their Module via DEFINES, Methods via Class -[:DEFINES_METHOD].
        cypher = """
        MATCH (m:Module)-[:DEFINES]->(n:Function)
        WHERE n.qualified_name IN $qnames
        RETURN n.qualified_name AS qn, m.path AS path,
               n.start_line AS start_line, n.end_line AS end_line
        UNION ALL
        MATCH (m:Module)-[:DEFINES]->(:Class)-[:DEFINES_METHOD]->(n:Method)
        WHERE n.qualified_name IN $qnames
        RETURN n.qualified_name AS qn, m.path AS path,
               n.start_line AS start_line, n.end_line AS end_line
        """
        conn = _get_conn(repo)
        rows = _result_to_rows(conn.execute(cypher, {"qnames": qnames}))  # type: ignore[attr-defined]
    except Exception:
        return {}

    out: dict[str, tuple[str, int, int]] = {}
    for r in rows:
        qn = r.get("qn")
        if not qn or qn in out:
            continue
        sl = r.get("start_line")
        el = r.get("end_line")
        out[qn] = (
            r.get("path") or "",
            int(sl) if isinstance(sl, (int, float)) else 0,
            int(el) if isinstance(el, (int, float)) else 0,
        )
    return out


@router.get("/centrality", response_model=CentralityResponse)
def centrality_top_n(
    limit: int = Query(default=10, ge=1, le=500),
    repo: str | None = Query(
        default=None,
        description="Repo slug to scope the query to. Omit for first indexed DB.",
    ),
) -> CentralityResponse:
    """Return the ``limit`` most-central symbols by PageRank score.

    Reads from the per-repo ``.duck`` ``centrality`` table (populated
    post-index by Plan J — see ``codebase_rag/services/pagerank.py``).
    File paths and line ranges are looked up via LadybugDB best-effort —
    a LadybugDB failure produces a degraded but still-useful response
    (qualified_name + score, with empty ``file_path`` and ``line_range``).

    Args:
        limit: Max number of rows to return (1–500; default 10).
              Raised from 200 → 500 (NAVI-93) — ``_enrich_centrality_locations``
              is O(limit) (one batch KùZu IN query) so this is safe.
        repo: Optional repo slug.

    Returns:
        CentralityResponse: Empty list when the centrality table is empty
        (PageRank not yet computed for this repo); the FE has copy for
        that case.
    """
    vec_path = _resolve_vec_path(repo)
    if not vec_path or not Path(vec_path).exists():
        return CentralityResponse(results=[])

    try:
        from codebase_rag.storage.vector_store import open_or_create  # type: ignore[import-untyped]
    except ImportError:
        return CentralityResponse(results=[])

    rows: list[tuple[str, float]] = []
    try:
        conn = open_or_create(vec_path)
        try:
            res = conn.execute(
                "SELECT qualified_name, pagerank FROM centrality "
                "ORDER BY pagerank DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
            rows = [(r[0], float(r[1])) for r in res]
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("Centrality read failed for %s: %s", repo, exc)
        return CentralityResponse(results=[])

    if not rows:
        return CentralityResponse(results=[])

    locations = _enrich_centrality_locations(repo, [qn for qn, _ in rows])

    return CentralityResponse(
        results=[
            CentralityResult(
                qualified_name=qn,
                pagerank=score,
                file_path=locations.get(qn, ("", 0, 0))[0],
                line_range=(
                    locations.get(qn, ("", 0, 0))[1],
                    locations.get(qn, ("", 0, 0))[2],
                ),
            )
            for qn, score in rows
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
    _t0 = time.monotonic()
    _status_code = 200
    try:
        return _symbol_lookup_impl(fqn=fqn, repo=repo)
    except HTTPException as _e:
        _status_code = _e.status_code
        raise
    finally:
        _metrics.record_search("symbol", reranked=False, duration_seconds=time.monotonic() - _t0, status_code=_status_code)


def _symbol_lookup_impl(*, fqn: str, repo: str | None) -> SymbolResponse:
    """Inner implementation for symbol_lookup (extracted for metrics wrapping)."""
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
    docstring: str | None = row.get("docstring") or None

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
        docstring=docstring,
    )


# ---------------------------------------------------------------------------
# GET /search/lexical — Tantivy BM25 (Phase 1.1)
# ---------------------------------------------------------------------------


def _open_tantivy_for_repo(repo: str | None) -> TantivyIndex | None:
    """Open the Tantivy index for ``repo``; return ``None`` if unavailable.

    The index lives at ``<LADYBUG_DB_DIR>/<slug>.tantivy/``.  ``None``
    return paths cause callers to skip the lexical arm — never to error,
    so an un-migrated repo degrades gracefully to the existing search
    surfaces.
    """
    if not repo:
        return None
    try:
        idx = TantivyIndex(settings.LADYBUG_DB_DIR, slugify_repo(repo))
        if idx._unavailable:  # type: ignore[attr-defined]
            return None
        return idx
    except Exception:
        return None


@router.get("/lexical", response_model=LexicalSearchResponse)
def lexical_search(
    q: str = Query(description="Free-text query; ranked by Tantivy BM25"),
    repo: str | None = Query(
        default=None,
        description="Repo slug to scope the search to. Required for cross-repo isolation.",
    ),
    limit: int = Query(default=20, ge=1, le=200),
) -> LexicalSearchResponse:
    """BM25 lexical search over the Tantivy index — Phase 1.1.

    Returns ranked hits with ``score`` ordered descending.  Catches what
    the dense semantic arm misses: rare identifiers, exact substrings,
    error codes, file-path fragments.

    Best-effort: a missing / unavailable index returns ``results: []``
    rather than 503 so the orchestrator's hybrid retrieval merge stays
    resilient when one repo hasn't been migrated yet.

    Args:
        q: Free-text query; tokenised and matched against ``content``,
            ``symbol_qname``, and ``file_path`` Tantivy fields.
        repo: Required for safe multi-tenant isolation — without it, a
            query against repo A could match repo B's documents (every
            repo's index is on disk under the same data dir).
        limit: Max ranked results to return (1–200; default 20).

    Returns:
        LexicalSearchResponse: Empty list when no index, no query tokens,
        or no matches — never raises.
    """
    _t0 = time.monotonic()
    _status_code = 200
    try:
        if not q.strip():
            return LexicalSearchResponse(results=[])

        idx = _open_tantivy_for_repo(repo)
        if idx is None:
            return LexicalSearchResponse(results=[])

        try:
            hits = idx.search(q, k=limit, repo=slugify_repo(repo) if repo else None)
        finally:
            idx.close()

        return LexicalSearchResponse(
            results=[
                LexicalHit(
                    symbol_qname=h["symbol_qname"],
                    file_path=h["file_path"],
                    symbol_kind=h["symbol_kind"],
                    score=h["score"],
                    start_line=h["start_line"],
                    end_line=h["end_line"],
                )
                for h in hits
            ]
        )
    except Exception as exc:
        # Best-effort: never fail the lexical surface — return empty.
        logger.debug("lexical_search swallowed error: %s", exc)
        return LexicalSearchResponse(results=[])
    finally:
        _metrics.record_search(
            "lexical",
            reranked=False,
            duration_seconds=time.monotonic() - _t0,
            status_code=_status_code,
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
    # Phase 1.1 — when a non-empty filter is supplied, try the Tantivy
    # ``file_path`` field first.  Tantivy returns BM25-ranked path hits
    # that are dramatically more accurate than the case-insensitive
    # ``CONTAINS`` substring scan when the filter is multi-token (e.g.
    # ``github-app-client``).  We fall back to the existing Cypher
    # behaviour on any failure (no index, query parse error, etc.) so the
    # contract is preserved bit-for-bit.
    needle = filter.strip()
    if needle and repo:
        try:
            idx = _open_tantivy_for_repo(repo)
            if idx is not None:
                try:
                    hits = idx.search(
                        needle,
                        k=min(int(limit) + int(offset), 5000),
                        repo=slugify_repo(repo),
                    )
                finally:
                    idx.close()
                if hits:
                    # Optional extension post-filter (mirrors Cypher path).
                    ext_norm = extension.strip().lower()
                    if ext_norm and not ext_norm.startswith("."):
                        ext_norm = "." + ext_norm
                    seen: set[str] = set()
                    files_acc: list[FileEntry] = []
                    for h in hits:
                        fp = h["file_path"]
                        if not fp or fp in seen:
                            continue
                        if ext_norm and not fp.lower().endswith(ext_norm):
                            continue
                        seen.add(fp)
                        base = fp.rsplit("/", 1)[-1]
                        ext_part = ""
                        if "." in base:
                            ext_part = "." + base.rsplit(".", 1)[-1]
                        files_acc.append(
                            FileEntry(path=fp, name=base, extension=ext_part)
                        )
                    if files_acc:
                        total = len(files_acc)
                        page = files_acc[int(offset) : int(offset) + int(limit)]
                        return FileListResponse(files=page, total=total)
        except Exception:
            # Non-fatal — fall through to the Cypher implementation.
            pass

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
    """Derive a stable string ID for a raw LadybugDB node dict.

    Prefers ``qualified_name`` (unique across Functions/Methods/Classes),
    falls back to ``path`` (unique for Files/Folders), and finally ``name``.
    """
    return str(
        v.get("qualified_name")
        or v.get("path")
        or v.get("name")
        or id(v)
    )


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
    rel_types: str = Query(
        default="CALLS,IMPORTS,DEFINES_METHOD",
        description=(
            "Comma-separated relationship labels to include. Defaults to the "
            "semantic subset (CALLS, IMPORTS, DEFINES_METHOD); excludes "
            "structural containment edges (CONTAINS_FILE, CONTAINS_FOLDER, "
            "DEFINES) which dominate a raw graph query but carry no "
            "code-navigation signal."
        ),
    ),
) -> GraphOverviewResponse:
    """Return a compact graph (nodes + edges) for repo-wide canvas rendering.

    Pulls relationships whose label matches ``rel_types`` (default: semantic
    code-navigation edges), derives stable node IDs from each node's
    ``qname`` or ``path`` property, and caps output at ``max_nodes``.
    Nodes with no relationships in the requested set are not included —
    overview emphasises connectivity rather than exhaustive enumeration.

    Args:
        repo: Repo slug to scope to.
        max_nodes: Cap on unique nodes included in the response.
        rel_types: Comma-separated relationship labels to include.

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

    # Parse rel_types — accept both list-style and comma-separated input.
    wanted_types: set[str] = {
        t.strip().upper() for t in rel_types.split(",") if t.strip()
    }
    if not wanted_types:
        # Empty filter degenerates into "all rels" rather than "none".
        wanted_types = set()

    # Fetch relationship triples.  We pull more than max_nodes worth so we
    # can include nodes that appear only as targets. When a rel-type filter is
    # active, push it into Cypher so we don't pay the transport cost for
    # rels we'll discard anyway.
    edge_limit = max_nodes * 5
    if wanted_types:
        rel_pattern = "|".join(sorted(wanted_types))
        cypher = (
            f"MATCH (a)-[r:{rel_pattern}]->(b) "
            f"RETURN a, r, b LIMIT {edge_limit}"
        )
    else:
        cypher = f"MATCH (a)-[r]->(b) RETURN a, r, b LIMIT {edge_limit}"
    try:
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
                qname=a_raw.get("qualified_name"),
                path=a_raw.get("path"),
            )
        if dst_id not in nodes and len(nodes) < max_nodes:
            nodes[dst_id] = GraphNode(
                id=dst_id,
                label=b_raw.get("_LABEL", "Node"),
                name=str(b_raw.get("name") or b_raw.get("path") or dst_id),
                qname=b_raw.get("qualified_name"),
                path=b_raw.get("path"),
            )

        # Only include edges where both endpoints are in our node set.
        if src_id in nodes and dst_id in nodes:
            # LadybugDB stores relationship type under ``_LABEL`` (same key
            # as nodes) — not ``_TYPE``. Use ``_LABEL`` with a conservative
            # fallback for rows where it's unexpectedly missing.
            rel_type = str(r_raw.get("_LABEL", "RELATES_TO"))
            edges.append(GraphEdge(source=src_id, target=dst_id, type=rel_type))

    node_list = list(nodes.values())
    return GraphOverviewResponse(
        nodes=node_list,
        edges=edges,
        node_count=len(node_list),
        edge_count=len(edges),
    )
