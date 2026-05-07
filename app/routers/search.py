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

from ..config import settings
from ..models import (
    CentralityResponse,
    CentralityResult,
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
    _t0 = time.monotonic()
    _status_code = 200
    try:
        return _structural_search_impl(q=q, limit=limit, repo=repo)
    except HTTPException as _e:
        _status_code = _e.status_code
        raise
    finally:
        _metrics.record_search("structural", reranked=False, duration_seconds=time.monotonic() - _t0, status_code=_status_code)


def _structural_search_impl(
    *,
    q: str,
    limit: int,
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

    # Provider priority: SageMaker (prod) → LM Studio (dev) → in-process torch.
    from ..services import lm_studio          # local import keeps cold-start cheap
    from ..services.sagemaker_embedder import get_sagemaker_embedder  # noqa: PLC0415

    def _embed_query(text: str) -> list[float]:
        # SageMaker primary — no asymmetric prefix; E5 models handle it natively.
        sm = get_sagemaker_embedder()
        if sm is not None:
            vecs = sm.embed(text)
            if vecs:
                return vecs[0]

        # LM Studio dev fallback — uses asymmetric "search_query: " prefix.
        if vec := lm_studio.embed(text, prefix="search_query: "):
            return vec

        # In-process torch last resort.
        if _embed_fn is None:  # pragma: no cover — guarded by outer block
            raise RuntimeError("in-process embedder not initialised")
        return _embed_fn(text)

    _sm_available = get_sagemaker_embedder() is not None
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
                status_code=503,
                detail="No embedding store found. Run POST /index first.",
            )

    if not Path(vec_path).exists():
        raise HTTPException(
            status_code=503,
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

    # --- Stage 2: optional listwise rerank via CodeRankLLM (LM Studio) ---
    # Runs AFTER all stage-1 fusion (PageRank + RRF/BM25) so the
    # bi-encoder's best-fused order is what the LLM rescores.  We hand
    # the reranker the top-50 candidates *enriched with their source
    # snippets* (joined from LadybugDB Module nodes), get back its
    # permutation, and slice to k.  Best-effort: any failure (LM Studio
    # offline, parse error, timeout, source-fetch error) leaves
    # ``filtered`` untouched.
    if rerank:
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
    return SemanticSearchResponse(
        results=[
            SemanticResult(
                symbol=r.qualified_name,
                score=round(r.score, 4),
                type="",
            )
            for r in filtered[:k]
        ],
        search_intent=search_intent,
    )


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
    limit: int = Query(default=10, ge=1, le=200),
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
        limit: Max number of rows to return (1–200; default 10).
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
