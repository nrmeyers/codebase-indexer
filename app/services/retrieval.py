"""Semantic retrieval pipeline shared by HTTP routers.

The full bi-encoder + PageRank + RRF/BM25 + optional listwise rerank flow
that powers ``GET /search/semantic`` lives here so multiple call sites
(``app/routers/search.py`` HTTP handler, ``app/routers/context_bundle.py``
seed builder) depend on a service rather than each other. Extracted from
``app/routers/search.py::_semantic_search_impl`` — pure refactor, no
behavioural change.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from .. import metrics as _metrics
from ..config import settings
from ..models import SemanticResult
from ..services.symbol_cards import fold_card_qname

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
_QUERY_STOP_WORDS: frozenset[str] = frozenset({
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


@dataclass(frozen=True)
class SemanticSearchResult:
    """Internal return shape for ``semantic_search``.

    Mirrors the fields of ``app.models.SemanticSearchResponse`` without
    coupling the service to the HTTP Pydantic layer — callers wrap as
    appropriate.
    """

    results: list[SemanticResult]
    search_intent: str


def semantic_search(
    *,
    q: str,
    k: int,
    repo: str | None,
    rerank: bool,
) -> SemanticSearchResult:
    """Run the full semantic retrieval pipeline for ``q``.

    Pipeline stages (each independently best-effort except embedding):

    1. Descriptive-query rewrite (strip filler words on long queries).
    2. Embed query via the configured backend (``role="query"``).
    3. Bi-encoder ANN search against the per-repo DuckDB vector store.
    4. Noise filter (fixtures / anonymous / self-recursive names).
    5. Symbol-card fold so card proxies collapse into their parent qname.
    6. FQN intent pin (bare ``a.b.c`` queries surface exact/prefix matches).
    7. PageRank fusion (0.7 cosine + 0.3 normalised pagerank).
    8. RRF/BM25 fusion across the candidate pool.
    9. Optional listwise rerank via the CodeRankLLM reranker (top-50).

    Args:
        q: Raw query string from the caller.
        k: Desired result count (1..N). The pipeline over-fetches
            ``max(k * 50, 500)`` candidates internally before slicing.
        repo: Repo slug. ``None`` falls back to the first ``.duck`` on disk.
        rerank: When True, runs stage 9 if ``settings.RERANK_ENABLED``.

    Returns:
        SemanticSearchResult: ranked results plus an internal
            ``search_intent`` label (``"fqn"`` or ``"semantic"``).

    Raises:
        HTTPException: 404 when no ``.duck`` exists for the requested repo,
            503 when the embedder is unavailable or the vector store
            module cannot be imported.
    """
    from ..embedders.sync_bridge import embed_text_sync

    def _embed_query(text: str) -> list[float]:
        # Configured embedder is the sole provider. ``role="query"`` lets
        # the local backend prepend the model's query prefix (e5
        # ``query: ``, CodeRankEmbed instruction, nomic ``search_query: ``)
        # — symmetric with the ``document`` prefix the index pass applies;
        # see app.embedders.prefixes. No-op for prod backends and symmetric
        # models. ``embed_text_sync`` returns the *full* 768-dim vector (it
        # unwraps the single-element async batch internally), so no ``[0]``
        # indexing here — regression guard from BUC-1570, where ``[0]``
        # sliced one float out of the 768-dim vector and crashed downstream
        # in ``_l2_normalise``.
        vec = embed_text_sync(text, role="query")
        if vec:
            return vec
        # Never fall back to a different vector space — a foreign embedding
        # query against the DuckDB index returns garbage at best and
        # silently-wrong rankings at worst.
        raise HTTPException(
            status_code=503,
            detail=(
                "Semantic search unavailable: configured embedder backend "
                "failed. For local installs run: uv sync "
                "(installs sentence-transformers for EMBEDDER_BACKEND=local). "
                "Check server logs for the exact initialisation error."
            ),
        )

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
    _ANON_RE = re.compile(r"^anonymous_\d+_\d+$")

    def _is_noise(sym: str) -> bool:
        parts = sym.split(".")
        if any(seg in _FIXTURE_SEGMENTS for seg in parts):
            return True
        if any(_ANON_RE.match(seg) for seg in parts):
            return True
        if len(parts) >= 2 and parts[-1] == parts[-2]:
            return True
        return False

    _BARE_FQN_RE = re.compile(r"^[\w][\w.]*[\w]$")

    # Single vec_conn spans both cosine search and PageRank centrality read —
    # opening one DuckDB connection per query (was three) is cheaper and avoids
    # races when the .duck is being concurrently written by an indexer job.
    _pr_scores: dict[str, float] = {}
    search_intent: str = "semantic"
    filtered: list[Any]
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
                _pqn = fold_card_qname(_r.qualified_name)
                _prev = _by_parent.get(_pqn)
                if _prev is None or _r.score > _prev.score:
                    _r.qualified_name = _pqn
                    _by_parent[_pqn] = _r
            filtered = sorted(_by_parent.values(), key=lambda r: r.score, reverse=True)

            # Intent routing: if the query looks like a bare qualified name (e.g.
            # "myapp.utils.retry"), pin exact / prefix matches to the top.
            _q_stripped = q.strip()
            if _BARE_FQN_RE.match(_q_stripped) and "." in _q_stripped:
                _exact = [r for r in filtered if r.qualified_name == _q_stripped
                          or r.qualified_name.endswith("." + _q_stripped)
                          or r.qualified_name.startswith(_q_stripped + ".")]
                _rest = [r for r in filtered if r not in _exact]
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
            from ..services import reranker, source_fetch
            from ..routers.search import _get_conn

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
    results = [
        SemanticResult(symbol=r.qualified_name, score=round(r.score, 4), type="")
        for r in filtered[:k]
    ]
    return SemanticSearchResult(results=results, search_intent=search_intent)
