"""PageRank centrality — pure-python core + LadybugDB wrapper.

Phase 1.5 of the optimization roadmap (TheForge ``docs/OPTIMIZATION_ROADMAP.md``
§1.5). The compute pipeline already shipped under Plan J in the sibling
``code-graph-rag`` package — this module is the locally-owned, testable
surface that:

- Pins the algorithm contract (alpha=0.85, max_iter=100, abort >100k nodes)
  to this repo's CI rather than to an editable cross-repo install.
- Splits compute (pure, graph-only) from normalisation (pure, list-only) so
  unit tests can hit either half without LadybugDB.
- Exposes ``compute_pagerank_for_repo(repo_db_path)`` as the LadybugDB-aware
  wrapper, reusing ``codebase_rag.storage.centrality.compute_pagerank`` when
  the sibling package is available so we have a single source of truth at
  runtime.

The TheForge ``mergeAndRank`` integration is intentionally deferred until
PR #224 (Phase 1.1 Tantivy) lands — that PR changes ``mergeAndRank``'s
signature, and a follow-up keeps both diffs clean.

This module **never raises** for callers in the indexing hot path. Callers
that want fail-loud semantics (tests) can use the pure helpers directly.
"""
from __future__ import annotations

import logging
from typing import Iterable

logger = logging.getLogger(__name__)

# Bounded compute — protect the indexer hot path from pathological graphs.
# Mirrors the brief's constraints: alpha=0.85, max_iter=100, hard ceiling at
# 100k nodes (warn + skip rather than spend minutes inside networkx).
_PR_ALPHA = 0.85
_PR_MAX_ITER = 100
_PR_NODE_CEILING = 100_000


def compute_pagerank(
    edges: Iterable[tuple[str, str]],
    nodes: Iterable[str] | None = None,
) -> dict[str, float]:
    """Compute raw PageRank over an in-memory CALLS graph.

    Pure: no I/O, no DB. Good for unit tests against a fixture graph.

    Args:
        edges: Iterable of ``(caller_qname, callee_qname)`` tuples. An empty
            iterable returns ``{}``.
        nodes: Optional explicit node set. Useful when a repo has callable
            symbols that participate in zero CALLS edges — they still get a
            (low) PageRank score so downstream fusion has a row for them.
            When ``None``, nodes are inferred from the edge endpoints.

    Returns:
        dict mapping ``qualified_name`` → raw PageRank score (sums to ~1.0
        across all nodes; not yet normalised to [0, 1]).

        Returns ``{}`` for an empty graph, a graph that exceeds the
        ``_PR_NODE_CEILING`` (logged at WARN), or a graph with no edges
        AND no explicit nodes.
    """
    edge_list = [(s, t) for s, t in edges if isinstance(s, str) and isinstance(t, str)]
    node_list: list[str] = []
    if nodes is not None:
        node_list = [n for n in nodes if isinstance(n, str)]

    if not edge_list and not node_list:
        return {}

    inferred = {n for e in edge_list for n in e}
    total_nodes = len(inferred | set(node_list))
    if total_nodes > _PR_NODE_CEILING:
        logger.warning(
            "pagerank.skipped reason=node_ceiling nodes=%d ceiling=%d",
            total_nodes,
            _PR_NODE_CEILING,
        )
        return {}

    # Defer the import so a missing networkx dep doesn't break import-time
    # callers that only want ``normalize_pagerank``.
    import networkx as nx

    g: nx.DiGraph = nx.DiGraph()
    if node_list:
        g.add_nodes_from(node_list)
    g.add_edges_from(edge_list)

    if g.number_of_nodes() == 0:
        return {}

    try:
        return nx.pagerank(g, alpha=_PR_ALPHA, max_iter=_PR_MAX_ITER)
    except nx.PowerIterationFailedConvergence as exc:
        # Best-effort: convergence failures still produce a usable score
        # via networkx's ``tol``-relaxed fallback. We log and return ``{}``
        # so the caller falls through to "no centrality" gracefully.
        logger.warning("pagerank.no_convergence nodes=%d err=%s", total_nodes, exc)
        return {}


def normalize_pagerank(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalise PageRank scores to [0.0, 1.0].

    Uses true min-max — floor maps to 0.0, ceiling to 1.0. This is a stricter
    contract than Plan J's divide-by-max (which has a non-zero floor when
    raw scores are uniform). The floor=0 guarantee matters for downstream
    fusion where the boost should be "no signal" for the least-central
    nodes, not a small constant.

    Args:
        scores: Raw PageRank scores keyed by qualified name. Empty input
            returns ``{}`` without exception.

    Returns:
        dict mapping ``qualified_name`` → score in [0.0, 1.0]. When all
        input scores are equal (rare — only for fully symmetric graphs),
        every output is ``1.0``.
    """
    if not scores:
        return {}

    values = list(scores.values())
    lo = min(values)
    hi = max(values)
    span = hi - lo

    if span == 0:
        # Degenerate input — every node has equal centrality. Returning all
        # zeros would mean "no signal" for every symbol; returning all ones
        # is safer for fusion (still neutral relative to siblings).
        return {qn: 1.0 for qn in scores}

    return {qn: (score - lo) / span for qn, score in scores.items()}


def compute_pagerank_for_repo(repo_db_path: str) -> dict[str, float]:
    """Compute normalised PageRank for a repo's LadybugDB graph.

    Thin wrapper that defers to the sibling ``code-graph-rag`` package's
    implementation when it is installed (the runtime path used by the
    indexer hot loop in ``app/routers/index.py``), then re-normalises with
    this module's stricter min-max so the contract on this layer is
    consistent regardless of which compute path produced the raw scores.

    Args:
        repo_db_path: Filesystem path to the per-repo ``.db`` LadybugDB file.

    Returns:
        dict mapping ``qualified_name`` → normalised PageRank in [0.0, 1.0].
        Empty dict on any failure — never raises. Callers in the indexing
        hot loop must treat absent centrality as a degraded-but-OK state.
    """
    try:
        from codebase_rag.storage.centrality import (  # type: ignore[import-untyped]
            compute_pagerank as _sibling_compute,
        )
    except ImportError:
        logger.warning("pagerank.no_sibling_pkg path=%s", repo_db_path)
        return {}

    try:
        raw = _sibling_compute(repo_db_path)
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("pagerank.sibling_failed path=%s err=%s", repo_db_path, exc)
        return {}

    if not raw:
        return {}
    return normalize_pagerank(raw)
