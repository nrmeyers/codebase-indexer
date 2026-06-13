"""Tests for the cross-encoder bundle reranker (app/services/bundle_reranker).

The reranker did NOT clear the benchmark gate (see
docs/reranker-bundle-tiebreak-spike.md) and ships flag-OFF. These tests
pin the two invariants that make it safe to carry dormant: it is
fail-open (no endpoint → no change), and the score-rescale keeps every
neighbour strictly below every seed.
"""
from __future__ import annotations

from unittest.mock import patch

from app.routers.context_bundle import _apply_neighbour_rerank
from app.services import bundle_reranker


def test_rerank_scores_fail_open_when_unavailable() -> None:
    """Endpoint down → None, so the caller keeps bi-encoder order."""
    with patch.object(bundle_reranker, "is_available", return_value=False):
        out = bundle_reranker.rerank_scores(
            "some query", [("app.mod.fn", "fn\nbody")]
        )
    assert out is None


def test_rerank_scores_none_on_empty_input() -> None:
    assert bundle_reranker.rerank_scores("", []) is None
    assert bundle_reranker.rerank_scores("q", []) is None


def test_build_doc_strips_summary_marker_and_caps() -> None:
    doc = bundle_reranker.build_doc("a.b.c.myFunc::Module::summary", "x" * 5000)
    assert doc.startswith("myFunc\n")
    # capped near the doc cap, not the full 5000 chars
    assert len(doc) <= bundle_reranker._DOC_CAP_CHARS + len("myFunc\n") + 1


def test_apply_neighbour_rerank_preserves_seed_precedence() -> None:
    """Reranked neighbours must stay strictly below the lowest seed, even
    when the reranker scores a neighbour 1.0."""
    scores = {
        "seed.low": 0.50,   # depth 0 — lowest seed
        "seed.high": 0.90,
        "nbr.a": 0.40,      # depth 1 neighbours
        "nbr.b": 0.40,
    }
    depth = {"seed.low": 0, "seed.high": 0, "nbr.a": 1, "nbr.b": 1}
    _apply_neighbour_rerank(scores, {"nbr.a": 1.0, "nbr.b": 0.0}, depth)
    seed_floor = min(scores["seed.low"], scores["seed.high"])
    assert scores["nbr.a"] < seed_floor   # even at r=1.0
    assert scores["nbr.b"] < scores["nbr.a"]  # relevance ordering applied
    assert scores["seed.low"] == 0.50     # seeds untouched


def test_apply_neighbour_rerank_noop_without_scores() -> None:
    scores = {"seed": 0.5, "nbr": 0.4}
    depth = {"seed": 0, "nbr": 1}
    _apply_neighbour_rerank(scores, {}, depth)
    assert scores == {"seed": 0.5, "nbr": 0.4}
