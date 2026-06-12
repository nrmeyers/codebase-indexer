"""Unit tests for the design-intent facet-coverage grader in scripts/grade_queries.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "grade_queries",
    Path(__file__).resolve().parents[1] / "scripts" / "grade_queries.py",
)
grade_queries = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(grade_queries)


def _record(facets, top_k):
    return {
        "id": "dsg-test",
        "intent": "design",
        "expected_facets": facets,
        "top_k": top_k,
    }


FACETS = [
    {"name": "lexical", "any": ["bm25_index", "tantivy"]},
    {"name": "semantic", "any": ["array_cosine_distance"]},
    {"name": "router", "any": ["search"]},
    {"name": "fusion", "any": ["rrf"]},
]


def test_full_coverage_passes() -> None:
    top_k = [
        {"symbol": "app.services.bm25_index.rrf_fuse", "snippet": "def rrf_fuse(): ..."},
        {"symbol": "app.routers.search.semantic", "snippet": "array_cosine_distance(...)"},
    ]
    g = grade_queries._grade(_record(FACETS, top_k))
    assert g["facet_coverage"] == 1.0
    assert g["design_pass"] is True
    assert g["facets_missed"] == []
    assert g["topk_relevant"] is True


def test_partial_coverage_below_threshold_fails() -> None:
    # Only 2/4 facets covered (0.5 < 0.7).
    top_k = [{"symbol": "app.routers.search", "snippet": "uses tantivy index"}]
    g = grade_queries._grade(_record(FACETS, top_k))
    assert g["facet_coverage"] == 0.5
    assert g["design_pass"] is False
    assert sorted(g["facets_missed"]) == ["fusion", "semantic"]
    assert g["top1_relevant"] is False


def test_partial_coverage_at_threshold_passes() -> None:
    # 3/4 = 0.75 >= 0.7.
    top_k = [
        {"symbol": "app.routers.search", "snippet": "tantivy + rrf fusion"},
    ]
    g = grade_queries._grade(_record(FACETS, top_k))
    assert g["facet_coverage"] == 0.75
    assert g["design_pass"] is True


def test_zero_coverage() -> None:
    top_k = [{"symbol": "app.main.lifespan", "snippet": "startup checks"}]
    g = grade_queries._grade(_record(FACETS, top_k))
    assert g["facet_coverage"] == 0.0
    assert g["design_pass"] is False
    assert len(g["facets_missed"]) == 4


def test_empty_facets_never_passes() -> None:
    g = grade_queries._grade(_record([], [{"symbol": "anything"}]))
    assert g["facet_coverage"] == 0.0
    assert g["design_pass"] is False


def test_empty_bundle_fails() -> None:
    g = grade_queries._grade(_record(FACETS, []))
    assert g["facet_coverage"] == 0.0
    assert g["design_pass"] is False


def test_matching_is_case_insensitive() -> None:
    facets = [{"name": "gate", "any": ["GateStateMachine"]}]
    top_k = [{"symbol": "theforge.gatestatemachine.transition", "snippet": ""}]
    g = grade_queries._grade(_record(facets, top_k))
    assert g["facet_coverage"] == 1.0


def test_non_design_intent_unaffected() -> None:
    g = grade_queries._grade(
        {
            "id": "sem-test",
            "intent": "semantic",
            "expected_topk_substrings": ["foo"],
            "top_k": [{"symbol": "foo.bar"}],
        }
    )
    assert g["top1_relevant"] is True
    assert "facet_coverage" not in g
