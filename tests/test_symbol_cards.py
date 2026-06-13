"""Tests for symbol-card folding (methodology §5).

A {qn}::Symbol::card doc is a never-emitted, task-vocabulary retrieval
proxy. These pin the two load-bearing invariants: a card hit maps to its
PARENT symbol, and a card qname is never emitted from search or seeding.
"""
from __future__ import annotations

from app.routers.search import _fold_card_qname
from app.routers.context_bundle import _card_parent, _SYMBOL_CARD_MARKER


def test_fold_card_qname_maps_to_parent() -> None:
    assert _fold_card_qname("pkg.mod.requireRole::Symbol::card") == "pkg.mod.requireRole"
    # Non-card qnames pass through untouched.
    assert _fold_card_qname("pkg.mod.requireRole") == "pkg.mod.requireRole"
    # A summary qname is NOT a card (different mechanism — summaries are emitted).
    assert _fold_card_qname("pkg.mod.Foo::Class::summary") == "pkg.mod.Foo::Class::summary"


def test_card_parent_matches_search_fold() -> None:
    qn = "a.b.c.handler::Symbol::card"
    assert _card_parent(qn) == "a.b.c.handler"
    assert _card_parent("a.b.c.handler") == "a.b.c.handler"


def test_marker_is_distinct_from_summary() -> None:
    assert _SYMBOL_CARD_MARKER == "::Symbol::card"
    assert "::summary" not in _SYMBOL_CARD_MARKER
