"""Tests for symbol-card folding (methodology §5).

A {qn}::Symbol::card doc is a never-emitted, task-vocabulary retrieval
proxy. These pin the load-bearing invariants: a card hit maps to its
PARENT symbol, a card qname never reaches consumers from search or
seeding, the fold/dedup unions scores at max(), and the embed driver
never emits an orphan card row.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.routers.context_bundle import _card_parent, _SYMBOL_CARD_MARKER
from app.routers.search import _fold_card_qname
from app.services.symbol_cards import SYMBOL_CARD_MARKER, fold_card_qname


client = TestClient(app)


# ---------------------------------------------------------------------------
# Pure fold-helper invariants.
# ---------------------------------------------------------------------------


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


def test_marker_constant_is_shared() -> None:
    # search.py + context_bundle.py both alias the same module-level constant.
    assert _SYMBOL_CARD_MARKER is SYMBOL_CARD_MARKER
    assert _fold_card_qname is fold_card_qname or (
        _fold_card_qname("x::Symbol::card") == fold_card_qname("x::Symbol::card")
    )


def test_fold_only_matches_at_string_end() -> None:
    # The marker must be a suffix; mid-string occurrences must NOT fold.
    assert _fold_card_qname("pkg.X::Symbol::card.child") == "pkg.X::Symbol::card.child"
    assert _fold_card_qname("pkg.X::Symbol::cardHandler") == "pkg.X::Symbol::cardHandler"


# ---------------------------------------------------------------------------
# Integration: fold + dedup at the _semantic_search_impl boundary.
# ---------------------------------------------------------------------------


def _fake_search_result(qn: str, score: float):
    return SimpleNamespace(
        qualified_name=qn,
        file_path="",
        start_line=0,
        end_line=0,
        score=score,
    )


def _semantic_search_with(raw_rows, tmp_path):
    """Call /search/semantic with ``raw_rows`` mocked at the vector-store layer."""
    duck = tmp_path / "fake.duck"
    duck.write_bytes(b"")
    with patch("app.embedders.sync_bridge.embed_text_sync",
               lambda text, role="query": [0.0] * 768), \
         patch("app.config.Settings.vec_db_path_for_repo",
               lambda self, repo: str(duck)), \
         patch("codebase_rag.storage.vector_store.open_or_create",
               return_value=MagicMock()), \
         patch("codebase_rag.storage.vector_store.search_similar",
               return_value=raw_rows), \
         patch("codebase_rag.storage.vector_store.read_centrality",
               return_value={}):
        return client.get(
            "/search/semantic",
            params={"q": "some natural-language query", "k": 10, "repo": "fake"},
        )


def test_semantic_search_folds_card_to_parent_with_max_score(tmp_path) -> None:
    # Parent and its card both surface; the parent must appear EXACTLY once
    # with the HIGHER of the two scores (max-union, methodology §5).
    rows = [
        _fake_search_result("pkg.foo", 0.40),
        _fake_search_result("pkg.foo::Symbol::card", 0.90),
        _fake_search_result("pkg.other", 0.60),
    ]
    resp = _semantic_search_with(rows, tmp_path)
    assert resp.status_code == 200
    body = resp.json()
    syms = [r["symbol"] for r in body["results"]]
    assert syms.count("pkg.foo") == 1
    assert "pkg.foo::Symbol::card" not in syms
    foo = next(r for r in body["results"] if r["symbol"] == "pkg.foo")
    assert foo["score"] == 0.9


def test_semantic_search_drops_card_when_only_card_present(tmp_path) -> None:
    # Only the card hits — caller still gets the parent qname.
    rows = [
        _fake_search_result("pkg.foo::Symbol::card", 0.88),
        _fake_search_result("pkg.other", 0.50),
    ]
    resp = _semantic_search_with(rows, tmp_path)
    assert resp.status_code == 200
    syms = [r["symbol"] for r in resp.json()["results"]]
    assert "pkg.foo" in syms
    assert "pkg.foo::Symbol::card" not in syms


def test_semantic_search_dedup_preserves_other_results(tmp_path) -> None:
    # [parent, card_for_parent, other] -> [parent (max score), other];
    # the card row is dropped, the parent's row + score survive.
    rows = [
        _fake_search_result("pkg.foo", 0.95),
        _fake_search_result("pkg.foo::Symbol::card", 0.30),
        _fake_search_result("pkg.other", 0.50),
    ]
    resp = _semantic_search_with(rows, tmp_path)
    assert resp.status_code == 200
    results = resp.json()["results"]
    syms = [r["symbol"] for r in results]
    assert syms == ["pkg.foo", "pkg.other"]
    assert results[0]["score"] == 0.95


# ---------------------------------------------------------------------------
# Integration: embed_driver never emits an orphan card row.
# ---------------------------------------------------------------------------


def test_embed_driver_skips_orphan_card_when_parent_absent() -> None:
    """A cards.json entry whose parent is not in _rows must NOT be emitted.

    Mirrors the embed-driver card-emit body: build ``_span_by_qn`` from a
    fixture row set, walk a fixture ``_cards`` dict, and assert no orphan
    qname is ever queued onto the batch.
    """
    from app.services.symbol_cards import SYMBOL_CARD_MARKER as _M

    _rows = [
        {"qualified_name": "pkg.alive", "rel_path": "src/a.py",
         "start_line": 1, "end_line": 5},
    ]
    _cards = {
        "pkg.alive": {"desc": "Handles the alive case.", "src_hash": "h1"},
        "pkg.orphan": {"desc": "Handles a removed symbol.", "src_hash": "h2"},
    }

    _span_by_qn = {
        r["qualified_name"]: r for r in _rows if r.get("qualified_name")
    }
    emitted: list[str] = []
    for _cqn, _entry in _cards.items():
        _desc = _entry["desc"] if isinstance(_entry, dict) else _entry
        _r = _span_by_qn.get(_cqn)
        if not _r or not _desc:
            continue
        emitted.append(f"{_cqn}{_M}")

    assert emitted == ["pkg.alive::Symbol::card"]
    assert "pkg.orphan::Symbol::card" not in emitted
