"""Tests for GET /search/structural, /search/semantic, /search/symbol."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# /search/structural
# ---------------------------------------------------------------------------


def _mock_conn(rows: list[dict]) -> MagicMock:
    """Build a mock LadybugDB connection that returns ``rows``."""
    col_names = list(rows[0].keys()) if rows else []

    result = MagicMock()
    result.get_column_names.return_value = col_names
    remaining = list(rows)

    def has_next():
        return bool(remaining)

    def get_next():
        row = remaining.pop(0)
        return list(row.values())

    result.has_next.side_effect = has_next
    result.get_next.side_effect = get_next

    conn = MagicMock()
    conn.execute.return_value = result
    return conn


# ---------------------------------------------------------------------------
# _rewrite_descriptive_query — descriptive→tighter rewrite + outcome label
# ---------------------------------------------------------------------------


def test_rewrite_descriptive_query_strips_filler_words() -> None:
    from app.routers.search import _rewrite_descriptive_query as rw
    # 4+ tokens, no symbol-name signal → stop-words drop, outcome=applied.
    assert rw("how do AAD groups map to Forge roles") == (
        "AAD groups map Forge roles", "applied",
    )
    assert rw("JWT validation against AAD JWKS") == (
        "JWT validation AAD JWKS", "applied",
    )
    # CamelCase common nouns (WebSocket / MSAL) do NOT short-circuit.
    assert rw("WebSocket reconnect with fresh token") == (
        "WebSocket reconnect fresh token", "applied",
    )


def test_rewrite_descriptive_query_outcome_short_token_count() -> None:
    from app.routers.search import _rewrite_descriptive_query as rw
    assert rw("createIdentityProvider") == ("createIdentityProvider", "skip-short")
    assert rw("rate limit") == ("rate limit", "skip-short")
    assert rw("error envelope construction") == (
        "error envelope construction", "skip-short",
    )


def test_rewrite_descriptive_query_outcome_dotted_or_snake() -> None:
    from app.routers.search import _rewrite_descriptive_query as rw
    # ≥4 tokens with a dotted FQN present — preserve verbatim, the
    # dotted token is an explicit symbol-name signal that overrides
    # rewriting.
    assert rw("look up module.path.fnName in current scope") == (
        "look up module.path.fnName in current scope", "skip-symbol-like",
    )
    # snake_case symbol — same skip path.
    assert rw("set up setup_test_env helper for new tests") == (
        "set up setup_test_env helper for new tests", "skip-symbol-like",
    )
    # Hyphenated identifier-shape token — same skip path.
    assert rw("verify aad-provider integration with JWKS cache") == (
        "verify aad-provider integration with JWKS cache", "skip-symbol-like",
    )


def test_rewrite_descriptive_query_outcome_overstrip_falls_back() -> None:
    from app.routers.search import _rewrite_descriptive_query as rw
    # 4+ tokens, no symbol-shape, but everything is a stop-word →
    # rewriter would leave 0-1 tokens; safer to return original.
    rewritten, outcome = rw("the of a an")
    assert rewritten == "the of a an"
    assert outcome in {"skip-short", "skip-overstrip"}


def test_rewrite_descriptive_query_outcomes_are_exclusive() -> None:
    from app.routers.search import _rewrite_descriptive_query as rw
    # Every outcome must be one of the four documented labels — Prometheus
    # counter cardinality stays bounded at 4 × intent count.
    valid = {"applied", "skip-short", "skip-symbol-like", "skip-overstrip"}
    for q in [
        "createIdentityProvider",                       # short
        "module.path.fn name with extra words",         # symbol-like
        "how do AAD groups map to Forge roles",         # applied
        "the of a an",                                  # short
    ]:
        _, outcome = rw(q)
        assert outcome in valid, f"unknown outcome {outcome!r} for {q!r}"


def test_structural_search_returns_rows() -> None:
    rows = [{"name": "foo", "qualified_name": "mymod.foo"}]
    with patch("app.routers.search._get_conn", return_value=_mock_conn(rows)):
        resp = client.get("/search/structural", params={"q": "MATCH (n:Function) RETURN n.name AS name, n.qualified_name AS qualified_name"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["row_count"] == 1
    assert body["nodes"][0]["name"] == "foo"


def test_structural_search_cypher_error() -> None:
    conn = MagicMock()
    conn.execute.side_effect = RuntimeError("syntax error")
    with patch("app.routers.search._get_conn", return_value=conn):
        resp = client.get("/search/structural", params={"q": "BAD CYPHER"})
    assert resp.status_code == 422
    assert "Cypher error" in resp.json()["detail"]


def test_structural_search_appends_limit() -> None:
    conn = MagicMock()
    result = MagicMock()
    result.get_column_names.return_value = []
    result.has_next.return_value = False
    conn.execute.return_value = result

    with patch("app.routers.search._get_conn", return_value=conn):
        client.get("/search/structural", params={"q": "MATCH (n) RETURN n"})

    executed_query: str = conn.execute.call_args[0][0]
    assert "LIMIT" in executed_query.upper()


# ---------------------------------------------------------------------------
# /search/semantic
# ---------------------------------------------------------------------------


def _fake_search_result(qn: str, score: float):
    """Lightweight stand-in for ``codebase_rag.storage.vector_store.SearchResult``.

    The router only reads ``.qualified_name`` and ``.score`` (it mutates
    ``.score`` during PageRank fusion), so a SimpleNamespace works without
    importing the real dataclass.
    """
    from types import SimpleNamespace
    return SimpleNamespace(
        qualified_name=qn,
        file_path="",
        start_line=0,
        end_line=0,
        score=score,
    )


def test_semantic_search_returns_results(tmp_path) -> None:
    # The semantic endpoint pipeline:  embed_query -> open_or_create(.duck)
    # -> search_similar -> read_centrality -> response.  Mock all four,
    # plus point the .duck path at a real (but empty) file so the
    # Path.exists() guard passes.
    duck = tmp_path / "fake.duck"
    duck.write_bytes(b"")  # exists() must return True; contents unused

    fake_results = [
        _fake_search_result("mymod.foo", 0.95),
        _fake_search_result("mymod.bar", 0.80),
    ]

    with patch("app.routers.search._embed_fn", lambda q: [0.0] * 768), \
         patch("app.routers.search._embed_unavailable", False), \
         patch("app.config.Settings.vec_db_path_for_repo",
               lambda self, repo: str(duck)), \
         patch("codebase_rag.storage.vector_store.open_or_create",
               return_value=MagicMock()), \
         patch("codebase_rag.storage.vector_store.search_similar",
               return_value=fake_results), \
         patch("codebase_rag.storage.vector_store.read_centrality",
               return_value={}):
        resp = client.get(
            "/search/semantic",
            params={"q": "find all functions", "k": 5, "repo": "fake"},
        )
    assert resp.status_code == 200
    body = resp.json()
    results = body["results"]
    assert len(results) == 2
    assert results[0]["symbol"] == "mymod.foo"
    assert results[0]["score"] == pytest.approx(0.95)
    # search_intent surfaces the internal routing label — a natural-language
    # query ("find all functions") routes through the default semantic path.
    assert "search_intent" in body
    assert body["search_intent"] == "semantic"


def test_semantic_search_surfaces_fqn_intent(tmp_path) -> None:
    """A bare-qualified-name query (e.g. ``mymod.foo``) must trigger the
    FQN-pinning branch and surface ``search_intent="fqn"`` in the response.
    """
    duck = tmp_path / "fake.duck"
    duck.write_bytes(b"")

    fake_results = [
        _fake_search_result("mymod.foo", 0.50),
        _fake_search_result("other.bar", 0.95),
    ]

    with patch("app.routers.search._embed_fn", lambda q: [0.0] * 768), \
         patch("app.routers.search._embed_unavailable", False), \
         patch("app.config.Settings.vec_db_path_for_repo",
               lambda self, repo: str(duck)), \
         patch("codebase_rag.storage.vector_store.open_or_create",
               return_value=MagicMock()), \
         patch("codebase_rag.storage.vector_store.search_similar",
               return_value=fake_results), \
         patch("codebase_rag.storage.vector_store.read_centrality",
               return_value={}):
        resp = client.get(
            "/search/semantic",
            params={"q": "mymod.foo", "k": 5, "repo": "fake"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["search_intent"] == "fqn"
    # FQN-pinning must hoist the exact match to position 0 even though its
    # raw cosine score (0.50) is lower than the unrelated hit (0.95).
    assert body["results"][0]["symbol"] == "mymod.foo"


def test_semantic_search_empty(tmp_path) -> None:
    duck = tmp_path / "fake.duck"
    duck.write_bytes(b"")

    with patch("app.routers.search._embed_fn", lambda q: [0.0] * 768), \
         patch("app.routers.search._embed_unavailable", False), \
         patch("app.config.Settings.vec_db_path_for_repo",
               lambda self, repo: str(duck)), \
         patch("codebase_rag.storage.vector_store.open_or_create",
               return_value=MagicMock()), \
         patch("codebase_rag.storage.vector_store.search_similar",
               return_value=[]), \
         patch("codebase_rag.storage.vector_store.read_centrality",
               return_value={}):
        resp = client.get(
            "/search/semantic", params={"q": "nothing", "repo": "fake"}
        )
    assert resp.status_code == 200
    assert resp.json()["results"] == []


# ---------------------------------------------------------------------------
# /search/symbol
# ---------------------------------------------------------------------------


def test_symbol_lookup_found(tmp_path) -> None:
    src_file = tmp_path / "mymod.py"
    src_file.write_text("def foo():\n    pass\n")

    rows = [{"qualified_name": "mymod.foo", "start_line": 1, "end_line": 2, "path": str(src_file)}]
    with patch("app.routers.search._get_conn", return_value=_mock_conn(rows)):
        resp = client.get("/search/symbol", params={"fqn": "mymod.foo"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["qualified_name"] == "mymod.foo"
    assert "def foo" in body["source"]


def test_symbol_lookup_not_found() -> None:
    conn = MagicMock()
    result = MagicMock()
    result.get_column_names.return_value = ["qualified_name", "start_line", "end_line", "path"]
    result.has_next.return_value = False
    conn.execute.return_value = result

    with patch("app.routers.search._get_conn", return_value=conn):
        resp = client.get("/search/symbol", params={"fqn": "does.not.exist"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /search/semantic — graceful degradation when ML deps unavailable
# ---------------------------------------------------------------------------


def test_semantic_search_503_when_import_fails() -> None:
    """When codebase_rag.embedder cannot be imported (e.g. torch missing),
    the endpoint must return 503 with a valid JSON body — not 500.

    Strategy: set sys.modules entry to None which makes Python raise
    ImportError on ``from codebase_rag.embedder import embed_query``.
    """
    import sys
    import app.routers.search as _search_mod

    # Reset cached state so the lazy-load branch is taken.
    original_fn = _search_mod._embed_fn
    original_unavail = _search_mod._embed_unavailable
    _search_mod._embed_fn = None
    _search_mod._embed_unavailable = False

    # Save the real module so we can restore it after the test.
    real_mod = sys.modules.get("codebase_rag.embedder")

    try:
        # Setting sys.modules[name] = None causes `from name import …` to
        # raise ImportError — this simulates a missing ML dependency.
        sys.modules["codebase_rag.embedder"] = None  # type: ignore[assignment]
        # Force every provider unavailable so the fallback chain reaches
        # the in-process torch path that the import is faking out.
        # BUC-1605: the default backend is now ``local`` (no env vars
        # required), so an unmocked ``get_embedder_or_none`` would return
        # a valid backend and bypass the 503 path under test.
        with patch("app.embedders.sync_bridge.get_embedder_or_none",
                   return_value=None), \
             patch("app.embedders.sync_bridge.embed_text_sync",
                   return_value=None), \
             patch("app.services.lm_studio.can_embed",
                   return_value=False), \
             patch("app.services.lm_studio.embed",
                   return_value=None):
            resp = client.get("/search/semantic", params={"q": "retry http"})
    finally:
        # Restore everything regardless of outcome.
        _search_mod._embed_fn = original_fn
        _search_mod._embed_unavailable = original_unavail
        if real_mod is not None:
            sys.modules["codebase_rag.embedder"] = real_mod
        else:
            sys.modules.pop("codebase_rag.embedder", None)

    assert resp.status_code == 503
    body = resp.json()
    # Response must be valid JSON with a 'detail' key (FastAPI HTTPException shape)
    assert "detail" in body
    assert isinstance(body["detail"], str)


def test_semantic_search_503_uses_fast_fail_after_first_import_failure() -> None:
    """Once the import fails, _embed_unavailable=True and subsequent calls
    skip the import attempt and return 503 immediately."""
    import app.routers.search as _search_mod

    original_fn = _search_mod._embed_fn
    original_unavail = _search_mod._embed_unavailable

    # Simulate a prior import failure having set the flag
    _search_mod._embed_fn = None
    _search_mod._embed_unavailable = True

    try:
        # BUC-1605: force the embedder + LM Studio paths unavailable so the
        # fast-fail branch can be reached. The default ``local`` backend
        # would otherwise satisfy ``_sm_available`` and bypass this branch.
        with patch("app.embedders.sync_bridge.get_embedder_or_none",
                   return_value=None), \
             patch("app.embedders.sync_bridge.embed_text_sync",
                   return_value=None), \
             patch("app.services.lm_studio.can_embed",
                   return_value=False), \
             patch("app.services.lm_studio.embed",
                   return_value=None):
            resp = client.get("/search/semantic", params={"q": "anything"})
    finally:
        _search_mod._embed_fn = original_fn
        _search_mod._embed_unavailable = original_unavail

    assert resp.status_code == 503
    body = resp.json()
    assert "detail" in body
    assert "unavailable" in body["detail"].lower()


# ---------------------------------------------------------------------------
# /search/semantic — regression: SageMaker embedder must not be double-indexed
# (BUC-1570: ``'float' object is not iterable`` when the response of
# ``SageMakerEmbedder.embed()`` was treated as a *list of vectors* and
# indexed with ``[0]``, sliced one float out, and passed downstream where
# ``_l2_normalise`` choked on a scalar.)
# ---------------------------------------------------------------------------


def test_semantic_search_passes_full_vector_from_embedder(tmp_path) -> None:
    """The embedder code path must hand the full embedding to ``search_similar``.

    Regression for BUC-1570 — we previously did ``vecs[0]`` on the result
    of the legacy ``SageMakerEmbedder.embed(text)`` (which already returned
    a single vector), leaking a single ``float`` into the search pipeline.
    After the BUC-1605 migration the search route calls
    :func:`app.embedders.sync_bridge.embed_text_sync` which itself unwraps
    the async batched response. This test asserts that the vector handed
    to ``search_similar`` is the full 768-dim list.
    """
    duck = tmp_path / "fake.duck"
    duck.write_bytes(b"")

    captured: dict[str, object] = {}

    def _capture_search_similar(conn, query_vec, k=10):
        captured["query_vec"] = query_vec
        return []

    fake_backend = MagicMock()
    fake_backend.name = "fake"

    with patch("app.embedders.sync_bridge.embed_text_sync",
               return_value=[0.1] * 768), \
         patch("app.embedders.sync_bridge.get_embedder_or_none",
               return_value=fake_backend), \
         patch("app.routers.search._embed_unavailable", False), \
         patch("app.config.Settings.vec_db_path_for_repo",
               lambda self, repo: str(duck)), \
         patch("codebase_rag.storage.vector_store.open_or_create",
               return_value=MagicMock()), \
         patch("codebase_rag.storage.vector_store.search_similar",
               side_effect=_capture_search_similar), \
         patch("codebase_rag.storage.vector_store.read_centrality",
               return_value={}):
        resp = client.get(
            "/search/semantic",
            params={"q": "code indexer client", "k": 5, "repo": "fake"},
        )

    assert resp.status_code == 200, resp.text
    qv = captured.get("query_vec")
    # Must be the *full* list — not a single float that some prior buggy
    # version sliced out with ``vecs[0]``.
    assert isinstance(qv, list), f"expected list, got {type(qv).__name__}"
    assert len(qv) == 768
    assert all(isinstance(x, float) for x in qv)


def test_semantic_search_does_not_500_on_embedder_path(tmp_path) -> None:
    """End-to-end: backend-driven semantic search must not raise the
    BUC-1570 ``'float' object is not iterable`` error.

    Uses the *real* ``_l2_normalise`` (no patching of ``search_similar`` arg
    handling) and a stub bridge whose ``embed_text_sync`` returns a full
    vector.
    """
    from codebase_rag.storage import vector_store as _vs

    duck = tmp_path / "fake.duck"
    duck.write_bytes(b"")

    fake_backend = MagicMock()
    fake_backend.name = "fake"

    # Real _l2_normalise will be called inside our patched search_similar.
    def _real_search_similar(conn, query_vec, k=10):
        # Will raise 'float' object is not iterable if BUC-1570 regresses.
        _vs._l2_normalise(query_vec)
        return []

    with patch("app.embedders.sync_bridge.embed_text_sync",
               return_value=[0.5] * 768), \
         patch("app.embedders.sync_bridge.get_embedder_or_none",
               return_value=fake_backend), \
         patch("app.routers.search._embed_unavailable", False), \
         patch("app.config.Settings.vec_db_path_for_repo",
               lambda self, repo: str(duck)), \
         patch("codebase_rag.storage.vector_store.open_or_create",
               return_value=MagicMock()), \
         patch("codebase_rag.storage.vector_store.search_similar",
               side_effect=_real_search_similar), \
         patch("codebase_rag.storage.vector_store.read_centrality",
               return_value={}):
        resp = client.get(
            "/search/semantic",
            params={"q": "code indexer client", "k": 5, "repo": "TheForge"},
        )

    assert resp.status_code == 200, resp.text
    # No "'float' object is not iterable" anywhere in the response.
    body = resp.json()
    detail = str(body.get("detail", ""))
    assert "not iterable" not in detail
