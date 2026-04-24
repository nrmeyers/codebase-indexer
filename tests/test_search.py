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


def test_semantic_search_returns_results() -> None:
    # Patch _semantic_fn directly — the module-level cache stores a function
    # reference after first call, so patching the module attr alone is unreliable.
    mock_results = [
        {"qualified_name": "mymod.foo", "score": 0.95, "node_id": "mymod.foo", "name": "foo", "type": "Function"},
        {"qualified_name": "mymod.bar", "score": 0.80, "node_id": "mymod.bar", "name": "bar", "type": "Method"},
    ]
    with patch("app.routers.search._semantic_fn", lambda q, top_k=10: mock_results):
        resp = client.get("/search/semantic", params={"q": "find all functions", "k": 5})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 2
    assert results[0]["symbol"] == "mymod.foo"
    assert results[0]["score"] == pytest.approx(0.95)


def test_semantic_search_empty() -> None:
    # Patch _semantic_fn directly — the module-level cache stores a function
    # reference after first call, so patching the module attr alone is unreliable.
    with patch("app.routers.search._semantic_fn", lambda q, top_k=10: []):
        resp = client.get("/search/semantic", params={"q": "nothing"})
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
    """When codebase_rag.tools.semantic_search cannot be imported (e.g. torch
    missing), the endpoint must return 503 with a valid JSON body — not 500.

    Strategy: set sys.modules entry to None which makes Python raise
    ImportError on ``from codebase_rag.tools.semantic_search import …``.
    """
    import sys
    import app.routers.search as _search_mod

    # Reset cached state so the lazy-load branch is taken.
    original_fn = _search_mod._semantic_fn
    original_unavail = _search_mod._semantic_unavailable
    _search_mod._semantic_fn = None
    _search_mod._semantic_unavailable = False

    # Save the real module so we can restore it after the test.
    real_mod = sys.modules.get("codebase_rag.tools.semantic_search")

    try:
        # Setting sys.modules[name] = None causes `from name import …` to
        # raise ImportError — this simulates a missing ML dependency.
        sys.modules["codebase_rag.tools.semantic_search"] = None  # type: ignore[assignment]
        resp = client.get("/search/semantic", params={"q": "retry http"})
    finally:
        # Restore everything regardless of outcome.
        _search_mod._semantic_fn = original_fn
        _search_mod._semantic_unavailable = original_unavail
        if real_mod is not None:
            sys.modules["codebase_rag.tools.semantic_search"] = real_mod
        else:
            sys.modules.pop("codebase_rag.tools.semantic_search", None)

    assert resp.status_code == 503
    body = resp.json()
    # Response must be valid JSON with a 'detail' key (FastAPI HTTPException shape)
    assert "detail" in body
    assert isinstance(body["detail"], str)


def test_semantic_search_503_uses_fast_fail_after_first_import_failure() -> None:
    """Once the import fails, _semantic_unavailable=True and subsequent calls
    skip the import attempt and return 503 immediately."""
    import app.routers.search as _search_mod

    original_fn = _search_mod._semantic_fn
    original_unavail = _search_mod._semantic_unavailable

    # Simulate a prior import failure having set the flag
    _search_mod._semantic_fn = None
    _search_mod._semantic_unavailable = True

    try:
        resp = client.get("/search/semantic", params={"q": "anything"})
    finally:
        _search_mod._semantic_fn = original_fn
        _search_mod._semantic_unavailable = original_unavail

    assert resp.status_code == 503
    body = resp.json()
    assert "detail" in body
    assert "unavailable" in body["detail"].lower()
