"""Tests for POST /context-bundle."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _mock_conn_with_calls(callee_map: dict[str, list[str]], source_rows: list[dict]) -> MagicMock:
    """Return a mock LadybugDB connection.

    callee_map: { caller_qn → [callee_qn, ...] }
    source_rows: rows returned for CYPHER_GET_FUNCTION_SOURCE_LOCATION queries
    """

    def execute_side_effect(query: str, params: dict | None = None):
        result = MagicMock()

        if "CALLS" in query and params:
            qn = params.get("qn", "")
            callees = callee_map.get(qn, [])
            rows = [{"callee": c} for c in callees]
            col_names = ["callee"]
        elif "start_line" in query and params:
            qn = params.get("node_id", "")
            rows = [r for r in source_rows if r.get("qualified_name") == qn]
            col_names = ["qualified_name", "start_line", "end_line", "path"]
        else:
            rows = []
            col_names = []

        remaining = list(rows)
        result.get_column_names.return_value = col_names
        result.has_next.side_effect = lambda: bool(remaining)
        result.get_next.side_effect = lambda: [remaining.pop(0).get(c) for c in col_names]
        return result

    conn = MagicMock()
    conn.execute.side_effect = execute_side_effect
    return conn


def test_context_bundle_returns_symbols(tmp_path: Path) -> None:
    src_file = tmp_path / "mymod.py"
    src_file.write_text("def foo():\n    pass\n")

    seed = [
        {"qualified_name": "mymod.foo", "score": 0.9, "node_id": "mymod.foo", "name": "foo", "type": "Function"}
    ]
    callee_map: dict[str, list[str]] = {"mymod.foo": ["mymod.bar"]}
    source_rows = [
        {"qualified_name": "mymod.foo", "start_line": 1, "end_line": 2, "path": str(src_file)},
        {"qualified_name": "mymod.bar", "start_line": 1, "end_line": 2, "path": str(src_file)},
    ]
    conn = _mock_conn_with_calls(callee_map, source_rows)

    import codebase_rag.tools.semantic_search as _sem

    with (
        patch.object(_sem, "semantic_code_search", return_value=seed),
        patch("app.routers.context_bundle._get_conn", return_value=conn),
    ):
        resp = client.post(
            "/context-bundle",
            json={"repo_path": str(tmp_path), "task_description": "implement foo"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "mymod.foo" in body["symbols"]
    assert "mymod.bar" in body["symbols"]
    assert "mymod.foo" in body["call_graph"]
    assert "total_tokens" in body


def test_context_bundle_empty_when_no_semantic_results(tmp_path: Path) -> None:
    import codebase_rag.tools.semantic_search as _sem

    with patch.object(_sem, "semantic_code_search", return_value=[]):
        resp = client.post(
            "/context-bundle",
            json={"repo_path": str(tmp_path), "task_description": "nothing matches"},
        )

    assert resp.status_code == 200
    assert resp.json()["symbols"] == []
    assert resp.json()["total_tokens"] == 0


def test_context_bundle_depth_zero(tmp_path: Path) -> None:
    """With depth=0 the call graph should not be expanded."""
    seed = [
        {"qualified_name": "mod.fn", "score": 0.8, "node_id": "mod.fn", "name": "fn", "type": "Function"}
    ]
    conn = MagicMock()
    result = MagicMock()
    result.get_column_names.return_value = ["qualified_name", "start_line", "end_line", "path"]
    result.has_next.return_value = False
    conn.execute.return_value = result

    import codebase_rag.tools.semantic_search as _sem

    with (
        patch.object(_sem, "semantic_code_search", return_value=seed),
        patch("app.routers.context_bundle._get_conn", return_value=conn),
    ):
        resp = client.post(
            "/context-bundle",
            json={"repo_path": str(tmp_path), "task_description": "do fn", "depth": 0},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["symbols"] == ["mod.fn"]
    # No CALLS query should have been made
    for call in conn.execute.call_args_list:
        assert "CALLS" not in (call[0][0] if call[0] else "")
