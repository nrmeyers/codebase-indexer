"""Tests for POST /context-bundle."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.models import SemanticResult, SemanticSearchResponse

client = TestClient(app)


def _semantic_response(seed: list[dict]) -> SemanticSearchResponse:
    """Build a SemanticSearchResponse from ``[{qualified_name, score}, ...]``.

    Context-bundle seeds from the search router's ``_semantic_search_impl``
    (LE-180 — same embedding path + fusion that /search/semantic uses), so
    tests patch that function rather than the legacy ``semantic_code_search``.
    """
    return SemanticSearchResponse(
        results=[
            SemanticResult(symbol=r["qualified_name"], score=r.get("score", 0.5), type="")
            for r in seed
        ],
        search_intent="semantic",
    )


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

    with (
        patch(
            "app.routers.search._semantic_search_impl",
            return_value=_semantic_response(seed),
        ),
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
    with patch(
        "app.routers.search._semantic_search_impl",
        return_value=_semantic_response([]),
    ):
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

    with (
        patch(
            "app.routers.search._semantic_search_impl",
            return_value=_semantic_response(seed),
        ),
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


def test_context_bundle_rejects_nonexistent_repo() -> None:
    """POST /context-bundle with a path that does not exist should return 422."""
    resp = client.post(
        "/context-bundle",
        json={
            "repo_path": "/this/path/absolutely/does/not/exist/on/disk",
            "task_description": "retry HTTP requests",
        },
    )
    assert resp.status_code == 422
    # Pydantic wraps field_validator errors in the standard 422 detail array
    detail = resp.json().get("detail", "")
    detail_str = str(detail)
    assert "does not exist" in detail_str or "repo_path" in detail_str


def test_is_test_or_script_path_classification() -> None:
    """Unit-level: the path classifier flags test/script FQNs but not impl."""
    from app.routers.context_bundle import _is_test_or_script_path

    # Test / script paths — should be flagged.
    assert _is_test_or_script_path("TheForge.scripts.calibrate-refusal.pct")
    assert _is_test_or_script_path("TheForge.scripts.latency-bench.runSolo")
    assert _is_test_or_script_path("repo.src.foo.bar.test.helper")
    assert _is_test_or_script_path("repo.src.foo.bar.spec.case")
    assert _is_test_or_script_path("repo.tests.unit.checkThing")
    assert _is_test_or_script_path("repo.src.__tests__.fixture")

    # Real implementation — must NOT be flagged (regression guard: a symbol
    # that merely *contains* "test" inside a longer identifier in production
    # code should pass through unpenalised).
    assert not _is_test_or_script_path(
        "TheForge.src.services.orchestration.zero-retrieval-refusal.evaluateRetrievalSignal"
    )
    assert not _is_test_or_script_path("repo.src.services.runTestSuite")


def test_context_bundle_seeds_implementation_over_scripts(tmp_path: Path) -> None:
    """LE-180 regression: for an NL query whose implementation lives in a
    non-script file, the bundle seeds must include the implementation symbol
    AHEAD of script/test files that merely mention the query terms.

    Reproduces the live bug: /search/semantic ranked
    ``…zero-retrieval-refusal.evaluateRetrievalSignal`` at ~0.83 while
    /context-bundle's exact-name boost flooded the seed cap with
    ``scripts.calibrate-refusal.*`` and dropped the implementation entirely.
    """
    src_file = tmp_path / "impl.py"
    src_file.write_text("def evaluate():\n    return 0.30\n")

    impl = "myrepo.src.services.orchestration.zero-retrieval-refusal.evaluateRetrievalSignal"
    floor = "myrepo.src.services.orchestration.zero-retrieval-refusal.resolveSemanticFloor"
    script_a = "myrepo.scripts.calibrate-refusal.pct"
    script_b = "myrepo.scripts.latency-bench.runSolo"

    # Semantic ranking (what /search/semantic uses) correctly puts the
    # implementation at the top with the script ranked lower.
    seed = [
        {"qualified_name": impl, "score": 0.83, "node_id": impl, "name": "evaluateRetrievalSignal", "type": "Function"},
        {"qualified_name": floor, "score": 0.82, "node_id": floor, "name": "resolveSemanticFloor", "type": "Function"},
        {"qualified_name": script_a, "score": 0.80, "node_id": script_a, "name": "pct", "type": "Function"},
    ]

    # The boost connection returns the scripts as exact-name hits (the bug
    # trigger: words like "refusal"/"gate"/"threshold" match script symbols),
    # and CALLS/source queries behave normally.
    def execute_side_effect(query: str, params: dict | None = None):
        result = MagicMock()
        if "n.name IN" in query:  # exact-name boost — return scripts
            rows = [{"qn": script_a}, {"qn": script_b}]
            cols = ["qn"]
        elif "CALLS" in query and params:
            rows = []
            cols = ["callee"]
        elif "start_line" in query and params:
            qn = params.get("node_id", "")
            rows = [{"qualified_name": qn, "start_line": 1, "end_line": 2, "path": str(src_file)}]
            cols = ["qualified_name", "start_line", "end_line", "path"]
        else:
            rows, cols = [], []
        remaining = list(rows)
        result.get_column_names.return_value = cols
        result.has_next.side_effect = lambda: bool(remaining)
        result.get_next.side_effect = lambda: [remaining.pop(0).get(c) for c in cols]
        return result

    conn = MagicMock()
    conn.execute.side_effect = execute_side_effect

    with (
        patch(
            "app.routers.search._semantic_search_impl",
            return_value=_semantic_response(seed),
        ),
        patch("app.routers.context_bundle._get_conn", return_value=conn),
    ):
        resp = client.post(
            "/context-bundle",
            json={
                "repo_path": str(tmp_path),
                "task_description": "Where is the zero-retrieval refusal gate implemented and what threshold does it use?",
                "k": 20,
                "depth": 0,
            },
        )

    assert resp.status_code == 200
    symbols = resp.json()["symbols"]

    # The implementation symbol must be present in the seeds — the original
    # bug dropped it entirely in favour of the scripts.
    assert impl in symbols, f"implementation symbol missing from seeds: {symbols}"

    from app.routers.context_bundle import _is_test_or_script_path

    assert not _is_test_or_script_path(impl)
    assert _is_test_or_script_path(script_a)

    # Assert RANKING directly: the response ``symbols`` field is sorted
    # alphabetically (a deliberate determinism choice in the router), which
    # would hide the seed ordering. So we exercise the scored seed selection
    # at a cap tight enough that the down-weighted scripts are excluded while
    # the implementation survives. The intent upgrade floors effective_k at
    # 12, but only 3 semantic + 2 script candidates exist here, so we assert
    # the down-weight relationship holds: impl scores 0.83/0.82 vs scripts
    # 0.80 * 0.4 = 0.32, so impl symbols MUST sort ahead of the scripts.
    from app.routers import context_bundle as _cb

    captured: dict[str, list[str]] = {}
    orig_expand = _cb._expand_call_graph

    def _capture_expand(conn_, seed_symbols, depth, **kwargs):  # type: ignore[no-untyped-def]
        captured["seeds"] = list(seed_symbols)
        return orig_expand(conn_, seed_symbols, depth, **kwargs)

    with (
        patch(
            "app.routers.search._semantic_search_impl",
            return_value=_semantic_response(seed),
        ),
        patch("app.routers.context_bundle._get_conn", return_value=conn),
        patch.object(_cb, "_expand_call_graph", side_effect=_capture_expand),
    ):
        resp2 = client.post(
            "/context-bundle",
            json={
                "repo_path": str(tmp_path),
                "task_description": "Where is the zero-retrieval refusal gate implemented and what threshold does it use?",
                "k": 20,
                "depth": 1,
            },
        )
    assert resp2.status_code == 200
    ranked_seeds = captured["seeds"]
    impl_idx = ranked_seeds.index(impl)
    script_idx = ranked_seeds.index(script_a)
    assert impl_idx < script_idx, (
        f"implementation must rank ahead of down-weighted script; "
        f"got order {ranked_seeds}"
    )


def test_context_bundle_symbols_are_relevance_ordered(tmp_path: Path) -> None:
    """LE-182: ``symbols`` must come back in relevance order — the top semantic
    hit at the FRONT, ahead of alphabetically-earlier call-graph neighbours.

    Reproduces the live bug: the bundle returned ``sorted(all_symbols)``
    (alphabetical), so a high-signal seed like ``…zero_retrieval_refusal.*``
    sat far below alphabetically-earlier neighbours (``api_server``,
    ``audit_trail`` …). A consumer truncating by array order dropped the real
    hit before it reached the model.
    """
    src_file = tmp_path / "impl.py"
    src_file.write_text("def evaluate():\n    return 0.30\n")

    # The true top hit. Its FQN sorts LATE alphabetically (z…), so under the
    # old alphabetical ordering it landed at the bottom of ``symbols``.
    top_hit = "myrepo.src.services.orchestration.zero_retrieval_refusal.evaluateRetrievalSignal"
    # Lower-ranked seeds whose FQNs sort EARLIER alphabetically (a…).
    seed_b = "myrepo.src.services.audit_trail.auditChatTurnRefused"
    seed_c = "myrepo.src.services.api_server.broadcast"
    # A call-graph neighbour of the top hit, alphabetically earliest of all.
    neighbour = "myrepo.src.services.aaa_helper.format"

    seed = [
        {"qualified_name": top_hit, "score": 0.83, "node_id": top_hit, "name": "evaluateRetrievalSignal", "type": "Function"},
        {"qualified_name": seed_b, "score": 0.40, "node_id": seed_b, "name": "auditChatTurnRefused", "type": "Function"},
        {"qualified_name": seed_c, "score": 0.35, "node_id": seed_c, "name": "broadcast", "type": "Function"},
    ]
    callee_map = {top_hit: [neighbour]}
    source_rows = [
        {"qualified_name": qn, "start_line": 1, "end_line": 2, "path": str(src_file)}
        for qn in (top_hit, seed_b, seed_c, neighbour)
    ]
    conn = _mock_conn_with_calls(callee_map, source_rows)

    with (
        patch(
            "app.routers.search._semantic_search_impl",
            return_value=_semantic_response(seed),
        ),
        patch("app.routers.context_bundle._get_conn", return_value=conn),
    ):
        resp = client.post(
            "/context-bundle",
            json={
                "repo_path": str(tmp_path),
                "task_description": "Where is the zero retrieval refusal gate implemented?",
                "k": 20,
                "depth": 1,
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    symbols = body["symbols"]

    # The top semantic hit must be FIRST — not buried alphabetically.
    assert symbols[0] == top_hit, f"top hit not at front: {symbols}"

    # Every seed must rank ahead of the pure call-graph neighbour, even though
    # the neighbour sorts alphabetically first (aaa_helper).
    assert symbols.index(top_hit) < symbols.index(neighbour)
    assert symbols.index(seed_b) < symbols.index(neighbour)
    assert symbols.index(seed_c) < symbols.index(neighbour)

    # Seed ordering follows merged score descending.
    assert symbols.index(top_hit) < symbols.index(seed_b) < symbols.index(seed_c)

    # Additive scores field (backward-compatible) is present and parallel to
    # ``symbols``: ``symbols`` equals the score-descending order of ``scores``.
    scores = body["scores"]
    assert set(scores) == set(symbols)
    assert scores[top_hit] >= scores[seed_b] >= scores[seed_c]
    # Pure neighbour scores strictly below the lowest seed.
    assert scores[neighbour] < scores[seed_c]
    # symbols is exactly sorted(scores, by value desc, tie-break FQN).
    expected = sorted(scores, key=lambda s: (-scores[s], s))
    assert symbols == expected


def test_context_bundle_503_when_semantic_search_unavailable(tmp_path: Path) -> None:
    """When the semantic seed search raises, the endpoint returns 503."""
    with patch(
        "app.routers.search._semantic_search_impl",
        side_effect=RuntimeError("model not loaded"),
    ):
        resp = client.post(
            "/context-bundle",
            json={"repo_path": str(tmp_path), "task_description": "anything"},
        )

    assert resp.status_code == 503
    assert "unavailable" in resp.json()["detail"].lower()


def test_expand_call_graph_caller_expansion() -> None:
    """caller_cap > 0 pulls inbound callers (depth 1, tests excluded) into the
    BFS frontier so their callees — the wiring siblings of the seed — are
    reachable on later hops."""
    from app.routers.context_bundle import _expand_call_graph

    edges_out = {
        "app.mount.mountRoutes": ["app.routes.makeRouter", "app.mw.requireRole"],
    }
    edges_in = {
        "app.routes.makeRouter": [
            "app.mount.mountRoutes",
            "tests.unit.mount.test.makeApp",  # must be filtered
        ],
    }

    conn = MagicMock()

    def _execute(query: str, params: dict[str, str]):  # type: ignore[no-untyped-def]
        qn = params["qn"]
        if "-[:CALLS]->(n" in query:  # inbound caller lookup
            return [{"caller": c} for c in edges_in.get(qn, [])]
        return [{"callee": c} for c in edges_out.get(qn, [])]

    conn.execute.side_effect = _execute
    with patch(
        "app.routers.context_bundle._result_to_rows", side_effect=lambda rows: rows
    ):
        all_symbols, call_graph, symbol_depth = _expand_call_graph(
            conn, ["app.routes.makeRouter"], 2, caller_cap=3
        )

    assert "app.mount.mountRoutes" in all_symbols
    assert symbol_depth["app.mount.mountRoutes"] == 1
    # The caller's other callee (the middleware sibling) is reached.
    assert "app.mw.requireRole" in all_symbols
    # Test-file caller filtered out.
    assert "tests.unit.mount.test.makeApp" not in all_symbols
    # Edge direction recorded caller -> seed.
    assert "app.routes.makeRouter" in call_graph["app.mount.mountRoutes"]

    # caller_cap=0 (default) keeps the old callee-only behaviour.
    with patch(
        "app.routers.context_bundle._result_to_rows", side_effect=lambda rows: rows
    ):
        all_symbols0, _, _ = _expand_call_graph(conn, ["app.routes.makeRouter"], 2)
    assert "app.mount.mountRoutes" not in all_symbols0


def test_context_bundle_lexical_seed_leg(tmp_path: Path) -> None:
    """Design-intent bundles include BM25 lexical hits as guaranteed seeds —
    the leg that catches comment-only signal (no embedding, no CALLS edge)."""
    src_file = tmp_path / "auth.py"
    src_file.write_text("# AAD provider\ndef verify_session():\n    pass\n")

    # Semantic returns plenty of unrelated-but-high-scoring symbols so the
    # lexical hit cannot make the seed window on score alone.
    seed = [
        {"qualified_name": f"app.other.fn{i}", "score": 0.9 - i * 0.001}
        for i in range(24)
    ]
    source_rows = [
        {"qualified_name": s["qualified_name"], "start_line": 1, "end_line": 2, "path": str(src_file)}
        for s in seed
    ] + [
        {"qualified_name": "app.auth.session.verify_session", "start_line": 1, "end_line": 3, "path": str(src_file)},
    ]
    conn = _mock_conn_with_calls({}, source_rows)

    with (
        patch(
            "app.routers.search._semantic_search_impl",
            return_value=_semantic_response(seed),
        ),
        patch("app.routers.context_bundle._get_conn", return_value=conn),
        patch(
            "app.routers.context_bundle._lexical_seed_hits",
            return_value=[
                {
                    "qn": "app.auth.session.verify_session",
                    "file_path": str(src_file),
                    "start_line": 1,
                    "end_line": 3,
                    "kind": "Function",
                }
            ],
        ) as lex_mock,
    ):
        resp = client.post(
            "/context-bundle",
            json={
                "repo_path": str(tmp_path),
                "task_description": "Add AAD role checks to the skill API",
                "intent": "design",
                "depth": 0,
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert lex_mock.called
    assert "app.auth.session.verify_session" in body["symbols"]

    # symbol intent has no lexical leg — the helper must not be consulted.
    with (
        patch(
            "app.routers.search._semantic_search_impl",
            return_value=_semantic_response(seed),
        ),
        patch("app.routers.context_bundle._get_conn", return_value=conn),
        patch(
            "app.routers.context_bundle._lexical_seed_hits",
            return_value=[
                {
                    "qn": "app.auth.session.verify_session",
                    "file_path": str(src_file),
                    "start_line": 1,
                    "end_line": 3,
                    "kind": "Function",
                }
            ],
        ) as lex_mock_sym,
    ):
        resp2 = client.post(
            "/context-bundle",
            json={
                "repo_path": str(tmp_path),
                "task_description": "what does fn1 do",
                "intent": "symbol",
                "depth": 0,
            },
        )

    assert resp2.status_code == 200
    assert not lex_mock_sym.called
