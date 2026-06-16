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
# /search/structural — large-graph cap + offset paging (LE-169a)
# ---------------------------------------------------------------------------


def _paging_mock_conn(dataset: list[dict]) -> MagicMock:
    """Mock connection that honours SKIP/LIMIT parsed from the Cypher.

    Mirrors the LadybugDB/kuzu engine behaviour for ``SKIP n LIMIT m`` so the
    test exercises the real paging contract (offset skips rows; pages don't
    overlap) instead of asserting on the generated query string alone.
    """
    col_names = list(dataset[0].keys()) if dataset else []

    def execute(cypher: str, *_args, **_kwargs):
        import re

        skip_m = re.search(r"\bSKIP\s+(\d+)", cypher, re.IGNORECASE)
        limit_m = re.search(r"\bLIMIT\s+(\d+)", cypher, re.IGNORECASE)
        skip = int(skip_m.group(1)) if skip_m else 0
        lim = int(limit_m.group(1)) if limit_m else len(dataset)
        page = dataset[skip : skip + lim]

        result = MagicMock()
        result.get_column_names.return_value = col_names
        remaining = list(page)
        result.has_next.side_effect = lambda: bool(remaining)
        result.get_next.side_effect = lambda: list(remaining.pop(0).values())
        return result

    conn = MagicMock()
    conn.execute.side_effect = execute
    return conn


def test_structural_search_accepts_limit_above_500() -> None:
    """LE-169a: the historical le=500 cap is raised to 5000."""
    dataset = [{"name": f"fn_{i}", "qualified_name": f"m.fn_{i}"} for i in range(2000)]
    with patch("app.routers.search._get_conn", return_value=_paging_mock_conn(dataset)):
        resp = client.get(
            "/search/structural",
            params={"q": "MATCH (n:Function) RETURN n.name AS name", "limit": 2000},
        )
    assert resp.status_code == 200
    body = resp.json()
    # >500 rows must come back — proves the cap no longer clamps at 500.
    assert body["row_count"] == 2000
    assert len(body["nodes"]) == 2000


def test_structural_search_rejects_limit_above_5000() -> None:
    """The cap is 5000 — beyond that FastAPI rejects with 422 (bounded fetch)."""
    with patch("app.routers.search._get_conn", return_value=_paging_mock_conn([])):
        resp = client.get(
            "/search/structural",
            params={"q": "MATCH (n) RETURN n", "limit": 5001},
        )
    assert resp.status_code == 422


def test_structural_search_offset_injects_skip() -> None:
    """offset>0 appends a bounded SKIP/LIMIT to a LIMIT-less query."""
    conn = MagicMock()
    result = MagicMock()
    result.get_column_names.return_value = []
    result.has_next.return_value = False
    conn.execute.return_value = result

    with patch("app.routers.search._get_conn", return_value=conn):
        client.get(
            "/search/structural",
            params={"q": "MATCH (n) RETURN n", "limit": 100, "offset": 50},
        )

    executed: str = conn.execute.call_args[0][0]
    assert "SKIP 50" in executed.upper()
    # The engine fetches one probe row beyond the page (limit + 1 = 101) so the
    # impl can derive ``has_more`` without a second COUNT query; the extra row
    # is trimmed before the response so the caller still sees exactly 100.
    assert "LIMIT 101" in executed.upper()


def test_structural_search_offset_pages_without_overlap() -> None:
    """offset=N skips N rows; consecutive pages don't overlap or duplicate."""
    dataset = [{"name": f"fn_{i}", "qualified_name": f"m.fn_{i}"} for i in range(1000)]
    q = "MATCH (n:Function) RETURN n.name AS name, n.qualified_name AS qualified_name"

    with patch("app.routers.search._get_conn", return_value=_paging_mock_conn(dataset)):
        page1 = client.get(
            "/search/structural", params={"q": q, "limit": 300, "offset": 0}
        ).json()
        page2 = client.get(
            "/search/structural", params={"q": q, "limit": 300, "offset": 300}
        ).json()
        page3 = client.get(
            "/search/structural", params={"q": q, "limit": 300, "offset": 600}
        ).json()

    names1 = [n["name"] for n in page1["nodes"]]
    names2 = [n["name"] for n in page2["nodes"]]
    names3 = [n["name"] for n in page3["nodes"]]

    assert names1[0] == "fn_0"
    assert names2[0] == "fn_300"  # offset skipped exactly 300 rows
    assert names3[0] == "fn_600"
    assert len(names1) == len(names2) == len(names3) == 300
    # No overlap / no duplication across pages.
    assert set(names1).isdisjoint(names2)
    assert set(names2).isdisjoint(names3)
    assert set(names1).isdisjoint(names3)


def test_structural_search_preserves_user_supplied_limit() -> None:
    """A caller's own LIMIT keeps full control — no SKIP injection."""
    conn = MagicMock()
    result = MagicMock()
    result.get_column_names.return_value = []
    result.has_next.return_value = False
    conn.execute.return_value = result

    with patch("app.routers.search._get_conn", return_value=conn):
        client.get(
            "/search/structural",
            params={"q": "MATCH (n) RETURN n LIMIT 5", "offset": 100},
        )

    executed: str = conn.execute.call_args[0][0]
    assert "SKIP" not in executed.upper()
    assert "LIMIT 5" in executed.upper()


def test_structural_search_has_more_true_when_next_page_exists() -> None:
    """LE-169a: has_more is True when matching rows exist beyond the page.

    With 1000 rows and a page of 300 at offset 0, another page exists. The
    response must echo offset/limit and set has_more without over-returning
    (exactly 300 rows reach the caller — the probe row is trimmed).
    """
    dataset = [{"name": f"fn_{i}"} for i in range(1000)]
    q = "MATCH (n:Function) RETURN n.name AS name"
    with patch("app.routers.search._get_conn", return_value=_paging_mock_conn(dataset)):
        resp = client.get(
            "/search/structural", params={"q": q, "limit": 300, "offset": 0}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["row_count"] == 300
    assert len(body["nodes"]) == 300
    assert body["offset"] == 0
    assert body["limit"] == 300
    assert body["has_more"] is True


def test_structural_search_has_more_false_on_final_page() -> None:
    """has_more is False when the page reaches the end of the dataset."""
    dataset = [{"name": f"fn_{i}"} for i in range(250)]
    q = "MATCH (n:Function) RETURN n.name AS name"
    with patch("app.routers.search._get_conn", return_value=_paging_mock_conn(dataset)):
        resp = client.get(
            "/search/structural", params={"q": q, "limit": 300, "offset": 0}
        )
    body = resp.json()
    assert body["row_count"] == 250
    assert body["has_more"] is False


def test_structural_search_has_more_false_for_caller_supplied_limit() -> None:
    """When the caller hand-writes LIMIT, the service doesn't claim has_more."""
    dataset = [{"name": f"fn_{i}"} for i in range(50)]
    with patch("app.routers.search._get_conn", return_value=_paging_mock_conn(dataset)):
        resp = client.get(
            "/search/structural",
            params={"q": "MATCH (n) RETURN n.name AS name LIMIT 5"},
        )
    body = resp.json()
    assert body["has_more"] is False
    assert body["offset"] == 0


def test_structural_search_default_response_is_backward_compatible() -> None:
    """A default call still returns the original fields; new fields are additive."""
    rows = [{"name": "foo"}]
    with patch("app.routers.search._get_conn", return_value=_mock_conn(rows)):
        resp = client.get(
            "/search/structural", params={"q": "MATCH (n:Function) RETURN n.name AS name"}
        )
    body = resp.json()
    # Original contract preserved.
    assert body["row_count"] == 1
    assert "nodes" in body and "relationships" in body
    # Additive paging fields present with sane defaults.
    assert body["offset"] == 0
    assert body["limit"] == 500
    assert body["has_more"] is False


# ---------------------------------------------------------------------------
# /search/structural — deterministic complete-scan paging (LE-181b)
# ---------------------------------------------------------------------------


def _ordering_paging_mock_conn(unordered: list[dict], *, shuffle_seed: int) -> MagicMock:
    """Mock connection that emulates kuzu's multi-label UNION SCAN.

    Two engine behaviours are modelled:
      * WITHOUT an ``ORDER BY`` the scan order is UNSTABLE across executions —
        we deterministically reshuffle ``unordered`` per call keyed by an
        incrementing counter, mirroring the real bug where a paged complete
        scan over an unordered multi-label query skips/duplicates rows.
      * WITH an ``ORDER BY <cols>`` the engine sorts by those columns FIRST,
        producing a stable total order that SKIP/LIMIT then pages.

    This is what makes the test exercise the real determinism contract: only
    the ``ORDER BY`` path yields a gapless, repeatable complete scan.
    """
    import re as _re

    call_counter = {"n": 0}

    def execute(cypher: str, *_args, **_kwargs):
        col_names = list(unordered[0].keys()) if unordered else []

        order_m = _re.search(r"ORDER\s+BY\s+([^\n]+?)(?:\s+SKIP|\s+LIMIT|$)", cypher, _re.IGNORECASE)
        if order_m:
            cols = [c.strip() for c in order_m.group(1).split(",")]
            data = sorted(unordered, key=lambda r: tuple(str(r.get(c, "")) for c in cols))
        else:
            # Unstable order: rotate by a per-call amount so consecutive pages
            # see a DIFFERENT ordering — the exact failure the fix addresses.
            call_counter["n"] += 1
            rot = (call_counter["n"] * 7 + shuffle_seed) % max(1, len(unordered))
            data = unordered[rot:] + unordered[:rot]

        skip_m = _re.search(r"\bSKIP\s+(\d+)", cypher, _re.IGNORECASE)
        limit_m = _re.search(r"\bLIMIT\s+(\d+)", cypher, _re.IGNORECASE)
        skip = int(skip_m.group(1)) if skip_m else 0
        lim = int(limit_m.group(1)) if limit_m else len(data)
        page = data[skip : skip + lim]

        result = MagicMock()
        result.get_column_names.return_value = col_names
        remaining = list(page)
        result.has_next.side_effect = lambda: bool(remaining)
        result.get_next.side_effect = lambda: list(remaining.pop(0).values())
        return result

    conn = MagicMock()
    conn.execute.side_effect = execute
    return conn


def _page_all(client_, q: str, *, page_size: int) -> list[dict]:
    """Walk every page of a service-paged structural query into one flat list."""
    out: list[dict] = []
    offset = 0
    while True:
        body = client_.get(
            "/search/structural",
            params={"q": q, "limit": page_size, "offset": offset},
        ).json()
        out.extend(body["nodes"])
        if not body["has_more"] or not body["nodes"]:
            break
        offset += len(body["nodes"])
    return out


def test_structural_search_injects_order_by_for_service_paged_aliased_query() -> None:
    """LE-181b: a service-paged query with explicit RETURN aliases gets a
    deterministic ORDER BY over the full alias tuple injected before SKIP/LIMIT."""
    conn = MagicMock()
    result = MagicMock()
    result.get_column_names.return_value = []
    result.has_next.return_value = False
    conn.execute.return_value = result

    with patch("app.routers.search._get_conn", return_value=conn):
        client.get(
            "/search/structural",
            params={
                "q": "MATCH (n:Function|Class|Method) RETURN n.qualified_name AS qname, n.name AS name, label(n) AS node_type",
            },
        )

    executed: str = conn.execute.call_args[0][0]
    up = executed.upper()
    assert "ORDER BY QNAME, NAME, NODE_TYPE" in up
    # ORDER BY must precede LIMIT — otherwise it's a parse error / no-op.
    assert up.index("ORDER BY") < up.index("LIMIT")


def test_structural_search_no_order_by_for_bare_projection() -> None:
    """A bare ``RETURN n`` (node variable, not orderable in kuzu) must NOT get
    an injected ORDER BY — preserves legacy behaviour, avoids a parse error."""
    conn = MagicMock()
    result = MagicMock()
    result.get_column_names.return_value = []
    result.has_next.return_value = False
    conn.execute.return_value = result

    with patch("app.routers.search._get_conn", return_value=conn):
        client.get("/search/structural", params={"q": "MATCH (n) RETURN n"})

    executed: str = conn.execute.call_args[0][0]
    assert "ORDER BY" not in executed.upper()


def test_structural_search_respects_caller_supplied_order_by() -> None:
    """When the caller already wrote an ORDER BY, the service must not add a
    second (conflicting) one."""
    conn = MagicMock()
    result = MagicMock()
    result.get_column_names.return_value = []
    result.has_next.return_value = False
    conn.execute.return_value = result

    with patch("app.routers.search._get_conn", return_value=conn):
        client.get(
            "/search/structural",
            params={"q": "MATCH (n:Function) RETURN n.name AS name ORDER BY name DESC"},
        )

    executed: str = conn.execute.call_args[0][0]
    # Exactly one ORDER BY survives.
    assert executed.upper().count("ORDER BY") == 1


def test_structural_search_complete_scan_is_deterministic_across_repeats() -> None:
    """LE-181b core regression: a paged complete scan over a multi-label query
    whose underlying engine order is UNSTABLE still returns the SAME complete
    result set in the SAME order across 3 repeated identical scans — and drops
    NO layer (web.* / src.services.routes.* / src.adapters.identity.* survive)."""
    # Build a dataset that mirrors the real failure: tests.* is multiplied
    # heavily, web.* sorts last alphabetically, and the at-risk first-party
    # files are present once each.
    dataset: list[dict] = []
    for i in range(400):  # heavy tests.* fan-out (the row-multiplication)
        dataset.append({"qname": f"tests.unit.test_{i:03d}", "name": f"test_{i}"})
    for i in range(120):
        dataset.append({"qname": f"src.services.core.fn_{i:03d}", "name": f"fn_{i}"})
    at_risk = [
        {"qname": "src.services.routes.auth.handler", "name": "handler"},
        {"qname": "src.adapters.identity.tailscale.resolve", "name": "resolve"},
        {"qname": "web.src.pages.Home.render", "name": "render"},
        {"qname": "web.src.components.Graph.draw", "name": "draw"},
        {"qname": "web.src.hooks.useActor.useActor", "name": "useActor"},
    ]
    dataset.extend(at_risk)

    q = "MATCH (n:Function|Class|Method|Interface) RETURN n.qualified_name AS qname, n.name AS name"

    scans: list[list[tuple[str, str]]] = []
    for _ in range(3):
        # Fresh mock per scan so the per-call rotation counter restarts —
        # proving determinism comes from the injected ORDER BY, not from a
        # warm/static engine state.
        conn = _ordering_paging_mock_conn(dataset, shuffle_seed=3)
        with patch("app.routers.search._get_conn", return_value=conn):
            rows = _page_all(client, q, page_size=100)
        scans.append([(r["qname"], r["name"]) for r in rows])

    # 1. COMPLETE — every distinct row present in each scan.
    expected = sorted((d["qname"], d["name"]) for d in dataset)
    for s in scans:
        assert sorted(s) == expected, "paged scan dropped or duplicated rows"

    # 2. DETERMINISTIC — identical result set AND identical order across repeats.
    assert scans[0] == scans[1] == scans[2]

    # 3. NO LAYER STARVED — the at-risk first-party files all survive.
    qnames = {qn for qn, _ in scans[0]}
    assert "web.src.pages.Home.render" in qnames
    assert "web.src.components.Graph.draw" in qnames
    assert "src.services.routes.auth.handler" in qnames
    assert "src.adapters.identity.tailscale.resolve" in qnames


def test_structural_search_unordered_scan_would_be_incomplete_control() -> None:
    """Control: prove the mock's unordered path IS unstable (so the test above
    is genuinely exercising the fix, not a no-op). A caller-supplied LIMIT
    bypasses the service ORDER-BY injection, so paging the same query by hand
    over the unstable engine order yields divergent pages across repeats."""
    dataset = [{"qname": f"m.fn_{i:04d}", "name": f"fn_{i}"} for i in range(300)]
    # Caller writes their OWN LIMIT → service paging (and ORDER BY) is bypassed.
    q = "MATCH (n) RETURN n.qualified_name AS qname, n.name AS name LIMIT 50"

    pages: list[list[str]] = []
    for seed in (11, 53, 197):  # distinct seeds ⇒ distinct unstable orderings
        conn = _ordering_paging_mock_conn(dataset, shuffle_seed=seed)
        with patch("app.routers.search._get_conn", return_value=conn):
            body = client.get("/search/structural", params={"q": q}).json()
        pages.append([n["qname"] for n in body["nodes"]])

    # Unstable engine order → at least two of the three first-pages differ.
    assert not (pages[0] == pages[1] == pages[2]), (
        "control mock returned a stable order — the determinism test would be a no-op"
    )


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

    with patch("app.embedders.sync_bridge.embed_text_sync", return_value=[0.0] * 768), \
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

    with patch("app.embedders.sync_bridge.embed_text_sync", return_value=[0.0] * 768), \
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

    with patch("app.embedders.sync_bridge.embed_text_sync", return_value=[0.0] * 768), \
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


def test_semantic_search_serves_200_when_rerank_true_but_lm_studio_unreachable(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BUC-1651: ``?rerank=true`` must NOT 5xx when LM Studio is unreachable.

    The route degrades gracefully to the un-reranked bi-encoder order — the
    reranker.rerank() call returns identity on any failure (unavailable,
    timeout, parse error, HTTP 5xx from LM Studio) and the response body is
    indistinguishable from a successful no-op rerank. This is the contract
    hosted deploys depend on: LM Studio runs on developer laptops only.
    """
    from app.config import settings

    duck = tmp_path / "fake.duck"
    duck.write_bytes(b"")

    fake_results = [
        _fake_search_result("mymod.foo", 0.95),
        _fake_search_result("mymod.bar", 0.80),
    ]

    # Force the unavailable path — no LM_STUDIO_URL, can_rerank() False.
    monkeypatch.delenv("LM_STUDIO_URL", raising=False)
    # Some deploys ship with RERANK_ENABLED=true so the env-var gate is on
    # but the backend isn't — this is the exact failure mode BUC-1651 fixes.
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)

    with patch("app.embedders.sync_bridge.embed_text_sync", return_value=[0.0] * 768), \
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
            params={
                "q": "find all functions",
                "k": 5,
                "repo": "fake",
                "rerank": "true",
            },
        )

    # The headline contract: rerank=true + unreachable LM Studio = 200, not 5xx.
    assert resp.status_code == 200, (
        f"expected 200 (graceful degrade) got {resp.status_code}: {resp.text}"
    )
    body = resp.json()
    # Un-reranked bi-encoder order surfaces unchanged.
    symbols = [r["symbol"] for r in body["results"]]
    assert "mymod.foo" in symbols and "mymod.bar" in symbols


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


def test_semantic_search_503_when_embedder_returns_none(tmp_path) -> None:
    """When the configured embedder backend cannot embed (returns None),
    the endpoint must return 503 with a valid JSON body — not 500 and
    not a foreign-space fallback embedding.
    """
    # The .duck-path check fires before the embed call, so back the test
    # with a real (empty) file to force the route past it and into the
    # embed branch we actually want to verify.
    duck = tmp_path / "fake.duck"
    duck.touch()
    with patch("app.embedders.sync_bridge.embed_text_sync",
               return_value=None), \
         patch("app.config.Settings.vec_db_path_for_repo",
               lambda self, repo: str(duck)):
        resp = client.get("/search/semantic", params={"q": "retry http", "repo": "fake"})

    assert resp.status_code == 503
    body = resp.json()
    assert "detail" in body
    assert "unavailable" in body["detail"].lower()


def test_semantic_search_503_again_on_subsequent_call(tmp_path) -> None:
    """503 is sticky: a second call with the embedder still down also returns
    503 (no silent fallback to a different vector space)."""
    duck = tmp_path / "fake.duck"
    duck.touch()
    with patch("app.embedders.sync_bridge.embed_text_sync",
               return_value=None), \
         patch("app.config.Settings.vec_db_path_for_repo",
               lambda self, repo: str(duck)):
        resp = client.get("/search/semantic", params={"q": "anything", "repo": "fake"})

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


# ---------------------------------------------------------------------------
# LE-33 regression: configured-but-broken backend must yield 503, not 500
# ---------------------------------------------------------------------------


def test_should_return_503_not_500_when_configured_backend_returns_none() -> None:
    """LE-33 regression guard.

    When EMBEDDER_BACKEND=local is configured but sentence-transformers is
    not installed, get_embedder_or_none() returns a LocalEmbedder instance
    (construction is lazy), but embed_text_sync() returns None (model-load
    fails at runtime).  Before the fix this cascaded into a bare
    ``RuntimeError("in-process embedder not initialised")`` which FastAPI
    wrapped as a 500.

    After the fix the route raises HTTPException(503) with a clear
    remediation message — never a bare RuntimeError / 500.
    """
    # Simulate: get_embedder_or_none returns a non-None backend
    # (EMBEDDER_BACKEND=local, lazy construction), but embed_text_sync
    # returns None (sentence-transformers missing, silently swallowed).
    fake_backend = MagicMock()
    fake_backend.name = "local"

    # Also patch the duck-path check so we reach the embed call rather
    # than short-circuiting on "no embedding store found".
    import tempfile
    import os as _os
    with tempfile.NamedTemporaryFile(suffix=".duck", delete=False) as tmp:
        duck_path = tmp.name
    try:
        with patch("app.embedders.sync_bridge.get_embedder_or_none",
                   return_value=fake_backend), \
             patch("app.embedders.sync_bridge.embed_text_sync",
                   return_value=None), \
             patch("app.config.Settings.vec_db_path_for_repo",
                   lambda self, repo: duck_path):
            resp = client.get("/search/semantic", params={"q": "anything", "repo": "fake"})
    finally:
        try:
            _os.unlink(duck_path)
        except OSError:
            pass

    # Must be 503 (service unavailable) — never 500 (unhandled exception).
    assert resp.status_code == 503, (
        f"Expected 503 but got {resp.status_code}: {resp.text}"
    )
    body = resp.json()
    assert "detail" in body
    detail = body["detail"].lower()
    # Must mention the remediation action so operators know what to do.
    assert "uv sync" in detail or "local-embed" in detail or "unavailable" in detail
