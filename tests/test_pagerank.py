"""Tests for Phase 1.5 PageRank centrality.

Surgical coverage (4 tests, per the brief):
    1. ``compute_pagerank`` on a small in-memory graph — caller of a heavily-
       called sink should rank above a leaf with no callers.
    2. Empty graph → empty dict, no exception.
    3. ``GET /repos/{name}/centrality`` returns the expected envelope shape
       after a fixture write to the ``.duck`` ``centrality`` table.
    4. ``GET /repos/{name}/centrality`` degrades to ``{symbols: []}`` (not
       404) when the table is empty, so TheForge can poll a freshly-indexed
       repo without special-casing.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.pagerank import compute_pagerank, normalize_pagerank


client = TestClient(app)


# ---------------------------------------------------------------------------
# Pure-core tests — no DB, no FastAPI
# ---------------------------------------------------------------------------


def test_should_rank_caller_of_hub_above_isolated_leaf_when_graph_has_clear_hub() -> None:
    """A 10-node graph where ``hub`` is called by 5 distinct callers should
    score ``hub`` above any leaf node that participates in zero edges as a
    callee. PageRank's whole purpose is to surface high-incoming-degree
    nodes; a regression here would mean the algorithm wiring broke.
    """
    # Build: callers c1..c5 → hub; orphans o1..o4 are isolated nodes.
    edges = [(f"c{i}", "hub") for i in range(1, 6)]
    nodes = [f"c{i}" for i in range(1, 6)] + ["hub"] + [f"o{i}" for i in range(1, 5)]

    raw = compute_pagerank(edges, nodes=nodes)
    assert raw, "expected non-empty PageRank result"
    assert "hub" in raw

    # Hub should outrank every caller (callers point AT hub but receive nothing).
    for c in (f"c{i}" for i in range(1, 6)):
        assert raw["hub"] > raw[c], f"hub ({raw['hub']}) not greater than {c} ({raw[c]})"

    # And hub should outrank every orphan.
    for o in (f"o{i}" for i in range(1, 5)):
        assert raw["hub"] > raw[o]

    # Normalisation contract: floor=0, ceiling=1.
    norm = normalize_pagerank(raw)
    assert max(norm.values()) == pytest.approx(1.0)
    assert min(norm.values()) == pytest.approx(0.0)
    # Hub is the most central, so it should normalise to the ceiling.
    assert norm["hub"] == pytest.approx(1.0)


def test_should_return_empty_dict_when_graph_has_no_edges_or_nodes() -> None:
    """Empty input must not raise — the indexer hot path swallows pagerank
    failures, but a noisy ValueError still pollutes logs and triggers the
    ``pagerank.failed`` warning path unnecessarily.
    """
    assert compute_pagerank([], nodes=None) == {}
    assert compute_pagerank([], nodes=[]) == {}
    # normalize on empty input is also a contract surface.
    assert normalize_pagerank({}) == {}


# ---------------------------------------------------------------------------
# Endpoint tests — use a real DuckDB ``.duck`` file via the sibling helper
# when available; skip cleanly otherwise so this test file works in CI
# environments where ``code-graph-rag`` editable install is missing.
# ---------------------------------------------------------------------------


def _write_centrality_fixture(duck_path: Path, rows: list[tuple[str, float]]) -> bool:
    """Best-effort fixture writer. Returns True on success.

    Uses the sibling package's helpers so the schema stays in lockstep with
    the production code path. Skips (returns False) if the package is not
    importable in the current environment.
    """
    try:
        from codebase_rag.storage.vector_store import (  # type: ignore[import-untyped]
            open_or_create,
            write_centrality,
            clear_centrality,
        )
    except ImportError:
        return False

    conn = open_or_create(str(duck_path))
    try:
        clear_centrality(conn)
        write_centrality(conn, dict(rows))
    finally:
        conn.close()
    return True


def test_should_return_top_n_centrality_when_table_is_populated(tmp_path: Path) -> None:
    """End-to-end: write a small centrality table, hit the endpoint, assert
    shape + ordering. Skips when the sibling write helper isn't available
    so this test never blocks a CI image that installs only this repo.
    """
    slug = "fixture__phase15"
    duck_path = tmp_path / f"{slug}.duck"

    fixture = [
        ("pkg.module.alpha", 1.0),
        ("pkg.module.bravo", 0.6),
        ("pkg.module.charlie", 0.2),
    ]
    if not _write_centrality_fixture(duck_path, fixture):
        pytest.skip("codebase_rag sibling pkg not importable in this environment")

    with patch("app.routers.repos.settings") as mock_settings:
        mock_settings.vec_db_path_for_repo.return_value = str(duck_path)
        resp = client.get(f"/repos/{slug}/centrality?limit=20")

    assert resp.status_code == 200
    body = resp.json()
    assert "symbols" in body
    assert len(body["symbols"]) == 3
    # Ordered by pagerank DESC.
    qnames = [s["qname"] for s in body["symbols"]]
    assert qnames == ["pkg.module.alpha", "pkg.module.bravo", "pkg.module.charlie"]
    # Shape contract — each row has exactly the two declared fields.
    for sym in body["symbols"]:
        assert set(sym.keys()) == {"qname", "centrality"}
        assert isinstance(sym["centrality"], float)


def test_should_return_empty_array_when_centrality_not_yet_computed(tmp_path: Path) -> None:
    """A repo whose ``.duck`` file does not exist (or exists without a
    centrality row) must yield ``{symbols: []}`` and a 200 status — never a
    404. This is the graceful-degradation path TheForge polls during the
    window between ingest start and PageRank's post-embed compute phase.
    """
    slug = "never_indexed__phase15"
    missing_duck = tmp_path / f"{slug}.duck"
    assert not missing_duck.exists()

    with patch("app.routers.repos.settings") as mock_settings:
        mock_settings.vec_db_path_for_repo.return_value = str(missing_duck)
        resp = client.get(f"/repos/{slug}/centrality")

    assert resp.status_code == 200
    body = resp.json()
    assert body["symbols"] == []
    # last_computed_at must be null when no .duck file exists.
    assert body.get("last_computed_at") is None


# ---------------------------------------------------------------------------
# LE-32 regression: codebase_rag.storage.centrality.compute_pagerank
# must use per-label queries (not the invalid OR-label syntax)
# ---------------------------------------------------------------------------


def test_should_compute_pagerank_via_per_label_queries(tmp_path: Path) -> None:
    """LE-32 regression guard.

    The original ``compute_pagerank`` used ``MATCH (n) WHERE (n:Function OR
    n:Method)`` which the LadybugDB Cypher parser rejects with a parse
    exception.  After the fix the implementation uses separate MATCH queries
    per label and merges the results in Python.

    Strategy: mock ``lb.Connection.execute`` to return label-appropriate
    mock results for Function and Method node queries and the four CALLS edge
    queries.  Assert that ``compute_pagerank`` returns a non-empty dict and
    that the highest-ranked symbol is the one targeted by the most CALLS.
    """
    import types
    from unittest.mock import MagicMock, call

    # Import after fix — if the old OR-syntax is still present the import
    # itself won't fail, but the mocked execute would need to handle the
    # combined query which we explicitly do not provide.
    from codebase_rag.storage.centrality import compute_pagerank as cgr_compute_pagerank

    # Build a tiny call graph:
    #   fn_a, fn_b → fn_hub (fn_hub is the most-called function)
    #   meth_x → meth_hub
    fn_nodes = ["fn_a", "fn_b", "fn_hub"]
    meth_nodes = ["meth_x", "meth_hub"]

    # The fix issues 2 node queries (Function, Method) + 4 edge queries
    # (Function→Function, Function→Method, Method→Function, Method→Method).
    # We must return the right data for each call.
    _call_count = [0]

    def _fake_execute(query: str):
        """Return mock results keyed on which label the query targets."""
        mock_result = MagicMock()
        q = query.strip()

        if "MATCH (n:Function)" in q:
            rows = [[qn] for qn in fn_nodes]
        elif "MATCH (n:Method)" in q:
            rows = [[qn] for qn in meth_nodes]
        elif "Function)-[:CALLS]->(t:Function)" in q:
            # fn_a→fn_hub, fn_b→fn_hub
            rows = [["fn_a", "fn_hub"], ["fn_b", "fn_hub"]]
        elif "Function)-[:CALLS]->(t:Method)" in q:
            rows = []
        elif "Method)-[:CALLS]->(t:Function)" in q:
            rows = []
        elif "Method)-[:CALLS]->(t:Method)" in q:
            # meth_x → meth_hub
            rows = [["meth_x", "meth_hub"]]
        else:
            rows = []

        remaining = list(rows)

        mock_result.has_next.side_effect = lambda: bool(remaining)
        mock_result.get_next.side_effect = lambda: remaining.pop(0)
        return mock_result

    fake_conn = MagicMock()
    fake_conn.execute.side_effect = _fake_execute
    fake_conn.close.return_value = None

    fake_db = MagicMock()

    with patch("ladybug.Database", return_value=fake_db), \
         patch("ladybug.Connection", return_value=fake_conn):
        scores = cgr_compute_pagerank("/fake/path.db")

    assert scores, "expected non-empty PageRank scores"
    # fn_hub has 2 incoming CALLS edges → highest score
    assert "fn_hub" in scores
    assert scores["fn_hub"] == pytest.approx(1.0)  # normalised to max
    # meth_hub has 1 incoming edge — lower than fn_hub but > meth_x
    assert scores.get("meth_hub", 0) > scores.get("meth_x", 0)


# ---------------------------------------------------------------------------
# POST /repos/{name}/recompute-centrality — LE-32 manual trigger endpoint
# ---------------------------------------------------------------------------


def test_should_return_200_and_score_count_when_recompute_succeeds(tmp_path: Path) -> None:
    """LE-32: POST /repos/{name}/recompute-centrality should write rows and
    return a JSON body with ``scores_written > 0``.
    """
    slug = "fixture__recompute"
    db_file = tmp_path / f"{slug}.db"
    duck_file = tmp_path / f"{slug}.duck"
    db_file.write_bytes(b"")  # must exist for 404 guard

    fixture_scores = {"fn.alpha": 1.0, "fn.beta": 0.5}

    with patch("app.routers.repos.settings") as mock_settings, \
         patch("codebase_rag.storage.centrality.compute_pagerank",
               return_value=fixture_scores) as mock_pr, \
         patch("codebase_rag.storage.vector_store.open_or_create",
               return_value=MagicMock()) as mock_open, \
         patch("codebase_rag.storage.vector_store.clear_centrality") as mock_clear, \
         patch("codebase_rag.storage.vector_store.write_centrality",
               return_value=len(fixture_scores)) as mock_write:
        mock_settings.db_path_for_repo.return_value = str(db_file)
        mock_settings.vec_db_path_for_repo.return_value = str(duck_file)

        resp = client.post(f"/repos/{slug}/recompute-centrality")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["repo"] == slug
    assert body["scores_written"] == 2
    mock_pr.assert_called_once()
    mock_clear.assert_called_once()
    mock_write.assert_called_once()


def test_should_return_404_when_db_not_indexed(tmp_path: Path) -> None:
    """LE-32 endpoint: missing ``.db`` file → 404 (not 500)."""
    slug = "not_indexed__recompute"
    missing_db = tmp_path / f"{slug}.db"
    assert not missing_db.exists()

    with patch("app.routers.repos.settings") as mock_settings:
        mock_settings.db_path_for_repo.return_value = str(missing_db)
        resp = client.post(f"/repos/{slug}/recompute-centrality")

    assert resp.status_code == 404


def test_should_return_200_with_zero_scores_when_graph_has_no_calls(tmp_path: Path) -> None:
    """LE-32 endpoint: zero CALLS edges → 200 with scores_written=0, not 500."""
    slug = "no_calls__recompute"
    db_file = tmp_path / f"{slug}.db"
    db_file.write_bytes(b"")

    with patch("app.routers.repos.settings") as mock_settings, \
         patch("codebase_rag.storage.centrality.compute_pagerank",
               return_value={}):
        mock_settings.db_path_for_repo.return_value = str(db_file)
        mock_settings.vec_db_path_for_repo.return_value = str(tmp_path / f"{slug}.duck")

        resp = client.post(f"/repos/{slug}/recompute-centrality")

    assert resp.status_code == 200
    body = resp.json()
    assert body["scores_written"] == 0
    assert "no CALLS" in body["message"] or "0" in body["message"]
