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
from unittest.mock import patch

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
