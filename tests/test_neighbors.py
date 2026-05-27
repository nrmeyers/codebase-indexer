"""Tests for the embedding neighbors + clusters endpoints (LE-158 Stream C).

Coverage (per the brief):

    1. /neighbors returns K results sorted descending by score, excluding
       the seed symbol itself.
    2. /neighbors with an unknown fqn → empty list (repo IS indexed, the
       symbol just isn't embedded — distinct from a missing-store 404).
    3. /neighbors and /clusters return 404 when the ``.duck`` store is absent.
    4. /clusters partitions every input fqn into exactly one cluster (the
       union of all returned fqns equals the seeded set).

Tests skip cleanly when the sibling ``codebase_rag`` package is not
importable, mirroring ``test_centroid.py``.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

_DIM = 768


def _can_open_duck() -> bool:
    try:
        from codebase_rag.storage.vector_store import open_or_create  # noqa: F401

        return True
    except ImportError:
        return False


def _seed_embeddings(
    duck_path: Path, rows: list[tuple[str, list[float]]]
) -> bool:
    """Write ``(qname, embedding)`` tuples into a fresh ``.duck`` store.

    Writes both ``embedding`` and ``embedding_v2`` so the production
    COALESCE read path is exercised. Returns False when the sibling
    package is unavailable so callers can skip.
    """
    if not _can_open_duck():
        return False

    from codebase_rag.storage.vector_store import open_or_create  # type: ignore[import-untyped]

    conn = open_or_create(str(duck_path))
    try:
        conn.execute("DELETE FROM embeddings")
        existing_cols = {
            r[0]
            for r in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'embeddings'"
            ).fetchall()
        }
        if "embedding_v2" not in existing_cols:
            conn.execute(
                f"ALTER TABLE embeddings ADD COLUMN embedding_v2 FLOAT[{_DIM}]"
            )

        now = int(time.time())
        for qname, vec in rows:
            assert len(vec) == _DIM, "fixture vectors must be 768-dim"
            conn.execute(
                "INSERT INTO embeddings "
                "(qualified_name, embedding, embedding_v2, symbol_type, "
                " file_path, start_line, end_line, indexed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    qname,
                    vec,
                    vec,
                    "Function",
                    f"{qname.replace('.', '/')}.py",
                    1,
                    10,
                    now,
                ),
            )
    finally:
        conn.close()
    return True


def _axis_vec(dim_on: int, magnitude: float = 1.0) -> list[float]:
    """Unit-ish vector with one hot dimension — gives controllable cosine."""
    vec = [0.0] * _DIM
    vec[dim_on] = magnitude
    return vec


# ---------------------------------------------------------------------------
# 1. /neighbors — top-K sorted desc, seed excluded.
# ---------------------------------------------------------------------------


def test_should_return_k_neighbors_sorted_desc_excluding_seed(
    tmp_path: Path,
) -> None:
    if not _can_open_duck():
        pytest.skip("codebase_rag sibling pkg not importable in this environment")

    slug = "fixture__neighbors_sorted"
    duck_path = tmp_path / f"{slug}.duck"

    # seed points along dim 0. near1 mostly dim 0 (high cosine), near2 less,
    # far points along dim 1 (cosine ~0).
    rows = [
        ("pkg.seed", _axis_vec(0, 1.0)),
        ("pkg.near1", [0.9 if i == 0 else (0.1 if i == 1 else 0.0) for i in range(_DIM)]),
        ("pkg.near2", [0.5 if i == 0 else (0.5 if i == 1 else 0.0) for i in range(_DIM)]),
        ("pkg.far", _axis_vec(1, 1.0)),
    ]
    assert _seed_embeddings(duck_path, rows)

    with patch("app.routers.repos.settings") as mock_settings:
        mock_settings.vec_db_path_for_repo.return_value = str(duck_path)
        resp = client.get(f"/repos/{slug}/neighbors?fqn=pkg.seed&k=2")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) == 2  # k=2

    fqns = [item["fqn"] for item in body]
    assert "pkg.seed" not in fqns, "seed must be excluded from its own neighbors"
    # near1 (cosine ~0.99) ranks above near2 (cosine ~0.71).
    assert fqns[0] == "pkg.near1"
    assert fqns[1] == "pkg.near2"
    # Sorted strictly descending by score.
    scores = [item["score"] for item in body]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > scores[1]


# ---------------------------------------------------------------------------
# 2. /neighbors — unknown fqn → empty list (store present).
# ---------------------------------------------------------------------------


def test_should_return_empty_list_when_fqn_unknown(tmp_path: Path) -> None:
    if not _can_open_duck():
        pytest.skip("codebase_rag sibling pkg not importable in this environment")

    slug = "fixture__neighbors_unknown"
    duck_path = tmp_path / f"{slug}.duck"
    assert _seed_embeddings(duck_path, [("pkg.only", _axis_vec(0))])

    with patch("app.routers.repos.settings") as mock_settings:
        mock_settings.vec_db_path_for_repo.return_value = str(duck_path)
        resp = client.get(f"/repos/{slug}/neighbors?fqn=pkg.does_not_exist&k=5")

    assert resp.status_code == 200, resp.text
    assert resp.json() == []


# ---------------------------------------------------------------------------
# 3. 404 when the .duck store is absent.
# ---------------------------------------------------------------------------


def test_should_return_404_for_neighbors_when_repo_not_indexed(
    tmp_path: Path,
) -> None:
    slug = "never_indexed__neighbors"
    missing = tmp_path / f"{slug}.duck"
    assert not missing.exists()

    with patch("app.routers.repos.settings") as mock_settings:
        mock_settings.vec_db_path_for_repo.return_value = str(missing)
        resp = client.get(f"/repos/{slug}/neighbors?fqn=pkg.x")

    assert resp.status_code == 404
    assert resp.json().get("detail", {}).get("code") == "repo_not_indexed"


def test_should_return_404_for_clusters_when_repo_not_indexed(
    tmp_path: Path,
) -> None:
    slug = "never_indexed__clusters"
    missing = tmp_path / f"{slug}.duck"
    assert not missing.exists()

    with patch("app.routers.repos.settings") as mock_settings:
        mock_settings.vec_db_path_for_repo.return_value = str(missing)
        resp = client.get(f"/repos/{slug}/clusters?n=4")

    assert resp.status_code == 404
    assert resp.json().get("detail", {}).get("code") == "repo_not_indexed"


# ---------------------------------------------------------------------------
# 4. /clusters — every input fqn lands in exactly one cluster.
# ---------------------------------------------------------------------------


def test_should_partition_all_fqns_into_clusters(tmp_path: Path) -> None:
    if not _can_open_duck():
        pytest.skip("codebase_rag sibling pkg not importable in this environment")

    slug = "fixture__clusters_partition"
    duck_path = tmp_path / f"{slug}.duck"

    # Two well-separated groups along orthogonal axes.
    rows = [
        ("grpA.one", _axis_vec(0, 1.0)),
        ("grpA.two", [0.95 if i == 0 else (0.05 if i == 5 else 0.0) for i in range(_DIM)]),
        ("grpA.three", [0.9 if i == 0 else (0.1 if i == 7 else 0.0) for i in range(_DIM)]),
        ("grpB.one", _axis_vec(1, 1.0)),
        ("grpB.two", [0.95 if i == 1 else (0.05 if i == 9 else 0.0) for i in range(_DIM)]),
        ("grpB.three", [0.9 if i == 1 else (0.1 if i == 11 else 0.0) for i in range(_DIM)]),
    ]
    assert _seed_embeddings(duck_path, rows)
    seeded_fqns = {qn for qn, _ in rows}

    with patch("app.routers.repos.settings") as mock_settings:
        mock_settings.vec_db_path_for_repo.return_value = str(duck_path)
        resp = client.get(f"/repos/{slug}/clusters?n=2")

    assert resp.status_code == 200, resp.text
    clusters = resp.json()
    assert isinstance(clusters, list)
    assert len(clusters) >= 1

    # Union of all cluster fqns == seeded set, with no symbol in two clusters.
    all_fqns: list[str] = []
    for c in clusters:
        assert "cluster_id" in c and "label" in c and "fqns" in c
        assert c["label"] in c["fqns"]  # representative is a member
        all_fqns.extend(c["fqns"])

    assert set(all_fqns) == seeded_fqns, "every symbol must be assigned"
    assert len(all_fqns) == len(seeded_fqns), "no symbol may appear twice"

    # cluster_id contiguous from 0, sorted by size desc.
    ids = [c["cluster_id"] for c in clusters]
    assert ids == list(range(len(clusters)))
    sizes = [len(c["fqns"]) for c in clusters]
    assert sizes == sorted(sizes, reverse=True)
