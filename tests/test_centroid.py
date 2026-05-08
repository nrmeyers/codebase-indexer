"""Tests for the per-repo topic centroid endpoint (BUC-1581).

Surgical coverage (4 tests, per the brief):

    1. Centroid math correctness — synthetic embeddings whose mean is
       analytically known should round-trip through the endpoint without
       distortion.
    2. 404 when the ``.duck`` file is absent (repo never indexed).
    3. 503 when the ``.duck`` file exists but the centrality table or
       embedding column is empty.
    4. Cache hit — a repeat call serves from the in-process LRU and reports
       ``cache_age_seconds > 0``.

Tests skip cleanly when the sibling ``codebase_rag`` package is not
importable in the current environment, mirroring ``test_pagerank.py``.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import centroid as centroid_service

client = TestClient(app)


# Embedding dim must match the production constant. Hard-coded here so the
# test file has no upstream import that would force a sibling-package
# dependency at collection time.
_DIM = 768


@pytest.fixture(autouse=True)
def _isolate_centroid_cache():
    """Each test starts with a clean cache so cache hits/misses are deterministic."""
    centroid_service.clear_cache()
    yield
    centroid_service.clear_cache()


def _can_open_duck() -> bool:
    """Return True when ``codebase_rag.storage.vector_store.open_or_create``
    is importable. Used to skip rather than fail in CI environments that
    install only this repo without the sibling ``code-graph-rag`` editable.
    """
    try:
        from codebase_rag.storage.vector_store import open_or_create  # noqa: F401

        return True
    except ImportError:
        return False


def _seed_centrality_and_embeddings(
    duck_path: Path,
    rows: list[tuple[str, float, list[float]]],
) -> bool:
    """Write ``(qname, pagerank, embedding)`` tuples into a fresh ``.duck``.

    Bypasses ``bulk_insert`` so our test embeddings are NOT L2-normalised
    — that lets us assert exact mean-pool arithmetic without doing the
    normalisation in the test fixture.

    Returns True on success; False when the sibling package is missing
    so callers can ``pytest.skip`` cleanly.
    """
    if not _can_open_duck():
        return False

    from codebase_rag.storage.vector_store import open_or_create  # type: ignore[import-untyped]

    conn = open_or_create(str(duck_path))
    try:
        # Wipe any pre-existing rows so repeat fixture calls are deterministic.
        conn.execute("DELETE FROM embeddings")
        conn.execute("DELETE FROM centrality")

        # Add embedding_v2 column the same way ensure_v2_schema does — we
        # write to it directly so this test pins the production read path.
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
        for qname, pr, vec in rows:
            assert len(vec) == _DIM, "fixture vectors must be 768-dim"
            # Insert into embeddings.embedding_v2 (the production read path
            # prefers v2 when present).
            conn.execute(
                "INSERT INTO embeddings "
                "(qualified_name, embedding, embedding_v2, symbol_type, "
                " file_path, start_line, end_line, indexed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    qname,
                    vec,  # ``embedding`` (legacy) — populate so search paths still work
                    vec,  # ``embedding_v2`` — what centroid will read
                    "Function",
                    f"{qname.replace('.', '/')}.py",
                    1,
                    10,
                    now,
                ),
            )
            conn.execute(
                "INSERT INTO centrality "
                "(qualified_name, pagerank, updated_at) VALUES (?, ?, ?)",
                (qname, float(pr), now),
            )
    finally:
        conn.close()

    return True


# ---------------------------------------------------------------------------
# 1. Centroid math correctness — synthetic embeddings → expected mean.
# ---------------------------------------------------------------------------


def test_should_return_arithmetic_mean_when_centroid_computed_from_known_embeddings(
    tmp_path: Path,
) -> None:
    """Three embeddings: all-ones, all-twos, all-threes (768-dim each).
    Mean is all-twos. The endpoint must return exactly that vector — proves
    the COALESCE-then-mean-pool path is wired correctly.
    """
    if not _can_open_duck():
        pytest.skip("codebase_rag sibling pkg not importable in this environment")

    slug = "fixture__centroid_math"
    duck_path = tmp_path / f"{slug}.duck"

    # Three symbols whose embeddings have a clean closed-form mean.
    rows = [
        ("pkg.alpha.fn", 1.0, [1.0] * _DIM),
        ("pkg.bravo.fn", 0.6, [2.0] * _DIM),
        ("pkg.charlie.fn", 0.2, [3.0] * _DIM),
    ]
    assert _seed_centrality_and_embeddings(duck_path, rows)

    with patch("app.routers.repos.settings") as mock_settings:
        mock_settings.vec_db_path_for_repo.return_value = str(duck_path)
        resp = client.get(f"/repos/{slug}/centroid?k=20")

    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["repo"] == slug
    assert body["k"] == 3  # only 3 rows seeded; effective k clamped down
    assert body["cache_age_seconds"] == 0  # fresh compute
    assert "computed_at" in body and body["computed_at"].endswith("Z")

    centroid = body["centroid"]
    assert isinstance(centroid, list)
    assert len(centroid) == _DIM
    # Mean of (1, 2, 3) per-dim = 2.0; allow float tolerance for L2-norm
    # rounding inside DuckDB's FLOAT[N] storage.
    for x in centroid:
        assert x == pytest.approx(2.0, abs=1e-5), (
            f"expected 2.0 across all dims, got {x}"
        )


# ---------------------------------------------------------------------------
# 2. 404 on unknown repo (no .duck file).
# ---------------------------------------------------------------------------


def test_should_return_404_when_repo_has_never_been_indexed(tmp_path: Path) -> None:
    """A repo whose ``.duck`` file does not exist must yield a 404. This
    distinguishes "we don't know about this repo" (TheForge should index it)
    from "we know about it but the centroid isn't ready yet" (TheForge
    should poll — that's the 503 case).
    """
    slug = "never_indexed__phase25"
    missing_duck = tmp_path / f"{slug}.duck"
    assert not missing_duck.exists()

    with patch("app.routers.repos.settings") as mock_settings:
        mock_settings.vec_db_path_for_repo.return_value = str(missing_duck)
        resp = client.get(f"/repos/{slug}/centroid")

    assert resp.status_code == 404
    detail = resp.json().get("detail", {})
    assert detail.get("code") == "repo_not_indexed"


# ---------------------------------------------------------------------------
# 3. 503 when the .duck exists but no embeddings are present.
# ---------------------------------------------------------------------------


def test_should_return_503_when_repo_indexed_but_no_embeddings_yet(
    tmp_path: Path,
) -> None:
    """A ``.duck`` file with a populated centrality table but zero embedding
    rows for the top-k qnames must yield 503 with ``code=centroid_unavailable``.
    This is the window between PageRank persistence and the embed pass
    completing on a freshly-indexed repo. TheForge polls through this state.
    """
    if not _can_open_duck():
        pytest.skip("codebase_rag sibling pkg not importable in this environment")

    from codebase_rag.storage.vector_store import open_or_create  # type: ignore[import-untyped]

    slug = "fixture__no_embeddings"
    duck_path = tmp_path / f"{slug}.duck"

    # Seed centrality only — leave embeddings empty.
    conn = open_or_create(str(duck_path))
    try:
        conn.execute("DELETE FROM centrality")
        conn.execute("DELETE FROM embeddings")
        now = int(time.time())
        conn.execute(
            "INSERT INTO centrality "
            "(qualified_name, pagerank, updated_at) VALUES (?, ?, ?)",
            ("pkg.alpha.fn", 1.0, now),
        )
    finally:
        conn.close()

    with patch("app.routers.repos.settings") as mock_settings:
        mock_settings.vec_db_path_for_repo.return_value = str(duck_path)
        resp = client.get(f"/repos/{slug}/centroid")

    assert resp.status_code == 503
    detail = resp.json().get("detail", {})
    assert detail.get("code") == "centroid_unavailable"


# ---------------------------------------------------------------------------
# 4. Cache hit — repeat call returns cached result with age > 0.
# ---------------------------------------------------------------------------


def test_should_serve_from_cache_when_called_twice_with_same_params(
    tmp_path: Path,
) -> None:
    """Two requests for the same (slug, k) within the TTL window must hit
    the in-process cache. We assert the second response has
    ``cache_age_seconds > 0`` and the centroid matches byte-for-byte —
    proving the cache path doesn't accidentally recompute and round-trip
    differently through DuckDB's FLOAT[N] storage.
    """
    if not _can_open_duck():
        pytest.skip("codebase_rag sibling pkg not importable in this environment")

    slug = "fixture__cache_hit"
    duck_path = tmp_path / f"{slug}.duck"

    rows = [
        ("pkg.alpha.fn", 1.0, [0.5] * _DIM),
        ("pkg.bravo.fn", 0.5, [1.5] * _DIM),
    ]
    assert _seed_centrality_and_embeddings(duck_path, rows)

    with patch("app.routers.repos.settings") as mock_settings:
        mock_settings.vec_db_path_for_repo.return_value = str(duck_path)

        first = client.get(f"/repos/{slug}/centroid?k=20")
        # Force a measurable age delta on the cache hit. The TTL is 1h so
        # this is well within the cache window.
        time.sleep(1.1)
        second = client.get(f"/repos/{slug}/centroid?k=20")

    assert first.status_code == 200
    assert second.status_code == 200

    first_body = first.json()
    second_body = second.json()

    assert first_body["cache_age_seconds"] == 0, "first call should be a cold compute"
    assert second_body["cache_age_seconds"] >= 1, (
        "second call should serve from cache with measurable age "
        f"(got {second_body['cache_age_seconds']})"
    )
    # ``computed_at`` should be identical on the cache hit — proves we
    # served the cached entry rather than recomputing.
    assert first_body["computed_at"] == second_body["computed_at"]
    # Centroid byte-for-byte equality on the cache hit.
    assert first_body["centroid"] == second_body["centroid"]
