"""Phase 1.3 — code-specific embedding model A/B path.

Four targeted tests covering the behaviour that ships in this PR:
    1. E5BaseV2Embedder default path returns a 768-dim vector via the
       upstream SageMaker wrapper.
    2. BgeCodeV1Embedder.embed() returns None when the v2 endpoint env vars
       are unset, and emits the missing-endpoint WARN exactly once.
    3. ``get_embedder()`` factory dispatches to the correct concrete class.
    4. Search reads ``embedding_v2`` when ``EMBEDDING_MODEL_ACTIVE=bge-code-v1``
       (in-memory DuckDB asserting the v2 column is queried).
"""
from __future__ import annotations

import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from app.services import embedder as emb_mod


@pytest.fixture(autouse=True)
def _clear_factory_cache() -> None:
    """Reset the lru_cache so env-driven tests see a fresh embedder."""
    emb_mod.get_embedder.cache_clear()
    emb_mod.reset_v2_warning_for_tests()
    yield
    emb_mod.get_embedder.cache_clear()


def test_e5_base_v2_embedder_returns_768_dim_vector() -> None:
    """E5 path delegates to the upstream SageMakerEmbedder verbatim."""
    fake_vec = [0.01] * 768
    fake_sm = MagicMock()
    fake_sm.embed.return_value = fake_vec

    with patch(
        "app.services.sagemaker_embedder.get_sagemaker_embedder",
        return_value=fake_sm,
    ):
        e5 = emb_mod.E5BaseV2Embedder()
        result = e5.embed("def foo(): return 1")

    assert result is not None
    assert len(result) == 768
    assert e5.cost_calls == 1
    assert e5.cost_tokens > 0
    fake_sm.embed.assert_called_once_with("def foo(): return 1")


def test_bge_code_v1_embedder_returns_none_when_endpoint_unset(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """v2 endpoint absent → embed() is None and exactly one WARN is emitted."""
    # Ensure no v2 env vars leak in from the host environment.
    for key in (
        "SAGEMAKER_BGE_CODE_URL",
        "SAGEMAKER_BGE_CODE_ENDPOINT",
    ):
        os.environ.pop(key, None)

    with caplog.at_level(logging.WARNING, logger="app.services.embedder"):
        bge = emb_mod.BgeCodeV1Embedder()
        first = bge.embed("def foo(): pass")
        second = bge.embed("def bar(): pass")

    assert first is None
    assert second is None
    # Cost counters must not advance on misses.
    assert bge.cost_calls == 0

    warn_msgs = [
        r.message for r in caplog.records
        if "SAGEMAKER_BGE_CODE_ENDPOINT" in r.message
    ]
    assert len(warn_msgs) == 1, (
        f"expected exactly one missing-endpoint WARN, got {len(warn_msgs)}: {warn_msgs}"
    )


def test_get_embedder_factory_dispatches_to_correct_class() -> None:
    """Factory returns the right concrete class for each canonical model id."""
    e5 = emb_mod.get_embedder(emb_mod.MODEL_E5_BASE_V2)
    bge = emb_mod.get_embedder(emb_mod.MODEL_BGE_CODE_V1)

    assert isinstance(e5, emb_mod.E5BaseV2Embedder)
    assert isinstance(bge, emb_mod.BgeCodeV1Embedder)
    assert e5.model_name == "e5-base-v2"
    assert bge.model_name == "bge-code-v1"

    with pytest.raises(ValueError, match="Unknown embedding model"):
        emb_mod.get_embedder("text-embedding-ada-002")


def test_search_reads_embedding_v2_when_active(tmp_path) -> None:
    """When EMBEDDING_MODEL_ACTIVE=bge-code-v1, search hits embedding_v2.

    Builds a tiny in-memory-style DuckDB file, populates one row with a
    distinguishable v2 vector, and verifies the v2-aware search path
    returns it (proving the SQL touched the v2 column).
    """
    duckdb = pytest.importorskip("duckdb")

    db_path = tmp_path / "v2_search.duck"
    conn = duckdb.connect(str(db_path))
    try:
        # Minimal schema mirror — enough to exercise the v2 query path.
        conn.execute(
            """
            CREATE TABLE embeddings (
                qualified_name TEXT PRIMARY KEY,
                embedding      FLOAT[768],
                symbol_type    TEXT,
                file_path      TEXT,
                start_line     INTEGER,
                end_line       INTEGER,
                indexed_at     BIGINT
            )
            """
        )
        # Phase 1.3 migration applies the new columns.
        emb_mod.ensure_v2_schema(conn)
        assert emb_mod.has_v2_column(conn)

        # v1 (legacy) embedding is a "wrong" direction; v2 matches the query.
        wrong = [0.0] * 767 + [1.0]      # unit along axis 767
        right = [1.0] + [0.0] * 767      # unit along axis 0
        query = [1.0] + [0.0] * 767      # identical to the v2 vector

        conn.execute(
            "INSERT INTO embeddings VALUES (?, ?::FLOAT[768], ?, ?, ?, ?, ?, ?, ?)",
            (
                "pkg.module.func_v2_match",
                right,                   # legacy column also matches — keeps
                "Function",              # the test focused on v2 read path,
                "pkg/module.py",         # not on column-isolation semantics.
                10,
                20,
                0,
                wrong,                   # embedding_v2 — the column under test
                "bge-code-v1",
            ),
        )
        # Update so embedding_v2 is the one matching `query` (cosine 1.0)
        # while `embedding` does not — proves search read v2 specifically.
        conn.execute(
            "UPDATE embeddings SET embedding = ?::FLOAT[768], "
            "embedding_v2 = ?::FLOAT[768] WHERE qualified_name = ?",
            (wrong, right, "pkg.module.func_v2_match"),
        )

        with patch.dict(os.environ, {"EMBEDDING_MODEL_ACTIVE": "bge-code-v1"}):
            assert emb_mod.is_v2_active() is True
            results = emb_mod.search_similar_v2(conn, query, k=5)

        assert len(results) == 1
        top = results[0]
        assert top.qualified_name == "pkg.module.func_v2_match"
        # Cosine of identical unit vectors is 1.0 — proves v2 was the column read.
        assert top.score == pytest.approx(1.0, abs=1e-5)
    finally:
        conn.close()
