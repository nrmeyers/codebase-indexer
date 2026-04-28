"""Tests for the in-memory BM25 lexical index (Plan E).

Each test builds a real ``.duck`` via ``open_or_create`` + ``bulk_insert``
inside ``tmp_path`` so we exercise the same DuckDB stack the production
router uses.  ``importorskip`` keeps the suite green on machines without
the optional semantic stack installed.
"""
from __future__ import annotations

import os
import time

import pytest

bm25s = pytest.importorskip("bm25s")
pytest.importorskip("duckdb")

from app.services.bm25_index import BM25Service, _tokenise  # noqa: E402


# Lazy import — only when a test actually builds a fixture DB.
def _make_db(path: str, rows: list[tuple[str, str]]) -> None:
    """Populate a fresh ``.duck`` at ``path`` with ``(qname, type)`` rows."""
    from codebase_rag.storage.vector_store import (
        EmbeddingRow,
        bulk_insert,
        open_or_create,
    )

    conn = open_or_create(path)
    try:
        embed_rows = [
            EmbeddingRow(
                qualified_name=qn,
                embedding=[0.0] * 768,
                file_path=f"src/{qn.replace('.', '/')}.py",
                start_line=1,
                end_line=10,
                symbol_type=stype,
            )
            for qn, stype in rows
        ]
        bulk_insert(conn, embed_rows)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# tokeniser
# ---------------------------------------------------------------------------


def test_should_tokenise_dotted_and_underscored_names() -> None:
    assert _tokenise("myapp.utils.retry_with_backoff") == [
        "myapp",
        "utils",
        "retry",
        "with",
        "backoff",
    ]
    assert _tokenise("FooHandler") == ["foohandler"]
    assert _tokenise("") == []
    # Underscore wrappers are stripped; the inner alpha run survives.
    assert _tokenise("__init__") == ["init"]
    # Pure-numeric / single-char runs are filtered by the regex.
    assert _tokenise("a 1 22 ab") == ["ab"]


# ---------------------------------------------------------------------------
# BM25Service.search
# ---------------------------------------------------------------------------


def test_should_return_empty_when_repo_has_no_vec_db(tmp_path) -> None:
    svc = BM25Service()
    missing = str(tmp_path / "nope.duck")
    assert svc.search(missing, "anything") == []


def test_should_handle_empty_query_string(tmp_path) -> None:
    svc = BM25Service()
    db = str(tmp_path / "x.duck")
    _make_db(db, [("myapp.utils.retry", "Function")])
    assert svc.search(db, "") == []
    assert svc.search(db, "   ") == []


def test_should_return_results_when_query_matches_qualified_name(tmp_path) -> None:
    svc = BM25Service()
    db = str(tmp_path / "x.duck")
    _make_db(
        db,
        [
            ("myapp.utils.retry_with_backoff", "Function"),
            ("myapp.handlers.foo_handler", "Function"),
            ("myapp.models.user", "Class"),
        ],
    )

    results = svc.search(db, "retry_with_backoff", k=10)
    assert results, "expected at least one BM25 hit"
    top = results[0][0]
    assert top == "myapp.utils.retry_with_backoff"

    results2 = svc.search(db, "foo_handler", k=10)
    assert results2[0][0] == "myapp.handlers.foo_handler"


def test_should_invalidate_cache_when_vec_db_mtime_changes(tmp_path) -> None:
    svc = BM25Service()
    db = str(tmp_path / "x.duck")
    _make_db(db, [("alpha.one", "Function")])

    first = svc.search(db, "alpha", k=5)
    assert first and first[0][0] == "alpha.one"

    # Bump mtime forward — must be greater than the cached value or the cache
    # would (correctly) consider itself fresh.  Filesystems can have 1s mtime
    # resolution so we explicitly set a future timestamp.
    future = time.time() + 5
    os.utime(db, (future, future))

    # Rewrite contents so the cache rebuild surfaces a different doc set.
    _make_db(db, [("beta.two", "Function")])
    os.utime(db, (future, future))  # second write may reset mtime, lock it again

    second = svc.search(db, "beta", k=5)
    assert second and second[0][0] == "beta.two"

    # Internal cache should have refreshed — the cached mtime must equal the
    # value we forced on disk (or the latest mtime after the rebuild).
    cached = svc._cache.get(db)
    assert cached is not None
    assert cached[2] == os.path.getmtime(db)
