"""Tests for ``app.services.ladybug_pool``.

BUC-1571 — read-only consumers (search, repos, health, context-bundle,
index/stats) must coexist with a concurrent writer rather than
contending on the exclusive file lock.

These tests exercise *real* LadybugDB files because the lock semantics
live entirely in the C library and would not surface against a mock.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.ladybug_pool import open_read_conn, open_rw_conn


@pytest.fixture()
def seeded_db(tmp_path: Path) -> str:
    """Spin up a tiny real LadybugDB so we can exercise the file lock."""
    import ladybug as lb  # type: ignore[import-untyped]
    from codebase_rag.services.ladybug_schema import migrate

    db_path = str(tmp_path / "lock.db")
    migrate(db_path)

    db = lb.Database(db_path)
    conn = lb.Connection(db)
    conn.execute("CREATE (p:Project {name: 'lockrepo'})")
    conn.execute(
        "CREATE (f:File {path: 'lockrepo/a.py', name: 'a.py', extension: '.py'})"
    )
    # Drop the writer reference so subsequent open_rw_conn calls in the
    # tests below can re-acquire the lock cleanly.
    conn = None  # noqa: F841
    db = None  # noqa: F841
    import gc as _gc
    _gc.collect()
    return db_path


def test_read_during_write_does_not_deadlock(seeded_db: str) -> None:
    """A read-only connection opens cleanly while a writer holds the file.

    Regression for BUC-1571: previously the read path called
    ``lb.Database(path)`` (RW mode), which raised
    ``IO exception: Could not set lock on file`` whenever a re-index was
    running.  With ``read_only=True`` the shared-lock path is taken and
    both connections coexist.
    """
    # Hold the writer for the duration of this test.
    writer_db, writer_conn = open_rw_conn(seeded_db)
    try:
        # The fix: read-only mode coexists with the held writer.
        # Without ``read_only=True`` this raises
        # ``IO exception: Could not set lock on file`` when the writer
        # belongs to another process (the BUC-1571 reproduction).  In a
        # single-process test ladybug's writer-vs-reader semantics still
        # exercise the shared-lock code path, so the read-only call
        # below must not raise.
        reader_db, reader_conn = open_read_conn(seeded_db)
        try:
            res = reader_conn.execute("MATCH (n) RETURN count(n) AS cnt")
            assert res.has_next()
            cnt = int(res.get_next()[0])
            # We seeded 1 Project + 1 File, but ladybug also tracks
            # internal nodes — only assert ≥ 2 to keep the test robust.
            assert cnt >= 2
        finally:
            reader_conn.close()
            del reader_conn, reader_db
    finally:
        writer_conn.close()
        del writer_conn, writer_db
        import gc as _gc
        _gc.collect()


def test_search_files_endpoint_works_during_active_write(
    seeded_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: GET /search/files succeeds while a writer holds the lock.

    Without the BUC-1571 fix this returns 503 with::

        IO exception: Could not set lock on file ...
        See https://docs.ladybugdb.com/concurrency
    """
    # Point the resolver at our seeded DB regardless of repo slug.
    monkeypatch.setattr(
        "app.routers.search._resolve_db_path",
        lambda repo: seeded_db,
    )

    # Hold the writer lock for the lifetime of the request.
    writer_db, writer_conn = open_rw_conn(seeded_db)
    try:
        with TestClient(app) as client:
            resp = client.get("/search/files", params={"repo": "lockrepo"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # The seeded File row must come back even with the writer holding
        # the lock — that is the entire point of the read-only fix.
        paths = [f["path"] for f in body["files"]]
        assert "lockrepo/a.py" in paths
    finally:
        writer_conn.close()
        del writer_conn, writer_db
        import gc as _gc
        _gc.collect()
