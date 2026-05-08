"""Tantivy BM25 lexical arm — Phase 1.1 unit tests.

Five sharp tests, ordered cheap → expensive:

1. Schema round-trip: a doc added & committed comes back with all fields.
2. BM25 ranking sanity: a document containing a rare token outranks one
   that contains only common tokens for a query on the rare token.
3. Repo isolation: searching with ``repo=A`` never returns a doc indexed
   into repo B (multi-tenant correctness).
4. Empty / whitespace queries return ``[]`` (no exceptions).
5. Reopen-after-close: closing the writer and constructing a fresh
   ``TantivyIndex`` over the same dir sees previously-committed docs.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.services.tantivy_index import TantivyIndex


@pytest.fixture()
def tmp_root(tmp_path: Path) -> Path:
    """Per-test scratch dir to host the .tantivy index folder."""
    return tmp_path


def _add(idx: TantivyIndex, *, qname: str, path: str, content: str, repo: str = "repoA") -> None:
    """Compact helper — every test calls add() with the same defaults."""
    idx.add(
        symbol_qname=qname,
        file_path=path,
        symbol_kind="Function",
        content=content,
        start_line=1,
        end_line=10,
        repo=repo,
    )


def test_schema_round_trip_returns_all_stored_fields(tmp_root: Path) -> None:
    idx = TantivyIndex(tmp_root, "repoA")
    if idx._unavailable:  # tantivy not installed in this env — skip
        pytest.skip("tantivy not available")

    _add(idx, qname="pkg.mod.foo", path="src/foo.py", content="hello world rare_token")
    idx.commit()

    hits = idx.search("rare_token", k=5, repo="repoA")
    idx.close()

    assert len(hits) == 1
    h = hits[0]
    assert h["symbol_qname"] == "pkg.mod.foo"
    assert h["file_path"] == "src/foo.py"
    assert h["symbol_kind"] == "Function"
    assert h["start_line"] == 1
    assert h["end_line"] == 10
    assert h["score"] > 0


def test_bm25_ranks_rare_token_above_common_only_doc(tmp_root: Path) -> None:
    idx = TantivyIndex(tmp_root, "repoA")
    if idx._unavailable:
        pytest.skip("tantivy not available")

    # Common-only doc: contains the common word but not the rare one.
    _add(idx, qname="pkg.common", path="src/common.py", content="setup helper utility common")
    # Rare-token doc: contains the distinctive term we will query for.
    _add(idx, qname="pkg.special", path="src/special.py",
         content="setup helper getInstallationOctokit rare_distinctive_term")
    idx.commit()

    hits = idx.search("getInstallationOctokit", k=5, repo="repoA")
    idx.close()

    assert len(hits) >= 1
    # The doc that actually contains the token must be ranked first.
    assert hits[0]["symbol_qname"] == "pkg.special"


def test_repo_filter_isolates_documents_across_repos(tmp_root: Path) -> None:
    idx_a = TantivyIndex(tmp_root, "repoA")
    idx_b = TantivyIndex(tmp_root, "repoB")
    if idx_a._unavailable or idx_b._unavailable:
        pytest.skip("tantivy not available")

    _add(idx_a, qname="repoA.fn", path="a/x.py", content="alpha beta shared_token", repo="repoA")
    idx_a.commit()
    _add(idx_b, qname="repoB.fn", path="b/y.py", content="alpha beta shared_token", repo="repoB")
    idx_b.commit()

    hits_a = idx_a.search("shared_token", k=10, repo="repoA")
    hits_b = idx_b.search("shared_token", k=10, repo="repoB")
    idx_a.close()
    idx_b.close()

    assert all(h["symbol_qname"] == "repoA.fn" for h in hits_a)
    assert all(h["symbol_qname"] == "repoB.fn" for h in hits_b)
    # And neither side should be empty — the data is there, just isolated.
    assert hits_a and hits_b


def test_empty_query_returns_empty_list(tmp_root: Path) -> None:
    idx = TantivyIndex(tmp_root, "repoA")
    if idx._unavailable:
        pytest.skip("tantivy not available")
    _add(idx, qname="pkg.x", path="x.py", content="anything")
    idx.commit()
    assert idx.search("", k=5, repo="repoA") == []
    assert idx.search("   ", k=5, repo="repoA") == []
    idx.close()


def test_reopen_sees_previously_committed_documents(tmp_root: Path) -> None:
    idx1 = TantivyIndex(tmp_root, "repoA")
    if idx1._unavailable:
        pytest.skip("tantivy not available")
    _add(idx1, qname="pkg.persistent", path="src/p.py", content="persisted_token alpha")
    idx1.commit()
    idx1.close()

    # Fresh handle, same on-disk dir — must see the prior commit.
    idx2 = TantivyIndex(tmp_root, "repoA")
    hits = idx2.search("persisted_token", k=5, repo="repoA")
    idx2.close()

    assert len(hits) == 1
    assert hits[0]["symbol_qname"] == "pkg.persistent"
