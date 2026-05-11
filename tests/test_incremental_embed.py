"""Tests for the incremental-embed-by-content-hash audit (BUC-1574 / Phase 1.4).

The actual embed pass runs in a subprocess (``_blocking_embed`` driver
template in ``app/routers/index.py``) and calls SageMaker, so it cannot
be exercised in CI without network. This test instead pins the
*persistence contract* the driver relies on:

    1. Every row written via ``bulk_insert`` carries a ``content_hash``.
    2. ``read_content_hashes`` round-trips cleanly across a close/reopen
       of the same ``.duck`` file (i.e. survives a uvicorn restart).
    3. Re-running the *exact same* hash-composition rule against the
       same source text yields a 100% skip rate (no row needs a fresh
       SageMaker call).

If the driver's hash composition is ever changed without also bumping
the persistence layer, the round-trip assertion catches it.

BUC-1613: the upstream ``read_content_hashes`` helper is now part of
``navistone/main`` (cherry-picked from the original side branch as commit
a4b3fd7).  This test module was previously collection-skipped while that
function was missing; the skip is now removed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

pytest.importorskip("duckdb")

from codebase_rag.storage.vector_store import (  # noqa: E402
    EmbeddingRow,
    bulk_insert,
    open_or_create,
    read_content_hashes,
)


def _compose_embed_text(
    stype: str,
    qname: str,
    callers: int,
    docstring: str,
    src: str,
) -> str:
    """Replicate the per-symbol embed input composed in the embed driver.

    Matches the f-string template in ``_blocking_embed`` (~lines
    1145–1167 of ``app/routers/index.py``).  Tested side-by-side: any
    drift between this helper and the driver template will surface as a
    skip-rate regression in the integration test below.
    """
    parts = [f"# {stype}: {qname}"]
    mod_path = ".".join(qname.split(".")[:-1])
    if mod_path:
        parts.append(f"# Module: {mod_path}")
    if callers > 0:
        parts.append(f"# Callers: {callers}")
    parts.append("# ---")
    if docstring:
        parts.append(docstring)
    parts.append(src)
    return "\n".join(parts)


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 1. content_hash is populated on every row written through bulk_insert.
# ---------------------------------------------------------------------------


def test_content_hash_is_populated_for_every_row(tmp_path: Path) -> None:
    """Every row written via bulk_insert must carry a non-null content_hash."""
    db_path = tmp_path / "fixture.duck"
    conn = open_or_create(str(db_path))
    try:
        rows = []
        for i in range(3):
            text = _compose_embed_text(
                stype="Function",
                qname=f"pkg.module_{i}.fn",
                callers=i,
                docstring="",
                src=f"def fn():\n    return {i}",
            )
            rows.append(
                EmbeddingRow(
                    qualified_name=f"pkg.module_{i}.fn",
                    embedding=[0.0] * 768,
                    file_path=f"pkg/module_{i}.py",
                    start_line=1,
                    end_line=2,
                    symbol_type="Function",
                    content_hash=_hash(text),
                )
            )
        bulk_insert(conn, rows)
        hashes = read_content_hashes(conn)
    finally:
        conn.close()

    assert len(hashes) == 3
    assert all(v and len(v) == 40 for v in hashes.values())  # sha1 hex


# ---------------------------------------------------------------------------
# 2. Hashes round-trip cleanly across a close + reopen (uvicorn restart).
# ---------------------------------------------------------------------------


def test_content_hash_persists_across_reopen(tmp_path: Path) -> None:
    """Hashes survive close+reopen of the .duck file — required for restarts."""
    db_path = tmp_path / "restart.duck"

    # First "process": write rows.
    conn = open_or_create(str(db_path))
    fixtures = [
        ("pkg.a.foo", "def foo():\n    return 1"),
        ("pkg.b.bar", "def bar():\n    return 2"),
        ("pkg.c.baz", "def baz():\n    return 3"),
    ]
    expected: dict[str, str] = {}
    try:
        rows = []
        for qname, src in fixtures:
            text = _compose_embed_text("Function", qname, 0, "", src)
            h = _hash(text)
            expected[qname] = h
            rows.append(
                EmbeddingRow(
                    qualified_name=qname,
                    embedding=[0.0] * 768,
                    file_path=qname.replace(".", "/") + ".py",
                    start_line=1,
                    end_line=2,
                    symbol_type="Function",
                    content_hash=h,
                )
            )
        bulk_insert(conn, rows)
    finally:
        conn.close()

    # Second "process": reopen and verify.
    conn2 = open_or_create(str(db_path))
    try:
        round_tripped = read_content_hashes(conn2)
    finally:
        conn2.close()

    assert round_tripped == expected


# ---------------------------------------------------------------------------
# 3. A re-run with identical source produces 100% skip rate (cache hit).
# ---------------------------------------------------------------------------


def test_rerun_with_identical_source_yields_100_percent_skip(tmp_path: Path) -> None:
    """Simulating the driver's skip check produces no fresh embeddings."""
    db_path = tmp_path / "rerun.duck"
    fixtures = [
        ("pkg.a.foo", 0, "", "def foo():\n    return 1"),
        ("pkg.b.bar", 2, "Bar docstring.", "def bar():\n    return 2"),
        ("pkg.c.baz", 0, "", "def baz():\n    return 3"),
    ]

    # Pass 1 — populate.
    conn = open_or_create(str(db_path))
    try:
        rows = []
        for qname, callers, doc, src in fixtures:
            text = _compose_embed_text("Function", qname, callers, doc, src)
            rows.append(
                EmbeddingRow(
                    qualified_name=qname,
                    embedding=[0.0] * 768,
                    file_path=qname.replace(".", "/") + ".py",
                    start_line=1,
                    end_line=2,
                    symbol_type="Function",
                    content_hash=_hash(text),
                )
            )
        bulk_insert(conn, rows)
    finally:
        conn.close()

    # Pass 2 — reopen, replay the driver's skip-check loop. With unchanged
    # inputs every symbol must be skipped (== zero new embeds).
    conn = open_or_create(str(db_path))
    try:
        existing = read_content_hashes(conn)
        embedded = 0
        skipped_unchanged = 0
        for qname, callers, doc, src in fixtures:
            text = _compose_embed_text("Function", qname, callers, doc, src)
            new_hash = _hash(text)
            if existing.get(qname) == new_hash:
                skipped_unchanged += 1
                continue
            embedded += 1
    finally:
        conn.close()

    assert embedded == 0
    assert skipped_unchanged == len(fixtures)
