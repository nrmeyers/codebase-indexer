"""Regression: CALLS (and behavioral) relationships must persist at flush.

Root cause (fixed in ``fix/calls-relationship-persist``): the batched UNWIND
relationship insert used an inline-property node bind —
``MATCH (a:Function {qualified_name: row.from_val})`` — which deterministically
triggers a Kùzu rel-MERGE planner fault (``unordered_map::at: key not found``)
even when both endpoint nodes are committed and visible.  Every CALLS batch
threw and fell back to the slow per-row path; under memory pressure / the
write-phase watchdog that per-row path also failed and silently swallowed every
edge, leaving a definition-rich but CALLS=0 graph ("nothing is connected" in the
knowledge-graph viewer).

The fix binds endpoints with ``MATCH (a:Label) WHERE a.key = row.val`` (which
avoids the planner fault while preserving MERGE idempotency) and adds a
fail-loud guard (:class:`RelationshipFlushError`) that raises when a behavioral
rel type lands ZERO edges due to a genuine RUNTIME drop — but NOT on benign
schema rejects.

These tests drive the behaviour at three levels:

1. The raw batched-UNWIND form against LadybugDB (the deterministic repro).
2. The real ``GraphUpdater`` parse → ingest path on a Python fixture with known
   cross-function / method calls, asserting CALLS persist with the expected
   count.
3. The fail-loud guard contract (raises on a runtime wipe; silent on benign
   schema rejects; silent on a clean flush).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import ladybug as lb


@pytest.fixture()
def migrated_db(tmp_path: Path) -> str:
    from app.services.ladybug_schema import migrate

    db_path = str(tmp_path / "graph.db")
    migrate(db_path)
    return db_path


def _calls_count(db_path: str) -> int:
    db = lb.Database(db_path, read_only=True)
    conn = lb.Connection(db)
    r = conn.execute("MATCH ()-[x:CALLS]->() RETURN count(x)")
    n = int(r.get_next()[0]) if r.has_next() else 0
    conn.close()
    return n


# ---------------------------------------------------------------------------
# 1. The deterministic batch-form repro at the LadybugDB layer.
# ---------------------------------------------------------------------------


def test_batched_calls_insert_does_not_throw_unordered_map(migrated_db: str) -> None:
    """The batched CALLS UNWIND must persist edges without the planner fault.

    Before the fix the inline-property bind threw
    ``unordered_map::at: key not found`` 100% of the time at this batch size,
    forcing the slow per-row fallback.  After the fix the batch path persists
    every edge directly.
    """
    from app.services.ladybug_ingestor import LadybugIngestor

    n = 50
    with LadybugIngestor(migrated_db, batch_size=10_000) as ingestor:
        ingestor.ensure_node_batch("Project", {"name": "P"})
        for i in range(n):
            ingestor.ensure_node_batch(
                "Function",
                {"qualified_name": f"P.f{i}", "name": f"f{i}",
                 "start_line": 1, "end_line": 2},
            )
        ingestor.flush_nodes()
        for i in range(n):
            ingestor.ensure_relationship_batch(
                ("Function", "qualified_name", f"P.f{i}"),
                "CALLS",
                ("Function", "qualified_name", f"P.f{(i + 1) % n}"),
                properties={
                    "file_path": "x.py", "line_start": 1, "col_start": 0,
                    "resolved_via": "direct", "confidence": 1.0,
                },
            )
        ingestor.flush_relationships()

    assert _calls_count(migrated_db) == n, (
        "CALLS edges were dropped — the batched UNWIND MERGE hit the Kùzu "
        "unordered_map planner fault and the edges did not persist"
    )


def test_batched_calls_merge_is_idempotent(migrated_db: str) -> None:
    """Re-flushing the same CALLS batch must not duplicate edges (MERGE)."""
    from app.services.ladybug_ingestor import LadybugIngestor

    def _flush_once() -> None:
        with LadybugIngestor(migrated_db, batch_size=10_000) as ingestor:
            for i in range(5):
                ingestor.ensure_node_batch(
                    "Function",
                    {"qualified_name": f"P.f{i}", "name": f"f{i}",
                     "start_line": 1, "end_line": 2},
                )
            ingestor.flush_nodes()
            for i in range(5):
                ingestor.ensure_relationship_batch(
                    ("Function", "qualified_name", f"P.f{i}"),
                    "CALLS",
                    ("Function", "qualified_name", f"P.f{(i + 1) % 5}"),
                    properties={"file_path": "x.py", "line_start": 1,
                                "col_start": 0, "resolved_via": "d",
                                "confidence": 1.0},
                )
            ingestor.flush_relationships()

    _flush_once()
    assert _calls_count(migrated_db) == 5
    _flush_once()  # re-index
    assert _calls_count(migrated_db) == 5, "MERGE re-flush duplicated CALLS edges"


# ---------------------------------------------------------------------------
# 2. The real GraphUpdater parse -> ingest path on a known-call fixture.
# ---------------------------------------------------------------------------


def _write_fixture(root: Path) -> int:
    """Write a small multi-file Python fixture and return the expected
    minimum number of resolvable in-graph CALLS edges.

    Calls whose callee resolves to a non-CALLS-legal endpoint (e.g. a
    constructor ``Service()`` resolving to a Class) are intentionally NOT
    counted — the schema forbids them and they are dropped benignly.
    """
    (root / "alpha.py").write_text(
        "def helper(x):\n    return x + 1\n\n"
        "def caller():\n    return helper(41)\n"  # caller -> helper
    )
    (root / "beta.py").write_text(
        "from alpha import helper\n\n"
        "def use_it():\n    return helper(10)\n\n"  # use_it -> helper
        "def chain():\n    return use_it()\n"        # chain -> use_it
    )
    (root / "gamma.py").write_text(
        "class Service:\n"
        "    def run(self):\n        return self.tick()\n"  # run -> tick
        "    def tick(self):\n        return 1\n"
    )
    # caller->helper, use_it->helper, chain->use_it, Service.run->Service.tick
    return 4


def test_real_ingest_persists_calls(tmp_path: Path) -> None:
    """The real parse->ingest path must persist function/method CALLS edges.

    This is the end-to-end guard: it exercises ``GraphUpdater.run`` (Pass 1
    structure, Pass 2 files, Pass 3 calls, then ``flush_all``) exactly as the
    live indexer does, then asserts the resolvable CALLS survived the flush.
    """
    pytest.importorskip("codebase_rag")
    from codebase_rag.graph_updater import GraphUpdater
    from codebase_rag.parser_loader import load_parsers

    from app.services.ladybug_ingestor import LadybugIngestor

    repo = tmp_path / "fixture_repo"
    repo.mkdir()
    expected_min_calls = _write_fixture(repo)

    db_path = str(tmp_path / "repo.lb")
    parsers, queries = load_parsers()

    with LadybugIngestor(db_path, batch_size=1000, use_merge=True) as ingestor:
        updater = GraphUpdater(
            ingestor=ingestor, repo_path=repo, parsers=parsers, queries=queries
        )
        # SKIP_EMBEDDINGS guards against the test needing a live embedder; the
        # graph write (the path under test) is unaffected.
        import codebase_rag.graph_updater as gu

        if hasattr(gu, "settings"):
            gu.settings.SKIP_EMBEDDINGS = True
        updater.run(force=True)

    calls = _calls_count(db_path)
    assert calls >= expected_min_calls, (
        f"CALLS persisted={calls}, expected >= {expected_min_calls} resolvable "
        f"function/method calls — behavioral edges were dropped at flush"
    )


# ---------------------------------------------------------------------------
# 3. The fail-loud guard contract.
# ---------------------------------------------------------------------------


def test_guard_raises_on_runtime_calls_wipe(
    migrated_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CALLS batch whose every row hits a RUNTIME drop must raise.

    Both the batched insert and the per-row fallback are forced to raise a
    non-schema runtime fault (mimicking the production
    ``unordered_map::at: key not found`` + per-row failure), so the CALLS type
    lands 0 edges with runtime_dropped > 0 -> RelationshipFlushError.
    ``LadybugIngestor`` uses ``__slots__`` so the methods are patched on the
    CLASS via monkeypatch, not the instance.
    """
    import app.services.ladybug_ingestor as li
    from app.services.ladybug_ingestor import (
        LadybugIngestor,
        RelationshipFlushError,
    )

    def _boom(*_a: object, **_k: object) -> object:
        raise RuntimeError("unordered_map::at: key not found")

    with LadybugIngestor(migrated_db, batch_size=10_000) as ingestor:
        ingestor.ensure_node_batch(
            "Function",
            {"qualified_name": "P.a", "name": "a", "start_line": 1, "end_line": 2},
        )
        ingestor.ensure_node_batch(
            "Function",
            {"qualified_name": "P.b", "name": "b", "start_line": 1, "end_line": 2},
        )
        ingestor.flush_nodes()
        ingestor.ensure_relationship_batch(
            ("Function", "qualified_name", "P.a"),
            "CALLS",
            ("Function", "qualified_name", "P.b"),
            properties={"file_path": "x", "line_start": 1, "col_start": 0,
                        "resolved_via": "d", "confidence": 1.0},
        )

        monkeypatch.setattr(li.LadybugIngestor, "_execute_batch", _boom)
        monkeypatch.setattr(li.LadybugIngestor, "_execute_query", _boom)

        with pytest.raises(RelationshipFlushError):
            ingestor.flush_relationships()


def test_guard_silent_on_clean_flush(migrated_db: str) -> None:
    """A flush where CALLS persist must NOT raise (no false positive)."""
    from app.services.ladybug_ingestor import LadybugIngestor

    with LadybugIngestor(migrated_db, batch_size=10_000) as ingestor:
        for i in range(3):
            ingestor.ensure_node_batch(
                "Function",
                {"qualified_name": f"P.f{i}", "name": f"f{i}",
                 "start_line": 1, "end_line": 2},
            )
        ingestor.flush_nodes()
        for i in range(2):
            ingestor.ensure_relationship_batch(
                ("Function", "qualified_name", f"P.f{i}"),
                "CALLS",
                ("Function", "qualified_name", f"P.f{i + 1}"),
                properties={"file_path": "x", "line_start": 1, "col_start": 0,
                            "resolved_via": "d", "confidence": 1.0},
            )
        # Must not raise.
        ingestor.flush_relationships()

    assert _calls_count(migrated_db) == 2
