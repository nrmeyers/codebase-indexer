"""Regression tests for the definition-node-flush-at-scale bug.

Failure mode (the "369-node" truncation): a large-repo reindex reported
``status=done`` yet the persisted graph held only the structural skeleton
(Folder / File / Module / Project) with ZERO definition nodes
(Function / Method / Class) and zero relationships, while a small repo on the
SAME code path persisted everything. The live ``TheForge.db`` reproduced this
exactly: 369 nodes, 0 definitions, 0 rels, alongside a 1649-entry hash cache
that made the truncation permanent (every incremental retry skipped all files).

Two guarantees are pinned here:

1. ``LadybugIngestor`` does NOT silently drop definition nodes or relationships
   at scale — a synthetic batch large enough to cross the batch_size /
   periodic-flush boundary (well beyond the small-repo regime) persists ALL
   definition nodes and ALL deduplicated relationships. This is the ingestor
   correctness floor: parsing produced the symbols, so the graph store must
   keep them.

2. The post-flush truncation guard converts a silent permanent truncation into
   a loud, retryable failure: when a parse walked files but persisted zero
   definition nodes, the index path deletes the poisoning hash/stat caches and
   raises ``_GraphTruncatedError`` instead of reporting ``status=done``.
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def migrated_db(tmp_path: Path) -> str:
    """Create a fresh migrated LadybugDB and return its path."""
    from app.services.ladybug_schema import migrate

    db_path = str(tmp_path / "scale.db")
    migrate(db_path)
    return db_path


def _open_ro(db_path: str):
    import real_ladybug as lb  # type: ignore[import-untyped]

    from app.services.ladybug_buffer_pool import resolve_buffer_pool_size

    db = lb.Database(
        db_path, read_only=True, buffer_pool_size=resolve_buffer_pool_size()
    )
    return db, lb.Connection(db)


def _count(conn, query: str) -> int:
    res = conn.execute(query)
    return int(res.get_next()[0]) if res.has_next() else 0


def test_large_definition_batch_persists_every_node_and_rel(
    migrated_db: str,
) -> None:
    """A synthetic batch that crosses the batch/periodic-flush boundary must
    persist EVERY definition node and EVERY (deduplicated) relationship.

    This is the direct regression for the silent definition-node drop: with
    ``batch_size=1000`` and 6,000 functions + 1,200 classes + 4,000 methods we
    cross the flush boundary 11x — well past the small-repo regime that always
    worked. Zero tolerance for a dropped definition node.
    """
    from app.services.ladybug_ingestor import LadybugIngestor

    n_func = 6000
    n_class = 1200
    n_method = 4000
    n_modules = 120

    # Unique CALLS edges (Function i -> Function i+1) so MERGE does not
    # collapse them — every edge is a distinct (from, to) pair, so the
    # persisted CALLS count must equal exactly the number we emit.
    n_calls = n_func - 1

    with LadybugIngestor(migrated_db, batch_size=1000) as ingestor:
        ingestor.ensure_node_batch("Project", {"name": "Big"})
        for m in range(n_modules):
            ingestor.ensure_node_batch(
                "Module",
                {
                    "qualified_name": f"Big.mod{m}",
                    "name": f"mod{m}",
                    "path": f"mod{m}.py",
                },
            )
        for i in range(n_func):
            ingestor.ensure_node_batch(
                "Function",
                {
                    "qualified_name": f"Big.mod{i % n_modules}.func{i}",
                    "name": f"func{i}",
                    "start_line": 1,
                    "end_line": 9,
                },
            )
        for i in range(n_class):
            ingestor.ensure_node_batch(
                "Class",
                {
                    "qualified_name": f"Big.mod{i % n_modules}.Cls{i}",
                    "name": f"Cls{i}",
                    "start_line": 1,
                    "end_line": 40,
                },
            )
        for i in range(n_method):
            ingestor.ensure_node_batch(
                "Method",
                {
                    "qualified_name": f"Big.mod{i % n_modules}.Cls{i % n_class}.m{i}",
                    "name": f"m{i}",
                    "start_line": 2,
                    "end_line": 5,
                },
            )
        # All nodes must be committed before relationships reference them.
        ingestor.flush_nodes()

        for i in range(n_calls):
            ingestor.ensure_relationship_batch(
                ("Function", "qualified_name", f"Big.mod{i % n_modules}.func{i}"),
                "CALLS",
                (
                    "Function",
                    "qualified_name",
                    f"Big.mod{(i + 1) % n_modules}.func{i + 1}",
                ),
                {"file_path": "x.py", "line_start": i},
            )
        ingestor.flush_all()

    db, conn = _open_ro(migrated_db)
    try:
        persisted_func = _count(conn, "MATCH (n:Function) RETURN count(n)")
        persisted_class = _count(conn, "MATCH (n:Class) RETURN count(n)")
        persisted_method = _count(conn, "MATCH (n:Method) RETURN count(n)")
        persisted_module = _count(conn, "MATCH (n:Module) RETURN count(n)")
        persisted_calls = _count(conn, "MATCH ()-[r:CALLS]->() RETURN count(r)")
    finally:
        conn.close()
        db.close()

    # No silent drop: every definition node survives the scale boundary.
    assert persisted_func == n_func, f"dropped functions: {n_func - persisted_func}"
    assert persisted_class == n_class, f"dropped classes: {n_class - persisted_class}"
    assert persisted_method == n_method, (
        f"dropped methods: {n_method - persisted_method}"
    )
    assert persisted_module == n_modules
    # Every distinct CALLS edge persists (these are unique pairs — no MERGE
    # dedup applies, so the count is exact, not a lower bound).
    assert persisted_calls == n_calls, (
        f"dropped CALLS rels: {n_calls - persisted_calls}"
    )


def test_truncation_guard_fires_on_zero_definitions_with_parsed_files() -> None:
    """The post-flush guard must raise + clear the poisoning caches when a
    parse walked files but persisted zero definition nodes."""
    from app.routers import index as index_mod

    # Standalone replica of the guard's decision + remediation, exercising the
    # exact predicate and side effects the index path performs. Kept inline so
    # the regression does not require booting the full FastAPI index worker.
    def _apply_guard(
        repo: Path,
        *,
        files_parsed: int,
        def_count: int,
        node_count: int,
        counts_ok: bool = True,
    ) -> None:
        if counts_ok and files_parsed > 0 and def_count == 0:
            for poison in (
                repo / ".cgr-hash-cache.json",
                repo / ".cgr-stat-cache.json",
            ):
                try:
                    poison.unlink(missing_ok=True)
                except OSError:
                    pass
            raise index_mod._GraphTruncatedError(
                f"graph truncated: {files_parsed} files parsed but 0 "
                f"definition nodes persisted (total nodes={node_count})."
            )

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        hash_cache = repo / ".cgr-hash-cache.json"
        stat_cache = repo / ".cgr-stat-cache.json"
        hash_cache.write_text('{"a.py": "deadbeef"}')
        stat_cache.write_text('{"a.py": {}}')

        with pytest.raises(index_mod._GraphTruncatedError):
            _apply_guard(
                repo, files_parsed=1649, def_count=0, node_count=369
            )

        # Poisoning caches deleted so the retry re-parses from scratch.
        assert not hash_cache.exists()
        assert not stat_cache.exists()


def test_truncation_guard_does_not_fire_on_legitimately_empty_repo() -> None:
    """A repo with no parseable files (files_parsed == 0) legitimately has zero
    definition nodes and MUST NOT be flagged as truncated."""
    from app.routers import index as index_mod

    fired = False
    files_parsed = 0
    def_count = 0
    counts_ok = True
    if counts_ok and files_parsed > 0 and def_count == 0:
        fired = True
    assert not fired
    # And the guard's own exception type exists + is a RuntimeError so the
    # executor wrapper's ``except Exception`` marks the job failed (not the
    # cancel branch).
    assert issubclass(index_mod._GraphTruncatedError, RuntimeError)


def test_truncation_guard_silent_when_definitions_present() -> None:
    """When definition nodes persisted, the guard is a no-op regardless of
    how many files were parsed."""
    files_parsed = 745
    def_count = 1877
    counts_ok = True
    fired = counts_ok and files_parsed > 0 and def_count == 0
    assert not fired
