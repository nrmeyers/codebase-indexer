"""Tests for LadybugIngestor.node_count / rel_count accumulators.

Exercises the counters added to address the AttributeError that
watch_manager._blocking_partial_index raised after flush_all().

Uses a real on-disk LadybugDB (same approach as test_ladybug_pool.py)
because the counter increments are inside flush paths that interact with
the DB.
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def migrated_db(tmp_path: Path) -> str:
    """Create a fresh migrated LadybugDB and return its path."""
    from app.services.ladybug_schema import migrate

    db_path = str(tmp_path / "test.db")
    migrate(db_path)
    return db_path


def test_node_count_increments_on_flush(migrated_db: str) -> None:
    """node_count should reflect the number of successfully flushed nodes."""
    from app.services.ladybug_ingestor import LadybugIngestor

    with LadybugIngestor(migrated_db, batch_size=100) as ingestor:
        assert ingestor.node_count == 0

        ingestor.ensure_node_batch("Project", {"name": "proj-a", "root_path": "/a"})
        ingestor.ensure_node_batch("Project", {"name": "proj-b", "root_path": "/b"})
        ingestor.flush_nodes()

        assert ingestor.node_count == 2

        # A second flush adds to the running total.
        ingestor.ensure_node_batch("Project", {"name": "proj-c", "root_path": "/c"})
        ingestor.flush_nodes()

        assert ingestor.node_count == 3


def test_rel_count_increments_on_flush(migrated_db: str) -> None:
    """rel_count should reflect the number of successfully flushed relationships."""
    from app.services.ladybug_ingestor import LadybugIngestor

    with LadybugIngestor(migrated_db, batch_size=100) as ingestor:
        assert ingestor.rel_count == 0

        # Seed nodes that the relationship will reference.
        ingestor.ensure_node_batch("Project", {"name": "proj-x", "root_path": "/x"})
        ingestor.ensure_node_batch(
            "Package",
            {"qualified_name": "pkg.a", "name": "a", "path": "/x/a"},
        )
        ingestor.flush_nodes()

        ingestor.ensure_relationship_batch(
            ("Project", "name", "proj-x"),
            "CONTAINS_PACKAGE",
            ("Package", "qualified_name", "pkg.a"),
        )
        ingestor.flush_relationships()

        assert ingestor.rel_count == 1


def test_flush_all_accumulates_both_counters(migrated_db: str) -> None:
    """flush_all() should leave node_count and rel_count accessible with correct totals."""
    from app.services.ladybug_ingestor import LadybugIngestor

    with LadybugIngestor(migrated_db, batch_size=100) as ingestor:
        ingestor.ensure_node_batch("Project", {"name": "proj-y", "root_path": "/y"})
        ingestor.ensure_node_batch(
            "Package",
            {"qualified_name": "pkg.b", "name": "b", "path": "/y/b"},
        )
        ingestor.ensure_relationship_batch(
            ("Project", "name", "proj-y"),
            "CONTAINS_PACKAGE",
            ("Package", "qualified_name", "pkg.b"),
        )
        ingestor.flush_all()

        assert ingestor.node_count == 2
        assert ingestor.rel_count == 1


def test_idempotent_nodes_still_counted(migrated_db: str) -> None:
    """Nodes that already exist (idempotent upsert) should still count toward node_count."""
    from app.services.ladybug_ingestor import LadybugIngestor

    with LadybugIngestor(migrated_db, batch_size=100) as ingestor:
        for _ in range(3):
            ingestor.ensure_node_batch(
                "Project", {"name": "dup-proj", "root_path": "/dup"}
            )
        ingestor.flush_nodes()

        # All three attempts counted — MERGE semantics treat each as a success.
        assert ingestor.node_count == 3
