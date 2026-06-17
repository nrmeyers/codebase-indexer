"""Reproduce the 369-node truncation: a full structure+definition+rel ingest.

The live TheForge graph after a status=done reindex held only 369 nodes
(Folder 191 / File 144 / Module 33 / Project 1) and ZERO relationships, while
the semantic store held ~7k symbol embeddings. i.e. the structural pass
persisted but every Function/Method/Class node AND every relationship was lost.

This test mirrors the real GraphUpdater ingest order — structure nodes flushed
periodically, definition nodes added, then flush_all() writes definitions +
relationships — and asserts the definitions and relationships actually persist.
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


def _counts(db_path: str) -> tuple[int, int]:
    db = lb.Database(db_path)
    conn = lb.Connection(db)
    n = conn.execute("MATCH (n) RETURN count(n)")
    node_count = int(n.get_next()[0]) if n.has_next() else 0
    r = conn.execute("MATCH ()-[x]->() RETURN count(x)")
    rel_count = int(r.get_next()[0]) if r.has_next() else 0
    conn.close()
    return node_count, rel_count


def test_definitions_and_relationships_persist(migrated_db: str) -> None:
    """A full ingest must persist Function nodes AND relationships, not just structure."""
    from app.services.ladybug_ingestor import LadybugIngestor

    n_modules = 30
    funcs_per_module = 40  # 1200 functions

    with LadybugIngestor(migrated_db, batch_size=500) as ingestor:
        ingestor.ensure_node_batch("Project", {"name": "TheForge"})

        # --- structural pass (flushed periodically, like _process_files) ---
        for m in range(n_modules):
            mod_qn = f"TheForge.mod_{m}"
            ingestor.ensure_node_batch("Module", {"qualified_name": mod_qn})
            ingestor.ensure_relationship_batch(
                ("Project", "name", "TheForge"),
                "CONTAINS_MODULE",
                ("Module", "qualified_name", mod_qn),
            )
            # mimic periodic structural flush
            ingestor.flush_nodes()

        # --- definition pass: Function nodes + DEFINES + CALLS ---
        for m in range(n_modules):
            mod_qn = f"TheForge.mod_{m}"
            for f in range(funcs_per_module):
                fn_qn = f"{mod_qn}.func_{f}"
                ingestor.ensure_node_batch(
                    "Function", {"qualified_name": fn_qn, "name": f"func_{f}"}
                )
                ingestor.ensure_relationship_batch(
                    ("Module", "qualified_name", mod_qn),
                    "DEFINES",
                    ("Function", "qualified_name", fn_qn),
                )
                if f > 0:
                    ingestor.ensure_relationship_batch(
                        ("Function", "qualified_name", fn_qn),
                        "CALLS",
                        ("Function", "qualified_name", f"{mod_qn}.func_{f - 1}"),
                    )

        # final flush (the real flush_all on GraphUpdater.run line 459)
        ingestor.flush_all()

    expected_nodes = 1 + n_modules + (n_modules * funcs_per_module)
    node_count, rel_count = _counts(migrated_db)

    assert node_count == expected_nodes, (
        f"truncated: {node_count} nodes persisted, expected {expected_nodes} "
        f"(definition nodes lost — the 369-node bug)"
    )
    assert rel_count > 0, "all relationships were lost (rel_count=0 — the 369-node bug)"
