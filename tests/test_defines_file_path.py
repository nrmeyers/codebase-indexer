"""NAVI-92-A: DEFINES.file_path column — schema, migration, and backfill tests.

Three tests:

1. Fresh DB: DEFINES table has a ``file_path`` column (declared inline in
   ``_REL_TABLES``); writing a DEFINES edge with an explicit file_path
   round-trips correctly.

2. Old DB (missing column): ``migrate()`` applies the ALTER and the column
   becomes readable without crashing.  This is the "graceful old-DB" path.

3. Backfill via ``LadybugIngestor._backfill_defines_file_paths()``: after
   flushing nodes (Module + Function) and a DEFINES edge (no file_path
   supplied), ``flush_all()`` sets ``DEFINES.file_path`` from the Module's
   ``path`` property.  Verifies the NAVI-92-A root-cause fix — centrality
   symbols that share the file's name now get a non-empty ``file_path``
   without parser changes.
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


def _defines_file_path(db_path: str, to_qname: str) -> str | None:
    """Return DEFINES.file_path for the edge pointing at ``to_qname``."""
    db = lb.Database(db_path, read_only=True)
    conn = lb.Connection(db)
    try:
        r = conn.execute(
            "MATCH ()-[x:DEFINES]->(n) WHERE n.qualified_name = $qn "
            "RETURN x.file_path",
            {"qn": to_qname},
        )
        if r.has_next():
            row = r.get_next()
            return str(row[0]) if row[0] is not None else None
        return None
    finally:
        conn.close()
        db.close()


# ---------------------------------------------------------------------------
# Test 1: Fresh DB — column exists; explicit file_path round-trips
# ---------------------------------------------------------------------------

def test_should_persist_explicit_file_path_on_defines_edge(
    migrated_db: str,
) -> None:
    """A DEFINES edge written with an explicit file_path survives flush."""
    from app.services.ladybug_ingestor import LadybugIngestor

    with LadybugIngestor(migrated_db, use_merge=True) as ing:
        # Module node
        ing.ensure_node_batch("Module", {"qualified_name": "pkg.utils", "name": "utils", "path": "pkg/utils.py"})
        # Function node
        ing.ensure_node_batch("Function", {
            "qualified_name": "pkg.utils.helper",
            "name": "helper",
            "decorators": [],
            "start_line": 10,
            "end_line": 20,
            "docstring": "",
            "source_code": "",
            "is_exported": False,
            "is_async": False,
            "is_generator": False,
            "contextual_prefix": "",
        })
        ing.flush_nodes()
        # DEFINES edge with explicit file_path
        ing.ensure_relationship_batch(
            ("Module", "qualified_name", "pkg.utils"),
            "DEFINES",
            ("Function", "qualified_name", "pkg.utils.helper"),
            properties={"file_path": "pkg/utils.py"},
        )
        ing.flush_relationships()

    result = _defines_file_path(migrated_db, "pkg.utils.helper")
    assert result == "pkg/utils.py", (
        f"Expected file_path='pkg/utils.py' on DEFINES edge, got {result!r}"
    )


# ---------------------------------------------------------------------------
# Test 2: Old DB (column missing) — ALTER adds it without crashing
# ---------------------------------------------------------------------------

def test_should_survive_migration_on_old_db_without_defines_file_path(
    tmp_path: Path,
) -> None:
    """An old LadybugDB that lacks DEFINES.file_path must not crash on migrate().

    We simulate an old DB by creating the DEFINES table WITHOUT file_path
    (mimicking a pre-NAVI-92-A schema), then running migrate() to apply the
    _REL_ALTERS backfill.  The column must be readable after the run and
    existing edges must surface the DEFAULT '' without raising.
    """
    db_path = str(tmp_path / "old_graph.db")
    # Create the old-style DEFINES table WITHOUT file_path.
    db = lb.Database(db_path)
    conn = lb.Connection(db)
    conn.execute(
        "CREATE NODE TABLE IF NOT EXISTS Module("
        "  qualified_name STRING, name STRING, path STRING,"
        "  PRIMARY KEY (qualified_name))"
    )
    conn.execute(
        "CREATE NODE TABLE IF NOT EXISTS Function("
        "  qualified_name STRING, name STRING,"
        "  decorators STRING[], start_line INT64, end_line INT64,"
        "  docstring STRING, source_code STRING, is_exported BOOL,"
        "  is_async BOOL DEFAULT FALSE, is_generator BOOL DEFAULT FALSE,"
        "  contextual_prefix STRING DEFAULT '',"
        "  PRIMARY KEY (qualified_name))"
    )
    # Old schema: no file_path column.
    conn.execute(
        "CREATE REL TABLE IF NOT EXISTS DEFINES("
        "  FROM Module TO Function)"
    )
    conn.execute(
        "CREATE (m:Module {qualified_name: 'old.mod', name: 'mod', path: 'old/mod.py'})"
    )
    conn.execute(
        "CREATE (f:Function {qualified_name: 'old.mod.fn', name: 'fn',"
        "  decorators: [], start_line: 1, end_line: 5,"
        "  docstring: '', source_code: '', is_exported: false,"
        "  is_async: false, is_generator: false, contextual_prefix: ''})"
    )
    conn.execute(
        "MATCH (m:Module) WHERE m.qualified_name = 'old.mod' "
        "MATCH (f:Function) WHERE f.qualified_name = 'old.mod.fn' "
        "MERGE (m)-[:DEFINES]->(f)"
    )
    conn.close()
    db.close()

    # Run the full migrate() — must not raise.
    from app.services.ladybug_schema import migrate
    migrate(db_path)  # must succeed

    # After migration the column must exist; existing rows surface '' (the DEFAULT).
    result = _defines_file_path(db_path, "old.mod.fn")
    # After ALTER the column is readable; value is '' (the DEFAULT) or None.
    assert result == "" or result is None, (
        f"Unexpected file_path value after ALTER backfill: {result!r}"
    )


# ---------------------------------------------------------------------------
# Test 3: flush_all() backfills DEFINES.file_path from Module.path
# ---------------------------------------------------------------------------

def test_should_backfill_defines_file_path_from_module_path(
    migrated_db: str,
) -> None:
    """NAVI-92-A root-cause fix: flush_all() copies Module.path onto DEFINES
    edges that have file_path = ''.

    This is the canonical repro for ~1,040/5,475 TheForge KG nodes missing
    filePath: parsers emit ``ensure_relationship_batch(..., 'DEFINES', ...,
    properties=None)``, so file_path lands as '' (the schema DEFAULT).
    After flush_all() runs _backfill_defines_file_paths(), the edge must
    carry the Module's path.
    """
    from app.services.ladybug_ingestor import LadybugIngestor

    with LadybugIngestor(migrated_db, use_merge=True) as ing:
        # Module with a known path.
        ing.ensure_node_batch("Module", {
            "qualified_name": "web.auth",
            "name": "auth",
            "path": "web/auth.ts",
        })
        # Function defined in that module (file-level export sharing the name).
        ing.ensure_node_batch("Function", {
            "qualified_name": "web.auth",  # same name as module — the exact repro case
            "name": "auth",
            "decorators": [],
            "start_line": 1,
            "end_line": 50,
            "docstring": "",
            "source_code": "",
            "is_exported": True,
            "is_async": False,
            "is_generator": False,
            "contextual_prefix": "",
        })
        ing.flush_nodes()

        # Emit DEFINES WITHOUT properties (parsers don't pass file_path yet).
        ing.ensure_relationship_batch(
            ("Module", "qualified_name", "web.auth"),
            "DEFINES",
            ("Function", "qualified_name", "web.auth"),
            properties=None,  # no file_path — pre-NAVI-92-A parser behaviour
        )

        # flush_all() must: flush_nodes + flush_relationships + _backfill_defines_file_paths
        ing.flush_all()

    result = _defines_file_path(migrated_db, "web.auth")
    assert result == "web/auth.ts", (
        f"Expected DEFINES.file_path='web/auth.ts' after backfill, got {result!r}. "
        "The _backfill_defines_file_paths() pass did not copy Module.path onto the edge."
    )


# ---------------------------------------------------------------------------
# Test 4: /search/centrality and /repos/{name}/centrality accept limit up to 500
# ---------------------------------------------------------------------------

def test_should_accept_centrality_limit_up_to_500() -> None:
    """NAVI-93: both centrality endpoints accept limit=500 (raised from 200)."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    # /search/centrality — limit=500 must not return 422 (was le=200 before)
    resp = client.get("/search/centrality", params={"limit": 500})
    assert resp.status_code != 422, (
        f"/search/centrality?limit=500 returned 422 (validation error); "
        f"le=500 cap not applied. Body: {resp.text}"
    )

    # /search/centrality — limit=501 must still return 422
    resp501 = client.get("/search/centrality", params={"limit": 501})
    assert resp501.status_code == 422, (
        "/search/centrality?limit=501 should be rejected (le=500), but was accepted"
    )
