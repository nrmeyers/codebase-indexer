"""CI-3: LadybugDB schema migration (DEV-1171).

Declares all node tables and relationship tables that match the
Code-Graph-RAG schema. Must be run once before any ingestion.
Safe to call on an existing DB — every DDL uses ``IF NOT EXISTS`` guards
and every column added after the initial cut is mirrored by an idempotent
``ALTER TABLE … ADD`` so that re-indexing an old DB picks up new columns
without a drop-and-recreate.

Embeddings are stored in per-repo numpy files alongside the DB file
(see ``vector_store.py``), not in LadybugDB. This avoids the chicken-and-egg
problem where opening a DB with a persisted vector index requires the VECTOR
extension pre-loaded, but the extension can only be loaded after the DB is
opened.

Schema layout:
    Node tables
        Project, Package, Folder, File, Module, Class, Function, Method,
        Interface, Enum, ExternalPackage.

    Relationship tables
        CONTAINS_FILE, CONTAINS_FOLDER, CONTAINS_PACKAGE, CONTAINS_MODULE,
        DEFINES, DEFINES_METHOD, CALLS, IMPORTS, INHERITS, IMPLEMENTS,
        OVERRIDES, BELONGS_TO, REBINDS, RE_EXPORTS.

Migration completeness contract (BUC-1621):
    Every CREATE NODE/REL TABLE statement runs unconditionally on every
    startup — ``IF NOT EXISTS`` is the safety net.  Every ALTER also runs
    unconditionally; duplicate-column errors are swallowed.  After all DDL
    has executed, ``_audit_schema()`` logs the present/absent state of every
    expected table at INFO level so drift between the declared schema and
    the on-disk DB is visible in service logs.
"""

from __future__ import annotations

import real_ladybug as lb
from loguru import logger

# ---------------------------------------------------------------------------
# Node table definitions
# ---------------------------------------------------------------------------
# Order matters only indirectly: rel tables below reference these node
# tables, so node DDL is always executed first in migrate().
_NODE_TABLES: list[str] = [
    """CREATE NODE TABLE IF NOT EXISTS Project(
        name STRING,
        root_path STRING,
        PRIMARY KEY (name)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Package(
        qualified_name STRING,
        name STRING,
        path STRING,
        PRIMARY KEY (qualified_name)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Folder(
        path STRING,
        name STRING,
        PRIMARY KEY (path)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS File(
        path STRING,
        name STRING,
        extension STRING,
        PRIMARY KEY (path)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Module(
        qualified_name STRING,
        name STRING,
        path STRING,
        PRIMARY KEY (qualified_name)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Class(
        qualified_name STRING,
        name STRING,
        decorators STRING[],
        start_line INT64,
        end_line INT64,
        docstring STRING,
        is_exported BOOL,
        PRIMARY KEY (qualified_name)
    )""",
    # BUC-1602: Function and Method carry is_async / is_generator flags so
    # downstream consumers can filter async vs sync call sites without
    # re-parsing source. Defaulted to FALSE for legacy rows; populated by
    # ``ingest_method`` (utils.py) and ``ingest_function`` (function_ingest.py).
    # Without these columns declared, every flush silently fails with
    # "Cannot find property is_async" — which is exactly the BUC-1621 symptom
    # for Method nodes on existing DBs.
    """CREATE NODE TABLE IF NOT EXISTS Function(
        qualified_name STRING,
        name STRING,
        decorators STRING[],
        start_line INT64,
        end_line INT64,
        docstring STRING,
        source_code STRING,
        is_exported BOOL,
        is_async BOOL DEFAULT FALSE,
        is_generator BOOL DEFAULT FALSE,
        contextual_prefix STRING DEFAULT '',
        PRIMARY KEY (qualified_name)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Method(
        qualified_name STRING,
        name STRING,
        decorators STRING[],
        start_line INT64,
        end_line INT64,
        docstring STRING,
        source_code STRING,
        is_exported BOOL,
        is_async BOOL DEFAULT FALSE,
        is_generator BOOL DEFAULT FALSE,
        contextual_prefix STRING DEFAULT '',
        PRIMARY KEY (qualified_name)
    )""",
    # Interface / Enum mirror Class's shape — C# emits the full property
    # set (decorators, start_line, docstring, is_exported) on these node
    # types.  Without the columns declared, every flush silently drops the
    # node with a "Binder exception: Cannot find property decorators"
    # error and stalls ingestion with log spam.
    """CREATE NODE TABLE IF NOT EXISTS Interface(
        qualified_name STRING,
        name STRING,
        decorators STRING[],
        start_line INT64,
        end_line INT64,
        docstring STRING,
        is_exported BOOL,
        PRIMARY KEY (qualified_name)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Enum(
        qualified_name STRING,
        name STRING,
        decorators STRING[],
        start_line INT64,
        end_line INT64,
        docstring STRING,
        is_exported BOOL,
        PRIMARY KEY (qualified_name)
    )""",
    # ExternalPackage uses ``name`` as its natural identifier (e.g.
    # "NSwag.AspNetCore") — code-graph-rag's C# parser emits ``name`` as the
    # only key.  Matching PK to the parser's contract here avoids a "Create
    # node expects primary key qualified_name" error on every external-dep
    # reference, which would poison every IMPORTS-edge batch.
    """CREATE NODE TABLE IF NOT EXISTS ExternalPackage(
        name STRING,
        qualified_name STRING,
        PRIMARY KEY (name)
    )""",
]

# ---------------------------------------------------------------------------
# Relationship table definitions
# ---------------------------------------------------------------------------
# Each REL TABLE may have multiple FROM/TO pairs because LadybugDB requires
# declaring every valid endpoint combination up front (unlike some graph
# DBs that allow ad-hoc typing at edge-insert time).
_REL_TABLES: list[str] = [
    """CREATE REL TABLE IF NOT EXISTS CONTAINS_FILE(
        FROM Project TO File,
        FROM Package TO File,
        FROM Folder TO File
    )""",
    """CREATE REL TABLE IF NOT EXISTS CONTAINS_FOLDER(
        FROM Project TO Folder,
        FROM Folder TO Folder
    )""",
    """CREATE REL TABLE IF NOT EXISTS CONTAINS_PACKAGE(
        FROM Project TO Package,
        FROM Package TO Package
    )""",
    """CREATE REL TABLE IF NOT EXISTS CONTAINS_MODULE(
        FROM Project TO Module,
        FROM Package TO Module
    )""",
    """CREATE REL TABLE IF NOT EXISTS DEFINES(
        FROM Module TO Class,
        FROM Module TO Function,
        FROM Module TO Interface,
        FROM Module TO Enum
    )""",
    """CREATE REL TABLE IF NOT EXISTS DEFINES_METHOD(
        FROM Class TO Method
    )""",
    # BUC-1603 + BUC-1609: CALLS edges carry call-site provenance (file +
    # line + column from BUC-1603) and resolver provenance (resolved_via +
    # confidence from BUC-1609) so downstream consumers (blast-radius,
    # agent context bundles, mergeAndRank) can show "this call happens at
    # <file>:<line>" without re-parsing source AND can deprioritize
    # low-confidence bindings (e.g. trie-suffix fallbacks, import-star
    # wildcards) when ranking results.  The columns are nullable from
    # the reader's POV — when a row was written before this migration ran
    # it surfaces as the schema DEFAULT (see ``_REL_ALTERS`` below for the
    # backfill defaults).
    #
    # ``resolved_via`` taxonomy (see ``call_resolver`` for the canonical
    # constants):
    #   - ``'exact'``     — fully-resolved qname (direct import, same-module,
    #                       type-bound method, super, builtin, IIFE)
    #   - ``'heuristic'`` — name matched within scope but ambiguity existed
    #                       (trie-suffix fallback picked best candidate)
    #   - ``'wildcard'``  — resolved through ``from foo import *``
    #   - ``'fallback'``  — couldn't resolve, edge written against best-effort
    #                       External node
    #   - ``'rebound'``   — RESERVED for BUC-1611 (method rebinding); never
    #                       emitted yet
    #   - ``'scip'``      — RESERVED for BUC-1615 (scip-typescript); never
    #                       emitted yet
    #   - ``'unknown'``   — DEFAULT for pre-BUC-1609 rows
    #
    # ``confidence`` is a DOUBLE in [0.0, 1.0].  Suggested mapping per
    # resolved_via: exact=1.0, heuristic=0.6, wildcard=0.5, fallback=0.2,
    # unknown=1.0 (don't penalize pre-existing edges; backfill default).
    """CREATE REL TABLE IF NOT EXISTS CALLS(
        FROM Function TO Function,
        FROM Function TO Method,
        FROM Method TO Function,
        FROM Method TO Method,
        FROM Module TO Function,
        FROM Module TO Method,
        file_path STRING DEFAULT '',
        line_start INT64 DEFAULT 0,
        col_start INT64 DEFAULT 0,
        resolved_via STRING DEFAULT 'unknown',
        confidence DOUBLE DEFAULT 1.0
    )""",
    """CREATE REL TABLE IF NOT EXISTS IMPORTS(
        FROM Module TO Module,
        FROM Module TO ExternalPackage
    )""",
    """CREATE REL TABLE IF NOT EXISTS INHERITS(
        FROM Class TO Class
    )""",
    """CREATE REL TABLE IF NOT EXISTS IMPLEMENTS(
        FROM Class TO Interface
    )""",
    """CREATE REL TABLE IF NOT EXISTS OVERRIDES(
        FROM Method TO Method
    )""",
    """CREATE REL TABLE IF NOT EXISTS BELONGS_TO(
        FROM File TO Module,
        FROM File TO Package
    )""",
    # BUC-1611: module-level method rebinding (monkey-patching).
    # When a module's top-level scope does ``Widget.render = custom``, we
    # emit a REBINDS edge from the *rebinding module* to the *original*
    # callable that was overwritten.  The replacement target is carried
    # as a string property (``new_target``) rather than a second endpoint
    # because LadybugDB rel-table FROM/TO pairs have to be declared up
    # front; storing the qname as a STRING lets the resolver look it up
    # in the function_registry at call-resolution time.
    #
    # Endpoint pairs:
    #   • FROM Module — every rebinding lives at module scope.
    #   • TO Method   — the common case (``Class.method = ...``).
    #   • TO Function — for symmetry when the original callable was
    #     classified as a Function (graph_updater treats module-scope
    #     ``def`` as Function; we keep both endpoints so the resolver
    #     can emit either without scheme acrobatics).
    #
    # Ordering note: "latest wins" within a single module is a *resolver*
    # decision (last assignment encountered in source order — see
    # ``rebind_processor.RebindRegistry``).  Inter-module ordering uses
    # last-module-encountered semantics; the schema does not encode
    # ordering — the resolver does.
    """CREATE REL TABLE IF NOT EXISTS REBINDS(
        FROM Module TO Method,
        FROM Module TO Function,
        new_target STRING DEFAULT '',
        file_path STRING DEFAULT '',
        line_start INT64 DEFAULT 0
    )""",
    # BUC-1610: module-to-module re-export topology.  Emitted by
    # ``import_processor`` when a barrel / package-init forwards symbols from
    # another module (e.g. TS ``export * from './foo'``, Python
    # ``__all__ = [...]``).  Symbol-level chaining at lookup time lives in
    # ``CallResolver._walk_reexport_chain``.  The edge is bare — endpoints
    # only — because chain traversal is qname-based and doesn't need
    # per-edge metadata.  Listed AFTER REBINDS so older DBs that already
    # have everything up to REBINDS pick this up cleanly via
    # ``CREATE REL TABLE IF NOT EXISTS`` on the next startup.
    """CREATE REL TABLE IF NOT EXISTS RE_EXPORTS(
        FROM Module TO Module
    )""",
]

# ---------------------------------------------------------------------------
# Backfill ALTERs for existing databases (BUC-1603 + BUC-1609)
# ---------------------------------------------------------------------------
# Existing CALLS rows written before BUC-1603 / BUC-1609 lack the provenance
# columns.
# LadybugDB (Kuzu fork) supports ``ALTER REL TABLE <name> ADD <col> <type>``
# but, unlike ``CREATE``, it has no ``IF NOT EXISTS`` guard.  We run these
# in a try/except and treat "already exists" / "duplicate column" errors as
# success.  For fresh DBs the columns are declared inline in ``_REL_TABLES``
# so each ALTER is a no-op (and the idempotent-success branch fires).  For
# pre-BUC-1603 DBs the ALTER backfills the columns with empty-string / 0
# defaults, preserving every existing CALLS row.
#
# Consumers (e.g. code-indexer-service) should re-index after deploy to
# populate the new columns on existing edges — there is no in-place backfill
# of file_path / line_start / col_start for rows written before the parser
# wiring landed.  This is flagged as a follow-up: a future migration could
# add a "schema_version" metadata table and trigger a re-index, but is out
# of scope here.
_REL_ALTERS: list[str] = [
    "ALTER TABLE CALLS ADD file_path STRING DEFAULT ''",
    "ALTER TABLE CALLS ADD line_start INT64 DEFAULT 0",
    "ALTER TABLE CALLS ADD col_start INT64 DEFAULT 0",
    # BUC-1609: resolver provenance — pre-existing rows surface as
    # ('unknown', 1.0).  The DEFAULT of 1.0 (not 0.0) is intentional: a
    # downstream ``min_confidence`` filter that gates on >= 0.5 should not
    # accidentally drop legacy edges just because they were ingested before
    # the resolver-tagging code shipped.  Once the consumer is re-indexed
    # the column converges on the resolver-supplied values.
    "ALTER TABLE CALLS ADD resolved_via STRING DEFAULT 'unknown'",
    "ALTER TABLE CALLS ADD confidence DOUBLE DEFAULT 1.0",
]

# BUC-1621: backfill ALTERs for node tables.  CREATE NODE TABLE IF NOT EXISTS
# is a no-op on an existing table — it will NOT add new columns to a table
# that was created with an older schema.  When BUC-1602 added is_async /
# is_generator inline above, fresh DBs were correct but every existing
# on-disk DB (created before BUC-1602 landed) was left with a Function /
# Method table missing those columns, so ``ingest_method`` flushes failed
# silently with "Cannot find property is_async" → 0 Methods landed.
#
# Same idempotent-success pattern as ``_REL_ALTERS``: duplicate-column
# errors are swallowed; anything else is hard-fail.
_NODE_ALTERS: list[str] = [
    "ALTER TABLE Function ADD is_async BOOL DEFAULT FALSE",
    "ALTER TABLE Function ADD is_generator BOOL DEFAULT FALSE",
    "ALTER TABLE Method ADD is_async BOOL DEFAULT FALSE",
    "ALTER TABLE Method ADD is_generator BOOL DEFAULT FALSE",
    # Anthropic Contextual Retrieval — 50-100 token LLM-generated summary
    # prepended to chunk text before embedding.  DEFAULT '' so legacy rows
    # surface as empty (no behaviour change unless CONTEXTUAL_RETRIEVAL_ENABLED
    # is set and the repo is re-indexed).  See
    # ``codebase_rag/services/contextual_prefix.py``.
    "ALTER TABLE Function ADD contextual_prefix STRING DEFAULT ''",
    "ALTER TABLE Method ADD contextual_prefix STRING DEFAULT ''",
]

# ---------------------------------------------------------------------------
# Audit — expected schema surface (BUC-1621)
# ---------------------------------------------------------------------------
# Source-of-truth list of table names the ingestor expects to find after
# ``migrate()``.  Logged at INFO level by ``_audit_schema()`` so drift between
# declared and on-disk schema is visible in service logs without needing to
# attach a debugger.  Order matches the DDL order above for log readability.
_EXPECTED_NODE_TABLES: tuple[str, ...] = (
    "Project",
    "Package",
    "Folder",
    "File",
    "Module",
    "Class",
    "Function",
    "Method",
    "Interface",
    "Enum",
    "ExternalPackage",
)

_EXPECTED_REL_TABLES: tuple[str, ...] = (
    "CONTAINS_FILE",
    "CONTAINS_FOLDER",
    "CONTAINS_PACKAGE",
    "CONTAINS_MODULE",
    "DEFINES",
    "DEFINES_METHOD",
    "CALLS",
    "IMPORTS",
    "INHERITS",
    "IMPLEMENTS",
    "OVERRIDES",
    "BELONGS_TO",
    "REBINDS",
    "RE_EXPORTS",
)

# Substrings that indicate "the column you tried to add already exists".
# LadybugDB error messages vary across versions, so match common fragments
# rather than an exact string.
_ALTER_IDEMPOTENT_SUBSTRINGS: tuple[str, ...] = (
    "already exists",
    "duplicate",
    "already has property",
)


def _run_alter(conn: lb.Connection, alter_ddl: str, kind: str) -> None:
    """Execute a single ALTER, treating duplicate-column errors as success.

    Extracted so node and rel ALTER loops share one code path.  Any error
    whose message does not match ``_ALTER_IDEMPOTENT_SUBSTRINGS`` is
    re-raised — schema-migration failures must be loud.
    """
    try:
        conn.execute(alter_ddl)
        logger.debug(f"  Applied {kind} ALTER: {alter_ddl}")
    except Exception as e:
        err_str = str(e).lower()
        if any(s in err_str for s in _ALTER_IDEMPOTENT_SUBSTRINGS):
            logger.debug(
                f"  {kind} ALTER skipped (column already present): {alter_ddl}"
            )
        else:
            logger.error(f"  {kind} ALTER failed: {alter_ddl}: {e}")
            raise


def _table_exists(conn: lb.Connection, table_name: str) -> bool:
    """Best-effort probe for table presence.

    LadybugDB does not expose ``information_schema``; the cheapest portable
    probe is ``CALL show_tables() RETURN *`` (Kuzu/Lugawugu compatible).  If
    that fails (older / forked binaries that don't expose ``show_tables``),
    we fall back to a no-op MATCH against the label — a non-existent table
    raises, an existing one returns 0 rows in O(1).
    """
    try:
        result = conn.execute("CALL show_tables() RETURN *")
        for row in _iter_rows(result):
            # ``show_tables`` columns vary by version; the table name is
            # always the second column ("name") in Kuzu/Lugawugu.
            for cell in row:
                if isinstance(cell, str) and cell == table_name:
                    return True
        return False
    except Exception:
        # show_tables unavailable — fall back to probe query.
        try:
            conn.execute(f"MATCH (n:{table_name}) RETURN count(n) LIMIT 1")
            return True
        except Exception:
            try:
                conn.execute(
                    f"MATCH ()-[r:{table_name}]->() RETURN count(r) LIMIT 1"
                )
                return True
            except Exception:
                return False


def _iter_rows(result: object) -> list[list[object]]:
    """Adapt LadybugDB result objects to a plain list-of-rows.

    LadybugDB query results expose ``get_next``/``has_next`` (iterator) or
    ``rows`` (eager) depending on version.  We try both shapes; on a totally
    unknown shape we return an empty list so the caller falls back to the
    MATCH-probe branch.
    """
    rows: list[list[object]] = []
    try:
        if hasattr(result, "has_next") and hasattr(result, "get_next"):
            while result.has_next():  # type: ignore[attr-defined]
                rows.append(list(result.get_next()))  # type: ignore[attr-defined]
            return rows
    except Exception:
        rows = []
    try:
        eager_rows = getattr(result, "rows", None)
        if eager_rows is not None:
            return [list(r) for r in eager_rows]
    except Exception:
        pass
    return rows


def _audit_schema(conn: lb.Connection) -> None:
    """Log the present/absent state of every expected table at INFO.

    BUC-1621: makes schema drift between the declared list and the on-disk
    DB visible in service logs.  Never raises — an audit failure must not
    abort the migration; the worst case is an empty audit, which is itself
    a signal.
    """
    try:
        node_present = [t for t in _EXPECTED_NODE_TABLES if _table_exists(conn, t)]
        node_missing = [t for t in _EXPECTED_NODE_TABLES if t not in node_present]
        rel_present = [t for t in _EXPECTED_REL_TABLES if _table_exists(conn, t)]
        rel_missing = [t for t in _EXPECTED_REL_TABLES if t not in rel_present]

        logger.info(
            f"LadybugDB schema audit — nodes: {len(node_present)}/"
            f"{len(_EXPECTED_NODE_TABLES)} present "
            f"[{', '.join(node_present)}]"
        )
        if node_missing:
            logger.warning(
                f"LadybugDB schema audit — MISSING node tables: "
                f"[{', '.join(node_missing)}]"
            )
        logger.info(
            f"LadybugDB schema audit — rels:  {len(rel_present)}/"
            f"{len(_EXPECTED_REL_TABLES)} present "
            f"[{', '.join(rel_present)}]"
        )
        if rel_missing:
            logger.warning(
                f"LadybugDB schema audit — MISSING rel tables: "
                f"[{', '.join(rel_missing)}]"
            )
    except Exception as e:
        logger.warning(f"LadybugDB schema audit failed (non-fatal): {e}")


def migrate(db_path: str) -> None:
    """Run schema migration on the given LadybugDB database path.

    Idempotent — safe to call on an existing database. Every CREATE TABLE
    uses ``IF NOT EXISTS``; every ALTER swallows duplicate-column errors.
    The full DDL set runs unconditionally on every startup so newly-added
    node columns and rel tables land on existing DBs without manual
    intervention.

    After all DDL has executed, the present/absent state of every expected
    table is logged at INFO level by ``_audit_schema()``.

    No VECTOR extension is loaded here; embeddings are stored in per-repo
    numpy files (see ``vector_store.py``).

    Args:
        db_path: Filesystem path to the LadybugDB database file. Created if
            it does not exist.
    """
    logger.info(f"Running LadybugDB schema migration on: {db_path}")
    db = lb.Database(db_path)
    conn = lb.Connection(db)

    # 1. Node DDL — rel tables below reference these types.
    #    ``CREATE NODE TABLE IF NOT EXISTS`` is a no-op when the table
    #    already exists; new columns on existing tables are handled by the
    #    ALTER pass below (step 3).
    for ddl in _NODE_TABLES:
        # Extract the table name from the DDL for logging only — LadybugDB
        # does not echo the created object name back to the caller.
        table_name = ddl.split("TABLE IF NOT EXISTS")[1].split("(")[0].strip()
        conn.execute(ddl)
        logger.debug(f"  Node table: {table_name}")

    # 2. Rel DDL — same semantics as node DDL.
    for ddl in _REL_TABLES:
        table_name = ddl.split("TABLE IF NOT EXISTS")[1].split("(")[0].strip()
        conn.execute(ddl)
        logger.debug(f"  Rel table: {table_name}")

    # 3a. Node-table backfill ALTERs (BUC-1621).  Runs unconditionally; each
    #     ALTER is independent — a failure on one column should not prevent
    #     the next from being tried.  Idempotent "already exists" errors
    #     are swallowed.
    for alter_ddl in _NODE_ALTERS:
        _run_alter(conn, alter_ddl, kind="NODE")

    # 3b. Rel-table backfill ALTERs (BUC-1603 + BUC-1609).  Same contract.
    for alter_ddl in _REL_ALTERS:
        _run_alter(conn, alter_ddl, kind="REL")

    # 4. Audit — log present/absent state of every expected table so drift
    #    between the declared list and the on-disk DB is visible in logs.
    _audit_schema(conn)

    logger.info("LadybugDB schema migration complete ✓")
