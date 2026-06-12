"""DuckDB-backed per-repo vector store (v5.3 §6.5 + §8.4).

Replaces the previous SQLite + sqlite-vec implementation with a single
``.duck`` file per repo holding ``FLOAT[768]`` embedding vectors plus
symbol metadata, repo metadata, and PageRank centrality scores.

v5.3 §17.2 explicitly rejects sqlite-vec; §8.4 mandates DuckDB
``FLOAT[768]`` columns and the built-in ``array_cosine_distance``
function with L2-normalised embeddings written at insert time so that
cosine similarity is equivalent to inner product at query time.

Schema
------
    embeddings(qualified_name TEXT PK, embedding FLOAT[768],
               symbol_type TEXT, file_path TEXT,
               start_line INTEGER, end_line INTEGER,
               indexed_at BIGINT)
    repo_metadata(key TEXT PK, value TEXT NOT NULL, updated_at BIGINT)
    centrality(qualified_name TEXT PK, pagerank REAL NOT NULL,
               updated_at BIGINT)

Public API (preserved from the previous backend so callers don't change):
    open_or_create(path) -> duckdb.DuckDBPyConnection
    insert_embedding(conn, row)
    bulk_insert(conn, rows) -> int
    search_similar(conn, query_vec, k) -> list[SearchResult]
    write_metadata(conn, **fields)
    read_metadata(conn, key, default=None)
    read_all_metadata(conn) -> dict[str, str]
    write_centrality(conn, scores) -> int
    read_centrality(conn, qualified_names) -> dict[str, float]
    clear_centrality(conn)
    row_count(conn) -> int
    verify_stored_ids(conn, qualified_names) -> set[str]
    delete_embeddings(conn, qualified_names) -> int

Phase 8 — HNSW scaffold (OFF by default):
    _ensure_vss_extension(conn)       — INSTALL/LOAD vss idempotently
    _hnsw_active(conn) -> bool        — reads repo_metadata.hnsw_active
    create_hnsw_index(conn, ...)      — CREATE INDEX IF NOT EXISTS via HNSW

The HNSW query path is activated only when BOTH of the following are true:
    1. Global env  HNSW_ENABLED=true
    2. Per-repo    repo_metadata key "hnsw_active" = "true"

When either gate is false, search_similar falls back to the existing
brute-force array_cosine_distance path.  Activation triggers (p95 > 200 ms
or repo > 50k symbols) are not yet met; this scaffold lands off-by-default
for future operator use.

DuckDB connections are not safe to share across threads; callers should
open a per-request connection (or a per-thread cursor) and close it
when done.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import duckdb  # noqa: F401  (type-only; actual import is lazy)

_EMBEDDING_DIM = 768

# ---------------------------------------------------------------------------
# Phase 8 — HNSW scaffold constants
# ---------------------------------------------------------------------------

_HNSW_INDEX_NAME = "hnsw_function_embed"
_HNSW_M = 16
_HNSW_EF_CONSTRUCTION = 200


def _ensure_vss_extension(conn: Any) -> None:
    """Install and load the DuckDB VSS extension idempotently.

    Safe to call multiple times; DuckDB's ``INSTALL`` and ``LOAD`` are no-ops
    when the extension is already present.

    Also enables ``hnsw_enable_experimental_persistence`` so that HNSW indexes
    can be created on persistent (file-backed) databases.  This setting is
    required by DuckDB VSS >= 1.1.3 for any non-in-memory connection; it is
    harmless on in-memory connections.

    Args:
        conn: Open connection from ``open_or_create``.
    """
    conn.execute("INSTALL vss")
    conn.execute("LOAD vss")
    conn.execute("SET hnsw_enable_experimental_persistence = true")


def _hnsw_active(conn: Any) -> bool:
    """Return True only when BOTH activation gates are open.

    Gate 1 — global env: ``HNSW_ENABLED=true`` (case-insensitive).
    Gate 2 — per-repo:   ``repo_metadata`` row with key ``"hnsw_active"``
              and value ``"true"`` (case-insensitive).

    Returns False (safe fallback) if either gate is absent or falsy, and
    also if the ``repo_metadata`` column/row does not exist.

    Args:
        conn: Open connection from ``open_or_create``.

    Returns:
        bool: True when HNSW query path should be used; False otherwise.
    """
    # Gate 1: global env flag
    if os.environ.get("HNSW_ENABLED", "false").lower() != "true":
        return False

    # Gate 2: per-repo metadata flag
    try:
        row = conn.execute(
            "SELECT value FROM repo_metadata WHERE key = 'hnsw_active'"
        ).fetchone()
        return bool(row and row[0].lower() == "true")
    except Exception:
        # Table missing or transient error — fail closed
        return False


def create_hnsw_index(
    conn: Any,
    table: str = "function_embeddings",
    col: str = "embedding",
) -> None:
    """Create the HNSW index idempotently (CREATE INDEX IF NOT EXISTS).

    Loads the VSS extension before issuing DDL.  The index uses cosine
    distance metric with M=16 and ef_construction=200 as specified in
    Phase 8 (.planning/phase-plans/PHASE_8_HNSW_VSS.md §5).

    This function operates on the ``embeddings`` table by default.  Pass
    ``table="embeddings"`` explicitly when calling from application code;
    the default ``"function_embeddings"`` is kept for backward-compat with
    any callers that were drafted against an earlier name.

    Args:
        conn: Open connection from ``open_or_create``.
        table: Table to index.  Defaults to ``"function_embeddings"`` (alias
            for ``"embeddings"`` in the HNSW DDL; pass ``"embeddings"`` for
            the live schema).
        col: Embedding column name.  Defaults to ``"embedding"``.
    """
    _ensure_vss_extension(conn)
    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS {_HNSW_INDEX_NAME}
        ON {table} USING HNSW ({col})
        WITH (metric = 'cosine', M = {_HNSW_M}, ef_construction = {_HNSW_EF_CONSTRUCTION})
        """
    )


@dataclass
class EmbeddingRow:
    """One row in the ``embeddings`` table.

    The embedding vector is L2-normalised at write time (see ``bulk_insert``)
    so that cosine similarity equals the inner product at query time.
    """

    qualified_name: str
    embedding: list[float]
    file_path: str
    start_line: int
    end_line: int
    symbol_type: str
    indexed_at: int = field(default_factory=lambda: int(time.time()))
    # BUC-1518 C2: SHA-1 of the symbol's source range, used by the embed
    # driver to skip re-embedding unchanged symbols on incremental re-index.
    # Optional for backwards compat — None means "unknown, treat as changed".
    content_hash: str | None = None


@dataclass
class SearchResult:
    """One result from ``search_similar``."""

    qualified_name: str
    file_path: str
    start_line: int
    end_line: int
    score: float  # cosine similarity in [-1, 1]; higher = more similar


def _l2_normalise(vec: list[float]) -> list[float]:
    """Return a unit-norm copy of ``vec``.

    Zero vectors are returned unchanged so callers don't have to special-case
    them; sqrt of a tiny float still produces a defined result.
    """
    mag = math.sqrt(sum(x * x for x in vec))
    if mag == 0.0:
        return list(vec)
    return [x / mag for x in vec]


# Benign ALTER failure: another writer applied the migration between our
# presence check and the ALTER. Anything else (syntax error, corrupt DB,
# lock failure) must raise — a blanket suppress here once hid a broken
# migration for years in a sibling project.
_DUPLICATE_COLUMN_SUBSTRINGS = ("already exists", "duplicate column")


def _add_column_if_missing(conn: Any, table: str, column: str, alter_ddl: str) -> None:
    """Apply an ADD COLUMN migration, suppressing only duplicate-column errors."""
    existing_cols = {
        r[0]
        for r in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = ?",
            [table],
        ).fetchall()
    }
    if column in existing_cols:
        return
    try:
        conn.execute(alter_ddl)
    except Exception as exc:
        if any(s in str(exc).lower() for s in _DUPLICATE_COLUMN_SUBSTRINGS):
            return
        raise


def open_or_create(path: str | Path) -> Any:
    """Open (or create) a ``.duck`` file and ensure the schema exists.

    Args:
        path: Filesystem path to the DuckDB file.  Parent directory is
            created automatically if missing.

    Returns:
        duckdb.DuckDBPyConnection: An open connection with the schema applied.
        Callers are responsible for closing it.

    Raises:
        RuntimeError: When the ``duckdb`` module is not installed.
    """
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover — surfaced clearly to callers
        raise RuntimeError(
            "duckdb is not installed. Add `duckdb>=1.1.0` to your project."
        ) from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(path))
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS embeddings (
            qualified_name TEXT PRIMARY KEY,
            embedding      FLOAT[{_EMBEDDING_DIM}],
            symbol_type    TEXT,
            file_path      TEXT,
            start_line     INTEGER,
            end_line       INTEGER,
            indexed_at     BIGINT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS repo_metadata (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            updated_at BIGINT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS centrality (
            qualified_name TEXT PRIMARY KEY,
            pagerank       REAL NOT NULL,
            updated_at     BIGINT NOT NULL
        )
        """
    )
    # Phase 8 schema migration: add hnsw_active flag column idempotently.
    # DuckDB does not support ADD COLUMN IF NOT EXISTS, so we guard with a
    # presence check against information_schema.
    _add_column_if_missing(
        conn,
        "repo_metadata",
        "hnsw_active",
        "ALTER TABLE repo_metadata ADD COLUMN hnsw_active BOOLEAN DEFAULT FALSE",
    )

    # BUC-1518 C2 schema migration: add content_hash column to embeddings.
    # Drives incremental embedding — when re-indexing, the embed driver hashes
    # each symbol's source range and skips any symbol whose stored hash matches
    # (no SageMaker call, no recompute).  For typical commits touching a few
    # files, this skips 95-99% of the work.
    _add_column_if_missing(
        conn,
        "embeddings",
        "content_hash",
        "ALTER TABLE embeddings ADD COLUMN content_hash TEXT",
    )
    return conn


def insert_embedding(conn: Any, row: EmbeddingRow) -> None:
    """Insert or replace a single embedding row.

    The embedding vector is L2-normalised before write so that
    ``1 - array_cosine_distance(...)`` at query time is the inner product
    against unit vectors (matches v5.3 §8.4 exactly).

    Args:
        conn: Open connection from ``open_or_create``.
        row: Embedding data including the 768-dim vector and metadata.
    """
    bulk_insert(conn, [row])


def bulk_insert(conn: Any, rows: list[EmbeddingRow]) -> int:
    """Insert (upsert) many rows inside a single transaction.

    Implementation note (perf — measured 2026-04-27):
        DuckDB's per-row binding of ``FLOAT[768]`` parameters from a Python
        list is the bottleneck of ``executemany`` — at 1000 rows it costs
        ~44 ms/row (≈44 s total) regardless of how many rows go through one
        ``executemany`` call.

        When ``pyarrow`` is importable, this function transparently delegates
        to :func:`vector_store_arrow.bulk_insert_arrow`, which stages the
        same data through a registered Arrow table and uses DuckDB's
        columnar bulk-load path.  Measured speedup at 100/500/1000 rows is
        ~324×/382×/390× respectively (linear scaling — see
        ``scripts/BENCH_RESULTS_2026-04-27.md``).

        Without ``pyarrow``, falls back to the executemany path (one batched
        DELETE + one ``executemany`` INSERT) which is correct but slow.
        Install the Arrow extra to opt in::

            pip install code-graph-rag[arrow]

    Args:
        conn: Open connection from ``open_or_create``.
        rows: Embedding rows to insert.  Empty list is a no-op.

    Returns:
        int: Number of rows inserted.
    """
    if not rows:
        return 0

    # Fast path: delegate to the Arrow-staged implementation when pyarrow
    # is installed.  ~380× faster on FLOAT[768] payloads (see docstring).
    try:
        import pyarrow  # noqa: F401  (sentinel — avoids the cost when absent)

        from codebase_rag.storage.vector_store_arrow import bulk_insert_arrow

        return bulk_insert_arrow(conn, rows)
    except ImportError:
        pass  # pyarrow not installed — use the executemany fallback below.

    now = int(time.time())
    qnames = [row.qualified_name for row in rows]
    insert_params = [
        (
            row.qualified_name,
            _l2_normalise(row.embedding),
            row.symbol_type,
            row.file_path,
            row.start_line,
            row.end_line,
            row.indexed_at or now,
            row.content_hash,  # BUC-1518 C2 — None on legacy callers, fine
        )
        for row in rows
    ]
    placeholders = ",".join("?" for _ in qnames)

    conn.execute("BEGIN")
    try:
        conn.execute(
            f"DELETE FROM embeddings WHERE qualified_name IN ({placeholders})",
            qnames,
        )
        conn.executemany(
            """
            INSERT INTO embeddings
                (qualified_name, embedding, symbol_type, file_path,
                 start_line, end_line, indexed_at, content_hash)
            VALUES (?, ?::FLOAT[768], ?, ?, ?, ?, ?, ?)
            """,
            insert_params,
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return len(rows)


def search_similar(
    conn: Any,
    query_vec: list[float],
    k: int = 10,
) -> list[SearchResult]:
    """Return the top-k most cosine-similar symbols.

    Stored embeddings are L2-normalised at write time, so for a normalised
    query vector ``1 - array_cosine_distance(stored, query)`` yields the
    cosine similarity in ``[-1, 1]`` (higher = more similar).  We normalise
    the query here defensively in case callers pass an un-normalised vector.

    Phase 8 — HNSW dispatch:
        When BOTH ``HNSW_ENABLED=true`` (env) AND ``repo_metadata`` key
        ``"hnsw_active"`` is ``"true"``, the query uses the HNSW index via
        the ``<=>`` cosine-distance operator (lower = more similar; we invert
        to a similarity score for API consistency).  Otherwise the existing
        brute-force ``array_cosine_distance`` path is used unchanged.

    Args:
        conn: Open connection from ``open_or_create``.
        query_vec: 768-dim query embedding.
        k: Max number of nearest neighbours to return.

    Returns:
        list[SearchResult]: Ranked results, highest similarity first.
    """
    normalised = _l2_normalise(query_vec)

    if _hnsw_active(conn):
        # HNSW path: <=> is cosine-distance (0 = identical, 2 = opposite).
        # Invert to similarity for API consistency with the brute-force path.
        rows = conn.execute(
            f"""
            SELECT qualified_name, file_path, start_line, end_line,
                   1.0 - (embedding <=> ?::FLOAT[{_EMBEDDING_DIM}]) AS score
            FROM embeddings
            ORDER BY embedding <=> ?::FLOAT[{_EMBEDDING_DIM}]
            LIMIT ?
            """,
            (normalised, normalised, int(k)),
        ).fetchall()
    else:
        # Brute-force path (default): unchanged from pre-Phase-8 behaviour.
        rows = conn.execute(
            f"""
            SELECT qualified_name, file_path, start_line, end_line,
                   1.0 - array_cosine_distance(embedding, ?::FLOAT[{_EMBEDDING_DIM}]) AS score
            FROM embeddings
            ORDER BY score DESC
            LIMIT ?
            """,
            (normalised, int(k)),
        ).fetchall()

    return [
        SearchResult(
            qualified_name=r[0],
            file_path=r[1] or "",
            start_line=int(r[2]) if r[2] is not None else 0,
            end_line=int(r[3]) if r[3] is not None else 0,
            score=float(r[4]),
        )
        for r in rows
    ]


def write_metadata(conn: Any, **fields: Any) -> None:
    """Upsert key-value pairs into ``repo_metadata``.

    Args:
        conn: Open connection from ``open_or_create``.
        **fields: Arbitrary key=value pairs.  All values are coerced to str.
    """
    if not fields:
        return
    now = int(time.time())
    keys = list(fields.keys())
    placeholders = ",".join("?" for _ in keys)
    insert_params = [(k, str(fields[k]), now) for k in keys]

    conn.execute("BEGIN")
    try:
        conn.execute(
            f"DELETE FROM repo_metadata WHERE key IN ({placeholders})",
            keys,
        )
        conn.executemany(
            """
            INSERT INTO repo_metadata (key, value, updated_at)
            VALUES (?, ?, ?)
            """,
            insert_params,
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def read_metadata(
    conn: Any, key: str, default: str | None = None
) -> str | None:
    """Read a single value from ``repo_metadata``.

    Args:
        conn: Open connection from ``open_or_create``.
        key: Metadata key (e.g. ``"last_indexed_at"``).
        default: Value to return when the key is absent.

    Returns:
        str | None: The stored value, or ``default`` when not found.
    """
    row = conn.execute(
        "SELECT value FROM repo_metadata WHERE key = ?", (key,)
    ).fetchone()
    return row[0] if row else default


def read_all_metadata(conn: Any) -> dict[str, str]:
    """Return all rows from ``repo_metadata`` as a plain dict.

    Args:
        conn: Open connection from ``open_or_create``.

    Returns:
        dict[str, str]: All key/value pairs, empty dict on miss.
    """
    rows = conn.execute("SELECT key, value FROM repo_metadata").fetchall()
    return {r[0]: r[1] for r in rows}


def write_centrality(conn: Any, scores: dict[str, float]) -> int:
    """Bulk upsert PageRank centrality scores.

    Args:
        conn: Open connection from ``open_or_create``.
        scores: Mapping of ``qualified_name`` → normalised PageRank score
            in ``[0.0, 1.0]``.  Empty dict is a no-op.

    Returns:
        int: Number of rows written.
    """
    if not scores:
        return 0
    now = int(time.time())
    qnames = list(scores.keys())
    placeholders = ",".join("?" for _ in qnames)
    insert_params = [(q, float(scores[q]), now) for q in qnames]

    conn.execute("BEGIN")
    try:
        conn.execute(
            f"DELETE FROM centrality WHERE qualified_name IN ({placeholders})",
            qnames,
        )
        conn.executemany(
            """
            INSERT INTO centrality (qualified_name, pagerank, updated_at)
            VALUES (?, ?, ?)
            """,
            insert_params,
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return len(scores)


def read_centrality(
    conn: Any, qualified_names: list[str]
) -> dict[str, float]:
    """Return PageRank scores for the requested qualified names.

    Args:
        conn: Open connection from ``open_or_create``.
        qualified_names: Symbol names to look up.  Missing keys are absent
            from the returned dict — callers should default to 0.0.

    Returns:
        dict[str, float]: Subset of ``qualified_names`` that have a stored
        score, mapped to the score.
    """
    if not qualified_names:
        return {}
    placeholders = ",".join("?" for _ in qualified_names)
    rows = conn.execute(
        f"SELECT qualified_name, pagerank FROM centrality "
        f"WHERE qualified_name IN ({placeholders})",
        tuple(qualified_names),
    ).fetchall()
    return {r[0]: float(r[1]) for r in rows}


def clear_centrality(conn: Any) -> None:
    """Delete every row from the ``centrality`` table.

    Used before recomputing scores so stale qualified names from a previous
    indexing run don't linger after files are deleted upstream.

    Args:
        conn: Open connection from ``open_or_create``.
    """
    conn.execute("BEGIN")
    try:
        conn.execute("DELETE FROM centrality")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def row_count(conn: Any) -> int:
    """Return the number of embeddings stored.

    Args:
        conn: Open connection from ``open_or_create``.

    Returns:
        int: Row count (0 when table is empty or missing).
    """
    try:
        result = conn.execute("SELECT count(*) FROM embeddings").fetchone()
        return int(result[0]) if result else 0
    except Exception:
        return 0


def verify_stored_ids(
    conn: Any, qualified_names: set[str]
) -> set[str]:
    """Return the subset of ``qualified_names`` that have a stored embedding.

    Lane-C migration helper — replaces the legacy ``vector_store.verify_stored_ids``
    which keyed by Memgraph ``node_id``.  The DuckDB schema is keyed by
    ``qualified_name``, so callers porting from the numpy backend should
    translate their node-id set to qualified-names via the ingestor before
    calling this.

    Used by ``graph_updater._reconcile_embeddings`` to detect rows that
    were generated but failed to persist (so it can warn the operator
    rather than silently lose data).

    Args:
        conn: Open connection from ``open_or_create``.
        qualified_names: The full set of names the caller expected to write.
            Empty set returns an empty set.

    Returns:
        set[str]: Intersection of ``qualified_names`` and rows present in the
        ``embeddings`` table.  Missing names = ``qualified_names - returned``.
    """
    if not qualified_names:
        return set()
    placeholders = ",".join("?" for _ in qualified_names)
    try:
        rows = conn.execute(
            f"SELECT qualified_name FROM embeddings "
            f"WHERE qualified_name IN ({placeholders})",
            tuple(qualified_names),
        ).fetchall()
    except Exception:
        # Table missing or transient DB error — treat as nothing stored so
        # the reconciliation pass logs every expected id as missing.
        return set()
    return {r[0] for r in rows}


def read_content_hashes(conn: Any) -> dict[str, str]:
    """Return ``{qualified_name: content_hash}`` for every row that has one.

    BUC-1518 C2 — drives incremental embedding.  The embed driver hashes each
    symbol's source range and looks it up here; if the stored hash matches,
    the SageMaker call is skipped entirely (no recompute, no cost).

    Returns an empty dict if the column is missing (legacy .duck files) or
    on any error — callers fall back to "always re-embed" semantics, which
    is correct just slower.

    Args:
        conn: Open connection from ``open_or_create``.

    Returns:
        Mapping from qualified_name to its stored ``content_hash``.  Rows
        with NULL ``content_hash`` are omitted (treated as "unknown,
        re-embed when next seen").
    """
    try:
        rows = conn.execute(
            "SELECT qualified_name, content_hash FROM embeddings "
            "WHERE content_hash IS NOT NULL"
        ).fetchall()
    except Exception:
        # content_hash column missing (pre-migration .duck), or transient DB
        # error.  Returning {} forces full re-embed, which is safe.
        return {}
    return {r[0]: r[1] for r in rows if r[0] and r[1]}


def delete_embeddings(conn: Any, qualified_names: list[str] | set[str]) -> int:
    """Delete embedding rows by qualified name.

    Lane-C migration helper — replaces ``vector_store.delete_project_embeddings``
    which keyed by Memgraph ``node_id`` and required the project name as a
    namespacing prefix.  In the DuckDB store every ``.duck`` file is already
    per-repo, so the project-name dimension is implicit in the file path.

    Args:
        conn: Open connection from ``open_or_create``.
        qualified_names: Names to delete.  Empty input is a no-op.

    Returns:
        int: Number of rows deleted (0 when input was empty or none matched).
    """
    if not qualified_names:
        return 0
    names = list(qualified_names)
    placeholders = ",".join("?" for _ in names)
    conn.execute("BEGIN")
    try:
        # DuckDB doesn't return rowcount on DELETE, so count first.
        existing = conn.execute(
            f"SELECT count(*) FROM embeddings "
            f"WHERE qualified_name IN ({placeholders})",
            tuple(names),
        ).fetchone()
        deleted = int(existing[0]) if existing else 0
        conn.execute(
            f"DELETE FROM embeddings WHERE qualified_name IN ({placeholders})",
            tuple(names),
        )
        conn.execute("COMMIT")
        return deleted
    except Exception:
        conn.execute("ROLLBACK")
        raise
