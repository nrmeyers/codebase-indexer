"""Arrow-staged bulk-insert for the DuckDB vector store.

This module is **opt-in** and lives separately from ``vector_store.py`` so the
core path has zero new dependencies.  Install with::

    pip install code-graph-rag[arrow]

When pyarrow is installed, ``vector_store.bulk_insert`` automatically delegates
here for a ~380× speedup over the executemany fallback (see
``scripts/BENCH_RESULTS_2026-04-27.md``).  The columnar staging path bypasses
DuckDB's per-row Python-list parameter binding for ``FLOAT[768]`` arrays,
which microbenches identified as the dominant bottleneck.

Public API:
    bulk_insert_arrow(conn, rows) -> int

Contract notes:
- Embeddings are L2-normalised before staging so the on-disk vector is a
  unit vector — identical to ``vector_store.bulk_insert``.  This keeps
  ``1 - array_cosine_distance`` equivalent to inner product at query time.
- Upserts are handled the same way as ``bulk_insert``: a single batched
  DELETE on ``qualified_name`` clears prior rows in one shot, then a
  ``INSERT INTO ... SELECT FROM staging`` copies from the registered Arrow
  table.
- Wraps in BEGIN/COMMIT with rollback on exception, mirroring ``bulk_insert``.
"""
from __future__ import annotations

import time
from typing import Any

from codebase_rag.storage.vector_store import EmbeddingRow

_EMBEDDING_DIM = 768
_STAGING = "_vsa_staging"


def _import_pyarrow() -> Any:
    """Import and return the ``pyarrow`` module, raising RuntimeError when absent."""
    try:
        import pyarrow as pa  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover — explicit operator guidance
        raise RuntimeError(
            "pyarrow is not installed. Install the optional dep with "
            "`pip install code-graph-rag[arrow]`."
        ) from exc
    return pa


def _normalise_matrix(rows: list[EmbeddingRow]) -> list[list[float]]:
    """L2-normalise each embedding; preserve zero vectors."""
    import numpy as np  # numpy is a hard dep of code-graph-rag

    mat = np.asarray(
        [r.embedding for r in rows], dtype=np.float32
    )  # (N, 768)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit = mat / norms
    return unit.tolist()


def bulk_insert_arrow(conn: Any, rows: list[EmbeddingRow]) -> int:
    """Insert (upsert) many rows via an Arrow staging table.

    Args:
        conn: Open DuckDB connection (from ``vector_store.open_or_create``).
        rows: Embedding rows.  Empty list is a no-op.

    Returns:
        int: Number of rows inserted.

    Raises:
        RuntimeError: When pyarrow is not installed.
    """
    if not rows:
        return 0

    pa = _import_pyarrow()
    now = int(time.time())

    qnames = [r.qualified_name for r in rows]
    embeddings = _normalise_matrix(rows)
    symbol_types = [r.symbol_type for r in rows]
    file_paths = [r.file_path for r in rows]
    start_lines = [int(r.start_line) for r in rows]
    end_lines = [int(r.end_line) for r in rows]
    indexed_at = [int(r.indexed_at or now) for r in rows]
    # BUC-1518 C2: persist content_hash so future incremental re-indexes can
    # skip rows whose source range hasn't changed.  None on legacy callers
    # (Arrow handles None via nullable string).
    content_hashes = [r.content_hash for r in rows]

    embedding_type = pa.list_(pa.float32(), _EMBEDDING_DIM)
    embedding_array = pa.array(embeddings, type=embedding_type)

    tbl = pa.table(
        {
            "qualified_name": pa.array(qnames, type=pa.string()),
            "embedding": embedding_array,
            "symbol_type": pa.array(symbol_types, type=pa.string()),
            "file_path": pa.array(file_paths, type=pa.string()),
            "start_line": pa.array(start_lines, type=pa.int32()),
            "end_line": pa.array(end_lines, type=pa.int32()),
            "indexed_at": pa.array(indexed_at, type=pa.int64()),
            "content_hash": pa.array(content_hashes, type=pa.string()),
        }
    )

    placeholders = ",".join("?" for _ in qnames)

    conn.execute("BEGIN")
    try:
        conn.execute(
            f"DELETE FROM embeddings WHERE qualified_name IN ({placeholders})",
            qnames,
        )
        conn.register(_STAGING, tbl)
        try:
            conn.execute(
                f"""
                INSERT INTO embeddings
                    (qualified_name, embedding, symbol_type, file_path,
                     start_line, end_line, indexed_at, content_hash)
                SELECT qualified_name,
                       embedding::FLOAT[{_EMBEDDING_DIM}],
                       symbol_type,
                       file_path,
                       start_line,
                       end_line,
                       indexed_at,
                       content_hash
                FROM {_STAGING}
                """
            )
        finally:
            try:
                conn.unregister(_STAGING)
            except Exception:
                # unregister is best-effort; never let cleanup mask the
                # original error.
                pass
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return len(rows)
