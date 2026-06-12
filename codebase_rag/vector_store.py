"""Compatibility shim — delegates to ``codebase_rag.storage.vector_store``.

Current backend: **DuckDB** (one ``.duck`` file per repo, FLOAT[768] embeddings
keyed by ``qualified_name``, cosine via ``array_cosine_distance``). This is
the v5.3 §6.5 + §8.4 mandated store and the only path consulted at query time.

Historical context
------------------
This module's original implementation persisted embeddings to a pair of sidecar
numpy files (``{slug}.embeddings.npy`` + ``{slug}.embeddings_idx.json``); a
brief intermediate sqlite-vec design was rejected per v5.3 §17 before the
DuckDB cutover. Both predecessors are gone; only this shim's public API
remains for legacy in-process callers.

Two production callers continued importing from this legacy path —
``graph_updater._generate_semantic_embeddings()`` (the in-process embedding
pass used by the ``cgr`` CLI / realtime updater) and
``codebase_rag.mcp.tools._cleanup_project_embeddings()``.  ``code-indexer-service``
already bypasses this module entirely (it sets ``skip_embeddings=True`` and
runs an external embedder subprocess that writes directly to the ``.duck``
store via ``storage.vector_store``).

Rather than rewrite the call-sites in ``graph_updater`` and ``mcp/tools`` and
risk a subtle behaviour drift, this shim re-implements the original public API
on top of the DuckDB store so all writes/reads end up in the same per-repo
``.duck`` file the rest of the system already uses.

Public API preserved (signatures match the previous numpy implementation):
    store_embedding(node_id, vector, qualified_name)
    store_embedding_batch(embeddings)            # 3-tuple OR 2-tuple
    flush_embeddings(db_path=None) -> int
    search_embeddings(query_vector, k=10, db_path=None, top_k=None)
    verify_stored_ids(expected_ids, db_path=None)
    delete_project_embeddings(project_name, node_ids, db_path=None)

Path resolution mirrors ``tools.semantic_search``: a ``.db`` LadybugDB path is
mapped to its sibling ``.duck`` file.  Callers that pass a ``.duck`` path
directly are honoured unchanged.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_duck_path(db_path: str | None) -> str | None:
    """Map a LadybugDB ``.db`` path to its sibling ``.duck`` file.

    Returns ``None`` when no path is configured.  ``.duck`` paths are returned
    unchanged so test fixtures and direct callers keep working.
    """
    path = db_path or settings.LADYBUG_DB_PATH
    if not path:
        return None
    if path.endswith(".db"):
        return path[:-3] + ".duck"
    if path.endswith(".duck"):
        return path
    # Defensive: prefer an explicit `.duck` sibling if the configured path is
    # something else entirely (e.g. an old test fixture pointing at a directory).
    return path + ".duck"


# In-memory accumulator preserved for API compatibility with callers that
# expect ``store_embedding_batch`` followed by ``flush_embeddings``.  The
# legacy contract permitted accumulating without a known db_path and then
# resolving it at flush time, so we mirror that here.
_pending: dict[str, list[float]] = {}


def _flush_pending_to(duck_path: str) -> int:
    """Write the in-memory ``_pending`` dict to the given ``.duck`` file.

    Returns the number of rows written.  Clears ``_pending`` on success.
    """
    if not _pending:
        return 0

    from .storage.vector_store import EmbeddingRow, bulk_insert, open_or_create

    rows = [
        EmbeddingRow(
            qualified_name=qn,
            embedding=list(vec),
            file_path="",
            start_line=0,
            end_line=0,
            symbol_type="",
        )
        for qn, vec in _pending.items()
    ]

    conn = open_or_create(duck_path)
    try:
        written = bulk_insert(conn, rows)
    finally:
        conn.close()

    _pending.clear()
    logger.info("Flushed %d embeddings to %s", written, duck_path)
    return written


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def store_embedding(
    node_id: object,  # noqa: ARG001  (legacy param, unused)
    vector: list[float] | None,
    qualified_name: str,
) -> None:
    """Accumulate a single embedding in memory.

    ``flush_embeddings()`` must be called afterwards to persist the rows to
    the per-repo ``.duck`` file.  ``node_id`` is accepted for API compatibility
    with the original Qdrant-backed store but is ignored — ``qualified_name``
    is the lookup key.
    """
    if vector is not None:
        _pending[qualified_name] = list(vector)


def store_embedding_batch(
    embeddings: list[tuple[Any, ...]],
    *args: object,  # noqa: ARG001
    **kwargs: object,  # noqa: ARG001
) -> int:
    """Accumulate a batch of embeddings in memory.

    Accepts both the legacy 3-tuple signature ``(node_id, vector, qualified_name)``
    used by ``graph_updater`` and the modern 2-tuple ``(qualified_name, vector)``.
    Returns the number of non-``None`` vectors accepted.
    """
    count = 0
    for item in embeddings:
        if len(item) == 3:
            _node_id, vec, qn = item
        else:
            qn, vec = item[0], item[1]

        if vec is not None:
            _pending[qn] = list(vec)
            count += 1

    logger.debug(
        "Accumulated %d embeddings in memory (total pending: %d)", count, len(_pending)
    )
    return count


def flush_embeddings(db_path: str | None = None) -> int:
    """Persist accumulated embeddings to the per-repo ``.duck`` file."""
    duck_path = _resolve_duck_path(db_path)
    if not duck_path:
        logger.warning("flush_embeddings: no db_path configured; dropping %d pending", len(_pending))
        _pending.clear()
        return 0

    return _flush_pending_to(duck_path)


def search_embeddings(
    query_vector: list[float],
    k: int = 10,
    db_path: str | None = None,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Top-k cosine-similarity search over the per-repo ``.duck`` file.

    Mirrors the previous numpy-backed return shape ``[(qualified_name, score)]``
    so callers remain unchanged.  In-memory ``_pending`` rows are flushed first
    so a search immediately after ``store_embedding_batch()`` still sees them
    (matches the legacy contract).
    """
    effective_k = top_k if top_k is not None else k
    duck_path = _resolve_duck_path(db_path)

    if not duck_path:
        logger.warning("search_embeddings: no db_path configured")
        return []

    if _pending:
        _flush_pending_to(duck_path)

    if not Path(duck_path).exists():
        logger.warning("search_embeddings: no embedding file at %s", duck_path)
        return []

    from .storage.vector_store import open_or_create, search_similar

    conn = open_or_create(duck_path)
    try:
        results = search_similar(conn, list(query_vector), k=effective_k)
    finally:
        conn.close()

    return [(r.qualified_name, float(r.score)) for r in results]


def verify_stored_ids(
    expected_ids: set[object],
    db_path: str | None = None,
) -> set[object]:
    """Return the subset of ``expected_ids`` that have stored embeddings.

    The legacy contract treated integer IDs as opaque (Qdrant point IDs) and
    passed them through unchanged.  String IDs are checked against both the
    in-memory ``_pending`` accumulator and the on-disk DuckDB store.
    """
    if not expected_ids:
        return set()

    int_ids = {i for i in expected_ids if isinstance(i, int)}
    str_ids = {i for i in expected_ids if isinstance(i, str)}

    if not str_ids:
        return int_ids

    found_in_pending = {qn for qn in str_ids if qn in _pending}

    duck_path = _resolve_duck_path(db_path)
    found_on_disk: set[object] = set()
    if duck_path and Path(duck_path).exists():
        from .storage.vector_store import open_or_create
        from .storage.vector_store import verify_stored_ids as _verify_stored

        conn = open_or_create(duck_path)
        try:
            found_on_disk = set(_verify_stored(conn, str_ids))
        finally:
            conn.close()

    return int_ids | found_in_pending | found_on_disk


def delete_project_embeddings(
    project_name: str,
    node_ids: list[object],  # noqa: ARG001  (legacy param, unused)
    db_path: str | None = None,
) -> None:
    """Delete every embedding whose ``qualified_name`` belongs to ``project_name``.

    Project ownership is determined by the legacy prefix convention
    (``"<project_name>."`` / exact match).  ``node_ids`` is accepted for API
    compatibility with the old Qdrant-backed store but is unused.
    """
    prefix = project_name + "."

    # Drop matching pending rows first so a subsequent flush doesn't resurrect them.
    for qn in list(_pending.keys()):
        if qn.startswith(prefix) or qn == project_name:
            del _pending[qn]

    duck_path = _resolve_duck_path(db_path)
    if not duck_path or not Path(duck_path).exists():
        return

    from .storage.vector_store import open_or_create

    conn = open_or_create(duck_path)
    try:
        # DuckDB has no native LIKE-prefix delete helper here; do it inline
        # so we don't have to add a public method to the typed store API just
        # for this legacy shim.
        try:
            conn.execute("BEGIN")
            removed_row = conn.execute(
                "SELECT count(*) FROM embeddings "
                "WHERE qualified_name = ? OR qualified_name LIKE ?",
                (project_name, prefix + "%"),
            ).fetchone()
            removed = int(removed_row[0]) if removed_row else 0
            conn.execute(
                "DELETE FROM embeddings "
                "WHERE qualified_name = ? OR qualified_name LIKE ?",
                (project_name, prefix + "%"),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()

    if removed:
        logger.info("Deleted %d embeddings for project '%s'", removed, project_name)
