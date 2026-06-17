"""Centralised LadybugDB connection helpers.

LadybugDB takes a single-writer file lock when a ``Database`` is opened in
read-write mode (the default).  Read-only consumers — every router that
serves ``GET`` traffic — used to open the same lock, which meant a
``POST /index`` job blocked all subsequent reads for the duration of the
re-index with::

    IO exception: Could not set lock on file ...
    See https://docs.ladybugdb.com/concurrency

The fix is to give read paths a connection opened with ``read_only=True``,
which acquires only a shared lock and coexists with the writer.  The
indexer subprocess template at ``app/routers/index.py`` line ~1019
already does this; this module exposes the same primitive to API routers
so we have exactly one call site for the kwarg.

Public surface:

    open_read_conn(db_path)  -> (db, conn)   read-only — for /search, /repos, /health, /context-bundle
    open_rw_conn(db_path)    -> (db, conn)   read-write — for indexer/embed/stamp paths only

Both helpers return the underlying ``Database`` alongside the
``Connection`` so callers can ``del`` / ``close()`` both before spawning a
subprocess that needs the file lock (see the GC dance in
``index.py`` around the ``_blocking_embed`` call site).
"""
from __future__ import annotations

from typing import Any, Tuple


def open_read_conn(db_path: str) -> Tuple[Any, Any]:
    """Open a LadybugDB connection in read-only mode.

    Read-only mode acquires a shared file lock, so multiple readers can
    coexist with a single writer (BUC-1571 — reads previously blocked
    while ``POST /index`` held the exclusive lock).

    Args:
        db_path: Filesystem path to the ``.db`` file.

    Returns:
        Tuple of ``(Database, Connection)``.  Callers are responsible for
        closing the connection (and dropping references to the database
        before any subprocess that needs the file).
    """
    import ladybug as lb  # type: ignore[import-untyped]

    from .ladybug_buffer_pool import resolve_buffer_pool_size

    db = lb.Database(
        db_path, read_only=True, buffer_pool_size=resolve_buffer_pool_size()
    )
    conn = lb.Connection(db)
    return db, conn


def open_rw_conn(db_path: str) -> Tuple[Any, Any]:
    """Open a LadybugDB connection in read-write mode.

    Use sparingly — only the ingester, embed-stamp path, and migrator
    need this.  HTTP read endpoints must use ``open_read_conn``.

    Args:
        db_path: Filesystem path to the ``.db`` file.

    Returns:
        Tuple of ``(Database, Connection)``.
    """
    import ladybug as lb  # type: ignore[import-untyped]

    from .ladybug_buffer_pool import resolve_buffer_pool_size

    db = lb.Database(db_path, buffer_pool_size=resolve_buffer_pool_size())
    conn = lb.Connection(db)
    return db, conn
