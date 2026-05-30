"""Centralised, bounded LadybugDB (Kùzu fork) buffer-pool sizing.

LadybugDB / Kùzu's ``Database(buffer_pool_size=0)`` default auto-sizes the
buffer-manager memory map to roughly 80% of *physical* RAM.  On a box
co-tenanted with a 24 GB local LLM that over-reservation cannot get its
mmap and the engine hard-fails at open time with::

    Buffer manager exception: Mmap for size 8796093022208 failed

(the 8 TiB figure is Kùzu's ``max_db_size`` virtual reservation; the real
failure is that the requested *buffer pool* mapping can't be backed under
memory pressure).  Once that happens, every graph / structural / path query
breaks for *every* repo until the service is restarted — silently degrading
retrieval to vector-only.

The fix is to ALWAYS open the database with an explicit, BOUNDED buffer pool
so it degrades gracefully (spilling / smaller cache) instead of asking the
OS for an unbacked map.  The bound is sourced from ``KUZU_BUFFER_POOL_SIZE``
(bytes) and clamped to a sane default when unset / invalid.

This module is the single source of truth for that sizing.  Every
``lb.Database(...)`` open site in the codebase passes
``buffer_pool_size=resolve_buffer_pool_size()`` so the cap is consistent.

Env var:

    KUZU_BUFFER_POOL_SIZE   Buffer-pool size in BYTES.  Must be a positive
                            integer.  Unset / empty / non-numeric /
                            non-positive falls back to ``DEFAULT_BUFFER_POOL_SIZE``.

The kwarg name was verified against real_ladybug 0.15.3:
``Database.__init__(..., buffer_pool_size: int = 0, ...)`` — 0 means
"auto-size", which is exactly the unbounded behaviour we are replacing.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

#: Environment variable controlling the bounded buffer-pool size (bytes).
ENV_VAR = "KUZU_BUFFER_POOL_SIZE"

#: Safe bounded default (2 GiB) used when the env var is unset or invalid.
#: 2 GiB comfortably holds the graph working set for our largest indexed
#: repos while leaving headroom on a box that also hosts a ~24 GB local LLM.
#: It is a fixed cap rather than a fraction of RAM precisely so the engine
#: never over-reserves under memory pressure.
DEFAULT_BUFFER_POOL_SIZE = 2 * 1024 * 1024 * 1024  # 2_147_483_648 bytes


def resolve_buffer_pool_size() -> int:
    """Resolve the bounded LadybugDB buffer-pool size in bytes.

    Reads ``KUZU_BUFFER_POOL_SIZE`` from the environment.  A valid positive
    integer is returned as-is; anything else (unset, empty, non-numeric,
    zero, or negative) falls back to :data:`DEFAULT_BUFFER_POOL_SIZE`.

    The value is read live from ``os.environ`` on every call so tests and
    operators can override it without re-importing the module.

    Returns:
        int: A strictly positive buffer-pool size in bytes.  Never returns
        0 (Kùzu's "auto-size to ~80% of RAM" sentinel), which is the
        unbounded behaviour this helper exists to prevent.
    """
    raw = os.environ.get(ENV_VAR)
    if raw is None or raw.strip() == "":
        return DEFAULT_BUFFER_POOL_SIZE

    try:
        value = int(raw.strip())
    except (TypeError, ValueError):
        logger.warning(
            "%s=%r is not an integer — falling back to default %d bytes.",
            ENV_VAR,
            raw,
            DEFAULT_BUFFER_POOL_SIZE,
        )
        return DEFAULT_BUFFER_POOL_SIZE

    if value <= 0:
        logger.warning(
            "%s=%d is non-positive (0 = Kùzu auto-size, the unbounded "
            "behaviour we are guarding against) — falling back to default "
            "%d bytes.",
            ENV_VAR,
            value,
            DEFAULT_BUFFER_POOL_SIZE,
        )
        return DEFAULT_BUFFER_POOL_SIZE

    return value
