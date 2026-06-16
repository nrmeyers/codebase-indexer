"""Sync wrapper around the async ``EmbedderBackend`` protocol.

The BUC-1605 embedder factory (:mod:`app.embedders`) exposes an *async*
``embed(texts: list[str]) -> list[list[float]]`` method on every backend so
the FastAPI / asyncio hot path doesn't block on HTTP / GPU work.

Several pre-BUC-1605 call sites are synchronous (background warmup threads,
the search path's helper functions, the A/B embedder wrapper). Refactoring
all of them to async would balloon the migration; instead this module
provides two tiny synchronous shims:

* :func:`get_embedder_or_none` — returns the configured backend, or
  ``None`` when no backend is configured (the legacy
  ``get_sagemaker_embedder()`` contract). Tests and call sites that need
  "is the backend ready?" can branch on this without catching
  :class:`EmbedderError`.
* :func:`embed_text_sync` — synchronous single-text embed that returns
  ``list[float] | None`` (the legacy ``SageMakerEmbedder.embed(text)``
  contract). Internally runs the async ``embed`` on a private event loop
  via ``asyncio.run`` and unwraps the single-element batch.

Both helpers preserve the *exact* return shapes the pre-migration call
sites expected so the migration is a near-mechanical import swap rather
than a deep refactor.

When the call site is *already* async (FastAPI route handlers, background
asyncio tasks) prefer :func:`app.embedders.get_embedder` directly and
``await backend.embed([text])`` — skip this bridge.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from . import get_embedder
from .base import EmbedderBackend, EmbedderError

if TYPE_CHECKING:
    from .prefixes import Role

logger = logging.getLogger(__name__)


def get_embedder_or_none() -> EmbedderBackend | None:
    """Return the configured embedder, or ``None`` when none is configured.

    Mirrors the legacy ``get_sagemaker_embedder()`` contract: sync callers
    that need to *probe* whether an embedder is available (e.g. the
    semantic-search route deciding between SageMaker, LM Studio, and the
    in-process torch fallback) can call this without catching
    :class:`EmbedderError`.

    Configuration failures (e.g. ``EMBEDDER_BACKEND=sagemaker`` with no
    endpoint env var set) collapse to ``None`` with a one-line DEBUG log
    — the same fail-soft behaviour the legacy shim had.

    Returns:
        EmbedderBackend | None: The cached singleton backend, or ``None``
        when construction fails. Subsequent calls are O(1) (the underlying
        factory is ``lru_cache``-d).
    """
    try:
        return get_embedder()
    except EmbedderError as exc:
        logger.debug("get_embedder_or_none: backend unavailable: %s", exc)
        return None


def embed_text_sync(
    text: str,
    *,
    role: Role | None = None,
) -> list[float] | None:
    """Embed a single text synchronously; return a 768-dim vector or ``None``.

    Drop-in replacement for the legacy ``SageMakerEmbedder.embed(text)``
    contract used by the search path and the prewarm threads. Internally:

    1. Resolve the configured backend via :func:`get_embedder_or_none`.
       ``None`` → return ``None`` immediately (no error, no log).
    2. Apply the model's role-appropriate prefix when ``role`` is given
       (see :mod:`app.embedders.prefixes`).
    3. Run the async batched ``embed([text])`` on a fresh event loop
       (``asyncio.run``). The fresh-loop pattern is safe here because
       every call site that uses this helper is *outside* an asyncio
       context (worker threads, sync route handlers).
    4. Unwrap the single-element batch and return the vector.

    Failures (network, protocol mismatch) collapse to ``None`` with a
    WARN log — matching the legacy contract where transient SageMaker
    errors were swallowed and the caller fell back to LM Studio / torch.

    Args:
        text: Input string to embed. Empty strings short-circuit to
            ``None`` (mirrors the legacy behaviour; no network call).
        role: ``"query"`` for query-side callers (semantic search), so
            instruction-tuned local models (e5, CodeRankEmbed) get their
            required query prefix — symmetric with the ``"document"`` prefix
            the index pass applies. ``None`` (default) preserves the legacy
            raw-text contract for warmup probes and the v2 A/B embedder, and
            is also a no-op for prod backends and symmetric models.

    Returns:
        list[float] | None: 768-dim vector on success, ``None`` on
        any failure (backend not configured, network error, protocol
        violation).
    """
    if not text:
        return None

    backend = get_embedder_or_none()
    if backend is None:
        return None

    if role is None:
        inputs = [text]
    else:
        from .prefixes import apply_prefix

        inputs = apply_prefix(backend, [text], role=role)

    try:
        vectors = asyncio.run(backend.embed(inputs))
    except EmbedderError as exc:
        logger.warning("embed_text_sync: backend %s failed: %s", backend.name, exc)
        return None
    except Exception as exc:  # noqa: BLE001 — match legacy fail-soft behaviour
        logger.warning(
            "embed_text_sync: unexpected error from backend %s: %s",
            backend.name,
            exc,
        )
        return None

    if not vectors:
        return None
    return vectors[0]


__all__ = ["embed_text_sync", "get_embedder_or_none"]
