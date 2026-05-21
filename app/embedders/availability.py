"""Embedder availability probe + cached status surfaced in ``/health``.

A dev box can boot the Code Indexer with ``EMBEDDER_BACKEND=local`` while
``sentence-transformers`` is **not** installed (the optional ``[local-embed]``
extras group). In that state semantic search silently 503s with
``in-process embedder not initialised`` and retrieval recall collapses to
0%. This module makes that failure mode LOUD:

* :func:`probe_embedder` is called once at startup (see ``app.main.lifespan``)
  and records the result — backend name, configured dim, availability, last
  error, fallback flags — into a module-level :data:`_status` dict. Cached
  for the process lifetime so ``/health`` calls are O(1).
* :func:`current_status` returns the cached snapshot as an
  :class:`~app.models.EmbedderStatus` payload for the health response.
* :func:`emit_startup_warning` prints the operator-visible banner when
  no backend is reachable AND no LM Studio embed fallback is configured.

The probe is intentionally fail-soft: ``get_embedder()`` raising
``EmbedderError`` (or anything else) does NOT block startup. The service
must continue running so structural search, /health probes, and re-indexes
can still operate while the operator fixes the install.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

from . import get_embedder
from .base import EMBEDDING_DIM, EmbedderError

logger = logging.getLogger(__name__)


# Module-level cache, populated by :func:`probe_embedder` at startup. Read
# by ``/health`` on every request. Tests can reset via
# :func:`reset_for_tests`.
_status: dict[str, Any] = {
    "backend": (os.environ.get("EMBEDDER_BACKEND") or "local").strip().lower(),
    "model": "",
    "dim": 0,
    "configured": False,
    "error": None,
    "available": False,
    "last_error": None,
    "fallback_lm_studio": False,
    "last_check_at": None,
    "check_latency_ms": None,
}


def _validate_backend_dependency(backend_name: str) -> None:
    """Verify the selected backend's heavy dependency is reachable.

    Cheap by design — never triggers a model download or a real
    inference call. For ``local`` this just imports
    ``sentence_transformers``; for ``sagemaker`` it imports ``boto3`` and
    checks the endpoint env var; for ``tei`` it does NOT pre-flight the
    HTTP sidecar (the TEI URL may be wrong but reachable later — let the
    first ``embed()`` surface that).

    Raises:
        EmbedderError: When the dependency is missing or misconfigured.
            Wraps the original exception via ``__cause__`` so the probe
            can walk the chain and surface the root error to the operator.
    """
    if backend_name == "local":
        try:
            import sentence_transformers  # noqa: F401 — import-only check
        except ImportError as exc:
            raise EmbedderError(
                "EMBEDDER_BACKEND=local requires the 'sentence-transformers' "
                "package (uv sync --group local-embed)."
            ) from exc
    elif backend_name == "sagemaker":
        endpoint = (
            os.environ.get("SAGEMAKER_ENDPOINT_NAME")
            or os.environ.get("SAGEMAKER_EMBED_URL")
        )
        if not endpoint:
            raise EmbedderError(
                "EMBEDDER_BACKEND=sagemaker requires SAGEMAKER_ENDPOINT_NAME "
                "(or legacy SAGEMAKER_EMBED_URL) to be set."
            )
        try:
            import boto3  # noqa: F401 — import-only check
        except ImportError as exc:
            raise EmbedderError(
                "EMBEDDER_BACKEND=sagemaker requires the 'boto3' package."
            ) from exc
    elif backend_name == "tei":
        if not os.environ.get("TEI_URL"):
            raise EmbedderError(
                "EMBEDDER_BACKEND=tei requires TEI_URL to be set."
            )
        try:
            import httpx  # noqa: F401 — import-only check
        except ImportError as exc:
            raise EmbedderError(
                "EMBEDDER_BACKEND=tei requires the 'httpx' package."
            ) from exc


def _probe_lm_studio_fallback() -> bool:
    """Return True iff LM Studio is configured AND has an embed model loaded.

    Defensive: any import or probe failure collapses to False so a broken
    LM Studio install can never crash startup.
    """
    try:
        from ..services import lm_studio

        if not lm_studio.base_url():
            return False
        return bool(lm_studio.can_embed())
    except Exception as exc:  # noqa: BLE001 — defensive
        logger.debug("LM Studio fallback probe failed: %s", exc)
        return False


def probe_embedder() -> dict[str, Any]:
    """Probe the configured embedder backend and cache the outcome.

    Called once at startup (``app.main.lifespan``); safe to call again from
    tests after :func:`reset_for_tests`. Mutates and returns the module-
    level :data:`_status` dict so ``/health`` reflects the latest probe.

    Returns:
        dict[str, Any]: Snapshot of ``{backend, available, dim, last_error,
        fallback_lm_studio, last_check_at, check_latency_ms}``.
    """
    backend_name = (os.environ.get("EMBEDDER_BACKEND") or "local").strip().lower()
    t0 = time.monotonic()
    available = False
    configured = False
    dim: int = 0
    model: str = ""
    last_error: str | None = None

    try:
        backend = get_embedder()
        # Factory succeeded — record the legacy ``configured/model/dim``
        # fields immediately so callers reading those keep working even if
        # the dep-validation step below flips ``available`` to False.
        configured = backend is not None
        if configured:
            backend_name = getattr(backend, "name", backend_name) or backend_name
            model = str(getattr(backend, "model", "") or "")
            dim = int(getattr(backend, "dim", EMBEDDING_DIM) or EMBEDDING_DIM)
        # Construction succeeded, but several backends (notably ``local``)
        # defer the heavy work — ``LocalEmbedder.__init__`` only sets
        # attributes; the ``sentence_transformers`` import happens inside
        # ``_load_model`` on the first ``embed()`` call. A bare
        # ``get_embedder()`` therefore reports ``available=true`` even when
        # the optional ``[local-embed]`` extras group is missing, which is
        # exactly the silent-503 bug we are trying to surface.
        #
        # Probe the *real* dependency without paying the model-download
        # cost: try the import (or analogous lightweight check) for the
        # selected backend. Failures here flip ``available`` to ``false``
        # with the captured ``last_error``.
        _validate_backend_dependency(backend_name)
        available = configured
    except EmbedderError as exc:
        last_error = f"{type(exc).__name__}: {exc}"
        # Walk the cause chain so the operator sees the *root* import
        # error (e.g. ``ModuleNotFoundError: No module named
        # 'sentence_transformers'``) rather than just the wrapper message.
        cause = exc.__cause__
        if cause is not None:
            last_error = f"{type(cause).__name__}: {cause}"
    except Exception as exc:  # noqa: BLE001 — never let startup crash here
        last_error = f"{type(exc).__name__}: {exc}"

    latency_ms = (time.monotonic() - t0) * 1000.0

    _status.update(
        {
            "backend": backend_name,
            "model": model,
            "dim": dim,
            "configured": configured,
            # ``error`` (legacy field) mirrors the construction-only error
            # when the factory itself raised; falls back to last_error
            # otherwise so the PR #69 field stays meaningful.
            "error": last_error if not configured else None,
            "available": available,
            "last_error": last_error,
            "fallback_lm_studio": _probe_lm_studio_fallback(),
            "last_check_at": datetime.now(timezone.utc).isoformat(),
            "check_latency_ms": round(latency_ms, 2),
        }
    )
    return dict(_status)


def current_status() -> dict[str, Any]:
    """Return a snapshot of the cached embedder status.

    Returns:
        dict[str, Any]: Shallow copy of the module-level status dict.
        Empty/default values when :func:`probe_embedder` has not yet run.
    """
    return dict(_status)


def emit_startup_warning(status: dict[str, Any] | None = None) -> None:
    """Print the operator-visible banner when no embedder is reachable.

    The banner goes to ``stderr`` AND the structured logger (``logger.error``
    with an ``action_required`` extra field) so it's loud in both
    development (terminal) and production (CloudWatch / journald).

    Args:
        status: Optional override; defaults to the cached module status.
            Tests inject a synthetic dict.
    """
    s = status if status is not None else current_status()

    # If either the primary backend OR the LM Studio dev fallback is
    # available the operator can still get embeddings — no banner.
    if s.get("available") or s.get("fallback_lm_studio"):
        return

    backend = s.get("backend") or "local"
    last_error = s.get("last_error") or "(no error captured)"

    banner = (
        "\n"
        "====================================================================\n"
        "WARN  Code Indexer started but NO EMBEDDER IS AVAILABLE.\n"
        "Semantic search will return 503 for every query.\n"
        "\n"
        f"EMBEDDER_BACKEND={backend}\n"
        f"last_error: {last_error}\n"
        "\n"
        "Fix:\n"
        "  - For local dev:  uv sync --group local-embed\n"
        "  - For SageMaker:  set AWS creds + EMBEDDER_BACKEND=sagemaker\n"
        "                    + SAGEMAKER_ENDPOINT_NAME=forge-e5-embed-v2\n"
        "  - For TEI:        start TEI sidecar + EMBEDDER_BACKEND=tei\n"
        "                    + TEI_URL=http://localhost:8080\n"
        "====================================================================\n"
    )
    # The banner intentionally uses the U+26A0 WARNING SIGN to make the
    # "your install is broken" signal hard to miss on a busy terminal.
    # Operator visibility outweighs the project-wide no-emoji style rule
    # here — every other surface stays clean.
    print("⚠ EMBEDDER UNAVAILABLE", banner, sep="\n", flush=True)  # noqa: T201
    logger.error(
        "embedder unavailable at startup (backend=%s): %s",
        backend,
        last_error,
        extra={
            "action_required": (
                "Install an embedder backend. For local dev run: "
                "uv sync --group local-embed"
            ),
            "embedder_backend": backend,
        },
    )


def reset_for_tests() -> None:
    """Reset the cached status so tests can re-probe with a fresh state.

    Intentionally NOT exported in ``__all__`` — production callers should
    never reach into the cache.
    """
    _status.clear()
    _status.update(
        {
            "backend": (os.environ.get("EMBEDDER_BACKEND") or "local").strip().lower(),
            "model": "",
            "dim": 0,
            "configured": False,
            "error": None,
            "available": False,
            "last_error": None,
            "fallback_lm_studio": False,
            "last_check_at": None,
            "check_latency_ms": None,
        }
    )


__all__ = [
    "current_status",
    "emit_startup_warning",
    "probe_embedder",
]
