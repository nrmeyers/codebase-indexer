"""Pluggable embedder backends for the Code Indexer service.

Four interchangeable backends sit behind a single ``EmbedderBackend``
protocol so the indexer can run anywhere from a laptop (no AWS, no
model download) to a GPU box (TEI) to Navistone's production AWS
account (SageMaker):

    local       sentence-transformers in-process. Zero external deps.
                Default for standalone installs. 768-dim (e5-base-v2;
                local backend retains E5 until a Jina HF artifact ships).
    sagemaker   Navistone's AWS SageMaker Serverless Inference endpoint
                (jina-code-v2-serverless, us-east-1; was forge-e5-embed-v2,
                swapped 2026-05-26 LE-129). 768-dim. Default for the
                Navistone production deploy.
    tei         Hugging Face Text-Embeddings-Inference HTTP sidecar
                (http://localhost:8080). 768-dim. For GPU-batched
                embedding without AWS or local CPU load.
    openai      OpenAI ``/v1/embeddings``. 1536-dim (3-small, default) or
                3072-dim (3-large). Cheapest "bring your own" path — no
                local model download, no AWS account. Requires
                ``OPENAI_API_KEY``.

Selection
---------
The backend is chosen by the ``EMBEDDER_BACKEND`` env var (case-insensitive,
values ``local`` | ``sagemaker`` | ``tei``; default ``local``). Use
``get_embedder()`` for the module-level singleton; the factory is cached so
the heavy import / network probe only happens once per process.

Backends expose their output dim via ``embedder.dim``. The three
default backends (``local``, ``sagemaker``, ``tei``) all return 768-dim
``list[float]`` vectors so the existing LadybugDB / DuckDB ``FLOAT[768]``
schema needs zero migration. The ``openai`` backend produces 1536 or
3072 dim vectors and requires a schema migration before use (see
``docs/EMBEDDERS.md``).

Examples
--------
::

    from app.embedders import get_embedder

    embedder = get_embedder()
    vectors = await embedder.embed(["def foo(): pass", "class Bar: ..."])
    assert len(vectors[0]) == 768
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

from .base import EMBEDDING_DIM, EmbedderBackend, EmbedderError

logger = logging.getLogger(__name__)

DEFAULT_BACKEND = "local"
VALID_BACKENDS = ("local", "sagemaker", "tei", "openai", "llama_server")


def _resolve_backend_name() -> str:
    """Read ``EMBEDDER_BACKEND`` from env, normalise, validate.

    Unknown values log a one-time warning and fall back to ``local`` so a
    typo cannot silently break ingestion.
    """
    raw = (os.environ.get("EMBEDDER_BACKEND") or DEFAULT_BACKEND).strip().lower()
    if raw not in VALID_BACKENDS:
        logger.warning(
            "EMBEDDER_BACKEND=%r is not recognised; falling back to %s. "
            "Valid values: %s",
            raw,
            DEFAULT_BACKEND,
            ", ".join(VALID_BACKENDS),
        )
        return DEFAULT_BACKEND
    return raw


@lru_cache(maxsize=1)
def get_embedder() -> EmbedderBackend:
    """Return the configured embedder backend as a process-wide singleton.

    Cached so subsequent calls in hot paths skip the (potentially heavy)
    backend construction. Tests can call ``get_embedder.cache_clear()`` to
    re-read env vars between cases.

    Raises:
        EmbedderError: If the selected backend cannot be initialised (e.g.
            ``local`` requested but ``sentence-transformers`` is not
            installed; ``sagemaker`` requested but no endpoint configured).
    """
    name = _resolve_backend_name()
    if name == "local":
        from .local import LocalEmbedder

        return LocalEmbedder()
    if name == "sagemaker":
        from .sagemaker import SageMakerEmbedder

        return SageMakerEmbedder.from_env()
    if name == "tei":
        from .tei import TEIEmbedder

        return TEIEmbedder.from_env()
    if name == "openai":
        from .openai import OpenAIEmbedder

        return OpenAIEmbedder.from_env()
    if name == "llama_server":
        from .llama_server import LlamaServerEmbedder

        return LlamaServerEmbedder.from_env()
    # Unreachable — _resolve_backend_name normalises to a valid value.
    raise EmbedderError(f"Unknown EMBEDDER_BACKEND: {name!r}")


__all__ = [
    "EMBEDDING_DIM",
    "EmbedderBackend",
    "EmbedderError",
    "VALID_BACKENDS",
    "get_embedder",
]
