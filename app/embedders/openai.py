"""OpenAI embeddings HTTP backend (BYO embedder).

Calls OpenAI's ``/v1/embeddings`` endpoint via the official ``openai`` SDK.
This backend is the recommended "bring your own" option for operators who
want to run the Code Indexer Service standalone without provisioning AWS
SageMaker, downloading a 440MB sentence-transformers model, or running a
GPU TEI sidecar — just paste an API key and go.

Models
------
The backend auto-detects the model from ``OPENAI_EMBED_MODEL`` (default
``text-embedding-3-small``). The two production-grade options are:

* ``text-embedding-3-small`` — 1536-dim, $0.02 / 1M tokens (default).
  Beats the legacy ``text-embedding-ada-002`` on every MTEB axis while
  being 5x cheaper.
* ``text-embedding-3-large`` — 3072-dim, $0.13 / 1M tokens. Best-in-class
  retrieval quality; pick this if accuracy beats latency / cost.

Both 3-series models support custom output dimensions via the ``dimensions``
parameter (Matryoshka representation learning). This backend exposes that
via ``OPENAI_EMBED_DIM`` — set it to e.g. ``1024`` to truncate
``text-embedding-3-small`` from 1536 to 1024 dim, which is useful for
matching an existing vector-index schema.

Schema compatibility
--------------------
The default Code Indexer DuckDB schema is ``FLOAT[768]``. None of the
OpenAI models default to 768; if you switch to this backend you MUST
either:

1. Force the dim via ``OPENAI_EMBED_DIM=768`` (only works for 3-series
   models — Matryoshka truncation), OR
2. Re-create the DuckDB tables with the new dim (delete the per-repo
   ``.duck`` files and re-index).

The ``/health`` endpoint surfaces the configured dim so an operator can
verify the index/embedder pair match before searching.

Configuration
-------------
::

    OPENAI_API_KEY        Required. Standard OpenAI API key (sk-...).
    OPENAI_EMBED_MODEL    Default ``text-embedding-3-small``.
                          Also accepts ``text-embedding-3-large``.
    OPENAI_EMBED_DIM      Optional. Override the model's native dim
                          (3-series only — uses Matryoshka truncation).
    OPENAI_BASE_URL       Optional. For OpenAI-compatible gateways
                          (e.g. Azure OpenAI, vLLM, LiteLLM proxy).
    OPENAI_EMBED_BATCH_SIZE  Inputs per request (default 96).
                             OpenAI's hard limit is 2048 inputs / 300k tokens
                             per call; 96 keeps requests well under both.
    OPENAI_TIMEOUT_S      Hard per-request timeout (default 30.0).
"""
from __future__ import annotations

import logging
import os
from typing import Any

from .base import EmbedderBackend, EmbedderError

logger = logging.getLogger(__name__)

#: Default model — 1536-dim, cheap, beats ada-002 on every MTEB axis.
DEFAULT_MODEL = "text-embedding-3-small"

#: Native output dims for the production OpenAI embedding models. Used both
#: to fill in :attr:`OpenAIEmbedder.dim` when ``OPENAI_EMBED_DIM`` is unset
#: and to validate that custom dims don't exceed the model's native size.
_NATIVE_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    # Legacy — kept for completeness. Most operators should pick a 3-series
    # model instead.
    "text-embedding-ada-002": 1536,
}

_DEFAULT_BATCH_SIZE = 96
_DEFAULT_TIMEOUT_S = 30.0


class OpenAIEmbedder(EmbedderBackend):
    """OpenAI embeddings client.

    Lazy-imports the ``openai`` package so installs without the
    ``[byo]`` extra still boot cleanly when this backend isn't selected.
    """

    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        dim: int | None = None,
        base_url: str | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        if not api_key:
            raise EmbedderError(
                "EMBEDDER_BACKEND=openai requires OPENAI_API_KEY to be set."
            )
        self.api_key = api_key
        self.model: str = model
        self.base_url = base_url or None
        # OpenAI's hard cap is 2048 inputs / 300k tokens. 96 keeps us
        # under both with realistic ~1000-char code chunks.
        self.batch_size = min(max(1, batch_size), 2048)
        self.timeout_s = max(1.0, float(timeout_s))

        native_dim = _NATIVE_DIMS.get(model)
        if dim is not None:
            self.dim = int(dim)
            # Matryoshka truncation only works downward and only on 3-series
            # models; reject obviously broken configurations loudly.
            if native_dim is not None and self.dim > native_dim:
                raise EmbedderError(
                    f"OPENAI_EMBED_DIM={self.dim} exceeds {model!r}'s native "
                    f"dim ({native_dim}). Pick a smaller value or omit "
                    f"OPENAI_EMBED_DIM to use the native dim."
                )
        elif native_dim is not None:
            self.dim = native_dim
        else:
            # Unknown model (custom fine-tune, Azure deployment, etc.) — defer
            # dim discovery to the first embed() call.
            self.dim = 0

        self._client: Any | None = None

    @classmethod
    def from_env(cls) -> "OpenAIEmbedder":
        """Build the backend from process env vars.

        Raises:
            EmbedderError: when ``OPENAI_API_KEY`` is unset. Other env
                vars have sensible defaults so a key is the only required
                input for the happy path.
        """
        api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise EmbedderError(
                "EMBEDDER_BACKEND=openai but OPENAI_API_KEY is unset. "
                "Set it to your OpenAI API key (sk-...) and restart."
            )
        model = (os.environ.get("OPENAI_EMBED_MODEL") or DEFAULT_MODEL).strip()
        base_url = (os.environ.get("OPENAI_BASE_URL") or "").strip() or None

        dim_env = os.environ.get("OPENAI_EMBED_DIM")
        try:
            dim = int(dim_env) if dim_env else None
        except ValueError:
            raise EmbedderError(
                f"OPENAI_EMBED_DIM={dim_env!r} is not an integer."
            ) from None

        try:
            batch_size = int(
                os.environ.get("OPENAI_EMBED_BATCH_SIZE") or _DEFAULT_BATCH_SIZE
            )
        except (TypeError, ValueError):
            batch_size = _DEFAULT_BATCH_SIZE

        try:
            timeout_s = float(
                os.environ.get("OPENAI_TIMEOUT_S") or _DEFAULT_TIMEOUT_S
            )
        except (TypeError, ValueError):
            timeout_s = _DEFAULT_TIMEOUT_S

        return cls(
            api_key=api_key,
            model=model,
            dim=dim,
            base_url=base_url,
            batch_size=batch_size,
            timeout_s=timeout_s,
        )

    def _get_client(self) -> Any:
        """Lazy-init the ``openai.OpenAI`` client.

        Deferred so the openai package is only imported when this backend
        is actually selected — keeps service boot fast for the default
        ``local`` and ``sagemaker`` paths.
        """
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise EmbedderError(
                "EMBEDDER_BACKEND=openai requires the 'openai' package. "
                "Install it with: uv sync --extra byo  "
                "(or: uv pip install 'openai>=1.0')"
            ) from exc

        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout_s,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    def _embed_batch_sync(self, chunk: list[str]) -> list[list[float]]:
        """One blocking ``embeddings.create`` call. Wrapped via to_thread."""
        client = self._get_client()
        kwargs: dict[str, Any] = {"model": self.model, "input": chunk}
        # Only pass `dimensions` when the operator explicitly opted into a
        # non-native size — passing the native dim is a no-op but does emit
        # a warning on some OpenAI-compatible gateways.
        native = _NATIVE_DIMS.get(self.model)
        if self.dim and native is not None and self.dim != native:
            kwargs["dimensions"] = self.dim

        try:
            resp = client.embeddings.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 — surface every SDK error path
            raise EmbedderError(
                f"OpenAIEmbedder.embeddings.create failed "
                f"({type(exc).__name__}: {exc})"
            ) from exc

        data = getattr(resp, "data", None)
        if data is None or len(data) != len(chunk):
            raise EmbedderError(
                f"OpenAIEmbedder returned "
                f"{len(data) if data is not None else 'no'} embeddings for "
                f"{len(chunk)} inputs"
            )

        result: list[list[float]] = []
        for i, item in enumerate(data):
            vec = list(getattr(item, "embedding", None) or [])
            if not vec:
                raise EmbedderError(
                    f"OpenAIEmbedder: empty embedding for input {i}"
                )
            # First call sets self.dim for unknown models; subsequent calls
            # enforce it.
            if self.dim == 0:
                self.dim = len(vec)
            elif len(vec) != self.dim:
                raise EmbedderError(
                    f"OpenAIEmbedder returned {len(vec)}-dim vector for "
                    f"input {i}; expected {self.dim} (model={self.model!r})"
                )
            result.append([float(v) for v in vec])
        return result

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        import asyncio

        results: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            batch = await asyncio.to_thread(self._embed_batch_sync, chunk)
            results.extend(batch)
        return results
