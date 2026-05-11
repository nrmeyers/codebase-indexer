"""In-process ``sentence-transformers`` backend (BUC-1605).

Runs ``intfloat/e5-base-v2`` on the host CPU/GPU with zero external
dependencies — perfect for standalone evaluation or a laptop install.

The first call downloads the model (~440 MB) into the HuggingFace cache
(``~/.cache/huggingface/hub`` by default) and may take 30-60s; subsequent
calls hit the cached weights and respond in tens of milliseconds per
batch on a modern CPU.

The blocking sentence-transformers ``encode()`` call is shoved into a
worker thread via :func:`asyncio.to_thread` so the FastAPI event loop
stays responsive.

Dependencies
------------
Requires the ``sentence-transformers`` package, which is in the optional
``[local-embed]`` extras group. If you set ``EMBEDDER_BACKEND=local``
without installing the extra you'll see :class:`EmbedderError` with the
exact pip command at startup.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from .base import EMBEDDING_DIM, EmbedderBackend, EmbedderError

logger = logging.getLogger(__name__)

#: HuggingFace model id. Pinned so the local backend stays bit-compatible
#: with the SageMaker / TEI endpoints (which all serve the same weights).
DEFAULT_MODEL = "intfloat/e5-base-v2"


class LocalEmbedder(EmbedderBackend):
    """Runs ``sentence-transformers`` in-process; 768-dim L2-normalised output.

    Construction is cheap — model loading is deferred until the first
    ``embed()`` call so ``get_embedder()`` can return immediately at
    startup. The model handle is then cached for the process lifetime.
    """

    name = "local"

    def __init__(self, model_name: str | None = None) -> None:
        self.model: str = (
            model_name
            or os.environ.get("LOCAL_EMBED_MODEL")
            or DEFAULT_MODEL
        ).strip()
        self._model: Any | None = None
        self._lock = asyncio.Lock()

    def _load_model(self) -> Any:
        """Import and instantiate ``SentenceTransformer`` lazily.

        Raises:
            EmbedderError: ``sentence-transformers`` is not installed, or
                the model failed to download / load.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbedderError(
                "EMBEDDER_BACKEND=local requires the 'sentence-transformers' "
                "package. Install it with: "
                "uv pip install 'code-indexer-service[local-embed]' "
                "(or: uv pip install 'sentence-transformers>=3.2')"
            ) from exc

        try:
            return SentenceTransformer(self.model)
        except Exception as exc:  # noqa: BLE001 — surface the original cause
            raise EmbedderError(
                f"LocalEmbedder failed to load model {self.model!r}: {exc}"
            ) from exc

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        """Blocking sentence-transformers encode call. Called via to_thread."""
        if self._model is None:
            self._model = self._load_model()
        # ``normalize_embeddings=True`` produces L2-normalised vectors so the
        # output matches the SageMaker / TEI endpoints (both also normalise
        # server-side). Returning ``list[list[float]]`` rather than ndarray
        # keeps the protocol JSON-serialisable end-to-end.
        vectors = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [[float(x) for x in row] for row in vectors]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # Serialise model construction under a lock so two concurrent first-
        # call requests don't both pay the 30-60s load cost.
        if self._model is None:
            async with self._lock:
                if self._model is None:
                    # ``_load_model`` is cheap once cached; offload anyway
                    # so the event loop doesn't stall on first call.
                    self._model = await asyncio.to_thread(self._load_model)

        vectors = await asyncio.to_thread(self._encode_sync, texts)

        # Defensive: enforce protocol contract loudly so a future model swap
        # cannot silently corrupt the 768-dim DuckDB schema.
        for i, vec in enumerate(vectors):
            if len(vec) != EMBEDDING_DIM:
                raise EmbedderError(
                    f"LocalEmbedder produced {len(vec)}-dim vector for input {i}; "
                    f"expected {EMBEDDING_DIM} (model={self.model!r})"
                )
        return vectors
