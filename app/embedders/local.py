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

Embed-phase robustness (fix/embedding-phase-stall)
--------------------------------------------------
Two changes guard against the watchdog false-kill observed on 1654-file,
18707-node repos:

1. **Per-batch progress ticks** — ``_encode_sync`` processes the input list
   one batch at a time (``ENCODE_BATCH_SIZE=32``) and invokes an optional
   ``batch_callback`` after each batch.  The embed driver passes a callback
   that emits a ``PROGRESS`` line; the parent heartbeat thread tails that
   file and bumps ``job.last_progress_at`` so the watchdog sees a live job.

2. **Input truncation** — each text is capped at ``EMBED_MAX_CHARS``
   (4096 characters, conservative proxy for ~512 BPE tokens on e5-base-v2)
   before encode.  A minified / generated single-line file can be megabytes;
   without this cap a single ``encode()`` call can block for minutes on CPU,
   burning through the watchdog budget.

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
from typing import Any, Callable

from .base import EMBEDDING_DIM, EmbedderBackend, EmbedderError

logger = logging.getLogger(__name__)

#: HuggingFace model id. Pinned so the local backend stays bit-compatible
#: with the SageMaker / TEI endpoints (which all serve the same weights).
DEFAULT_MODEL = "intfloat/e5-base-v2"

#: Number of texts passed to one ``model.encode()`` call.  Matches the
#: embed driver's ``_BATCH`` so each outer flush triggers roughly one
#: ``encode()`` call → one progress tick → one heartbeat bump.
ENCODE_BATCH_SIZE = 32

#: Maximum character length for a single embed input text.  e5-base-v2
#: silently truncates at 512 BPE tokens (~2–4 chars each); feeding it a
#: megabyte-sized minified file wastes tokenisation time and can block a
#: CPU encode for tens of seconds.  4096 chars is a conservative cap that
#: keeps encode time predictable while retaining the meaningful leading
#: portion of any realistic function/class body.
EMBED_MAX_CHARS = 4096


class LocalEmbedder(EmbedderBackend):
    """Runs ``sentence-transformers`` in-process; 768-dim L2-normalised output.

    Construction is cheap — model loading is deferred until the first
    ``embed()`` call so ``get_embedder()`` can return immediately at
    startup. The model handle is then cached for the process lifetime.
    """

    name = "local"

    def __init__(
        self,
        model_name: str | None = None,
        dim: int | None = None,
    ) -> None:
        self.model: str = (
            model_name
            or os.environ.get("LOCAL_EMBED_MODEL")
            or DEFAULT_MODEL
        ).strip()
        # Allow an operator to override dim when running a non-default
        # sentence-transformers model. Defaults to e5-base-v2's 768.
        env_dim = os.environ.get("LOCAL_EMBED_DIM")
        if dim is not None:
            self.dim = int(dim)
        elif env_dim:
            try:
                self.dim = int(env_dim)
            except ValueError:
                self.dim = EMBEDDING_DIM
        else:
            self.dim = EMBEDDING_DIM
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

    def _truncate_texts(self, texts: list[str]) -> list[str]:
        """Cap each text at ``EMBED_MAX_CHARS`` and log a warning on truncation.

        e5-base-v2 silently truncates at ~512 BPE tokens.  Passing it a
        multi-megabyte string wastes tokenisation time and can wedge a CPU
        encode for tens of seconds, burning through the phase watchdog budget.
        This method makes the truncation explicit and loud so the operator
        knows which symbol triggered the cap.

        Args:
            texts: Raw embed input strings.

        Returns:
            The same list with any over-length strings replaced by their
            first ``EMBED_MAX_CHARS`` characters.
        """
        out: list[str] = []
        for t in texts:
            if len(t) > EMBED_MAX_CHARS:
                logger.warning(
                    "LocalEmbedder: input truncated from %d to %d chars "
                    "(model max ~512 tokens). Truncation is logged but does "
                    "not fail the embed.",
                    len(t),
                    EMBED_MAX_CHARS,
                )
                out.append(t[:EMBED_MAX_CHARS])
            else:
                out.append(t)
        return out

    def _encode_sync(
        self,
        texts: list[str],
        *,
        batch_callback: Callable[[int], None] | None = None,
    ) -> list[list[float]]:
        """Blocking sentence-transformers encode; processes one batch at a time.

        Each input text is truncated to ``EMBED_MAX_CHARS`` before encoding so
        pathologically long strings (minified files, generated code) cannot
        stall a single ``encode()`` call for minutes on CPU.

        The optional ``batch_callback`` is invoked with the cumulative embedded
        count after each batch completes.  The embed-driver passes a function
        that emits a ``PROGRESS`` line; the parent heartbeat thread tails the
        log and bumps ``job.last_progress_at`` on each new PROGRESS line,
        keeping the phase watchdog from false-killing a slow-but-alive encode.

        Args:
            texts: Input strings to embed.
            batch_callback: Optional callable invoked with the cumulative count
                of texts encoded so far, after each batch.  Never raises —
                any exception from the callback is swallowed.

        Returns:
            One 768-dim vector per input text, in input order.
        """
        if self._model is None:
            self._model = self._load_model()

        texts = self._truncate_texts(texts)

        result: list[list[float]] = []
        for batch_start in range(0, len(texts), ENCODE_BATCH_SIZE):
            batch = texts[batch_start : batch_start + ENCODE_BATCH_SIZE]
            vectors = self._model.encode(
                batch,
                batch_size=len(batch),
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            result.extend([float(x) for x in row] for row in vectors)
            if batch_callback is not None:
                try:
                    batch_callback(len(result))
                except Exception:  # noqa: BLE001 — callback failure must not kill the encode
                    pass

        return result

    async def embed(
        self,
        texts: list[str],
        *,
        batch_callback: Callable[[int], None] | None = None,
    ) -> list[list[float]]:
        """Embed a batch of texts asynchronously.

        Delegates the blocking encode to a worker thread via
        :func:`asyncio.to_thread`.

        Args:
            texts: Input strings to embed.  Empty list yields ``[]`` without
                loading the model.
            batch_callback: Forwarded to ``_encode_sync`` — called with the
                cumulative count after each internal batch.  See
                ``_encode_sync`` for semantics.

        Returns:
            One 768-dim L2-normalised vector per input text, in input order.

        Raises:
            EmbedderError: If the model fails to load or produces a
                dimension mismatch.
        """
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

        vectors = await asyncio.to_thread(
            self._encode_sync, texts, batch_callback=batch_callback
        )

        # Defensive: enforce protocol contract loudly so a future model swap
        # cannot silently corrupt the 768-dim DuckDB schema.
        for i, vec in enumerate(vectors):
            if len(vec) != self.dim:
                raise EmbedderError(
                    f"LocalEmbedder produced {len(vec)}-dim vector for input {i}; "
                    f"expected {self.dim} (model={self.model!r})"
                )
        return vectors
