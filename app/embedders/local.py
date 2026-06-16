"""In-process ``sentence-transformers`` backend (BUC-1605).

Runs ``nomic-ai/nomic-embed-text-v1.5`` on the host CPU/GPU with zero
external dependencies — perfect for standalone evaluation or a laptop
install. Switched from ``intfloat/e5-base-v2`` after the 768-dim POC
(``.planning/runs/768-poc/``): nomic-v1.5 wins recall@10 by +3.9pp,
trades 1–2pp at @25/@50, and is instruction-tuned with Matryoshka
support for future dim truncation. Schema is unchanged (FLOAT[768]).

The first call downloads the model (~520 MB) into the HuggingFace cache
(``~/.cache/huggingface/hub`` by default) and may take 30-60s; subsequent
calls hit the cached weights and respond in tens of milliseconds per
batch on a modern CPU.

``nomic-bert`` ships custom modeling code, so the backend auto-enables
``trust_remote_code=True`` when ``LOCAL_EMBED_MODEL`` resolves to one of
the vetted nomic-family ids (see :data:`AUTO_TRUST_REMOTE_CODE_MODELS`).
Any other model still requires the explicit ``LOCAL_TRUST_REMOTE_CODE``
opt-in.

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
   (4096 characters, conservative cap) before encode.  A minified /
   generated single-line file can be megabytes; without this cap a single
   ``encode()`` call can block for minutes on CPU, burning through the
   watchdog budget.

Dependencies
------------
Requires the ``sentence-transformers`` package (a default dependency).
If it is missing from the environment you'll see :class:`EmbedderError`
with the exact pip command at startup.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
from typing import Any, Callable

from .base import EMBEDDING_DIM, EmbedderBackend, EmbedderError

logger = logging.getLogger(__name__)

#: HuggingFace model id. Pinned so the local backend stays bit-compatible
#: with the prod path. Swapped from ``intfloat/e5-base-v2`` to
#: ``nomic-ai/nomic-embed-text-v1.5`` after the 768-dim POC — see module
#: docstring. The SageMaker / TEI paths still serve their own weights;
#: this default governs the in-process ``local`` backend only.
DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"

#: HF model ids that ship custom modeling code and require
#: ``trust_remote_code=True`` to load. The local backend auto-enables the
#: flag for these so the operator does not have to remember the
#: ``LOCAL_TRUST_REMOTE_CODE`` env knob. Membership is the only signal —
#: an unknown model still requires the explicit env opt-in.
AUTO_TRUST_REMOTE_CODE_MODELS: frozenset[str] = frozenset(
    {
        "nomic-ai/nomic-embed-text-v1.5",
        "nomic-ai/nomic-embed-text-v1",
        "nomic-ai/CodeRankEmbed",
        "jinaai/jina-embeddings-v2-base-code",
    }
)

#: Number of texts passed to one ``model.encode()`` call.  Matches the
#: embed driver's ``_BATCH`` so each outer flush triggers roughly one
#: ``encode()`` call → one progress tick → one heartbeat bump.  Override via
#: ``LOCAL_ENCODE_BATCH_SIZE``: long-context code models (CodeRankEmbed,
#: jina-v2-base-code — 8K ctx) can OOM a small GPU at 32; drop to 8.
from ._env_utils import env_int as _env_int

ENCODE_BATCH_SIZE = _env_int("LOCAL_ENCODE_BATCH_SIZE", 32)

#: Maximum character length for a single embed input text. nomic-v1.5
#: supports 8K tokens (Matryoshka) but the chunk strategies in
#: ``app/services/chunk_strategies.py`` never emit anything close to that;
#: 4096 chars (~1K BPE tokens) is a safety valve against pathological
#: minified / generated single-line files that would otherwise wedge a
#: single ``encode()`` call for tens of seconds on CPU and burn the phase
#: watchdog budget. Tuned conservatively rather than to the model ceiling.
EMBED_MAX_CHARS = 4096


class LocalEmbedder(EmbedderBackend):
    """Runs ``sentence-transformers`` in-process; 768-dim L2-normalised output.

    Construction is cheap — model loading is deferred until the first
    ``embed()`` call so ``get_embedder()`` can return immediately at
    startup. The model handle is then cached for the process lifetime.

    Thread-safety note
    ------------------
    ``embed_driver.py`` runs this backend from multiple ``asyncio.run()``
    calls in different ``ThreadPoolExecutor`` threads (one per concurrent
    batch).  Each ``asyncio.run()`` creates a private event loop, so an
    ``asyncio.Lock`` is NOT safe here: a waiter Future added to the lock
    on loop L1 can never be woken up by a ``release()`` that fires on loop
    L0, causing the second thread to hang indefinitely.

    The model-loading guard therefore uses ``threading.Lock`` — a plain OS
    primitive that is loop-agnostic and safe across concurrent
    ``asyncio.run()`` invocations.  The fast path (model already loaded)
    checks ``self._model is not None`` without acquiring the lock, which is
    safe because ``_model`` is only ever written once (from ``None`` to a
    loaded object) under the lock.
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
        # sentence-transformers model. Defaults to nomic-v1.5's 768
        # (Matryoshka native; truncation to 512/256 is opt-in via env).
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
        # Some code/nomic embedders ship custom modeling code that
        # sentence-transformers only loads with trust_remote_code=True
        # (which executes code from the model repo).  Auto-enabled for the
        # vetted set in ``AUTO_TRUST_REMOTE_CODE_MODELS`` (currently the
        # nomic family + CodeRankEmbed + jina-v2-base-code) so the default
        # ``nomic-ai/nomic-embed-text-v1.5`` works out of the box.  An
        # unknown ``LOCAL_EMBED_MODEL`` still requires the explicit
        # ``LOCAL_TRUST_REMOTE_CODE=1`` opt-in.
        env_trust = os.environ.get(
            "LOCAL_TRUST_REMOTE_CODE", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._trust_remote_code = (
            env_trust or self.model in AUTO_TRUST_REMOTE_CODE_MODELS
        )
        self._model: Any | None = None
        # threading.Lock — NOT asyncio.Lock.
        # This guard must work across concurrent asyncio.run() calls in
        # different ThreadPoolExecutor threads (the embed_driver.py use
        # case).  asyncio.Lock waiters are tied to the event loop they were
        # created on; a release() on L0 cannot wake a waiter registered on
        # L1, producing a permanent hang.  threading.Lock has no such
        # restriction and is the correct primitive for cross-thread,
        # cross-loop mutual exclusion.
        self._load_lock = threading.Lock()

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
                "package. Install it with: uv sync "
                "(or: uv pip install 'sentence-transformers>=3.2')"
            ) from exc

        try:
            return SentenceTransformer(
                self.model, trust_remote_code=self._trust_remote_code
            )
        except Exception as exc:  # noqa: BLE001 — surface the original cause
            raise EmbedderError(
                f"LocalEmbedder failed to load model {self.model!r}: {exc}"
            ) from exc

    def _truncate_texts(self, texts: list[str]) -> list[str]:
        """Cap each text at ``EMBED_MAX_CHARS`` and log a warning on truncation.

        The model itself has its own (much higher) token cap; this cap
        exists to keep a single pathological input — multi-megabyte
        minified file, generated source — from wedging a CPU encode for
        tens of seconds and burning through the phase watchdog budget.
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
                    "(watchdog cap; the model's own token limit is higher). "
                    "Truncation is logged but does not fail the embed.",
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
            One ``self.dim``-dim vector per input text, in input order.
            When ``LOCAL_EMBED_DIM`` selects a smaller-than-native dim
            (Matryoshka truncation, e.g. 512 or 256 on nomic-v1.5), the
            vector is truncated AFTER encoding and re-L2-normalised so
            cosine similarity stays comparable across embeddings.
        """
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
            for row in vectors:
                vec = [float(x) for x in row]
                # Matryoshka truncation: if the operator asked for a
                # smaller dim than the model's native output, slice and
                # re-normalise so the truncated vector stays unit-norm.
                if len(vec) > self.dim:
                    vec = vec[: self.dim]
                    norm = math.sqrt(sum(v * v for v in vec))
                    if norm > 0.0:
                        vec = [v / norm for v in vec]
                result.append(vec)
            if batch_callback is not None:
                try:
                    batch_callback(len(result))
                except Exception:  # noqa: BLE001 — callback failure must not kill the encode
                    pass

        return result

    def _ensure_model_loaded(self) -> None:
        """Load the sentence-transformers model exactly once, thread-safely.

        Uses ``threading.Lock`` (not ``asyncio.Lock``) so the guard works
        correctly across concurrent ``asyncio.run()`` calls in different
        threads — the ``embed_driver.py`` subprocess pattern.  The fast
        path (``self._model is not None``) is a plain attribute read that
        races safely because the attribute transitions from ``None`` to a
        non-``None`` object exactly once and is never set back to ``None``.

        Raises:
            EmbedderError: If ``sentence-transformers`` is not installed or
                the model fails to download / load.
        """
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is None:
                self._model = self._load_model()

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

        # Ensure the model is loaded.  _ensure_model_loaded uses a
        # threading.Lock internally so it is safe to call from concurrent
        # asyncio.run() contexts in different threads (the embed_driver.py
        # subprocess pattern).  Offload to a worker thread so the event
        # loop is not stalled during the 30-60 s first-load download.
        if self._model is None:
            await asyncio.to_thread(self._ensure_model_loaded)

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
