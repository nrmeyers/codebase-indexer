"""llama-server embedder backend (768 POC — nomic-embed-text-v1.5 GGUF).

Talks to a ``llama.cpp`` ``llama-server`` process over its OpenAI-compatible
``POST /v1/embeddings`` endpoint. Lets us evaluate GGUF/quantised models
(specifically ``nomic-embed-text-v1.5.Q8_0.gguf``) without pulling them
in-process — the in-process ``sentence-transformers`` path was both slow and
OOM-prone on the long-context code corpus.

Serve recipe (768-dim full-quality Matryoshka, mean pooled)::

    /mnt/ai-data/llama/llama-server-cuda.sh \\
        -m /mnt/ai-data/llama/models/nomic-embed-text-v1.5.Q8_0.gguf \\
        --embeddings --pooling mean --ctx-size 2048 --ubatch-size 2048 \\
        -ngl 99 --host 127.0.0.1 --port 8090

Gotchas baked into this client:

* ``--ubatch-size 2048`` on the server is mandatory; the default 512 makes
  any input chunk >512 tokens return HTTP 500 ``"input too large to
  process"``. The driver enforces the flag; this client adds belt-and-
  braces *client-side* truncation to ``LLAMA_SERVER_MAX_TOKENS`` (default
  2048) using the model's HuggingFace tokenizer, applied AFTER the
  query/document prefix.
* Prefixes (``search_query: `` / ``search_document: ``) are applied
  upstream by :mod:`app.embedders.prefixes`. This backend never injects
  prefixes itself.
* nomic-embed-text-v1.5 is Matryoshka — the server returns the *full*
  768-dim vector; we do not slice it. Any other dim is treated as a
  protocol error.

Configuration
-------------
::

    LLAMA_SERVER_URL          Base URL (no trailing slash). Default
                              http://127.0.0.1:8090.
    LLAMA_SERVER_MODEL        Model name used in the /v1/embeddings ``model``
                              field; surfaced on the backend as ``.model``.
                              Default ``nomic-embed-text-v1.5.Q8_0.gguf``.
    LLAMA_SERVER_TOKENIZER    HF id used for client-side truncation. Default
                              ``nomic-ai/nomic-embed-text-v1.5``.
    LLAMA_SERVER_MAX_TOKENS   Hard cap per input AFTER prefix. Default 2048
                              (matches ``--ctx-size``).
    LLAMA_SERVER_TIMEOUT_MS   Per-request timeout. Default 60000 (60s — the
                              first request after server start can be slow
                              while CUDA kernels warm).
    LLAMA_SERVER_BATCH_SIZE   Inputs per HTTP request. Default 16. The
                              server flattens internally; this is just to
                              keep request bodies and tail latencies sane.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

import httpx

from .base import EMBEDDING_DIM, EmbedderBackend, EmbedderError

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://127.0.0.1:8090"
DEFAULT_MODEL = "nomic-embed-text-v1.5.Q8_0.gguf"
DEFAULT_TOKENIZER = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TIMEOUT_MS = 60_000
DEFAULT_BATCH_SIZE = 16


class LlamaServerEmbedder(EmbedderBackend):
    """HTTP client for a ``llama.cpp`` ``llama-server`` embedding endpoint."""

    name = "llama_server"
    dim = EMBEDDING_DIM

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        model: str = DEFAULT_MODEL,
        tokenizer_id: str = DEFAULT_TOKENIZER,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        # HF id consulted by :mod:`app.embedders.prefixes` so the GGUF
        # filename in ``model`` still resolves to the correct prefix entry.
        self.prefix_model = tokenizer_id
        self._tokenizer_id = tokenizer_id
        self.max_tokens = max(1, max_tokens)
        self.timeout_s = max(0.5, timeout_ms / 1000.0)
        self.batch_size = min(max(1, batch_size), 128)
        # NB: we intentionally DO NOT cache an ``httpx.AsyncClient`` across
        # ``embed()`` invocations. The embed driver and sync_bridge call us
        # via ``asyncio.run()`` per batch, which creates and tears down a
        # fresh event loop each time; a cached client is bound to the FIRST
        # loop and every subsequent call raises ``RuntimeError: Event loop
        # is closed``. Instead each ``embed()`` builds its own client inside
        # ``async with`` so the loop binding lines up. Keepalive within a
        # single ``embed()`` call (multiple batches) still applies.
        # The HF tokenizer is loaded lazily on first embed() — keeps
        # construction cheap and avoids paying the import cost when the
        # backend is just being probed at startup.
        self._tokenizer: Any = None
        self._tok_lock = threading.Lock()
        self._truncated_warned = False

    @classmethod
    def from_env(cls) -> "LlamaServerEmbedder":
        base_url = (os.environ.get("LLAMA_SERVER_URL") or DEFAULT_URL).strip()
        model = (os.environ.get("LLAMA_SERVER_MODEL") or DEFAULT_MODEL).strip()
        tok = (os.environ.get("LLAMA_SERVER_TOKENIZER") or DEFAULT_TOKENIZER).strip()
        try:
            max_tokens = int(os.environ.get("LLAMA_SERVER_MAX_TOKENS") or DEFAULT_MAX_TOKENS)
        except (TypeError, ValueError):
            max_tokens = DEFAULT_MAX_TOKENS
        try:
            timeout_ms = int(os.environ.get("LLAMA_SERVER_TIMEOUT_MS") or DEFAULT_TIMEOUT_MS)
        except (TypeError, ValueError):
            timeout_ms = DEFAULT_TIMEOUT_MS
        try:
            batch_size = int(os.environ.get("LLAMA_SERVER_BATCH_SIZE") or DEFAULT_BATCH_SIZE)
        except (TypeError, ValueError):
            batch_size = DEFAULT_BATCH_SIZE
        return cls(
            base_url=base_url,
            model=model,
            tokenizer_id=tok,
            max_tokens=max_tokens,
            timeout_ms=timeout_ms,
            batch_size=batch_size,
        )

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        with self._tok_lock:
            if self._tokenizer is not None:
                return self._tokenizer
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise EmbedderError(
                    "EMBEDDER_BACKEND=llama_server requires 'transformers' "
                    "(installed as a sentence-transformers transitive dep)."
                ) from exc
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._tokenizer_id, use_fast=True
                )
            except Exception as exc:  # noqa: BLE001
                raise EmbedderError(
                    f"LlamaServerEmbedder could not load tokenizer "
                    f"{self._tokenizer_id!r}: {type(exc).__name__}: {exc}"
                ) from exc
        return self._tokenizer

    def _truncate(self, texts: list[str]) -> list[str]:
        """Hard-cap each text at ``max_tokens`` using the model's tokenizer.

        The cap is applied AFTER the upstream prefix (callers pass already-
        prefixed strings), which is exactly what the server tokenises.
        """
        tok = self._get_tokenizer()
        out: list[str] = []
        for t in texts:
            ids = tok.encode(t, add_special_tokens=False)
            if len(ids) > self.max_tokens:
                if not self._truncated_warned:
                    logger.warning(
                        "LlamaServerEmbedder: truncating input from %d to %d tokens "
                        "(future truncations logged at DEBUG only)",
                        len(ids),
                        self.max_tokens,
                    )
                    self._truncated_warned = True
                else:
                    logger.debug(
                        "LlamaServerEmbedder: truncated %d -> %d tokens",
                        len(ids),
                        self.max_tokens,
                    )
                ids = ids[: self.max_tokens]
                # Decode back to text; ``clean_up_tokenization_spaces`` keeps
                # the surface form close to the original.
                t = tok.decode(ids, skip_special_tokens=True)
            out.append(t)
        return out

    async def aclose(self) -> None:
        # No-op: clients are per-call now (see __init__ comment).
        return None

    async def _embed_one(self, client: httpx.AsyncClient, text: str) -> list[float]:
        """Per-string fallback when a batch request returns 5xx."""
        payload: dict[str, Any] = {"model": self.model, "input": text}
        try:
            resp = await client.post("/v1/embeddings", json=payload)
        except httpx.HTTPError as exc:
            raise EmbedderError(
                f"LlamaServerEmbedder POST /v1/embeddings failed (single): "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if resp.status_code != 200:
            raise EmbedderError(
                f"LlamaServerEmbedder HTTP {resp.status_code} (single): "
                f"{resp.text[:400]}"
            )
        try:
            body = resp.json()
        except ValueError as exc:
            raise EmbedderError(
                f"LlamaServerEmbedder: malformed JSON (single): {exc}"
            ) from exc
        return self._extract_vectors(body, expected=1)[0]

    @staticmethod
    def _extract_vectors(body: Any, *, expected: int) -> list[list[float]]:
        """Pull the ``data[*].embedding`` arrays out of an OpenAI-shape body."""
        if not isinstance(body, dict):
            raise EmbedderError(
                f"LlamaServerEmbedder: expected JSON object, got {type(body).__name__}"
            )
        data = body.get("data")
        if not isinstance(data, list) or len(data) != expected:
            raise EmbedderError(
                f"LlamaServerEmbedder returned "
                f"{len(data) if isinstance(data, list) else type(data).__name__} "
                f"entries for {expected} inputs"
            )
        out: list[list[float]] = []
        for i, entry in enumerate(data):
            if not isinstance(entry, dict):
                raise EmbedderError(
                    f"LlamaServerEmbedder: data[{i}] is "
                    f"{type(entry).__name__}, expected dict"
                )
            vec = entry.get("embedding")
            if not isinstance(vec, list) or len(vec) != EMBEDDING_DIM:
                raise EmbedderError(
                    f"LlamaServerEmbedder returned "
                    f"{len(vec) if isinstance(vec, list) else type(vec).__name__}-"
                    f"dim vector at index {i}; expected {EMBEDDING_DIM}"
                )
            out.append([float(v) for v in vec])
        return out

    async def _embed_batch(self, client: httpx.AsyncClient, chunk: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {"model": self.model, "input": chunk}
        try:
            resp = await client.post("/v1/embeddings", json=payload)
        except httpx.HTTPError as exc:
            raise EmbedderError(
                f"LlamaServerEmbedder POST /v1/embeddings failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if 500 <= resp.status_code < 600:
            # Batch failure → fall back to per-string. Common cause is a
            # single oversize input we couldn't tokenise (no HF tokenizer,
            # tokenizer mismatch). One bad input shouldn't sink the run.
            logger.warning(
                "LlamaServerEmbedder: HTTP %d on batch of %d; falling back to per-string",
                resp.status_code,
                len(chunk),
            )
            out: list[list[float]] = []
            for t in chunk:
                out.append(await self._embed_one(client, t))
            return out

        if resp.status_code != 200:
            raise EmbedderError(
                f"LlamaServerEmbedder HTTP {resp.status_code}: {resp.text[:400]}"
            )

        try:
            body = resp.json()
        except ValueError as exc:
            raise EmbedderError(
                f"LlamaServerEmbedder: malformed JSON response: {exc}"
            ) from exc

        return self._extract_vectors(body, expected=len(chunk))

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        truncated = self._truncate(texts)
        results: list[list[float]] = []
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_s,
            headers={"Content-Type": "application/json"},
        ) as client:
            for start in range(0, len(truncated), self.batch_size):
                chunk = truncated[start : start + self.batch_size]
                results.extend(await self._embed_batch(client, chunk))
        return results
