"""Hugging Face Text-Embeddings-Inference (TEI) HTTP backend (BUC-1605).

TEI is a Rust-based HTTP server that runs embedding models with GPU
batching and an OpenAI-compatible-ish API. The Code Indexer hits its
``POST /embed`` endpoint. The intended deployment is a Docker sidecar:

::

    docker run -d --name tei \\
        -p 8080:80 \\
        --gpus all \\
        ghcr.io/huggingface/text-embeddings-inference:1.5 \\
        --model-id intfloat/e5-base-v2

Endpoint contract
-----------------
::

    POST {TEI_URL}/embed
    Content-Type: application/json
    Body:        {"inputs": ["text1", "text2", ...], "normalize": true}
    Response:    [[0.01, -0.98, ...], [...]]
                 — one vector per input, in input order.

Configuration
-------------
::

    TEI_URL          Base URL (no trailing slash). Default http://localhost:8080.
    TEI_TIMEOUT_MS   Hard timeout per HTTP request (default 30000 = 30s).
    TEI_BATCH_SIZE   Inputs per request (default 32). TEI batches further
                     server-side; this caps the request body size.

Failure modes are surfaced as :class:`EmbedderError` so the operator sees
exactly which sidecar is down — unlike SageMaker we don't auto-retry; TEI
is expected to be a hot local sidecar.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from .base import EMBEDDING_DIM, EmbedderBackend, EmbedderError

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://localhost:8080"
DEFAULT_TIMEOUT_MS = 30_000
DEFAULT_BATCH_SIZE = 32


class TEIEmbedder(EmbedderBackend):
    """HTTP client for a Hugging Face Text-Embeddings-Inference sidecar."""

    name = "tei"
    model = "e5-base-v2"

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = max(0.1, timeout_ms / 1000.0)
        # Clamp to a sane window — TEI happily accepts 1, but huge batches
        # blow up request body size and HTTP keepalive timing.
        self.batch_size = min(max(1, batch_size), 256)
        self._client: httpx.AsyncClient | None = None

    @classmethod
    def from_env(cls) -> "TEIEmbedder":
        base_url = (os.environ.get("TEI_URL") or DEFAULT_URL).strip()
        try:
            timeout_ms = int(
                (os.environ.get("TEI_TIMEOUT_MS") or DEFAULT_TIMEOUT_MS)
            )
        except (TypeError, ValueError):
            timeout_ms = DEFAULT_TIMEOUT_MS
        try:
            batch_size = int(
                (os.environ.get("TEI_BATCH_SIZE") or DEFAULT_BATCH_SIZE)
            )
        except (TypeError, ValueError):
            batch_size = DEFAULT_BATCH_SIZE
        return cls(
            base_url=base_url,
            timeout_ms=timeout_ms,
            batch_size=batch_size,
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-init the httpx.AsyncClient; reused across calls for keepalive."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout_s,
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def aclose(self) -> None:
        """Close the underlying httpx client. Idempotent."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def _embed_batch(self, chunk: list[str]) -> list[list[float]]:
        client = await self._get_client()
        payload: dict[str, Any] = {"inputs": chunk, "normalize": True}
        try:
            resp = await client.post("/embed", json=payload)
        except httpx.HTTPError as exc:
            raise EmbedderError(
                f"TEIEmbedder POST {self.base_url}/embed failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if resp.status_code != 200:
            raise EmbedderError(
                f"TEIEmbedder HTTP {resp.status_code} from {self.base_url}/embed: "
                f"{resp.text[:400]}"
            )

        try:
            raw = resp.json()
        except ValueError as exc:
            raise EmbedderError(
                f"TEIEmbedder: malformed JSON response: {exc}"
            ) from exc

        if not isinstance(raw, list) or len(raw) != len(chunk):
            raise EmbedderError(
                f"TEIEmbedder returned "
                f"{len(raw) if isinstance(raw, list) else type(raw).__name__} "
                f"embeddings for {len(chunk)} inputs"
            )

        result: list[list[float]] = []
        for i, vec in enumerate(raw):
            if not isinstance(vec, list) or len(vec) != EMBEDDING_DIM:
                raise EmbedderError(
                    f"TEIEmbedder returned "
                    f"{len(vec) if isinstance(vec, list) else type(vec).__name__}-"
                    f"dim vector for input {i}; expected {EMBEDDING_DIM}"
                )
            result.append([float(v) for v in vec])
        return result

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            results.extend(await self._embed_batch(chunk))
        return results
