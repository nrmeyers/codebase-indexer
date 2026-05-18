"""AWS SageMaker Serverless Inference backend (BUC-1605).

Calls Navistone's ``forge-e5-embed-v2`` endpoint in ``us-east-1`` via
``boto3.client('sagemaker-runtime')``. This is the default for the
Navistone production deploy; standalone installs without AWS creds
should leave this alone and use the ``local`` backend instead.

Endpoint contract
-----------------
::

    POST /endpoints/forge-e5-embed-v2/invocations
    Content-Type: application/json
    Body:        {"inputs": ["chunk1", "chunk2", ...]}
    Response:    [[0.01, -0.98, ...], [...]]
                 — one 768-float L2-normalised vector per input
                   (mean-pooled server-side by the custom inference handler).

Configuration
-------------
::

    SAGEMAKER_ENDPOINT_NAME    Endpoint name, e.g. "forge-e5-embed-v2".
                               Preferred over the legacy URL form.
    SAGEMAKER_EMBED_ENDPOINT   Legacy alias (still read for backwards compat).
    SAGEMAKER_EMBED_URL        Full invocation URL; endpoint name extracted.
    SAGEMAKER_EMBED_REGION     AWS region (default us-east-1).
    SAGEMAKER_EMBED_BATCH_SIZE Inputs per request (1-64, default 16).

Timeouts (boto3 ``Config``) are tuned for batch=32 on ml.m5.large:
    connect_timeout=10s, read_timeout=90s, retries=3 (standard mode).

Truncation
----------
e5-base-v2 has a hard 512-token position-embedding limit; SageMaker's HF
inference toolkit does NOT truncate automatically. We hard-cap each text
at 1000 chars (~300 tokens for Python code) on the client before sending,
matching the pre-existing ``codebase_rag.embedder`` behaviour.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

from .base import EMBEDDING_DIM, EmbedderBackend, EmbedderError

logger = logging.getLogger(__name__)

#: Per-text character cap before sending to the endpoint. Above ~1200 chars
#: e5-base-v2 hits its 512-token position-embedding limit and SageMaker
#: returns a 400. Binary search settled on 1000 as the safe ceiling
#: (~3.3 chars/token for Python).
_MAX_CHARS = 1000
_DEFAULT_BATCH_SIZE = 16
_CONNECT_TIMEOUT = 10
_READ_TIMEOUT = 90


class SageMakerEmbedder(EmbedderBackend):
    """Boto3-backed client for Navistone's SageMaker e5-base-v2 endpoint."""

    name = "sagemaker"
    model = "e5-base-v2"
    dim = EMBEDDING_DIM

    def __init__(
        self,
        endpoint_name: str,
        region: str = "us-east-1",
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        if not endpoint_name:
            raise EmbedderError("SageMakerEmbedder: endpoint_name must be non-empty")
        self.endpoint_name = endpoint_name
        self.region = region
        # Clamp to the SageMaker contract (1-64). Larger batches trip the
        # 60s serverless timeout; smaller ones throttle ingest throughput.
        self.batch_size = min(max(1, batch_size), 64)
        self._client: Any | None = None

    @classmethod
    def from_env(cls) -> "SageMakerEmbedder":
        """Construct from env vars; raise if no endpoint is configured.

        Resolution priority (highest first):
            1. ``SAGEMAKER_ENDPOINT_NAME`` — preferred BUC-1605 name.
            2. ``SAGEMAKER_EMBED_ENDPOINT`` — legacy alias kept for
               backwards-compat with existing Navistone .env files.
            3. ``SAGEMAKER_EMBED_URL`` — full URL, endpoint name extracted.
        """
        url = (os.environ.get("SAGEMAKER_EMBED_URL") or "").strip()
        endpoint = (
            os.environ.get("SAGEMAKER_ENDPOINT_NAME")
            or os.environ.get("SAGEMAKER_EMBED_ENDPOINT")
            or ""
        ).strip()
        if not endpoint and url:
            endpoint = cls._extract_endpoint_name(url)
        if not endpoint:
            raise EmbedderError(
                "EMBEDDER_BACKEND=sagemaker but no endpoint is configured. "
                "Set SAGEMAKER_ENDPOINT_NAME (preferred) or SAGEMAKER_EMBED_URL."
            )

        region = (os.environ.get("SAGEMAKER_EMBED_REGION") or "us-east-1").strip()
        try:
            batch_size = int(
                os.environ.get("SAGEMAKER_EMBED_BATCH_SIZE") or _DEFAULT_BATCH_SIZE
            )
        except (TypeError, ValueError):
            batch_size = _DEFAULT_BATCH_SIZE

        return cls(endpoint_name=endpoint, region=region, batch_size=batch_size)

    @staticmethod
    def _extract_endpoint_name(url_or_name: str) -> str:
        """Pull endpoint name out of a SageMaker invocation URL, or pass through."""
        match = re.search(r"/endpoints/([^/]+)/invocations", url_or_name)
        return match.group(1) if match else url_or_name

    def _get_client(self) -> Any:
        """Lazy-init the boto3 sagemaker-runtime client with proper timeouts.

        Deferred so importing the module doesn't pay the boto3 service-model
        load cost (~200ms) when the sagemaker backend is not selected.
        """
        if self._client is not None:
            return self._client
        try:
            import boto3  # type: ignore[import-untyped]
            from botocore.config import Config  # type: ignore[import-untyped]
        except ImportError as exc:
            raise EmbedderError(
                "EMBEDDER_BACKEND=sagemaker requires 'boto3'. "
                "It is a core dep of code-indexer-service — reinstall to fix."
            ) from exc

        self._client = boto3.client(
            "sagemaker-runtime",
            region_name=self.region,
            config=Config(
                connect_timeout=_CONNECT_TIMEOUT,
                read_timeout=_READ_TIMEOUT,
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )
        return self._client

    def _invoke_sync(self, body: bytes) -> bytes:
        """One blocking ``invoke_endpoint`` call. Wrapped in to_thread."""
        client = self._get_client()
        resp = client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=body,
        )
        return resp["Body"].read()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            safe_chunk = [
                t[:_MAX_CHARS] if len(t) > _MAX_CHARS else t for t in chunk
            ]
            body = json.dumps({"inputs": safe_chunk}).encode("utf-8")
            try:
                raw_bytes = await asyncio.to_thread(self._invoke_sync, body)
                raw = json.loads(raw_bytes)
            except Exception as exc:  # noqa: BLE001
                raise EmbedderError(
                    f"SageMakerEmbedder.invoke_endpoint failed "
                    f"({type(exc).__name__}: {exc})"
                ) from exc

            if not isinstance(raw, list) or len(raw) != len(chunk):
                raise EmbedderError(
                    f"SageMaker returned {len(raw) if isinstance(raw, list) else type(raw).__name__} "
                    f"embeddings for {len(chunk)} inputs"
                )

            for i, vec in enumerate(raw):
                if not isinstance(vec, list) or len(vec) != EMBEDDING_DIM:
                    raise EmbedderError(
                        f"SageMaker returned {len(vec) if isinstance(vec, list) else type(vec).__name__}-"
                        f"dim vector for input {start + i}; expected {EMBEDDING_DIM}"
                    )
                results.append([float(v) for v in vec])

        return results
