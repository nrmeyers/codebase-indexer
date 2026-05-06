"""SageMaker embedding adapter for the ``forge-e5-embed-v1`` endpoint.

Production primary embedding provider for the Code Indexer Service. Reads
``SAGEMAKER_EMBED_ENDPOINT`` and ``SAGEMAKER_EMBED_REGION`` from the
environment; returns ``None`` gracefully when unconfigured so callers fall
back to LM Studio or in-process torch.

Endpoint contract (forge-e5-embed-v1):
    Request:  POST InvokeEndpoint  body=``{"inputs": ["chunk1", ...]}}``
    Response: ``[[0.01, -0.98, ...], [...]]`` — one vector per input, same order.

Batch size guidance: 16–64 inputs per request. Default: 32.

This module has zero FastAPI / pydantic-settings imports so it can be
imported from background jobs and subprocesses without the full settings
stack.
"""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)

# Forge contract: 16–64 inputs per InvokeEndpoint call.
_DEFAULT_BATCH_SIZE = 32
_MAX_BATCH_SIZE = 64


class SageMakerEmbedder:
    """Thin boto3 client for the SageMaker embedding endpoint.

    Args:
        endpoint: SageMaker endpoint name (e.g. ``"forge-e5-embed-v1"``).
        region: AWS region (default ``"us-east-1"``).
        batch_size: Max inputs per InvokeEndpoint call (1–64, default 32).
    """

    __slots__ = ("_endpoint", "_region", "_batch_size")

    def __init__(
        self,
        endpoint: str,
        region: str = "us-east-1",
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        self._endpoint = endpoint
        self._region = region
        self._batch_size = min(max(1, batch_size), _MAX_BATCH_SIZE)

    @classmethod
    def from_env(cls) -> "SageMakerEmbedder | None":
        """Construct from env vars.  Returns ``None`` when endpoint is unset."""
        endpoint = (os.environ.get("SAGEMAKER_EMBED_ENDPOINT") or "").strip()
        if not endpoint:
            return None
        region = (os.environ.get("SAGEMAKER_EMBED_REGION") or "us-east-1").strip()
        try:
            batch_size = int(os.environ.get("SAGEMAKER_EMBED_BATCH_SIZE") or _DEFAULT_BATCH_SIZE)
        except (TypeError, ValueError):
            batch_size = _DEFAULT_BATCH_SIZE
        return cls(endpoint=endpoint, region=region, batch_size=batch_size)

    def embed(self, texts: str | list[str]) -> list[list[float]] | None:
        """Embed one or more text strings.

        Args:
            texts: A single string or a list of strings to embed.

        Returns:
            A list of float vectors in the same order as ``texts``, or
            ``None`` on any error so the caller can fall back to LM Studio /
            in-process torch.
        """
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return []

        try:
            import boto3  # type: ignore[import-untyped]

            runtime = boto3.client("sagemaker-runtime", region_name=self._region)
            results: list[list[float] | None] = [None] * len(texts)

            for start in range(0, len(texts), self._batch_size):
                chunk = texts[start : start + self._batch_size]
                body = json.dumps({"inputs": chunk}).encode("utf-8")
                response = runtime.invoke_endpoint(
                    EndpointName=self._endpoint,
                    ContentType="application/json",
                    Body=body,
                )
                raw: list[list[float]] = json.loads(response["Body"].read())

                if len(raw) != len(chunk):
                    raise RuntimeError(
                        f"SageMaker returned {len(raw)} vectors for {len(chunk)} inputs"
                    )
                for offset, vec in enumerate(raw):
                    results[start + offset] = [float(v) for v in vec]

        except Exception as exc:
            logger.warning(
                "SageMakerEmbedder.embed failed (endpoint=%s, error=%s) — caller should fall back",
                self._endpoint,
                exc,
            )
            return None

        # Verify all slots populated.
        out: list[list[float]] = []
        for i, v in enumerate(results):
            if v is None:
                logger.warning("SageMakerEmbedder: missing slot %d — caller should fall back", i)
                return None
            out.append(v)
        return out

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def region(self) -> str:
        return self._region


@lru_cache(maxsize=1)
def get_sagemaker_embedder() -> SageMakerEmbedder | None:
    """Module-level singleton.  Returns ``None`` when ``SAGEMAKER_EMBED_ENDPOINT`` is unset.

    Call ``get_sagemaker_embedder.cache_clear()`` in tests to force re-init.
    """
    embedder = SageMakerEmbedder.from_env()
    if embedder is not None:
        logger.info(
            "SageMakerEmbedder initialised (endpoint=%s, region=%s, batch_size=%d)",
            embedder.endpoint,
            embedder.region,
            embedder._batch_size,
        )
    return embedder
