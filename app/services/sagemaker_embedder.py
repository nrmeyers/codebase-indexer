"""SageMaker embedding adapter for the ``forge-e5-embed-v2`` endpoint.

Production primary embedding provider for the Code Indexer Service.

Uses boto3's ``sagemaker-runtime`` client for invocation.  Boto3's
``Config(read_timeout=...)`` is a true total-time-without-bytes timeout
enforced by botocore — unlike ``urllib.urlopen(timeout=...)``, which is a
per-recv timeout that does not fire when MMS keepalive packets keep the
TCP socket alive between data bytes.  The urllib variant caused indefinite
hangs on the first batch in 2026-05-06's re-index attempts; boto3's
read_timeout reliably fires after the configured ceiling.

Endpoint contract (after BUC-1509 inference-handler fix):
    POST  /endpoints/forge-e5-embed-v2/invocations
    Body:        {"inputs": ["snippet1", "snippet2", ...]}
    Response:    [[f0, f1, ..., f767], [...]]   one 768-float L2-normalized
                 vector per input (mean-pooled server-side).

Provider priority (in the indexer service):
    SageMaker (this) → LM Studio → in-process torch

Env vars:
    SAGEMAKER_EMBED_URL        Full invocation URL (preferred when present;
                               endpoint name is extracted automatically).
    SAGEMAKER_EMBED_ENDPOINT   Endpoint name; fallback when URL is unset.
    SAGEMAKER_EMBED_REGION     AWS region (default: us-east-1).
    SAGEMAKER_EMBED_BATCH_SIZE Inputs per request (1–64, default 16).

This module has zero FastAPI / pydantic-settings imports so it can be
imported from background jobs and subprocesses without the full settings stack.
"""
from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache

logger = logging.getLogger(__name__)

# Batch size: 16 is a safe production default for ml.m5.large with realistic
# 1000-char Python/TypeScript inputs (~25-30s per call).  ml.c6i.2xlarge or
# Serverless Inference can comfortably handle 32-64.
_DEFAULT_BATCH_SIZE = 16
_MAX_BATCH_SIZE = 64
_CONTENT_TYPE = "application/json"

# Connect timeout: very short — TCP handshake should be sub-second on AWS LAN.
# Read timeout: 90s is enough for batch=32 (~30-50s) plus some cold-start margin
# but short enough to recover from a stuck worker before users notice.  Boto3's
# read_timeout fires when no bytes arrive for this many seconds; unlike urllib
# this is wall-clock, not per-recv, so MMS keepalive packets cannot defeat it.
_CONNECT_TIMEOUT = 10
_READ_TIMEOUT = 90

# e5-base-v2 (BERT) has a 512-token position-embedding ceiling.  The SageMaker
# HF inference toolkit does NOT truncate automatically and returns HTTP 400 on
# overflow.  Binary search confirmed: 1000 chars → OK, 1200 chars → FAIL for
# Python source.  1000 is the safe client-side cap.
_MAX_INPUT_CHARS = 1000


def _extract_endpoint_name(url_or_name: str) -> str:
    """Pull the endpoint name out of a full invocation URL, or return as-is.

    Examples:
        >>> _extract_endpoint_name("forge-e5-embed-v2")
        'forge-e5-embed-v2'
        >>> _extract_endpoint_name("https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/forge-e5-embed-v2/invocations")
        'forge-e5-embed-v2'
    """
    m = re.search(r"/endpoints/([^/]+)/invocations", url_or_name)
    return m.group(1) if m else url_or_name


class SageMakerEmbedder:
    """Boto3-backed client for the SageMaker embedding endpoint.

    Falls back gracefully — ``embed()`` returns ``None`` on any error so the
    caller can fall through to the next provider (LM Studio or in-process torch).
    """

    __slots__ = ("_endpoint_name", "_region", "_batch_size", "_client")

    def __init__(
        self,
        endpoint_name: str,
        region: str = "us-east-1",
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        self._endpoint_name = endpoint_name
        self._region = region
        self._batch_size = min(max(1, batch_size), _MAX_BATCH_SIZE)
        # Lazy-init the boto3 client on first call so importing this module
        # doesn't pay the boto3 service-model load cost up front.
        self._client = None  # type: ignore[assignment]

    @classmethod
    def from_env(cls) -> "SageMakerEmbedder | None":
        """Construct from env vars.  Returns ``None`` when no endpoint is set."""
        region = (os.environ.get("SAGEMAKER_EMBED_REGION") or "us-east-1").strip()

        # Prefer explicit URL (extract name); fall back to bare endpoint name.
        url = (os.environ.get("SAGEMAKER_EMBED_URL") or "").strip()
        endpoint = (os.environ.get("SAGEMAKER_EMBED_ENDPOINT") or "").strip()
        endpoint_name = _extract_endpoint_name(url) if url else endpoint
        if not endpoint_name:
            return None

        try:
            batch_size = int(
                os.environ.get("SAGEMAKER_EMBED_BATCH_SIZE") or _DEFAULT_BATCH_SIZE
            )
        except (TypeError, ValueError):
            batch_size = _DEFAULT_BATCH_SIZE

        return cls(endpoint_name=endpoint_name, region=region, batch_size=batch_size)

    def _get_client(self):
        """Lazy-init the boto3 sagemaker-runtime client with proper timeouts."""
        if self._client is not None:
            return self._client

        import boto3  # type: ignore[import-untyped]
        from botocore.config import Config  # type: ignore[import-untyped]

        self._client = boto3.client(
            "sagemaker-runtime",
            region_name=self._region,
            config=Config(
                connect_timeout=_CONNECT_TIMEOUT,
                read_timeout=_READ_TIMEOUT,
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )
        return self._client

    def _invoke(self, body: bytes) -> bytes:
        """POST ``body`` to the endpoint via boto3.  Returns raw response bytes.

        Raises ``botocore.exceptions.ReadTimeoutError`` if no response bytes
        arrive within ``_READ_TIMEOUT``.  This is the boto3 contract — unlike
        urllib, it IS a true wall-clock timeout that fires regardless of TCP
        keepalive activity.
        """
        client = self._get_client()
        resp = client.invoke_endpoint(
            EndpointName=self._endpoint_name,
            ContentType=_CONTENT_TYPE,
            Accept=_CONTENT_TYPE,
            Body=body,
        )
        return resp["Body"].read()

    def embed(self, texts: str | list[str]) -> list[list[float]] | None:
        """Embed one or more text strings.

        Args:
            texts: Single string or list of strings to embed.

        Returns:
            List of float vectors (same order as input), or ``None`` on error.
        """
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return []

        try:
            results: list[list[float] | None] = [None] * len(texts)
            for start in range(0, len(texts), self._batch_size):
                chunk = texts[start : start + self._batch_size]
                safe_chunk = [
                    t[:_MAX_INPUT_CHARS] if len(t) > _MAX_INPUT_CHARS else t
                    for t in chunk
                ]
                body = json.dumps({"inputs": safe_chunk}).encode("utf-8")
                raw: list[list[float]] = json.loads(self._invoke(body))
                if len(raw) != len(chunk):
                    raise RuntimeError(
                        f"SageMaker returned {len(raw)} vectors for {len(chunk)} inputs"
                    )
                for offset, vec in enumerate(raw):
                    # forge-e5-embed-v2 returns flat [batch, 768] floats —
                    # the custom inference handler does mean-pool + L2-normalize
                    # server-side, so no client-side unwrapping is needed.
                    results[start + offset] = [float(v) for v in vec]

        except Exception as exc:
            # f-string so the actual exception type+message surfaces; the previous
            # %s-style was being swallowed by loguru elsewhere in the codebase.
            logger.warning(
                f"SageMakerEmbedder.embed failed (endpoint={self._endpoint_name}, "
                f"err={type(exc).__name__}: {exc}) — caller should fall back"
            )
            return None

        out: list[list[float]] = []
        for i, v in enumerate(results):
            if v is None:
                logger.warning(
                    f"SageMakerEmbedder: missing slot {i} — caller should fall back"
                )
                return None
            out.append(v)
        return out

    @property
    def endpoint_name(self) -> str:
        return self._endpoint_name

    @property
    def region(self) -> str:
        return self._region

    # Keep ``url`` for backwards compat with any caller that printed it
    # for diagnostics.  Synthesises the standard invocation URL.
    @property
    def url(self) -> str:
        return (
            f"https://runtime.sagemaker.{self._region}.amazonaws.com"
            f"/endpoints/{self._endpoint_name}/invocations"
        )


@lru_cache(maxsize=1)
def get_sagemaker_embedder() -> SageMakerEmbedder | None:
    """Module-level singleton.  ``None`` when no URL/endpoint env var is set.

    Call ``get_sagemaker_embedder.cache_clear()`` in tests to force re-init.
    """
    embedder = SageMakerEmbedder.from_env()
    if embedder is not None:
        logger.info(
            "SageMakerEmbedder active (endpoint=%s, region=%s, batch_size=%d)",
            embedder.endpoint_name,
            embedder.region,
            embedder._batch_size,
        )
    return embedder
