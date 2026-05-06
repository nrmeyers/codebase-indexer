"""SageMaker embedding adapter for the ``forge-e5-embed-v1`` endpoint.

Production primary embedding provider for the Code Indexer Service.

Calls the invocation URL directly via ``urllib`` + botocore SigV4 signing
rather than going through the full boto3 ``sagemaker-runtime`` client.
This keeps the import footprint minimal while still using the standard
AWS credential chain.

Endpoint:
    POST https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/forge-e5-embed-v1/invocations
    Request body:  {"inputs": ["chunk1", "chunk2", ...]}
    Response body: [[0.01, -0.98, ...], [...]]  — one vector per input.

Provider priority (in the indexer service):
    SageMaker (this) → LM Studio → in-process torch

Env vars:
    SAGEMAKER_EMBED_URL        Full invocation URL (preferred).
    SAGEMAKER_EMBED_ENDPOINT   Endpoint name; URL derived automatically when URL unset.
    SAGEMAKER_EMBED_REGION     AWS region (default: us-east-1).
    SAGEMAKER_EMBED_BATCH_SIZE Inputs per request (1–64, default 32).

This module has zero FastAPI / pydantic-settings imports so it can be
imported from background jobs and subprocesses without the full settings stack.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from functools import lru_cache

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 8
_MAX_BATCH_SIZE = 64
_CONTENT_TYPE = "application/json"
_INVOKE_TIMEOUT = 30  # seconds


class SageMakerEmbedder:
    """Direct-HTTPS client for the SageMaker embedding endpoint.

    Prefer setting ``SAGEMAKER_EMBED_URL`` to the full invocation URL:
        https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/forge-e5-embed-v1/invocations

    Falls back gracefully — ``embed()`` returns ``None`` on any error.
    """

    __slots__ = ("_url", "_region", "_batch_size")

    def __init__(self, url: str, region: str = "us-east-1", batch_size: int = _DEFAULT_BATCH_SIZE) -> None:
        self._url = url
        self._region = region
        self._batch_size = min(max(1, batch_size), _MAX_BATCH_SIZE)

    @classmethod
    def from_env(cls) -> "SageMakerEmbedder | None":
        """Construct from env vars.  Returns ``None`` when neither URL nor endpoint is set."""
        region = (os.environ.get("SAGEMAKER_EMBED_REGION") or "us-east-1").strip()

        url = (os.environ.get("SAGEMAKER_EMBED_URL") or "").strip()
        if not url:
            endpoint = (os.environ.get("SAGEMAKER_EMBED_ENDPOINT") or "").strip()
            if not endpoint:
                return None
            url = f"https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{endpoint}/invocations"

        try:
            batch_size = int(os.environ.get("SAGEMAKER_EMBED_BATCH_SIZE") or _DEFAULT_BATCH_SIZE)
        except (TypeError, ValueError):
            batch_size = _DEFAULT_BATCH_SIZE

        return cls(url=url, region=region, batch_size=batch_size)

    def _signed_post(self, body: bytes) -> bytes:
        """POST ``body`` to the invocation URL with AWS SigV4.  Returns raw bytes."""
        from botocore.auth import SigV4Auth  # type: ignore[import-untyped]
        from botocore.awsrequest import AWSRequest  # type: ignore[import-untyped]
        from botocore.session import Session as BotocoreSession  # type: ignore[import-untyped]

        session = BotocoreSession()
        credentials = session.get_credentials()
        if credentials is None:
            raise RuntimeError("No AWS credentials found in credential chain")

        aws_req = AWSRequest(
            method="POST",
            url=self._url,
            data=body,
            headers={"Content-Type": _CONTENT_TYPE},
        )
        SigV4Auth(credentials, "sagemaker", self._region).add_auth(aws_req)

        http_req = urllib.request.Request(
            self._url,
            data=body,
            headers=dict(aws_req.headers),
            method="POST",
        )
        with urllib.request.urlopen(http_req, timeout=_INVOKE_TIMEOUT) as resp:
            return resp.read()

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
                # e5-base-v2 (BERT) has a hard 512-token position-embedding
                # limit. The SageMaker HF inference toolkit does NOT truncate
                # automatically and returns HTTP 400 on overflow.
                # Binary search: 1000 chars → OK, 1200 chars → FAIL.
                # 1000 is the safe ceiling (~3.3 chars/token for Python code).
                _MAX_CHARS = 1000
                safe_chunk = [t[:_MAX_CHARS] if len(t) > _MAX_CHARS else t for t in chunk]
                body = json.dumps({"inputs": safe_chunk}).encode("utf-8")
                raw: list[list[float]] = json.loads(self._signed_post(body))
                if len(raw) != len(chunk):
                    raise RuntimeError(
                        f"SageMaker returned {len(raw)} vectors for {len(chunk)} inputs"
                    )
                for offset, vec in enumerate(raw):
                    # forge-e5-embed-v1 wraps embeddings in extra list levels:
                    # e.g. raw[i] may be [[[ f0, f1, ...]]] not [f0, f1, ...].
                    # Unwrap until the innermost non-nested float list.
                    actual = vec
                    while actual and isinstance(actual[0], list):
                        actual = actual[0]
                    results[start + offset] = [float(v) for v in actual]

        except Exception as exc:
            logger.warning(
                "SageMakerEmbedder.embed failed (url=%s): %s — caller should fall back",
                self._url,
                exc,
            )
            return None

        out: list[list[float]] = []
        for i, v in enumerate(results):
            if v is None:
                logger.warning("SageMakerEmbedder: missing slot %d — caller should fall back", i)
                return None
            out.append(v)
        return out

    @property
    def url(self) -> str:
        return self._url

    @property
    def region(self) -> str:
        return self._region


@lru_cache(maxsize=1)
def get_sagemaker_embedder() -> SageMakerEmbedder | None:
    """Module-level singleton.  ``None`` when no URL/endpoint env var is set.

    Call ``get_sagemaker_embedder.cache_clear()`` in tests to force re-init.
    """
    embedder = SageMakerEmbedder.from_env()
    if embedder is not None:
        logger.info(
            "SageMakerEmbedder active (url=%s, region=%s, batch_size=%d)",
            embedder.url,
            embedder.region,
            embedder._batch_size,
        )
    return embedder
