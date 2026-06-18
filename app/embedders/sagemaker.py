"""AWS SageMaker Serverless Inference backend (BUC-1605).

Calls the ``forge-e5-embed-v2`` endpoint in ``us-east-1`` via
``boto3.client('sagemaker-runtime')``. This is the default for the
production deploy; standalone installs without AWS creds
should leave this alone and use the ``local`` backend instead.

Endpoint contract
-----------------
::

    POST /endpoints/forge-e5-embed-v2/invocations
    Content-Type: application/json
    Body:        {"inputs": ["chunk1", "chunk2", ...]}
    Response:    [[0.01, -0.98, ...], [...]]
                 — nominally one 768-float vector per input.

LE-129d: the HF ``feature-extraction`` task returns TOKEN-LEVEL embeddings,
shaped ``[batch][tokens][dim]`` (e.g. ``[1][11][768]``), NOT a pre-pooled
``[dim]`` sentence vector. ``embed()`` below recursively mean-pools any
nesting above ``[dim]`` down to one vector per input and L2-normalises the
result client-side so the cosine band matches the LE-123/124 refusal
thresholds. (No-op when the handler already returns flat, normalised
vectors.)

Configuration
-------------
::

    SAGEMAKER_ENDPOINT_NAME    Endpoint name, e.g. "forge-e5-embed-v2".
                               Preferred over the legacy URL form.
    SAGEMAKER_EMBED_ENDPOINT   Legacy alias (still read for backwards compat).
    SAGEMAKER_EMBED_URL        Full invocation URL; endpoint name extracted.
    SAGEMAKER_EMBED_REGION     AWS region (default us-east-1).
    SAGEMAKER_EMBED_BATCH_SIZE Inputs per request (1-64, default 16).
    SAGEMAKER_EMBED_MAX_ATTEMPTS
                               Per-batch invoke attempts incl. the first
                               (1-5, default 5). LE-151b adds explicit
                               exponential-backoff retry for transient
                               model-container failures ("Worker died."
                               500s, 503s, throttling) that botocore's
                               standard retry mode does NOT cover.

Timeouts (boto3 ``Config``) are tuned for batch=32 on ml.m5.large:
    connect_timeout=10s, read_timeout=90s, retries=3 (standard mode).
The botocore ``retries`` config covers connection-level transients and
some throttling; the explicit per-batch retry above (LE-151b) additionally
covers the model-container "Worker died." 500 that a serverless endpoint
emits under bulk-ingest concurrency.

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
import math
import os
import random
import re
import time
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

# ---------------------------------------------------------------------------
# Transient-failure retry (LE-151b).
#
# A serverless SageMaker endpoint under bulk-ingest load returns model-
# container failures that botocore's standard retry mode does NOT cover:
#
#   * ModelError / InternalServerException — HTTP 500, body
#     ``{"message": "Worker died."}``. The model worker OOM'd or crashed.
#     botocore treats a 500 from a *successful* HTTP exchange with the
#     endpoint as a non-retryable ClientError, so the bulk re-embed of
#     PR #82 crashed the endpoint and surfaced the failure to the operator.
#   * ServiceUnavailableException — HTTP 503, the endpoint is scaling /
#     all workers busy. Transient by definition.
#   * ThrottlingException / 429 — too many concurrent invocations.
#
# We retry these explicitly with capped exponential backoff + jitter.
# Everything else (400 ValidationError, dim-mismatch, auth failure) is a
# hard error and propagates immediately — retrying it would just waste
# time and money.
# ---------------------------------------------------------------------------

#: Max attempts (1 initial + up to 4 retries) per batch invocation.
_DEFAULT_MAX_ATTEMPTS = 5
#: Base backoff seconds; sleep = base * 2**(attempt-1) capped at _BACKOFF_CAP,
#: i.e. ~1s, 2s, 4s, 8s, 16s (+ jitter).
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_CAP_SECONDS = 16.0

#: SageMaker-runtime error codes / substrings that are safe to retry.
_RETRYABLE_ERROR_CODES = frozenset(
    {
        "ModelError",
        "InternalServerException",
        "InternalFailure",
        "ServiceUnavailableException",
        "ServiceUnavailable",
        "ThrottlingException",
        "TooManyRequestsException",
        "ModelNotReadyException",
    }
)
#: HTTP status codes that indicate a transient endpoint-side failure.
_RETRYABLE_STATUS = frozenset({429, 500, 503})


def _is_transient_sagemaker_error(exc: BaseException) -> bool:
    """True when ``exc`` is a transient SageMaker failure worth retrying.

    Inspects ``botocore.exceptions.ClientError.response`` for either a
    retryable error ``Code`` (e.g. ``ModelError``) or a retryable HTTP
    status (429/500/503), and matches the model-container "Worker died."
    500 that botocore does NOT auto-retry. Connection/read timeouts
    (``EndpointConnectionError``, ``ReadTimeoutError``, etc.) are also
    treated as transient.

    Args:
        exc: The exception raised by a single ``invoke_endpoint`` call.

    Returns:
        True if the call should be retried; False for hard errors
        (validation, auth, dim mismatch) that must propagate.
    """
    # botocore ClientError carries a structured ``response`` dict.
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        err = response.get("Error") or {}
        code = str(err.get("Code") or "")
        if code in _RETRYABLE_ERROR_CODES:
            return True
        status = (
            response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        )
        if isinstance(status, int) and status in _RETRYABLE_STATUS:
            return True
        message = str(err.get("Message") or "")
        if "Worker died" in message:
            return True

    # Connection-level transients (timeouts, dropped sockets). Match by
    # class name so we don't have to import every botocore exception.
    transient_names = {
        "EndpointConnectionError",
        "ConnectionClosedError",
        "ReadTimeoutError",
        "ConnectTimeoutError",
        "ResponseStreamingError",
    }
    if type(exc).__name__ in transient_names:
        return True

    # The model-container "Worker died." text can also reach us via a bare
    # exception message (depending on botocore version / streaming body).
    if "Worker died" in str(exc):
        return True

    return False


def _mean_pool(x: Any) -> Any:
    """Recursively mean-pool any nesting above ``[dim]`` to one vector.

    The HF ``feature-extraction`` task returns token-level embeddings
    (``[tokens][dim]`` or ``[batch][tokens][dim]``) rather than a single
    pooled sentence vector. This collapses every dimension above the
    innermost float list by averaging element-wise.

    A flat ``[dim]`` float list is returned unchanged, so the helper is a
    no-op when the endpoint already pools server-side.
    """
    if isinstance(x, list) and x and isinstance(x[0], list):
        sub = [_mean_pool(e) for e in x]  # each -> [dim]
        return [sum(c) / len(c) for c in zip(*sub)]
    return x


class SageMakerEmbedder(EmbedderBackend):
    """Boto3-backed client for the SageMaker embedding endpoint."""

    name = "sagemaker"
    dim = EMBEDDING_DIM

    def __init__(
        self,
        endpoint_name: str,
        region: str = "us-east-1",
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        model: str = "e5-base-v2",
        prefix_model: str = "",
    ) -> None:
        if not endpoint_name:
            raise EmbedderError("SageMakerEmbedder: endpoint_name must be non-empty")
        self.endpoint_name = endpoint_name
        self.region = region
        # Instance attr so swapping the underlying SageMaker deployment
        # (e.g. e5-base-v2 -> nomic-embed-text-v1.5) is a single env var
        # change rather than a class-attr lie about what the endpoint serves.
        self.model = model
        # Consulted by :mod:`app.embedders.prefixes`: a SageMaker endpoint
        # serving an instruction-tuned model (e.g. nomic-v1.5) can advertise
        # the HF model id explicitly without depending on the friendlier
        # ``model`` label happening to match a registry key.
        self.prefix_model = prefix_model
        # Clamp to the SageMaker contract (1-64). Larger batches trip the
        # 60s serverless timeout; smaller ones throttle ingest throughput.
        self.batch_size = min(max(1, batch_size), 64)
        # LE-151b: at least one attempt; capped at 5 to bound worst-case
        # latency for a genuinely-dead endpoint (~31s of backoff total).
        self.max_attempts = min(max(1, max_attempts), 5)
        self._client: Any | None = None
        self._nk_coalesce_warned = False

    @classmethod
    def from_env(cls) -> "SageMakerEmbedder":
        """Construct from env vars; raise if no endpoint is configured.

        Resolution priority (highest first):
            1. ``SAGEMAKER_ENDPOINT_NAME`` — preferred BUC-1605 name.
            2. ``SAGEMAKER_EMBED_ENDPOINT`` — legacy alias kept for
               backwards-compat with existing .env files.
            3. ``SAGEMAKER_EMBED_URL`` — full URL, endpoint name extracted.

        Model labelling (independent of endpoint resolution):
            * ``SAGEMAKER_MODEL_ID`` — human label exposed as ``.model``
              (defaults to ``e5-base-v2`` for backwards compat).
            * ``SAGEMAKER_PREFIX_MODEL`` — HF model id consulted by
              :mod:`app.embedders.prefixes` (e.g.
              ``nomic-ai/nomic-embed-text-v1.5``). Required to enable
              query/document prefixing for SageMaker-served instruction-
              tuned models; unset means raw-text behaviour preserved.
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

        from ._env_utils import env_int

        region = (os.environ.get("SAGEMAKER_EMBED_REGION") or "us-east-1").strip()
        model = (os.environ.get("SAGEMAKER_MODEL_ID") or "e5-base-v2").strip()
        prefix_model = (os.environ.get("SAGEMAKER_PREFIX_MODEL") or "").strip()

        return cls(
            endpoint_name=endpoint,
            region=region,
            batch_size=env_int("SAGEMAKER_EMBED_BATCH_SIZE", _DEFAULT_BATCH_SIZE),
            max_attempts=env_int("SAGEMAKER_EMBED_MAX_ATTEMPTS", _DEFAULT_MAX_ATTEMPTS),
            model=model,
            prefix_model=prefix_model,
        )

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

    @staticmethod
    def _sleep(seconds: float) -> None:
        """Blocking sleep seam — overridable in tests to avoid real waits."""
        time.sleep(seconds)

    def _backoff_seconds(self, attempt: int) -> float:
        """Capped exponential backoff with full jitter for retry ``attempt``.

        ``attempt`` is 1-indexed (1 = first retry). Yields ~1s, 2s, 4s, 8s,
        16s ceilings, each randomised across ``[0, ceiling]`` (full jitter)
        so concurrent ingest workers don't synchronise their retries and
        re-stampede a recovering serverless worker.
        """
        ceiling = min(
            _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), _BACKOFF_CAP_SECONDS
        )
        return random.uniform(0.0, ceiling)

    def _invoke_with_retry(self, body: bytes) -> bytes:
        """Invoke the endpoint, retrying transient failures with backoff.

        Retries only failures classified transient by
        :func:`_is_transient_sagemaker_error` (model "Worker died." 500s,
        503s, throttling, connection timeouts) up to ``self.max_attempts``.
        Hard errors propagate on the first failure. After the final attempt
        the last transient exception is re-raised so the caller surfaces a
        loud failure — it is NEVER swallowed into an empty result.

        Args:
            body: JSON-encoded ``{"inputs": [...]}`` request payload.

        Returns:
            The raw response body bytes from a successful invocation.

        Raises:
            The originating exception (transient after exhausting retries,
            or non-transient on first occurrence).
        """
        last_exc: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._invoke_sync(body)
            except Exception as exc:  # noqa: BLE001 — classified below
                last_exc = exc
                transient = _is_transient_sagemaker_error(exc)
                if not transient or attempt >= self.max_attempts:
                    # Hard error, or out of retries — let it propagate so the
                    # batch is recorded as a failure (never stored empty).
                    raise
                delay = self._backoff_seconds(attempt)
                logger.warning(
                    "sagemaker.invoke transient failure (attempt %d/%d): "
                    "%s: %s — retrying in %.2fs",
                    attempt,
                    self.max_attempts,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                self._sleep(delay)
        # Defensive: the loop either returns or raises. If we somehow fall
        # through, re-raise the last seen exception rather than returning.
        assert last_exc is not None  # pragma: no cover
        raise last_exc

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
                # Retry transient endpoint failures ("Worker died." 500s,
                # 503s, throttling) inside the worker thread before
                # converting any surviving failure to a loud EmbedderError.
                raw_bytes = await asyncio.to_thread(self._invoke_with_retry, body)
                raw = json.loads(raw_bytes)
            except Exception as exc:  # noqa: BLE001
                raise EmbedderError(
                    f"SageMakerEmbedder.invoke_endpoint failed after "
                    f"{self.max_attempts} attempt(s) "
                    f"({type(exc).__name__}: {exc})"
                ) from exc

            if not isinstance(raw, list):
                raise EmbedderError(
                    f"SageMaker returned {type(raw).__name__}, expected list"
                )

            # LE-129d: some custom inference handlers (dual-head models, or a
            # batched feature-extraction pass) return K rows per input. Detect
            # the N*K case and keep the first row per input. Server-side fix is
            # to pin a single-vector pooling task in predict_fn.
            if len(raw) > len(chunk) and len(raw) % len(chunk) == 0:
                stride = len(raw) // len(chunk)
                if not self._nk_coalesce_warned:
                    logger.warning(
                        "SageMaker endpoint %r returned %d rows for %d inputs "
                        "(stride=%d) — falling back to first-row-per-input. "
                        "Pin a single-vector pooling task in predict_fn to fix.",
                        self.endpoint_name,
                        len(raw),
                        len(chunk),
                        stride,
                    )
                    self._nk_coalesce_warned = True
                raw = [raw[i * stride] for i in range(len(chunk))]

            if len(raw) != len(chunk):
                raise EmbedderError(
                    f"SageMaker returned {len(raw)} embeddings for {len(chunk)} inputs"
                )

            for i, vec in enumerate(raw):
                # LE-129d: some handlers return inner vectors as JSON-encoded
                # strings instead of float lists. Decode before pooling.
                if isinstance(vec, str):
                    try:
                        vec = json.loads(vec)
                    except (json.JSONDecodeError, ValueError) as exc:
                        raise EmbedderError(
                            f"SageMaker returned non-decodable string for input "
                            f"{start + i}: {type(exc).__name__}"
                        ) from exc
                # LE-129d: E5 via the HF feature-extraction task returns
                # TOKEN-LEVEL embeddings, e.g. [tokens][dim] or
                # [batch][tokens][dim] ([1][11][768]). Recursively mean-pool any
                # nesting above [dim] until one pooled sentence vector remains.
                # (No-op when already a flat [dim] float list.)
                vec = _mean_pool(vec)
                if not isinstance(vec, list) or len(vec) != EMBEDDING_DIM:
                    raise EmbedderError(
                        f"SageMaker returned {len(vec) if isinstance(vec, list) else type(vec).__name__}-"
                        f"dim vector for input {start + i}; expected {EMBEDDING_DIM}"
                    )
                floats = [float(v) for v in vec]
                # LE-129d: L2-normalise client-side so ingest AND query vectors
                # share one cosine range, keeping the calibrated LE-123/124
                # refusal-score thresholds portable. No-op for already-unit
                # vectors (E5 is normalised server-side at magnitude ~1.0).
                norm = math.sqrt(sum(v * v for v in floats))
                if norm > 0.0:
                    floats = [v / norm for v in floats]
                results.append(floats)

        return results
