"""Unit tests for :class:`app.embedders.sagemaker.SageMakerEmbedder`.

boto3 is fully mocked — the suite never opens a real AWS connection.
The interesting behaviour is the env-resolution priority (preferred name
> legacy alias > URL-derived name) and the dim-mismatch guard.
"""
from __future__ import annotations

import json
import math
import os
from unittest.mock import MagicMock

import pytest

from app.embedders import EMBEDDING_DIM, EmbedderError
from app.embedders.base import EmbedderBackend
from app.embedders.sagemaker import SageMakerEmbedder


@pytest.fixture(autouse=True)
def _scrub_env() -> None:
    """Strip any SAGEMAKER_* env vars that may leak in from the host's .env."""
    keys = (
        "SAGEMAKER_ENDPOINT_NAME",
        "SAGEMAKER_EMBED_ENDPOINT",
        "SAGEMAKER_EMBED_URL",
        "SAGEMAKER_EMBED_REGION",
        "SAGEMAKER_EMBED_BATCH_SIZE",
    )
    saved = {k: os.environ.pop(k, None) for k in keys}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v


def test_sagemaker_embedder_satisfies_protocol() -> None:
    sm = SageMakerEmbedder(endpoint_name="forge-e5-embed-v2")
    assert isinstance(sm, EmbedderBackend)


def test_sagemaker_embedder_exposes_dim() -> None:
    """dim is 768 — SageMaker serves e5-base-v2 by contract."""
    sm = SageMakerEmbedder(endpoint_name="forge-e5-embed-v2")
    assert sm.dim == EMBEDDING_DIM
    assert sm.name == "sagemaker"


def test_sagemaker_from_env_raises_when_endpoint_unset() -> None:
    """No endpoint configured → loud EmbedderError (vs. silent None)."""
    with pytest.raises(EmbedderError, match="no endpoint is configured"):
        SageMakerEmbedder.from_env()


def test_sagemaker_from_env_prefers_endpoint_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SAGEMAKER_ENDPOINT_NAME wins over the legacy aliases."""
    monkeypatch.setenv("SAGEMAKER_ENDPOINT_NAME", "forge-e5-embed-v2")
    monkeypatch.setenv("SAGEMAKER_EMBED_ENDPOINT", "legacy-name")
    monkeypatch.setenv(
        "SAGEMAKER_EMBED_URL",
        "https://runtime.sagemaker.us-east-1.amazonaws.com"
        "/endpoints/url-derived-name/invocations",
    )
    sm = SageMakerEmbedder.from_env()
    assert sm.endpoint_name == "forge-e5-embed-v2"


def test_sagemaker_from_env_extracts_endpoint_from_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SAGEMAKER_EMBED_URL alone → endpoint name parsed from URL."""
    monkeypatch.setenv(
        "SAGEMAKER_EMBED_URL",
        "https://runtime.sagemaker.us-east-1.amazonaws.com"
        "/endpoints/forge-e5-embed-v2/invocations",
    )
    sm = SageMakerEmbedder.from_env()
    assert sm.endpoint_name == "forge-e5-embed-v2"


@pytest.mark.asyncio
async def test_sagemaker_embedder_returns_768_dim_vectors() -> None:
    """Mock invoke_endpoint; verify request/response shape end-to-end."""
    fake_vec = [0.001 * i for i in range(EMBEDDING_DIM)]
    fake_body = MagicMock()
    fake_body.read.return_value = json.dumps([fake_vec, fake_vec]).encode("utf-8")
    fake_client = MagicMock()
    fake_client.invoke_endpoint.return_value = {"Body": fake_body}

    sm = SageMakerEmbedder(endpoint_name="forge-e5-embed-v2")
    sm._client = fake_client  # bypass lazy-init / boto3

    result = await sm.embed(["def foo(): return 1", "class Bar: pass"])

    assert len(result) == 2
    assert len(result[0]) == EMBEDDING_DIM
    fake_client.invoke_endpoint.assert_called_once()
    call_kwargs = fake_client.invoke_endpoint.call_args.kwargs
    assert call_kwargs["EndpointName"] == "forge-e5-embed-v2"
    payload = json.loads(call_kwargs["Body"])
    assert payload == {"inputs": ["def foo(): return 1", "class Bar: pass"]}


@pytest.mark.asyncio
async def test_sagemaker_embedder_raises_on_dim_mismatch() -> None:
    """Wrong-dim response → EmbedderError (not silent corruption)."""
    fake_body = MagicMock()
    fake_body.read.return_value = json.dumps([[0.0] * 512]).encode("utf-8")
    fake_client = MagicMock()
    fake_client.invoke_endpoint.return_value = {"Body": fake_body}

    sm = SageMakerEmbedder(endpoint_name="forge-e5-embed-v2")
    sm._client = fake_client

    with pytest.raises(EmbedderError, match="512-dim vector"):
        await sm.embed(["text"])


@pytest.mark.asyncio
async def test_sagemaker_embedder_empty_input_short_circuits() -> None:
    """Empty list returns [] without an invoke_endpoint call."""
    fake_client = MagicMock()
    sm = SageMakerEmbedder(endpoint_name="forge-e5-embed-v2")
    sm._client = fake_client
    assert await sm.embed([]) == []
    fake_client.invoke_endpoint.assert_not_called()


# ---------------------------------------------------------------------------
# LE-151b — transient-failure retry with exponential backoff.
#
# A serverless endpoint under bulk-ingest load returns a model-container
# "Worker died." 500 (ModelError / InternalServerException) that botocore's
# standard retry mode does NOT auto-retry. These tests prove the explicit
# per-batch retry recovers on a transient error, exhausts after max_attempts,
# and never retries a hard (non-transient) error.
# ---------------------------------------------------------------------------


def _client_error(code: str, status: int, message: str = "") -> Exception:
    """Build a botocore-shaped ClientError for the given code/status."""
    from botocore.exceptions import ClientError

    return ClientError(
        error_response={
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        operation_name="InvokeEndpoint",
    )


@pytest.mark.asyncio
async def test_sagemaker_retries_worker_died_500_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient 'Worker died.' 500 is retried with backoff, then succeeds.

    Pins the LE-151b contract: the model-container failure that crashed the
    PR #82 bulk re-embed is now recovered instead of aborting the job.
    """
    fake_vec = [0.001 * i for i in range(EMBEDDING_DIM)]
    ok_body = MagicMock()
    ok_body.read.return_value = json.dumps([fake_vec]).encode("utf-8")

    calls = {"n": 0}

    def _flaky(**_kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            # Fails twice with the exact serverless model-worker OOM error.
            raise _client_error(
                "ModelError", 500, "Received server error: Worker died."
            )
        return {"Body": ok_body}

    fake_client = MagicMock()
    fake_client.invoke_endpoint.side_effect = _flaky

    sm = SageMakerEmbedder(endpoint_name="forge-e5-embed-v2", max_attempts=5)
    sm._client = fake_client

    slept: list[float] = []
    monkeypatch.setattr(
        SageMakerEmbedder, "_sleep", staticmethod(lambda s: slept.append(s))
    )

    result = await sm.embed(["def foo(): return 1"])

    assert calls["n"] == 3, "should retry twice before the third call succeeds"
    assert len(result) == 1 and len(result[0]) == EMBEDDING_DIM
    # Backoff was applied between the two failures (2 sleeps).
    assert len(slept) == 2
    assert all(s >= 0 for s in slept)


@pytest.mark.asyncio
async def test_sagemaker_raises_after_exhausting_retries_never_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent transient failure exhausts retries and raises loudly.

    The call must NEVER fall through to an empty / partial result — a failed
    embed has to be detectable, not silently stored.
    """
    fake_client = MagicMock()
    fake_client.invoke_endpoint.side_effect = _client_error(
        "InternalServerException", 500, "Worker died."
    )

    sm = SageMakerEmbedder(endpoint_name="forge-e5-embed-v2", max_attempts=3)
    sm._client = fake_client
    monkeypatch.setattr(
        SageMakerEmbedder, "_sleep", staticmethod(lambda _s: None)
    )

    with pytest.raises(EmbedderError, match="after 3 attempt"):
        await sm.embed(["text"])

    # Exactly max_attempts invocations — no more, no fewer.
    assert fake_client.invoke_endpoint.call_count == 3


@pytest.mark.asyncio
async def test_sagemaker_does_not_retry_hard_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-transient 400 ValidationError fails immediately (no retry).

    Retrying a malformed-request error just wastes time and money.
    """
    fake_client = MagicMock()
    fake_client.invoke_endpoint.side_effect = _client_error(
        "ValidationError", 400, "bad input"
    )

    sm = SageMakerEmbedder(endpoint_name="forge-e5-embed-v2", max_attempts=5)
    sm._client = fake_client
    slept: list[float] = []
    monkeypatch.setattr(
        SageMakerEmbedder, "_sleep", staticmethod(lambda s: slept.append(s))
    )

    with pytest.raises(EmbedderError):
        await sm.embed(["text"])

    # One attempt only; no backoff sleeps.
    assert fake_client.invoke_endpoint.call_count == 1
    assert slept == []


def test_sagemaker_classifies_transient_errors() -> None:
    """_is_transient_sagemaker_error: retryable codes/status vs hard errors."""
    from app.embedders.sagemaker import _is_transient_sagemaker_error

    assert _is_transient_sagemaker_error(
        _client_error("ModelError", 500, "Worker died.")
    )
    assert _is_transient_sagemaker_error(
        _client_error("ServiceUnavailableException", 503)
    )
    assert _is_transient_sagemaker_error(
        _client_error("ThrottlingException", 429)
    )
    # 500 status with an unknown code is still transient.
    assert _is_transient_sagemaker_error(_client_error("SomethingElse", 500))
    # Hard errors are NOT transient.
    assert not _is_transient_sagemaker_error(
        _client_error("ValidationError", 400)
    )
    assert not _is_transient_sagemaker_error(ValueError("nope"))


def test_sagemaker_from_env_reads_max_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SAGEMAKER_EMBED_MAX_ATTEMPTS is honoured and clamped to [1, 5]."""
    monkeypatch.setenv("SAGEMAKER_ENDPOINT_NAME", "forge-e5-embed-v2")
    monkeypatch.setenv("SAGEMAKER_EMBED_MAX_ATTEMPTS", "3")
    assert SageMakerEmbedder.from_env().max_attempts == 3

    monkeypatch.setenv("SAGEMAKER_EMBED_MAX_ATTEMPTS", "99")
    assert SageMakerEmbedder.from_env().max_attempts == 5

    monkeypatch.setenv("SAGEMAKER_EMBED_MAX_ATTEMPTS", "garbage")
    assert SageMakerEmbedder.from_env().max_attempts == 5
def _is_unit_vector(vec: list[float], tol: float = 1e-6) -> bool:
    """True when ``vec`` is L2-normalised (magnitude 1.0 within ``tol``)."""
    norm = math.sqrt(sum(v * v for v in vec))
    return abs(norm - 1.0) < tol


@pytest.mark.asyncio
async def test_sagemaker_embedder_mean_pools_token_level_response() -> None:
    """LE-129d: E5 feature-extraction returns [1][N][768] token-level output.

    The HF ``feature-extraction`` task does NOT pool server-side, so a single
    input comes back shaped ``[batch][tokens][dim]`` (here ``[1][11][768]``).
    ``embed()`` must recursively mean-pool that down to ONE 768-dim unit
    vector — without the fix it returns a malformed nested/1-dim result and
    reindex + search fail.
    """
    n_tokens = 11
    # Deterministic, distinct per-token vectors so the mean is a real average,
    # not a degenerate constant: token t contributes (t + 1) in every dim.
    token_level = [[float(t + 1)] * EMBEDDING_DIM for t in range(n_tokens)]
    # Wrap one extra time to mimic the [batch][tokens][dim] = [1][11][768] shape.
    payload = [[token_level]]  # raw[0] == [tokens][dim] for the single input

    fake_body = MagicMock()
    fake_body.read.return_value = json.dumps(payload).encode("utf-8")
    fake_client = MagicMock()
    fake_client.invoke_endpoint.return_value = {"Body": fake_body}

    sm = SageMakerEmbedder(endpoint_name="forge-e5-embed-v2")
    sm._client = fake_client

    result = await sm.embed(["def foo(): return 1"])

    assert len(result) == 1, "one input -> one pooled vector"
    assert len(result[0]) == EMBEDDING_DIM, "pooled down to a single [768] vector"
    # Mean of token values 1..11 is 6.0 in every dim before normalisation;
    # after L2-normalise each component is 6 / sqrt(768 * 36) = 1/sqrt(768).
    expected_component = 1.0 / math.sqrt(EMBEDDING_DIM)
    assert all(abs(v - expected_component) < 1e-6 for v in result[0])
    assert _is_unit_vector(result[0])


@pytest.mark.asyncio
async def test_sagemaker_embedder_pooling_is_noop_for_already_pooled() -> None:
    """LE-129d: an already-pooled [batch][768] response passes through.

    When the handler pools server-side and returns a flat ``[dim]`` vector per
    input, ``_mean_pool`` is a no-op; only L2-normalisation is applied so the
    cosine band matches the calibrated refusal thresholds.
    """
    flat_vec = [0.001 * i for i in range(EMBEDDING_DIM)]  # unnormalised
    fake_body = MagicMock()
    fake_body.read.return_value = json.dumps([flat_vec, flat_vec]).encode("utf-8")
    fake_client = MagicMock()
    fake_client.invoke_endpoint.return_value = {"Body": fake_body}

    sm = SageMakerEmbedder(endpoint_name="forge-e5-embed-v2")
    sm._client = fake_client

    result = await sm.embed(["a", "b"])

    assert len(result) == 2
    assert len(result[0]) == EMBEDDING_DIM
    assert _is_unit_vector(result[0]), "flat input is normalised, not re-pooled"
    # Direction preserved: normalised vector is proportional to the input.
    norm = math.sqrt(sum(v * v for v in flat_vec))
    assert all(abs(result[0][i] - flat_vec[i] / norm) < 1e-9 for i in range(EMBEDDING_DIM))
