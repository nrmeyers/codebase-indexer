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
