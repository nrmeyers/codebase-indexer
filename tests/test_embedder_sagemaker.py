"""Unit tests for :class:`app.embedders.sagemaker.SageMakerEmbedder`.

boto3 is fully mocked — the suite never opens a real AWS connection.
The interesting behaviour is the env-resolution priority (preferred name
> legacy alias > URL-derived name) and the dim-mismatch guard.
"""
from __future__ import annotations

import json
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
