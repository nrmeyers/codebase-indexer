"""Unit tests for the pluggable embedder layer (BUC-1605).

Covers:
    1. Factory dispatch — EMBEDDER_BACKEND env var selects the right class.
    2. Unknown values fall back to ``local`` (with a warning).
    3. Each backend produces 768-dim vectors for a known input string.
    4. SageMaker raises EmbedderError when no endpoint is configured.
    5. Empty input yields ``[]`` without a network round-trip.

The ``local`` backend test is gated on ``sentence-transformers`` being
installed — if the optional extra isn't present, we skip rather than fail.
The ``sagemaker`` and ``tei`` backends are exercised against mocks so the
suite stays fast and offline.
"""
from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app import embedders
from app.embedders import EMBEDDING_DIM, EmbedderError, get_embedder
from app.embedders.base import EmbedderBackend
from app.embedders.local import LocalEmbedder
from app.embedders.sagemaker import SageMakerEmbedder
from app.embedders.tei import TEIEmbedder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_factory_cache() -> None:
    """Reset the lru_cache so env-driven tests see a fresh embedder.

    Also strips backend-selector env vars that may leak in from the
    host's .env so test cases get a clean baseline.
    """
    get_embedder.cache_clear()
    for key in (
        "EMBEDDER_BACKEND",
        "SAGEMAKER_ENDPOINT_NAME",
        "SAGEMAKER_EMBED_ENDPOINT",
        "SAGEMAKER_EMBED_URL",
        "SAGEMAKER_EMBED_REGION",
        "TEI_URL",
        "TEI_TIMEOUT_MS",
    ):
        os.environ.pop(key, None)
    yield
    get_embedder.cache_clear()


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------


def test_factory_returns_local_by_default() -> None:
    """Unset EMBEDDER_BACKEND → LocalEmbedder (the BUC-1605 default)."""
    backend = get_embedder()
    assert isinstance(backend, LocalEmbedder)
    assert backend.name == "local"


def test_factory_returns_sagemaker_when_selected() -> None:
    """EMBEDDER_BACKEND=sagemaker + endpoint set → SageMakerEmbedder."""
    with patch.dict(
        os.environ,
        {
            "EMBEDDER_BACKEND": "sagemaker",
            "SAGEMAKER_ENDPOINT_NAME": "forge-e5-embed-v2",
            "SAGEMAKER_EMBED_REGION": "us-east-1",
        },
    ):
        get_embedder.cache_clear()
        backend = get_embedder()
    assert isinstance(backend, SageMakerEmbedder)
    assert backend.name == "sagemaker"
    assert backend.endpoint_name == "forge-e5-embed-v2"
    assert backend.region == "us-east-1"


def test_factory_returns_tei_when_selected() -> None:
    """EMBEDDER_BACKEND=tei → TEIEmbedder pointed at TEI_URL."""
    with patch.dict(
        os.environ,
        {"EMBEDDER_BACKEND": "tei", "TEI_URL": "http://tei-sidecar:8080"},
    ):
        get_embedder.cache_clear()
        backend = get_embedder()
    assert isinstance(backend, TEIEmbedder)
    assert backend.name == "tei"
    assert backend.base_url == "http://tei-sidecar:8080"


def test_factory_falls_back_to_local_for_unknown_backend(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Typo'd EMBEDDER_BACKEND value → fall back to local with a warning."""
    import logging

    with patch.dict(os.environ, {"EMBEDDER_BACKEND": "openai"}):
        get_embedder.cache_clear()
        with caplog.at_level(logging.WARNING, logger="app.embedders"):
            backend = get_embedder()
    assert isinstance(backend, LocalEmbedder)
    assert any(
        "EMBEDDER_BACKEND=" in r.message and "not recognised" in r.message
        for r in caplog.records
    )


def test_factory_caches_singleton() -> None:
    """Repeated calls return the same instance (lru_cache contract)."""
    a = get_embedder()
    b = get_embedder()
    assert a is b


# ---------------------------------------------------------------------------
# Protocol conformance — every backend implements EmbedderBackend
# ---------------------------------------------------------------------------


def test_local_embedder_satisfies_protocol() -> None:
    assert isinstance(LocalEmbedder(), EmbedderBackend)


def test_sagemaker_embedder_satisfies_protocol() -> None:
    sm = SageMakerEmbedder(endpoint_name="forge-e5-embed-v2")
    assert isinstance(sm, EmbedderBackend)


def test_tei_embedder_satisfies_protocol() -> None:
    assert isinstance(TEIEmbedder(), EmbedderBackend)


# ---------------------------------------------------------------------------
# Local backend
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_embedder_produces_768_dim_vector() -> None:
    """Real sentence-transformers run — skipped when the extra isn't installed.

    This is intentionally a small smoke test (one short string) so the
    suite stays fast. Model load is cached across pytest invocations via
    the HuggingFace cache.
    """
    pytest.importorskip("sentence_transformers")

    backend = LocalEmbedder()
    vectors = await backend.embed(["def greet(): return 'hello'"])
    assert len(vectors) == 1
    assert len(vectors[0]) == EMBEDDING_DIM
    assert all(isinstance(x, float) for x in vectors[0])


@pytest.mark.asyncio
async def test_local_embedder_empty_input_short_circuits() -> None:
    """Empty list returns [] without loading the model."""
    backend = LocalEmbedder()
    assert await backend.embed([]) == []
    # Model must not have been loaded on an empty call.
    assert backend._model is None


@pytest.mark.asyncio
async def test_local_embedder_missing_sentence_transformers_raises() -> None:
    """Clear error when the optional extra isn't installed."""
    backend = LocalEmbedder()
    # Force the ImportError path by patching the import target name in
    # sys.modules to None.
    import sys

    saved = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = None  # type: ignore[assignment]
    try:
        with pytest.raises(EmbedderError, match="sentence-transformers"):
            await backend.embed(["hello"])
    finally:
        if saved is not None:
            sys.modules["sentence_transformers"] = saved
        else:
            sys.modules.pop("sentence_transformers", None)


# ---------------------------------------------------------------------------
# SageMaker backend
# ---------------------------------------------------------------------------


def test_sagemaker_from_env_raises_when_endpoint_unset() -> None:
    """No endpoint configured → loud EmbedderError (vs. silent None)."""
    with pytest.raises(EmbedderError, match="no endpoint is configured"):
        SageMakerEmbedder.from_env()


def test_sagemaker_from_env_prefers_endpoint_name() -> None:
    """SAGEMAKER_ENDPOINT_NAME wins over the legacy aliases."""
    with patch.dict(
        os.environ,
        {
            "SAGEMAKER_ENDPOINT_NAME": "forge-e5-embed-v2",
            "SAGEMAKER_EMBED_ENDPOINT": "legacy-name",
            "SAGEMAKER_EMBED_URL": (
                "https://runtime.sagemaker.us-east-1.amazonaws.com"
                "/endpoints/url-derived-name/invocations"
            ),
        },
    ):
        sm = SageMakerEmbedder.from_env()
    assert sm.endpoint_name == "forge-e5-embed-v2"


def test_sagemaker_from_env_extracts_endpoint_from_url() -> None:
    """SAGEMAKER_EMBED_URL alone → endpoint name parsed from URL."""
    with patch.dict(
        os.environ,
        {
            "SAGEMAKER_EMBED_URL": (
                "https://runtime.sagemaker.us-east-1.amazonaws.com"
                "/endpoints/forge-e5-embed-v2/invocations"
            ),
        },
    ):
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
    assert len(result[1]) == EMBEDDING_DIM
    fake_client.invoke_endpoint.assert_called_once()
    call_kwargs = fake_client.invoke_endpoint.call_args.kwargs
    assert call_kwargs["EndpointName"] == "forge-e5-embed-v2"
    payload = json.loads(call_kwargs["Body"])
    assert payload == {"inputs": ["def foo(): return 1", "class Bar: pass"]}


@pytest.mark.asyncio
async def test_sagemaker_embedder_raises_on_dim_mismatch() -> None:
    """Wrong-dim response from SageMaker → EmbedderError (not silent corruption)."""
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
# TEI backend
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tei_embedder_returns_768_dim_vectors() -> None:
    """Mock httpx.AsyncClient; verify request shape + response parsing."""
    fake_vec = [0.001 * i for i in range(EMBEDDING_DIM)]
    fake_response = MagicMock(spec=httpx.Response)
    fake_response.status_code = 200
    fake_response.json.return_value = [fake_vec, fake_vec]

    fake_client = MagicMock(spec=httpx.AsyncClient)
    fake_client.is_closed = False
    fake_client.post = AsyncMock(return_value=fake_response)

    tei = TEIEmbedder(base_url="http://tei:8080")
    tei._client = fake_client  # bypass _get_client lazy-init

    result = await tei.embed(["def foo(): pass", "class Bar: pass"])

    assert len(result) == 2
    assert len(result[0]) == EMBEDDING_DIM
    fake_client.post.assert_awaited_once_with(
        "/embed",
        json={"inputs": ["def foo(): pass", "class Bar: pass"], "normalize": True},
    )


@pytest.mark.asyncio
async def test_tei_embedder_raises_on_http_error() -> None:
    """Non-200 → EmbedderError carrying status + body snippet."""
    fake_response = MagicMock(spec=httpx.Response)
    fake_response.status_code = 503
    fake_response.text = "model warming up"

    fake_client = MagicMock(spec=httpx.AsyncClient)
    fake_client.is_closed = False
    fake_client.post = AsyncMock(return_value=fake_response)

    tei = TEIEmbedder(base_url="http://tei:8080")
    tei._client = fake_client

    with pytest.raises(EmbedderError, match="HTTP 503"):
        await tei.embed(["hello"])


@pytest.mark.asyncio
async def test_tei_embedder_raises_on_dim_mismatch() -> None:
    """Wrong-dim response → EmbedderError before vectors leak into DuckDB."""
    fake_response = MagicMock(spec=httpx.Response)
    fake_response.status_code = 200
    fake_response.json.return_value = [[0.0] * 256]

    fake_client = MagicMock(spec=httpx.AsyncClient)
    fake_client.is_closed = False
    fake_client.post = AsyncMock(return_value=fake_response)

    tei = TEIEmbedder(base_url="http://tei:8080")
    tei._client = fake_client

    with pytest.raises(EmbedderError, match="256-dim vector"):
        await tei.embed(["text"])


@pytest.mark.asyncio
async def test_tei_embedder_empty_input_short_circuits() -> None:
    """Empty list returns [] without an HTTP request."""
    fake_client = MagicMock(spec=httpx.AsyncClient)
    fake_client.post = AsyncMock()
    tei = TEIEmbedder()
    tei._client = fake_client
    assert await tei.embed([]) == []
    fake_client.post.assert_not_called()


# ---------------------------------------------------------------------------
# Module-level surface
# ---------------------------------------------------------------------------


def test_module_exports() -> None:
    """Public API surface is stable — guard against accidental removals."""
    assert hasattr(embedders, "get_embedder")
    assert hasattr(embedders, "EmbedderBackend")
    assert hasattr(embedders, "EmbedderError")
    assert embedders.EMBEDDING_DIM == 768
    assert set(embedders.VALID_BACKENDS) == {"local", "sagemaker", "tei"}
