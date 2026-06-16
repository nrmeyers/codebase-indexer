"""Unit tests for :class:`app.embedders.tei.TEIEmbedder`.

Network is fully mocked via ``unittest.mock.AsyncMock`` so the suite runs
offline. The TEI sidecar contract is small (POST /embed with {"inputs":
[...], "normalize": true}); the tests assert the request shape, dim
validation, and HTTP error mapping.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.embedders import EMBEDDING_DIM, EmbedderError
from app.embedders.base import EmbedderBackend
from app.embedders.tei import TEIEmbedder


def _patch_async_client(monkeypatch: pytest.MonkeyPatch, fake_client: MagicMock) -> None:
    """Patch httpx.AsyncClient so ``async with`` yields ``fake_client``."""
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=fake_client)
    ctx.__aexit__ = AsyncMock(return_value=None)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: ctx)


def test_tei_embedder_satisfies_protocol() -> None:
    assert isinstance(TEIEmbedder(), EmbedderBackend)


def test_tei_embedder_exposes_dim() -> None:
    """dim is 768 — TEI serves e5-base-v2 by contract."""
    tei = TEIEmbedder()
    assert tei.dim == EMBEDDING_DIM
    assert tei.name == "tei"
    assert tei.model == "e5-base-v2"


@pytest.mark.asyncio
async def test_tei_embedder_returns_768_dim_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock httpx.AsyncClient; verify request shape + response parsing."""
    fake_vec = [0.001 * i for i in range(EMBEDDING_DIM)]
    fake_response = MagicMock(spec=httpx.Response)
    fake_response.status_code = 200
    fake_response.json.return_value = [fake_vec, fake_vec]

    fake_client = MagicMock(spec=httpx.AsyncClient)
    fake_client.post = AsyncMock(return_value=fake_response)
    _patch_async_client(monkeypatch, fake_client)

    tei = TEIEmbedder(base_url="http://tei:8080")

    result = await tei.embed(["def foo(): pass", "class Bar: pass"])

    assert len(result) == 2
    assert len(result[0]) == EMBEDDING_DIM
    fake_client.post.assert_awaited_once_with(
        "/embed",
        json={"inputs": ["def foo(): pass", "class Bar: pass"], "normalize": True},
    )


@pytest.mark.asyncio
async def test_tei_embedder_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-200 → EmbedderError carrying status + body snippet."""
    fake_response = MagicMock(spec=httpx.Response)
    fake_response.status_code = 503
    fake_response.text = "model warming up"

    fake_client = MagicMock(spec=httpx.AsyncClient)
    fake_client.post = AsyncMock(return_value=fake_response)
    _patch_async_client(monkeypatch, fake_client)

    tei = TEIEmbedder(base_url="http://tei:8080")

    with pytest.raises(EmbedderError, match="HTTP 503"):
        await tei.embed(["hello"])


@pytest.mark.asyncio
async def test_tei_embedder_raises_on_dim_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrong-dim response → EmbedderError before vectors leak into DuckDB."""
    fake_response = MagicMock(spec=httpx.Response)
    fake_response.status_code = 200
    fake_response.json.return_value = [[0.0] * 256]

    fake_client = MagicMock(spec=httpx.AsyncClient)
    fake_client.post = AsyncMock(return_value=fake_response)
    _patch_async_client(monkeypatch, fake_client)

    tei = TEIEmbedder(base_url="http://tei:8080")

    with pytest.raises(EmbedderError, match="256-dim vector"):
        await tei.embed(["text"])


@pytest.mark.asyncio
async def test_tei_embedder_empty_input_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty list returns [] without an HTTP request."""
    fake_client = MagicMock(spec=httpx.AsyncClient)
    fake_client.post = AsyncMock()
    _patch_async_client(monkeypatch, fake_client)
    tei = TEIEmbedder()
    assert await tei.embed([]) == []
    fake_client.post.assert_not_called()
