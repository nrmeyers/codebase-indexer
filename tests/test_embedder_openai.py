"""Unit tests for :class:`app.embedders.openai.OpenAIEmbedder`.

The OpenAI SDK is mocked end-to-end via ``unittest.mock`` so the suite
runs offline and never burns real API quota. We assert:

    * ``from_env`` requires ``OPENAI_API_KEY``.
    * Default model is ``text-embedding-3-small`` (1536-dim).
    * ``text-embedding-3-large`` correctly reports 3072-dim.
    * ``OPENAI_EMBED_DIM`` truncates dim via Matryoshka and the
      ``dimensions`` kwarg is forwarded to the SDK.
    * The SDK ``embeddings.create`` is called with the expected payload.
    * Wrong-dim response → :class:`EmbedderError`.
    * Empty input short-circuits without a network call.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.embedders import EmbedderError
from app.embedders.base import EmbedderBackend
from app.embedders.openai import (
    DEFAULT_MODEL,
    OpenAIEmbedder,
    _NATIVE_DIMS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_openai_client(vectors: list[list[float]]) -> MagicMock:
    """Return a MagicMock that mimics ``openai.OpenAI()`` for one call.

    ``embeddings.create`` returns an object with a ``.data`` list of items,
    each carrying an ``.embedding`` attribute — exactly the shape the
    OpenAI SDK >=1.0 returns.
    """
    items = [SimpleNamespace(embedding=v) for v in vectors]
    resp = SimpleNamespace(data=items)
    client = MagicMock()
    client.embeddings.create.return_value = resp
    return client


# ---------------------------------------------------------------------------
# Construction / env resolution
# ---------------------------------------------------------------------------


def test_openai_embedder_satisfies_protocol() -> None:
    assert isinstance(OpenAIEmbedder(api_key="sk-test"), EmbedderBackend)


def test_openai_embedder_default_dim_is_1536() -> None:
    """Default model = text-embedding-3-small → 1536-dim."""
    e = OpenAIEmbedder(api_key="sk-test")
    assert e.model == DEFAULT_MODEL
    assert e.dim == 1536
    assert e.name == "openai"


def test_openai_embedder_large_model_is_3072() -> None:
    """text-embedding-3-large → 3072-dim."""
    e = OpenAIEmbedder(api_key="sk-test", model="text-embedding-3-large")
    assert e.dim == 3072


def test_openai_embedder_matryoshka_dim_override() -> None:
    """Operator can truncate dim via the constructor."""
    e = OpenAIEmbedder(
        api_key="sk-test", model="text-embedding-3-small", dim=768
    )
    assert e.dim == 768


def test_openai_embedder_rejects_dim_above_native() -> None:
    """Asking for more dims than the model produces is a config error."""
    with pytest.raises(EmbedderError, match="exceeds"):
        OpenAIEmbedder(
            api_key="sk-test", model="text-embedding-3-small", dim=4096
        )


def test_openai_embedder_unknown_model_defers_dim() -> None:
    """Custom / Azure model → dim discovered on first embed call."""
    e = OpenAIEmbedder(api_key="sk-test", model="my-custom-fine-tune")
    assert e.dim == 0


def test_openai_embedder_requires_api_key() -> None:
    """Empty key fails construction loudly."""
    with pytest.raises(EmbedderError, match="OPENAI_API_KEY"):
        OpenAIEmbedder(api_key="")


def test_openai_from_env_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """from_env() with no key raises before touching the SDK."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(EmbedderError, match="OPENAI_API_KEY"):
        OpenAIEmbedder.from_env()


def test_openai_from_env_reads_model_and_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All env vars get plumbed through from_env."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example.com/v1")
    monkeypatch.setenv("OPENAI_EMBED_DIM", "1024")
    e = OpenAIEmbedder.from_env()
    assert e.model == "text-embedding-3-large"
    assert e.base_url == "https://gateway.example.com/v1"
    assert e.dim == 1024


def test_openai_from_env_rejects_garbage_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-integer OPENAI_EMBED_DIM → EmbedderError at construction."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_EMBED_DIM", "not-a-number")
    with pytest.raises(EmbedderError, match="not an integer"):
        OpenAIEmbedder.from_env()


def test_native_dims_table_covers_production_models() -> None:
    """Guard against accidentally dropping a model from the dim table."""
    assert _NATIVE_DIMS["text-embedding-3-small"] == 1536
    assert _NATIVE_DIMS["text-embedding-3-large"] == 3072


# ---------------------------------------------------------------------------
# embed() — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_embed_calls_sdk_with_expected_payload() -> None:
    """embed() forwards inputs to the SDK; dim matches native model size."""
    fake_vec = [0.001 * i for i in range(1536)]
    fake_client = _fake_openai_client([fake_vec, fake_vec])

    e = OpenAIEmbedder(api_key="sk-test")
    e._client = fake_client  # bypass lazy import / construction

    result = await e.embed(["def foo(): pass", "class Bar: pass"])

    assert len(result) == 2
    assert len(result[0]) == 1536
    fake_client.embeddings.create.assert_called_once()
    kwargs = fake_client.embeddings.create.call_args.kwargs
    assert kwargs["model"] == DEFAULT_MODEL
    assert kwargs["input"] == ["def foo(): pass", "class Bar: pass"]
    # No explicit dimensions kwarg when running at the native size.
    assert "dimensions" not in kwargs


@pytest.mark.asyncio
async def test_openai_embed_passes_dimensions_when_truncated() -> None:
    """Matryoshka truncation → dimensions kwarg is forwarded to the SDK."""
    fake_vec = [0.001 * i for i in range(768)]
    fake_client = _fake_openai_client([fake_vec])

    e = OpenAIEmbedder(api_key="sk-test", dim=768)
    e._client = fake_client

    await e.embed(["hello"])

    kwargs = fake_client.embeddings.create.call_args.kwargs
    assert kwargs["dimensions"] == 768


@pytest.mark.asyncio
async def test_openai_embed_unknown_model_records_dim_on_first_call() -> None:
    """Unknown model: dim is filled in from the first response."""
    fake_vec = [0.5] * 1024
    fake_client = _fake_openai_client([fake_vec])

    e = OpenAIEmbedder(api_key="sk-test", model="my-azure-deployment")
    e._client = fake_client
    assert e.dim == 0

    result = await e.embed(["hello"])
    assert len(result[0]) == 1024
    assert e.dim == 1024


# ---------------------------------------------------------------------------
# embed() — error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_embed_raises_on_dim_mismatch() -> None:
    """Wrong-dim response → EmbedderError before vectors leak downstream."""
    fake_vec = [0.0] * 999  # not 1536
    fake_client = _fake_openai_client([fake_vec])

    e = OpenAIEmbedder(api_key="sk-test")
    e._client = fake_client

    with pytest.raises(EmbedderError, match="999-dim vector"):
        await e.embed(["text"])


@pytest.mark.asyncio
async def test_openai_embed_raises_on_sdk_failure() -> None:
    """SDK exceptions are wrapped in EmbedderError with type+msg context."""
    fake_client = MagicMock()
    fake_client.embeddings.create.side_effect = RuntimeError("rate limited")

    e = OpenAIEmbedder(api_key="sk-test")
    e._client = fake_client

    with pytest.raises(EmbedderError, match="rate limited"):
        await e.embed(["text"])


@pytest.mark.asyncio
async def test_openai_embed_empty_input_short_circuits() -> None:
    """Empty input → [] with no SDK call."""
    fake_client = MagicMock()
    e = OpenAIEmbedder(api_key="sk-test")
    e._client = fake_client
    assert await e.embed([]) == []
    fake_client.embeddings.create.assert_not_called()


@pytest.mark.asyncio
async def test_openai_embed_raises_when_openai_pkg_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing openai dep → EmbedderError pointing at the install command."""
    e = OpenAIEmbedder(api_key="sk-test")
    # Force the lazy `from openai import OpenAI` to fail.
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(EmbedderError, match="'openai' package"):
        await e.embed(["hello"])
