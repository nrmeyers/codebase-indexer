"""Unit tests for :class:`app.embedders.local.LocalEmbedder`.

Split out from ``tests/test_embedders.py`` as part of the BYO-embedder
config pass so each backend's contract sits in its own file. The factory
dispatch tests remain in ``test_embedders.py``.

The "produces 768-dim vector" test pulls a real sentence-transformers
model from the HuggingFace cache. It is skipped when the optional
``sentence-transformers`` package isn't installed — keeps CI green on slim
installs.
"""
from __future__ import annotations

import os

import pytest

from app.embedders import EMBEDDING_DIM, EmbedderError
from app.embedders.base import EmbedderBackend
from app.embedders.local import LocalEmbedder


def test_local_embedder_satisfies_protocol() -> None:
    assert isinstance(LocalEmbedder(), EmbedderBackend)


def test_local_embedder_exposes_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    """dim attribute defaults to EMBEDDING_DIM (nomic-v1.5's 768)."""
    monkeypatch.delenv("LOCAL_EMBED_MODEL", raising=False)
    monkeypatch.delenv("LOCAL_EMBED_DIM", raising=False)
    backend = LocalEmbedder()
    assert backend.dim == EMBEDDING_DIM
    assert backend.name == "local"
    assert backend.model == "nomic-ai/nomic-embed-text-v1.5"


def test_local_embedder_dim_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    """LOCAL_EMBED_DIM lets operators run a non-default model."""
    monkeypatch.setenv("LOCAL_EMBED_DIM", "1024")
    backend = LocalEmbedder()
    assert backend.dim == 1024


def test_local_embedder_dim_falls_back_on_garbage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-integer LOCAL_EMBED_DIM falls back silently to EMBEDDING_DIM."""
    monkeypatch.setenv("LOCAL_EMBED_DIM", "not-a-number")
    assert LocalEmbedder().dim == EMBEDDING_DIM


@pytest.mark.asyncio
async def test_local_embedder_produces_768_dim_vector() -> None:
    """Real sentence-transformers smoke — skipped when extra isn't installed."""
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
    assert backend._model is None


@pytest.mark.asyncio
async def test_local_embedder_missing_sentence_transformers_raises() -> None:
    """Clear EmbedderError when the optional extra is absent."""
    import sys

    backend = LocalEmbedder()
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


def test_local_embedder_auto_trusts_default_nomic_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default model is in AUTO_TRUST_REMOTE_CODE_MODELS -> trust_remote_code on."""
    monkeypatch.delenv("LOCAL_EMBED_MODEL", raising=False)
    monkeypatch.delenv("LOCAL_TRUST_REMOTE_CODE", raising=False)
    backend = LocalEmbedder()
    assert backend._trust_remote_code is True


def test_local_embedder_does_not_trust_arbitrary_model_without_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unvetted LOCAL_EMBED_MODEL stays opt-in: trust_remote_code stays False."""
    monkeypatch.setenv("LOCAL_EMBED_MODEL", "some/other-model")
    monkeypatch.delenv("LOCAL_TRUST_REMOTE_CODE", raising=False)
    backend = LocalEmbedder()
    assert backend._trust_remote_code is False


def test_local_embedder_env_opt_in_trusts_arbitrary_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LOCAL_TRUST_REMOTE_CODE=1 forces trust on for any model id."""
    monkeypatch.setenv("LOCAL_EMBED_MODEL", "some/other-model")
    monkeypatch.setenv("LOCAL_TRUST_REMOTE_CODE", "1")
    backend = LocalEmbedder()
    assert backend._trust_remote_code is True
