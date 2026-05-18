"""Unit tests for :class:`app.embedders.local.LocalEmbedder`.

Split out from ``tests/test_embedders.py`` as part of the BYO-embedder
config pass so each backend's contract sits in its own file. The factory
dispatch tests remain in ``test_embedders.py``.

The "produces 768-dim vector" test pulls a real sentence-transformers
model from the HuggingFace cache. It is skipped when the optional
``[local-embed]`` extra isn't installed — keeps CI green on slim
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


def test_local_embedder_exposes_dim() -> None:
    """dim attribute defaults to EMBEDDING_DIM (e5-base-v2's 768)."""
    backend = LocalEmbedder()
    assert backend.dim == EMBEDDING_DIM
    assert backend.name == "local"


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
