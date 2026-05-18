"""Factory dispatch tests for ``app.embedders.get_embedder``.

Per-backend behaviour now lives in ``tests/test_embedder_{local,sagemaker,
tei,openai}.py`` — this module only verifies the BUC-1605 + BYO factory
contract:

    * Default is ``local`` (no env vars set).
    * Each backend name routes to the right concrete class.
    * Unknown values fall back to ``local`` with a warning.
    * The factory is an lru_cache singleton.
    * Module-level surface (``EMBEDDING_DIM``, ``VALID_BACKENDS``) is
      stable.
"""
from __future__ import annotations

import logging
import os
from unittest.mock import patch

import pytest

from app import embedders
from app.embedders import EMBEDDING_DIM, get_embedder
from app.embedders.local import LocalEmbedder
from app.embedders.openai import OpenAIEmbedder
from app.embedders.sagemaker import SageMakerEmbedder
from app.embedders.tei import TEIEmbedder


@pytest.fixture(autouse=True)
def _clear_factory_cache() -> None:
    """Reset the lru_cache and strip selector env vars per test."""
    get_embedder.cache_clear()
    for key in (
        "EMBEDDER_BACKEND",
        "SAGEMAKER_ENDPOINT_NAME",
        "SAGEMAKER_EMBED_ENDPOINT",
        "SAGEMAKER_EMBED_URL",
        "SAGEMAKER_EMBED_REGION",
        "TEI_URL",
        "TEI_TIMEOUT_MS",
        "OPENAI_API_KEY",
        "OPENAI_EMBED_MODEL",
        "OPENAI_EMBED_DIM",
        "OPENAI_BASE_URL",
    ):
        os.environ.pop(key, None)
    yield
    get_embedder.cache_clear()


def test_factory_returns_local_by_default() -> None:
    backend = get_embedder()
    assert isinstance(backend, LocalEmbedder)
    assert backend.name == "local"


def test_factory_returns_sagemaker_when_selected() -> None:
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
    assert backend.endpoint_name == "forge-e5-embed-v2"


def test_factory_returns_tei_when_selected() -> None:
    with patch.dict(
        os.environ,
        {"EMBEDDER_BACKEND": "tei", "TEI_URL": "http://tei-sidecar:8080"},
    ):
        get_embedder.cache_clear()
        backend = get_embedder()
    assert isinstance(backend, TEIEmbedder)
    assert backend.base_url == "http://tei-sidecar:8080"


def test_factory_returns_openai_when_selected() -> None:
    """BYO embedder path — EMBEDDER_BACKEND=openai + key → OpenAIEmbedder."""
    with patch.dict(
        os.environ,
        {
            "EMBEDDER_BACKEND": "openai",
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_EMBED_MODEL": "text-embedding-3-small",
        },
    ):
        get_embedder.cache_clear()
        backend = get_embedder()
    assert isinstance(backend, OpenAIEmbedder)
    assert backend.name == "openai"
    assert backend.dim == 1536


def test_factory_falls_back_to_local_for_unknown_backend(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with patch.dict(os.environ, {"EMBEDDER_BACKEND": "totally-fake"}):
        get_embedder.cache_clear()
        with caplog.at_level(logging.WARNING, logger="app.embedders"):
            backend = get_embedder()
    assert isinstance(backend, LocalEmbedder)
    assert any(
        "EMBEDDER_BACKEND=" in r.message and "not recognised" in r.message
        for r in caplog.records
    )


def test_factory_caches_singleton() -> None:
    a = get_embedder()
    b = get_embedder()
    assert a is b


def test_module_exports() -> None:
    """Public API surface — guard against accidental removals."""
    assert hasattr(embedders, "get_embedder")
    assert hasattr(embedders, "EmbedderBackend")
    assert hasattr(embedders, "EmbedderError")
    assert embedders.EMBEDDING_DIM == 768
    assert set(embedders.VALID_BACKENDS) == {
        "local",
        "sagemaker",
        "tei",
        "openai",
    }
