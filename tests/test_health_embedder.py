"""Tests for the embedder availability surface in ``GET /health``.

Covers three operator-visible scenarios:

1. The configured backend constructs cleanly → ``available: true``.
2. The configured backend raises ``ModuleNotFoundError`` (the optional
   ``sentence-transformers`` package is missing) → ``available: false``,
   ``last_error`` is populated, the service still boots.
3. LM Studio is configured with an embed model loaded → ``fallback_lm_studio:
   true``, independent of the primary backend's state.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.embedders import availability as embedder_availability
from app.embedders.base import EMBEDDING_DIM, EmbedderError
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_availability_cache() -> None:
    """Reset the module-level probe cache between tests."""
    embedder_availability.reset_for_tests()
    yield
    embedder_availability.reset_for_tests()


def test_health_embedder_block_present_with_defaults() -> None:
    """The ``embedder`` block is always present with the documented keys."""
    with patch("app.routers.health._get_indexed_repos", return_value=[]):
        resp = client.get("/health")
    assert resp.status_code == 200
    block = resp.json()["embedder"]
    for key in (
        "backend",
        "model",
        "dim",
        "configured",
        "error",
        "available",
        "last_error",
        "fallback_lm_studio",
        "last_check_at",
        "check_latency_ms",
    ):
        assert key in block, f"missing key {key!r} in embedder block"


def test_health_embedder_available_when_factory_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clean ``get_embedder()`` → ``available: true`` + dim populated."""

    class _FakeBackend:
        name = "local"
        model = "intfloat/e5-base-v2"

        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * EMBEDDING_DIM for _ in texts]

    monkeypatch.setenv("EMBEDDER_BACKEND", "local")
    monkeypatch.setattr(
        "app.embedders.availability.get_embedder",
        lambda: _FakeBackend(),
    )
    # Skip the real dependency probe — the test backend is synthetic.
    monkeypatch.setattr(
        "app.embedders.availability._validate_backend_dependency",
        lambda _name: None,
    )
    monkeypatch.setattr(
        "app.embedders.availability._probe_lm_studio_fallback",
        lambda: False,
    )

    embedder_availability.probe_embedder()

    with patch("app.routers.health._get_indexed_repos", return_value=[]):
        resp = client.get("/health")
    block = resp.json()["embedder"]
    assert block["backend"] == "local"
    assert block["available"] is True
    assert block["dim"] == EMBEDDING_DIM
    assert block["last_error"] is None
    assert block["fallback_lm_studio"] is False
    # Probe latency is recorded as a non-negative float.
    assert isinstance(block["check_latency_ms"], (int, float))
    assert block["check_latency_ms"] >= 0
    # Timestamp is an ISO 8601 string with a TZ offset.
    assert isinstance(block["last_check_at"], str)
    assert block["last_check_at"].endswith("+00:00")


def test_health_embedder_unavailable_when_module_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ModuleNotFoundError`` from sentence-transformers → service still boots.

    Mirrors the exact failure mode on a dev box where ``EMBEDDER_BACKEND=local``
    is set but the ``sentence-transformers`` package is not installed.
    The probe MUST capture the error, the service MUST stay alive, and
    ``available`` MUST be ``false``.
    """
    monkeypatch.setenv("EMBEDDER_BACKEND", "local")

    def _explode() -> object:
        raise EmbedderError(
            "EMBEDDER_BACKEND=local requires the 'sentence-transformers' package."
        ) from ModuleNotFoundError("No module named 'sentence_transformers'")

    monkeypatch.setattr("app.embedders.availability.get_embedder", _explode)
    monkeypatch.setattr(
        "app.embedders.availability._probe_lm_studio_fallback",
        lambda: False,
    )

    status = embedder_availability.probe_embedder()

    # Service-level assertion: probe call did not raise.
    assert status["available"] is False
    assert status["configured"] is False
    assert status["dim"] == 0
    assert "sentence_transformers" in (status["last_error"] or "")

    with patch("app.routers.health._get_indexed_repos", return_value=[]):
        resp = client.get("/health")
    assert resp.status_code == 200
    block = resp.json()["embedder"]
    assert block["available"] is False
    assert block["configured"] is False
    assert block["dim"] == 0
    assert block["fallback_lm_studio"] is False
    # The cause-chain walker should surface the root ModuleNotFoundError
    # rather than just the wrapper EmbedderError.
    assert "ModuleNotFoundError" in (block["last_error"] or "")
    assert "sentence_transformers" in (block["last_error"] or "")


def test_health_embedder_reports_lm_studio_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LM Studio with an embed model loaded → ``fallback_lm_studio: true``."""
    monkeypatch.setenv("EMBEDDER_BACKEND", "local")

    def _explode() -> object:
        raise EmbedderError("backend unavailable in this test")

    monkeypatch.setattr("app.embedders.availability.get_embedder", _explode)
    # Simulate LM Studio reachable + embed model loaded.
    monkeypatch.setattr(
        "app.services.lm_studio.base_url",
        lambda: "http://localhost:1234/v1",
    )
    monkeypatch.setattr("app.services.lm_studio.can_embed", lambda: True)

    embedder_availability.probe_embedder()

    with patch("app.routers.health._get_indexed_repos", return_value=[]):
        resp = client.get("/health")
    block = resp.json()["embedder"]
    assert block["available"] is False
    assert block["fallback_lm_studio"] is True


def test_emit_startup_warning_silent_when_available() -> None:
    """No banner when the primary backend is available."""
    import io
    import contextlib

    status = {
        "backend": "local",
        "available": True,
        "dim": EMBEDDING_DIM,
        "last_error": None,
        "fallback_lm_studio": False,
        "last_check_at": None,
        "check_latency_ms": 1.0,
    }
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        embedder_availability.emit_startup_warning(status)
    assert buf.getvalue() == ""


def test_emit_startup_warning_silent_when_lm_studio_fallback() -> None:
    """No banner when LM Studio fallback covers the gap."""
    import io
    import contextlib

    status = {
        "backend": "local",
        "available": False,
        "dim": 0,
        "last_error": "ModuleNotFoundError",
        "fallback_lm_studio": True,
        "last_check_at": None,
        "check_latency_ms": 1.0,
    }
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        embedder_availability.emit_startup_warning(status)
    assert buf.getvalue() == ""


def test_emit_startup_warning_loud_when_no_backend() -> None:
    """Banner is printed AND includes the operator fix when nothing is available."""
    import io
    import contextlib

    status = {
        "backend": "local",
        "available": False,
        "dim": 0,
        "last_error": "ModuleNotFoundError: No module named 'sentence_transformers'",
        "fallback_lm_studio": False,
        "last_check_at": None,
        "check_latency_ms": 0.1,
    }
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        embedder_availability.emit_startup_warning(status)
    out = buf.getvalue()
    assert "EMBEDDER UNAVAILABLE" in out
    assert "uv sync" in out
    assert "EMBEDDER_BACKEND=local" in out
    assert "sentence_transformers" in out
