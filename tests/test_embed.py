"""Tests for the POST /embed endpoint (BUC-1592).

Surgical coverage:

    1. Happy path: 200 + 768-dim float vector + ``model="e5-base-v2"``.
    2. Empty text: 422 (Pydantic min_length).
    3. Oversized text: 422 (Pydantic max_length).
    4. SageMaker outage (backend raises): 503.
    5. SageMaker returns ``None`` (cold-start failure swallowed inside the
       embedder): 503.
    6. SageMaker not configured (singleton returns ``None``): 503.

Tests stub out ``get_sagemaker_embedder`` so no real SageMaker / AWS call
is made — the endpoint is a thin wrapper around the existing helper.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixture: a stand-in SageMakerEmbedder. Only ``embed()`` is exercised by
# the route; we don't need to mimic the full upstream surface.
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Test double that returns a deterministic 768-dim vector."""

    def __init__(self, vec: list[float] | None = None, *, raise_exc: bool = False) -> None:
        self._vec = vec if vec is not None else [0.1] * 768
        self._raise_exc = raise_exc

    def embed(self, text: str) -> list[float] | None:  # noqa: ARG002
        if self._raise_exc:
            raise RuntimeError("simulated SageMaker outage")
        return self._vec


# ---------------------------------------------------------------------------
# 1. Happy path.
# ---------------------------------------------------------------------------


def test_should_return_768_dim_vector_when_sagemaker_succeeds() -> None:
    """POST /embed with a normal short string returns 200 and a 768-dim
    float vector tagged with ``model="e5-base-v2"``.
    """
    fake = _FakeEmbedder(vec=[0.5] * 768)
    with patch("app.routers.embed.get_sagemaker_embedder", return_value=fake):
        resp = client.post("/embed", json={"text": "hello world"})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body["embedding"], list)
    assert len(body["embedding"]) == 768
    assert all(isinstance(x, float) for x in body["embedding"])
    assert body["dims"] == 768
    assert body["model"] == "e5-base-v2"


# ---------------------------------------------------------------------------
# 2. Empty text — Pydantic min_length validation.
# ---------------------------------------------------------------------------


def test_should_reject_empty_text_with_422() -> None:
    """An empty string violates ``min_length=1`` and must yield a 422
    before any SageMaker call is attempted.
    """
    resp = client.post("/embed", json={"text": ""})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 3. Oversized text — Pydantic max_length validation.
# ---------------------------------------------------------------------------


def test_should_reject_oversized_text_with_422() -> None:
    """Strings >4000 chars must be rejected by validation. This keeps
    request bodies bounded and prevents a misbehaving caller from
    blasting the SageMaker endpoint with multi-MB payloads.
    """
    too_long = "x" * 4001
    resp = client.post("/embed", json={"text": too_long})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 4. SageMaker outage — backend raises an exception.
# ---------------------------------------------------------------------------


def test_should_return_503_when_sagemaker_raises() -> None:
    """Any exception out of ``SageMakerEmbedder.embed`` must surface as a
    503 so the orchestrator can fail-open. We never want a backend
    transient turning into a user-visible 500.
    """
    fake = _FakeEmbedder(raise_exc=True)
    with patch("app.routers.embed.get_sagemaker_embedder", return_value=fake):
        resp = client.post("/embed", json={"text": "hello"})

    assert resp.status_code == 503
    assert "embed failed" in resp.json().get("detail", "")


# ---------------------------------------------------------------------------
# 5. SageMaker returns None (transient miss, no exception).
# ---------------------------------------------------------------------------


def test_should_return_503_when_sagemaker_returns_none() -> None:
    """The upstream embedder swallows certain transient failures (cold-
    start timeouts, etc.) and returns ``None``. The route must translate
    that into a 503 — never a 200 with an empty vector.
    """
    fake = _FakeEmbedder(vec=None)
    # Wire the fake to return None directly without raising.
    with patch.object(fake, "embed", return_value=None):
        with patch("app.routers.embed.get_sagemaker_embedder", return_value=fake):
            resp = client.post("/embed", json={"text": "hello"})

    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# 6. SageMaker not configured — singleton returns None.
# ---------------------------------------------------------------------------


def test_should_return_503_when_sagemaker_not_configured() -> None:
    """When ``get_sagemaker_embedder()`` itself returns ``None`` (no env
    vars set) the route must 503 with a descriptive message rather than
    raising AttributeError on ``.embed()``.
    """
    with patch("app.routers.embed.get_sagemaker_embedder", return_value=None):
        resp = client.post("/embed", json={"text": "hello"})

    assert resp.status_code == 503
    detail = resp.json().get("detail", "")
    assert "not configured" in detail


# ---------------------------------------------------------------------------
# 7. Missing ``text`` field — Pydantic required-field validation.
# ---------------------------------------------------------------------------


def test_should_reject_missing_text_field_with_422() -> None:
    """The ``text`` field is required by the Pydantic model. A POST that
    omits it must yield 422 before any SageMaker call is made.
    """
    resp = client.post("/embed", json={})
    assert resp.status_code == 422
