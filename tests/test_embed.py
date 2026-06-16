"""Tests for the POST /embed endpoint (BUC-1592).

Surgical coverage:

    1. Happy path: 200 + 768-dim float vector + ``model`` mirrors backend.model.
    2. Empty text: 422 (Pydantic min_length).
    3. Oversized text: 422 (Pydantic max_length).
    4. Embedder outage (backend raises): 503.
    5. Embedder returns empty vector (cold-start failure swallowed inside
       the backend): 503.
    6. Embedder not configured (factory raises EmbedderError): 503.
    7. ``model`` response field is dynamic w.r.t. the configured backend.
    8. Local-backend query prefix wiring (``apply_prefix`` role=query).

Tests stub out the embedder factory so no real SageMaker / AWS call is
made — the endpoint is a thin wrapper around the BUC-1605 embedder
factory (post-shim migration).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.embedders.base import EmbedderError
from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixture: a stand-in async embedder backend. Only ``embed()`` is exercised
# by the route; we don't need to mimic the full protocol surface.
# ---------------------------------------------------------------------------


class _FakeBackend:
    """Async test double that returns a deterministic 768-dim vector.

    Implements the :class:`app.embedders.base.EmbedderBackend` protocol
    surface that the route actually touches: ``name``, ``model``, and
    ``async embed(texts) -> list[list[float]]``.
    """

    name = "fake"
    model = "nomic-ai/nomic-embed-text-v1.5"

    def __init__(
        self,
        vec: list[float] | None = None,
        *,
        raise_exc: bool = False,
        empty: bool = False,
        name: str = "fake",
        model: str = "nomic-ai/nomic-embed-text-v1.5",
    ) -> None:
        self._vec = vec if vec is not None else [0.1] * 768
        self._raise_exc = raise_exc
        self._empty = empty
        self.name = name
        self.model = model
        self.recorded_texts: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.recorded_texts.append(list(texts))
        if self._raise_exc:
            raise EmbedderError("simulated backend outage")
        if self._empty:
            # Match the backend contract: one entry per input. An "empty
            # vector" failure is represented by a [] entry, which the
            # route translates to 503.
            return [[] for _ in texts]
        return [self._vec for _ in texts]


# ---------------------------------------------------------------------------
# 1. Happy path.
# ---------------------------------------------------------------------------


def test_should_return_768_dim_vector_when_backend_succeeds() -> None:
    """POST /embed with a normal short string returns 200 and a 768-dim
    float vector tagged with the backend's current ``model`` id.
    """
    fake = _FakeBackend(vec=[0.5] * 768)
    with patch("app.routers.embed.get_embedder", return_value=fake):
        resp = client.post("/embed", json={"text": "hello world"})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body["embedding"], list)
    assert len(body["embedding"]) == 768
    assert all(isinstance(x, float) for x in body["embedding"])
    assert body["dims"] == 768
    assert body["model"] == "nomic-ai/nomic-embed-text-v1.5"


# ---------------------------------------------------------------------------
# 2. Empty text — Pydantic min_length validation.
# ---------------------------------------------------------------------------


def test_should_reject_empty_text_with_422() -> None:
    """An empty string violates ``min_length=1`` and must yield a 422
    before any embedder call is attempted.
    """
    resp = client.post("/embed", json={"text": ""})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 3. Oversized text — Pydantic max_length validation.
# ---------------------------------------------------------------------------


def test_should_reject_oversized_text_with_422() -> None:
    """Strings >4000 chars must be rejected by validation. This keeps
    request bodies bounded and prevents a misbehaving caller from
    blasting the embedder with multi-MB payloads.
    """
    too_long = "x" * 4001
    resp = client.post("/embed", json={"text": too_long})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 4. Backend outage — embed() raises EmbedderError.
# ---------------------------------------------------------------------------


def test_should_return_503_when_backend_raises() -> None:
    """Any exception out of the backend's ``embed`` must surface as a
    503 so the orchestrator can fail-open. We never want a backend
    transient turning into a user-visible 500.
    """
    fake = _FakeBackend(raise_exc=True)
    with patch("app.routers.embed.get_embedder", return_value=fake):
        resp = client.post("/embed", json={"text": "hello"})

    assert resp.status_code == 503
    assert "embed failed" in resp.json().get("detail", "")


# ---------------------------------------------------------------------------
# 5. Backend returns an empty vector (transient miss, no exception).
# ---------------------------------------------------------------------------


def test_should_return_503_when_backend_returns_empty_vector() -> None:
    """When the backend returns an entry with no floats (transient
    upstream failure swallowed inside the backend) the route must
    translate that into a 503 — never a 200 with an empty vector.
    """
    fake = _FakeBackend(empty=True)
    with patch("app.routers.embed.get_embedder", return_value=fake):
        resp = client.post("/embed", json={"text": "hello"})

    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# 6. Backend not configured — factory raises EmbedderError.
# ---------------------------------------------------------------------------


def test_should_return_503_when_backend_not_configured() -> None:
    """When ``get_embedder()`` raises ``EmbedderError`` (no backend
    configured, e.g. ``EMBEDDER_BACKEND=sagemaker`` with no endpoint set)
    the route must 503 with a descriptive message rather than 500.
    """
    err = EmbedderError("no endpoint configured")
    with patch("app.routers.embed.get_embedder", side_effect=err):
        resp = client.post("/embed", json={"text": "hello"})

    assert resp.status_code == 503
    detail = resp.json().get("detail", "")
    assert "no endpoint configured" in detail


# ---------------------------------------------------------------------------
# 7. Missing ``text`` field — Pydantic required-field validation.
# ---------------------------------------------------------------------------


def test_should_reject_missing_text_field_with_422() -> None:
    """The ``text`` field is required by the Pydantic model. A POST that
    omits it must yield 422 before any backend call is made.
    """
    resp = client.post("/embed", json={})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 8. Dynamic ``model`` field — mirrors backend.model, not a hardcoded const.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fake_model",
    [
        "nomic-ai/nomic-embed-text-v1.5",
        "jina-code-v2-serverless",
    ],
)
def test_should_reflect_backend_model_in_response(fake_model: str) -> None:
    """``body['model']`` mirrors whatever the configured backend reports."""
    fake = _FakeBackend(model=fake_model)
    with patch("app.routers.embed.get_embedder", return_value=fake):
        resp = client.post("/embed", json={"text": "hello"})

    assert resp.status_code == 200, resp.text
    assert resp.json()["model"] == fake_model


# ---------------------------------------------------------------------------
# 9. Query-prefix wiring — local + e5 model -> "query: " prefix is applied.
# ---------------------------------------------------------------------------


def test_should_apply_query_prefix_for_local_e5_backend() -> None:
    """``apply_prefix`` runs before backend.embed for the local backend."""
    fake = _FakeBackend(name="local", model="intfloat/e5-base-v2")
    with patch("app.routers.embed.get_embedder", return_value=fake):
        resp = client.post("/embed", json={"text": "hello"})

    assert resp.status_code == 200, resp.text
    assert fake.recorded_texts == [["query: hello"]]
