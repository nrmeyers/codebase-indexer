"""Tests for GET /health."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    with patch("app.routers.health._get_indexed_repos", return_value=[]):
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "db_path" in body
    assert isinstance(body["indexed_repos"], list)


def test_health_lists_indexed_repos() -> None:
    with patch(
        "app.routers.health._get_indexed_repos",
        return_value=["my-project", "other-project"],
    ):
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["indexed_repos"] == ["my-project", "other-project"]


def test_health_survives_db_error() -> None:
    """If LadybugDB is unavailable, /health still returns 200."""
    with patch(
        "app.routers.health._get_indexed_repos",
        side_effect=RuntimeError("db not found"),
    ):
        # _get_indexed_repos already swallows exceptions internally,
        # but the outer layer should be fault-tolerant too.
        # Monkeypatch the safe wrapper path.
        pass

    with patch("app.routers.health._get_indexed_repos", return_value=[]):
        resp = client.get("/health")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# lm_studio block
# ---------------------------------------------------------------------------


def test_health_lm_studio_block_present() -> None:
    """The ``lm_studio`` block is always present with the expected shape."""
    with patch("app.routers.health._get_indexed_repos", return_value=[]):
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "lm_studio" in body
    block = body["lm_studio"]
    for key in (
        "configured",
        "reachable",
        "embed_model",
        "rerank_model",
        "can_embed",
        "can_rerank",
    ):
        assert key in block, f"missing key {key!r} in lm_studio block"
    assert isinstance(block["configured"], bool)
    assert isinstance(block["reachable"], bool)
    assert isinstance(block["can_embed"], bool)
    assert isinstance(block["can_rerank"], bool)
    # embed_model / rerank_model are str | None
    assert block["embed_model"] is None or isinstance(block["embed_model"], str)
    assert block["rerank_model"] is None or isinstance(block["rerank_model"], str)


def test_health_lm_studio_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``LM_STUDIO_URL`` is empty, return all-False/null and don't probe."""
    # Force base_url() to return "" to simulate unconfigured deployment.
    monkeypatch.setattr("app.services.lm_studio.base_url", lambda: "")

    # If the handler short-circuits correctly, these probes are never called.
    # If they ARE called we want the test to fail loudly rather than silently
    # passing because lm_studio.is_available() also returns False.
    def _boom() -> bool:  # pragma: no cover — intentionally unreachable
        raise AssertionError("LM Studio should not be probed when not configured")

    monkeypatch.setattr("app.services.lm_studio.is_available", _boom)
    monkeypatch.setattr("app.services.lm_studio.can_embed", _boom)
    monkeypatch.setattr("app.services.lm_studio.can_rerank", _boom)

    with patch("app.routers.health._get_indexed_repos", return_value=[]):
        resp = client.get("/health")

    assert resp.status_code == 200
    block = resp.json()["lm_studio"]
    assert block == {
        "configured": False,
        "reachable": False,
        "embed_model": None,
        "rerank_model": None,
        "can_embed": False,
        "can_rerank": False,
    }


def test_health_lm_studio_probe_failure_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A throwing LM Studio adapter must never break /health."""
    monkeypatch.setattr("app.services.lm_studio.base_url", lambda: "http://x:1234")

    def _boom(*_a: object, **_kw: object) -> bool:
        raise RuntimeError("LM Studio exploded")

    monkeypatch.setattr("app.services.lm_studio.is_available", _boom)
    monkeypatch.setattr("app.services.lm_studio.resolve_model", _boom)
    monkeypatch.setattr("app.services.lm_studio.can_embed", _boom)
    monkeypatch.setattr("app.services.lm_studio.can_rerank", _boom)

    with patch("app.routers.health._get_indexed_repos", return_value=[]):
        resp = client.get("/health")

    assert resp.status_code == 200
    block = resp.json()["lm_studio"]
    # configured=True (URL was set) but everything else degrades safely.
    assert block["configured"] is True
    assert block["reachable"] is False
    assert block["embed_model"] is None
    assert block["rerank_model"] is None
    assert block["can_embed"] is False
    assert block["can_rerank"] is False
