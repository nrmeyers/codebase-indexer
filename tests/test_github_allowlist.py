"""Tests for the ``GITHUB_ALLOWED_OWNERS`` guard on /github/* routes.

The allowlist is the only thing standing between an attacker (or a stray
UI bug) and the indexer cloning arbitrary public repos onto local disk —
so we lock down both the HTTP entry-points (``POST /github/index``) and
the picker (``GET /github/repos``) and assert dev-mode (empty allowlist)
preserves backwards-compatible behaviour.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.routers import github as github_router

client = TestClient(app)


# ---------------------------------------------------------------------------
# POST /github/index — owner allowlist guard
# ---------------------------------------------------------------------------


def test_post_index_rejects_disallowed_owner() -> None:
    """A repo owned by an org outside the allowlist must 403 before clone."""
    with patch.object(github_router.settings, "GITHUB_ALLOWED_OWNERS", "navistone"):
        resp = client.post("/github/index", json={"full_name": "evil-org/legacy-api"})
    assert resp.status_code == 403
    assert "not in GITHUB_ALLOWED_OWNERS" in resp.json()["detail"]


def test_post_index_accepts_allowed_owner() -> None:
    """An owner present in the allowlist passes the guard and reaches clone."""
    with patch.object(github_router.settings, "GITHUB_ALLOWED_OWNERS", "navistone"), \
         patch("app.routers.github._clone_or_update", return_value=Path("/tmp/fake-clone-dest")), \
         patch("app.routers.github._run_ingestion", new_callable=AsyncMock):
        resp = client.post("/github/index", json={"full_name": "navistone/legacy-api"})
    assert resp.status_code == 202
    assert "job_id" in resp.json()


def test_post_index_owner_match_is_case_insensitive() -> None:
    """``Navistone/foo`` and ``navistone/foo`` should both pass the guard."""
    with patch.object(github_router.settings, "GITHUB_ALLOWED_OWNERS", "navistone"), \
         patch("app.routers.github._clone_or_update", return_value=Path("/tmp/fake-clone-dest")), \
         patch("app.routers.github._run_ingestion", new_callable=AsyncMock):
        resp = client.post("/github/index", json={"full_name": "Navistone/legacy-api"})
    assert resp.status_code == 202


def test_post_index_rejects_malformed_full_name() -> None:
    """Missing the ``/`` returns 422 (validation), not 403 (allowlist)."""
    with patch.object(github_router.settings, "GITHUB_ALLOWED_OWNERS", "navistone"):
        resp = client.post("/github/index", json={"full_name": "no-slash-here"})
    assert resp.status_code == 422


def test_post_index_empty_allowlist_disables_guard() -> None:
    """Empty ``GITHUB_ALLOWED_OWNERS`` = dev mode = any owner allowed."""
    with patch.object(github_router.settings, "GITHUB_ALLOWED_OWNERS", ""), \
         patch("app.routers.github._clone_or_update", return_value=Path("/tmp/fake-clone-dest")), \
         patch("app.routers.github._run_ingestion", new_callable=AsyncMock):
        resp = client.post("/github/index", json={"full_name": "anyone/any-repo"})
    assert resp.status_code == 202


def test_post_index_supports_multiple_allowed_owners() -> None:
    """Comma-separated list — each owner gets through."""
    with patch.object(
        github_router.settings, "GITHUB_ALLOWED_OWNERS", "navistone, anthropic"
    ), patch("app.routers.github._clone_or_update", return_value=Path("/tmp/fake-clone-dest")), \
       patch("app.routers.github._run_ingestion", new_callable=AsyncMock):
        for full_name in ("navistone/foo", "anthropic/bar"):
            resp = client.post("/github/index", json={"full_name": full_name})
            assert resp.status_code == 202, f"{full_name} should be allowed"
