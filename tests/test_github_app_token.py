"""Tests for GitHub App installation token support on the ``/github/*`` routes.

GitHub App installation tokens (``ghs_*``) have a 1-hour TTL and authenticate
as the App's installation rather than as a user.  They CANNOT call
``/user/repos`` — GitHub returns 403 "Resource not accessible by
integration".  The indexer must detect the credential family by prefix and
route to ``/installation/repositories`` instead.

These tests lock down:

* ``_token_type`` prefix detection for every documented family.
* ``GET /github/repos`` calling the right endpoint per token family.
* ``GET /github/status`` probing the right endpoint and surfacing
  ``token_type`` so the UI can render the correct hint.
* The 401 recovery hint changing per token family.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers import github as github_router

client = TestClient(app)


# ---------------------------------------------------------------------------
# _token_type — prefix classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token, expected",
    [
        ("ghp_abcdef0123456789", "pat"),
        ("gho_abcdef0123456789", "pat"),
        ("github_pat_11ABCDEFG_xyz", "pat"),
        ("ghs_installationToken123", "github_app"),
        ("just-some-string", "unknown"),
        ("", "none"),
        (None, "none"),
    ],
)
def test_token_type_classifies_each_prefix(token: str | None, expected: str) -> None:
    """Every documented GitHub credential prefix maps to a stable family."""
    assert github_router._token_type(token) == expected


# ---------------------------------------------------------------------------
# GET /github/repos — endpoint routing by token type
# ---------------------------------------------------------------------------


@pytest.fixture
def repo_payload() -> list[dict[str, Any]]:
    """A minimal but realistic repo record (shape shared by both endpoints)."""
    return [
        {
            "full_name": "navistone/legacy-api",
            "name": "legacy-api",
            "owner": {"login": "navistone"},
            "private": True,
            "default_branch": "main",
            "clone_url": "https://github.com/navistone/legacy-api.git",
            "ssh_url": "git@github.com:navistone/legacy-api.git",
            "stargazers_count": 3,
            "updated_at": "2026-05-08T00:00:00Z",
            "description": "Legacy API",
        },
    ]


def test_repos_uses_installation_endpoint_for_app_token(
    repo_payload: list[dict[str, Any]],
) -> None:
    """``ghs_*`` token routes to ``/installation/repositories``."""
    captured: dict[str, Any] = {}

    async def fake_gh_get(client: Any, path: str, params: Any = None) -> Any:
        captured["path"] = path
        captured["params"] = params
        # /installation/repositories returns an envelope, not a bare list.
        return {"total_count": 1, "repositories": repo_payload}

    with patch(
        "app.routers.github._github_token",
        return_value="ghs_installationToken123",
    ), patch.object(github_router.settings, "GITHUB_ALLOWED_OWNERS", ""), patch(
        "app.routers.github._gh_get", new=AsyncMock(side_effect=fake_gh_get)
    ):
        resp = client.get("/github/repos")

    assert resp.status_code == 200
    assert captured["path"] == "/installation/repositories"
    body = resp.json()
    assert body["total"] == 1
    assert body["repos"][0]["full_name"] == "navistone/legacy-api"


def test_repos_uses_user_repos_endpoint_for_pat(
    repo_payload: list[dict[str, Any]],
) -> None:
    """Classic / fine-grained PAT keeps using ``/user/repos``."""
    captured: dict[str, Any] = {}

    async def fake_gh_get(client: Any, path: str, params: Any = None) -> Any:
        captured["path"] = path
        captured["params"] = params
        return repo_payload

    with patch(
        "app.routers.github._github_token", return_value="ghp_classicPersonalToken"
    ), patch.object(github_router.settings, "GITHUB_ALLOWED_OWNERS", ""), patch(
        "app.routers.github._gh_get", new=AsyncMock(side_effect=fake_gh_get)
    ):
        resp = client.get("/github/repos")

    assert resp.status_code == 200
    assert captured["path"] == "/user/repos"
    # Sanity: PAT call also passes the affiliation filter.
    assert captured["params"]["affiliation"].startswith("owner")


def test_repos_app_token_handles_dict_envelope_with_no_repositories_key() -> None:
    """Defensive: empty installation returns an envelope with empty list."""
    async def fake_gh_get(client: Any, path: str, params: Any = None) -> Any:
        return {"total_count": 0, "repositories": []}

    with patch(
        "app.routers.github._github_token", return_value="ghs_emptyInstall"
    ), patch.object(github_router.settings, "GITHUB_ALLOWED_OWNERS", ""), patch(
        "app.routers.github._gh_get", new=AsyncMock(side_effect=fake_gh_get)
    ):
        resp = client.get("/github/repos")

    assert resp.status_code == 200
    assert resp.json() == {"repos": [], "total": 0}


# ---------------------------------------------------------------------------
# GET /github/status — token_type surfacing + correct probe endpoint
# ---------------------------------------------------------------------------


def _make_response(status_code: int, json_body: Any = None, headers: dict | None = None):
    """Tiny stub matching the subset of httpx.Response the router consumes."""
    class _R:
        def __init__(self) -> None:
            self.status_code = status_code
            self._json = json_body
            self.headers = headers or {}

        def json(self) -> Any:
            return self._json

    return _R()


def test_status_app_token_probes_installation_repositories() -> None:
    """``ghs_*`` token causes status to probe /installation/repositories."""
    calls: list[str] = []

    async def fake_get(url: str, headers: dict, params: Any = None) -> Any:
        calls.append(url)
        if url.endswith("/installation/repositories"):
            return _make_response(200, {"total_count": 0, "repositories": []})
        if url.endswith("/rate_limit"):
            return _make_response(
                200,
                {"resources": {"core": {"limit": 5000, "remaining": 4999, "reset": 1}}},
            )
        return _make_response(404)

    class _Client:
        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def get(self, url: str, headers: dict, params: Any = None):
            return await fake_get(url, headers, params)

    with patch(
        "app.routers.github._github_token", return_value="ghs_installationToken123"
    ), patch("app.routers.github.httpx.AsyncClient", return_value=_Client()):
        resp = client.get("/github/status")

    assert resp.status_code == 200
    body = resp.json()
    assert body["connected"] is True
    assert body["token_type"] == "github_app"
    # User identity is intentionally None for App tokens.
    assert body["user"] is None
    # The status endpoint must NOT have called /user for an App token.
    assert any(c.endswith("/installation/repositories") for c in calls)
    assert not any(c.endswith("/user") for c in calls)


def test_status_pat_token_probes_user_endpoint() -> None:
    """PAT keeps the legacy ``/user`` probe and surfaces the login + scopes."""
    calls: list[str] = []

    async def fake_get(url: str, headers: dict, params: Any = None) -> Any:
        calls.append(url)
        if url.endswith("/user"):
            return _make_response(
                200,
                {"login": "octocat"},
                headers={"X-OAuth-Scopes": "repo, read:org"},
            )
        if url.endswith("/rate_limit"):
            return _make_response(
                200,
                {"resources": {"core": {"limit": 5000, "remaining": 4998, "reset": 1}}},
            )
        return _make_response(404)

    class _Client:
        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def get(self, url: str, headers: dict, params: Any = None):
            return await fake_get(url, headers, params)

    with patch(
        "app.routers.github._github_token", return_value="ghp_classicPersonalToken"
    ), patch("app.routers.github.httpx.AsyncClient", return_value=_Client()):
        resp = client.get("/github/status")

    body = resp.json()
    assert body["connected"] is True
    assert body["token_type"] == "pat"
    assert body["user"] == "octocat"
    assert "repo" in body["scopes"]
    assert any(c.endswith("/user") for c in calls)


def test_status_no_token_reports_none() -> None:
    """No token configured surfaces ``token_type='none'``."""
    with patch("app.routers.github._github_token", return_value=None):
        resp = client.get("/github/status")
    body = resp.json()
    assert body["connected"] is False
    assert body["token_type"] == "none"


def test_status_app_token_403_uses_app_specific_hint() -> None:
    """When App token is rejected, the recovery hint mentions installation TTL."""
    async def fake_get(url: str, headers: dict, params: Any = None) -> Any:
        # GitHub returns 403 for App tokens whose installation has been revoked
        # or that hit an endpoint the App doesn't have permission for.
        return _make_response(403, {"message": "Resource not accessible by integration"})

    class _Client:
        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def get(self, url: str, headers: dict, params: Any = None):
            return await fake_get(url, headers, params)

    with patch(
        "app.routers.github._github_token", return_value="ghs_revokedToken"
    ), patch("app.routers.github.httpx.AsyncClient", return_value=_Client()):
        resp = client.get("/github/status")

    body = resp.json()
    assert body["connected"] is False
    assert body["token_type"] == "github_app"
    assert "App installation token" in body["message"]
