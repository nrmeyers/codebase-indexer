"""Tests for ``GET /github/orgs`` — the org-listing endpoint that powers
the Settings allowlist editor.

The endpoint must:
  * Reject requests when no PAT is configured (401).
  * Surface the user's orgs sorted by login.
  * Mark each org with an ``allowlisted`` flag matching the current
    ``GITHUB_ALLOWED_OWNERS`` setting (case-insensitive).
  * Echo the parsed allowlist back so the UI can cross-check.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers import github as github_router

client = TestClient(app)


@pytest.fixture
def gh_orgs_response() -> list[dict]:
    """A minimal but realistic ``/user/orgs`` payload from GitHub."""
    return [
        {
            "login": "navistone",
            "description": "Navistone Engineering",
            "avatar_url": "https://avatars.githubusercontent.com/u/1?v=4",
        },
        {
            "login": "anthropic",
            "description": "Anthropic",
            "avatar_url": "https://avatars.githubusercontent.com/u/2?v=4",
        },
        {
            # Missing description/avatar — must still surface.
            "login": "side-org",
        },
    ]


def test_orgs_returns_401_without_token() -> None:
    """No token = no upstream call = 401."""
    with patch("app.routers.github._github_token", return_value=None):
        resp = client.get("/github/orgs")
    assert resp.status_code == 401
    assert "GITHUB_TOKEN not set" in resp.json()["detail"]


def test_orgs_returns_sorted_list_with_allowlist_flag(
    gh_orgs_response: list[dict],
) -> None:
    """Orgs come back alphabetised; allowlisted flag tracks settings."""
    with patch("app.routers.github._github_token", return_value="ghp_fake"), \
         patch.object(github_router.settings, "GITHUB_ALLOWED_OWNERS", "navistone"), \
         patch(
             "app.routers.github._gh_get",
             new=AsyncMock(return_value=gh_orgs_response),
         ):
        resp = client.get("/github/orgs")

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 3
    assert body["allowlist"] == ["navistone"]
    # Alphabetised: anthropic, navistone, side-org
    logins = [o["login"] for o in body["orgs"]]
    assert logins == ["anthropic", "navistone", "side-org"]
    flags = {o["login"]: o["allowlisted"] for o in body["orgs"]}
    assert flags == {"anthropic": False, "navistone": True, "side-org": False}


def test_orgs_allowlist_match_is_case_insensitive(
    gh_orgs_response: list[dict],
) -> None:
    """``Navistone`` (mixed case in env) still flags ``navistone`` org."""
    with patch("app.routers.github._github_token", return_value="ghp_fake"), \
         patch.object(github_router.settings, "GITHUB_ALLOWED_OWNERS", "Navistone"), \
         patch(
             "app.routers.github._gh_get",
             new=AsyncMock(return_value=gh_orgs_response),
         ):
        resp = client.get("/github/orgs")

    flags = {o["login"]: o["allowlisted"] for o in resp.json()["orgs"]}
    assert flags["navistone"] is True


def test_orgs_skips_records_missing_login() -> None:
    """Defensive: GitHub edge case where an entry has no ``login`` field."""
    bogus = [{"description": "no login"}, {"login": "valid"}]
    with patch("app.routers.github._github_token", return_value="ghp_fake"), \
         patch.object(github_router.settings, "GITHUB_ALLOWED_OWNERS", ""), \
         patch("app.routers.github._gh_get", new=AsyncMock(return_value=bogus)):
        resp = client.get("/github/orgs")

    body = resp.json()
    assert body["total"] == 1
    assert body["orgs"][0]["login"] == "valid"


def test_orgs_empty_response_is_ok() -> None:
    """User belongs to zero orgs — the endpoint must still return 200."""
    with patch("app.routers.github._github_token", return_value="ghp_fake"), \
         patch.object(github_router.settings, "GITHUB_ALLOWED_OWNERS", "navistone"), \
         patch("app.routers.github._gh_get", new=AsyncMock(return_value=[])):
        resp = client.get("/github/orgs")

    assert resp.status_code == 200
    assert resp.json() == {"orgs": [], "total": 0, "allowlist": ["navistone"]}
