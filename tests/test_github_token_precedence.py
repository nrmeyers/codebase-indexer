"""Tests for the per-request ``github_token`` precedence rule on POST /github/index.

BUC-1591: TheForge sends a short-lived GitHub App installation token in the
request body (``req.github_token``) and the indexer MUST honour it instead
of falling back to its own ``GITHUB_TOKEN`` env var.  The env-var resolver
is reserved for the local-dev case where TheForge isn't in the loop.

This file locks down the precedence rule in both directions:

* Request-supplied token wins over the env-var fallback (the BUC-1578 contract).
* Env-var resolver fires only when the request body has no token.
* The audit log emitted by ``_clone_or_update`` correctly classifies the
  token source and never leaks the full secret (last-4 only).
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers import github as github_router

client = TestClient(app)


# ---------------------------------------------------------------------------
# Precedence — request-supplied token wins
# ---------------------------------------------------------------------------


def test_request_token_takes_precedence_over_env_var() -> None:
    """``req.github_token`` MUST be passed to ``_clone_or_update``, not the env PAT."""
    captured: dict[str, object] = {}

    def fake_clone(
        full_name: str,
        branch: str | None,
        token: str | None,
        token_source: str = "unknown",
    ) -> Path:
        captured["full_name"] = full_name
        captured["token"] = token
        captured["token_source"] = token_source
        return Path("/tmp/fake-clone-dest")

    with patch.object(github_router.settings, "GITHUB_ALLOWED_OWNERS", ""), patch(
        "app.routers.github._github_token", return_value="env-pat-FAKE"
    ), patch("app.routers.github._token_source", return_value="env"), patch(
        "app.routers.github._clone_or_update", side_effect=fake_clone
    ), patch(
        "app.routers.github._run_ingestion", new_callable=AsyncMock
    ):
        resp = client.post(
            "/github/index",
            json={
                "full_name": "navistone/legacy-api",
                "github_token": "req-app-FAKE",
            },
        )

    assert resp.status_code == 202
    assert captured["token"] == "req-app-FAKE", (
        "Request-body token must take precedence over the env-var PAT "
        "(BUC-1578 contract — TheForge mints a per-request App token)."
    )
    assert captured["token_source"] == "request"


def test_env_fallback_fires_when_request_token_absent() -> None:
    """No ``github_token`` in the body → indexer falls back to its env-var PAT."""
    captured: dict[str, object] = {}

    def fake_clone(
        full_name: str,
        branch: str | None,
        token: str | None,
        token_source: str = "unknown",
    ) -> Path:
        captured["token"] = token
        captured["token_source"] = token_source
        return Path("/tmp/fake-clone-dest")

    with patch.object(github_router.settings, "GITHUB_ALLOWED_OWNERS", ""), patch(
        "app.routers.github._github_token", return_value="env-pat-FAKE"
    ), patch("app.routers.github._token_source", return_value="env"), patch(
        "app.routers.github._clone_or_update", side_effect=fake_clone
    ), patch(
        "app.routers.github._run_ingestion", new_callable=AsyncMock
    ):
        resp = client.post(
            "/github/index",
            json={"full_name": "navistone/legacy-api"},
        )

    assert resp.status_code == 202
    assert captured["token"] == "env-pat-FAKE"
    assert captured["token_source"] == "env"


def test_env_fallback_marks_source_none_when_no_token_anywhere() -> None:
    """No request token AND no env token → ``token_source="none"`` (public clone)."""
    captured: dict[str, object] = {}

    def fake_clone(
        full_name: str,
        branch: str | None,
        token: str | None,
        token_source: str = "unknown",
    ) -> Path:
        captured["token"] = token
        captured["token_source"] = token_source
        return Path("/tmp/fake-clone-dest")

    with patch.object(github_router.settings, "GITHUB_ALLOWED_OWNERS", ""), patch(
        "app.routers.github._github_token", return_value=None
    ), patch("app.routers.github._token_source", return_value="none"), patch(
        "app.routers.github._clone_or_update", side_effect=fake_clone
    ), patch(
        "app.routers.github._run_ingestion", new_callable=AsyncMock
    ):
        resp = client.post(
            "/github/index",
            json={"full_name": "navistone/legacy-api"},
        )

    assert resp.status_code == 202
    assert captured["token"] is None
    assert captured["token_source"] == "none"


def test_empty_string_token_in_body_falls_back_to_env() -> None:
    """An empty-string ``github_token`` is treated as absent — env fallback fires."""
    captured: dict[str, object] = {}

    def fake_clone(
        full_name: str,
        branch: str | None,
        token: str | None,
        token_source: str = "unknown",
    ) -> Path:
        captured["token"] = token
        captured["token_source"] = token_source
        return Path("/tmp/fake-clone-dest")

    with patch.object(github_router.settings, "GITHUB_ALLOWED_OWNERS", ""), patch(
        "app.routers.github._github_token", return_value="env-pat-FAKE"
    ), patch("app.routers.github._token_source", return_value="env"), patch(
        "app.routers.github._clone_or_update", side_effect=fake_clone
    ), patch(
        "app.routers.github._run_ingestion", new_callable=AsyncMock
    ):
        resp = client.post(
            "/github/index",
            json={
                "full_name": "navistone/legacy-api",
                "github_token": "",
            },
        )

    assert resp.status_code == 202
    # Empty strings are falsy → precedence check skips them.
    assert captured["token"] == "env-pat-FAKE"
    assert captured["token_source"] == "env"


# ---------------------------------------------------------------------------
# Audit logging — token_source surfaced, full secret never logged
# ---------------------------------------------------------------------------


def test_clone_logs_token_source_and_last4_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``_clone_or_update`` logs the source + last-4 only — never the full token."""
    # Drive the real ``_clone_or_update`` but stub out subprocess so no real
    # git fork happens.  We just want to capture the audit-log line.
    with caplog.at_level(logging.INFO, logger="app.routers.github"), patch(
        "app.routers.github.subprocess.run"
    ):
        github_router._clone_or_update(
            full_name="navistone/legacy-api",
            branch=None,
            token="ghs_supersecret_abcd1234",
            token_source="request",
        )

    audit_lines = [r.getMessage() for r in caplog.records if "github_clone" in r.getMessage()]
    assert audit_lines, "expected at least one github_clone audit log line"
    line = audit_lines[0]
    assert "target=navistone/legacy-api" in line
    assert "token_source=request" in line
    assert "token_type=github_app" in line
    assert "last4=1234" in line
    # The full secret MUST NOT appear in the log.
    assert "ghs_supersecret_abcd1234" not in line
    assert "supersecret" not in line


def test_clone_logs_none_when_no_token_supplied(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Anonymous clone (public repo, no token) logs ``last4=none``."""
    with caplog.at_level(logging.INFO, logger="app.routers.github"), patch(
        "app.routers.github.subprocess.run"
    ):
        github_router._clone_or_update(
            full_name="navistone/public-repo",
            branch=None,
            token=None,
            token_source="none",
        )

    audit_lines = [r.getMessage() for r in caplog.records if "github_clone" in r.getMessage()]
    assert audit_lines
    line = audit_lines[0]
    assert "token_source=none" in line
    assert "token_type=none" in line
    assert "last4=none" in line
