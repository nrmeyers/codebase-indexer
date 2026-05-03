"""Integration tests for Phase 5 — /repos/{slug}/watch endpoints.

Uses FastAPI's TestClient for synchronous requests and patches out the
underlying watch_manager to avoid spawning real Watchdog observer threads
in the test suite.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.services.jobs_store import _reset_for_tests


# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app_client():
    """Create a test client using the real app but with a fresh jobs DB."""
    _reset_for_tests(":memory:")
    from app.main import create_app
    application = create_app()
    with TestClient(application, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture(autouse=True)
def reset_watches_between_tests():
    """Empty _watches before every test to avoid state leaking between cases."""
    import app.services.watch_manager as wm
    with wm._watches_lock:
        wm._watches.clear()
    yield
    with wm._watches_lock:
        wm._watches.clear()


@pytest.fixture(autouse=True)
def reset_jobs():
    _reset_for_tests(":memory:")
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _register_indexed_repo(slug: str, path: str) -> None:
    """Inject a slug → path mapping into the index router's in-memory dict."""
    from app.routers.index import indexed_repo_paths
    indexed_repo_paths[slug] = path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_post_watch_503_when_disabled(app_client: TestClient, monkeypatch, tmp_path):
    """should return 503 when WATCH_ENABLED=false."""
    monkeypatch.setattr("app.config.settings.WATCH_ENABLED", False)
    resp = app_client.post("/repos/some-repo/watch")
    assert resp.status_code == 503
    assert resp.json()["detail"]["code"] == "watch_disabled"


def test_post_watch_404_when_not_indexed(app_client: TestClient, monkeypatch):
    """should return 404 when the repo has no recorded source path."""
    monkeypatch.setattr("app.config.settings.WATCH_ENABLED", True)
    resp = app_client.post("/repos/nonexistent-repo/watch")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "repo_not_indexed"


def test_post_watch_202_when_valid(app_client: TestClient, monkeypatch, tmp_path):
    """should return 202 WatchAccepted when watch starts successfully."""
    monkeypatch.setattr("app.config.settings.WATCH_ENABLED", True)
    monkeypatch.setattr("app.config.settings.WATCH_DEBOUNCE_MS", 200)

    slug = "test-valid-repo"
    repo = tmp_path / slug
    repo.mkdir()
    _register_indexed_repo(slug, str(repo))

    # Stub the watch_manager so no real threads are created.
    from app.services.watch_manager import WatchHandle
    fake_handle = WatchHandle(
        repo_slug=slug,
        repo_path=str(repo),
        actor_oid="anon",
        actor_email="anon@local",
        started_at=time.time(),
        last_event_at=None,
        last_partial_job_id=None,
        debounce_ms=200,
        pending_paths_count=0,
        state="active",
    )
    with patch("app.services.watch_manager.start_watch", new=AsyncMock(return_value=fake_handle)):
        resp = app_client.post(f"/repos/{slug}/watch")

    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["watcher_id"] == slug
    assert body["debounce_ms"] == 200


def test_post_watch_409_when_already_active(app_client: TestClient, monkeypatch, tmp_path):
    """should return 409 when a watcher is already running for the slug."""
    monkeypatch.setattr("app.config.settings.WATCH_ENABLED", True)

    slug = "test-duplicate-repo"
    repo = tmp_path / slug
    repo.mkdir()
    _register_indexed_repo(slug, str(repo))

    from app.services.watch_manager import WatchAlreadyActiveError

    # Patch at the watch_manager module level — the endpoint does a local import
    # from there, so this is the correct interception point.
    with patch(
        "app.services.watch_manager.start_watch",
        new=AsyncMock(side_effect=WatchAlreadyActiveError("already active")),
    ):
        with patch("app.services.watch_manager.get_watch", return_value=None):
            resp = app_client.post(f"/repos/{slug}/watch")

    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "watch_already_active"


def test_post_watch_429_when_capacity_exceeded(app_client: TestClient, monkeypatch, tmp_path):
    """should return 429 when WATCH_MAX_REPOS is exceeded."""
    monkeypatch.setattr("app.config.settings.WATCH_ENABLED", True)

    slug = "test-cap-repo"
    repo = tmp_path / slug
    repo.mkdir()
    _register_indexed_repo(slug, str(repo))

    from app.services.watch_manager import WatchCapacityError

    with patch(
        "app.services.watch_manager.start_watch",
        new=AsyncMock(side_effect=WatchCapacityError("cap exceeded")),
    ):
        resp = app_client.post(f"/repos/{slug}/watch")

    assert resp.status_code == 429
    assert resp.json()["detail"]["code"] == "watch_capacity_exceeded"


def test_get_watch_404_when_not_active(app_client: TestClient):
    """should return 404 when no watcher is running for the slug."""
    resp = app_client.get("/repos/ghost-repo/watch")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "watch_not_active"


def test_get_watch_200_when_active(app_client: TestClient, tmp_path):
    """should return 200 WatchStatus when a watcher is running."""
    slug = "status-repo"
    repo = tmp_path / slug
    repo.mkdir()

    import app.services.watch_manager as wm

    entry = wm._WatchEntry(
        repo_slug=slug,
        repo_path=str(repo),
        actor_oid="u1",
        actor_email="u1@test",
        debounce_ms=1500,
        loop=None,  # not needed for snapshot
    )
    entry.state = "active"
    with wm._watches_lock:
        wm._watches[slug] = entry

    resp = app_client.get(f"/repos/{slug}/watch")
    assert resp.status_code == 200
    body = resp.json()
    assert body["repo_slug"] == slug
    assert body["state"] == "active"
    assert body["debounce_ms"] == 1500


def test_delete_watch_404_when_not_active(app_client: TestClient):
    """should return 404 when there is no watcher to stop."""
    resp = app_client.delete("/repos/phantom-repo/watch")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "watch_not_active"


def test_delete_watch_200_when_active(app_client: TestClient, tmp_path):
    """should return 200 with stopped_at when the watcher is stopped."""
    slug = "stop-me-repo"
    repo = tmp_path / slug
    repo.mkdir()

    import app.services.watch_manager as wm

    entry = wm._WatchEntry(
        repo_slug=slug,
        repo_path=str(repo),
        actor_oid="u1",
        actor_email="u1@test",
        debounce_ms=1500,
        loop=None,
    )
    entry.state = "active"
    with wm._watches_lock:
        wm._watches[slug] = entry

    with patch(
        "app.services.watch_manager.stop_watch",
        new=AsyncMock(return_value=True),
    ):
        resp = app_client.delete(f"/repos/{slug}/watch")

    assert resp.status_code == 200
    body = resp.json()
    assert "stopped_at" in body
