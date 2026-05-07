"""Tests for BUC-1561b — GET /repos listing + App-authenticated clones.

Covers:
    1. GET /repos returns an empty envelope on a fresh DB.
    2. GET /repos surfaces a repo after a successful index run (mocked).
    3. POST /index accepts ``github_token`` + ``full_name`` and routes
       through the clone helper.
    4. The github_token never appears in any persisted state — neither
       the in-memory ``_jobs`` registry nor the SQLite jobs_store row.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers.index import _jobs, indexed_repo_paths, indexed_repos
from app.services import jobs_store


client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    """Reset every shared mutable so tests don't leak into each other."""
    _jobs.clear()
    indexed_repos.clear()
    indexed_repo_paths.clear()
    jobs_store._reset_for_tests(":memory:")
    yield
    _jobs.clear()
    indexed_repos.clear()
    indexed_repo_paths.clear()


# ---------------------------------------------------------------------------
# GET /repos
# ---------------------------------------------------------------------------


def test_should_return_empty_repos_when_db_is_fresh(tmp_path: Path) -> None:
    # Point the indexer at an empty DB dir so the on-disk scan finds nothing.
    with patch("app.routers.repos.settings") as mock_settings:
        mock_settings.LADYBUG_DB_DIR = str(tmp_path)
        resp = client.get("/repos")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"repos": []}


def test_should_list_one_repo_after_successful_index(tmp_path: Path) -> None:
    """Simulate a completed index by populating the in-memory registry +
    DuckDB metadata, then verify GET /repos surfaces a fresh entry."""
    slug = "navistone__example"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    indexed_repos.add(slug)
    indexed_repo_paths[slug] = str(repo_root)

    fake_meta = {
        "last_indexed_at": "1700000000.0",
        "last_indexed_sha": "abc1234567890abcdef1234567890abcdef12345",
        "root_path": str(repo_root),
    }

    with (
        patch("app.routers.index._read_meta", return_value=fake_meta),
        patch("app.routers.repos._git_sha", return_value=fake_meta["last_indexed_sha"]),
        patch("app.routers.repos.subprocess.run") as mock_run,
        patch("app.routers.repos.settings") as mock_settings,
    ):
        mock_settings.LADYBUG_DB_DIR = str(tmp_path / "no-such-dir")
        # symbolic-ref returns "main"
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "main\n"
        resp = client.get("/repos")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["repos"]) == 1
    item = body["repos"][0]
    assert item["slug"] == slug
    assert item["full_name"] == "navistone/example"
    assert item["indexed"] is True
    assert item["status"] == "fresh"
    assert item["last_indexed_sha"] == fake_meta["last_indexed_sha"]
    assert item["last_indexed_at"] is not None
    assert item["last_indexed_at"].endswith("Z")
    assert item["default_branch"] == "main"


def test_should_mark_repo_stale_when_head_sha_differs(tmp_path: Path) -> None:
    slug = "navistone__example"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    indexed_repos.add(slug)
    indexed_repo_paths[slug] = str(repo_root)

    with (
        patch(
            "app.routers.index._read_meta",
            return_value={
                "last_indexed_at": "1700000000.0",
                "last_indexed_sha": "old_sha",
                "root_path": str(repo_root),
            },
        ),
        patch("app.routers.repos._git_sha", return_value="new_sha_from_local_head"),
        patch("app.routers.repos.subprocess.run") as mock_run,
        patch("app.routers.repos.settings") as mock_settings,
    ):
        mock_settings.LADYBUG_DB_DIR = str(tmp_path / "no-such-dir")
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "main\n"
        resp = client.get("/repos")

    assert resp.status_code == 200
    item = resp.json()["repos"][0]
    assert item["status"] == "stale"


# ---------------------------------------------------------------------------
# POST /index — App-authenticated clones
# ---------------------------------------------------------------------------


def test_should_accept_github_token_and_use_token_bearing_clone_url(tmp_path: Path) -> None:
    """When github_token + full_name are present, the clone helper must be
    called with the token, and the index job should kick off against the
    cloned working tree."""
    cloned_path = tmp_path / "navistone__example"
    cloned_path.mkdir()

    captured: dict[str, object] = {}

    def _fake_clone(full_name: str, branch: str | None, token: str | None) -> Path:
        captured["full_name"] = full_name
        captured["branch"] = branch
        captured["token"] = token
        return cloned_path

    with (
        patch("app.routers.github._clone_or_update", side_effect=_fake_clone),
        patch("app.routers.index._run_ingestion", new_callable=AsyncMock),
    ):
        resp = client.post(
            "/index",
            json={
                "repo_path": "",
                "github_token": "ghs_installation_token_xyz",
                "full_name": "navistone/example",
                "branch": "main",
            },
        )

    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert "job_id" in body

    assert captured["full_name"] == "navistone/example"
    assert captured["branch"] == "main"
    assert captured["token"] == "ghs_installation_token_xyz"

    # The kicked-off job's repo_path is the cloned working tree.
    job = next(iter(_jobs.values()))
    assert job.repo_path == str(cloned_path)


def test_should_reject_github_token_without_full_name(tmp_path: Path) -> None:
    resp = client.post(
        "/index",
        json={
            "repo_path": str(tmp_path),
            "github_token": "ghs_xxx",
        },
    )
    assert resp.status_code == 422
    assert "full_name" in resp.json()["detail"]


def test_github_token_is_never_persisted_in_jobs_store(tmp_path: Path) -> None:
    """The opaque token is ephemeral — it must not show up in any persisted
    field (repo_path, error, etc) of the jobs_store row."""
    cloned_path = tmp_path / "navistone__example"
    cloned_path.mkdir()
    secret = "ghs_super_secret_installation_token"

    with (
        patch("app.routers.github._clone_or_update", return_value=cloned_path),
        patch("app.routers.index._run_ingestion", new_callable=AsyncMock),
    ):
        resp = client.post(
            "/index",
            json={
                "github_token": secret,
                "full_name": "navistone/example",
            },
        )

    assert resp.status_code == 202, resp.text

    # The in-memory _Job's repo_path is the cloned tree (no token in there).
    job = next(iter(_jobs.values()))
    assert secret not in job.repo_path

    # Check every persisted jobs_store row.
    rows = jobs_store.list_jobs(limit=500)
    for row in rows:
        # Every string field must be free of the token.
        for value in (
            row.repo_path,
            row.repo_slug,
            row.actor_oid,
            row.actor_email,
            row.error or "",
            row.current_file or "",
            row.phase or "",
            row.status,
        ):
            assert secret not in value, f"token leaked into {value!r}"


def test_local_path_mode_still_works_without_token(tmp_path: Path) -> None:
    """Backward-compat: existing callers that pass only repo_path keep working."""
    with patch("app.routers.index._run_ingestion", new_callable=AsyncMock):
        resp = client.post("/index", json={"repo_path": str(tmp_path)})
    assert resp.status_code == 202
    assert "job_id" in resp.json()
