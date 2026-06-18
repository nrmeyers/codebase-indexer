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
    # LE-111: repo_path surfaces root_path so TheForge drift detection has
    # a source-tree path to work with.
    assert item["repo_path"] == str(repo_root)


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
    # LE-111: repo_path still surfaces even when stale, so TheForge can
    # probe the current local HEAD itself.
    assert item["repo_path"] == str(repo_root)


def test_should_report_indexed_when_no_git_sha_on_either_side(tmp_path: Path) -> None:
    """A non-git directory (no recorded SHA, no local HEAD) reports
    ``indexed`` — not the misleading ``stale``. There is nothing to drift
    against; this is the standalone-CLI "index a plain folder" path. TheForge
    always indexes git repos with a SHA, so it never reaches this branch."""
    slug = "local__plain_dir"
    repo_root = tmp_path / "plain"
    repo_root.mkdir()

    indexed_repos.add(slug)
    indexed_repo_paths[slug] = str(repo_root)

    with (
        patch(
            "app.routers.index._read_meta",
            return_value={
                "last_indexed_at": "1700000000.0",
                # no ``last_indexed_sha`` — the indexed path is not a git repo
                "root_path": str(repo_root),
            },
        ),
        patch("app.routers.repos._git_sha", return_value=None),
        patch("app.routers.repos.subprocess.run") as mock_run,
        patch("app.routers.repos.settings") as mock_settings,
    ):
        mock_settings.LADYBUG_DB_DIR = str(tmp_path / "no-such-dir")
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""
        resp = client.get("/repos")

    assert resp.status_code == 200
    item = resp.json()["repos"][0]
    assert item["indexed"] is True
    assert item["status"] == "indexed"


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


# ---------------------------------------------------------------------------
# LE-111 — last_indexed_sha + repo_path persistence
# ---------------------------------------------------------------------------


def test_repo_path_is_null_when_root_path_meta_is_missing(tmp_path: Path) -> None:
    """A repo listed only via on-disk .db (no .duck meta yet) reports
    repo_path=None — never crashes, never invents a path."""
    slug = "navistone__example"
    # Create a .db file but no .duck so meta is empty.
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    (db_dir / f"{slug}.db").write_bytes(b"")

    with (
        patch("app.routers.index._read_meta", return_value={}),
        patch("app.routers.repos._git_sha", return_value=None),
        patch("app.routers.repos.subprocess.run") as mock_run,
        patch("app.routers.repos.settings") as mock_settings,
    ):
        mock_settings.LADYBUG_DB_DIR = str(db_dir)
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""
        resp = client.get("/repos")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["repos"]) == 1
    item = body["repos"][0]
    assert item["slug"] == slug
    assert item["repo_path"] is None
    # No meta + no SHA → indexed:false → status:unindexed
    assert item["indexed"] is False
    assert item["status"] == "unindexed"


def test_capture_head_sha_returns_none_for_non_git_path(tmp_path: Path) -> None:
    """_capture_head_sha never raises; returns None for non-git directories."""
    from app.routers.index import _capture_head_sha

    # tmp_path is not a git checkout.
    assert _capture_head_sha(tmp_path) is None


def test_capture_head_sha_returns_sha_for_git_checkout(tmp_path: Path) -> None:
    """A real git init+commit produces a real SHA that we can read back."""
    import subprocess as _sub

    from app.routers.index import _capture_head_sha

    repo = tmp_path / "repo"
    repo.mkdir()
    _sub.run(["git", "init", "-q"], cwd=repo, check=True)
    _sub.run(["git", "config", "user.email", "test@local"], cwd=repo, check=True)
    _sub.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    (repo / "README").write_text("hi\n")
    _sub.run(["git", "add", "."], cwd=repo, check=True)
    _sub.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "-q", "-m", "init"],
        cwd=repo,
        check=True,
    )

    sha = _capture_head_sha(repo)
    assert isinstance(sha, str)
    assert len(sha) == 40
    # All hex chars.
    int(sha, 16)


def test_capture_head_sha_returns_none_for_empty_string() -> None:
    """Guard against ``str(repo)`` evaluating to an empty string."""
    from app.routers.index import _capture_head_sha

    assert _capture_head_sha("") is None


def test_repos_response_includes_sha_and_path_after_index(tmp_path: Path) -> None:
    """End-to-end: simulate the index path's _write_meta call writing both
    root_path and last_indexed_sha, then GET /repos reports both."""
    slug = "navistone__example"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    indexed_repos.add(slug)
    indexed_repo_paths[slug] = str(repo_root)

    fake_meta = {
        "last_indexed_at": "1700000000.0",
        "last_indexed_sha": "deadbeef" * 5,
        "root_path": str(repo_root),
        "node_count": "100",
        "rel_count": "200",
    }

    with (
        patch("app.routers.index._read_meta", return_value=fake_meta),
        patch("app.routers.repos._git_sha", return_value=fake_meta["last_indexed_sha"]),
        patch("app.routers.repos.subprocess.run") as mock_run,
        patch("app.routers.repos.settings") as mock_settings,
    ):
        mock_settings.LADYBUG_DB_DIR = str(tmp_path / "no-such-dir")
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "main\n"
        resp = client.get("/repos")

    assert resp.status_code == 200
    item = resp.json()["repos"][0]
    assert item["last_indexed_sha"] == fake_meta["last_indexed_sha"]
    assert item["repo_path"] == str(repo_root)
    assert item["status"] == "fresh"
