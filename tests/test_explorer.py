"""tests/test_explorer.py — unit tests for GET /explorer/info.

Verifies that the endpoint correctly reports viewer availability based on
whether the LadybugDB file exists and contains indexed projects.  No Docker,
no live kuzu-explorer required — only the info payload is under test.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import create_app


def test_explorer_info_reports_unavailable_when_db_missing(tmp_path: Path) -> None:
    """When the LadybugDB file does not exist, available must be False."""
    missing = str(tmp_path / "does-not-exist.db")
    with patch("app.routers.explorer.settings") as s1, patch("app.routers.health.settings") as s2:
        s1.LADYBUG_DB_PATH = missing
        s2.LADYBUG_DB_PATH = missing
        client = TestClient(create_app())
        resp = client.get("/explorer/info")

    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is False
    assert body["indexed_repos"] == []
    assert body["db_path"] == missing


def test_explorer_info_reports_unavailable_when_db_empty(tmp_path: Path) -> None:
    """File exists but no Project nodes — should still be unavailable (viewer would be empty)."""
    db_path = str(tmp_path / "empty.db")
    # Create an empty file so Path.exists() returns True
    Path(db_path).touch()

    with patch("app.routers.explorer.settings") as s1, patch("app.routers.health.settings") as s2, \
         patch("app.routers.explorer._get_indexed_repos", return_value=[]):
        s1.LADYBUG_DB_PATH = db_path
        s2.LADYBUG_DB_PATH = db_path
        client = TestClient(create_app())
        resp = client.get("/explorer/info")

    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is False
    assert body["indexed_repos"] == []


def test_explorer_info_reports_available_when_indexed(tmp_path: Path) -> None:
    """DB exists + Project nodes present → available=True, indexed_repos populated."""
    db_path = str(tmp_path / "has-data.db")
    Path(db_path).touch()

    with patch("app.routers.explorer.settings") as s1, patch("app.routers.health.settings") as s2, \
         patch("app.routers.explorer._get_indexed_repos", return_value=["myproject", "other-repo"]):
        s1.LADYBUG_DB_PATH = db_path
        s2.LADYBUG_DB_PATH = db_path
        # settings.db_path_for_repo(target) is called when a specific repo is
        # selected. Mock it to return the real per-repo path (same file for
        # this test) — without this, MagicMock returns a MagicMock and
        # ExplorerInfoResponse rejects it with a string_type validation error.
        s1.db_path_for_repo.return_value = db_path
        client = TestClient(create_app())
        resp = client.get("/explorer/info")

    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert body["indexed_repos"] == ["myproject", "other-repo"]


def test_explorer_info_returns_launch_command_with_db_path(tmp_path: Path) -> None:
    """The launch_command must reference the resolved DB path so the user can copy-paste."""
    db_path = str(tmp_path / "graph.db")
    Path(db_path).touch()

    with patch("app.routers.explorer.settings") as s1, patch("app.routers.health.settings") as s2, \
         patch("app.routers.explorer._get_indexed_repos", return_value=["proj"]):
        s1.LADYBUG_DB_PATH = db_path
        s2.LADYBUG_DB_PATH = db_path
        s1.db_path_for_repo.return_value = db_path
        client = TestClient(create_app())
        resp = client.get("/explorer/info")

    body = resp.json()
    assert "docker run" in body["launch_command"]
    # Service uses the ladybugdb/explorer image now (previously kuzudb/explorer
    # before the Memgraph→LadybugDB migration).
    assert "ladybugdb/explorer" in body["launch_command"]
    # Mount should point at the parent directory (ladybug stores DB as a file
    # whose parent dir is mounted as /database).
    assert str(tmp_path) in body["launch_command"]


def test_explorer_info_viewer_url_uses_expected_port(tmp_path: Path) -> None:
    """viewer_url must match the port inside launch_command (default 7001).

    Port 7001 is used instead of 7000 because macOS ControlCenter / AirPlay
    Receiver reserves 7000 by default — see explorer.py _EXPLORER_PORT.
    """
    db_path = str(tmp_path / "graph.db")
    Path(db_path).touch()

    with patch("app.routers.explorer.settings") as s1, patch("app.routers.health.settings") as s2:
        s1.LADYBUG_DB_PATH = db_path
        s2.LADYBUG_DB_PATH = db_path
        client = TestClient(create_app())
        resp = client.get("/explorer/info")

    body = resp.json()
    assert body["viewer_url"] == "http://localhost:7001"
    assert "7001:" in body["launch_command"]


def test_explorer_info_always_returns_docs_url(tmp_path: Path) -> None:
    """docs_url must always be present so the UI can link to the kuzu-explorer docs."""
    with patch("app.routers.explorer.settings") as s1, patch("app.routers.health.settings") as s2:
        s1.LADYBUG_DB_PATH = str(tmp_path / "x.db")
        s2.LADYBUG_DB_PATH = str(tmp_path / "x.db")
        client = TestClient(create_app())
        resp = client.get("/explorer/info")

    body = resp.json()
    assert body["docs_url"].startswith("https://")
    # After the Memgraph→LadybugDB migration, explorer links to LadybugDB.
    assert "ladybugdb" in body["docs_url"].lower() or "ladybug" in body["docs_url"].lower()
