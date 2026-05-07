"""Tests for POST /admin/s3/snapshot."""
from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestAdminS3Snapshot:
    """POST /admin/s3/snapshot — manual snapshot trigger."""

    def test_returns_ok_when_snapshot_succeeds(self) -> None:
        """Happy path: return ok=True and file count."""
        with patch("app.routers.admin.snapshot_indexes", return_value=3):
            resp = client.post("/admin/s3/snapshot")

        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["files_pushed"] == 3
        assert body["error"] is None

    def test_returns_zero_files_when_all_current(self) -> None:
        """When all files are up-to-date, return ok=True with 0 files."""
        with patch("app.routers.admin.snapshot_indexes", return_value=0):
            resp = client.post("/admin/s3/snapshot")

        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["files_pushed"] == 0
        assert body["error"] is None

    def test_returns_error_on_snapshot_failure(self) -> None:
        """When snapshot fails, return ok=False with error message."""
        with patch(
            "app.routers.admin.snapshot_indexes",
            side_effect=RuntimeError("S3 auth failed"),
        ):
            resp = client.post("/admin/s3/snapshot")

        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is False
        assert body["files_pushed"] == 0
        assert body["error"] is not None
        assert "S3 auth failed" in body["error"]

    def test_response_schema(self) -> None:
        """Verify response always includes ok, files_pushed, error fields."""
        with patch("app.routers.admin.snapshot_indexes", return_value=0):
            resp = client.post("/admin/s3/snapshot")

        assert resp.status_code == 200
        body = resp.json()
        assert "ok" in body
        assert "files_pushed" in body
        assert "error" in body
        assert isinstance(body["ok"], bool)
        assert isinstance(body["files_pushed"], int)
