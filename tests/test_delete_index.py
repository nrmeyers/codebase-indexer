"""Integration tests for DELETE /index/{repo} — cascading delete."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import jobs_store as _jobs_store

client = TestClient(app)


class TestDeleteIndex:
    """DELETE /index/{repo} — cascading resource cleanup."""

    def test_404_when_no_index_exists(self) -> None:
        """Return 404 when trying to delete a repo with no index."""
        resp = client.delete("/index/nonexistent-repo")
        assert resp.status_code == 404
        assert "No index found" in resp.json()["detail"]

    def test_returns_ok_when_ladybug_db_exists(self, tmp_path: Path) -> None:
        """Return ok=True when LadybugDB file exists."""
        with patch("app.config.settings.LADYBUG_DB_DIR", str(tmp_path)):
            # Create a dummy DB file
            db_file = tmp_path / "test-repo.db"
            db_file.touch()

            resp = client.delete("/index/test-repo")

        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["repo"] == "test-repo"
        assert "ladybug_db" in body["cleanup"]
        assert "duckdb" in body["cleanup"]
        assert "s3" in body["cleanup"]
        assert "embedding_cache" in body["cleanup"]
        assert "embed_logs" in body["cleanup"]
        assert "jobs_store" in body["cleanup"]
        assert "repo_meta" in body["cleanup"]

    def test_deletes_ladybug_db_files(self, tmp_path: Path) -> None:
        """Delete LadybugDB .db, .wal, and .shadow files."""
        with patch("app.config.settings.LADYBUG_DB_DIR", str(tmp_path)):
            db_file = tmp_path / "test-repo.db"
            wal_file = tmp_path / "test-repo.db.wal"
            shadow_file = tmp_path / "test-repo.db.shadow"

            db_file.touch()
            wal_file.touch()
            shadow_file.touch()

            assert db_file.exists()
            assert wal_file.exists()
            assert shadow_file.exists()

            resp = client.delete("/index/test-repo")

        assert resp.status_code == 200
        assert not db_file.exists()
        assert not wal_file.exists()
        assert not shadow_file.exists()
        assert "deleted" in resp.json()["cleanup"]["ladybug_db"]

    def test_deletes_duckdb_files(self, tmp_path: Path) -> None:
        """Delete DuckDB .duck and .duck.wal files."""
        with patch("app.config.settings.LADYBUG_DB_DIR", str(tmp_path)):
            # Create minimal files to trigger the delete
            db_file = tmp_path / "test-repo.db"
            duck_file = tmp_path / "test-repo.duck"
            duck_wal = tmp_path / "test-repo.duck.wal"

            db_file.touch()
            duck_file.touch()
            duck_wal.touch()

            assert duck_file.exists()
            assert duck_wal.exists()

            resp = client.delete("/index/test-repo")

        assert resp.status_code == 200
        assert not duck_file.exists()
        assert not duck_wal.exists()
        assert "deleted" in resp.json()["cleanup"]["duckdb"]

    def test_cleanup_dict_includes_all_resource_types(self, tmp_path: Path) -> None:
        """Verify response includes cleanup status for all 7 resource types."""
        with patch("app.config.settings.LADYBUG_DB_DIR", str(tmp_path)):
            db_file = tmp_path / "test-repo.db"
            db_file.touch()

            resp = client.delete("/index/test-repo")

        assert resp.status_code == 200
        body = resp.json()
        cleanup = body["cleanup"]

        # All 7 resource types should be present
        assert set(cleanup.keys()) == {
            "ladybug_db",
            "duckdb",
            "s3",
            "embedding_cache",
            "embed_logs",
            "jobs_store",
            "repo_meta",
        }

        # Each should have a string status
        for key, status in cleanup.items():
            assert isinstance(status, str)
            assert status in (
                "not found",
                "not applicable",
                "not applicable (in duckdb)",
            ) or status.startswith("deleted") or status.startswith("error:")

    def test_deletes_jobs_store_entries(self, tmp_path: Path) -> None:
        """Delete all job records for the repo."""
        with patch("app.config.settings.LADYBUG_DB_DIR", str(tmp_path)):
            db_file = tmp_path / "test-repo.db"
            db_file.touch()

            # Mock jobs_store to track the call
            with patch("app.routers.index._jobs_store.delete_by_repo") as mock_delete:
                mock_delete.return_value = 3

                resp = client.delete("/index/test-repo")

            assert resp.status_code == 200
            assert mock_delete.called
            assert "deleted 3 row(s)" in resp.json()["cleanup"]["jobs_store"]

    def test_s3_delete_integration(self, tmp_path: Path) -> None:
        """Test S3 backup deletion."""
        with patch("app.config.settings.LADYBUG_DB_DIR", str(tmp_path)):
            db_file = tmp_path / "test-repo.db"
            db_file.touch()

            # Mock the S3 delete
            with patch("app.services.s3_store.delete_repo_backup") as mock_s3:
                mock_s3.return_value = "deleted 2 file(s)"

                resp = client.delete("/index/test-repo")

            assert resp.status_code == 200
            assert mock_s3.called
            assert resp.json()["cleanup"]["s3"] == "deleted 2 file(s)"

    def test_embed_logs_cleanup(self, tmp_path: Path) -> None:
        """Test embed log file cleanup returns status."""
        with patch("app.config.settings.LADYBUG_DB_DIR", str(tmp_path)):
            db_file = tmp_path / "test-repo.db"
            db_file.touch()

            resp = client.delete("/index/test-repo")

        assert resp.status_code == 200
        # embed_logs should have a status (either "not found" or "error: ...")
        assert "embed_logs" in resp.json()["cleanup"]


class TestJobsStoreDeleteByRepo:
    """Test app.services.jobs_store.delete_by_repo helper."""

    def test_delete_by_repo_returns_count(self) -> None:
        """Delete all jobs for a repo and return row count."""
        # Use in-memory database for testing
        with patch.dict("os.environ", {"JOBS_DB_PATH": ":memory:"}):
            _jobs_store._reset_for_tests()
            _jobs_store.init(":memory:")

            # Create some test jobs
            job1 = _jobs_store.create_job(
                kind="index",
                actor_oid="user1",
                actor_email="user1@example.com",
                repo_path="/path/to/test-repo",
                force_reindex=False,
            )
            job2 = _jobs_store.create_job(
                kind="index",
                actor_oid="user2",
                actor_email="user2@example.com",
                repo_path="/path/to/test-repo",
                force_reindex=False,
            )
            job3 = _jobs_store.create_job(
                kind="index",
                actor_oid="user1",
                actor_email="user1@example.com",
                repo_path="/path/to/other-repo",
                force_reindex=False,
            )

            # Verify jobs exist
            assert _jobs_store.get_job(job1.job_id) is not None
            assert _jobs_store.get_job(job2.job_id) is not None
            assert _jobs_store.get_job(job3.job_id) is not None

            # Delete jobs for test-repo (derived from repo_path)
            deleted_count = _jobs_store.delete_by_repo("test-repo")

            # Verify count
            assert deleted_count == 2

            # Verify correct jobs were deleted
            assert _jobs_store.get_job(job1.job_id) is None
            assert _jobs_store.get_job(job2.job_id) is None
            # Other repo's job should still exist
            assert _jobs_store.get_job(job3.job_id) is not None
