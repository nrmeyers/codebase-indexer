"""Unit tests for app.services.s3_store.

Tests use moto (or a lightweight manual mock) to stub out boto3's S3 client
so no real AWS calls are made.  Moto is an optional dev dependency; when it
isn't installed the tests that need it are skipped gracefully.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from app.services.s3_store import restore_indexes, snapshot_indexes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


# ---------------------------------------------------------------------------
# restore_indexes
# ---------------------------------------------------------------------------


class TestRestoreIndexes:
    """restore_indexes: pull absent/stale files from S3."""

    def _make_s3_client(
        self,
        objects: list[dict],
        content_map: dict[str, bytes],
        last_modified_map: dict[str, object] | None = None,
    ) -> MagicMock:
        """Build a minimal mock boto3 S3 client.

        ``last_modified_map`` (optional) maps S3 key → a value to expose as
        the object's ``LastModified`` on both ``list_objects_v2`` and
        ``head_object``.  When omitted the field is absent, exercising the
        legacy content-only code path.
        """
        last_modified_map = last_modified_map or {}
        client = MagicMock()

        # paginator for list_objects_v2 — ensure all objects have Size (copy to avoid in-place mutation)
        enriched_objects = []
        for obj in objects:
            enriched = dict(obj)  # copy
            if "Size" not in enriched and "Key" in enriched:
                key = enriched["Key"]
                enriched["Size"] = len(content_map.get(key, b""))
            if "LastModified" not in enriched and enriched.get("Key") in last_modified_map:
                enriched["LastModified"] = last_modified_map[enriched["Key"]]
            enriched_objects.append(enriched)
        page = {"Contents": enriched_objects}
        paginator = MagicMock()
        paginator.paginate.return_value = [page]
        client.get_paginator.return_value = paginator

        def _download(bucket, key, dest):  # type: ignore[override]
            path = Path(dest)
            path.write_bytes(content_map[key])

        client.download_file.side_effect = _download

        # Mock head_object for multipart detection
        def _head_object(Bucket, Key):  # type: ignore[override]
            resp: dict[str, object] = {"Metadata": {}}
            if Key in content_map:
                resp["Metadata"] = {"content-md5": _md5(content_map[Key])}
            if Key in last_modified_map:
                resp["LastModified"] = last_modified_map[Key]
            return resp
        client.head_object.side_effect = _head_object

        return client

    def test_downloads_absent_file(self, tmp_path: Path) -> None:
        content = b"fake-db-content"
        prefix = "code-indexer/indexes"
        objects = [
            {"Key": f"{prefix}/repo1.db", "ETag": f'"{_md5(content)}"'},
        ]
        client = self._make_s3_client(objects, {f"{prefix}/repo1.db": content})

        with (
            patch("app.services.s3_store._make_client", return_value=client),
            patch.dict(os.environ, {"S3_INDEX_BUCKET": "test-bucket", "S3_INDEX_PREFIX": prefix}),
        ):
            n = restore_indexes(tmp_path)

        assert n == 1
        assert (tmp_path / "repo1.db").read_bytes() == content

    def test_skips_up_to_date_file(self, tmp_path: Path) -> None:
        content = b"identical-content"
        prefix = "code-indexer/indexes"
        local_file = tmp_path / "repo1.db"
        local_file.write_bytes(content)
        key = f"{prefix}/repo1.db"
        objects = [
            {"Key": key, "ETag": f'"{_md5(content)}"'},
        ]
        client = self._make_s3_client(objects, {key: content})

        with (
            patch("app.services.s3_store._make_client", return_value=client),
            patch.dict(os.environ, {"S3_INDEX_BUCKET": "test-bucket", "S3_INDEX_PREFIX": prefix}),
        ):
            n = restore_indexes(tmp_path)

        assert n == 0
        client.download_file.assert_not_called()

    def test_refreshes_stale_file(self, tmp_path: Path) -> None:
        old_content = b"old-content"
        new_content = b"new-content-from-s3"
        prefix = "code-indexer/indexes"
        local_file = tmp_path / "repo1.db"
        local_file.write_bytes(old_content)
        objects = [
            {"Key": f"{prefix}/repo1.db", "ETag": f'"{_md5(new_content)}"'},
        ]
        client = self._make_s3_client(objects, {f"{prefix}/repo1.db": new_content})

        with (
            patch("app.services.s3_store._make_client", return_value=client),
            patch.dict(os.environ, {"S3_INDEX_BUCKET": "test-bucket", "S3_INDEX_PREFIX": prefix}),
        ):
            n = restore_indexes(tmp_path)

        assert n == 1
        assert local_file.read_bytes() == new_content

    def test_preserves_newer_local_when_s3_is_stale(self, tmp_path: Path) -> None:
        """Regression (fix/embed-rows-vanish-live): a freshly-written local
        ``.duck`` must NOT be clobbered by an older, differing S3 copy.

        Reproduces the "embeddings silently vanish" production failure: an
        empty ``<repo>.duck`` left in S3 from a prior force-reindex snapshot
        was downloaded over a just-embedded local file on the next boot,
        zeroing the embeddings table with no visible job.  The recency guard
        keeps the newer local copy.
        """
        import datetime
        import os as _os
        import time as _time

        stale_s3 = b"EMPTY-DUCK-SCHEMA-ONLY"          # what S3 wrongly holds
        fresh_local = b"FRESH-EMBED-3618-ROWS" * 50    # newer + different
        prefix = "code-indexer/indexes"
        key = f"{prefix}/TheForge.duck"

        local_file = tmp_path / "TheForge.duck"
        local_file.write_bytes(fresh_local)
        now = _time.time()
        _os.utime(local_file, (now, now))  # local mtime = "now"

        # S3 object is one hour OLDER than the local file.
        s3_dt = datetime.datetime.fromtimestamp(now - 3600, tz=datetime.timezone.utc)
        objects = [{"Key": key, "ETag": f'"{_md5(stale_s3)}"'}]
        client = self._make_s3_client(
            objects, {key: stale_s3}, last_modified_map={key: s3_dt}
        )

        with (
            patch("app.services.s3_store._make_client", return_value=client),
            patch.dict(os.environ, {"S3_INDEX_BUCKET": "test-bucket", "S3_INDEX_PREFIX": prefix}),
        ):
            n = restore_indexes(tmp_path)

        assert n == 0, "stale S3 copy must not be downloaded over a newer local"
        client.download_file.assert_not_called()
        assert local_file.read_bytes() == fresh_local, "fresh local .duck must survive"

    def test_refreshes_when_s3_is_genuinely_newer(self, tmp_path: Path) -> None:
        """The recency guard must NOT over-correct: when the S3 object is
        newer than the local file (e.g. another container re-indexed), a
        differing local copy IS refreshed from S3.
        """
        import datetime
        import os as _os
        import time as _time

        old_local = b"old-local-content"
        new_s3 = b"new-content-from-another-container"
        prefix = "code-indexer/indexes"
        key = f"{prefix}/repo1.duck"

        local_file = tmp_path / "repo1.duck"
        local_file.write_bytes(old_local)
        now = _time.time()
        _os.utime(local_file, (now - 3600, now - 3600))  # local is one hour OLD

        # S3 object is newer than local.
        s3_dt = datetime.datetime.fromtimestamp(now, tz=datetime.timezone.utc)
        objects = [{"Key": key, "ETag": f'"{_md5(new_s3)}"'}]
        client = self._make_s3_client(
            objects, {key: new_s3}, last_modified_map={key: s3_dt}
        )

        with (
            patch("app.services.s3_store._make_client", return_value=client),
            patch.dict(os.environ, {"S3_INDEX_BUCKET": "test-bucket", "S3_INDEX_PREFIX": prefix}),
        ):
            n = restore_indexes(tmp_path)

        assert n == 1, "a genuinely-newer S3 object must refresh a stale local"
        assert local_file.read_bytes() == new_s3

    def test_skips_non_index_extensions(self, tmp_path: Path) -> None:
        prefix = "code-indexer/indexes"
        objects = [
            {"Key": f"{prefix}/manifest.json", "ETag": '"abc123"'},
            {"Key": f"{prefix}/README.txt", "ETag": '"def456"'},
        ]
        client = self._make_s3_client(objects, {})

        with (
            patch("app.services.s3_store._make_client", return_value=client),
            patch.dict(os.environ, {"S3_INDEX_BUCKET": "test-bucket", "S3_INDEX_PREFIX": prefix}),
        ):
            n = restore_indexes(tmp_path)

        assert n == 0
        client.download_file.assert_not_called()

    def test_skips_sub_path_keys(self, tmp_path: Path) -> None:
        prefix = "code-indexer/indexes"
        objects = [
            {"Key": f"{prefix}/subdir/repo1.db", "ETag": '"abc123"'},
        ]
        client = self._make_s3_client(objects, {})

        with (
            patch("app.services.s3_store._make_client", return_value=client),
            patch.dict(os.environ, {"S3_INDEX_BUCKET": "test-bucket", "S3_INDEX_PREFIX": prefix}),
        ):
            n = restore_indexes(tmp_path)

        assert n == 0
        client.download_file.assert_not_called()

    def test_no_op_when_client_unavailable(self, tmp_path: Path) -> None:
        with patch("app.services.s3_store._make_client", return_value=None):
            n = restore_indexes(tmp_path)
        assert n == 0

    def test_returns_zero_on_s3_error(self, tmp_path: Path) -> None:
        client = MagicMock()
        paginator = MagicMock()
        paginator.paginate.side_effect = RuntimeError("network error")
        client.get_paginator.return_value = paginator

        with patch("app.services.s3_store._make_client", return_value=client):
            n = restore_indexes(tmp_path)

        assert n == 0


# ---------------------------------------------------------------------------
# snapshot_indexes
# ---------------------------------------------------------------------------


class TestSnapshotIndexes:
    """snapshot_indexes: push changed local files to S3."""

    def _make_s3_client(
        self,
        existing_etags: dict[str, str] | None = None,
        existing_sizes: dict[str, int] | None = None,
    ) -> MagicMock:
        """Build a minimal mock boto3 S3 client with an existing etag map."""
        client = MagicMock()
        existing_etags = existing_etags or {}
        existing_sizes = existing_sizes or {}

        # Build Contents with Size field so size comparison works
        contents = []
        for k, v in existing_etags.items():
            contents.append({
                "Key": k,
                "ETag": f'"{v}"',
                "Size": existing_sizes.get(k, 100),  # use provided size or default
            })
        page = {"Contents": contents}
        paginator = MagicMock()
        paginator.paginate.return_value = [page]
        client.get_paginator.return_value = paginator

        # Mock head_object for multipart detection — return Metadata with content-md5
        # so the snapshot code can verify multipart files
        def _head_object(Bucket, Key):  # type: ignore[override]
            if Key in existing_etags:
                return {"Metadata": {"content-md5": existing_etags[Key]}}
            return {"Metadata": {}}
        client.head_object.side_effect = _head_object

        return client

    def test_uploads_new_files(self, tmp_path: Path) -> None:
        (tmp_path / "repo1.db").write_bytes(b"db-content")
        (tmp_path / "repo1.duck").write_bytes(b"duck-content")
        client = self._make_s3_client()

        with (
            patch("app.services.s3_store._make_client", return_value=client),
            patch.dict(os.environ, {"S3_INDEX_BUCKET": "test-bucket", "S3_INDEX_PREFIX": "code-indexer/indexes"}),
        ):
            n = snapshot_indexes(tmp_path)

        assert n == 2
        assert client.upload_file.call_count == 2

    def test_skips_unchanged_files(self, tmp_path: Path) -> None:
        content = b"unchanged-db"
        (tmp_path / "repo1.db").write_bytes(content)

        # Mock client with the existing file
        client = MagicMock()
        existing_etags = {"code-indexer/indexes/repo1.db": _md5(content)}

        # Build Contents with Size field matching actual content
        contents = [
            {
                "Key": "code-indexer/indexes/repo1.db",
                "ETag": f'"{_md5(content)}"',
                "Size": len(content),
            }
        ]
        page = {"Contents": contents}
        paginator = MagicMock()
        paginator.paginate.return_value = [page]
        client.get_paginator.return_value = paginator

        # Mock head_object
        def _head_object(Bucket, Key):  # type: ignore[override]
            if Key in existing_etags:
                return {"Metadata": {"content-md5": existing_etags[Key]}}
            return {"Metadata": {}}
        client.head_object.side_effect = _head_object

        with (
            patch("app.services.s3_store._make_client", return_value=client),
            patch.dict(os.environ, {"S3_INDEX_BUCKET": "test-bucket", "S3_INDEX_PREFIX": "code-indexer/indexes"}),
        ):
            n = snapshot_indexes(tmp_path)

        assert n == 0
        client.upload_file.assert_not_called()

    def test_uploads_changed_files_only(self, tmp_path: Path) -> None:
        old_content = b"old-content"
        new_content = b"new-content"
        (tmp_path / "repo1.db").write_bytes(new_content)
        (tmp_path / "repo2.db").write_bytes(old_content)
        # repo2.db is up-to-date in S3
        client = self._make_s3_client(
            existing_etags={"code-indexer/indexes/repo2.db": _md5(old_content)},
            existing_sizes={"code-indexer/indexes/repo2.db": len(old_content)},
        )

        with (
            patch("app.services.s3_store._make_client", return_value=client),
            patch.dict(os.environ, {"S3_INDEX_BUCKET": "test-bucket", "S3_INDEX_PREFIX": "code-indexer/indexes"}),
        ):
            n = snapshot_indexes(tmp_path)

        assert n == 1
        uploaded_files = [c.args[0] for c in client.upload_file.call_args_list]
        assert any("repo1.db" in f for f in uploaded_files)

    def test_skips_wal_and_shadow_files(self, tmp_path: Path) -> None:
        (tmp_path / "repo1.db.wal").write_bytes(b"wal-data")
        (tmp_path / "repo1.db.shadow").write_bytes(b"shadow-data")
        client = self._make_s3_client()

        with (
            patch("app.services.s3_store._make_client", return_value=client),
            patch.dict(os.environ, {"S3_INDEX_BUCKET": "test-bucket", "S3_INDEX_PREFIX": "code-indexer/indexes"}),
        ):
            n = snapshot_indexes(tmp_path)

        assert n == 0
        client.upload_file.assert_not_called()

    def test_no_op_when_client_unavailable(self, tmp_path: Path) -> None:
        with patch("app.services.s3_store._make_client", return_value=None):
            n = snapshot_indexes("/nonexistent")
        assert n == 0

    def test_no_op_when_db_dir_missing(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "does-not-exist"
        client = self._make_s3_client()

        with patch("app.services.s3_store._make_client", return_value=client):
            n = snapshot_indexes(nonexistent)

        assert n == 0
        client.upload_file.assert_not_called()

    def test_continues_after_individual_upload_failure(self, tmp_path: Path) -> None:
        (tmp_path / "repo1.db").write_bytes(b"content-a")
        (tmp_path / "repo2.db").write_bytes(b"content-b")
        client = self._make_s3_client()

        upload_calls: list[str] = []

        def _upload(src, bucket, key, ExtraArgs=None):  # type: ignore[override]
            upload_calls.append(key)
            if "repo1" in key:
                raise RuntimeError("upload failed")

        client.upload_file.side_effect = _upload

        with (
            patch("app.services.s3_store._make_client", return_value=client),
            patch.dict(os.environ, {"S3_INDEX_BUCKET": "test-bucket", "S3_INDEX_PREFIX": "code-indexer/indexes"}),
        ):
            n = snapshot_indexes(tmp_path)

        # repo1 failed, repo2 succeeded — count reflects only successful uploads
        assert n == 1
        assert len(upload_calls) == 2

    def test_tracks_upload_error_in_sync_state(self, tmp_path: Path) -> None:
        """Verify that snapshot_indexes records error messages for /health."""
        from app.services.s3_store import get_sync_state
        (tmp_path / "repo1.db").write_bytes(b"content")
        client = self._make_s3_client()

        def _upload(src, bucket, key, ExtraArgs=None):  # type: ignore[override]
            raise RuntimeError("S3 auth failed")

        client.upload_file.side_effect = _upload

        with (
            patch("app.services.s3_store._make_client", return_value=client),
            patch.dict(os.environ, {"S3_INDEX_BUCKET": "test-bucket", "S3_INDEX_PREFIX": "code-indexer/indexes"}),
        ):
            snapshot_indexes(tmp_path)
            state = get_sync_state()

        assert state["last_error"] is not None
        assert "S3 auth failed" in state["last_error"]

    def test_clears_error_on_successful_snapshot(self, tmp_path: Path) -> None:
        """Verify that errors are cleared after a successful snapshot."""
        from app.services.s3_store import get_sync_state
        (tmp_path / "repo1.db").write_bytes(b"content")
        client = self._make_s3_client()

        with (
            patch("app.services.s3_store._make_client", return_value=client),
            patch.dict(os.environ, {"S3_INDEX_BUCKET": "test-bucket", "S3_INDEX_PREFIX": "code-indexer/indexes"}),
        ):
            snapshot_indexes(tmp_path)
            state = get_sync_state()

        assert state["last_error"] is None

    def test_refuses_to_push_empty_duck_over_larger_s3_copy(self, tmp_path: Path) -> None:
        """Regression (fix/embed-rows-vanish-live): the periodic snapshot must
        NOT overwrite a populated S3 ``.duck`` with a 0-row local ``.duck``.

        This is the S3-poisoning leg of the bug — a force-reindex / SIGKILL
        mid-embed leaves the local ``.duck`` empty, and the unconditional
        periodic snapshot would push it over the good S3 copy, after which
        every boot restores the empty file everywhere.
        """
        pytest.importorskip("duckdb")
        from codebase_rag.storage.vector_store import open_or_create

        # Real 0-row .duck (schema only).
        duck_path = tmp_path / "TheForge.duck"
        open_or_create(str(duck_path)).close()
        local_size = duck_path.stat().st_size
        assert local_size > 0  # schema-only files are non-empty on disk

        key = "code-indexer/indexes/TheForge.duck"
        # S3 already holds a LARGER (populated) copy.
        client = self._make_s3_client(
            existing_etags={key: "deadbeef"},
            existing_sizes={key: local_size + 5_000_000},
        )

        with (
            patch("app.services.s3_store._make_client", return_value=client),
            patch.dict(os.environ, {"S3_INDEX_BUCKET": "test-bucket", "S3_INDEX_PREFIX": "code-indexer/indexes"}),
        ):
            n = snapshot_indexes(tmp_path)

        assert n == 0, "empty .duck must not be pushed over a larger S3 copy"
        client.upload_file.assert_not_called()

    def test_pushes_empty_duck_when_no_prior_s3_copy(self, tmp_path: Path) -> None:
        """The poison guard only triggers when S3 already holds a LARGER copy.

        A first-ever push of a (legitimately) empty repo must still upload, so
        the guard cannot strand a brand-new repo.
        """
        pytest.importorskip("duckdb")
        from codebase_rag.storage.vector_store import open_or_create

        duck_path = tmp_path / "NewRepo.duck"
        open_or_create(str(duck_path)).close()

        client = self._make_s3_client()  # no existing objects

        with (
            patch("app.services.s3_store._make_client", return_value=client),
            patch.dict(os.environ, {"S3_INDEX_BUCKET": "test-bucket", "S3_INDEX_PREFIX": "code-indexer/indexes"}),
        ):
            n = snapshot_indexes(tmp_path)

        assert n == 1, "first push of an empty repo must still upload"
        client.upload_file.assert_called_once()
