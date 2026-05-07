"""End-to-end S3 snapshot+restore tests (BUC-1555b).

Exercises the full round-trip we never previously verified:

    populate db_dir → snapshot_bundle() → wipe local → restore_latest()
        → assert byte-identical recovery + sha256 verification.

Uses moto's ``mock_aws`` to stub S3 in-process so no real AWS calls leave
the test runner.  When moto is unavailable (e.g. in a stripped CI image)
the whole module is skipped — the legacy unit tests in test_s3_store.py
still cover individual functions.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

moto = pytest.importorskip("moto", reason="moto[s3] not installed; skipping restore round-trip")
boto3 = pytest.importorskip("boto3", reason="boto3 not installed")


BUCKET = "test-forge-data"
PREFIX = "code-indexer/indexes"


@pytest.fixture
def s3_env(monkeypatch):
    """Stub AWS creds + point s3_store at our mock bucket."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("S3_INDEX_BUCKET", BUCKET)
    monkeypatch.setenv("S3_INDEX_PREFIX", PREFIX)
    monkeypatch.setenv("S3_INDEX_REGION", "us-east-1")
    monkeypatch.setenv("S3_SNAPSHOT_RETAIN", "3")
    monkeypatch.setenv("S3_SNAPSHOT_MIN_AGE_HOURS", "0")  # disable floor for tests


@pytest.fixture
def mocked_s3(s3_env):
    """Spin up moto's S3 mock and create the bucket."""
    from moto import mock_aws  # type: ignore[import-untyped]

    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        # Reset module-level bookkeeping so each test starts clean.
        import app.services.s3_store as s3_store
        s3_store._LAST_SNAPSHOT_BUNDLE_AT = None
        s3_store._LAST_SNAPSHOT_BUNDLE_KEY = None
        s3_store._LAST_SNAPSHOT_AT = None
        s3_store._LAST_SNAPSHOT_COUNT = None
        s3_store._LAST_SNAPSHOT_ERROR = None
        yield client


def _write_fake_indexes(db_dir: Path, repos: list[str]) -> dict[str, bytes]:
    """Populate db_dir with .db + .duck files of distinct content.

    Returns ``{filename: bytes}`` so the test can assert on byte-identical
    restoration later.
    """
    db_dir.mkdir(parents=True, exist_ok=True)
    contents: dict[str, bytes] = {}
    for i, repo in enumerate(repos):
        db_payload = (f"ladybug-db:{repo}:" + "x" * (1024 * (i + 1))).encode()
        duck_payload = (f"duckdb:{repo}:" + "y" * (1024 * (i + 1))).encode()
        (db_dir / f"{repo}.db").write_bytes(db_payload)
        (db_dir / f"{repo}.duck").write_bytes(duck_payload)
        contents[f"{repo}.db"] = db_payload
        contents[f"{repo}.duck"] = duck_payload
    return contents


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_should_recover_byte_identical_indexes_when_restoring_latest_snapshot(
    mocked_s3, tmp_path
):
    """Snapshot a populated db_dir, wipe it, restore — assert byte-equal."""
    from app.services.s3_restore import restore_latest
    from app.services.s3_store import snapshot_bundle

    db_dir = tmp_path / "ladybug"
    expected = _write_fake_indexes(db_dir, ["alpha", "beta"])

    snap = snapshot_bundle(db_dir)
    assert snap["ok"] is True, snap
    assert snap["files_uploaded"] == 4  # 2 repos × 2 extensions
    assert snap["bytes"] > 0
    assert snap["error"] is None

    # Wipe local — this is the disaster scenario the test simulates.
    for f in db_dir.iterdir():
        f.unlink()
    assert list(db_dir.iterdir()) == []

    result = restore_latest(None, None, db_dir)
    assert result.ok is True, result
    assert result.files_restored == 4
    assert result.error is None
    assert sorted(result.verified) == sorted(expected.keys())

    for name, payload in expected.items():
        restored = (db_dir / name).read_bytes()
        assert restored == payload, f"byte mismatch on {name}"
        assert _sha256(restored) == _sha256(payload)


def test_should_skip_already_present_files_when_restore_is_rerun(mocked_s3, tmp_path):
    """Idempotency: a second restore is a no-op when local already matches."""
    from app.services.s3_restore import restore_latest
    from app.services.s3_store import snapshot_bundle

    db_dir = tmp_path / "ladybug"
    _write_fake_indexes(db_dir, ["gamma"])
    snapshot_bundle(db_dir)

    # First restore — local already matches snapshot, so all files are skipped.
    first = restore_latest(None, None, db_dir)
    assert first.ok is True
    assert sorted(first.skipped) == ["gamma.db", "gamma.duck"]
    assert first.files_restored == 0

    # Second restore — same expectation.
    second = restore_latest(None, None, db_dir)
    assert second.ok is True
    assert sorted(second.skipped) == ["gamma.db", "gamma.duck"]


def test_should_restore_specific_snapshot_when_key_supplied(mocked_s3, tmp_path):
    """Two snapshots; restore the older one by snapshot_key."""
    import time

    from app.services.s3_restore import restore_specific
    from app.services.s3_store import list_snapshots, snapshot_bundle

    db_dir = tmp_path / "ladybug"

    # Snapshot 1
    _write_fake_indexes(db_dir, ["v1"])
    first = snapshot_bundle(db_dir)
    assert first["ok"]
    time.sleep(1.1)  # ensure timestamp prefix differs

    # Snapshot 2
    for f in db_dir.iterdir():
        f.unlink()
    _write_fake_indexes(db_dir, ["v2"])
    second = snapshot_bundle(db_dir)
    assert second["ok"]

    bundles = list_snapshots()
    assert len(bundles) == 2

    # Wipe + restore the OLDER bundle (last in the list).
    target = tmp_path / "restore-target"
    older_key = first["snapshot_key"]
    result = restore_specific(None, None, older_key, target)
    assert result.ok is True
    assert result.files_restored == 2
    assert (target / "v1.db").exists()
    assert not (target / "v2.db").exists()


def test_should_fail_restore_when_sha256_mismatch_detected(mocked_s3, tmp_path):
    """Tamper with an S3 object after snapshot; restore must refuse it."""
    from app.services.s3_restore import restore_latest
    from app.services.s3_store import snapshot_bundle

    db_dir = tmp_path / "ladybug"
    _write_fake_indexes(db_dir, ["repo"])
    snap = snapshot_bundle(db_dir)
    assert snap["ok"]

    # Corrupt the .db file in S3 (manifest still has the correct SHA).
    tampered_key = f"{snap['snapshot_key']}/repo.db"
    mocked_s3.put_object(Bucket=BUCKET, Key=tampered_key, Body=b"corrupted")

    # Wipe local.
    for f in db_dir.iterdir():
        f.unlink()

    result = restore_latest(None, None, db_dir)
    assert result.ok is False
    assert result.error is not None
    assert "sha256 mismatch" in result.error


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


def test_should_retain_only_n_newest_snapshots_when_rotating(mocked_s3, tmp_path):
    """S3_SNAPSHOT_RETAIN=3, write 5 bundles → 2 oldest deleted."""
    import time

    from app.services.s3_store import list_snapshots, snapshot_bundle

    db_dir = tmp_path / "ladybug"
    _write_fake_indexes(db_dir, ["only"])

    keys: list[str] = []
    for _ in range(5):
        snap = snapshot_bundle(db_dir)
        assert snap["ok"]
        keys.append(snap["snapshot_key"])
        time.sleep(1.1)  # distinct timestamps

    bundles = list_snapshots()
    assert len(bundles) == 3, f"retain=3 violated: got {len(bundles)} bundles"

    # The 3 retained must be the 3 most recent we created.
    retained_keys = {b["key"] for b in bundles}
    assert retained_keys == set(keys[-3:])


def test_should_protect_recent_snapshots_when_floor_is_set(mocked_s3, tmp_path, monkeypatch):
    """24h floor blocks deletion of <24h-old bundles even when over retain."""
    import time

    from app.services.s3_store import list_snapshots, snapshot_bundle

    monkeypatch.setenv("S3_SNAPSHOT_RETAIN", "1")
    monkeypatch.setenv("S3_SNAPSHOT_MIN_AGE_HOURS", "24")

    db_dir = tmp_path / "ladybug"
    _write_fake_indexes(db_dir, ["only"])

    for _ in range(3):
        snap = snapshot_bundle(db_dir)
        assert snap["ok"]
        time.sleep(1.1)

    # Even with retain=1, the 24h floor protects all 3 bundles.
    bundles = list_snapshots()
    assert len(bundles) == 3


# ---------------------------------------------------------------------------
# Health probe
# ---------------------------------------------------------------------------


def test_should_report_health_after_successful_snapshot(mocked_s3, tmp_path):
    """get_snapshot_health() reflects the bundle we just wrote."""
    from app.services.s3_store import get_snapshot_health, snapshot_bundle

    db_dir = tmp_path / "ladybug"
    _write_fake_indexes(db_dir, ["healthy"])

    pre = get_snapshot_health()
    assert pre["retained_count"] == 0
    assert pre["last_successful_snapshot_at"] is None

    snap = snapshot_bundle(db_dir)
    assert snap["ok"]

    post = get_snapshot_health()
    assert post["retained_count"] == 1
    assert post["last_successful_snapshot_at"] is not None
    assert post["age_seconds"] is not None and post["age_seconds"] >= 0
    assert post["snapshot_key"] == snap["snapshot_key"]


def test_should_report_empty_health_when_no_snapshots_exist(mocked_s3):
    """Fresh deploy probe — bucket exists but no bundles yet."""
    from app.services.s3_store import get_snapshot_health

    h = get_snapshot_health()
    assert h["retained_count"] == 0
    assert h["last_successful_snapshot_at"] is None
    assert h["age_seconds"] is None
    assert h["oldest_retained_at"] is None
    assert h["snapshot_key"] is None


# ---------------------------------------------------------------------------
# Restore — error paths
# ---------------------------------------------------------------------------


def test_should_return_error_when_no_snapshots_exist_for_restore_latest(mocked_s3, tmp_path):
    from app.services.s3_restore import restore_latest

    target = tmp_path / "ladybug"
    result = restore_latest(None, None, target)
    assert result.ok is False
    assert result.error is not None
    assert "no snapshots found" in result.error


def test_should_be_no_op_when_s3_not_configured(monkeypatch, tmp_path):
    """Without S3_INDEX_BUCKET + boto3 importable, restore returns ok=False cleanly."""
    monkeypatch.delenv("S3_INDEX_BUCKET", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

    # Force boto3 to fail by patching _make_client directly.
    import app.services.s3_restore as s3_restore
    monkeypatch.setattr(s3_restore, "_make_client", lambda: None)

    result = s3_restore.restore_latest(None, None, tmp_path)
    assert result.ok is False
    assert result.error == "s3 not configured"
