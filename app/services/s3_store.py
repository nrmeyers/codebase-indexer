"""S3 snapshot / restore for LadybugDB and DuckDB per-repo index files.

On startup the service pulls any index files that are missing locally (or
stale) from S3 so a fresh container immediately inherits the last committed
graph.  On shutdown it pushes changed local files back so the next container
can pick up where this one left off.

Env vars (all optional — S3 sync is a no-op when S3_INDEX_BUCKET is empty):
    S3_INDEX_BUCKET   S3 bucket name (default: navistone-forge-data).
    S3_INDEX_PREFIX   Key prefix inside the bucket (default: code-indexer/indexes).
    S3_INDEX_REGION   AWS region for the S3 client (default: us-east-1).

File extensions synced:
    *.db    — LadybugDB graph store
    *.duck  — DuckDB vector store

WAL / shadow files (*.db.wal, *.db.shadow) are intentionally excluded;
they are recreated automatically from the base *.db on first open.

Provider priority: this module uses boto3 (standard AWS SDK) with the
default credential chain — env vars, ~/.aws/credentials, EC2 instance
metadata, ECS task role.  No special setup needed in production when the
container runs under ForgeServiceTaskRole.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_DEFAULT_BUCKET = "navistone-forge-data"
_DEFAULT_PREFIX = "code-indexer/indexes"
_DEFAULT_REGION = "us-east-1"
_SYNC_EXTENSIONS = {".db", ".duck"}


def _bucket() -> str:
    return (os.environ.get("S3_INDEX_BUCKET") or _DEFAULT_BUCKET).strip()


def _prefix() -> str:
    return (os.environ.get("S3_INDEX_PREFIX") or _DEFAULT_PREFIX).strip().rstrip("/")


def _region() -> str:
    return (os.environ.get("S3_INDEX_REGION") or _DEFAULT_REGION).strip()


def _make_client():  # type: ignore[return]
    """Create a boto3 S3 client.  Returns None on import failure."""
    try:
        import boto3  # type: ignore[import-untyped]
        return boto3.client("s3", region_name=_region())
    except Exception as exc:
        logger.warning("s3_store: boto3 unavailable — S3 sync disabled (%s)", exc)
        return None


def restore_indexes(db_dir: str | Path) -> int:
    """Pull index files from S3 into *db_dir* that are absent or stale locally.

    Compares S3 ETag (MD5 of the object) against the local file's MD5 digest
    to detect staleness.  Files already up-to-date are skipped.

    Args:
        db_dir: Local directory that holds per-repo ``.db`` / ``.duck`` files.

    Returns:
        int: Number of files downloaded.
    """
    bucket = _bucket()
    prefix = _prefix()
    client = _make_client()
    if client is None:
        return 0

    db_path = Path(db_dir)
    db_path.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                # Only sync the exact extension set — skip manifests, logs, etc.
                if not any(key.endswith(ext) for ext in _SYNC_EXTENSIONS):
                    continue

                filename = key[len(prefix) + 1:]  # strip "prefix/" to get basename
                if "/" in filename:
                    # Unexpected sub-path — skip for safety
                    continue

                local = db_path / filename
                s3_etag = obj.get("ETag", "").strip('"')

                if local.exists():
                    import hashlib
                    md5 = hashlib.md5(local.read_bytes()).hexdigest()
                    if md5 == s3_etag:
                        logger.debug("s3_store: %s already up-to-date, skipping", filename)
                        continue
                    logger.info("s3_store: %s stale (local=%s s3=%s) — refreshing", filename, md5, s3_etag)
                else:
                    logger.info("s3_store: %s absent locally — downloading", filename)

                client.download_file(bucket, key, str(local))
                downloaded += 1

    except Exception as exc:
        logger.warning(
            "s3_store.restore_indexes failed (bucket=%s prefix=%s): %s — continuing without S3 restore",
            bucket, prefix, exc,
        )

    if downloaded:
        logger.info("s3_store: restored %d index file(s) from s3://%s/%s", downloaded, bucket, prefix)
    else:
        logger.info("s3_store: all local index files are current (or bucket empty)")

    return downloaded


def snapshot_indexes(db_dir: str | Path) -> int:
    """Push local index files from *db_dir* to S3.

    Only uploads files whose MD5 digest differs from the current S3 ETag so
    that unchanged files don't generate unnecessary S3 PUT charges.

    WAL / shadow files are never uploaded (they are runtime-only artefacts).

    Args:
        db_dir: Local directory containing ``.db`` / ``.duck`` index files.

    Returns:
        int: Number of files uploaded.
    """
    bucket = _bucket()
    prefix = _prefix()
    client = _make_client()
    if client is None:
        return 0

    db_path = Path(db_dir)
    if not db_path.exists():
        logger.info("s3_store: db_dir %s does not exist — nothing to snapshot", db_dir)
        return 0

    # Collect current S3 ETags for fast diff checks
    s3_etags: dict[str, str] = {}
    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                s3_etags[key] = obj.get("ETag", "").strip('"')
    except Exception as exc:
        logger.warning(
            "s3_store.snapshot_indexes: could not list existing objects (%s) — will upload all files",
            exc,
        )

    import hashlib

    uploaded = 0
    for local in sorted(db_path.iterdir()):
        if local.suffix not in _SYNC_EXTENSIONS:
            continue
        if not local.is_file():
            continue

        key = f"{prefix}/{local.name}"
        local_md5 = hashlib.md5(local.read_bytes()).hexdigest()
        if s3_etags.get(key) == local_md5:
            logger.debug("s3_store: %s unchanged — skipping upload", local.name)
            continue

        try:
            client.upload_file(str(local), bucket, key)
            uploaded += 1
            logger.info("s3_store: uploaded %s → s3://%s/%s", local.name, bucket, key)
        except Exception as exc:
            logger.warning("s3_store: upload of %s failed: %s", local.name, exc)

    if uploaded:
        logger.info("s3_store: snapshot complete — %d file(s) pushed to S3", uploaded)
    else:
        logger.info("s3_store: snapshot complete — no changes to push")

    return uploaded
