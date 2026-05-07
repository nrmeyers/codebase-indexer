"""S3-primary storage for LadybugDB and DuckDB per-repo index files.

S3 is the source of truth.  Local files in LADYBUG_DB_DIR are an LRU cache.

Lifecycle:
    - On startup: restore_indexes() pulls absent or stale files from S3.
    - On every successful re-index: snapshot_indexes() pushes changed
      files back, so the cache and S3 stay in sync.
    - On graceful shutdown: snapshot_indexes() runs again as a safety net.
    - Periodically (or on demand): evict_local_cache() drops local files
      that haven't been touched in S3_INDEX_LOCAL_TTL_HOURS, keeping VM
      disk usage bounded regardless of how many repos get indexed.

The cache eviction makes this an "S3-primary" deployment: indexes don't
have to live on the VM disk forever.  Anyone who needs an evicted repo
just queries it — restore_indexes() pulls it on the next /index call,
or callers can use ensure_local() to lazy-load a single repo.

Env vars (all optional — S3 sync is a no-op when S3_INDEX_BUCKET is empty):
    S3_INDEX_BUCKET             default 'navistone-forge-data'
    S3_INDEX_PREFIX             default 'code-indexer/indexes'
    S3_INDEX_REGION             default 'us-east-1'
    S3_INDEX_LOCAL_TTL_HOURS    default '24' — evict cached files
                                untouched for this long
    S3_INDEX_LOCAL_MAX_GB       default '5' — evict oldest files until
                                local cache is under this size

File extensions synced:
    *.db    — LadybugDB graph store
    *.duck  — DuckDB vector store

WAL / shadow files (*.db.wal, *.db.shadow) are intentionally excluded;
they are recreated automatically from the base *.db on first open.

Provider priority: boto3 (standard AWS SDK) with the default credential
chain — env vars, ~/.aws/credentials, EC2 instance metadata, ECS task
role.  No special setup needed in production when the container runs
under ForgeServiceTaskRole.
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

# In-process bookkeeping for /health.  Updated by snapshot_indexes() on
# every successful push so operators can see when the local indexer last
# synced to S3 without having to query the bucket directly.
_LAST_SNAPSHOT_AT: float | None = None
_LAST_SNAPSHOT_COUNT: int | None = None


def get_sync_state() -> dict:
    """Snapshot of the S3 sync configuration + last push timestamp.

    Returned in /health so the frontend can render a "Synced 2 min ago"
    badge per indexer instance.  ``enabled`` is True when boto3 is
    importable AND a bucket name is set.  When False, the indexer is
    operating in local-only mode and indexes will be lost if the host
    is reset.
    """
    enabled = False
    try:
        import boto3  # type: ignore[import-untyped] # noqa: F401
        enabled = bool(_bucket())
    except Exception:
        enabled = False
    return {
        "enabled": enabled,
        "bucket": _bucket() if enabled else None,
        "prefix": _prefix() if enabled else None,
        "region": _region() if enabled else None,
        "last_snapshot_at": _LAST_SNAPSHOT_AT,
        "last_snapshot_count": _LAST_SNAPSHOT_COUNT,
    }


def _bucket() -> str:
    return (os.environ.get("S3_INDEX_BUCKET") or _DEFAULT_BUCKET).strip()


def _prefix() -> str:
    return (os.environ.get("S3_INDEX_PREFIX") or _DEFAULT_PREFIX).strip().rstrip("/")


def _region() -> str:
    return (os.environ.get("S3_INDEX_REGION") or _DEFAULT_REGION).strip()


def _local_ttl_hours() -> float:
    try:
        return float(os.environ.get("S3_INDEX_LOCAL_TTL_HOURS") or "24")
    except (TypeError, ValueError):
        return 24.0


def _local_max_gb() -> float:
    try:
        return float(os.environ.get("S3_INDEX_LOCAL_MAX_GB") or "5")
    except (TypeError, ValueError):
        return 5.0


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

    # Record the snapshot for /health.  We track 'attempted' (anything
    # we ran the snapshot loop on, even when 0 files changed) so the
    # frontend can show "Synced now" rather than only flipping state on
    # successful pushes.
    import time as _time
    global _LAST_SNAPSHOT_AT, _LAST_SNAPSHOT_COUNT
    _LAST_SNAPSHOT_AT = _time.time()
    _LAST_SNAPSHOT_COUNT = uploaded

    if uploaded:
        logger.info("s3_store: snapshot complete — %d file(s) pushed to S3", uploaded)
    else:
        logger.info("s3_store: snapshot complete — no changes to push")

    return uploaded


# ---------------------------------------------------------------------------
# Eviction + lazy-load — S3-as-primary semantics
# ---------------------------------------------------------------------------


def evict_local_cache(db_dir: str | Path) -> tuple[int, int]:
    """Drop locally-cached index files that are stale or over-quota.

    Two-pass eviction:
        1. TTL-based: any file whose mtime is older than ``_local_ttl_hours()``
           is removed.  Reflects the assumption that files we haven't touched
           in a day are unlikely to be queried soon.
        2. Quota-based: if the remaining files still exceed
           ``_local_max_gb()``, remove the oldest ones until under the quota.

    Files are only evicted when an UP-TO-DATE copy exists in S3, so eviction
    never destroys irrecoverable state.  Files whose S3 ETag does not match
    the local MD5 are skipped (they would be lost if evicted).

    Returns ``(evicted_count, kept_count)``.  When S3 is not configured the
    function is a no-op (returns ``(0, kept)``).
    """
    bucket = _bucket()
    prefix = _prefix()
    client = _make_client()
    if client is None:
        return (0, 0)

    db_path = Path(db_dir)
    if not db_path.exists():
        return (0, 0)

    # Index S3's view by key once — avoids one HEAD per file.
    s3_etags: dict[str, str] = {}
    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
            for obj in page.get("Contents", []):
                s3_etags[obj["Key"]] = obj.get("ETag", "").strip('"')
    except Exception as exc:
        logger.warning("s3_store.evict: list failed (%s) — skipping eviction", exc)
        return (0, 0)

    import hashlib
    import time as _time

    now = _time.time()
    ttl_sec = _local_ttl_hours() * 3600
    quota_bytes = int(_local_max_gb() * 1024 * 1024 * 1024)

    candidates = []
    for local in sorted(db_path.iterdir()):
        if local.suffix not in _SYNC_EXTENSIONS:
            continue
        if not local.is_file():
            continue
        st = local.stat()
        candidates.append((local, st.st_mtime, st.st_size))

    # Pass 1 — TTL eviction
    evicted = 0
    kept_after_ttl: list[tuple[Path, float, int]] = []
    for local, mtime, size in candidates:
        if now - mtime <= ttl_sec:
            kept_after_ttl.append((local, mtime, size))
            continue

        # Only evict if S3 has a matching copy
        key = f"{prefix}/{local.name}"
        if key not in s3_etags:
            logger.warning(
                "s3_store.evict: %s skipped — no S3 copy (would lose data)",
                local.name,
            )
            kept_after_ttl.append((local, mtime, size))
            continue
        local_md5 = hashlib.md5(local.read_bytes()).hexdigest()
        if s3_etags[key] != local_md5:
            logger.info(
                "s3_store.evict: %s skipped — local MD5 differs from S3 (push first)",
                local.name,
            )
            kept_after_ttl.append((local, mtime, size))
            continue

        try:
            local.unlink()
            evicted += 1
            logger.info("s3_store.evict: dropped %s (idle for %.1f h)",
                        local.name, (now - mtime) / 3600)
        except OSError as exc:
            logger.warning("s3_store.evict: unlink %s failed: %s", local.name, exc)
            kept_after_ttl.append((local, mtime, size))

    # Pass 2 — quota eviction (only if still over)
    total_bytes = sum(size for _, _, size in kept_after_ttl)
    if total_bytes <= quota_bytes:
        return (evicted, len(kept_after_ttl))

    # Sort by mtime ascending (oldest first) and drop until under quota.
    by_age = sorted(kept_after_ttl, key=lambda t: t[1])
    final_kept: list[Path] = []
    for local, mtime, size in by_age:
        if total_bytes <= quota_bytes:
            final_kept.append(local)
            continue

        key = f"{prefix}/{local.name}"
        if key not in s3_etags:
            final_kept.append(local)
            continue
        local_md5 = hashlib.md5(local.read_bytes()).hexdigest()
        if s3_etags[key] != local_md5:
            final_kept.append(local)
            continue

        try:
            local.unlink()
            evicted += 1
            total_bytes -= size
            logger.info(
                "s3_store.evict: dropped %s (quota — local cache was %.1f GB)",
                local.name, total_bytes / 1024 / 1024 / 1024,
            )
        except OSError:
            final_kept.append(local)

    return (evicted, len(final_kept))


def ensure_local(db_dir: str | Path, repo_name: str) -> bool:
    """Lazy-load: make sure a single repo's index files exist locally.

    Used by routes that need to query a repo whose local cache may have
    been evicted.  Pulls ``{repo_name}.db`` and ``{repo_name}.duck`` from
    S3 if they're missing locally.

    Returns True when the files are now present locally (either already
    or after a successful download), False when restore failed or S3 is
    not configured.
    """
    bucket = _bucket()
    prefix = _prefix()
    client = _make_client()
    if client is None:
        return False

    db_path = Path(db_dir)
    db_path.mkdir(parents=True, exist_ok=True)

    needed = [
        (f"{prefix}/{repo_name}.db",   db_path / f"{repo_name}.db"),
        (f"{prefix}/{repo_name}.duck", db_path / f"{repo_name}.duck"),
    ]

    all_ok = True
    for key, local in needed:
        if local.exists():
            continue
        try:
            client.download_file(bucket, key, str(local))
            logger.info("s3_store.ensure_local: pulled %s", local.name)
        except Exception as exc:
            logger.warning(
                "s3_store.ensure_local: failed to pull %s: %s — caller may "
                "need to re-index",
                key, exc,
            )
            all_ok = False

    return all_ok
