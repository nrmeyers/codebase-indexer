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
    S3_INDEX_BUCKET             default '' (S3 sync disabled)
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

_DEFAULT_BUCKET = ""
_DEFAULT_PREFIX = "code-indexer/indexes"
_DEFAULT_REGION = "us-east-1"
_DEFAULT_SNAPSHOT_RETAIN = 10
_DEFAULT_SNAPSHOT_MIN_AGE_HOURS = 24.0  # never purge anything younger than this
_SYNC_EXTENSIONS = {".db", ".duck"}
_SNAPSHOTS_SUBPREFIX = "snapshots"  # under _prefix(): {prefix}/snapshots/<ts>/...
_MANIFEST_NAME = "index.json"

# In-process bookkeeping for /health.  Updated by snapshot_indexes() on
# every successful push so operators can see when the local indexer last
# synced to S3 without having to query the bucket directly.
_LAST_SNAPSHOT_AT: float | None = None
_LAST_SNAPSHOT_COUNT: int | None = None
_LAST_SNAPSHOT_ERROR: str | None = None
# Snapshot-bundle bookkeeping for /admin/s3/health (BUC-1555b).  Tracks the
# most recent timestamped snapshot bundle written to {prefix}/snapshots/.
_LAST_SNAPSHOT_BUNDLE_AT: float | None = None
_LAST_SNAPSHOT_BUNDLE_KEY: str | None = None


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
        "last_error": _LAST_SNAPSHOT_ERROR,
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


def _snapshot_retain() -> int:
    """Number of snapshot bundles to retain (default 10)."""
    try:
        n = int(os.environ.get("S3_SNAPSHOT_RETAIN") or _DEFAULT_SNAPSHOT_RETAIN)
        return max(1, n)
    except (TypeError, ValueError):
        return _DEFAULT_SNAPSHOT_RETAIN


def _snapshot_min_age_hours() -> float:
    """Floor — snapshots younger than this are never purged regardless of count."""
    try:
        return float(
            os.environ.get("S3_SNAPSHOT_MIN_AGE_HOURS")
            or _DEFAULT_SNAPSHOT_MIN_AGE_HOURS
        )
    except (TypeError, ValueError):
        return _DEFAULT_SNAPSHOT_MIN_AGE_HOURS


def _snapshots_prefix() -> str:
    """Prefix under which timestamped snapshot bundles live."""
    return f"{_prefix()}/{_SNAPSHOTS_SUBPREFIX}"


def _make_client():  # type: ignore[return]
    """Create a boto3 S3 client.  Returns None on import failure."""
    try:
        import boto3  # type: ignore[import-untyped]
        return boto3.client("s3", region_name=_region())
    except Exception as exc:
        logger.warning("s3_store: boto3 unavailable — S3 sync disabled (%s)", exc)
        return None


def _is_in_sync(local_path: Path, s3_obj: dict, client, bucket: str) -> bool:
    """Decide whether the local file matches the S3 object.

    Uses object size + a custom ``x-amz-meta-content-md5`` header that
    snapshot_indexes() writes alongside every PUT.  Plain ETag comparison
    BREAKS for multipart uploads (any file > 8 MB by default): boto3's
    TransferManager auto-multiparts those, and the resulting ETag is
    ``MD5(MD5(part1) + … + MD5(partN))-N`` — never equal to the local
    file's plain MD5.  Without this fix, every startup re-downloads
    every file even when nothing changed.

    Falls back to size-only comparison when the metadata header is
    missing (e.g. files uploaded by an older version of this code).
    Size+presence is good enough — once a snapshot_indexes() pass runs,
    the metadata gets stamped on the next push and we move to MD5.
    """
    s3_size = int(s3_obj.get("Size", 0))
    if local_path.stat().st_size != s3_size:
        return False  # size mismatch = definitely changed

    s3_etag = s3_obj.get("ETag", "").strip('"')
    is_multipart = "-" in s3_etag

    if not is_multipart:
        # Single-part upload — ETag IS the plain MD5.  Cheap path.
        import hashlib
        local_md5 = hashlib.md5(local_path.read_bytes()).hexdigest()
        return local_md5 == s3_etag

    # Multipart — fetch the object's user metadata for our content-md5
    try:
        head = client.head_object(Bucket=bucket, Key=s3_obj["Key"])
    except Exception:
        # HEAD failed — fall back to size match (we already know that's true)
        return True
    s3_content_md5 = head.get("Metadata", {}).get("content-md5")
    if not s3_content_md5:
        # Old upload without our stamp — accept size match as good enough
        return True
    import hashlib
    local_md5 = hashlib.md5(local_path.read_bytes()).hexdigest()
    return local_md5 == s3_content_md5


def _duck_embeddings_row_count(path: Path) -> int | None:
    """Return the ``embeddings`` row count of a ``.duck`` file, or ``None``.

    Used by the snapshot path to avoid pushing an EMPTY ``<repo>.duck`` over a
    populated copy already in S3 (the second leg of the "embeddings vanish"
    bug: a force-reindex / SIGKILL mid-embed leaves a 0-row ``.duck`` locally,
    and the unconditional periodic snapshot would otherwise poison S3 with it).

    Opens read-only and is fully fail-open: any error (duckdb missing, file
    locked by the writer, no ``embeddings`` table) returns ``None`` so the
    caller falls back to the legacy "always upload" behaviour.
    """
    try:
        import duckdb
    except Exception:
        return None
    try:
        conn = duckdb.connect(str(path), read_only=True)
        try:
            row = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
            return int(row[0]) if row is not None else None
        finally:
            conn.close()
    except Exception:
        return None


def _s3_last_modified_epoch(obj: dict) -> float | None:
    """Best-effort UTC epoch of an S3 object's ``LastModified``.

    ``list_objects_v2`` (and ``head_object``) return a timezone-aware
    ``datetime``.  We convert to a POSIX timestamp so it can be compared
    directly against ``Path.stat().st_mtime`` (also UTC epoch).  Returns
    ``None`` when the field is missing or unparsable so callers can fall
    back to the legacy content-only behaviour.
    """
    lm = obj.get("LastModified")
    if lm is None:
        return None
    try:
        # botocore returns aware datetimes; .timestamp() yields the correct
        # UTC epoch regardless of tzinfo.  A naive datetime is assumed local
        # by Python, which is acceptable for the coarse staleness check.
        return float(lm.timestamp())  # type: ignore[union-attr]
    except (AttributeError, OSError, ValueError, OverflowError):
        return None


# A local file is treated as "newer than S3" only when it is at least this
# many seconds ahead of the S3 object's LastModified.  The slack absorbs
# clock skew and the lag between a local write and the subsequent S3 PUT so
# we don't flap on a byte-identical pair whose timestamps differ by a blip.
_RESTORE_MTIME_SLACK_SEC = 2.0


def restore_indexes(db_dir: str | Path) -> int:
    """Pull index files from S3 into *db_dir* that are absent or stale locally.

    Uses size + custom content-md5 metadata header for staleness detection
    (see ``_is_in_sync``) so multipart-uploaded files don't get re-downloaded
    every startup despite being byte-identical.

    Recency guard (fix/embed-rows-vanish-live): a local file that is NEWER
    than the S3 object is NEVER overwritten, even when their content differs.
    Without this, a freshly-embedded ``<repo>.duck`` (3618 rows, mtime = embed
    time) is clobbered on the very next process boot by an older, EMPTY
    ``<repo>.duck`` left in S3 from a prior force-reindex snapshot — the
    ``restore_indexes`` call in the lifespan compared content only (size+MD5)
    and "refreshed" the local file down to zero rows.  A crash-loop (or the
    LE-137 respawn) makes this fire seconds after every embed, so embeddings
    "silently vanish" with no visible job.  We now keep the local copy when it
    is at least ``_RESTORE_MTIME_SLACK_SEC`` newer than S3; a genuinely-newer
    S3 object (e.g. from another container) still refreshes a stale local.

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

                filename = key[len(prefix) + 1:]
                if "/" in filename:
                    continue  # unexpected sub-path

                local = db_path / filename

                if local.exists() and _is_in_sync(local, obj, client, bucket):
                    logger.debug("s3_store: %s already up-to-date, skipping", filename)
                    continue
                if local.exists():
                    # Recency guard — never clobber a NEWER local file with an
                    # older S3 copy.  This is the fix for the "embeddings
                    # silently vanish" bug: a fresh local <repo>.duck must
                    # survive a boot that finds a stale, empty copy in S3.
                    s3_epoch = _s3_last_modified_epoch(obj)
                    local_mtime = local.stat().st_mtime
                    if (
                        s3_epoch is not None
                        and local_mtime > s3_epoch + _RESTORE_MTIME_SLACK_SEC
                    ):
                        logger.info(
                            "s3_store: %s differs from S3 but local is newer "
                            "(local_mtime=%.0f > s3=%.0f) — keeping local, "
                            "skipping restore",
                            filename, local_mtime, s3_epoch,
                        )
                        continue
                    logger.info("s3_store: %s out of sync — refreshing", filename)
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

    # Index existing S3 objects so we can skip uploads of unchanged files.
    # Read the user-metadata 'content-md5' we stamp on every PUT (see below).
    # Plain ETag comparison breaks on multipart uploads (the "-N" suffix
    # variant); we fall back to size-only when the metadata is absent.
    s3_state: dict[str, dict] = {}  # key → { size, content_md5 (if known) }
    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
            for obj in page.get("Contents", []):
                s3_state[obj["Key"]] = {
                    "size": int(obj.get("Size", 0)),
                    "etag": obj.get("ETag", "").strip('"'),
                    # content_md5 lazy-fetched below only when we need it
                }
    except Exception as exc:
        logger.warning(
            "s3_store.snapshot_indexes: could not list existing objects (%s) — will upload all files",
            exc,
        )

    import hashlib

    uploaded = 0
    upload_errors = []
    for local in sorted(db_path.iterdir()):
        if local.suffix not in _SYNC_EXTENSIONS:
            continue
        if not local.is_file():
            continue

        key = f"{prefix}/{local.name}"
        local_md5 = hashlib.md5(local.read_bytes()).hexdigest()
        local_size = local.stat().st_size

        # Decide whether to skip the upload.  Cheap path: size mismatch =>
        # always upload.  Otherwise compare ETag (single-part) or fetched
        # metadata (multipart).
        existing = s3_state.get(key)
        skip = False
        if existing and existing["size"] == local_size:
            etag = existing["etag"]
            if "-" not in etag and etag == local_md5:
                skip = True
            else:
                # multipart — need a HEAD to read user metadata
                try:
                    head = client.head_object(Bucket=bucket, Key=key)
                    if head.get("Metadata", {}).get("content-md5") == local_md5:
                        skip = True
                except Exception:
                    pass  # treat as changed; safe to re-upload
        if skip:
            logger.debug("s3_store: %s unchanged — skipping upload", local.name)
            continue

        # Empty-.duck poison guard (fix/embed-rows-vanish-live): never push a
        # 0-row ``<repo>.duck`` over a LARGER copy already in S3.  A
        # force-reindex or a SIGKILL mid-embed can leave the local ``.duck``
        # empty; the unconditional periodic snapshot would otherwise overwrite
        # the good S3 copy with the empty one, which the next boot then
        # restores everywhere.  Only ``.duck`` files are checked, only when S3
        # already holds a larger object, and the row-count probe is fail-open
        # (None → fall through to upload), so a legitimately-emptied repo on a
        # first push (no prior S3 object) is unaffected.
        if (
            local.suffix == ".duck"
            and existing is not None
            and existing.get("size", 0) > local_size
        ):
            rows = _duck_embeddings_row_count(local)
            if rows == 0:
                logger.warning(
                    "s3_store: refusing to push EMPTY %s (0 embedding rows, "
                    "local_size=%d) over larger S3 copy (size=%d) — likely a "
                    "post-reindex / interrupted-embed clobber; keeping S3 copy",
                    local.name, local_size, existing.get("size", 0),
                )
                continue

        try:
            # Stamp the plain MD5 in user metadata so the next restore can
            # detect identity even when boto3 multiparts the upload.
            client.upload_file(
                str(local), bucket, key,
                ExtraArgs={"Metadata": {"content-md5": local_md5}},
            )
            uploaded += 1
            logger.info("s3_store: uploaded %s → s3://%s/%s", local.name, bucket, key)
        except Exception as exc:
            error_msg = f"upload of {local.name} failed: {exc}"
            logger.warning("s3_store: %s", error_msg)
            upload_errors.append(error_msg)

    # Record the snapshot for /health.  We track 'attempted' (anything
    # we ran the snapshot loop on, even when 0 files changed) so the
    # frontend can show "Synced now" rather than only flipping state on
    # successful pushes.
    import time as _time
    global _LAST_SNAPSHOT_AT, _LAST_SNAPSHOT_COUNT, _LAST_SNAPSHOT_ERROR
    _LAST_SNAPSHOT_AT = _time.time()
    _LAST_SNAPSHOT_COUNT = uploaded
    _LAST_SNAPSHOT_ERROR = None
    if upload_errors:
        _LAST_SNAPSHOT_ERROR = "; ".join(upload_errors[:3])  # surface first 3 errors

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


def delete_repo_backup(repo_name: str) -> str:
    """Delete a repo's backup copy from S3.

    Called during cascading delete to clean up the S3 copy.
    Returns a status string: "deleted", "not found", or "error: <msg>".

    Args:
        repo_name: Repository slug (matches filename stem in S3).

    Returns:
        Status message describing what happened.
    """
    bucket = _bucket()
    prefix = _prefix()
    client = _make_client()

    if client is None:
        return "not found"  # S3 not configured; nothing to delete

    try:
        # Try to delete both .db and .duck files
        deleted_count = 0
        for ext in [".db", ".duck"]:
            key = f"{prefix}/{repo_name}{ext}"
            try:
                client.delete_object(Bucket=bucket, Key=key)
                deleted_count += 1
                logger.info("s3_store.delete_repo_backup: deleted %s", key)
            except client.exceptions.NoSuchKey:
                pass  # File doesn't exist, which is fine
            except Exception as exc:
                logger.warning(
                    "s3_store.delete_repo_backup: failed to delete %s: %s",
                    key, exc,
                )

        return f"deleted {deleted_count} file(s)" if deleted_count > 0 else "not found"
    except Exception as exc:
        msg = str(exc)
        logger.warning("s3_store.delete_repo_backup: %s", msg)
        return f"error: {msg}"


# ---------------------------------------------------------------------------
# Snapshot bundles + rotation + lifecycle (BUC-1555b)
# ---------------------------------------------------------------------------
#
# Snapshot bundles are *point-in-time* archives of every .db / .duck file in
# the local index dir, written under a timestamped sub-prefix:
#
#     s3://{bucket}/{prefix}/snapshots/{ISO8601-Z}/
#         <repo>.db
#         <repo>.duck
#         index.json    -- manifest with sha256 + size for each file
#
# These exist alongside the per-file "primary" objects under {prefix}/ which
# the LRU cache layer maintains.  Bundles enable point-in-time restore and
# are subject to rotation (S3_SNAPSHOT_RETAIN, default 10) so they don't
# grow unbounded.


def _utcnow_iso() -> str:
    """RFC 3339 UTC timestamp without colons (S3-key safe)."""
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256_file(path: Path) -> str:
    """Stream-sha256 a file (no full read into memory for large indexes)."""
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_manifest(snapshot_key: str, files: list[dict]) -> dict:
    """Manifest schema written as ``index.json`` inside each snapshot bundle."""
    import time as _time
    return {
        "version": 1,
        "snapshot_key": snapshot_key,
        "created_at": _time.time(),
        "files": files,  # [{ name, sha256, size_bytes }]
    }


def snapshot_bundle(db_dir: str | Path) -> dict:
    """Write a timestamped point-in-time snapshot bundle to S3.

    Uploads every ``.db`` / ``.duck`` file under *db_dir* into a fresh
    ``{prefix}/snapshots/<timestamp>/`` sub-prefix and writes an
    ``index.json`` manifest containing per-file SHA-256 + size.  After
    a successful upload, runs :func:`rotate_snapshots` to enforce
    retention limits.

    Returns a dict shape::

        {
            "ok": bool,
            "snapshot_key": str,        # the snapshots/<timestamp>/ prefix
            "files_uploaded": int,
            "bytes": int,
            "error": Optional[str],
            "rotation": dict,           # output of rotate_snapshots()
        }

    No-op (returns ok=False, error="s3 not configured") when boto3 is
    unavailable or the bucket is unset.
    """
    bucket = _bucket()
    client = _make_client()
    if client is None:
        return {
            "ok": False,
            "snapshot_key": "",
            "files_uploaded": 0,
            "bytes": 0,
            "error": "s3 not configured",
            "rotation": {"deleted": [], "kept": []},
        }

    db_path = Path(db_dir)
    if not db_path.exists():
        return {
            "ok": False,
            "snapshot_key": "",
            "files_uploaded": 0,
            "bytes": 0,
            "error": f"db_dir {db_dir} does not exist",
            "rotation": {"deleted": [], "kept": []},
        }

    ts = _utcnow_iso()
    snapshot_prefix = f"{_snapshots_prefix()}/{ts}"
    files_meta: list[dict] = []
    total_bytes = 0
    upload_error: str | None = None

    for local in sorted(db_path.iterdir()):
        if local.suffix not in _SYNC_EXTENSIONS or not local.is_file():
            continue
        size = local.stat().st_size
        sha = _sha256_file(local)
        key = f"{snapshot_prefix}/{local.name}"
        try:
            client.upload_file(
                str(local), bucket, key,
                ExtraArgs={"Metadata": {"sha256": sha}},
            )
            files_meta.append({"name": local.name, "sha256": sha, "size_bytes": size})
            total_bytes += size
            logger.info("s3_store.snapshot_bundle: uploaded %s → s3://%s/%s",
                        local.name, bucket, key)
        except Exception as exc:
            upload_error = f"upload of {local.name} failed: {exc}"
            logger.warning("s3_store.snapshot_bundle: %s", upload_error)
            break

    # Always write the manifest, even when partial — so a restore can
    # see what *was* meant to be in this bundle and refuse to use it
    # if integrity is broken.  Mark partial bundles in the manifest.
    manifest = _build_manifest(snapshot_prefix, files_meta)
    if upload_error:
        manifest["partial"] = True
        manifest["error"] = upload_error
    try:
        import json as _json
        client.put_object(
            Bucket=bucket,
            Key=f"{snapshot_prefix}/{_MANIFEST_NAME}",
            Body=_json.dumps(manifest, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
    except Exception as exc:
        logger.warning("s3_store.snapshot_bundle: manifest write failed: %s", exc)
        if upload_error is None:
            upload_error = f"manifest write failed: {exc}"

    ok = upload_error is None and len(files_meta) > 0

    if ok:
        global _LAST_SNAPSHOT_BUNDLE_AT, _LAST_SNAPSHOT_BUNDLE_KEY
        import time as _time
        _LAST_SNAPSHOT_BUNDLE_AT = _time.time()
        _LAST_SNAPSHOT_BUNDLE_KEY = snapshot_prefix

    rotation = rotate_snapshots() if ok else {"deleted": [], "kept": [], "skipped": "snapshot failed"}

    # Best-effort lifecycle policy install.  Skipped silently when IAM
    # blocks PutBucketLifecycleConfiguration (e.g. read-only role).
    try:
        ensure_bucket_lifecycle_policy()
    except Exception as exc:
        logger.info("s3_store.snapshot_bundle: lifecycle policy not applied (%s)", exc)

    return {
        "ok": ok,
        "snapshot_key": snapshot_prefix,
        "files_uploaded": len(files_meta),
        "bytes": total_bytes,
        "error": upload_error,
        "rotation": rotation,
    }


def list_snapshots() -> list[dict]:
    """List timestamped snapshot bundles under ``{prefix}/snapshots/``.

    Returns newest-first.  Each entry::

        {
            "key": "code-indexer/indexes/snapshots/20260507T120000Z",
            "created_at": float,        # unix epoch (max LastModified of files)
            "size_bytes": int,          # sum of objects in bundle
            "file_count": int,          # excluding the manifest
        }
    """
    bucket = _bucket()
    client = _make_client()
    if client is None:
        return []

    prefix = _snapshots_prefix()
    bundles: dict[str, dict] = {}
    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Strip "{prefix}/" -> "<timestamp>/<filename>"
                rest = key[len(prefix) + 1:]
                if "/" not in rest:
                    continue  # not inside a bundle dir
                ts_part, fname = rest.split("/", 1)
                bundle_key = f"{prefix}/{ts_part}"
                slot = bundles.setdefault(
                    bundle_key,
                    {"key": bundle_key, "created_at": 0.0, "size_bytes": 0, "file_count": 0},
                )
                slot["size_bytes"] += int(obj.get("Size", 0))
                if fname != _MANIFEST_NAME:
                    slot["file_count"] += 1
                lm = obj.get("LastModified")
                if lm is not None:
                    epoch = lm.timestamp() if hasattr(lm, "timestamp") else float(lm)
                    if epoch > slot["created_at"]:
                        slot["created_at"] = epoch
    except Exception as exc:
        logger.warning("s3_store.list_snapshots: list failed: %s", exc)
        return []

    return sorted(bundles.values(), key=lambda b: b["created_at"], reverse=True)


def rotate_snapshots(
    retain: int | None = None,
    min_age_hours: float | None = None,
) -> dict:
    """Delete old snapshot bundles, keeping the newest *retain* and a 24h floor.

    The floor protects against accidental purge during a burst of snapshots:
    no bundle younger than ``min_age_hours`` (default 24h) is ever deleted,
    even if it pushes us above *retain*.

    Returns ``{"deleted": [keys], "kept": [keys]}`` with every action logged.
    """
    bucket = _bucket()
    client = _make_client()
    if client is None:
        return {"deleted": [], "kept": [], "skipped": "s3 not configured"}

    retain_n = retain if retain is not None else _snapshot_retain()
    floor_hours = min_age_hours if min_age_hours is not None else _snapshot_min_age_hours()

    bundles = list_snapshots()  # newest-first
    if len(bundles) <= retain_n:
        return {"deleted": [], "kept": [b["key"] for b in bundles]}

    import time as _time
    now = _time.time()
    floor_sec = floor_hours * 3600

    keep: list[dict] = bundles[:retain_n]
    candidates_for_delete: list[dict] = bundles[retain_n:]

    deleted: list[str] = []
    kept_keys: list[str] = [b["key"] for b in keep]

    for b in candidates_for_delete:
        age = now - b["created_at"]
        if age < floor_sec:
            logger.info(
                "s3_store.rotate_snapshots: keeping %s (age %.1fh < floor %.1fh)",
                b["key"], age / 3600, floor_hours,
            )
            kept_keys.append(b["key"])
            continue
        try:
            _delete_prefix_recursive(client, bucket, b["key"] + "/")
            deleted.append(b["key"])
            logger.info(
                "s3_store.rotate_snapshots: deleted %s (age %.1fh)",
                b["key"], age / 3600,
            )
        except Exception as exc:
            logger.warning(
                "s3_store.rotate_snapshots: failed to delete %s: %s",
                b["key"], exc,
            )
            kept_keys.append(b["key"])

    return {"deleted": deleted, "kept": kept_keys}


def _delete_prefix_recursive(client, bucket: str, prefix: str) -> int:
    """Delete every object under *prefix*.  Returns count deleted."""
    paginator = client.get_paginator("list_objects_v2")
    keys: list[dict] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append({"Key": obj["Key"]})
    deleted = 0
    # delete_objects max batch = 1000
    while keys:
        batch, keys = keys[:1000], keys[1000:]
        try:
            client.delete_objects(Bucket=bucket, Delete={"Objects": batch})
            deleted += len(batch)
        except Exception:
            # Fallback: per-object delete
            for k in batch:
                try:
                    client.delete_object(Bucket=bucket, Key=k["Key"])
                    deleted += 1
                except Exception as exc:
                    logger.warning("s3_store: delete %s failed: %s", k["Key"], exc)
    return deleted


def ensure_bucket_lifecycle_policy() -> dict:
    """Best-effort: install a lifecycle policy on the bucket.

    Rules:
        - AbortIncompleteMultipartUpload after 7 days (cost hygiene).
        - Transition snapshot bundles older than 90 days to GLACIER.

    Skipped silently and logged at INFO when the IAM role lacks
    ``s3:PutLifecycleConfiguration`` — this is expected on locked-down
    deployments where bucket-level config is managed out of band.
    """
    bucket = _bucket()
    client = _make_client()
    if client is None:
        return {"applied": False, "reason": "s3 not configured"}

    config = {
        "Rules": [
            {
                "ID": "abort-stuck-multipart-uploads-7d",
                "Status": "Enabled",
                "Filter": {"Prefix": ""},
                "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7},
            },
            {
                "ID": "snapshots-glacier-after-90d",
                "Status": "Enabled",
                "Filter": {"Prefix": f"{_snapshots_prefix()}/"},
                "Transitions": [{"Days": 90, "StorageClass": "GLACIER"}],
            },
        ]
    }
    try:
        client.put_bucket_lifecycle_configuration(
            Bucket=bucket, LifecycleConfiguration=config,
        )
        logger.info("s3_store.lifecycle: applied policy to bucket %s", bucket)
        return {"applied": True, "rules": [r["ID"] for r in config["Rules"]]}
    except Exception as exc:
        # AccessDenied / NotImplemented (e.g. moto without lifecycle support)
        # are non-fatal — the snapshots still rotate via rotate_snapshots().
        logger.info(
            "s3_store.lifecycle: skipping bucket policy (%s) — rotate_snapshots() still active",
            exc,
        )
        return {"applied": False, "reason": str(exc)}


def get_snapshot_health() -> dict:
    """Snapshot-bundle health for ``GET /admin/s3/health`` + deploy doctor.

    Shape::

        {
            "last_successful_snapshot_at": float | None,   # epoch
            "age_seconds": float | None,
            "retained_count": int,
            "oldest_retained_at": float | None,
            "snapshot_key": str | None,                    # most recent
        }
    """
    import time as _time
    bundles = list_snapshots()
    last_at: float | None = None
    last_key: str | None = None
    if bundles:
        last_at = bundles[0]["created_at"] or None
        last_key = bundles[0]["key"]
    elif _LAST_SNAPSHOT_BUNDLE_AT is not None:
        last_at = _LAST_SNAPSHOT_BUNDLE_AT
        last_key = _LAST_SNAPSHOT_BUNDLE_KEY

    age = (_time.time() - last_at) if last_at else None
    oldest_at = bundles[-1]["created_at"] if bundles else None

    return {
        "last_successful_snapshot_at": last_at,
        "age_seconds": age,
        "retained_count": len(bundles),
        "oldest_retained_at": oldest_at,
        "snapshot_key": last_key,
    }
