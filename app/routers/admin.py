"""Admin S3 endpoints (BUC-1555 + BUC-1555b) and slug migration (BUC-1580).

POST /admin/s3/snapshot   — manual per-file snapshot trigger (legacy, BUC-1555)
POST /admin/s3/snapshot-bundle — write a timestamped bundle + rotate
POST /admin/s3/restore    — restore a bundle (latest, or by snapshot_key)
GET  /admin/s3/snapshots  — list available bundles
GET  /admin/s3/health     — last successful snapshot age + retained count
                             (probed by TheForge `pnpm doctor`)
POST /admin/migrate-slugs — rename bare-basename slugs to canonical
                             ``{org}__{repo}`` form (BUC-1580)

All endpoints are best-effort and return HTTP 200 with an ``ok`` field;
upstream callers (deploy doctor, dashboards) read ``ok`` and ``error``
to render status without trapping HTTP errors.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from ..config import settings, slugify_repo
from ..services import jobs_store
from ..services.s3_restore import restore_latest, restore_specific
from ..services.s3_store import (
    get_snapshot_health,
    list_snapshots,
    snapshot_bundle,
    snapshot_indexes,
)
from ..services.slug import derive_slug

router = APIRouter(prefix="/admin", tags=["admin"])
logger = logging.getLogger(__name__)


class AdminS3SnapshotResponse(BaseModel):
    """Response for ``POST /admin/s3/snapshot`` (per-file legacy)."""

    ok: bool
    files_pushed: int
    error: str | None = None


class AdminS3BundleResponse(BaseModel):
    """Response for ``POST /admin/s3/snapshot-bundle``."""

    ok: bool
    snapshot_key: str
    files_uploaded: int
    bytes: int
    error: str | None = None
    rotation: dict


class AdminS3RestoreRequest(BaseModel):
    """Body for ``POST /admin/s3/restore`` — both fields optional."""

    snapshot_key: Optional[str] = None


class AdminS3RestoreResponse(BaseModel):
    """Response for ``POST /admin/s3/restore``."""

    ok: bool
    files_restored: int
    bytes: int
    snapshot_key: str
    error: str | None = None
    skipped: list[str] = []
    verified: list[str] = []


class SnapshotEntry(BaseModel):
    """One entry in the ``GET /admin/s3/snapshots`` list."""

    key: str
    size_bytes: int
    created_at: float
    file_count: int


class AdminS3HealthResponse(BaseModel):
    """Shape consumed by TheForge ``pnpm doctor`` and dashboards."""

    last_successful_snapshot_at: float | None
    age_seconds: float | None
    retained_count: int
    oldest_retained_at: float | None
    snapshot_key: str | None


@router.post("/s3/snapshot", response_model=AdminS3SnapshotResponse)
def admin_trigger_s3_snapshot() -> AdminS3SnapshotResponse:
    """Manually trigger a per-file S3 snapshot of local index files.

    Pushes any changed .db / .duck files from LADYBUG_DB_DIR to S3.
    Returns immediately with the count of files uploaded.

    Useful for testing, pre-deployment backups, or debugging S3 sync issues.
    Non-fatal failures in individual file uploads are logged but don't block
    the overall snapshot from completing.
    """
    try:
        count = snapshot_indexes(settings.LADYBUG_DB_DIR)
        return AdminS3SnapshotResponse(ok=True, files_pushed=count, error=None)
    except Exception as exc:
        error_msg = str(exc)
        logger.error("admin_trigger_s3_snapshot failed: %s", error_msg)
        return AdminS3SnapshotResponse(ok=False, files_pushed=0, error=error_msg)


@router.post("/s3/snapshot-bundle", response_model=AdminS3BundleResponse)
def admin_snapshot_bundle() -> AdminS3BundleResponse:
    """Write a timestamped snapshot bundle to S3 + rotate old bundles.

    Bundle layout:
        s3://{bucket}/{prefix}/snapshots/<UTC-ISO8601>/
            <repo>.db
            <repo>.duck
            index.json   (sha256 + size manifest)

    After the bundle uploads cleanly, rotation deletes everything older
    than ``S3_SNAPSHOT_RETAIN`` (default 10) — except snapshots younger
    than 24h, which are protected by the safety floor.
    """
    try:
        result = snapshot_bundle(settings.LADYBUG_DB_DIR)
        return AdminS3BundleResponse(**result)
    except Exception as exc:
        error_msg = str(exc)
        logger.error("admin_snapshot_bundle failed: %s", error_msg)
        return AdminS3BundleResponse(
            ok=False,
            snapshot_key="",
            files_uploaded=0,
            bytes=0,
            error=error_msg,
            rotation={"deleted": [], "kept": []},
        )


@router.post("/s3/restore", response_model=AdminS3RestoreResponse)
def admin_restore_snapshot(req: AdminS3RestoreRequest | None = None) -> AdminS3RestoreResponse:
    """Restore a snapshot bundle into LADYBUG_DB_DIR.

    Body:
        ``{ "snapshot_key": "<full prefix>" }`` to restore a specific bundle,
        or ``{}`` / no body to restore the most recent bundle.

    Each file is verified against the bundle's ``index.json`` SHA-256.
    A mismatch aborts the restore and returns ``ok=False`` with a partial
    file count — the caller can decide whether to retry or fall back.
    """
    try:
        target = settings.LADYBUG_DB_DIR
        if req and req.snapshot_key:
            result = restore_specific(None, None, req.snapshot_key, target)
        else:
            result = restore_latest(None, None, target)
        return AdminS3RestoreResponse(**result.to_dict())
    except Exception as exc:
        error_msg = str(exc)
        logger.error("admin_restore_snapshot failed: %s", error_msg)
        return AdminS3RestoreResponse(
            ok=False,
            files_restored=0,
            bytes=0,
            snapshot_key="",
            error=error_msg,
        )


@router.get("/s3/snapshots", response_model=list[SnapshotEntry])
def admin_list_snapshots() -> list[SnapshotEntry]:
    """List timestamped snapshot bundles, newest first."""
    try:
        return [SnapshotEntry(**b) for b in list_snapshots()]
    except Exception as exc:
        logger.error("admin_list_snapshots failed: %s", exc)
        return []


@router.get("/s3/health", response_model=AdminS3HealthResponse)
def admin_s3_health() -> AdminS3HealthResponse:
    """Snapshot-bundle health probe.

    Consumed by TheForge ``pnpm doctor`` to alert when the indexer hasn't
    successfully snapshotted in a long time.  All fields are nullable when
    the indexer has never produced a bundle (fresh deploy).
    """
    try:
        return AdminS3HealthResponse(**get_snapshot_health())
    except Exception as exc:
        logger.error("admin_s3_health failed: %s", exc)
        return AdminS3HealthResponse(
            last_successful_snapshot_at=None,
            age_seconds=None,
            retained_count=0,
            oldest_retained_at=None,
            snapshot_key=None,
        )


# ---------------------------------------------------------------------------
# BUC-1580 — canonical slug migration
# ---------------------------------------------------------------------------


class SlugMigrationEntry(BaseModel):
    """One entry in the ``POST /admin/migrate-slugs`` response."""

    old: str
    new: str
    files_renamed: list[str] = []
    s3_renamed: list[str] = []
    jobs_updated: int = 0


class SlugMigrationSkipped(BaseModel):
    """Reason a slug was not migrated."""

    slug: str
    reason: str


class SlugMigrationResponse(BaseModel):
    """Response for ``POST /admin/migrate-slugs``."""

    ok: bool
    migrated: list[SlugMigrationEntry]
    skipped: list[SlugMigrationSkipped]
    error: str | None = None


def _rename_local_artifacts(
    db_dir: Path, old_slug: str, new_slug: str
) -> list[str]:
    """Atomically rename per-slug local artefacts ``.db``, ``.duck``, ``.tantivy``.

    Args:
        db_dir: ``settings.LADYBUG_DB_DIR`` resolved as a Path.
        old_slug: Current on-disk slug (filename stem).
        new_slug: Target canonical slug.

    Returns:
        List of artefact paths that were actually renamed.

    Raises:
        FileExistsError: If a destination already exists.  Caller MUST
            check for collisions before invoking this helper.
    """
    moved: list[str] = []

    # Rename the canonical pair plus DuckDB sidecars (.wal / .tmp) that may
    # exist after a crash.  Tantivy is a directory.
    artefacts: list[tuple[Path, Path]] = [
        (db_dir / f"{old_slug}.db", db_dir / f"{new_slug}.db"),
        (db_dir / f"{old_slug}.db-wal", db_dir / f"{new_slug}.db-wal"),
        (db_dir / f"{old_slug}.db-shm", db_dir / f"{new_slug}.db-shm"),
        (db_dir / f"{old_slug}.duck", db_dir / f"{new_slug}.duck"),
        (db_dir / f"{old_slug}.duck.wal", db_dir / f"{new_slug}.duck.wal"),
        (db_dir / f"{old_slug}.duck.tmp", db_dir / f"{new_slug}.duck.tmp"),
        (db_dir / f"{old_slug}.tantivy", db_dir / f"{new_slug}.tantivy"),
    ]

    # Pre-flight: verify no destination exists.  Detected collisions surface
    # to the caller as FileExistsError BEFORE any rename runs, so we never
    # leave the filesystem half-migrated.
    for _src, dst in artefacts:
        if dst.exists():
            raise FileExistsError(str(dst))

    for src, dst in artefacts:
        if not src.exists():
            continue
        src.rename(dst)
        moved.append(dst.name)
    return moved


def _rename_s3_objects(old_slug: str, new_slug: str) -> list[str]:
    """Rename S3 objects by copying then deleting (S3 has no atomic move).

    Best-effort: returns an empty list when the bucket is unconfigured or
    the operation fails — local migration still wins, the operator can
    retry the S3 leg manually.

    Args:
        old_slug: Current on-disk slug (key stem under ``S3_INDEX_PREFIX``).
        new_slug: Target canonical slug.

    Returns:
        List of new S3 keys that were written.  Empty when no S3 work
        was performed.
    """
    bucket = (settings.S3_INDEX_BUCKET or "").strip()
    if not bucket:
        return []
    prefix = (settings.S3_INDEX_PREFIX or "").strip().rstrip("/")
    region = settings.S3_INDEX_REGION or "us-east-1"

    try:
        import boto3  # type: ignore[import-untyped]
    except ImportError:
        logger.info("admin.migrate-slugs: boto3 unavailable — skipping S3 leg")
        return []

    client = boto3.client("s3", region_name=region)
    renamed: list[str] = []

    try:
        paginator = client.get_paginator("list_objects_v2")
        # All keys directly under the prefix that start with old_slug. so
        # the per-file snapshot artefacts (`{slug}.db`, `{slug}.duck`) move
        # but timestamped snapshot bundles under `{prefix}/snapshots/...`
        # do NOT — those are retained as-is (they're forensic captures).
        candidates: list[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/{old_slug}"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Guard against ``{old_slug}-foo`` / ``{old_slug}_archive``
                # by requiring the next char to be ``.`` (matches .db,
                # .duck, .tantivy/, .db-wal, .db-shm).  Mirrors the local
                # artefact list above.
                tail = key[len(f"{prefix}/{old_slug}"):]
                if not tail or tail[0] not in (".", "-"):
                    continue
                candidates.append(key)

        for src_key in candidates:
            dst_key = f"{prefix}/{new_slug}{src_key[len(f'{prefix}/{old_slug}'):]}"
            client.copy_object(
                Bucket=bucket,
                CopySource={"Bucket": bucket, "Key": src_key},
                Key=dst_key,
            )
            client.delete_object(Bucket=bucket, Key=src_key)
            renamed.append(dst_key)
    except Exception as exc:
        logger.warning("admin.migrate-slugs: S3 rename failed for %s — %s", old_slug, exc)
    return renamed


@router.post("/migrate-slugs", response_model=SlugMigrationResponse)
def admin_migrate_slugs() -> SlugMigrationResponse:
    """Rename bare-basename slugs to canonical ``{org}__{repo}`` form (BUC-1580).

    Iterates every indexed repo (any ``.db`` file under ``LADYBUG_DB_DIR``
    plus the in-memory registry), looks up its ``root_path`` from the
    DuckDB sidecar, and runs :func:`derive_slug` to compute the canonical
    slug.  When the canonical slug differs from the on-disk slug:

        * Refuses migration when the canonical slug already exists with
          content (operator must manually decide which copy to keep).
        * Otherwise renames local artefacts (``.db``, ``.db-wal``,
          ``.db-shm``, ``.duck``, ``.duck.wal``, ``.duck.tmp``,
          ``.tantivy/``), copies S3 objects to the new key and deletes
          the old, and rewrites ``repo_slug`` rows in the jobs store.

    Returns:
        SlugMigrationResponse with two lists: ``migrated`` (one entry per
        renamed slug, with the artefact paths actually moved) and
        ``skipped`` (slug + reason for every repo that was left alone).
    """
    db_dir = Path(settings.LADYBUG_DB_DIR)
    migrated: list[SlugMigrationEntry] = []
    skipped: list[SlugMigrationSkipped] = []

    if not db_dir.exists():
        return SlugMigrationResponse(ok=True, migrated=[], skipped=[])

    # Lazy import to avoid the circular import (admin → routers/index → admin).
    from .index import _read_meta, indexed_repo_paths, indexed_repos

    # Build the candidate set: anything with a ``.db`` file on disk plus
    # the in-memory registry.
    on_disk_slugs: set[str] = {p.stem for p in db_dir.glob("*.db")}
    candidate_slugs = sorted(on_disk_slugs | set(indexed_repos))

    # Track new slugs we've already routed in this run so we can detect
    # in-batch collisions where two old slugs map to the same canonical.
    claimed_new: set[str] = set()

    for old_slug in candidate_slugs:
        # Resolve root_path: in-memory cache wins (most current), DuckDB
        # ``repo_metadata`` is the persistent fallback.
        root_path = indexed_repo_paths.get(old_slug)
        if not root_path:
            try:
                root_path = (_read_meta(old_slug) or {}).get("root_path") or ""
            except Exception:
                root_path = ""
        if not root_path:
            skipped.append(SlugMigrationSkipped(
                slug=old_slug,
                reason="no root_path on record — cannot probe git remote",
            ))
            continue

        try:
            new_slug = derive_slug(Path(root_path), old_slug)
        except Exception as exc:
            skipped.append(SlugMigrationSkipped(
                slug=old_slug,
                reason=f"derive_slug failed: {exc}",
            ))
            continue

        # Already canonical — nothing to do.
        if new_slug == old_slug or new_slug == slugify_repo(old_slug):
            skipped.append(SlugMigrationSkipped(
                slug=old_slug, reason="already canonical",
            ))
            continue

        # Collision detection: refuse migration when a *different* indexed
        # repo already owns the canonical slug.  The operator must decide
        # which to keep (typically the App-clone, since it's authoritative)
        # and delete the duplicate via DELETE /index/{slug} before retrying.
        new_db = db_dir / f"{new_slug}.db"
        new_duck = db_dir / f"{new_slug}.duck"
        if (
            (new_db.exists() and new_db.stat().st_size > 0)
            or (new_duck.exists() and new_duck.stat().st_size > 0)
            or new_slug in claimed_new
        ):
            skipped.append(SlugMigrationSkipped(
                slug=old_slug,
                reason=(
                    f"canonical slug '{new_slug}' already exists with content — "
                    "delete the duplicate manually before retrying"
                ),
            ))
            continue

        # Perform the rename.  Local first, then S3, then jobs_store.  Any
        # exception aborts THIS slug only — others continue.
        try:
            files_renamed = _rename_local_artifacts(db_dir, old_slug, new_slug)
        except FileExistsError as exc:
            skipped.append(SlugMigrationSkipped(
                slug=old_slug,
                reason=f"destination artefact already present: {exc}",
            ))
            continue
        except OSError as exc:
            skipped.append(SlugMigrationSkipped(
                slug=old_slug,
                reason=f"local rename failed: {exc}",
            ))
            continue

        s3_renamed = _rename_s3_objects(old_slug, new_slug)

        try:
            jobs_updated = jobs_store.rename_repo_slug(old_slug, new_slug)
        except Exception as exc:
            logger.warning(
                "admin.migrate-slugs: jobs_store rename failed for %s — %s",
                old_slug, exc,
            )
            jobs_updated = 0

        # Update in-memory registry so the next /repos call reflects the move.
        try:
            indexed_repos.discard(old_slug)
            indexed_repos.add(new_slug)
            if old_slug in indexed_repo_paths:
                indexed_repo_paths[new_slug] = indexed_repo_paths.pop(old_slug)
        except Exception:
            pass  # best-effort

        claimed_new.add(new_slug)
        migrated.append(SlugMigrationEntry(
            old=old_slug,
            new=new_slug,
            files_renamed=files_renamed,
            s3_renamed=s3_renamed,
            jobs_updated=jobs_updated,
        ))

    return SlugMigrationResponse(ok=True, migrated=migrated, skipped=skipped)
