"""Admin S3 endpoints (BUC-1555 + BUC-1555b).

POST /admin/s3/snapshot   — manual per-file snapshot trigger (legacy, BUC-1555)
POST /admin/s3/snapshot-bundle — write a timestamped bundle + rotate
POST /admin/s3/restore    — restore a bundle (latest, or by snapshot_key)
GET  /admin/s3/snapshots  — list available bundles
GET  /admin/s3/health     — last successful snapshot age + retained count
                             (probed by TheForge `pnpm doctor`)

All endpoints are best-effort and return HTTP 200 with an ``ok`` field;
upstream callers (deploy doctor, dashboards) read ``ok`` and ``error``
to render status without trapping HTTP errors.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from ..config import settings
from ..services.s3_restore import restore_latest, restore_specific
from ..services.s3_store import (
    get_snapshot_health,
    list_snapshots,
    snapshot_bundle,
    snapshot_indexes,
)

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
