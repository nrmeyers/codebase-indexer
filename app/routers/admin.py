"""POST /admin/s3/snapshot — manual S3 index snapshot trigger (BUC-1555).

Allows operators to manually trigger a snapshot of local index files to S3
outside the normal lifespan shutdown + periodic background task.  Useful for
testing and for on-demand backups when you know you've made changes.

Response always includes:
    ok: bool — true if the snapshot attempt succeeded (0 or more files pushed)
    files_pushed: int — number of .db / .duck files uploaded (0 if all current)
    error: str | None — error message if something failed during the snapshot
"""
from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel

from ..config import settings
from ..services.s3_store import snapshot_indexes

router = APIRouter(prefix="/admin", tags=["admin"])
logger = logging.getLogger(__name__)


class AdminS3SnapshotResponse(BaseModel):
    """Response for ``POST /admin/s3/snapshot``."""

    ok: bool
    files_pushed: int
    error: str | None = None


@router.post("/s3/snapshot", response_model=AdminS3SnapshotResponse)
def admin_trigger_s3_snapshot() -> AdminS3SnapshotResponse:
    """Manually trigger an S3 snapshot of local index files.

    Pushes any changed .db / .duck files from LADYBUG_DB_DIR to S3.
    Returns immediately with the count of files uploaded.

    Useful for testing, pre-deployment backups, or debugging S3 sync issues.
    Non-fatal failures in individual file uploads are logged but don't block
    the overall snapshot from completing.

    Returns:
        AdminS3SnapshotResponse: ok=True when the snapshot completed
        (even if 0 files needed pushing), ok=False on fatal errors.
    """
    try:
        count = snapshot_indexes(settings.LADYBUG_DB_DIR)
        return AdminS3SnapshotResponse(ok=True, files_pushed=count, error=None)
    except Exception as exc:
        error_msg = str(exc)
        logger.error("admin_trigger_s3_snapshot failed: %s", error_msg)
        return AdminS3SnapshotResponse(ok=False, files_pushed=0, error=error_msg)
