"""GET /disk-usage — capacity gauge for the .cgr/repos directory.

Powers the disk-usage gauge in TheForge Settings → Source control & Code
Indexer (BACKEND_HANDOVER §2.11). The gauge color-codes based on the
``used / (used + free)`` ratio.

Returns the disk-usage data even when ``LADYBUG_DB_DIR`` does not yet
exist — ``used_bytes = 0`` and ``free_bytes`` is the host filesystem's
free space (computed against the parent that DOES exist). The frontend
renders a "0% — 0 B / X" gauge in that case rather than an empty state.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..models import DiskUsageResponse

router = APIRouter(tags=["disk"])
logger = logging.getLogger(__name__)


def _dir_size_bytes(root: Path) -> int:
    """Recursively sum file sizes under ``root``.

    Uses ``os.scandir`` via ``rglob`` for portability. Symlinked targets
    are NOT followed — we only count bytes physically inside the dir.
    Permission errors on individual entries are swallowed; the caller
    still gets a partial-but-useful figure.
    """
    total = 0
    if not root.is_dir():
        return 0
    for p in root.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except (OSError, PermissionError):
            continue
    return total


def _existing_ancestor(p: Path) -> Path:
    """Walk up from ``p`` until we hit a directory that exists.

    ``shutil.disk_usage`` requires an existing path. The handler may run
    before any repo has been indexed (so ``LADYBUG_DB_DIR`` doesn't
    exist), so we walk up until we find something stat-able. The root
    ``/`` always exists, so this terminates.
    """
    cur = p.resolve() if p.is_absolute() else p.absolute()
    while not cur.exists():
        parent = cur.parent
        if parent == cur:  # hit filesystem root
            break
        cur = parent
    return cur


@router.get("/disk-usage", response_model=DiskUsageResponse)
def disk_usage() -> DiskUsageResponse:
    """Return disk usage under ``LADYBUG_DB_DIR`` plus host free space.

    Returns:
        DiskUsageResponse: ``used_bytes`` is the recursive sum of file
        sizes under the indexer's data directory; ``free_bytes`` is the
        free space available on the filesystem hosting that directory.

    Raises:
        HTTPException: 500 only when the host filesystem itself is
            unreadable (extremely unlikely — a stat() failure on ``/``).
    """
    db_dir = Path(settings.LADYBUG_DB_DIR)
    used = _dir_size_bytes(db_dir)

    try:
        usage = shutil.disk_usage(str(_existing_ancestor(db_dir)))
        free = int(usage.free)
    except OSError as exc:
        logger.warning("disk_usage probe failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"disk_usage probe failed: {exc}",
        ) from exc

    return DiskUsageResponse(used_bytes=int(used), free_bytes=free)
