"""GET /explorer/info — report whether the graph is visualisable via Kuzu Explorer.

LadybugDB (the fork's embedded graph store) is kuzu-compatible on disk, so any
existing kuzu-explorer instance can open our ``.db`` file directly.  This
endpoint exists so TheForge UI (and any curious developer) can:

1. Check whether an index already exists on disk — no point launching a
   viewer against an empty DB.
2. Retrieve the exact shell command that would spin up kuzu-explorer pointed
   at the current ``LADYBUG_DB_PATH`` on the developer's machine.
3. Confirm which repos would be visible once the explorer loads.

The endpoint never *launches* anything — it is intentionally inert.  Starting
a browser process from a FastAPI service would be surprising behaviour for a
headless HTTP gateway.  Instead, the frontend (or a CLI helper) is expected
to pick up the returned ``launch_command`` and exec it locally.

Returns a ``ExplorerInfoResponse`` with:
    * ``available`` — True iff the LadybugDB file exists on disk.
    * ``db_path`` — absolute filesystem path to the LadybugDB file.
    * ``indexed_repos`` — repos currently represented in the graph.
    * ``launch_command`` — shell command to start kuzu-explorer on port 7000.
    * ``viewer_url`` — URL to open once the explorer is running.
    * ``docs_url`` — upstream kuzu-explorer documentation link.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from ..config import settings
from ..models import ExplorerInfoResponse
from .health import _get_indexed_repos

router = APIRouter()


# Default port for the launched kuzu-explorer UI.  7000 is used instead of
# 8000 so it doesn't collide with the Code Indexer Service itself.
_EXPLORER_PORT: int = 7000


def _launch_command(db_path: str, port: int = _EXPLORER_PORT) -> str:
    """Build the shell command to launch kuzu-explorer against *db_path*.

    We use the official ``kuzudb/explorer`` Docker image — it is the only
    first-party distribution of the viewer, and since it runs read-only
    against a mounted volume it never writes to the DB.  The command mounts
    the **parent directory** of the DB file so the explorer can navigate
    sibling files (kuzu stores the DB as a directory of files, not a single
    blob, so the mount must be the directory itself).

    Args:
        db_path: Absolute or relative path to the LadybugDB file/directory.
        port: Host port to expose the explorer on.

    Returns:
        str: A single-line shell command the user can paste into a terminal.
    """
    resolved = Path(db_path).resolve()
    mount_source = str(resolved.parent)
    mount_target = "/database"
    return (
        f"docker run --rm -p {port}:8000 "
        f"-v {mount_source}:{mount_target} "
        f"-e KUZU_PATH={mount_target}/{resolved.name} "
        f"kuzudb/explorer:latest"
    )


@router.get("/explorer/info", response_model=ExplorerInfoResponse)
def explorer_info() -> ExplorerInfoResponse:
    """Report viewer availability + launch instructions for the current DB.

    Returns:
        ExplorerInfoResponse: ``available=True`` when the LadybugDB file
        exists on disk, plus the launch command and viewer URL so a caller
        (TheForge UI or a developer CLI) can open the graph.  When the DB
        has not been populated yet the response still succeeds but with
        ``available=False`` and ``indexed_repos=[]``.
    """
    db_path = settings.LADYBUG_DB_PATH
    exists = Path(db_path).exists()
    indexed = _get_indexed_repos() if exists else []

    return ExplorerInfoResponse(
        available=exists and len(indexed) > 0,
        db_path=db_path,
        indexed_repos=indexed,
        launch_command=_launch_command(db_path),
        viewer_url=f"http://localhost:{_EXPLORER_PORT}",
        docs_url="https://docs.kuzudb.com/visualization/kuzu-explorer/",
    )
