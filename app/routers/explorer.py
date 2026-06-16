"""GET /explorer/info — report whether the graph is visualisable via LadybugDB Explorer.

The LadybugDB Explorer (github.com/LadybugDB/explorer) is the first-party
viewer for LadybugDB databases.  It runs as a Docker container and opens
``.db`` files via the ``LBUG_FILE`` environment variable.

This endpoint exists so TheForge UI can:

1. Check whether an index already exists on disk — no point launching a
   viewer against an empty DB.
2. Retrieve the exact shell command that spins up lbugdb/explorer pointed
   at the current ``LADYBUG_DB_PATH`` on the developer's machine.
3. Confirm which repos would be visible once the explorer loads.

The endpoint never *launches* anything — it is intentionally inert.  Starting
a browser process from a FastAPI service would be surprising behaviour for a
headless HTTP gateway.  The frontend copies the returned ``launch_command``
for the developer to run in a terminal.

Returns a ``ExplorerInfoResponse`` with:
    * ``available`` — True iff the LadybugDB file exists on disk.
    * ``db_path`` — absolute filesystem path to the LadybugDB file.
    * ``indexed_repos`` — repos currently represented in the graph.
    * ``launch_command`` — shell command to start lbugdb/explorer on port 7001.
    * ``viewer_url`` — URL to open once the explorer is running.
    * ``docs_url`` — LadybugDB Explorer GitHub page.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Query

from ..config import settings
from ..models import ExplorerInfoResponse
from .health import _get_indexed_repos

router = APIRouter()


# Default port for the launched LadybugDB Explorer UI.  7001 avoids collision
# with the Code Indexer Service (8003) and macOS ControlCenter (7000 / AirPlay).
_EXPLORER_PORT: int = 7001


def _launch_command(db_path: str, port: int = _EXPLORER_PORT) -> str:
    """Build the shell command to launch lbugdb/explorer against *db_path*.

    Uses the official ``lbugdb/explorer`` Docker image with read-only mode.
    Mounts the parent directory of the DB file as ``/database`` and passes
    the filename via ``LBUG_FILE``.

    Args:
        db_path: Absolute or relative path to the LadybugDB ``.db`` file.
        port: Host port to expose the explorer on (default 7001).

    Returns:
        str: A single-line shell command the user can paste into a terminal.
    """
    resolved = Path(db_path).resolve()
    mount_source = str(resolved.parent)
    mount_target = "/database"
    # --platform linux/amd64 forces Rosetta emulation on Apple Silicon so the
    # x86_64 vector extension (libvector.lbug_extension) is available.  Without
    # this flag the arm64 image variant is pulled and the extension is missing,
    # causing a startup error even though the explorer still partially works.
    return (
        f"docker run --rm --platform linux/amd64 -p {port}:8000 "
        f"-v {mount_source}:{mount_target} "
        f"-e LBUG_FILE={resolved.name} "
        f"-e MODE=READ_ONLY "
        f"ghcr.io/ladybugdb/explorer:latest"
    )


@router.get("/explorer/info", response_model=ExplorerInfoResponse)
def explorer_info(
    repo: str | None = Query(
        default=None,
        description="Repo slug to target.  When omitted, returns info for the "
                    "first available indexed repo (or empty when none exist).",
    ),
) -> ExplorerInfoResponse:
    """Report viewer availability + launch instructions for ONE indexed repo.

    Args:
        repo: Optional repo slug (matches ``indexed_repos`` names).  When
            given, the returned launch command opens that specific repo's
            DB file — so the explorer shows only that graph.  When omitted,
            defaults to the first indexed repo, or an ``available=False``
            response if nothing is indexed yet.

    Returns:
        ExplorerInfoResponse: Per-repo DB path, launch command, and the full
        list of indexed repos so the UI can render a picker.  When the DB
        has not been populated yet the response still succeeds but with
        ``available=False`` and ``indexed_repos=[]``.
    """
    indexed = _get_indexed_repos()

    # Pick the repo to target: explicit param > first indexed > none.
    target: str | None = None
    if repo and repo in indexed:
        target = repo
    elif indexed:
        target = indexed[0]

    if target is None:
        # Nothing indexed yet — return a still-useful launch command pointed
        # at the DB directory so the UI can render the "what to do next" copy.
        fallback_path = str(Path(settings.LADYBUG_DB_DIR) / "graph.db")
        return ExplorerInfoResponse(
            available=False,
            db_path=fallback_path,
            indexed_repos=indexed,
            launch_command=_launch_command(fallback_path),
            viewer_url=f"http://localhost:{_EXPLORER_PORT}",
            docs_url="https://github.com/LadybugDB/explorer",
        )

    db_path = settings.db_path_for_repo(target)
    exists = Path(db_path).exists()

    return ExplorerInfoResponse(
        available=exists,
        db_path=db_path,
        indexed_repos=indexed,
        launch_command=_launch_command(db_path),
        viewer_url=f"http://localhost:{_EXPLORER_PORT}",
        docs_url="https://github.com/LadybugDB/explorer",
    )
