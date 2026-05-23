#!/usr/bin/env python3
"""LE-111 — best-effort backfill of ``last_indexed_sha`` in repo_metadata.

For every per-repo ``.duck`` file under ``LADYBUG_DB_DIR``:

1. Read ``root_path`` from the ``repo_metadata`` table.
2. If ``last_indexed_sha`` is already set, skip.
3. Otherwise run ``git rev-parse HEAD`` against ``root_path`` and persist
   the result. Failures (missing path, non-git checkout) are logged and
   skipped — they record nothing rather than guessing.

The backfill SHA is best-effort: the historical index ran against an
unknown HEAD that may have since moved. The captured SHA is at-least
"a SHA from the indexed working tree" — useful as a seed value so
``/repos`` drift detection has *something* to compare against, and the
next ``POST /repos/{name}/reindex`` will overwrite it with an
index-time SHA.

Usage
-----

    # Dry-run for every repo:
    python scripts/backfill-last-indexed-sha.py --dry-run

    # Apply for every repo:
    python scripts/backfill-last-indexed-sha.py

    # Apply for a single repo:
    python scripts/backfill-last-indexed-sha.py --repo TheForge
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("backfill-sha")


def _capture_head_sha(repo_path: str) -> str | None:
    """Return ``git rev-parse HEAD`` for ``repo_path`` or None on any failure."""
    if not repo_path:
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if result.returncode == 0:
            sha = result.stdout.strip()
            return sha or None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("git rev-parse failed for %s: %s", repo_path, exc)
    return None


def _backfill_one(duck_path: Path, dry_run: bool) -> str:
    """Backfill SHA for one ``.duck`` file.

    Returns one of: ``skipped-already-set``, ``skipped-no-root-path``,
    ``skipped-no-sha``, ``would-update``, ``updated``.
    """
    try:
        from codebase_rag.storage.vector_store import (  # type: ignore[import-untyped]
            open_or_create,
            read_all_metadata,
            write_metadata,
        )
    except ImportError as exc:
        logger.error("codebase_rag not installed: %s", exc)
        return "error-no-codebase-rag"

    conn = open_or_create(str(duck_path))
    try:
        meta = read_all_metadata(conn)
        existing_sha = meta.get("last_indexed_sha")
        if isinstance(existing_sha, str) and existing_sha.strip():
            logger.info("%s: already has last_indexed_sha=%s", duck_path.stem, existing_sha[:8])
            return "skipped-already-set"

        root_path = meta.get("root_path")
        if not (isinstance(root_path, str) and root_path):
            logger.info("%s: no root_path; cannot backfill", duck_path.stem)
            return "skipped-no-root-path"

        sha = _capture_head_sha(root_path)
        if not sha:
            logger.info("%s: git rev-parse failed at %s; cannot backfill", duck_path.stem, root_path)
            return "skipped-no-sha"

        if dry_run:
            logger.info("%s: would set last_indexed_sha=%s (root_path=%s)", duck_path.stem, sha[:8], root_path)
            return "would-update"

        write_metadata(conn, last_indexed_sha=sha)
        logger.info("%s: set last_indexed_sha=%s", duck_path.stem, sha[:8])
        return "updated"
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill last_indexed_sha in repo_metadata.")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing.")
    parser.add_argument("--repo", default=None, help="Restrict to one repo slug.")
    parser.add_argument(
        "--db-dir",
        default=None,
        help="Override LADYBUG_DB_DIR (defaults to the value from app.config).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.db_dir:
        db_dir = Path(args.db_dir)
    else:
        from app.config import settings  # type: ignore[import-not-found]
        db_dir = Path(settings.LADYBUG_DB_DIR)

    if not db_dir.exists():
        logger.error("LADYBUG_DB_DIR not found: %s", db_dir)
        return 2

    if args.repo:
        duck_files = [db_dir / f"{args.repo}.duck"]
        if not duck_files[0].exists():
            logger.error("No .duck file for repo %s at %s", args.repo, duck_files[0])
            return 2
    else:
        duck_files = sorted(db_dir.glob("*.duck"))

    counts: dict[str, int] = {}
    for duck in duck_files:
        outcome = _backfill_one(duck, dry_run=args.dry_run)
        counts[outcome] = counts.get(outcome, 0) + 1

    logger.info("summary: %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
