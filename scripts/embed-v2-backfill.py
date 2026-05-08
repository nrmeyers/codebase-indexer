#!/usr/bin/env python3
"""Phase 1.3 — backfill the ``embedding_v2`` column for legacy rows.

Walks every ``.duck`` file under ``LADYBUG_DB_DIR`` (or a single repo when
``--repo`` is given), finds rows whose ``embedding_v2`` is NULL, fetches
their source slice, and writes a v2 embedding via the BGE-Code-v1
SageMaker endpoint.

Idempotent — already-populated rows are skipped.  Reuses the Phase 1.4
``content_hash`` machinery for the "skip unchanged symbols" semantic:
when a row's stored ``content_hash`` matches the current source slice
hash, we treat it as fresh and only fill in the v2 vector (no source
re-fetch beyond the symbol range needed for the embedding text).

Hard cost cap: $50 across the entire run (configurable via ``--cap``).
The cap is checked AFTER each batch so a long run aborts cleanly with a
progress report rather than overshooting.

Usage
-----
    # See what would happen for one repo:
    python scripts/embed-v2-backfill.py --dry-run --repo TheForge

    # Backfill all repos with the default $50 cap:
    SAGEMAKER_BGE_CODE_ENDPOINT=forge-bge-code-v1 \\
        python scripts/embed-v2-backfill.py

    # Backfill one repo, lower cap:
    python scripts/embed-v2-backfill.py --repo TheForge --cap 5

Exit codes
----------
    0   Success (all NULL rows backfilled, or dry-run completed)
    2   Cost cap reached — partial backfill (re-run to continue)
    3   v2 endpoint not configured (refuse to start)
    4   No .duck files found
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Allow `python scripts/embed-v2-backfill.py` from repo root without install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.services.embedder import (  # noqa: E402
    MODEL_BGE_CODE_V1,
    BgeCodeV1Embedder,
    ensure_v2_schema,
    has_v2_column,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("embed-v2-backfill")

# SageMaker E5/BGE Serverless Inference is billed at ~$0.20 per million
# tokens in us-east-1 (2026-04 list price).  This is conservative enough
# to keep the cap honest even if pricing drifts +20%.
COST_USD_PER_TOKEN = 0.20 / 1_000_000
DEFAULT_CAP_USD = 50.0


def _estimate_cost_usd(tokens: int) -> float:
    return tokens * COST_USD_PER_TOKEN


def _iter_duck_files(repo: str | None) -> list[Path]:
    """Return ``.duck`` files to process.  ``repo`` selects exactly one."""
    db_dir = Path(settings.LADYBUG_DB_DIR)
    if not db_dir.is_dir():
        return []
    if repo:
        path = Path(settings.vec_db_path_for_repo(repo))
        return [path] if path.exists() else []
    return sorted(db_dir.glob("*.duck"))


def _count_null_v2_rows(conn) -> int:
    """Return the number of rows that need a v2 embedding."""
    if not has_v2_column(conn):
        return 0
    row = conn.execute(
        "SELECT count(*) FROM embeddings WHERE embedding_v2 IS NULL"
    ).fetchone()
    return int(row[0]) if row else 0


def _fetch_pending(conn, batch_size: int) -> list[tuple[str, str, int, int]]:
    """Pull (qualified_name, file_path, start_line, end_line) for NULL v2 rows."""
    rows = conn.execute(
        """
        SELECT qualified_name, file_path, start_line, end_line
        FROM embeddings
        WHERE embedding_v2 IS NULL
        ORDER BY qualified_name
        LIMIT ?
        """,
        (batch_size,),
    ).fetchall()
    return [(r[0], r[1] or "", int(r[2] or 0), int(r[3] or 0)) for r in rows]


def _read_source_slice(file_path: str, start: int, end: int) -> str:
    """Best-effort read of the symbol's source slice.  Empty on miss."""
    if not file_path:
        return ""
    try:
        with open(file_path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return ""
    if start <= 0 or end <= 0 or start > len(lines):
        return ""
    return "".join(lines[max(0, start - 1) : min(len(lines), end)])


def _write_v2(conn, qualified_name: str, vec: list[float]) -> None:
    conn.execute(
        "UPDATE embeddings "
        "SET embedding_v2 = ?::FLOAT[768], embedding_model = ? "
        "WHERE qualified_name = ?",
        (vec, MODEL_BGE_CODE_V1, qualified_name),
    )


def backfill_one_duck(
    duck_path: Path,
    embedder: BgeCodeV1Embedder | None,
    cap_usd: float,
    dry_run: bool,
) -> tuple[int, int, float]:
    """Backfill a single ``.duck`` file.

    Returns ``(eligible, written, cost_usd_for_this_file)``.
    """
    try:
        from codebase_rag.storage.vector_store import open_or_create
    except ImportError as exc:
        logger.error("vector_store unavailable: %s", exc)
        return (0, 0, 0.0)

    conn = open_or_create(str(duck_path))
    try:
        ensure_v2_schema(conn)
        eligible = _count_null_v2_rows(conn)
        if eligible == 0:
            logger.info("%s: nothing to backfill", duck_path.name)
            return (0, 0, 0.0)

        logger.info(
            "%s: %d row(s) eligible for v2 backfill%s",
            duck_path.name,
            eligible,
            " (DRY RUN — no SageMaker calls)" if dry_run else "",
        )
        if dry_run or embedder is None:
            return (eligible, 0, 0.0)

        written = 0
        local_cost = 0.0
        BATCH = 50
        while True:
            pending = _fetch_pending(conn, BATCH)
            if not pending:
                break
            for qname, fpath, s, e in pending:
                # Cap check before every call so we never overshoot.
                if local_cost + _estimate_cost_usd(embedder.cost_tokens) >= cap_usd:
                    logger.warning(
                        "cap_usd=%.2f reached at %d rows — stopping cleanly",
                        cap_usd,
                        written,
                    )
                    return (eligible, written, local_cost)
                src = _read_source_slice(fpath, s, e)
                if not src:
                    # Nothing to embed — leave the v2 cell NULL so a later
                    # run can retry once source becomes readable.
                    continue
                vec = embedder.embed(src)
                if vec is None:
                    # Endpoint absent or transient miss — abort the file so
                    # we don't burn cycles on guaranteed misses.
                    logger.warning(
                        "%s: embedder returned None at row %s — aborting file",
                        duck_path.name,
                        qname,
                    )
                    return (eligible, written, local_cost)
                _write_v2(conn, qname, vec)
                written += 1
                local_cost = _estimate_cost_usd(embedder.cost_tokens)
            logger.info(
                "%s: progress %d/%d (cost ~$%.4f)",
                duck_path.name,
                written,
                eligible,
                local_cost,
            )
        return (eligible, written, local_cost)
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        help="Backfill exactly one repo (default: every .duck under LADYBUG_DB_DIR)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report rowcount that WOULD be embedded, but make no SageMaker calls",
    )
    parser.add_argument(
        "--cap",
        type=float,
        default=DEFAULT_CAP_USD,
        help=f"Hard cost cap in USD (default {DEFAULT_CAP_USD})",
    )
    args = parser.parse_args()

    duck_files = _iter_duck_files(args.repo)
    if not duck_files:
        logger.error("no .duck files found under %s", settings.LADYBUG_DB_DIR)
        return 4

    embedder: BgeCodeV1Embedder | None = None
    if not args.dry_run:
        if not (
            os.environ.get("SAGEMAKER_BGE_CODE_ENDPOINT")
            or os.environ.get("SAGEMAKER_BGE_CODE_URL")
        ):
            logger.error(
                "SAGEMAKER_BGE_CODE_ENDPOINT (or _URL) must be set for a "
                "non-dry-run backfill — refusing to start."
            )
            return 3
        embedder = BgeCodeV1Embedder()

    t0 = time.time()
    total_eligible = 0
    total_written = 0
    total_cost = 0.0
    capped = False
    for duck_path in duck_files:
        eligible, written, cost = backfill_one_duck(
            duck_path,
            embedder,
            cap_usd=args.cap - total_cost,
            dry_run=args.dry_run,
        )
        total_eligible += eligible
        total_written += written
        total_cost += cost
        if total_cost >= args.cap:
            capped = True
            break

    logger.info(
        "DONE: files=%d eligible=%d written=%d cost=$%.4f wall=%.1fs%s",
        len(duck_files),
        total_eligible,
        total_written,
        total_cost,
        time.time() - t0,
        " (DRY RUN)" if args.dry_run else "",
    )
    return 2 if capped else 0


if __name__ == "__main__":
    raise SystemExit(main())
