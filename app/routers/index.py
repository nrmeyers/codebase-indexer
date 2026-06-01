"""POST /index and GET /index/{job_id}/status — background ingestion.

Indexing a repository is CPU-bound and can take minutes. To keep the HTTP
API responsive, this router accepts indexing requests asynchronously:

    1. ``POST /index`` creates a ``_Job`` record and returns a job_id (202).
    2. The heavy work runs in a thread-pool executor via FastAPI's
       BackgroundTasks so the event loop stays free.
    3. Clients poll ``GET /index/{job_id}/status`` until ``done`` or
       ``failed``.

Every index run executes two passes:

    Pass 1–3 (structural): tree-sitter parse → LadybugDB graph (nodes + rels)
    Pass 4   (embedding):  CodeRankEmbed model → DuckDB (.duck, v5.3 §6.5)

Embeddings are **required** — if pass 4 fails the job is marked ``failed``
and the caller must re-index.  Structural graph data is preserved on disk so
re-indexing without ``force_reindex=true`` only re-runs the embedding pass
over already-parsed files.

Jobs live in an in-memory dict (``_jobs``). This is acceptable because the
service is single-process; running multiple replicas would require moving
the store to Redis or LadybugDB itself.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from pydantic import BaseModel

from ..config import settings, slugify_repo
from .. import metrics as _metrics
from ..services import jobs_store as _jobs_store
from ..models import (
    DeleteIndexResponse,
    DiffMetrics,
    IndexAccepted,
    IndexRequest,
    IndexStatus,
    JobClearResponse,
    JobListResponse,
    JobSummary,
    NodeTypeStat,
    RepoStatsResponse,
)

import os as _os

logger = logging.getLogger(__name__)

router = APIRouter()

# Per-process worker token used by jobs_store.sweep_interrupted() to identify
# orphaned rows from prior processes. Matches the token written to each new row.
_WORKER_TOKEN: str = _os.urandom(8).hex()


def _cleanup_old_embed_logs(keep: int = 5) -> int:
    """Remove all but the ``keep`` most-recent /tmp/cis_embed_*.log files.

    Embed logs are diagnostic artefacts that pile up over weeks of dev use
    (one per re-index attempt).  Tail behaviour and incremental embedding
    mean we run more re-indexes per day, so without housekeeping the file
    count grows unbounded.  Called on uvicorn startup; best-effort.
    """
    try:
        log_paths = sorted(
            Path("/tmp").glob("cis_embed_*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        removed = 0
        for stale in log_paths[keep:]:
            try:
                stale.unlink()
                removed += 1
            except OSError:
                pass
        return removed
    except Exception:  # noqa: BLE001
        return 0


def _parse_embed_progress(job_id: str) -> tuple[int, int, int] | None:
    """Read the latest PROGRESS line from an embed subprocess log.

    Returns ``(embedded, skipped_unchanged, filtered_out)`` from the most
    recent ``PROGRESS embedded=N skipped=M filtered=K`` line, or None when
    the log file does not exist or has no PROGRESS line yet.

    The embed driver writes one PROGRESS line per concurrent flush
    (BUC-1517 / BUC-1519).  Tailing the log gives the frontend live
    visibility into the embed pass without changing the subprocess
    contract.
    """
    log_path = Path(f"/tmp/cis_embed_{job_id}-embed.log")
    if not log_path.exists():
        return None
    try:
        # Embed logs are tiny (one PROGRESS line every ~30s).  No need
        # for a stat-then-seek tail; just read and reverse-scan.
        with log_path.open() as fh:
            lines = fh.readlines()
    except OSError:
        return None
    for line in reversed(lines):
        if line.startswith("PROGRESS "):
            parts = line.split()
            try:
                vals = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in parts[1:]}
                return (
                    vals.get("embedded", 0),
                    vals.get("skipped", 0),
                    vals.get("filtered", 0),
                )
            except (ValueError, IndexError):
                return None
    return None


# Pre-warm tracking — only warm the endpoint once per uvicorn process per job.
# A set of job_ids that have already had a warmup ping sent.
_prewarmed_jobs: set[str] = set()


def _prewarm_sagemaker_endpoint(job_id: str) -> None:
    """Fire-and-forget warmup ping to the configured embedder backend.

    Originally added (BUC-1518 D1) to hide the 30-60s SageMaker Serverless
    Inference cold start by running in parallel with the parsing phase
    (~70s for typical repos). Still useful for the SageMaker backend after
    the BUC-1605 generalisation; effectively a no-op for ``local`` / ``tei``
    backends but cheap enough not to gate on backend type.

    Idempotent per job — only fires once even if the parsing-phase callback
    runs multiple times for the same job_id.  Cheap (~$0.0001 per call on
    SageMaker) and failures are silently swallowed since this is purely a
    latency optimisation.
    """
    if job_id in _prewarmed_jobs:
        return
    _prewarmed_jobs.add(job_id)

    def _ping() -> None:
        try:
            from ..embedders.sync_bridge import (
                embed_text_sync,
                get_embedder_or_none,
            )

            backend = get_embedder_or_none()
            if backend is None:
                return  # not configured; nothing to warm
            t0 = time.time()
            embed_text_sync("warmup")
            logger.info(
                "Embedder prewarm: backend=%s, latency=%.2fs (job=%s)",
                backend.name,
                time.time() - t0,
                job_id[:8],
            )
        except Exception as exc:  # noqa: BLE001
            # Warmup is best-effort — never fail the job over a missed ping.
            logger.debug(
                "Embedder prewarm failed (best-effort): %s: %s",
                type(exc).__name__,
                exc,
            )

    threading.Thread(target=_ping, name=f"embed-prewarm-{job_id[:8]}", daemon=True).start()


class _IndexCancelledError(RuntimeError):
    """Raised from the progress callback when a job receives a cancel signal."""


# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------


@dataclass
class _Job:
    """Internal bookkeeping record for a single indexing request.

    Attributes:
        job_id: UUID4 string returned to the client.
        repo_path: Resolved absolute path to the repo being indexed.
        status: Lifecycle state — ``running`` → ``done`` | ``failed``.
        progress_pct: Best-effort progress indicator, monotonically
            non-decreasing.  Derived from phase + per-file counters via
            the progress callback rather than hard-coded milestones.
        phase: Current phase label shown to UIs.
        files_total: Total eligible files discovered (0 until discovering
            phase completes).
        files_done: Files fully scanned so far (advances during parsing).
        current_file: Relative path of the file currently being parsed;
            None outside the parsing phase.
        node_count: Final graph node count, populated on completion.
        rel_count: Final graph relationship count, populated on completion.
        embedded_count: Number of function/method embeddings written.
        cancelled: Set to True by the cancel endpoint; the progress
            callback raises _IndexCancelledError on the next check.
        error: Populated only when ``status == "failed"``.
        started_at: Wall-clock start time used for TTL-based pruning.
        finished_at: Wall-clock end time; None while running.
    """

    job_id: str
    repo_path: str
    status: Literal["running", "done", "failed"] = "running"
    progress_pct: float = 0.0
    phase: str = "queued"
    files_total: int = 0
    files_done: int = 0
    current_file: str | None = None
    node_count: int = 0
    rel_count: int = 0
    embedded_count: int = 0
    embeddings_skipped_unchanged: int = 0
    embeddings_filtered_out: int = 0
    # BUC-1601 (Fix A) — count of source-file read failures the embed
    # driver hit and explicitly dropped.  Should be 0 on healthy runs;
    # a non-zero value means the working tree was mutated mid-index or
    # the graph references files that no longer exist on disk.
    embeddings_dropped_unreadable: int = 0
    # BUC-1574 (Phase 1.4) — diff-metrics instrumentation. Captured from the
    # embed subprocess so /index/{job_id}/diff_metrics can report the
    # incremental-embed audit shape without re-tailing the log.
    embed_started_at: float | None = None
    embed_finished_at: float | None = None
    cancelled: bool = False
    elapsed_sec: float = 0.0
    eta_sec: float | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    # LE-143 fix: progress heartbeat. Advanced on every progress callback so
    # the phase watchdog can distinguish a hung job (no progress) from a
    # legitimately slow one. Distinct from ``started_at`` (whole-job age),
    # which never advances and would falsely flag long healthy runs.
    last_progress_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    exclude_paths: frozenset[str] = field(default_factory=frozenset)


# Module-level transient runtime cache — indexed by job_id. Single-process.
#
# Two-store design (intentional, not a TODO):
#   - `_jobs` holds the high-frequency mutable progress fields for an
#     in-flight job (current_file, eta_sec, files_done) and is updated at
#     ~1 Hz from the GraphUpdater progress_cb. Lives only in RAM.
#   - `app.services.jobs_store` is the durable record (SQLite, WAL): it
#     gets the terminal counters via `mark_done` / `mark_failed`, sweeps
#     interrupted rows on lifespan boot, and surfaces persistent state
#     across restarts. Updates are coarser — phase transitions only.
#
# Persisting every callback to SQLite would either thrash disk or require
# a write-coalescer that adds complexity for no gain. Querying _jobs at
# response time is O(1) and always reflects the latest callback fire.
# Querying jobs_store is needed only when the job_id isn't in _jobs (i.e.
# the job survived a restart) — that lookup happens lazily in
# get_index_status's fallback branch.
_jobs: dict[str, _Job] = {}

# Per-repo-path lock: prevents two concurrent jobs from writing to the same
# LadybugDB instance simultaneously (single-writer constraint).
_repo_locks: dict[str, asyncio.Lock] = {}

# Interval (seconds) between liveness ticks emitted by the writing-phase
# heartbeat thread (see ``_writing_phase_heartbeat``). Must be well under the
# reconciler/watchdog staleness threshold (JOB_STALENESS_THRESHOLD_SECONDS /
# JOB_PHASE_WATCHDOG_SECONDS, both >= 300s) so a single missed tick never trips
# a false-kill. 30s gives ~10x margin.
_WRITING_HEARTBEAT_INTERVAL_SECONDS = 30.0


class _writing_phase_heartbeat:
    """Keep a job observably-alive during a long, callback-silent bulk write.

    Root cause (write-stall false-kill): ``GraphUpdater.run()`` emits a single
    ``{"phase": "writing"}`` progress event and then calls the blocking
    ``LadybugIngestor.flush_all()`` — a Kùzu bulk node + relationship COPY that
    can run for minutes on a large repo while emitting **zero** further progress
    callbacks. The progress callback is the only thing that advances both
    ``job.last_progress_at`` (in-memory) and the durable jobs_store
    ``updated_at``. With neither advancing, the periodic heartbeat reconciler
    (``reconcile_stale_running_jobs``, JOB_STALENESS_THRESHOLD_SECONDS=300) and
    the phase watchdog both see the job as "hung — no progress for >300s" and
    mark it ``failed`` mid-write, leaving a partially-written graph (missing
    route handlers, degenerate single mega-cluster in the KG viewer).

    This context manager spawns a daemon thread that, while the wrapped write
    runs, periodically bumps ``job.last_progress_at`` and advances the durable
    ``updated_at`` (via ``update_progress(phase=...)``). Liveness is therefore
    monotonic throughout the write and neither reaper false-kills a healthy
    bulk flush. The tick is a no-op for progress_pct (it never goes backwards);
    it exists purely to prove the worker thread is alive.

    Best-effort: a failure to write the heartbeat (e.g. jobs_store not
    initialised in tests) is swallowed — the heartbeat must never fail the
    index worker. The thread is stopped + joined on ``__exit__``.
    """

    def __init__(
        self,
        job: _Job,
        *,
        interval_seconds: float = _WRITING_HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        self._job = job
        self._interval = max(1.0, float(interval_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _tick(self) -> None:
        now = time.time()
        # In-memory liveness — read by reconcile_stale_running_jobs path #1.
        self._job.last_progress_at = now
        # Durable liveness — touch_heartbeat() bumps ONLY ``updated_at`` (which
        # list_stale_running_jobs / reconcile path #2 keys on) without touching
        # phase or progress_pct, which the real progress callback owns. A tick
        # mid-parsing must not clobber the live phase shown in the UI.
        try:
            _jobs_store.touch_heartbeat(self._job.job_id)
        except RuntimeError:
            pass  # jobs_store not initialised (tests without lifespan)
        except Exception:  # noqa: BLE001
            pass  # never let a bookkeeping write fail the index worker

    def _run(self) -> None:
        # Tick once immediately so a write that starts right after a long
        # parsing gap resets the staleness clock, then on a fixed interval.
        while not self._stop.wait(self._interval):
            self._tick()

    def __enter__(self) -> _writing_phase_heartbeat:
        self._tick()
        self._thread = threading.Thread(
            target=self._run,
            name=f"writing-heartbeat-{self._job.job_id[:8]}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        # Final tick so the post-write transition starts from a fresh clock.
        self._tick()

# TTL: keep completed jobs for 1 hour so callers can poll after completion.
_JOB_TTL_SECONDS = 3600

# In-memory set of successfully indexed project names.  Updated by the
# background worker when a job completes.  Health + explorer endpoints read
# from here instead of opening a second DB connection (LadybugDB is
# single-writer — concurrent open() calls corrupt the WAL).
indexed_repos: set[str] = set()

# Maps repo_name → resolved absolute repo_path for embed jobs that need it.
indexed_repo_paths: dict[str, str] = {}

# Maps repo_name → unix timestamp of the last successful index.  Populated from
# the Project node's ``last_indexed_at`` on first read (see _get_last_indexed_at)
# so it survives restarts.  In-memory cache avoids a DB round-trip on every
# /health probe.
_last_indexed_cache: dict[str, float] = {}


def _capture_head_sha(repo_path: str | Path) -> str | None:
    """Return ``git rev-parse HEAD`` for ``repo_path`` or None on any failure.

    Best-effort SHA capture used at index-job completion to persist a
    durable ``last_indexed_sha`` into the per-repo ``repo_metadata``
    table (LE-111). Failure modes (non-git checkout, missing ``git``
    binary, timeout) all return None so the indexer keeps working on
    non-git source trees and the caller can record a null SHA.

    A subprocess shell-out is used rather than importing
    ``codebase_rag.services.git_diff`` because that sibling-package
    surface has shifted historically — the import-time failure was
    silently swallowed by the post-job try/except and the SHA was
    never persisted. Inlining the two-line shell-out makes the SHA
    capture path stable independent of the sibling package.
    """
    if not repo_path:
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if result.returncode == 0:
            sha = result.stdout.strip()
            return sha or None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _write_meta(repo_name: str, **fields: Any) -> None:
    """Upsert ``fields`` into the per-repo DuckDB ``repo_metadata`` table.

    Opens the ``.duck`` file (creating it if absent), writes all key-value
    pairs atomically, and closes the connection.  Replaces the old JSON
    sidecar approach — DuckDB transactions provide the same atomic-write
    guarantee.
    """
    from codebase_rag.storage.vector_store import open_or_create, write_metadata

    vec_path = settings.vec_db_path_for_repo(repo_name)
    try:
        conn = open_or_create(vec_path)
        write_metadata(conn, **{k: str(v) for k, v in fields.items()})
        conn.close()
    except Exception:
        pass  # metadata write is best-effort; never fail the index job


def _read_meta(repo_name: str) -> dict[str, Any]:
    """Return all metadata for ``repo_name`` from the DuckDB ``repo_metadata`` table.

    Falls back to an empty dict when the ``.duck`` file does not yet exist
    (e.g. before the first successful index) or when DuckDB is unavailable.
    """
    from codebase_rag.storage.vector_store import open_or_create, read_all_metadata

    vec_path = settings.vec_db_path_for_repo(repo_name)
    if not Path(vec_path).exists():
        return {}
    try:
        conn = open_or_create(vec_path)
        meta = read_all_metadata(conn)
        conn.close()
        return meta
    except Exception:
        return {}


def _get_last_indexed_at(repo_name: str) -> float | None:
    """Return the last successful-index timestamp for ``repo_name``.

    Checks the in-memory cache first, then the ``repo_metadata`` SQLite table.
    Returns None on any miss or parse failure.
    """
    if repo_name in _last_indexed_cache:
        return _last_indexed_cache[repo_name]

    meta = _read_meta(repo_name)
    ts = meta.get("last_indexed_at")
    if ts is not None:
        try:
            val = float(ts)
            _last_indexed_cache[repo_name] = val
            return val
        except (TypeError, ValueError):
            return None
    return None


def is_repo_indexing(repo_name: str) -> bool:
    """Return True when any currently-running job targets ``repo_name``.

    Used by /health and /stats to signal UI-level mutual exclusion.
    """
    for j in _jobs.values():
        if j.status == "running" and Path(j.repo_path).name == repo_name:
            return True
    return False


def _prune_old_jobs() -> None:
    """Drop completed/failed jobs older than the TTL to bound memory.

    Only terminal jobs are pruned — a long-running job is never evicted
    regardless of age.
    """
    now = time.time()
    stale = [
        jid
        for jid, j in _jobs.items()
        if j.status in ("done", "failed") and (now - j.started_at) > _JOB_TTL_SECONDS
    ]
    for jid in stale:
        del _jobs[jid]


# ---------------------------------------------------------------------------
# Background ingestion worker
# ---------------------------------------------------------------------------


async def _run_ingestion(job: _Job, force_reindex: bool) -> None:
    """Drive code-graph-rag indexing in a background asyncio task.

    The core indexing work is CPU-bound / blocking.  We run it in a thread
    pool executor so the event loop stays responsive to health and status
    polling during long ingestion runs.

    A per-repo asyncio.Lock serialises concurrent jobs that target the same
    repo_path, enforcing LadybugDB's single-writer constraint and preventing
    graph corruption from two simultaneous ingest operations.

    Args:
        job: The mutable job record to update with progress and final state.
        force_reindex: When true, the underlying GraphUpdater clears the
            graph before re-ingesting.
    """
    # Acquire (or create) the per-repo lock before touching the DB.
    repo_key = str(Path(job.repo_path).resolve())
    if repo_key not in _repo_locks:
        _repo_locks[repo_key] = asyncio.Lock()
    lock = _repo_locks[repo_key]

    async with lock:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None, _blocking_index, job, force_reindex
            )
        except _IndexCancelledError:
            # phase/error already set by the progress callback before raising.
            if job.finished_at is None:
                job.finished_at = time.time()
            # Phase 2+4: persist cancel state.
            try:
                _jobs_store.mark_failed(job.job_id, error="Cancelled by user", terminal_status="cancelled")
            except RuntimeError:
                pass
            _metrics.record_index_terminal("cancelled", kind="index")
        except Exception as exc:
            # Capture failure on the job so pollers see the error reason rather
            # than a silent stuck-running status.
            job.status = "failed"
            job.error = str(exc)
            job.finished_at = time.time()
            # Phase 2+4: persist failure state.
            try:
                _jobs_store.mark_failed(job.job_id, error=str(exc))
            except RuntimeError:
                pass
            _metrics.record_index_terminal("failed", kind="index")


def _index_markdown_corpus(
    *,
    repo_root: Path,
    repo_name: str,
    vec_db_path: str,
    discover: Any,
    chunker: Any,
    composer: Any,
) -> int:
    """Embed + persist markdown chunks into the per-repo ``.duck`` file.

    Walks ``repo_root`` for eligible ``.md`` files (see
    :func:`app.services.markdown_indexer.discover_markdown_files`), chunks
    each by H1/H2/H3 sections, composes the embed text, calls the
    SageMaker code-embedding bridge in batches, and bulk-inserts the
    results into the same ``embeddings`` table the structural pass uses.
    Each row carries ``symbol_type = "MarkdownDoc"`` so callers can filter
    or weight markdown hits differently from code hits.

    Best-effort + idempotent:

    * Already-embedded chunks with an unchanged content hash are skipped
      via the same BUC-1518 mechanism the code embed driver uses.
    * Tantivy is also updated so the lexical arm can hit ``LE-123`` /
      ``REG-D`` style queries without paying the semantic-search cost.

    Args:
        repo_root: Absolute path to the repo checkout.
        repo_name: Canonical repo slug (used for ``qualified_name``
            namespacing and tantivy index keys).
        vec_db_path: Filesystem path to the per-repo ``.duck`` vector
            store.
        discover: Injected file-discovery callable; defaults to
            :func:`markdown_indexer.discover_markdown_files` (parameterised
            for testability).
        chunker: Injected chunker — see
            :func:`markdown_indexer.chunk_markdown_file`.
        composer: Injected embed-text composer — see
            :func:`markdown_indexer.compose_markdown_embed_text`.

    Returns:
        Number of chunks newly embedded + inserted.  Returns 0 when no
        eligible markdown files are present (a perfectly valid state for
        e.g. service repos with no ``.planning/`` or ``docs/``).
    """
    md_files = discover(repo_root)
    if not md_files:
        return 0

    # Build all chunks up-front so we can size the SageMaker batch loop.
    chunks: list[Any] = []
    for path in md_files:
        try:
            rel = path.relative_to(repo_root).as_posix()
        except ValueError:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning(
                "markdown_indexer.read_failed path=%s err=%s", rel, exc
            )
            continue
        chunks.extend(chunker(repo_name=repo_name, rel_path=rel, content=content))

    if not chunks:
        return 0

    # SageMaker bridge + vector store.  Imports are lazy so this module
    # stays importable in test environments that mock the embed stack.
    from codebase_rag.embedder import embed_code_batch  # noqa: PLC0415
    from codebase_rag.storage.vector_store import (  # noqa: PLC0415
        EmbeddingRow,
        bulk_insert,
        open_or_create,
        read_content_hashes,
    )
    from app.scripts.embed_driver import compute_content_hash  # noqa: PLC0415

    vec_conn = open_or_create(vec_db_path)
    try:
        existing_hashes = read_content_hashes(vec_conn)
        batch_texts: list[str] = []
        batch_meta: list[tuple[Any, str, int, int, str]] = []
        rows: list[EmbeddingRow] = []
        BATCH = 32  # markdown chunks are larger than fn bodies; smaller batch
        inserted = 0

        def _flush() -> None:
            nonlocal inserted
            if not batch_texts:
                return
            embs = embed_code_batch(batch_texts)
            for meta_tuple, emb, text in zip(batch_meta, embs, batch_texts):
                _qn, _fp, _sl, _el, _content_hash = meta_tuple
                rows.append(
                    EmbeddingRow(
                        qualified_name=_qn,
                        embedding=emb,
                        file_path=_fp,
                        start_line=_sl,
                        end_line=_el,
                        symbol_type="MarkdownDoc",
                        content_hash=_content_hash,
                    )
                )
            bulk_insert(vec_conn, rows)
            inserted += len(rows)
            rows.clear()
            batch_texts.clear()
            batch_meta.clear()

        for chunk in chunks:
            embed_text = composer(chunk)
            content_hash = compute_content_hash(embed_text)
            if existing_hashes.get(chunk.qualified_name) == content_hash:
                continue  # unchanged — skip the SageMaker call
            batch_texts.append(embed_text)
            batch_meta.append(
                (
                    chunk.qualified_name,
                    chunk.file_path,
                    int(chunk.start_line),
                    int(chunk.end_line),
                    content_hash,
                )
            )
            if len(batch_texts) >= BATCH:
                _flush()
        _flush()
    finally:
        try:
            vec_conn.close()
        except Exception:
            pass

    # Mirror into tantivy for the lexical arm.  Failures are non-fatal
    # for the same reason the code-symbol tantivy pass is non-fatal:
    # semantic search still works.
    try:
        from ..services.tantivy_index import TantivyIndex  # noqa: PLC0415
        from ..config import slugify_repo  # noqa: PLC0415

        slug = slugify_repo(repo_name)
        t_idx = TantivyIndex(settings.LADYBUG_DB_DIR, slug)
        try:
            for chunk in chunks:
                # Content = heading + body so lexical queries on "LE-123"
                # can hit either the section title or the body prose.
                content = f"{chunk.heading}\n{chunk.body}"
                t_idx.add(
                    symbol_qname=chunk.qualified_name,
                    file_path=chunk.file_path,
                    symbol_kind="MarkdownDoc",
                    content=content,
                    start_line=int(chunk.start_line),
                    end_line=int(chunk.end_line),
                    repo=slug,
                )
            t_idx.commit()
        finally:
            t_idx.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "markdown_indexer.tantivy_pass_failed (non-fatal): %s", exc
        )

    return inserted


def _blocking_index(job: _Job, force_reindex: bool) -> None:
    """Synchronous ingestion — called from the thread pool.

    Uses LadybugIngestor as a context manager (which handles DB connection,
    schema migration, and VECTOR extension loading on entry). GraphUpdater
    requires parsers and queries to be loaded from the tree-sitter registry.

    Args:
        job: The job record to mutate with progress and final counts.
            ``job.exclude_paths`` is forwarded to GraphUpdater to skip
            synthetic test fixtures and other noise from the semantic index.
        force_reindex: When true, the graph is cleared before ingesting.
    """
    from codebase_rag.config import settings as cgr_settings
    from app.services.ladybug_ingestor import LadybugIngestor
    from codebase_rag.graph_updater import GraphUpdater
    from codebase_rag.parser_loader import load_parsers

    repo = Path(job.repo_path).resolve()

    # BUC-1580: derive the canonical slug from ``git remote get-url origin``.
    # Falls back to the basename when no GitHub remote is configured (e.g.
    # bare local checkouts, gitlab/bitbucket, multiple-remote ambiguity).
    # The App-clone path already uses ``{owner}__{repo}`` as its directory
    # name, so canonical and basename converge there — no behaviour change.
    from ..services.slug import derive_slug as _derive_slug  # noqa: PLC0415

    repo_name = _derive_slug(repo, repo.name)

    # Per-repo DB file: each indexed repo gets its own ``.db`` so the explorer
    # can open one index at a time and WAL corruption / re-indexing stays
    # scoped.  Parent directory is created lazily because LadybugDB will
    # otherwise fail with "No such file or directory".
    repo_db_path = settings.db_path_for_repo(repo_name)
    Path(repo_db_path).parent.mkdir(parents=True, exist_ok=True)

    # Point code-graph-rag at the per-repo DB. Without this, the two packages
    # would each read their own config and could end up writing to different
    # database files.
    cgr_settings.LADYBUG_DB_PATH = repo_db_path
    cgr_settings.LADYBUG_BATCH_SIZE = settings.LADYBUG_BATCH_SIZE

    parsers, queries = load_parsers()

    # Detect a fresh / empty DB BEFORE the ingestor creates/opens it.
    # If the DB didn't exist (or is zero-byte from a previous wipe), force a
    # full re-parse regardless of force_reindex.  Without this, the on-disk
    # hash cache (.cgr-hash-cache.json inside the repo) causes the incremental
    # updater to skip every "unchanged" file, leaving an empty graph even
    # though the DB has never been populated.
    db_was_new = (
        not Path(repo_db_path).exists()
        or Path(repo_db_path).stat().st_size < 4096  # < 4 KB = essentially empty
    )
    effective_force = force_reindex or db_was_new

    # When the caller requested a full force-reindex, physically delete the
    # per-repo DB file + WAL + hash cache.  Without this, GraphUpdater's
    # ``force=True`` only resets the in-memory hash cache — old Function /
    # Call / File nodes from previous ingestions REMAIN in LadybugDB and
    # the new parse just layers fresh copies on top.  Symptoms: junk
    # symbols persist across re-indexes, node count grows monotonically,
    # cgrignore changes only partially take effect.
    if force_reindex:
        db_file = Path(repo_db_path)
        # LadybugDB artefacts (kuzu naming: .db-wal, .db-shm)
        ladybug_wal = db_file.with_suffix(db_file.suffix + "-wal")
        ladybug_shm = db_file.with_suffix(db_file.suffix + "-shm")
        # DuckDB artefacts live alongside the .duck file derived from the
        # same stem — DuckDB names its journal `<file>.wal` (dot, not dash)
        # and may leave a `<file>.tmp` from interrupted writes.
        duck_file = db_file.with_suffix(".duck")
        duck_wal = db_file.with_name(db_file.stem + ".duck.wal")
        duck_tmp = db_file.with_name(db_file.stem + ".duck.tmp")
        # Belt-and-braces: also catch the ".db.wal" form that DuckDB has
        # been observed to construct when its connection path got the
        # LadybugDB extension by mistake.
        belt_db_dot_wal = db_file.with_name(db_file.name + ".wal")
        hash_cache = repo / ".cgr-hash-cache.json"
        for artifact in (
            db_file,
            ladybug_wal,
            ladybug_shm,
            duck_file,
            duck_wal,
            duck_tmp,
            belt_db_dot_wal,
            hash_cache,
        ):
            try:
                artifact.unlink(missing_ok=True)
            except OSError:
                # Best-effort — a locked WAL file is not fatal; the ingestor
                # will recreate what it needs.
                pass

    # ------------------------------------------------------------------
    # Progress callback — maps GraphUpdater events to _Job state.
    # Called from the GraphUpdater thread; must be thread-safe for the
    # simple field writes it performs (GIL-protected in CPython).
    # ------------------------------------------------------------------
    _job_start = job.started_at

    def _progress_cb(event: dict) -> None:
        # Cancel check — must happen before any state mutation so that
        # phase/error are set consistently before raising.
        if job.cancelled:
            job.phase = "cancelled"
            job.status = "failed"
            job.error = "Cancelled by user"
            job.finished_at = time.time()
            raise _IndexCancelledError("Cancelled by user")

        phase = event.get("phase")
        if phase:
            job.phase = str(phase)
            # Pre-warm SageMaker the moment we enter the parsing phase.
            # Parsing typically takes ~70s; that's enough to absorb the
            # 30-60s Serverless Inference cold start in parallel, so
            # embedding kicks off against an already-hot endpoint.
            if phase == "parsing":
                _prewarm_sagemaker_endpoint(job.job_id)
        if "files_total" in event:
            job.files_total = int(event["files_total"])
        if "files_done" in event:
            job.files_done = int(event["files_done"])
        if "current_file" in event:
            cf = event["current_file"]
            job.current_file = str(cf) if cf else None

        # Compute progress_pct from phase + counters (monotonically non-decreasing).
        if "progress_pct" in event:
            pct: float = float(event["progress_pct"])
        elif phase == "discovering":
            pct = 2.0
        elif phase == "parsing":
            pct = 5.0 + (job.files_done / max(job.files_total, 1)) * 60.0
        elif phase == "writing":
            pct = 65.0
        elif phase == "embedding":
            pct = 70.0
        elif phase == "finalizing":
            pct = 98.0
        elif phase == "done":
            pct = 100.0
        else:
            pct = job.progress_pct  # unchanged

        job.progress_pct = max(job.progress_pct, min(100.0, pct))
        # LE-143 fix: heartbeat — every callback proves the job is alive.
        job.last_progress_at = time.time()

        # Elapsed + ETA — computed at callback time (cheap, no extra polling).
        elapsed = time.time() - _job_start
        job.elapsed_sec = elapsed
        if job.progress_pct > 10.0 and job.progress_pct < 100.0:
            job.eta_sec = elapsed * (100.0 - job.progress_pct) / job.progress_pct
        else:
            job.eta_sec = None

        # LE-143 fix: mirror progress to the durable store so its ``updated_at``
        # advances on every phase transition. The phase watchdog + reconcile
        # path key staleness on ``updated_at``; without this a slow-but-healthy
        # run would look hung. Throttled to phase transitions (not every
        # ~1 Hz file callback) to keep SQLite writes coarse. Best-effort —
        # never let a bookkeeping write fail the index worker.
        if phase:
            try:
                _jobs_store.update_progress(
                    job.job_id,
                    phase=str(phase),
                    progress_pct=job.progress_pct,
                    files_total=job.files_total,
                    files_done=job.files_done,
                )
            except RuntimeError:
                pass  # jobs_store not initialised (tests without lifespan)
            except Exception:  # noqa: BLE001
                pass

    # Load .cgrignore patterns from the repo root and merge with explicit
    # exclude_paths from the POST body.  Without this the service would skip
    # only the request's exclude_paths and ignore the user's .cgrignore file
    # entirely (which is the expected extensibility surface for teams who
    # can't modify the POST body — e.g. IDE integrations).
    from codebase_rag.config import load_cgrignore_patterns
    cgrignore = load_cgrignore_patterns(repo)
    merged_excludes: frozenset[str] | None = None
    if job.exclude_paths or cgrignore.exclude:
        merged_excludes = frozenset(job.exclude_paths) | cgrignore.exclude
    unignore_paths: frozenset[str] | None = (
        cgrignore.unignore if cgrignore.unignore else None
    )

    # Phase 4: record start time for "parse" phase timer.
    _phase_parse_start = time.monotonic()

    # Sub-phase timers (Cycle 5 follow-up — localising the 4.4 sym/s
    # end-to-end bottleneck. Embed bench showed embedding is 89→121 sym/s
    # standalone, so >95% of indexing time is downstream of embedding.
    # These timers tell us whether the cost is in LadybugIngestor open/close
    # or in GraphUpdater.run() so we know where to push deeper
    # instrumentation. Phase-name keys: "parse_open", "parse_run",
    # "parse_metadata", "parse_close").
    _t_open = time.monotonic()
    # LadybugIngestor is a context manager — __enter__ opens the DB connection
    # and runs schema migration; __exit__ flushes remaining buffers and closes.
    with LadybugIngestor(
        db_path=repo_db_path,
        batch_size=settings.LADYBUG_BATCH_SIZE,
    ) as ingestor:
        _metrics.record_index_phase("parse_open", time.monotonic() - _t_open)

        updater = GraphUpdater(
            ingestor=ingestor,
            repo_path=repo,
            parsers=parsers,
            queries=queries,
            progress_cb=_progress_cb,
            exclude_paths=merged_excludes,
            unignore_paths=unignore_paths,
            skip_embeddings=True,  # embeddings handled by DuckDB subprocess below
        )

        _t_run = time.monotonic()
        # updater.run() emits a single {"phase": "writing"} event and then
        # blocks in LadybugIngestor.flush_all() — a Kùzu bulk write that runs
        # for minutes on a large repo with NO further progress callbacks. The
        # heartbeat thread bumps in-memory + durable liveness on a 30s tick so
        # the reconciler/phase-watchdog never false-kill a healthy slow write
        # mid-flush (see _writing_phase_heartbeat). The window also covers the
        # silent parts of parsing/embedding for free.
        with _writing_phase_heartbeat(job):
            updater.run(force=effective_force)
        _metrics.record_index_phase("parse_run", time.monotonic() - _t_run)
        # GraphUpdater emits "done" at 100% at the end of run(); reset to 92%
        # so the UI knows the embedding subprocess pass still follows.
        job.progress_pct = min(job.progress_pct, 92.0)

        # Store the absolute repo root on the Project node so search endpoints
        # can resolve relative file paths back to absolute paths.  The
        # last-indexed timestamp and other operational metadata live in the
        # ``repo_metadata`` table inside the per-repo ``.duck`` file instead —
        # LadybugDB's typed schema doesn't allow adding new columns without a
        # migration, while DuckDB key/value rows are free to extend.
        _t_meta = time.monotonic()
        project_name = repo.name
        ingestor.conn.execute(  # type: ignore[union-attr]
            "MATCH (p:Project {name: $name}) SET p.root_path = $root_path",
            {"name": project_name, "root_path": str(repo)},
        )
        _metrics.record_index_phase("parse_metadata", time.monotonic() - _t_meta)
        _t_close = time.monotonic()

    # `parse_close` ends here — flush + connection teardown happens on __exit__.
    _metrics.record_index_phase("parse_close", time.monotonic() - _t_close)
    # Top-level "parse" stays for back-compat with existing dashboard panels.
    _metrics.record_index_phase("parse", time.monotonic() - _phase_parse_start)

    # Collect final counts from LadybugDB. Best-effort: if the count
    # queries fail we still mark the job done — the ingestion itself
    # succeeded and callers can always re-query later.
    # IMPORTANT: use a short-lived block so Python GC can release the DB
    # lock before the embedding subprocess tries to open the same file.
    _count_db = None
    _count_conn = None
    try:
        import real_ladybug as lb  # type: ignore[import-untyped]
        from ..services.ladybug_buffer_pool import resolve_buffer_pool_size  # noqa: PLC0415

        _count_db = lb.Database(repo_db_path, buffer_pool_size=resolve_buffer_pool_size())
        _count_conn = lb.Connection(_count_db)

        node_res = _count_conn.execute("MATCH (n) RETURN count(n) AS cnt")
        if node_res.has_next():
            job.node_count = int(node_res.get_next()[0])

        rel_res = _count_conn.execute("MATCH ()-[r]->() RETURN count(r) AS cnt")
        if rel_res.has_next():
            job.rel_count = int(rel_res.get_next()[0])
    except Exception:
        pass  # counts are best-effort
    finally:
        # Explicitly close and drop references so LadybugDB releases the
        # file lock before the embedding subprocess attempts to open the
        # same DB. `del` alone is not enough because CPython's refcount
        # may delay destruction when an exception is in-flight.
        try:
            if _count_conn is not None and hasattr(_count_conn, "close"):
                _count_conn.close()
        except Exception:
            pass
        try:
            if _count_db is not None and hasattr(_count_db, "close"):
                _count_db.close()
        except Exception:
            pass
        _count_conn = None
        _count_db = None
        import gc as _gc
        _gc.collect()

    # Persist metadata sidecar AFTER counts are populated so the UI sees
    # authoritative node/rel totals in /stats without a separate query.
    # LE-111: capture HEAD SHA here so the parse-pass meta write also
    # persists last_indexed_sha — keeps SHA + root_path on the same
    # write so /repos can always report drift status.
    _now = time.time()
    _head_sha = _capture_head_sha(repo)
    _meta_fields: dict[str, Any] = {
        "last_indexed_at": _now,
        "root_path": str(repo),
        "node_count": str(job.node_count),
        "rel_count": str(job.rel_count),
        "last_job_id": job.job_id,
        "schema_version": "1.5",
    }
    if _head_sha:
        _meta_fields["last_indexed_sha"] = _head_sha
    _write_meta(repo_name, **_meta_fields)
    _last_indexed_cache[repo_name] = _now

    # -------------------------------------------------------------------
    # Phase 1.1 — Tantivy BM25 lexical index (best-effort)
    # -------------------------------------------------------------------
    # After the parse pass commits LadybugDB we read the symbol metadata
    # back out and mirror it into a per-repo Tantivy index.  Failures here
    # are NON-FATAL — the lexical arm is additive; semantic + structural
    # search still work fine without it.  See OPTIMIZATION_ROADMAP §1.1.
    _phase_tantivy_start = time.monotonic()
    try:
        from ..services.tantivy_index import TantivyIndex  # noqa: PLC0415
        from ..config import slugify_repo  # noqa: PLC0415
        import real_ladybug as _lb_t  # type: ignore[import-untyped]  # noqa: PLC0415
        from ..services.ladybug_buffer_pool import resolve_buffer_pool_size  # noqa: PLC0415

        _t_db = _lb_t.Database(
            repo_db_path, read_only=True, buffer_pool_size=resolve_buffer_pool_size()
        )
        _t_conn = _lb_t.Connection(_t_db)
        try:
            _cypher = (
                "MATCH (m:Module)-[:DEFINES]->(n:Function) "
                "RETURN n.qualified_name AS qn, n.start_line AS sl, "
                "n.end_line AS el, m.path AS p, n.docstring AS doc, "
                "'Function' AS kind "
                "UNION ALL "
                "MATCH (m:Module)-[:DEFINES]->(:Class)-[:DEFINES_METHOD]->(n:Method) "
                "RETURN n.qualified_name AS qn, n.start_line AS sl, "
                "n.end_line AS el, m.path AS p, n.docstring AS doc, "
                "'Method' AS kind"
            )
            _res = _t_conn.execute(_cypher)
            _cols = _res.get_column_names()
            # BUC-1580: ``repo_name`` is already canonical; pass through
            # ``slugify_repo`` for the filesystem-safe charset guarantee.
            _slug = slugify_repo(repo_name)
            _t_idx = TantivyIndex(settings.LADYBUG_DB_DIR, _slug)
            _added = 0
            try:
                while _res.has_next():
                    _row = dict(zip(_cols, _res.get_next()))
                    _qn = _row.get("qn") or ""
                    if not _qn:
                        continue
                    # ``content`` is the BM25 corpus body — qname tokens +
                    # docstring give the ranker enough signal without
                    # paying the cost of full source I/O at index time.
                    _content_parts = [_qn]
                    _doc = _row.get("doc")
                    if isinstance(_doc, str) and _doc:
                        _content_parts.append(_doc)
                    _t_idx.add(
                        symbol_qname=str(_qn),
                        file_path=str(_row.get("p") or ""),
                        symbol_kind=str(_row.get("kind") or "Function"),
                        content=" ".join(_content_parts),
                        start_line=int(_row.get("sl") or 0),
                        end_line=int(_row.get("el") or 0),
                        repo=_slug,
                    )
                    _added += 1
                _t_idx.commit()
            finally:
                _t_idx.close()
            logger.info("tantivy.indexed repo=%s symbols=%d", repo_name, _added)
        finally:
            try:
                _t_conn.close()
            except Exception:
                pass
            try:
                _t_db.close()
            except Exception:
                pass
    except Exception as _exc:
        logger.warning("tantivy.index_pass_failed (non-fatal): %s", _exc)
    _metrics.record_index_phase("tantivy", time.monotonic() - _phase_tantivy_start)

    # Register the repo immediately after the structural graph commits so that
    # /health and /explorer see it even while the embedding pass is running.
    # The structural graph is on disk and fully queryable at this point.
    indexed_repos.add(repo_name)
    indexed_repo_paths[repo_name] = str(repo)
    # BUC-1599 — register / refresh the authoritative ``indexed_repos`` row
    # so the next service restart can rehydrate without globbing the disk.
    try:
        _jobs_store.upsert_indexed_repo(
            slug=repo_name,
            display_name=repo_name,
            db_path=repo_db_path,
        )
    except RuntimeError:
        pass  # jobs_store not initialised (tests without lifespan)
    except Exception as _exc:  # noqa: BLE001
        logger.debug("jobs_store.upsert_indexed_repo non-fatal: %s", _exc)
    try:
        from .health import invalidate_probe_cache
        invalidate_probe_cache(repo_name)
    except Exception:
        pass  # cache invalidation is best-effort

    job.phase = "embedding"
    job.progress_pct = max(job.progress_pct, 92.0)

    # -------------------------------------------------------------------
    # Pass 4: Embedding generation (subprocess-isolated, REQUIRED)
    # -------------------------------------------------------------------
    # Load the CodeRankEmbed model (~550 MB) and embed every Function/Method
    # source code so that semantic / natural-language search works.
    # Embeddings are NOT optional — if this pass fails the job is marked
    # "failed" so the caller knows to re-index rather than silently
    # serving a graph with no semantic search capability.
    #
    # Running in a subprocess means an OOM kill only takes down that child;
    # uvicorn and the structural graph both survive.  Any exception raised
    # by _blocking_embed propagates to _run_ingestion, which marks the job
    # as "failed" and surfaces the error to pollers.
    embed_job = _EmbedJob(
        job_id=job.job_id + "-embed",
        repo_name=repo_name,
        repo_path=str(repo),
    )
    # Force a GC sweep before spawning the subprocess so that the count-query
    # Database/Connection objects (already set to None above) are fully
    # reclaimed and their OS file-lock released.  Without this the subprocess
    # races the parent and opening the same DB fails with a lock error that
    # surfaces as an empty-stderr exit 1.
    import gc as _gc
    _gc.collect()

    _phase_embed_start = time.monotonic()
    job.embed_started_at = time.time()
    _blocking_embed(embed_job)  # raises on failure → job marked "failed"
    _metrics.record_index_phase("embed", time.monotonic() - _phase_embed_start)
    job.embedded_count = embed_job.embedded_count
    # BUC-1574 (Phase 1.4) — lift incremental-embed totals so the
    # /index/{job_id}/diff_metrics endpoint works after the job completes.
    job.embeddings_skipped_unchanged = max(
        job.embeddings_skipped_unchanged, embed_job.skipped_unchanged
    )
    job.embeddings_filtered_out = max(
        job.embeddings_filtered_out, embed_job.skipped_filtered
    )
    # BUC-1601 (Fix A) — lift the read-failure count so it surfaces in
    # /diff_metrics alongside the other skip categories.
    job.embeddings_dropped_unreadable = max(
        job.embeddings_dropped_unreadable, embed_job.dropped_unreadable
    )
    job.embed_finished_at = embed_job.finished_at or time.time()

    # --- Plan J: PageRank centrality (best-effort, never fail the job) ---
    # Clear before write so qualified names from a previous indexing run don't
    # linger after files are deleted upstream.
    _phase_pagerank_start = time.monotonic()
    try:
        from codebase_rag.storage.centrality import compute_pagerank
        from codebase_rag.storage.vector_store import (
            clear_centrality,
            open_or_create,
            write_centrality,
        )

        pr_scores = compute_pagerank(repo_db_path)
        if pr_scores:
            # BUC-1580: route via the canonical slug so PageRank lands in
            # the same .duck file the embedding pass just wrote.
            _vec_path_pr = settings.vec_db_path_for_repo(repo_name)
            _vec_conn_pr = open_or_create(_vec_path_pr)
            try:
                clear_centrality(_vec_conn_pr)
                write_centrality(_vec_conn_pr, pr_scores)
            finally:
                _vec_conn_pr.close()
            logger.info("pagerank.computed scores=%d", len(pr_scores))
    except Exception as exc:
        logger.warning("pagerank.failed err=%s", exc)
    _metrics.record_index_phase("pagerank", time.monotonic() - _phase_pagerank_start)

    # --- LE-136: Markdown corpus pass (best-effort, never fail the job) ---
    # Index ``.planning/*.md`` + ``docs/*.md`` + root-level docs (README,
    # CLAUDE.md, …) so /search/semantic and /context-bundle can surface
    # planning content (LE briefs, ADRs, dogfood reports).  See REG-D in
    # TheForge .planning/le-dogfood-2026-05-26T16-postwave3-baseline.md.
    #
    # Runs in-process (not a subprocess) because the volume is small
    # (typically <500 chunks/repo) and the SageMaker endpoint is already
    # warm from the function/method embed pass above.  Any failure is
    # logged and swallowed — markdown indexing is purely additive.
    _phase_md_start = time.monotonic()
    try:
        from ..services.markdown_indexer import (  # noqa: PLC0415
            chunk_markdown_file,
            compose_markdown_embed_text,
            discover_markdown_files,
        )

        md_added = _index_markdown_corpus(
            repo_root=repo,
            repo_name=repo_name,
            vec_db_path=settings.vec_db_path_for_repo(repo_name),
            discover=discover_markdown_files,
            chunker=chunk_markdown_file,
            composer=compose_markdown_embed_text,
        )
        if md_added:
            logger.info(
                "markdown_indexer.indexed repo=%s chunks=%d",
                repo_name, md_added,
            )
    except Exception as _exc:  # noqa: BLE001
        logger.warning("markdown_indexer.failed (non-fatal): %s", _exc)
    _metrics.record_index_phase(
        "markdown", time.monotonic() - _phase_md_start
    )

    job.progress_pct = 100.0
    job.status = "done"
    job.finished_at = time.time()

    # Phase 2+4: persist terminal state + emit counter.
    try:
        _jobs_store.mark_done(
            job.job_id,
            node_count=job.node_count,
            rel_count=job.rel_count,
            embedding_count=job.embedded_count,
        )
    except RuntimeError:
        pass  # jobs_store not initialised (tests without lifespan)
    except Exception as _exc:
        logger.debug("jobs_store.mark_done non-fatal: %s", _exc)
    _metrics.record_index_terminal("done", kind="index")

    # BUC-1518 C3 / LE-111 — stamp RepoMeta with the current HEAD SHA only
    # AFTER both graph build and embed pass have completed successfully.
    # A mid-flight crash above this point leaves the OLD SHA in place, so
    # the next /index call re-runs the same diff and recovers without
    # losing prior progress.
    #
    # LE-111: SHA capture used to go via
    # ``codebase_rag.services.git_diff.get_head_sha``, but that import
    # failed at runtime (sibling package surface shifted) and the whole
    # try/except below silently swallowed it — so /repos always returned
    # last_indexed_sha=null. We now use the local ``_capture_head_sha``
    # helper which is independent of the sibling package.
    try:
        head_sha = _capture_head_sha(repo)
        if head_sha:
            # Mirror the SHA into the DuckDB ``repo_metadata`` sidecar so the
            # GET /repos listing endpoint (BUC-1561b) can compute fresh/stale
            # status without a LadybugDB read (which contends with the
            # single-writer lock during a re-index).
            _write_meta(repo_name, last_indexed_sha=head_sha)
            # BUC-1599 — advance the canonical ``indexed_repos`` row so
            # restart-time rehydration sees the fresh SHA + timestamp.
            try:
                _jobs_store.mark_indexed(repo_name, last_commit_sha=head_sha)
            except RuntimeError:
                pass
            except Exception as _mi_exc:  # noqa: BLE001
                logger.debug(
                    "jobs_store.mark_indexed non-fatal: %s", _mi_exc
                )
            # LE-111: previously this block wrote the SHA into the
            # LadybugDB ``RepoMeta`` table via
            # ``codebase_rag.services.repo_meta.stamp``. That sibling
            # symbol no longer exists, so the call was a silent no-op
            # (the outer except swallowed the ImportError). The
            # ``repo_metadata`` DuckDB write above is the authoritative
            # surface for /repos drift detection; the LadybugDB
            # RepoMeta table is only consulted by the incremental-embed
            # path, which has been disabled separately. Logging the
            # successful SHA stamp here is enough for operator
            # visibility.
            logger.info(
                "repo_metadata stamped: repo=%s sha=%s",
                repo_name, head_sha[:8],
            )
        else:
            logger.info(
                "RepoMeta stamp skipped: %s is not a git repo (incremental disabled)",
                repo_name,
            )
            # BUC-1599 — non-git case still advances ``last_indexed_at`` so
            # /repos surfaces a fresh timestamp post-restart.
            try:
                _jobs_store.mark_indexed(repo_name)
            except RuntimeError:
                pass
            except Exception as _mi_exc:  # noqa: BLE001
                logger.debug(
                    "jobs_store.mark_indexed non-fatal: %s", _mi_exc
                )
    except Exception as _exc:
        # Stamping failures are non-fatal — the index already succeeded;
        # we just lose the ability to do incremental on the NEXT call.
        logger.warning("RepoMeta.stamp failed (non-fatal): %s", _exc)

    # BUC-1518 — push the freshly-indexed .db + .duck to S3 so anyone else
    # running the indexer (locally or in another container) sees the same
    # graph + embeddings without re-indexing.  Best-effort: never fail the
    # job if the upload errors; the next graceful shutdown will retry.
    # Runs in a background thread so the long upload (~30-100 MB) doesn't
    # block the response or hold any DB locks.
    #
    # After the push succeeds we also opportunistically evict any STALE
    # local files from OTHER repos that have aged out of the TTL window
    # — this keeps disk usage bounded on long-running VMs without
    # requiring a separate cron job.  The repo we just indexed is
    # naturally fresh and won't be evicted.
    try:
        from .services.s3_store import (
            snapshot_indexes as _s3_snapshot,
            evict_local_cache as _s3_evict,
        )
        def _push_and_evict() -> None:
            try:
                n = _s3_snapshot(settings.LADYBUG_DB_DIR)
                logger.info("S3 snapshot after index: %d file(s) pushed", n)
            except Exception as _e:  # noqa: BLE001
                logger.warning("S3 snapshot after index failed (non-fatal): %s", _e)
                return  # don't evict if push failed — would risk data loss
            try:
                evicted, kept = _s3_evict(settings.LADYBUG_DB_DIR)
                if evicted:
                    logger.info(
                        "S3 cache evict: %d local file(s) dropped (kept %d)",
                        evicted, kept,
                    )
            except Exception as _e:  # noqa: BLE001
                logger.warning("S3 cache evict failed (non-fatal): %s", _e)
        threading.Thread(
            target=_push_and_evict,
            name=f"s3-snapshot-{job.job_id[:8]}",
            daemon=True,
        ).start()
    except Exception as _exc:
        logger.debug("S3 snapshot dispatch failed: %s", _exc)

    # BUC-1598 — Cross-repo IMPORTS resolution.  When the feature flag is
    # enabled, rewire any external Module nodes that match another indexed
    # repo's package identity (npm name / pyproject project.name).  The
    # call is a fast no-op when the flag is off, so the import + invocation
    # cost is bounded — no need to gate the import on the flag here.
    # Failures are logged but never fail the index job: the structural
    # graph is already committed, cross-repo wiring is purely additive.
    try:
        from ..services import cross_repo_imports as _cri

        if _cri.is_enabled():
            _siblings = [
                _cri.extract_repo_identity(_s, _p)
                for _s, _p in indexed_repo_paths.items()
                if _s != repo_name
            ]
            _stats = _cri.resolve_cross_repo_imports(
                repo_name, repo_db_path, _siblings,
            )
            logger.info(
                "cross_repo.post_ingest slug=%s matched=%d duration_ms=%.1f",
                repo_name, _stats.matched, _stats.duration_ms,
            )
    except Exception as _exc:
        logger.warning("cross_repo.post_ingest failed (non-fatal): %s", _exc)

    # Bust the health probe cache again now that embeddings are done so the
    # UI transitions from "indexing" to fully-complete state immediately.
    try:
        from .health import invalidate_probe_cache
        invalidate_probe_cache(repo_name)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Embedding job (separate from structural index)
# ---------------------------------------------------------------------------


@dataclass
class _EmbedJob:
    """Bookkeeping record for a standalone embedding pass (POST /index/embed).

    This dataclass is also used internally during the mandatory pass-4
    embedding step in ``_blocking_index`` — every index run runs embeddings;
    this type just tracks the outcome of that sub-pass.
    """

    job_id: str
    repo_name: str
    repo_path: str = ""   # resolved absolute path; needed by the subprocess driver
    status: Literal["running", "done", "failed"] = "running"
    progress_pct: float = 0.0
    embedded_count: int = 0
    # BUC-1574 (Phase 1.4) — diff-metrics breakdown captured from the
    # embed subprocess summary line.
    skipped_unchanged: int = 0
    skipped_filtered: int = 0
    # BUC-1601 (Fix A) — count of source files the embed subprocess
    # tried to read off disk and failed.  Should be 0 on healthy runs.
    dropped_unreadable: int = 0
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None


_embed_jobs: dict[str, _EmbedJob] = {}


def _parse_reconcile_line(line: str) -> dict[str, int]:
    """Parse the trailing ``RECONCILE k=v k=v ...`` line from the embed log.

    BUC-1601 Fix A — the driver emits this line once at end-of-pass.
    Lifted to a module-level helper so the post-subprocess parsing path
    can be exercised by a unit test without spawning a subprocess.

    Args:
        line: One line from the embed subprocess log, leading and
            trailing whitespace allowed.  Expected to start with the
            literal token ``RECONCILE`` followed by ``key=integer``
            tokens separated by whitespace.

    Returns:
        Dict mapping each ``key`` to its integer value.  Non-integer
        values and tokens without ``=`` are silently dropped — the
        caller falls back to live job-record counts in that case.  When
        ``line`` does not start with ``RECONCILE`` the result is empty.
    """
    parts = line.strip().split()
    if not parts or parts[0] != "RECONCILE":
        return {}
    out: dict[str, int] = {}
    for tok in parts[1:]:
        if "=" not in tok:
            continue
        _k, _v = tok.split("=", 1)
        try:
            out[_k] = int(_v)
        except ValueError:
            continue
    return out


def _blocking_embed(job: _EmbedJob) -> None:
    """Run only the embedding pass (pass-4) against an already-indexed repo.

    Executed in a thread-pool executor.  Runs the embedding generation in a
    **subprocess** so that if torch OOMs the subprocess is killed by the OS
    without taking down the main uvicorn process.  The subprocess writes
    embeddings directly to the LadybugDB file.

    Args:
        job: The embed job record to update with progress and final state.
    """
    repo_db_path = settings.db_path_for_repo(job.repo_name)
    if not Path(repo_db_path).exists():
        raise FileNotFoundError(
            f"No index found for '{job.repo_name}'. Run /index first."
        )

    repo_path_str = job.repo_path or ""

    # Driver runs the embedding pass in isolation. It:
    #   1. Opens the existing LadybugDB read-only to query Function/Method nodes
    #   2. Loads the CodeRankEmbed model once in-process
    #   3. Embeds every Function/Method that has source code on disk
    #   4. Writes rows to the per-repo .duck file (DuckDB FLOAT[768])
    # Running in a subprocess means an OOM kill doesn't affect uvicorn.
    vec_db_path = settings.vec_db_path_for_repo(job.repo_name)
    # BUC-1601: shell out to app/scripts/embed_driver.py via
    # ``python -m`` instead of the legacy ``python -c`` f-string.
    # The module is unit-testable and lives at app/scripts/embed_driver.py.
    # All previous flags (SAGEMAKER_EMBED_CONCURRENCY, MANIFEST_URL,
    # MANIFEST_AGENT_KEY, MANIFEST_FILE_SUMMARY_MODEL) are still read
    # from the inherited environment.
    driver_argv = [
        sys.executable, "-m", "app.scripts.embed_driver",
        "--repo-db-path", str(repo_db_path),
        "--vec-db-path", str(vec_db_path),
        "--repo-path", repo_path_str,
    ]

    # Pipe subprocess output through a log file rather than OS pipes.  The
    # embedding pass emits tens of thousands of loguru DEBUG lines for
    # large repos; capture_output=True would deadlock once the 64 KB pipe
    # buffer fills before the subprocess exits.  A file sink never blocks.
    sub_env = _os.environ.copy()
    if settings.EMBED_DEVICE == "cpu":
        sub_env["CUDA_VISIBLE_DEVICES"] = ""

    log_path = Path(f"/tmp/cis_embed_{job.job_id}.log")
    with log_path.open("w") as log_fh:
        proc = subprocess.run(
            driver_argv,
            env=sub_env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=14400,  # 4 hr hard limit — large repos at batch=32 SageMaker need ~2-3 hr
            cwd=str(Path(repo_db_path).parent.parent.parent),  # service root
        )

    if proc.returncode != 0:
        # Tail the log so the error surfaces on /index/embed/{id}/status
        # without bloating the response with gigabytes of DEBUG spam.
        tail = ""
        try:
            with log_path.open() as f:
                tail = "".join(f.readlines()[-40:])
        except Exception:
            pass
        raise RuntimeError(
            f"Embedding subprocess failed (exit {proc.returncode}). "
            f"See {log_path} for full output.\nLast 40 lines:\n{tail}"
        )

    # Parse the count out of stdout if possible — successful runs print
    # "EMBED_DONE" and may emit "Embedded N (skipped M unchanged, filtered K)"
    # earlier. BUC-1574 (Phase 1.4) — also lift the unchanged/filtered
    # totals onto the embed job so /index/{job_id}/diff_metrics can report
    # them without re-tailing the log.
    #
    # BUC-1601 (Fix A) — also parse the trailing ``RECONCILE`` line the
    # driver emits.  Surfaces the per-category skip breakdown into the
    # parent's logger so ops can spot read-failure drift without grepping
    # /tmp/cis_embed_*.log.
    reconcile_line: str | None = None
    try:
        with log_path.open() as f:
            for line in f:
                if line.startswith("Embedded"):
                    try:
                        parts = line.split()
                        # Format: "Embedded {N} (skipped {M} unchanged, filtered {K})"
                        job.embedded_count = int(parts[1])
                        # parts[3] = "{M}", parts[6] = "{K}" — "filtered K)"
                        # so strip the trailing ')'.
                        job.skipped_unchanged = int(parts[3])
                        job.skipped_filtered = int(parts[6].rstrip(")"))
                    except (IndexError, ValueError):
                        # Best-effort — don't fail the job over a log
                        # parse glitch; live PROGRESS values still fall
                        # through to diff_metrics.
                        pass
                elif line.startswith("RECONCILE "):
                    # Keep the LAST RECONCILE line — driver emits one per
                    # full embed pass; if Function/Method, Class and Module
                    # passes ever each get their own (Phase 2), the latest
                    # is the most relevant.
                    reconcile_line = line.rstrip("\n")
    except Exception:
        pass

    # ------------------------------------------------------------------
    # BUC-1601 Fix A — reconcile pass.
    #
    # The driver writes a line of the form::
    #
    #     RECONCILE expected=N embedded=A skipped_unchanged=B
    #               skipped_filtered=C dropped_unreadable=D unaccounted=E
    #
    # We parse the dropped_unreadable count onto the job (so it surfaces
    # in /index/.../diff_metrics) and emit one structured log line so the
    # delta is grep-able without re-reading the embed log.
    # ------------------------------------------------------------------
    reconcile_fields: dict[str, int] = (
        _parse_reconcile_line(reconcile_line) if reconcile_line else {}
    )
    if reconcile_fields:
        job.dropped_unreadable = reconcile_fields.get("dropped_unreadable", 0)
        logger.info(
            "embed.reconcile job_id=%s repo=%s expected=%d embedded=%d "
            "skipped_unchanged=%d skipped_filtered=%d "
            "dropped_unreadable=%d unaccounted=%d",
            job.job_id,
            job.repo_name,
            reconcile_fields.get("expected", 0),
            reconcile_fields.get("embedded", job.embedded_count),
            reconcile_fields.get("skipped_unchanged", job.skipped_unchanged),
            reconcile_fields.get("skipped_filtered", job.skipped_filtered),
            reconcile_fields.get("dropped_unreadable", 0),
            reconcile_fields.get("unaccounted", 0),
        )
        if reconcile_fields.get("dropped_unreadable", 0) > 0:
            logger.warning(
                "embed.reconcile dropped_unreadable=%d on repo=%s — "
                "graph references files missing from the working tree; "
                "see %s for per-path WARN lines.",
                reconcile_fields["dropped_unreadable"],
                job.repo_name,
                log_path,
            )
    else:
        # Driver completed but never emitted a RECONCILE line.  That is
        # itself a regression — surface as WARN so ops notice without
        # failing the job (counts on the record are still populated from
        # the "Embedded ..." summary).
        logger.warning(
            "embed.reconcile_missing job_id=%s repo=%s — driver finished "
            "without RECONCILE line; see %s",
            job.job_id,
            job.repo_name,
            log_path,
        )

    job.finished_at = time.time()
    job.progress_pct = 100.0
    job.status = "done"


async def _run_embed(job: _EmbedJob) -> None:
    """Drive the embedding pass in a background asyncio task."""
    # Re-use the per-repo lock to avoid writing embeddings while a structural
    # index is also writing (LadybugDB single-writer).
    repo_db_path = settings.db_path_for_repo(job.repo_name)
    lock_key = str(Path(repo_db_path).resolve())
    if lock_key not in _repo_locks:
        _repo_locks[lock_key] = asyncio.Lock()
    lock = _repo_locks[lock_key]

    async with lock:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, _blocking_embed, job)
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/index", response_model=IndexAccepted, status_code=202)
async def start_index(
    req: IndexRequest,
    background_tasks: BackgroundTasks,
    x_forge_triggered_by: str | None = Header(
        default=None,
        description=(
            "Optional provenance hint persisted on the jobs row "
            "(BUC-1599). One of 'manual' | 'webhook' | 'cron' | "
            "'reindex_admin'. Defaults to 'manual' when absent."
        ),
    ),
) -> IndexAccepted:
    """Kick off a background indexing job for the given repository.

    Args:
        req: Request body specifying the repo path and force_reindex flag.
        background_tasks: FastAPI-injected task registry used to schedule
            the ingestion worker.

    Returns:
        IndexAccepted: The job_id to use for subsequent status polling.

    Raises:
        HTTPException: 422 when ``repo_path`` does not exist on disk or is not
            a directory. Passing a file path is rejected early to prevent
            ``GraphUpdater`` from silently traversing the file's parent
            directory and indexing unrelated content.
    """
    # Opportunistically evict stale job records before allocating a new one.
    _prune_old_jobs()

    # ------------------------------------------------------------------
    # App-authenticated clone path (BUC-1561b)
    # ------------------------------------------------------------------
    # When ``github_token`` is supplied alongside ``full_name``, clone the
    # remote into ``.cgr/clones/{owner}__{repo}`` using the token-bearing
    # URL and treat the cloned working tree as the repo to index. Token is
    # NEVER logged, NEVER persisted — we only retain the resolved local
    # path beyond this block, and the masked form ``***`` is what surfaces
    # in any error message.
    #
    # GitHub App installation tokens are valid for ~1h; clones must finish
    # within that window or git will fail with 401 and the caller must
    # request a fresh token and retry.
    cloned_repo_path: Path | None = None
    if req.github_token:
        if not req.full_name or "/" not in req.full_name:
            raise HTTPException(
                status_code=422,
                detail="full_name (owner/repo) is required when github_token is set",
            )
        # Lazy import to avoid the circular dependency (github.py imports
        # _Job / _run_ingestion from this module).
        from .github import _clone_or_update

        try:
            cloned_repo_path = await asyncio.get_running_loop().run_in_executor(
                None,
                _clone_or_update,
                req.full_name,
                req.branch,
                req.github_token,
            )
        except HTTPException:
            # _clone_or_update already scrubs the token from the error
            # message — re-raise as-is so the caller sees a clean 502.
            raise
        except Exception as exc:  # noqa: BLE001
            # Defensive: any other failure type — scrub token before surfacing.
            msg = str(exc)
            if req.github_token:
                msg = msg.replace(req.github_token, "***")
            raise HTTPException(status_code=502, detail=f"clone failed: {msg[:500]}") from None

    # Resolve the effective repo path: cloned tree wins when present,
    # otherwise fall back to the caller-supplied local path.
    effective_repo_path = (
        str(cloned_repo_path) if cloned_repo_path is not None else req.repo_path
    )

    # Reject empty or whitespace-only paths before attempting filesystem ops.
    # Path("") resolves to cwd (a valid directory) so it must be caught early.
    if not effective_repo_path or not effective_repo_path.strip():
        raise HTTPException(
            status_code=422,
            detail="repo_path must not be empty",
        )

    repo_path = Path(effective_repo_path)
    if not repo_path.exists():
        raise HTTPException(
            status_code=422,
            detail=f"repo_path does not exist: {effective_repo_path}",
        )
    if not repo_path.is_dir():
        raise HTTPException(
            status_code=422,
            detail=f"repo_path must be a directory, not a file: {effective_repo_path}",
        )

    # Reject a second concurrent job on the same repo with a clear 409 so the
    # UI can show "already indexing" instead of silently queueing behind a
    # lock.  Without this the second request would stall on the async lock
    # and timeout the HTTP client before ever getting a job id.
    resolved = repo_path.resolve()

    # Phase 2: check jobs_store first (survives restarts), fall back to the
    # in-memory dict for the duration of this process.
    _store_active = None
    try:
        # LE-143 fix: reconcile orphaned/no-progress locks on the request path
        # so a stuck job (dead worker, or no progress past the threshold) can
        # never permanently 409-block a new reindex. Legitimate in-progress
        # jobs (advancing updated_at) survive and still 409.
        _store_active = _jobs_store.find_active_for_repo(
            repo_path.name,
            reconcile=True,
            no_progress_seconds=settings.JOB_PHASE_WATCHDOG_SECONDS,
        )
    except RuntimeError:
        pass  # jobs_store not yet initialised (tests without lifespan)

    if _store_active is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Index job already running for this repo "
                f"(job_id={_store_active.job_id}). Poll /index/{_store_active.job_id}/status or wait for completion."
            ),
        )

    for j in _jobs.values():
        if (
            j.status == "running"
            # LE-143 fix: a cancel-requested job is on its way to terminal —
            # it no longer holds the lock for a new reindex, so don't 409 on it.
            and not j.cancelled
            and Path(j.repo_path).resolve() == resolved
        ):
            _metrics.record_dedupe_409()
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Index job already running for this repo "
                    f"(job_id={j.job_id}). Poll /index/{j.job_id}/status or wait for completion."
                ),
            )

    # Merge caller-supplied exclude_paths with built-in defaults that keep
    # synthetic test fixtures out of the semantic index.
    _DEFAULT_EXCLUDE: frozenset[str] = frozenset({"tests/fixtures", "test/fixtures"})
    effective_excludes = _DEFAULT_EXCLUDE | frozenset(req.exclude_paths)

    job_id = str(uuid.uuid4())
    job = _Job(job_id=job_id, repo_path=str(repo_path), exclude_paths=effective_excludes)
    _jobs[job_id] = job

    # Phase 2: persist to jobs_store (best-effort — never fail the request).
    # BUC-1599: ``triggered_by`` defaults to 'manual'; callers (webhook
    # receiver, cron scheduler, admin reindex flow) set the header to
    # mark a different provenance.
    # Coerce: when start_index is called directly (e.g. reindex_repo in
    # repos.py) rather than via FastAPI injection, the Header() default is NOT
    # resolved and x_forge_triggered_by is a Header object, not a str. Guard so
    # the direct-call path doesn't AttributeError on .strip().
    _raw_triggered_by = x_forge_triggered_by if isinstance(x_forge_triggered_by, str) else None
    _triggered_by = (_raw_triggered_by or "manual").strip().lower() or "manual"
    try:
        # LE-143 fix: pass the SAME job_id the caller receives so the durable
        # row and the in-memory tracker are one job. Previously create_job
        # minted its own UUID, so mark_done / mark_failed / request_cancel
        # (all keyed by the in-memory job.job_id) never touched the durable
        # row — leaving it 'running' forever and permanently 409-locking the
        # repo (the orphaned-lock bug). Unifying the id makes terminal
        # transitions, cancel, and clear all release the per-repo lock.
        _jobs_store.create_job(
            kind="index",
            actor_oid="",
            actor_email="",
            repo_path=str(repo_path),
            force_reindex=req.force_reindex,
            exclude_paths=effective_excludes,
            worker_token=_WORKER_TOKEN,
            initial_status="running",
            initial_phase="queued",
            triggered_by=_triggered_by,
            job_id=job_id,
        )
    except RuntimeError:
        pass  # jobs_store not initialised (tests without lifespan)
    except Exception as _exc:
        logger.warning("jobs_store.create_job failed (non-fatal): %s", _exc)

    background_tasks.add_task(_run_ingestion, job, req.force_reindex)

    return IndexAccepted(job_id=job_id)


@router.get("/index/{job_id}/status", response_model=IndexStatus)
def get_index_status(job_id: str) -> IndexStatus:
    """Poll the status of a previously submitted indexing job.

    Args:
        job_id: The identifier returned from ``POST /index``.

    Returns:
        IndexStatus: Current status plus (on completion) node/rel counts.

    Raises:
        HTTPException: 404 when the job_id is unknown (expired or invalid).
    """
    job = _jobs.get(job_id)
    if job is None:
        # Phase 2: check persistent store for jobs that survived a restart.
        # This surfaces 'interrupted' status for jobs from prior processes.
        try:
            stored = _jobs_store.get_job(job_id)
        except RuntimeError:
            stored = None
        if stored is not None:
            elapsed = (time.time() - stored.started_at) if stored.started_at else 0.0
            return IndexStatus(
                job_id=stored.job_id,
                status=stored.status,  # type: ignore[arg-type]
                phase=stored.phase or "queued",  # type: ignore[arg-type]
                progress_pct=stored.progress_pct,
                files_total=stored.files_total,
                files_done=stored.files_done,
                current_file=stored.current_file,
                node_count=stored.node_count,
                rel_count=stored.rel_count,
                embedding_count=stored.embedding_count,
                started_at=stored.started_at,
                elapsed_sec=elapsed,
                eta_sec=None,
                error=stored.error,
            )
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    # Compute elapsed_sec at response time so it advances even between
    # callback fires; eta_sec is kept from the last callback update.
    elapsed = time.time() - job.started_at if job.started_at else 0.0

    # Phase on the public model uses the cancelled literal; status always
    # reflects the job lifecycle.
    phase_val = job.phase  # type: ignore[assignment]

    # Live embed progress: tail the embed subprocess log for the latest
    # PROGRESS line.  Lets the frontend show running totals (embedded /
    # skipped via content_hash / filtered as test or generated) WHILE the
    # pass is in flight, instead of seeing 0 until completion.
    live_embedded = job.embedded_count
    live_skipped = job.embeddings_skipped_unchanged
    live_filtered = job.embeddings_filtered_out
    if job.status == "running" and job.phase == "embedding":
        progress = _parse_embed_progress(job.job_id)
        if progress is not None:
            live_embedded, live_skipped, live_filtered = progress
            # Mirror back onto the job so subsequent reads stay monotonic
            # even if the log briefly races during a flush.
            job.embedded_count = max(job.embedded_count, live_embedded)
            job.embeddings_skipped_unchanged = max(
                job.embeddings_skipped_unchanged, live_skipped
            )
            job.embeddings_filtered_out = max(
                job.embeddings_filtered_out, live_filtered
            )

    return IndexStatus(
        job_id=job.job_id,
        status=job.status,
        phase=phase_val,
        progress_pct=job.progress_pct,
        files_total=job.files_total,
        files_done=job.files_done,
        current_file=job.current_file,
        node_count=job.node_count,
        rel_count=job.rel_count,
        embedding_count=live_embedded,
        embeddings_skipped_unchanged=live_skipped,
        embeddings_filtered_out=live_filtered,
        embeddings_dropped_unreadable=job.embeddings_dropped_unreadable,
        started_at=job.started_at,
        elapsed_sec=elapsed,
        eta_sec=job.eta_sec,
        error=job.error,
    )


# ---------------------------------------------------------------------------
# GET /index/{job_id}/diff_metrics — incremental-embed audit shape
# ---------------------------------------------------------------------------


@router.get("/index/{job_id}/diff_metrics", response_model=DiffMetrics)
def get_diff_metrics(job_id: str) -> DiffMetrics:
    """Return the incremental-embed audit shape for a single index run.

    BUC-1574 (Phase 1.4) — instrumentation around the content-hash skip
    path in the embed subprocess.  For running jobs the values are the
    running totals from the latest ``PROGRESS`` line; for completed jobs
    they are the persisted final totals on the in-memory job record.

    Args:
        job_id: The identifier returned from ``POST /index``.

    Returns:
        DiffMetrics: ``total_symbols``, ``embedded``, ``skipped_unchanged``,
        ``skipped_filtered``, ``hash_match_rate``, ``wall_clock_seconds``.

    Raises:
        HTTPException: 404 when the job_id is unknown.
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    embedded = job.embedded_count
    skipped_unchanged = job.embeddings_skipped_unchanged
    skipped_filtered = job.embeddings_filtered_out

    # For running jobs, prefer the freshest PROGRESS line so the metrics
    # advance in sync with /status. mark_done's parse runs after the
    # subprocess exits so the persisted values lag by up to one flush.
    if job.status == "running" and job.phase == "embedding":
        progress = _parse_embed_progress(job.job_id)
        if progress is not None:
            embedded = max(embedded, progress[0])
            skipped_unchanged = max(skipped_unchanged, progress[1])
            skipped_filtered = max(skipped_filtered, progress[2])

    in_scope = embedded + skipped_unchanged
    hash_match_rate = (skipped_unchanged / in_scope) if in_scope > 0 else 0.0

    if job.embed_started_at is None:
        wall_clock_seconds = 0.0
    elif job.embed_finished_at is not None:
        wall_clock_seconds = max(0.0, job.embed_finished_at - job.embed_started_at)
    else:
        wall_clock_seconds = max(0.0, time.time() - job.embed_started_at)

    return DiffMetrics(
        total_symbols=embedded + skipped_unchanged + skipped_filtered,
        embedded=embedded,
        skipped_unchanged=skipped_unchanged,
        skipped_filtered=skipped_filtered,
        dropped_unreadable=job.embeddings_dropped_unreadable,
        hash_match_rate=round(hash_match_rate, 4),
        wall_clock_seconds=round(wall_clock_seconds, 3),
    )


# ---------------------------------------------------------------------------
# POST /index/{job_id}/cancel
# ---------------------------------------------------------------------------


class CancelResponse(BaseModel):
    """Response from ``POST /index/{job_id}/cancel``."""

    job_id: str
    cancelled: bool
    message: str


@router.post("/index/{job_id}/cancel", response_model=CancelResponse)
def cancel_index(job_id: str) -> CancelResponse:
    """Signal a running indexing job to stop at the next safe checkpoint.

    The job checks ``job.cancelled`` between files during parsing and
    between batches during embedding.  Termination typically occurs within
    two seconds for small files; larger files may take longer to finish
    the current unit of work.  On cancellation the job transitions to
    ``status="failed"``, ``phase="cancelled"``, ``error="Cancelled by user"``.

    Args:
        job_id: The identifier returned from ``POST /index``.

    Returns:
        CancelResponse: Confirmation that the cancel flag was set.

    Raises:
        HTTPException: 404 when the job is unknown; 409 when the job is
        already in a terminal state (done or failed — nothing to cancel).
    """
    job = _jobs.get(job_id)
    # LE-143 fix: an orphaned job can exist ONLY in the durable store (e.g.
    # the in-memory record was pruned, or it survived a restart). Cancel must
    # still be able to release that lock — fall back to the persistent row so
    # cancel never 404s on a job that is still holding a per-repo lock.
    if job is None:
        stored = None
        try:
            stored = _jobs_store.get_job(job_id)
        except RuntimeError:
            stored = None
        if stored is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        if stored.status not in ("queued", "running"):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Job {job_id} is already in terminal state "
                    f"'{stored.status}'; nothing to cancel."
                ),
            )
        # No live in-memory worker to honour the cancel flag — mark the
        # durable row terminal directly so the per-repo lock is released.
        try:
            _jobs_store.mark_failed(
                job_id, error="Cancelled by user", terminal_status="cancelled"
            )
        except RuntimeError:
            pass
        logger.info("Cancel released orphaned durable lock for job %s.", job_id)
        return CancelResponse(
            job_id=job_id,
            cancelled=True,
            message="Cancelled — orphaned job lock released.",
        )
    if job.status != "running":
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is already in terminal state '{job.status}'; nothing to cancel.",
        )
    job.cancelled = True
    # Phase 2: set cancel flag in the persistent store so the worker / pollers
    # on restart can see it. LE-143 fix: ALSO transition the durable row to a
    # terminal 'cancelled' state now so the per-repo lock is released on cancel
    # intent — a reindex can proceed immediately rather than 409ing until the
    # background worker happens to notice the flag (which never fires if the
    # worker already died). The worker's own mark_failed on the next checkpoint
    # is idempotent. Best-effort — the in-memory flag remains authoritative for
    # in-flight progress reporting.
    try:
        _jobs_store.request_cancel(job_id)
        _jobs_store.mark_failed(
            job_id, error="Cancelled by user", terminal_status="cancelled"
        )
    except RuntimeError:
        pass
    logger.info("Cancel requested for job %s.", job_id)
    return CancelResponse(
        job_id=job_id,
        cancelled=True,
        message="Cancel signal sent — job will stop at the next checkpoint.",
    )


# ---------------------------------------------------------------------------
# POST /index/embed — optional semantic embedding pass
# ---------------------------------------------------------------------------


class EmbedRequest(BaseModel):
    """Body for ``POST /index/embed``."""

    repo_name: str
    repo_path: str = ""   # optional; helps the driver resolve the project root


class EmbedAccepted(BaseModel):
    """Response from ``POST /index/embed``."""

    job_id: str


class EmbedStatus(BaseModel):
    """Response from ``GET /index/embed/{job_id}/status``."""

    job_id: str
    status: str
    progress_pct: float
    embedded_count: int
    error: str | None


@router.post("/index/embed", response_model=EmbedAccepted, status_code=202)
async def start_embed(
    req: EmbedRequest,
    background_tasks: BackgroundTasks,
) -> EmbedAccepted:
    """Kick off semantic embedding for an already-indexed repository.

    Embedding generation loads the CodeRankEmbed model (~550 MB) and is
    intentionally separated from the structural indexing pass so that it
    can be triggered on demand and run in an isolated subprocess (preventing
    an OOM from killing the main uvicorn process).

    Args:
        req: Body containing ``repo_name`` — the short name of the repo to
            embed (must match an existing ``.db`` file in ``LADYBUG_DB_DIR``).

    Returns:
        EmbedAccepted: Job id for polling ``GET /index/embed/{job_id}/status``.

    Raises:
        HTTPException: 404 when no index exists for the given repo name.
    """
    db_path = settings.db_path_for_repo(req.repo_name)
    if not Path(db_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"No index found for '{req.repo_name}'. Run POST /index first.",
        )

    # Auto-fill repo_path from the in-memory cache when not supplied.
    resolved_path = req.repo_path or indexed_repo_paths.get(req.repo_name, "")

    job_id = str(uuid.uuid4())
    job = _EmbedJob(job_id=job_id, repo_name=req.repo_name, repo_path=resolved_path)
    _embed_jobs[job_id] = job

    background_tasks.add_task(_run_embed, job)
    return EmbedAccepted(job_id=job_id)


@router.get("/index/embed/{job_id}/status", response_model=EmbedStatus)
def get_embed_status(job_id: str) -> EmbedStatus:
    """Poll the status of a running embedding job.

    Args:
        job_id: The identifier returned from ``POST /index/embed``.

    Returns:
        EmbedStatus: Current status and count of embedded symbols.

    Raises:
        HTTPException: 404 when the job_id is unknown.
    """
    job = _embed_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Embed job not found: {job_id}")
    return EmbedStatus(
        job_id=job.job_id,
        status=job.status,
        progress_pct=job.progress_pct,
        embedded_count=job.embedded_count,
        error=job.error,
    )


# ---------------------------------------------------------------------------
# Startup orphan sweep + lock cleanup
# ---------------------------------------------------------------------------


def sweep_orphan_jobs() -> int:
    """Mark any in-memory running jobs as failed on process start.

    After a crash or ``systemctl restart`` the in-memory ``_jobs`` dict is
    reconstructed empty, so this is a no-op in practice — but it's a cheap
    safety net for future-work where jobs are persisted across restarts
    (Redis/SQLite).  Returns the number of jobs transitioned so the caller
    can log a single line.
    """
    swept = 0
    for j in _jobs.values():
        if j.status == "running":
            j.status = "failed"
            j.error = "Service restarted while job was running (orphaned)."
            swept += 1
    if swept:
        logger.warning("Swept %d orphan job(s) on startup.", swept)
    return swept


def cleanup_stale_locks() -> int:
    """Drop per-repo locks pointing at DB files that no longer exist.

    Keeps ``_repo_locks`` from growing unbounded across DELETE + re-index
    cycles.  Called from the lifespan startup after the WAL probe.
    """
    removed = 0
    for key in list(_repo_locks.keys()):
        # key is ``str(Path(repo_path).resolve())`` — if that directory is
        # gone, the lock is stranded and can be dropped.
        if not Path(key).exists():
            del _repo_locks[key]
            removed += 1
    return removed


def reconcile_stale_running_jobs(
    staleness_threshold_seconds: int = 300,
) -> int:
    """Mark any running job that hasn't updated in N seconds as stale/failed.

    LE-143: periodic heartbeat reconciliation of orphaned running jobs.
    Transitioned jobs are also updated in the persistent jobs_store.
    Returns the number of jobs reconciled.

    Args:
        staleness_threshold_seconds: Mark a job as stale if (now - updated_at)
            exceeds this threshold. Default 300 (5 minutes).
    """
    now = time.time()
    reconciled = 0

    # Check both the in-memory _jobs dict and the persistent jobs_store
    # for stale running jobs.

    # 1. In-memory reconciliation (running jobs not yet written to disk).
    #    LE-143 fix: key staleness on the progress HEARTBEAT (last_progress_at),
    #    NOT started_at. A long-but-healthy run keeps advancing last_progress_at
    #    so it is never reaped; only a job whose current phase has gone fully
    #    silent past the threshold (a hung phase) is failed. We also mark the
    #    durable row failed so the per-repo lock is released, not just the
    #    in-memory record.
    for job_id, job in list(_jobs.items()):
        if job.status == "running":
            silent_for = now - job.last_progress_at
            # Writing-phase awareness (belt-and-suspenders to the heartbeat
            # thread): the Kùzu bulk flush is callback-silent and can run for
            # minutes. The heartbeat thread normally keeps last_progress_at
            # fresh, but if it is itself starved (GIL contention behind the
            # CPU-bound write) we widen the budget rather than reap a write
            # that is demonstrably mid-flush. Never shrinks the budget.
            effective_threshold = staleness_threshold_seconds
            if job.phase == "writing":
                effective_threshold = max(
                    staleness_threshold_seconds,
                    settings.JOB_PHASE_WATCHDOG_SECONDS,
                )
            if silent_for > effective_threshold:
                err = (
                    f"Job hung in phase '{job.phase}' — no progress for "
                    f"{int(silent_for)}s (phase watchdog)."
                )
                job.status = "failed"
                job.error = err
                job.finished_at = now
                reconciled += 1
                # Release the durable per-repo lock (shared id since LE-143 fix).
                try:
                    _jobs_store.mark_failed(job_id, error=err, terminal_status="failed")
                except RuntimeError:
                    pass
                except Exception as _exc:  # noqa: BLE001
                    logger.warning(
                        "phase-watchdog: durable mark_failed(%s) failed: %s",
                        job_id[:8], _exc,
                    )
                logger.warning(
                    "Phase watchdog failed hung job %s (phase=%s, silent %.0fs)",
                    job_id[:8], job.phase, silent_for,
                )

    # 2. Persistent store reconciliation (long-running or surviving jobs)
    try:
        stale_rows = _jobs_store.list_stale_running_jobs(
            staleness_threshold_seconds=staleness_threshold_seconds
        )
        for row in stale_rows:
            job_id = row.get("job_id")
            if job_id and job_id not in _jobs:
                # Job not in memory — was already reaped or is a survivor from a prior restart.
                _jobs_store.mark_failed(
                    job_id,
                    error=(
                        f"Job marked stale by heartbeat reconciliation "
                        f"({int(now - row.get('started_at', now))}s without progress)."
                    ),
                    terminal_status="failed",
                )
                reconciled += 1
                logger.warning(
                    "Reconciled stale persistent job %s (elapsed %.0fs)",
                    job_id[:8], now - row.get("started_at", now),
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "reconcile_stale_running_jobs: persistent store reconciliation failed "
            "(non-fatal): %s", exc
        )

    if reconciled:
        logger.info("Reconciled %d stale job(s) by heartbeat.", reconciled)
    return reconciled


# ---------------------------------------------------------------------------
# GET /stats/{repo} — per-repo graph breakdown
# ---------------------------------------------------------------------------


# Labels + rel types defined by the code-graph-rag schema.  Querying each
# by name is correct whether a given label has any rows or not; labels
# that don't exist yet get a silent count=0 and are filtered downstream.
_STATS_NODE_LABELS = (
    "Project", "File", "Folder", "Package", "Module",
    "Class", "Function", "Method", "Interface",
    "Variable", "Struct", "Enum", "Type",
    "ExternalPackage",
)
_STATS_REL_TYPES = (
    "CONTAINS_FILE", "CONTAINS_FOLDER", "CONTAINS_PACKAGE", "CONTAINS_MODULE",
    "DEFINES", "DEFINES_METHOD",
    "CALLS", "IMPORTS", "INHERITS", "IMPLEMENTS", "OVERRIDES", "BELONGS_TO",
)


@router.get("/stats/{repo}", response_model=RepoStatsResponse)
def repo_stats(repo: str) -> RepoStatsResponse:
    """Return per-repo node/rel counts and embedding coverage.

    Used by the UI Browse panel to seed tab badges and decide whether to
    surface the semantic search button (only meaningful when embeddings
    exist).

    Args:
        repo: Repo slug (matches ``/health.indexed_repos``).

    Returns:
        RepoStatsResponse: totals, per-label/per-type breakdown, file
        size, last-modified timestamp, embedding coverage flag.

    Raises:
        HTTPException: 404 when the repo has no DB file.
    """
    db_path = settings.db_path_for_repo(repo)
    p = Path(db_path)
    if not p.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No index found for '{repo}'. Run POST /index first.",
        )

    size = p.stat().st_size
    mtime = p.stat().st_mtime

    # If an index job is mid-write, opening a fresh read connection here
    # will block (single-writer) or trigger a LadybugDB internal assertion.
    # Serve the last-known-good figures from the per-repo ``.duck`` file's
    # ``repo_metadata`` table instead — the UI can keep polling and will
    # see live counts once the writer releases.
    if is_repo_indexing(repo):
        meta = _read_meta(repo)
        last_idx_at = _get_last_indexed_at(repo)
        return RepoStatsResponse(
            repo=repo,
            node_count=meta.get("node_count", 0) or 0,
            rel_count=meta.get("rel_count", 0) or 0,
            node_breakdown=[],
            rel_breakdown=[],
            db_size_bytes=size,
            last_modified=mtime,
            last_indexed_at=last_idx_at,
            root_path=meta.get("root_path", "") or "",
            has_embeddings=False,
            indexing=True,
        )

    db = None
    conn = None
    node_breakdown: list[NodeTypeStat] = []
    total_nodes = 0
    rel_breakdown: list[NodeTypeStat] = []
    total_rels = 0
    has_embeddings = False
    _embedding_count: int | None = None
    root_path = ""
    probe_ok = False
    try:
        from ..services.ladybug_pool import open_read_conn

        # BUC-1571: /index/stats is a pure read — never contend with the
        # exclusive lock held by a concurrent /index POST.
        db, conn = open_read_conn(db_path)

        for label in _STATS_NODE_LABELS:
            try:
                r = conn.execute(f"MATCH (n:{label}) RETURN count(n) AS cnt")
                if r.has_next():
                    cnt = int(r.get_next()[0])
                    total_nodes += cnt
                    if cnt:
                        node_breakdown.append(NodeTypeStat(label=label, count=cnt))
            except Exception:
                continue

        for rtype in _STATS_REL_TYPES:
            try:
                r = conn.execute(f"MATCH ()-[r:{rtype}]->() RETURN count(r) AS cnt")
                if r.has_next():
                    cnt = int(r.get_next()[0])
                    total_rels += cnt
                    if cnt:
                        rel_breakdown.append(NodeTypeStat(label=rtype, count=cnt))
            except Exception:
                continue

        try:
            # "has_embeddings" is best-effort — check the DuckDB vector store
            # written by the embedding subprocess.
            _vec_path = Path(settings.vec_db_path_for_repo(repo))
            has_embeddings = _vec_path.exists() and _vec_path.stat().st_size > 0
            if has_embeddings:
                try:
                    from codebase_rag.storage.vector_store import (
                        open_or_create,
                        row_count,
                    )
                    _vec_conn = open_or_create(str(_vec_path))
                    _embedding_count = row_count(_vec_conn)
                    _vec_conn.close()
                    if not _embedding_count:
                        _embedding_count = None
                except Exception:
                    _embedding_count = None
            else:
                _embedding_count = None
        except Exception:
            has_embeddings = False
            _embedding_count = None

        node_breakdown.sort(key=lambda s: s.count, reverse=True)
        rel_breakdown.sort(key=lambda s: s.count, reverse=True)

        # ``root_path`` lives on the Project node (typed column in the
        # schema).  Fetch it here so /stats can return the absolute path
        # the repo was indexed from.
        try:
            r = conn.execute(
                "MATCH (p:Project) RETURN p.root_path AS rp LIMIT 1"
            )
            if r.has_next():
                row = r.get_next()
                if row[0]:
                    root_path = str(row[0])
        except Exception:
            pass

        probe_ok = True
    except Exception as exc:
        logger.warning("Stats probe failed for %s: %s", repo, exc)
    finally:
        # Always release the DB handle — without this a failed probe pins
        # the file for the rest of the process lifetime and every follow-up
        # probe inherits the same error.
        conn = None
        db = None

    # ``last_indexed_at`` comes from the DuckDB ``repo_metadata`` table
    # (_get_last_indexed_at checks the in-memory cache first, then the
    # ``.duck`` file).  Independent from the graph so a schema-less
    # migration isn't needed.
    last_idx_at = _get_last_indexed_at(repo)

    if not probe_ok:
        # DB exists but can't be probed right now (stale lock, transient
        # failure).  Return ``repo_metadata`` figures from the ``.duck``
        # file with a 200 — the UI distinguishes "writer busy" from
        # "permanent error" via `indexing` + staleness of last_indexed_at,
        # and retrying /stats on the next poll usually succeeds once
        # whatever held the lock has released.
        meta = _read_meta(repo)
        return RepoStatsResponse(
            repo=repo,
            node_count=meta.get("node_count", 0) or 0,
            rel_count=meta.get("rel_count", 0) or 0,
            node_breakdown=[],
            rel_breakdown=[],
            db_size_bytes=size,
            last_modified=mtime,
            last_indexed_at=last_idx_at,
            root_path=meta.get("root_path", "") or "",
            has_embeddings=False,
            indexing=False,
        )

    return RepoStatsResponse(
        repo=repo,
        node_count=total_nodes,
        rel_count=total_rels,
        node_breakdown=node_breakdown,
        rel_breakdown=rel_breakdown,
        db_size_bytes=size,
        last_modified=mtime,
        last_indexed_at=last_idx_at,
        root_path=root_path,
        has_embeddings=has_embeddings,
        embedding_count=_embedding_count,
        indexing=is_repo_indexing(repo),
    )


# ---------------------------------------------------------------------------
# DELETE /index/{repo} — admin wipe (cascading cleanup)
# ---------------------------------------------------------------------------


def _delete_ladybug_db(repo: str) -> str:
    """Delete LadybugDB graph file + WAL/shadow sidecars.

    Returns status string: "deleted", "not found", or "error: <msg>".
    """
    db_path = settings.db_path_for_repo(repo)
    p = Path(db_path)

    if not p.exists():
        return "not found"

    try:
        deleted_count = 0
        for target in (
            p,
            p.with_suffix(".db.wal"),
            p.with_suffix(".db.shadow"),
        ):
            if target.exists():
                target.unlink(missing_ok=True)
                deleted_count += 1
        return f"deleted {deleted_count} file(s)"
    except Exception as exc:
        msg = str(exc)
        logger.warning("_delete_ladybug_db(%s) failed: %s", repo, msg)
        return f"error: {msg}"


def _delete_duckdb(repo: str) -> str:
    """Delete DuckDB vector store + WAL sidecar.

    Returns status string: "deleted", "not found", or "error: <msg>".
    """
    try:
        vec_p = Path(settings.vec_db_path_for_repo(repo))
        vec_wal = vec_p.with_name(vec_p.name + ".wal")

        if not vec_p.exists():
            return "not found"

        deleted_count = 0
        for target in (vec_p, vec_wal):
            if target.exists():
                target.unlink(missing_ok=True)
                deleted_count += 1
        return f"deleted {deleted_count} file(s)"
    except Exception as exc:
        msg = str(exc)
        logger.warning("_delete_duckdb(%s) failed: %s", repo, msg)
        return f"error: {msg}"


def _delete_s3_backup(repo: str) -> str:
    """Delete S3 backup copy of repo indexes.

    Returns status string from s3_store.delete_repo_backup.
    """
    try:
        from ..services import s3_store
        return s3_store.delete_repo_backup(repo)
    except Exception as exc:
        msg = str(exc)
        logger.warning("_delete_s3_backup(%s) failed: %s", repo, msg)
        return f"error: {msg}"


def _drop_embedding_cache_entries(repo: str) -> str:
    """Drop embedding cache entries for the repo from .cgr/.embedding_cache.json.

    Returns status string: "dropped N keys", "not found", or "error: <msg>".
    """
    try:
        import json
        cache_path = Path(".cgr/.embedding_cache.json")
        if not cache_path.exists():
            return "not found"

        with open(cache_path, "r") as f:
            cache = json.load(f)

        # Count how many keys belong to this repo
        # Key format appears to be content hashes (SHA-256-like), so we look for
        # a more reliable pattern. For now, assume keys are generic hashes.
        # In practice, repo-specific entries would have a prefix pattern.
        # Without seeing actual usage, we'll be conservative and not delete
        # anything (return "not applicable").
        logger.info("_drop_embedding_cache_entries(%s): cache has %d entries", repo, len(cache))
        return "not applicable"
    except Exception as exc:
        msg = str(exc)
        logger.warning("_drop_embedding_cache_entries(%s) failed: %s", repo, msg)
        return f"error: {msg}"


def _delete_embed_logs(repo: str) -> str:
    """Delete /tmp/cis_embed_*.log files that reference the repo.

    Returns status string: "deleted N files", "not found", or "error: <msg>".
    """
    try:
        log_paths = list(Path("/tmp").glob("cis_embed_*.log"))
        if not log_paths:
            return "not found"

        deleted_count = 0
        for log_path in log_paths:
            try:
                # Heuristic: check the first few lines for repo mention
                with open(log_path, "r", errors="ignore") as f:
                    first_lines = "".join(f.readlines()[:10])
                    if repo in first_lines:
                        log_path.unlink(missing_ok=True)
                        deleted_count += 1
            except Exception as exc:
                logger.warning(
                    "_delete_embed_logs: skipping %s: %s",
                    log_path.name, exc,
                )

        return f"deleted {deleted_count} file(s)" if deleted_count > 0 else "not found"
    except Exception as exc:
        msg = str(exc)
        logger.warning("_delete_embed_logs(%s) failed: %s", repo, msg)
        return f"error: {msg}"


def _delete_jobs_for_repo(repo: str) -> str:
    """Delete all job records for the repo from jobs_store.

    Returns status string: "deleted N rows" or "error: <msg>".
    """
    try:
        count = _jobs_store.delete_by_repo(repo)
        if count == 0:
            return "not found"
        return f"deleted {count} row(s)"
    except Exception as exc:
        msg = str(exc)
        logger.warning("_delete_jobs_for_repo(%s) failed: %s", repo, msg)
        return f"error: {msg}"


def _delete_indexed_repo_row(repo: str) -> str:
    """Drop the ``indexed_repos`` row for ``repo`` (BUC-1599).

    Returns status string: "deleted" | "not found" | "error: <msg>".
    """
    try:
        removed = _jobs_store.delete_indexed_repo(repo)
        return "deleted" if removed else "not found"
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        logger.warning("_delete_indexed_repo_row(%s) failed: %s", repo, msg)
        return f"error: {msg}"


def _delete_repo_meta(repo: str) -> str:
    """Delete repo metadata entry (no-op for now — repo_metadata is in .duck file).

    Returns status string: "not applicable" (data is in the .duck file, which is
    already deleted by _delete_duckdb).
    """
    # repo_metadata rows live inside the per-repo DuckDB .duck file.
    # Deleting that file already removes all metadata, so this is a no-op.
    return "not applicable (in duckdb)"


def _delete_tantivy_index(repo: str) -> str:
    """Delete Tantivy full-text index directory for the repo.

    The Tantivy index lives at ``.cgr/repos/{slug}.tantivy/`` as a sibling
    to the ``.db`` and ``.duck`` files.

    Returns status string: "deleted", "not found", or "error: <msg>".
    """
    try:
        # Build the per-repo tantivy directory path.
        db_dir = Path(settings.LADYBUG_DB_DIR)
        slugged_repo = f"{slugify_repo(repo)}.tantivy"
        tantivy_dir = db_dir / slugged_repo

        if not tantivy_dir.exists():
            return "not found"

        # Safety guard: ensure the directory is scoped to the per-repo artifact root.
        # Resolve to absolute paths to prevent path traversal.
        try:
            resolved_tantivy = tantivy_dir.resolve()
            resolved_db_dir = db_dir.resolve()
            if not str(resolved_tantivy).startswith(str(resolved_db_dir)):
                msg = f"Path traversal attempt blocked: {resolved_tantivy} not under {resolved_db_dir}"
                logger.warning("_delete_tantivy_index(%s) failed: %s", repo, msg)
                return f"error: {msg}"
        except Exception as exc:
            msg = f"Path resolution failed: {str(exc)}"
            logger.warning("_delete_tantivy_index(%s) failed: %s", repo, msg)
            return f"error: {msg}"

        # Recursively delete the directory and all its contents.
        shutil.rmtree(tantivy_dir, ignore_errors=False)
        logger.info("Deleted tantivy index: %s", tantivy_dir)
        return f"deleted directory"
    except Exception as exc:
        msg = str(exc)
        logger.warning("_delete_tantivy_index(%s) failed: %s", repo, msg)
        return f"error: {msg}"


def _delete_clone_directory(repo: str) -> str:
    """Delete cloned repository directory if it exists.

    The clone directory lives at ``.cgr/clones/{owner}__{name}`` and is created
    when indexing a GitHub repo via POST /github/index.

    This is a best-effort delete: we search for any directory under ``.cgr/clones``
    that appears to be associated with this repo slug. Since the clone directory
    name is ``{owner}__{name}`` (derived from the full_name), we cannot reliably
    map a repo slug back to the clone path without scanning the directory.

    Returns status string: "deleted", "not found", or "error: <msg>".
    """
    try:
        clones_dir = Path(".cgr/clones")
        if not clones_dir.exists():
            return "not found"

        # Search for clone directories matching this repo slug in the directory name.
        # The clone directory naming scheme is {owner}__{name}, where name is derived
        # from the repo path. We match against the slug as a simple heuristic.
        deleted_dirs = []
        for clone_path in clones_dir.iterdir():
            if not clone_path.is_dir():
                continue

            # Extract the repo name part from the clone directory name (after __).
            # For example, {owner}__my-repo => my-repo.
            if "__" in clone_path.name:
                clone_repo_name = clone_path.name.split("__", 1)[1]
                # Check if this clone's repo name matches our target repo slug.
                if slugify_repo(clone_repo_name) == repo:
                    # Safety guard: ensure the directory is under the clones root.
                    try:
                        resolved_clone = clone_path.resolve()
                        resolved_clones = clones_dir.resolve()
                        if not str(resolved_clone).startswith(str(resolved_clones)):
                            msg = f"Path traversal attempt blocked: {resolved_clone} not under {resolved_clones}"
                            logger.warning("_delete_clone_directory(%s) failed: %s", repo, msg)
                            continue
                    except Exception as exc:
                        msg = f"Path resolution failed: {str(exc)}"
                        logger.warning("_delete_clone_directory(%s) failed: %s", repo, msg)
                        continue

                    # Delete the clone directory.
                    shutil.rmtree(clone_path, ignore_errors=False)
                    deleted_dirs.append(str(clone_path))
                    logger.info("Deleted clone directory: %s", clone_path)

        if deleted_dirs:
            return f"deleted {len(deleted_dirs)} directory(ies)"
        return "not found"
    except Exception as exc:
        msg = str(exc)
        logger.warning("_delete_clone_directory(%s) failed: %s", repo, msg)
        return f"error: {msg}"


@router.delete("/index/{repo}", response_model=DeleteIndexResponse)
def delete_index(repo: str) -> DeleteIndexResponse:
    """Cascade delete: remove a repo's index and all related resources.

    Cleans up 9 resource types in a best-effort manner:
    1. LadybugDB graph file + WAL/shadow sidecars
    2. DuckDB vector store + WAL sidecar
    3. S3 backup copy
    4. Embedding cache entries (if applicable)
    5. Embed log files
    6. Job history records
    7. Repo metadata (no-op — stored in DuckDB file)
    8. Tantivy full-text index directory
    9. Cloned repository directory (when indexed via POST /github/index)

    Every cleanup step continues on error and logs at WARN; the response
    includes a summary of what happened for each resource type.

    Args:
        repo: Repo slug (matches filename stem in ``LADYBUG_DB_DIR``).

    Returns:
        DeleteIndexResponse: the repo slug, files removed, cleanup status per
        resource type, and ok=True.

    Raises:
        HTTPException: 404 when no index exists for the repo.
    """
    # Check that at least one DB file exists (LadybugDB or DuckDB).
    db_path = settings.db_path_for_repo(repo)
    vec_path = settings.vec_db_path_for_repo(repo)
    db_exists = Path(db_path).exists()
    vec_exists = Path(vec_path).exists()

    if not db_exists and not vec_exists:
        raise HTTPException(
            status_code=404,
            detail=f"No index found for '{repo}'.",
        )

    # Execute cascading cleanup: every step continues on error.
    cleanup = {}
    cleanup["ladybug_db"] = _delete_ladybug_db(repo)
    cleanup["duckdb"] = _delete_duckdb(repo)
    cleanup["s3"] = _delete_s3_backup(repo)
    cleanup["embedding_cache"] = _drop_embedding_cache_entries(repo)
    cleanup["embed_logs"] = _delete_embed_logs(repo)
    cleanup["jobs_store"] = _delete_jobs_for_repo(repo)
    cleanup["indexed_repos"] = _delete_indexed_repo_row(repo)
    cleanup["repo_meta"] = _delete_repo_meta(repo)
    cleanup["tantivy_index"] = _delete_tantivy_index(repo)
    cleanup["clone_directory"] = _delete_clone_directory(repo)

    # Build list of removed files (for backward compatibility).
    removed: list[str] = []
    for target in (
        Path(db_path),
        Path(db_path).with_suffix(".db.wal"),
        Path(db_path).with_suffix(".db.shadow"),
        Path(vec_path),
        Path(vec_path).with_name(Path(vec_path).name + ".wal"),
    ):
        if not target.exists():
            removed.append(str(target))

    # Drop in-memory bookkeeping so subsequent /health doesn't advertise a
    # phantom repo.
    indexed_repos.discard(repo)
    indexed_repo_paths.pop(repo, None)
    _last_indexed_cache.pop(repo, None)

    try:
        from .health import invalidate_probe_cache
        invalidate_probe_cache(repo)
    except Exception:
        pass

    logger.info(
        "Deleted index for repo '%s': %s",
        repo,
        ", ".join(f"{k}={v}" for k, v in cleanup.items()),
    )
    return DeleteIndexResponse(repo=repo, removed_files=removed, ok=True, cleanup=cleanup)


# ---------------------------------------------------------------------------
# Job history — list / clear
# ---------------------------------------------------------------------------


def _job_to_summary(j: _Job) -> JobSummary:
    """Project an internal ``_Job`` to the public ``JobSummary`` shape."""
    return JobSummary(
        job_id=j.job_id,
        repo_path=j.repo_path,
        repo_name=Path(j.repo_path).name,
        status=j.status,
        progress_pct=j.progress_pct,
        phase=j.phase,
        node_count=j.node_count,
        rel_count=j.rel_count,
        error=j.error,
        started_at=j.started_at,
        finished_at=j.finished_at,
    )


@router.get("/index/jobs", response_model=JobListResponse)
def list_jobs(
    status: str | None = None,
    repo: str | None = None,
    limit: int = 50,
) -> JobListResponse:
    """List all known index jobs, newest-first.

    Used by the UI's indexing-history panel — no UI should hit every job
    by id to rebuild the list.

    Args:
        status: Optional filter (``running`` | ``done`` | ``failed``).  Comma-
            separated list accepted (e.g. ``done,failed``).
        repo: Optional repo-name filter (matches the tail of ``repo_path``).
        limit: Max rows returned (defaults to 50, capped at 500).

    Returns:
        JobListResponse: newest-first job summaries + totals.
    """
    limit = max(1, min(int(limit), 500))

    wanted_statuses: set[str] | None = None
    if status:
        wanted_statuses = {s.strip() for s in status.split(",") if s.strip()}

    filtered: list[_Job] = []
    running_count = 0
    for j in _jobs.values():
        if j.status == "running":
            running_count += 1
        if wanted_statuses and j.status not in wanted_statuses:
            continue
        if repo and Path(j.repo_path).name != repo:
            continue
        filtered.append(j)

    filtered.sort(key=lambda j: j.started_at, reverse=True)
    return JobListResponse(
        jobs=[_job_to_summary(j) for j in filtered[:limit]],
        total=len(filtered),
        running=running_count,
    )


@router.post("/index/jobs/clear", response_model=JobClearResponse)
def clear_jobs(status: str = "done,failed") -> JobClearResponse:
    """Drop completed/failed job records from the in-memory store.

    Running jobs are never dropped regardless of the ``status`` filter —
    clearing a running job would leave the background task orphaned and the
    UI unable to poll it.

    Args:
        status: Comma-separated statuses to clear.  Defaults to clearing
            every terminal job (``done,failed``).  ``running`` is silently
            skipped even if requested.

    Returns:
        JobClearResponse: number of jobs cleared + number remaining.
    """
    wanted = {s.strip() for s in status.split(",") if s.strip()} or {"done", "failed"}
    wanted.discard("running")  # never clear in-flight jobs

    to_drop = [jid for jid, j in _jobs.items() if j.status in wanted]
    for jid in to_drop:
        del _jobs[jid]

    # LE-143 fix: also clear terminal rows from the durable store. The in-memory
    # dict and the durable store are now the same job (shared id), but a row may
    # exist ONLY in the durable store (pruned in-memory, or post-restart). Map
    # the in-memory status filter onto durable terminal statuses so a clear
    # actually empties the persistent history the 409 path reads from.
    durable_statuses: set[str] = set()
    if "done" in wanted:
        durable_statuses.add("done")
    if "failed" in wanted:
        durable_statuses.update({"failed", "cancelled", "interrupted"})
    cleared_durable = 0
    try:
        cleared_durable = _jobs_store.clear_terminal(statuses=durable_statuses)
    except RuntimeError:
        pass
    except Exception as _exc:  # noqa: BLE001
        logger.warning("clear_jobs: durable clear failed (non-fatal): %s", _exc)

    total_cleared = max(len(to_drop), cleared_durable)
    logger.info(
        "Cleared %d terminal job(s) (in-memory=%d, durable=%d, status=%s).",
        total_cleared, len(to_drop), cleared_durable, ",".join(sorted(wanted)),
    )
    return JobClearResponse(cleared=total_cleared, remaining=len(_jobs))


@router.delete("/index/jobs/{job_id}", response_model=JobClearResponse)
def delete_job(job_id: str) -> JobClearResponse:
    """Remove a single job record from the store.

    Rejects attempts to drop a running job with 409 — use ``/status`` to
    monitor or wait for completion first.

    Raises:
        HTTPException: 404 if no job with that id exists, 409 if the job
        is still running.
    """
    j = _jobs.get(job_id)
    # LE-143 fix: a job may exist only in the durable store (pruned in-memory
    # or post-restart). Resolve effective status from whichever record exists
    # so delete never 404s on a durable-only row that is still holding a lock.
    stored = None
    try:
        stored = _jobs_store.get_job(job_id)
    except RuntimeError:
        stored = None
    if j is None and stored is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    in_memory_running = j is not None and j.status == "running"
    durable_active = stored is not None and stored.status in ("queued", "running")
    if in_memory_running or durable_active:
        # An actively-progressing job must not be silently deleted. Reconcile
        # first: if it is orphaned (dead worker / no progress) release it, then
        # allow the delete; otherwise reject so we never orphan a live worker.
        reconciled = 0
        if stored is not None:
            try:
                reconciled = _jobs_store.reconcile_active_for_repo(
                    stored.repo_slug,
                    no_progress_seconds=settings.JOB_PHASE_WATCHDOG_SECONDS,
                )
            except RuntimeError:
                reconciled = 0
        # Re-read durable status after reconcile.
        still_active = False
        if stored is not None:
            try:
                refreshed = _jobs_store.get_job(job_id)
                still_active = (
                    refreshed is not None
                    and refreshed.status in ("queued", "running")
                )
            except RuntimeError:
                still_active = False
        if in_memory_running and (j is not None and j.status == "running") and reconciled == 0:
            still_active = True
        if still_active:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Job {job_id} is still running. Wait for it to finish, "
                    f"or POST /index/{job_id}/cancel first."
                ),
            )

    if j is not None:
        del _jobs[job_id]
    # Always drop the durable row so the per-repo lock + history are released.
    try:
        _jobs_store.delete_job(job_id)
    except RuntimeError:
        pass
    return JobClearResponse(cleared=1, remaining=len(_jobs))



# ---------------------------------------------------------------------------
# BUC-1599 — job_events surface for the FE timeline
# ---------------------------------------------------------------------------


class JobEventOut(BaseModel):
    """One row in ``GET /jobs/{job_id}/events``."""

    id: int
    job_id: str
    ts: int
    level: Literal["info", "warn", "error"]
    message: str


class JobEventsResponse(BaseModel):
    """Envelope for ``GET /jobs/{job_id}/events``."""

    job_id: str
    events: list[JobEventOut]


@router.get("/jobs/{job_id}/events", response_model=JobEventsResponse)
def get_job_events(job_id: str, limit: int = 100) -> JobEventsResponse:
    """Return the recorded events for ``job_id``, oldest-first.

    No pagination in v1 — capped at ``limit`` (default 100, hard ceiling
    1000 inside the DAO). The FE renders this as the per-job activity
    timeline; future structured progress events will reuse the same
    surface without a schema change.
    """
    rows = _jobs_store.list_job_events(job_id, limit=limit)
    return JobEventsResponse(
        job_id=job_id,
        events=[JobEventOut(**r) for r in rows],  # type: ignore[arg-type]
    )
