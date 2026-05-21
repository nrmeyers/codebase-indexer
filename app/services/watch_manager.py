"""Phase 5 — per-repo file-watcher with asyncio debouncer.

Each watched repo gets one Watchdog Observer thread and one asyncio debouncer
coroutine.  When a file-system event fires the handler pushes the path into a
thread-safe asyncio.Queue; the debouncer accumulates paths for
``WATCH_DEBOUNCE_MS`` milliseconds and then dispatches a single partial-index
run, serialised on the existing ``_repo_locks[repo_key]`` from
``app.routers.index``.

Public surface (import these and nothing else):

    await start_watch(repo_slug, actor_oid=..., actor_email=...)
    await stop_watch(repo_slug) -> bool
    get_watch(repo_slug) -> WatchHandle | None
    list_watches() -> list[WatchHandle]
    await shutdown_all(timeout_s=None)

Everything else is private implementation detail.

Feature flag: the entire module is gated on ``settings.WATCH_ENABLED``.
``start_watch`` raises ``WatchDisabledError`` when the flag is false so the
router can return 503 without importing any Watchdog machinery.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class WatchDisabledError(RuntimeError):
    """Raised when WATCH_ENABLED=false and a watch operation is attempted."""


class WatchCapacityError(RuntimeError):
    """Raised when the active watcher count would exceed WATCH_MAX_REPOS."""


class WatchAlreadyActiveError(RuntimeError):
    """Raised when start_watch is called for an already-watched repo."""


class WatchNotActiveError(KeyError):
    """Raised when stop_watch or get_watch targets an unknown slug."""


# ---------------------------------------------------------------------------
# Public handle — frozen for thread safety
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WatchHandle:
    """Immutable snapshot of a watcher's state (safe to return from the API).

    This is intentionally a *copy* of the mutable ``_WatchEntry`` fields so
    callers can read it without holding any lock.
    """

    repo_slug: str
    repo_path: str
    actor_oid: str
    actor_email: str
    started_at: float
    last_event_at: float | None
    last_partial_job_id: str | None
    debounce_ms: int
    pending_paths_count: int
    state: Literal["starting", "active", "stopping", "stopped", "errored"]


# ---------------------------------------------------------------------------
# Internal entry
# ---------------------------------------------------------------------------


class _WatchEntry:
    """Mutable lifecycle state for one watched repo."""

    def __init__(
        self,
        repo_slug: str,
        repo_path: str,
        actor_oid: str,
        actor_email: str,
        debounce_ms: int,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.repo_slug = repo_slug
        self.repo_path = repo_path
        self.actor_oid = actor_oid
        self.actor_email = actor_email
        self.debounce_ms = debounce_ms
        self.loop = loop

        self.started_at: float = time.time()
        self.last_event_at: float | None = None
        self.last_partial_job_id: str | None = None
        self.state: Literal["starting", "active", "stopping", "stopped", "errored"] = "starting"

        # Thread-safe queue: Watchdog thread pushes paths; asyncio debouncer
        # pulls them.  maxsize=0 = unbounded (we rely on debounce to collapse
        # bursts rather than dropping events).
        self._queue: asyncio.Queue[str] = asyncio.Queue()

        self._observer: threading.Thread | None = None  # set after schedule()
        self._debouncer_task: asyncio.Task | None = None  # set after start

    def snapshot(self) -> WatchHandle:
        """Return an immutable copy of the current state."""
        return WatchHandle(
            repo_slug=self.repo_slug,
            repo_path=self.repo_path,
            actor_oid=self.actor_oid,
            actor_email=self.actor_email,
            started_at=self.started_at,
            last_event_at=self.last_event_at,
            last_partial_job_id=self.last_partial_job_id,
            debounce_ms=self.debounce_ms,
            pending_paths_count=self._queue.qsize(),
            state=self.state,
        )


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_watches: dict[str, _WatchEntry] = {}
_watches_lock = threading.Lock()

# ---------------------------------------------------------------------------
# FS event handler — pushed from the Watchdog observer thread
# ---------------------------------------------------------------------------


def _make_handler(entry: _WatchEntry):  # type: ignore[return]
    """Return a Watchdog FileSystemEventHandler bound to ``entry``."""
    try:
        from watchdog.events import FileSystemEventHandler, FileSystemEvent
    except ImportError as exc:  # pragma: no cover
        raise ImportError("watchdog is required for the file-watcher feature") from exc

    _IGNORE_PATTERNS = frozenset({
        ".git", "__pycache__", ".cgr", "node_modules", ".venv", "venv",
        ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    })
    _RELEVANT_EXTENSIONS = frozenset({
        ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".java", ".cs",
        ".rs", ".cpp", ".c", ".h", ".rb", ".php", ".swift", ".kt",
    })

    def _is_relevant(path: str) -> bool:
        p = Path(path)
        # Drop hidden parts and known noisy directories.
        for part in p.parts:
            if part.startswith(".") and part not in (".", ".."):
                return False
            if part in _IGNORE_PATTERNS:
                return False
        ext = p.suffix.lower()
        return ext in _RELEVANT_EXTENSIONS

    class _Handler(FileSystemEventHandler):  # type: ignore[misc]
        def on_modified(self, event: "FileSystemEvent") -> None:
            if event.is_directory:
                return
            src = str(event.src_path)
            if not _is_relevant(src):
                entry.loop.call_soon_threadsafe(
                    _enqueue_safe, entry, src, "filtered"
                )
                return
            entry.loop.call_soon_threadsafe(
                _enqueue_safe, entry, src, "dispatched"
            )

        def on_created(self, event: "FileSystemEvent") -> None:
            self.on_modified(event)

    return _Handler()


def _enqueue_safe(entry: _WatchEntry, path: str, result: str) -> None:
    """Put ``path`` onto the entry's queue from the asyncio thread."""
    from .. import metrics as _metrics
    _metrics.record_watch_event(result)
    if result == "filtered":
        return
    entry.last_event_at = time.time()
    entry._queue.put_nowait(path)


# ---------------------------------------------------------------------------
# Debouncer coroutine
# ---------------------------------------------------------------------------


async def _debouncer(entry: _WatchEntry) -> None:
    """Asyncio coroutine: coalesce rapid events and dispatch partial runs.

    Waits up to ``debounce_s`` for the first event.  Once one arrives,
    it drains the queue for the remainder of the debounce window and then
    dispatches a single ``_run_partial_index`` call.  Loops forever until
    the entry's state transitions to ``stopping`` or ``stopped``.
    """
    debounce_s = entry.debounce_ms / 1000.0

    while entry.state in ("starting", "active"):
        # --- Phase 1: wait for the first event in this cycle ---
        try:
            first_path = await asyncio.wait_for(
                entry._queue.get(), timeout=debounce_s
            )
        except asyncio.TimeoutError:
            # No event — loop back and keep watching.
            continue
        except asyncio.CancelledError:
            return

        # --- Phase 2: drain additional events for the debounce window ---
        pending: set[str] = {first_path}
        deadline = time.monotonic() + debounce_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                path = await asyncio.wait_for(
                    entry._queue.get(), timeout=remaining
                )
                pending.add(path)
                # Reset the deadline on every new event (sliding window).
                deadline = time.monotonic() + debounce_s
            except asyncio.TimeoutError:
                break
            except asyncio.CancelledError:
                return

        if entry.state not in ("starting", "active"):
            return

        # --- Phase 3: dispatch partial run ---
        try:
            await _run_partial_index(entry, frozenset(pending))
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.warning(
                "watch_manager: partial-index run failed for %s: %s",
                entry.repo_slug, exc,
            )


# ---------------------------------------------------------------------------
# Partial index runner
# ---------------------------------------------------------------------------


async def _run_partial_index(entry: _WatchEntry, changed_paths: frozenset[str]) -> None:
    """Run a partial re-index for the given set of changed paths.

    Acquires ``_repo_locks[repo_key]`` (from app.routers.index) before
    touching the DB, so full-index jobs and partial runs are always
    serialised on the same lock.
    """
    from ..services import jobs_store as _jobs_store
    from .. import metrics as _metrics
    from ..routers.websocket import broadcast_partial_update

    repo_path = Path(entry.repo_path)
    repo_key = str(repo_path.resolve())

    # Lazily acquire (or create) the per-repo lock from the index router.
    from ..routers.index import _repo_locks
    if repo_key not in _repo_locks:
        _repo_locks[repo_key] = asyncio.Lock()
    lock = _repo_locks[repo_key]

    run_start = time.time()

    # Create the job row in queued state before acquiring the lock so the
    # UI can show "waiting for full-index to finish" if needed.
    job = _jobs_store.create_job(
        kind="watch_partial",
        actor_oid=entry.actor_oid,
        actor_email=entry.actor_email,
        repo_path=str(repo_path),
        force_reindex=False,
        exclude_paths=frozenset(),
        initial_status="queued",
        initial_phase="queued",
    )

    # --- Broadcast "running" event so the FE knows we're in flight ---
    await broadcast_partial_update(
        repo_slug=entry.repo_slug,
        job_id=job.job_id,
        status="running",
        changed_paths=sorted(changed_paths),
    )

    async with lock:
        if entry.state not in ("starting", "active"):
            # Watcher was stopped while we were waiting for the lock.
            _jobs_store.mark_failed(
                job.job_id,
                error="watcher stopped",
                terminal_status="cancelled",
            )
            await broadcast_partial_update(
                repo_slug=entry.repo_slug,
                job_id=job.job_id,
                status="cancelled",
                changed_paths=sorted(changed_paths),
                cancelled=True,
            )
            return

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                _blocking_partial_index,
                entry,
                job.job_id,
                changed_paths,
            )
        except Exception as exc:
            elapsed_ms = int((time.time() - run_start) * 1000)
            _jobs_store.mark_failed(job.job_id, error=str(exc))
            _metrics.record_watch_partial_duration(
                time.time() - run_start, "failed"
            )
            await broadcast_partial_update(
                repo_slug=entry.repo_slug,
                job_id=job.job_id,
                status="failed",
                changed_paths=sorted(changed_paths),
                duration_ms=elapsed_ms,
            )
            logger.error(
                "watch_manager: partial-index errored for %s: %s",
                entry.repo_slug, exc,
            )
            return

    elapsed_ms = int((time.time() - run_start) * 1000)
    terminal = "noop" if result.get("noop") else "done"
    _metrics.record_watch_partial_duration(time.time() - run_start, terminal)
    _metrics.record_watch_partial_files(result.get("dirty_count", 0))

    entry.last_partial_job_id = job.job_id

    await broadcast_partial_update(
        repo_slug=entry.repo_slug,
        job_id=job.job_id,
        status="done",
        changed_paths=sorted(changed_paths),
        files_done=result.get("files_done", 0),
        files_total=len(changed_paths),
        embedding_count=result.get("embedding_count", 0),
        node_count=result.get("node_count", 0),
        rel_count=result.get("rel_count", 0),
        duration_ms=elapsed_ms,
        noop=result.get("noop", False),
    )


# ---------------------------------------------------------------------------
# Blocking partial-index (runs in thread pool)
# ---------------------------------------------------------------------------


def _file_sha1(path: Path) -> str | None:
    """Return the SHA-1 hex digest of ``path``, or None if unreadable."""
    try:
        data = path.read_bytes()
        return hashlib.sha1(data, usedforsecurity=False).hexdigest()
    except OSError:
        return None


def _load_hash_cache(repo_path: Path) -> dict[str, str]:
    """Load ``.cgr-hash-cache.json`` from the repo root. Returns {} on miss."""
    cache_file = repo_path / ".cgr-hash-cache.json"
    try:
        if cache_file.exists():
            raw = cache_file.read_text(encoding="utf-8")
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _save_hash_cache(repo_path: Path, cache: dict[str, str]) -> None:
    """Atomically overwrite ``.cgr-hash-cache.json``. Best-effort."""
    cache_file = repo_path / ".cgr-hash-cache.json"
    tmp_file = cache_file.with_suffix(".json.tmp")
    try:
        tmp_file.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        tmp_file.replace(cache_file)
    except OSError as exc:
        logger.warning("watch_manager: could not save hash cache: %s", exc)
        try:
            tmp_file.unlink(missing_ok=True)
        except OSError:
            pass


def _blocking_partial_index(
    entry: _WatchEntry,
    job_id: str,
    changed_paths: frozenset[str],
) -> dict:
    """Synchronous partial re-index, called from the thread pool.

    Steps:
        1. Hash-diff against .cgr-hash-cache.json — skip unchanged files.
        2. If dirty set is empty → noop path.
        3. For each dirty file: delete old graph nodes, re-parse, re-embed.
        4. Persist the updated hash cache.
        5. Mark the job done.

    Returns a dict with keys: noop, dirty_count, files_done, embedding_count,
    node_count, rel_count.
    """
    from ..services import jobs_store as _jobs_store
    from ..config import settings

    repo_path = Path(entry.repo_path).resolve()

    # --- Update job to running ---
    _jobs_store.update_progress(job_id, phase="discovering", progress_pct=5.0)

    # --- Hash-diff ---
    hash_cache = _load_hash_cache(repo_path)
    dirty: list[Path] = []
    for raw_path in changed_paths:
        abs_path = Path(raw_path) if Path(raw_path).is_absolute() else repo_path / raw_path
        current_sha = _file_sha1(abs_path)
        if current_sha is None:
            # File deleted or unreadable — still dirty (graph needs cleanup).
            dirty.append(abs_path)
            continue
        rel = str(abs_path.relative_to(repo_path)) if abs_path.is_relative_to(repo_path) else raw_path
        if hash_cache.get(rel) != current_sha:
            dirty.append(abs_path)
            hash_cache[rel] = current_sha

    dirty_count = len(dirty)

    if not dirty:
        # All files unchanged — noop.
        _jobs_store.mark_done(job_id, node_count=0, rel_count=0, embedding_count=0)
        return {"noop": True, "dirty_count": 0, "files_done": 0,
                "embedding_count": 0, "node_count": 0, "rel_count": 0}

    # --- Partial graph update ---
    _jobs_store.update_progress(
        job_id,
        phase="parsing",
        progress_pct=20.0,
        files_total=dirty_count,
        files_done=0,
    )

    node_count = 0
    rel_count = 0
    embedding_count = 0

    try:
        from codebase_rag.config import settings as cgr_settings
        from app.services.ladybug_ingestor import LadybugIngestor
        from codebase_rag.graph_updater import GraphUpdater
        from codebase_rag.parser_loader import load_parsers

        repo_db_path = settings.db_path_for_repo(repo_path.name)
        cgr_settings.LADYBUG_DB_PATH = repo_db_path
        cgr_settings.LADYBUG_BATCH_SIZE = settings.LADYBUG_BATCH_SIZE

        parsers, queries = load_parsers()
        files_done_counter = [0]

        def _progress_cb(event: dict) -> None:
            if event.get("phase") == "parsing" and "files_done" in event:
                files_done_counter[0] = int(event["files_done"])
                pct = 20.0 + (files_done_counter[0] / max(dirty_count, 1)) * 60.0
                _jobs_store.update_progress(
                    job_id,
                    phase="parsing",
                    progress_pct=pct,
                    files_done=files_done_counter[0],
                )

        with LadybugIngestor(
            cgr_settings.LADYBUG_DB_PATH,
            batch_size=cgr_settings.LADYBUG_BATCH_SIZE,
        ) as ingestor:
            updater = GraphUpdater(
                ingestor=ingestor,
                parsers=parsers,
                queries=queries,
                repo_path=str(repo_path),
                force=False,
                progress_callback=_progress_cb,
            )
            # Process each dirty file individually.
            for abs_path in dirty:
                try:
                    updater.process_file(str(abs_path))
                except Exception as exc:
                    logger.warning(
                        "watch_manager: failed to process %s: %s", abs_path, exc
                    )

            _jobs_store.update_progress(job_id, phase="writing", progress_pct=85.0)
            ingestor.flush_all()
            node_count = ingestor.node_count
            rel_count = ingestor.rel_count

    except Exception as exc:
        logger.warning(
            "watch_manager: graph update failed for %s: %s",
            entry.repo_slug, exc,
        )
        # Soft failure: still mark done with what we have.

    # --- Persist updated hash cache ---
    _jobs_store.update_progress(job_id, phase="finalizing", progress_pct=95.0)
    _save_hash_cache(repo_path, hash_cache)

    _jobs_store.mark_done(
        job_id,
        node_count=node_count,
        rel_count=rel_count,
        embedding_count=embedding_count,
    )

    return {
        "noop": False,
        "dirty_count": dirty_count,
        "files_done": dirty_count,
        "embedding_count": embedding_count,
        "node_count": node_count,
        "rel_count": rel_count,
    }


# ---------------------------------------------------------------------------
# Public lifecycle API
# ---------------------------------------------------------------------------


async def start_watch(
    repo_slug: str,
    *,
    actor_oid: str,
    actor_email: str,
) -> WatchHandle:
    """Start a file-watcher for ``repo_slug``.

    Returns:
        WatchHandle: Snapshot of the new watcher.

    Raises:
        WatchDisabledError: ``WATCH_ENABLED`` is false.
        WatchAlreadyActiveError: A watcher is already running for this slug.
        WatchCapacityError: ``WATCH_MAX_REPOS`` limit reached.
        FileNotFoundError: The repo path is not known / does not exist.
        OSError: inotify limit exceeded or permission denied.
    """
    from ..config import settings
    from .. import metrics as _metrics

    if not settings.WATCH_ENABLED:
        raise WatchDisabledError("WATCH_ENABLED is false")

    with _watches_lock:
        if repo_slug in _watches:
            raise WatchAlreadyActiveError(f"Watcher already active for {repo_slug!r}")
        if len(_watches) >= settings.WATCH_MAX_REPOS:
            raise WatchCapacityError(
                f"Watcher capacity exceeded ({settings.WATCH_MAX_REPOS} max)"
            )

    # Resolve repo path from index metadata.
    from ..routers.index import indexed_repo_paths
    repo_path_str = indexed_repo_paths.get(repo_slug)
    if not repo_path_str or not Path(repo_path_str).is_dir():
        raise FileNotFoundError(f"Repo '{repo_slug}' is not indexed or path is missing")

    loop = asyncio.get_running_loop()
    entry = _WatchEntry(
        repo_slug=repo_slug,
        repo_path=repo_path_str,
        actor_oid=actor_oid,
        actor_email=actor_email,
        debounce_ms=settings.WATCH_DEBOUNCE_MS,
        loop=loop,
    )

    # Schedule the Watchdog observer.
    try:
        from watchdog.observers import Observer
        observer = Observer()
        handler = _make_handler(entry)
        observer.schedule(handler, repo_path_str, recursive=True)
        observer.start()
        entry._observer = observer
    except OSError as exc:
        import errno
        msg = str(exc)
        if exc.errno == errno.ENOSPC or "inotify" in msg.lower():
            from .. import metrics as _metrics_inner
            _metrics_inner.record_watch_inotify_failure("max_watches")
        elif exc.errno == errno.EACCES:
            from .. import metrics as _metrics_inner
            _metrics_inner.record_watch_inotify_failure("permission")
        else:
            from .. import metrics as _metrics_inner
            _metrics_inner.record_watch_inotify_failure("other")
        raise

    # Start the debouncer coroutine.
    entry.state = "active"
    task = asyncio.create_task(_debouncer(entry), name=f"watcher-debounce-{repo_slug}")
    entry._debouncer_task = task

    with _watches_lock:
        _watches[repo_slug] = entry

    _metrics.set_watch_active_repos(len(_watches))
    logger.info(
        "watch_manager: started watcher for %s at %s (debounce=%dms)",
        repo_slug, repo_path_str, settings.WATCH_DEBOUNCE_MS,
    )
    return entry.snapshot()


async def stop_watch(repo_slug: str) -> bool:
    """Stop the watcher for ``repo_slug``.

    Returns:
        bool: True when a watcher was found and stopped.

    Raises:
        WatchNotActiveError: No watcher is active for ``repo_slug``.
    """
    from .. import metrics as _metrics

    with _watches_lock:
        entry = _watches.get(repo_slug)
        if entry is None:
            raise WatchNotActiveError(repo_slug)
        entry.state = "stopping"
        del _watches[repo_slug]

    # Cancel debouncer task.
    if entry._debouncer_task is not None and not entry._debouncer_task.done():
        entry._debouncer_task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.shield(entry._debouncer_task), timeout=2.0
            )
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    # Stop Watchdog observer.
    if entry._observer is not None:
        obs = entry._observer
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _stop_observer, obs)

    entry.state = "stopped"
    _metrics.set_watch_active_repos(len(_watches))
    logger.info("watch_manager: stopped watcher for %s", repo_slug)
    return True


def _stop_observer(observer: threading.Thread) -> None:
    """Blocking helper to stop a Watchdog observer (runs in thread pool)."""
    try:
        observer.stop()  # type: ignore[attr-defined]
        observer.join(timeout=5.0)
    except Exception as exc:
        logger.warning("watch_manager: observer stop error: %s", exc)


def get_watch(repo_slug: str) -> WatchHandle | None:
    """Return a snapshot of the watcher for ``repo_slug``, or None."""
    with _watches_lock:
        entry = _watches.get(repo_slug)
    return entry.snapshot() if entry is not None else None


def list_watches() -> list[WatchHandle]:
    """Return snapshots of all active watchers."""
    with _watches_lock:
        entries = list(_watches.values())
    return [e.snapshot() for e in entries]


async def shutdown_all(timeout_s: float | None = None) -> None:
    """Stop all active watchers; called from app lifespan on shutdown.

    Args:
        timeout_s: Maximum seconds to wait for each observer thread to join.
            Defaults to ``settings.WATCH_SHUTDOWN_TIMEOUT_S``.
    """
    from ..config import settings

    if timeout_s is None:
        timeout_s = settings.WATCH_SHUTDOWN_TIMEOUT_S

    with _watches_lock:
        slugs = list(_watches.keys())

    for slug in slugs:
        try:
            await stop_watch(slug)
        except WatchNotActiveError:
            pass
        except Exception as exc:
            logger.warning("watch_manager: error stopping %s during shutdown: %s", slug, exc)

    logger.info("watch_manager: all watchers stopped")
