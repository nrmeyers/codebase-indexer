# Phase 2 — Persistent Job State

**Status:** ready for implementation
**Depends on:** Phase 1 (M365 OAuth) — provides `actor_oid` / `actor_email` claims used as job ownership keys.
**Targets:** survive restart, recover progress, dedupe concurrent same-repo requests, queryable history.
**Validation gate (TEAM_DEPLOYMENT_PLAN §7):** restart-mid-job test passes; cross-user isolation test passes; existing 90 + 64 tests stay green.

---

## 1. Schema design

Single SQLite database at `JOBS_DB_PATH` (defaults to `.cgr/jobs.sqlite` dev / `/var/lib/forge/jobs/jobs.sqlite` prod). One file, one writer, WAL mode.

```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous   = NORMAL;          -- WAL + NORMAL durable across crashes
PRAGMA foreign_keys  = ON;
PRAGMA busy_timeout  = 5000;            -- wait up to 5s on lock contention

CREATE TABLE IF NOT EXISTS jobs (
  job_id           TEXT PRIMARY KEY,                  -- UUID4
  kind             TEXT NOT NULL CHECK (kind IN ('index','embed','watch_partial')),
  actor_oid        TEXT NOT NULL,                     -- M365 oid (Phase 1)
  actor_email      TEXT NOT NULL,
  repo_slug        TEXT NOT NULL,                     -- Path(repo_path).name
  repo_path        TEXT NOT NULL,                     -- absolute, resolved
  status           TEXT NOT NULL CHECK (status IN
                     ('queued','running','done','failed','cancelled','interrupted')),
  phase            TEXT,                              -- "parsing","embedding",…
  progress_pct     REAL NOT NULL DEFAULT 0.0,
  files_total      INTEGER NOT NULL DEFAULT 0,
  files_done       INTEGER NOT NULL DEFAULT 0,
  current_file     TEXT,
  node_count       INTEGER NOT NULL DEFAULT 0,
  rel_count        INTEGER NOT NULL DEFAULT 0,
  embedding_count  INTEGER NOT NULL DEFAULT 0,
  force_reindex    INTEGER NOT NULL DEFAULT 0,
  exclude_paths    TEXT,                              -- JSON array
  error            TEXT,
  cancel_requested INTEGER NOT NULL DEFAULT 0,
  pid              INTEGER,
  worker_token     TEXT,                              -- random per-process
  started_at       REAL NOT NULL,                     -- unix ts (matches IndexStatus)
  updated_at       REAL NOT NULL,
  finished_at      REAL,
  schema_version   INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_jobs_actor       ON jobs(actor_oid);
CREATE INDEX IF NOT EXISTS idx_jobs_repo        ON jobs(repo_slug);
CREATE INDEX IF NOT EXISTS idx_jobs_status      ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_started_at  ON jobs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_repo_active
   ON jobs(repo_slug) WHERE status IN ('queued','running');

CREATE TABLE IF NOT EXISTS schema_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
INSERT OR IGNORE INTO schema_meta(key,value) VALUES ('version','1');
```

Notes:
- No FK to a `users` table — `actor_oid` is the source of truth from the JWT.
- `kind` lets the same store cover the existing `_embed_jobs` path and the Phase 5 `watch_partial` events.
- `worker_token` set at row-create to module-scope `WORKER_TOKEN = uuid.uuid4().hex`. On startup, any `running` row whose `worker_token` differs is fair game for the orphan sweep.
- `started_at` stays a float-unix-timestamp to match the existing `IndexStatus` shape — FE keeps working unchanged.

## 2. `app/services/jobs_store.py` — CRUD surface

Thin DAO around stdlib `sqlite3`. No ORM, no async wrapper.

```python
def init(db_path: str) -> None: ...                # opens, applies PRAGMAs, runs migration

def create_job(*, kind: str, actor_oid: str, actor_email: str,
               repo_path: str, force_reindex: bool,
               exclude_paths: frozenset[str]) -> Job: ...

def update_progress(job_id: str, *, phase: str | None = None,
                    progress_pct: float | None = None,
                    files_total: int | None = None,
                    files_done: int | None = None,
                    current_file: str | None = None,
                    node_count: int | None = None,
                    rel_count: int | None = None,
                    embedding_count: int | None = None) -> None: ...

def mark_done(job_id: str, *, node_count: int, rel_count: int,
              embedding_count: int) -> None: ...

def mark_failed(job_id: str, *, error: str,
                terminal_status: str = 'failed') -> None: ...

def request_cancel(job_id: str) -> bool: ...
def is_cancel_requested(job_id: str) -> bool: ...

def get_job(job_id: str) -> Job | None: ...
def find_active_for_repo(repo_slug: str) -> Job | None: ...

def list_jobs(*, actor_oid: str | None = None,
              repo_slug: str | None = None,
              status: set[str] | None = None,
              limit: int = 50,
              offset: int = 0) -> list[Job]: ...

def clear_terminal(actor_oid: str | None = None,
                   statuses: set[str] = {'done','failed','cancelled'}) -> int: ...

def sweep_interrupted(worker_token: str) -> int: ...
        # UPDATE jobs SET status='interrupted', error='service restart', finished_at=now
        # WHERE status IN ('running','queued') AND worker_token != ? RETURNING job_id;

@dataclass(frozen=True)
class Job:
    job_id: str; kind: str; actor_oid: str; actor_email: str
    repo_slug: str; repo_path: str; status: str
    phase: str | None; progress_pct: float
    files_total: int; files_done: int; current_file: str | None
    node_count: int; rel_count: int; embedding_count: int
    force_reindex: bool; exclude_paths: frozenset[str]
    error: str | None; cancel_requested: bool
    started_at: float; updated_at: float; finished_at: float | None
```

Throttling: `update_progress` is called per-file from GraphUpdater progress_cb. To avoid 50k tiny commits, the store coalesces writes — in-memory dirty-record buffer flushed every 250 ms or on phase change. Cancel reads bypass the buffer.

## 3. Concurrency model

- **Single writer, multiple readers, WAL mode.**
- **Process-level write lock** — `threading.Lock` around every write, even with single writer (removes the lock-timeout window where one indexing job gets `SQLITE_BUSY`).
- **Long-lived connection** opened with `check_same_thread=False`, guarded by the lock.
- **Dedupe on `POST /index`** — replaces the current `for j in _jobs.values()` scan with `find_active_for_repo(repo_slug)`. Three options considered:

  | Strategy | Behavior | Decision |
  |---|---|---|
  | **A. Reject 409** | "already indexing this repo, poll job_id=…" | **CHOSEN** — matches today's behavior, FE already handles it (`index.py:894`), zero migration risk. |
  | B. Coalesce | Return existing `job_id` with `200` | Breaks "202 = new job started" contract. |
  | C. Queue | Insert `status='queued'`, run after current finishes | Adds scheduler complexity; `_repo_locks` already does this implicitly. Defer. |

  Coalesce identical `(actor_oid, repo_slug, force_reindex)` requests within 2 s (typical FE double-submit), returning existing job_id with 202.

- **Queue across repos.** Different repos can index in parallel. Existing `_repo_locks` (asyncio.Lock keyed by resolved repo path) preserved.

## 4. Crash recovery

Decision: **mark `interrupted` and require re-trigger.** Rationale:
- The Python coroutine that owned the indexing thread is dead.
- Partial graph state is *already* on disk (LadybugDB writes incrementally); next `POST /index` without `force_reindex` picks up via hash-cache.
- Auto-restart hides operational signal: a restart loop would silently keep retrying a poison-pill repo.

Mechanics, in `lifespan` startup:
1. Generate fresh `WORKER_TOKEN`.
2. `jobs_store.init(settings.JOBS_DB_PATH)`.
3. `count = jobs_store.sweep_interrupted(WORKER_TOKEN)` — flips `running`/`queued` rows not owned by us to `interrupted`.
4. Existing `cleanup_stale_locks()` and DB-corruption probe stay.

Frontend impact: `IndexStatus.status` gains the literal `'interrupted'`. FE renders identical to `'failed'` (red badge, "Re-index" button). Brief at `.planning/FRONTEND_AGENT_BRIEF.md` updated in Phase 7.

## 5. Migration of the in-memory tracker

- Delete module-globals: `_jobs: dict[str, _Job]`, `_embed_jobs`, `_prune_old_jobs`, `_job_to_summary`.
- Replace `_Job` dataclass references with `jobs_store` calls.
- `start_index` becomes:
  ```python
  existing = jobs_store.find_active_for_repo(repo_path.name)
  if existing: raise HTTPException(409, ...)
  job_row = jobs_store.create_job(...)
  background_tasks.add_task(_run_ingestion, job_row.job_id, ...)
  return IndexAccepted(job_id=job_row.job_id)
  ```
- `_run_ingestion(job_id, force_reindex)` loads row, runs work, calls `mark_done` / `mark_failed`. Progress callback becomes `lambda evt: jobs_store.update_progress(job_id, **derived(evt))`.
- `get_index_status` reads DB row and projects to `IndexStatus`. **Field-for-field identical** to today's response.
- `cancel_index` calls `jobs_store.request_cancel(job_id)`; worker polls `is_cancel_requested(job_id)` between phases.
- `list_jobs`, `clear_jobs`, `delete_job` become DB-backed; `JobListResponse` shape unchanged. `scope=mine|all` gates on `actor_oid`.

API contract impact: **zero new fields, one new status value.** `'interrupted'` is the only addition. FE work to support it is "treat like failed" — half a day.

## 6. Test plan

Unit (`tests/test_jobs_store.py`, new):
- `create_job` round-trips all fields including `exclude_paths` JSON.
- `update_progress` is partial-update (COALESCE).
- `mark_done` / `mark_failed` are idempotent.
- `find_active_for_repo` ignores terminal statuses.
- `list_jobs` filters and orders newest-first.
- `sweep_interrupted` only touches rows whose `worker_token` differs from caller's.
- WAL mode set; PRAGMA values verified.
- Cross-user isolation: A's `actor_oid` doesn't see B's jobs.

Integration (`tests/test_index.py`, augment):
- Existing 90+64 must stay green — store swap is internal.
- New: `test_index_persists_across_restart` — start a job, fail mid-flight via monkey-patched progress_cb, recreate FastAPI app with same `JOBS_DB_PATH`, GET status, expect `status='interrupted'`.
- New: `test_concurrent_post_returns_409` — two `POST /index` for same repo back-to-back; second gets 409.
- New: `test_cross_user_isolation` (Phase 1 dep) — two `Authorization` headers, each lists only their own jobs.

E2E (`scripts/test_restart_recovery.sh`, new):
- Boot service, `POST /index` against a large repo, wait until phase=`parsing`, `kill -TERM` uvicorn, restart, `GET /index/jobs`, assert exactly one row with `status='interrupted'`.

## 7. Observability hooks (feeds Phase 4 dashboard)

| Metric | Type | Labels | Source |
|---|---|---|---|
| `code_indexer_jobs_total` | Counter | `status`, `kind` | `mark_done`/`mark_failed`/`request_cancel`/`sweep_interrupted` |
| `code_indexer_jobs_active` | Gauge | `kind` | recomputed every `update_progress` flush |
| `code_indexer_job_duration_seconds` | Histogram | `kind`, `terminal_status` | terminal transition: `finished_at - started_at` |
| `code_indexer_jobs_interrupted_total` | Counter | — | startup sweep |
| `code_indexer_jobs_dedupe_409_total` | Counter | — | rejected `POST /index` |
| `code_indexer_jobs_store_write_seconds` | Histogram | `op` | wraps every store write |

Phase 2 stubs the `app/metrics.py` calls behind a no-op import shim if metrics aren't wired yet.

## 8. Trade-offs called out

(a) **SQLite vs lightweight Postgres.** SQLite wins for Phase 2: zero ops surface, stdlib only, workload is at most a few writes/sec from a single process. Postgres only earns its keep above ~100 writes/sec sustained — neither true even at Phase 5's "realtime updater" load. Schema is portable; swap is a `jobs_store.py` rewrite, not an API change.

(b) **Persist the work-queue or just the envelope?** Just the envelope. The indexing pipeline is *deterministic given repo path and force_reindex flag* and re-entrant via the existing hash cache. Persisting per-file work items would explode row count, duplicate state LadybugDB's hash cache already encodes, and tempt us into auto-resume.

(c) **Phase 5 interaction (realtime updater).** Phase 5 creates one job per debounced watcher fire — ~1 every 1.5 s per active repo, terminal in seconds. With WAL + 250 ms write coalescing, ~3 writes per fire amortized. Two precautions: (i) Phase 5 jobs use `kind='watch_partial'` so `list_jobs` filters them by default. (ii) Phase 5 should `clear_terminal` `watch_partial` rows older than 24h.

## 9. Step-by-step implementation order

1. Add `JOBS_DB_PATH: str` to `app/config.py` `Settings` (already in `.env.example`; just unused — wire up).
2. Create `app/services/jobs_store.py` (~250 LOC).
3. Add `tests/test_jobs_store.py` — get unit tests green against `:memory:` SQLite.
4. Wire `jobs_store.init()` + `sweep_interrupted()` into `app/main.py` `lifespan`.
5. Refactor `app/routers/index.py`: replace `_jobs` reads/writes with `jobs_store` calls. Update `start_index`, `get_index_status`, `cancel_index`, `list_jobs`, `clear_jobs`, `delete_job`, `_run_ingestion`, progress_cb. Same treatment for `_embed_jobs` (use `kind='embed'`).
6. Update `IndexStatus` `Literal` to add `'interrupted'`.
7. Add integration test for restart recovery and dedupe-409.
8. Stub metrics shim (`app/metrics.py` no-op functions) so Phase 4 only swaps bodies.
9. Run full suite; verify 90 + 64 baseline stays green.
10. Update `.planning/FRONTEND_AGENT_BRIEF.md`'s `IndexStatus` snippet to add `'interrupted'`.

Estimate: 1 day, matching the TEAM_DEPLOYMENT_PLAN allowance.

---

## Critical Files for Implementation
- `code-indexer-service/app/services/jobs_store.py` (new)
- `code-indexer-service/app/routers/index.py` (refactor — biggest diff)
- `code-indexer-service/app/main.py` (lifespan: init + sweep_interrupted)
- `code-indexer-service/app/config.py` (wire `JOBS_DB_PATH`)
- `code-indexer-service/tests/test_jobs_store.py` (new) and `tests/test_index.py` (augment)
