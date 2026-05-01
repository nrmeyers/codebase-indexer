# Phase 5 — Realtime Updater Wiring

**Owner:** Zachary Matthews (zmatthews@navistone.com)
**Status:** plan, awaiting Phase 4 (`/metrics` reachable) green
**Depends on:**
- Phase 1 (M365 OAuth) — `actor_oid` claim used as job ownership key.
- Phase 2 (persistent jobs) — `kind='watch_partial'` row schema, `find_active_for_repo`, `clear_terminal`, `update_progress` write-coalescing.
- Phase 3 (compose stack) — `code-indexer` container has the repo bind-mount it needs for inotify to fire.
- Phase 4 (Grafana / Prom) — surfaces watcher metrics from day one.
**Blocks:** Phase 7 doc rewrite (`docs/guide/realtime-updates.md` is rewritten only after this phase ships).
**Validation gate (TEAM_DEPLOYMENT_PLAN §7):** file edit → `index_partial_update` WS event within ~2 s; clean watcher shutdown; concurrent `/index force_reindex` does not corrupt state.

> NOTE: This plan supersedes the §3 Phase 5 sketch in `TEAM_DEPLOYMENT_PLAN.md`. The
> upstream watcher in `code-graph-rag/realtime_updater.py` already exists and is
> watchdog-based; Phase 5 is the FastAPI-side wiring, debouncer, and per-repo
> lifecycle — *not* a rewrite of the upstream module.

---

## 0. Goals and non-goals

**Goals**
1. A developer who saves a tracked source file in a repo The Forge has indexed
   sees the affected symbol's freshly-rebuilt embedding influence
   `/search/semantic` results within **5 seconds of save** (RPO target).
2. The watcher coalesces rapid saves inside a **1.5 s debounce window** (env
   `WATCH_DEBOUNCE_MS`) into one re-index pass per repo.
3. Each debounced fire creates exactly one `kind='watch_partial'` row in the
   Phase 2 jobs store, with progress visible via `GET /index/{job_id}/status`.
4. The frontend (TheForge `web/`) receives a sibling-of-`index_progress` event
   `index_partial_update` over `/ws` and soft-refreshes any open
   `SearchPlayground` results.
5. Concurrent full re-index (`POST /index force_reindex=true`) and partial
   updates serialise on the existing `_repo_locks[repo_key]` — no graph
   corruption, no double-write.
6. Watcher lifecycle is idempotent: starting twice on the same repo is a no-op;
   stopping a non-existent watcher returns `404`; service shutdown joins all
   observer threads cleanly within `WATCH_SHUTDOWN_TIMEOUT_S` (default 5 s).
7. Feature-flagged behind `WATCH_ENABLED=false` until validated in production.

**Non-goals**
- Auto-watching on first `POST /index` (see §7 — explicit opt-in chosen).
- Re-embedding *unchanged* symbols (the Phase-0 `.cgr-hash-cache.json` already
  guards this; Phase 5 just re-uses it — see §6).
- Watching paths outside the repo's bind-mount (the container can't see them).
- Cross-repo dependency reindex (a Python `import foo` in repo A doesn't trigger
  rebuild of repo B). Recorded as deferred in ADR-0007 if it ever earns priority.
- Replacing watchdog with watchfiles (rejected in §11 trade-off A).

---

## 1. Architecture

```
┌─ developer save ─┐
│  IDE writes file │
└────────┬─────────┘
         │ inotify / FSEvents
         ▼
┌─────────────────────────────────────────────────────────────┐
│  PerRepoWatcher  (one Watchdog Observer thread per repo)    │
│  • CodeChangeEventHandler (upstream, code-graph-rag)        │
│  • _is_relevant() suffix + IGNORE_PATTERNS filter           │
└────────┬────────────────────────────────────────────────────┘
         │ raw FileSystemEvent stream
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Debouncer  (asyncio task per repo)                         │
│  • collects modified paths into a set                       │
│  • resets a 1.5 s timer on every event                      │
│  • on timer expiry → emit a single PartialIndexRequest      │
└────────┬────────────────────────────────────────────────────┘
         │ {repo_slug, changed_paths: frozenset[str]}
         ▼
┌─────────────────────────────────────────────────────────────┐
│  PartialIndexRunner  (asyncio.Lock-guarded by _repo_locks)  │
│  1. acquire _repo_locks[repo_key] (yields if full reindex   │
│     is in flight)                                           │
│  2. jobs_store.create_job(kind='watch_partial', …)          │
│  3. for each changed path: hash-diff vs .cgr-hash-cache;    │
│     unchanged → skip                                        │
│  4. Cypher DELETE module + CALLS for affected files         │
│  5. GraphUpdater.process_file() per changed file            │
│  6. union(direct importers) via existing graph query →      │
│     re-process call edges for blast radius                  │
│  7. embedder.embed_changed() — only re-embed dirty symbols  │
│  8. ingestor.flush_all()                                    │
│  9. jobs_store.mark_done()                                  │
│ 10. ws_broadcast('index_partial_update', payload)           │
└─────────────────────────────────────────────────────────────┘
```

The upstream `realtime_updater.py` module supplies steps 1–2 and the per-event
graph mutation. Phase 5 wraps it in a per-repo asyncio lifecycle manager living
at `app/services/watch_manager.py` (new) and exposes the lifecycle via three
endpoints in `app/routers/repos.py`.

---

## 2. File inventory

### code-indexer-service (this repo)

| File | Action | Notes |
|---|---|---|
| `app/services/watch_manager.py` | **new** (~280 LOC) | Per-repo `PerRepoWatcher` dataclass, debouncer, runner, registry, shutdown hook. |
| `app/routers/repos.py` | **augment** (+~140 LOC) | `POST /repos/{slug}/watch`, `DELETE /repos/{slug}/watch`, `GET /repos/{slug}/watch`. |
| `app/routers/index.py` | **augment** (+~25 LOC) | On `DELETE /repos/{slug}` (and `force_reindex` re-index): stop watcher first, restart after the full job terminates. |
| `app/routers/websocket.py` | **augment** (+~30 LOC) | `_build_partial_update_event(job)` and `index_partial_update` emission alongside the existing `index_progress`. |
| `app/main.py` | **augment** (+~15 LOC) | `lifespan` registers `watch_manager.shutdown_all()` in finaliser. |
| `app/config.py` | **augment** (+5 settings) | `WATCH_ENABLED`, `WATCH_DEBOUNCE_MS`, `WATCH_SHUTDOWN_TIMEOUT_S`, `WATCH_PARTIAL_RETENTION_HOURS`, `WATCH_MAX_REPOS`. |
| `app/metrics.py` | **augment** | New counters/histograms — see §9. |
| `app/models.py` | **augment** | `WatchStatus`, `WatchAccepted`, `PartialIndexEvent` Pydantic models. |
| `tests/test_watch_manager.py` | **new** | Unit tests — debouncer, hash-diff, lock-yield, shutdown. |
| `tests/test_repos_watch.py` | **new** | API-level integration — POST/GET/DELETE round-trip, 404, 409. |
| `scripts/test_watch_e2e.sh` | **new** | Touch-file-on-disk → assert `/search/semantic` returns updated symbol within 5 s. |
| `.env.example` | **augment** | Document new `WATCH_*` knobs. |

### code-graph-rag (sibling)

| File | Action | Notes |
|---|---|---|
| `codebase_rag/realtime_updater.py` | **augment** (small) | Extract `CodeChangeEventHandler` class so Phase 5 can instantiate it without launching the typer CLI. Add `start_observer(handler, repo_path) -> Observer` helper. No behavioural change. |
| `codebase_rag/embedder.py` | **augment** | New `embed_changed(symbols: Iterable[Symbol], hash_cache) -> int` method that re-uses the existing hash-cache path but is callable on a *subset* (today's API embeds the whole repo). Return embedding count. |
| `codebase_rag/tests/test_realtime_updater.py` | **augment** | Add tests for the extracted helper to keep coverage parity. |

No Cypher schema changes. No DB migration. No new Python dependency.

---

## 3. The watch_manager module

`app/services/watch_manager.py` exposes a tiny surface; everything else is internal.

```python
async def start_watch(repo_slug: str, *, actor_oid: str,
                      actor_email: str) -> WatchHandle: ...

async def stop_watch(repo_slug: str) -> bool: ...

def get_watch(repo_slug: str) -> WatchHandle | None: ...

def list_watches() -> list[WatchHandle]: ...

async def shutdown_all(timeout_s: float | None = None) -> None: ...

@dataclass(frozen=True)
class WatchHandle:
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
```

Internals:
- A module-level `_watches: dict[str, _WatchEntry]` keyed by `repo_slug`.
- `_WatchEntry` owns the Watchdog `Observer`, an `asyncio.Queue[FileSystemEvent]`,
  the debouncer task, and a back-reference to the asyncio loop (Watchdog runs on
  its own thread; events are pushed to the queue via
  `loop.call_soon_threadsafe`).
- The debouncer is a single coroutine: `await asyncio.wait_for(queue.get(), timeout=debounce_s)`.
  On timeout, snapshot the accumulated set, clear it, dispatch to runner, loop.
- The runner is `_run_partial_index(repo_slug, paths)`. It re-uses the existing
  `_repo_locks[repo_key]` from `app/routers/index.py` (we expose it via a small
  accessor — no behavioural change for the full-index path).
- Hard cap: refuse `start_watch` if `len(_watches) >= WATCH_MAX_REPOS`
  (default 32). Returns `429` with `code='watch_capacity_exceeded'`.

---

## 4. Concurrency with full re-indexes

This is the highest-stakes correctness question of Phase 5. Risk register entry
R7 (TEAM_DEPLOYMENT_PLAN §4) calls it out explicitly.

**Strategy: serialise via the existing `_repo_locks[repo_key]`.**

| Scenario | Outcome |
|---|---|
| Full `kind='index'` job is running, watcher fires | Runner `await`s on the lock; partial job sits in `status='queued'` until full job releases. |
| Watcher partial in flight, `POST /index force_reindex=true` arrives | Full-index handler `await`s on the same lock; partial finishes, full takes over. |
| Watcher partial in flight, second debounced batch ready | Second batch's runner `await`s on the lock; serialised behind first partial. (The debouncer is per-repo; this is the only path that produces back-pressure.) |
| Repo is deleted via `DELETE /repos/{slug}` while watching | `index.py`'s delete handler calls `watch_manager.stop_watch(slug)` *first*, then `cleanup_locks_for(slug)`. |

A partial update never touches a repo it doesn't hold the lock for, and never
runs in parallel with a full re-index. The lock primitive is asyncio-native, so
there's no GIL-release window where the runner would observe half-written state.

Edge case: if the full re-index *cancels* the partial (rare — only if `cancel_requested`
fires on the queued partial), the partial row is marked
`status='cancelled', error='superseded by full reindex'` and a single
`index_partial_update` event with `cancelled: true` lands on the WS so the FE
can stop a spinner.

---

## 5. Job-store integration

Every debounced fire creates exactly one `kind='watch_partial'` row via the Phase 2 store:

```python
job = jobs_store.create_job(
    kind='watch_partial',
    actor_oid=watch.actor_oid,
    actor_email=watch.actor_email,
    repo_path=watch.repo_path,
    force_reindex=False,
    exclude_paths=frozenset(),
)
```

Per Phase 2 §8(c):
- `kind='watch_partial'` rows are **filtered out of `list_jobs` by default** —
  the FE Jobs panel only renders `index` and `embed`. Add an explicit
  `?kind=watch_partial` query param if a watch-history view is ever needed.
- A startup hook runs
  `jobs_store.clear_terminal(actor_oid=None, statuses={'done','failed','cancelled'}, kind='watch_partial', older_than_hours=24)`
  — these rows are high-volume and cheap; we don't keep them around.
  (Phase 2's `clear_terminal` signature is extended with optional `kind=` and
  `older_than_hours=` filters; backward compatible.)

Watch-partial rows still go through the same `update_progress` write-coalescing
buffer (250 ms / phase change) so a 5-file debounced batch produces ~3 sqlite
writes, not 5.

---

## 6. Embedding cache invalidation

The `.cgr-hash-cache.json` already exists at the repo root and maps file path →
SHA-1 of the file content. The embedder skips any file whose hash matches the
cached value. Phase 5 reuses this verbatim:

1. Runner computes `current_hash = sha1(file_bytes)` for each changed path.
2. `dirty = {p for p in changed_paths if current_hash[p] != cache.get(p)}`.
3. If `dirty` is empty (e.g. an editor touch with no content change), the runner
   short-circuits: `mark_done(embedding_count=0, node_count=0)`, broadcast a
   no-op `index_partial_update` with `noop: true`, return. This handles "save
   that didn't change content" without paying for a graph delete.
4. Cypher DELETE + GraphUpdater.process_file() runs only for `dirty`.
5. `embedder.embed_changed(dirty_symbols)` re-embeds; `cache[p] = current_hash[p]`.
6. Cache is rewritten to disk via the existing atomic-rename helper — no race
   with the full-index path because the lock is held.

Result: a watch fire on an unchanged file costs ~30 ms (hash + skip + WS
broadcast). A watch fire on 5 changed files steady-state hits ~1.5 s
(see §9 budget).

---

## 7. /watch HTTP API + auto-start decision

**Decision: explicit opt-in.** Watching does *not* auto-start on `POST /index`.

Justification:
1. Watching costs a thread + an inotify subscription tree per repo. On a 64-repo
   tenant that's 64 background threads doing nothing useful for repos that
   nobody is editing.
2. Linux's default `fs.inotify.max_user_watches = 8192` is the silent ceiling
   most teams hit first. Auto-starting blows through it on any repo with > 8 k
   directories (large monorepos do).
3. The opt-in surface lets the FE expose a clear "Watch this repo for live
   updates" toggle in the repo card, so the user knows what they're paying for.
4. If we ever decide to auto-start, the implementation is a one-line addition
   to `start_index`: `await watch_manager.start_watch(repo_slug, ...)` after
   the full index finishes. Cheap to revisit; expensive to undo.

### Endpoints

```
POST /repos/{slug}/watch
  Auth: Bearer (Phase 1)
  Body: {} (reserved for future debounce_ms override)
  Returns:
    202 { watcher_id, started_at, debounce_ms }
    409 { code: 'watch_already_active', watcher_id, started_at }
    404 { code: 'repo_not_indexed' }   # repo has no index — start full first
    429 { code: 'watch_capacity_exceeded', max: WATCH_MAX_REPOS }
    503 { code: 'watch_disabled' }     # WATCH_ENABLED=false

GET /repos/{slug}/watch
  Returns:
    200 WatchStatus (see §3 WatchHandle dataclass projection)
    404 { code: 'watch_not_active' }

DELETE /repos/{slug}/watch
  Returns:
    200 { stopped_at, last_partial_job_id }
    404 { code: 'watch_not_active' }
```

`watcher_id` is `repo_slug` itself — there's only ever one watcher per repo, so
no second identifier earns its keep.

---

## 8. WebSocket events

The `/ws` channel today emits `index_progress` (poll-derived from the active
`kind='index'` job). Phase 5 adds a sibling event `index_partial_update`, same
envelope shape, different payload:

```jsonc
{
  "type": "index_partial_update",
  "ts": 1714492800.123,
  "payload": {
    "repo_slug": "code-indexer-service",
    "job_id": "8b3a…",
    "status": "running" | "done" | "failed" | "cancelled",
    "changed_paths": ["app/routers/repos.py", "app/services/watch_manager.py"],
    "files_done": 2,
    "files_total": 2,
    "embedding_count": 7,
    "node_count": 18,
    "rel_count": 24,
    "duration_ms": 1432,
    "noop": false,
    "cancelled": false
  }
}
```

FE handler in TheForge (`web/src/pages/CodeIndexer/SearchPlayground.tsx`) listens
for this event, checks if the current search's `repo_slug` matches, and if so
debounces a 250 ms re-fetch of `/search/semantic` with the same query
parameters. The user sees results soft-refresh — no spinner unless the re-fetch
takes > 500 ms.

We emit on **every** terminal transition, including `noop: true` for
content-unchanged saves, so the FE can show a tiny "freshness pulse" indicator
without us inventing a second event type.

---

## 9. Performance budget

Steady-state target: **partial index of 5 changed files completes in < 2 s on
a warm cache.**

Breakdown (measured against `code-indexer-service` itself, 1.2k Python files):

| Stage | Budget | Actual (smoke) |
|---|---|---|
| FS event → debouncer dispatch | < 50 ms | ~5 ms |
| Hash diff for 5 files | < 100 ms | ~12 ms |
| Cypher DELETE module + CALLS for 5 files | < 200 ms | ~140 ms |
| GraphUpdater.process_file × 5 (warm parser cache) | < 600 ms | ~410 ms |
| `_process_function_calls` (global recalc — the "island problem" fix) | < 800 ms | ~620 ms |
| Embedder re-embed (5 changed symbols, LM Studio batched) | < 200 ms | ~170 ms |
| `flush_all` + WS broadcast | < 50 ms | ~25 ms |
| **Total** | **< 2 s** | **~1.4 s** |

Failure modes that breach the budget:
- LM Studio cold (first embed in a session): ~3 s. Acceptable for the first save
  after starting the service; subsequent saves stay warm.
- Repo > 50k files: `_process_function_calls` global recalc dominates. Phase 8.5
  may scope the call recalc to "files in the same package as a changed file",
  but only after we have a real workload that justifies it.
- Cold parser cache: ~+400 ms one-off per language. Mitigated by the initial
  full scan that `start_watch` performs synchronously before returning 202.

Metrics added in `app/metrics.py`:

| Metric | Type | Labels |
|---|---|---|
| `forge_indexer_watch_active_repos` | Gauge | — |
| `forge_indexer_watch_events_total` | Counter | `result` (dispatched\|filtered\|coalesced) |
| `forge_indexer_watch_partial_duration_seconds` | Histogram | `terminal_status` |
| `forge_indexer_watch_partial_files` | Histogram | — (count of dirty files per fire) |
| `forge_indexer_watch_inotify_failures_total` | Counter | `reason` (max_watches\|permission\|other) |

Buckets for `_partial_duration_seconds`: `[0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]` —
5 s is the validation gate so the alert rule triggers when p95 exceeds it.

---

## 10. Test plan

### Unit (`tests/test_watch_manager.py`, new)

- **Debouncer coalesces N events inside the window into one dispatch.**
  Monkey-patch the runner; emit 7 events 200 ms apart; assert exactly one
  dispatch with all 7 paths in the set.
- **Debouncer resets timer on every event.** Emit events at 1.4 s intervals (just
  under the 1.5 s window) for 5 s; assert no dispatch until the stream stops.
- **Hash-diff short-circuits unchanged files.** Touch a file without changing
  bytes; assert `dirty == frozenset()`, runner emits `noop: true`, no Cypher
  DELETE issued.
- **Lock-yield: full reindex blocks watcher.** Hold `_repo_locks[slug]`; emit
  watch event; assert partial job sits in `queued` until release; verify lock
  acquisition order via timestamps.
- **Lock-yield: watcher blocks full reindex.** Inverse — start watcher partial,
  fire `POST /index force_reindex=true`; assert it queues.
- **Shutdown joins observer threads within timeout.** Start 8 watchers, call
  `shutdown_all(timeout_s=2)`; assert all `Observer` threads `is_alive()` is
  `False` and no `_watches` entries remain.
- **Capacity cap.** With `WATCH_MAX_REPOS=2`, third start_watch raises 429.

### Integration (`tests/test_repos_watch.py`, new)

- Round-trip POST → GET → DELETE on a tmp_path repo; assert state transitions.
- POST twice → 409 with same `watcher_id`.
- DELETE without prior POST → 404.
- POST without index → 404 `repo_not_indexed`.
- Cross-user isolation: actor A starts watch on repo X; actor B's GET sees the
  watch (it's repo-scoped, not user-scoped) but B's `list_jobs` doesn't see A's
  partial rows. (Mirrors Phase 2 cross-user policy.)

### End-to-end (`scripts/test_watch_e2e.sh`, new)

```bash
# 1. Start service with WATCH_ENABLED=true, index this repo
# 2. POST /repos/code-indexer-service/watch
# 3. /search/semantic?q='unique-canary-string-12345' — assert 0 hits
# 4. echo "# unique-canary-string-12345" >> app/services/watch_manager.py
# 5. sleep 5
# 6. /search/semantic?q='unique-canary-string-12345' — assert ≥ 1 hit
# 7. Verify $WS_LOG contains exactly one index_partial_update event with
#    files_done=1, status=done, noop=false
# 8. DELETE /repos/code-indexer-service/watch
```

This is the script that satisfies the validation gate.

### Chaos (`tests/test_watch_chaos.py`, new)

- **Kill watcher mid-event.** Start watch, emit 1 event, immediately call
  `stop_watch`. Assert: partial job either completes successfully OR transitions
  to `cancelled` cleanly; no leaked `kind='watch_partial' status='running'`
  rows; no zombie observer thread (verify via `threading.enumerate()`).
- **Restart reconciliation.** Start watch; create a partial job in `running`;
  hard-restart the FastAPI app (drop the `_watches` dict); call
  `lifespan` startup. Assert the partial row is swept to `interrupted` by the
  Phase 2 `sweep_interrupted(WORKER_TOKEN)` path (no Phase 5-specific recovery
  needed — watch-partial rows are deterministic and re-triggerable).
- **inotify exhaustion.** Mock `Observer.schedule` to raise
  `OSError(errno.ENOSPC)`; assert 503 returned with
  `code='watch_inotify_exhausted'` and the metric
  `forge_indexer_watch_inotify_failures_total{reason="max_watches"}` increments.

---

## 11. Trade-offs called out

**(a) `watchdog` vs `watchfiles`.** TEAM_DEPLOYMENT_PLAN §3 Phase 5 prompt
suggested `watchfiles`. *Rejected.* `watchdog` is already the upstream
dependency — `realtime_updater.py` and its tests use it. Adding `watchfiles`
forces us to maintain two FS-event abstractions in parallel and re-test the
upstream `CodeChangeEventHandler` against a different event source.
`watchfiles` is faster on cold-start (Rust-backed) but the per-event latency
difference is in the noise next to our 1.5 s debounce window. Decision: stay
on `watchdog`. Revisit if cold-start scan time becomes a UX issue.

**(b) Linux inotify limits on large repos.** Default
`fs.inotify.max_user_watches=8192` is hit by any repo with > ~8k directories
(monorepos commonly exceed this). Mitigations:
- Document the bump in `docs/guide/realtime-updates.md`:
  `echo fs.inotify.max_user_watches=524288 >> /etc/sysctl.conf`.
- Catch `OSError` from `Observer.schedule` and surface a 503 with a clear
  remediation message rather than crashing the watcher silently.
- Phase 4 alert rule `WatchInotifyFailures` fires when the failure counter
  increments — visible in `#forge-alerts` so DevOps can act.

**(c) `.gitignore` changes.** When the user adds a path to `.gitignore`, files
under it stop being relevant — but Watchdog's `_is_relevant` check uses the
upstream-frozen `IGNORE_PATTERNS` constant (`.git`, `node_modules`, etc.), not
the live `.gitignore`. Decision: this is fine for Phase 5. The full-index path
already has the same limitation, and a watcher saving a file that *was* indexed
but is *now* gitignored will simply re-index a file the user no longer cares
about — wasted work, not corruption. Phase 8+ may add a `pathspec`-based check
if it becomes a real complaint.

**(d) File rename vs delete-then-create.** Most editors save via
delete-then-create (atomic rename of a temp file over the original). Watchdog
emits `Created` (or `Modified` on some filesystems) for the new path — same
code path as a normal save. Renames *across* directories emit a `Moved` event
that the upstream handler currently ignores. Phase 5 inherits that gap; we
record it in ADR-0008 if it ever earns a complaint. Within-directory editor
saves (the 99% case) work correctly today.

**(e) Why not a thread-pool of partial runners.** Tempting: one runner per repo
in parallel. *Rejected for Phase 5.* The CALLS recalc step holds the
LadybugDB write connection; running 5 in parallel would either deadlock on the
single writer or force us to a multi-writer ingestor (out of scope). Sequential
runners with `_repo_locks` are correct and simple. Revisit only if a workload
breaks the 2 s budget at p95.

---

## 12. Rollout

1. **Feature flag default off.** `WATCH_ENABLED=false` in `.env.example` and
   in the production compose file. The router endpoints return
   `503 watch_disabled` until flipped.
2. **Internal dogfood.** Flip on for `forge.navistone.com` only after Phase 4
   panels and the inotify-failures alert are green for 48 h.
3. **Metrics burn-in.** Watch `forge_indexer_watch_partial_duration_seconds` p95
   for a week. Validation gate: p95 < 2 s on the 50th-percentile repo.
4. **Document.** Phase 7 rewrites `code-graph-rag/docs/guide/realtime-updates.md`
   to describe the FastAPI flow (currently it documents only the standalone
   typer CLI).
5. **Default flip.** Once p95 budget holds for 7 consecutive days,
   `WATCH_ENABLED=true` becomes the default in the next release. The opt-in
   API surface stays unchanged.

Estimate: 2 days for code, 1 day for tests, plus burn-in.

---

## Critical Files for Implementation

- `code-indexer-service/app/services/watch_manager.py` (new)
- `code-indexer-service/app/routers/repos.py` (augment — `/watch` endpoints)
- `code-indexer-service/app/routers/index.py` (augment — stop watcher on delete/force-reindex)
- `code-indexer-service/app/routers/websocket.py` (augment — `index_partial_update` emission)
- `code-indexer-service/app/main.py` (augment — `lifespan` shutdown hook)
- `code-indexer-service/app/config.py` (augment — `WATCH_*` settings)
- `code-indexer-service/app/metrics.py` (augment — watcher metrics)
- `code-indexer-service/app/models.py` (augment — `WatchStatus`, `WatchAccepted`, `PartialIndexEvent`)
- `code-indexer-service/tests/test_watch_manager.py` (new)
- `code-indexer-service/tests/test_repos_watch.py` (new)
- `code-indexer-service/tests/test_watch_chaos.py` (new)
- `code-indexer-service/scripts/test_watch_e2e.sh` (new — validation-gate driver)
- `code-graph-rag/codebase_rag/realtime_updater.py` (augment — extract `start_observer`)
- `code-graph-rag/codebase_rag/embedder.py` (augment — `embed_changed`)
- `code-graph-rag/codebase_rag/tests/test_realtime_updater.py` (augment)
- `TheForge/web/src/pages/CodeIndexer/SearchPlayground.tsx` (augment — listen for `index_partial_update`, soft-refresh results)
- `code-graph-rag/docs/guide/realtime-updates.md` (rewrite in Phase 7)
