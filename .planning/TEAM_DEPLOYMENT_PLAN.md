# Team Deployment & Productionization Plan

**Author:** assembled 2026-04-27 | **Status:** awaiting inputs (see §6)
**Scope:** lift the Code Indexer + TheForge stack from "single-dev localhost"
to "shared team environment with M365 SSO, Grafana monitoring, and the
documentation surface compacted to its final shape."

This plan is the working blueprint.  It will be deleted after Phase 7
ships; the durable docs it produces are inventoried in §5.

---

## Table of contents

1. [Frontend Integration Spec — current state](#1-frontend-integration-spec--current-state)
2. [Phased plan in order](#2-phased-plan-in-order)
3. [Phase-by-phase detail](#3-phase-by-phase-detail)
4. [Risk register](#4-risk-register)
5. [Documentation compaction inventory](#5-documentation-compaction-inventory)
6. [Inputs required from product owner](#6-inputs-required-from-product-owner)
7. [Validation gates per phase](#7-validation-gates-per-phase)

---

## 1. Frontend Integration Spec — current state

This is the contract a frontend agent (or human) needs to integrate
against the Code Indexer service **as it stands today**.  It will move
behind M365 OAuth in Phase 1 — this section describes the pre-OAuth
shape so any frontend work that lands before Phase 1 can be done
correctly.

### 1.1 Base URL & transport

| Concern | Value |
|---------|-------|
| Base URL (dev) | `http://localhost:8000` |
| Base URL (team env) | TBD per §6 (likely `https://forge.<corp>.com/api/code-indexer`) |
| Auth (today) | None |
| Auth (Phase 1+) | `Authorization: Bearer <m365 access token>` (audience: code-indexer service principal) |
| Content type | `application/json` for all bodies and responses |
| WebSocket | `ws://<host>/ws` — single multiplexed channel |

### 1.2 Error envelope (all error responses)

```ts
type ErrorEnvelope = {
  error: string;       // stable machine-readable code, e.g. "repo_not_found"
  message: string;     // human-readable, can change without a breaking version
  timestamp: string;   // ISO 8601 UTC
  traceId?: string;    // present when request hit the orchestrator
};
```

HTTP status conventions:

| Status | Meaning |
|--------|---------|
| `200` | success with body |
| `202` | accepted; job created (see `IndexAccepted`) |
| `400` | malformed request body / params |
| `401` | missing or invalid bearer (Phase 1+) |
| `403` | token valid but caller lacks the role needed |
| `404` | unknown repo / job / symbol |
| `409` | conflict (e.g. job already cancelled) |
| `429` | rate-limited (Phase 4+ once Grafana/Prom are in) |
| `503` | upstream embedder/reranker unavailable |

### 1.3 Endpoint catalog

**Health & metadata**

| Method | Path | Returns | Notes |
|--------|------|---------|-------|
| GET | `/health` | `HealthResponse` | includes `lm_studio` block; safe to poll |
| GET | `/repos` | `{ repos: RepoSummary[] }` | indexed repos with counts |
| GET | `/repos/{slug}` | `RepoStatsResponse` | per-repo node/embedding stats |

**Indexing**

| Method | Path | Body | Returns |
|--------|------|------|---------|
| POST | `/index` | `IndexRequest` | `202 IndexAccepted` |
| GET | `/index/{job_id}/status` | — | `IndexStatus` |
| POST | `/index/{job_id}/cancel` | — | `IndexStatus` |
| GET | `/index/jobs/list` | — | `JobListResponse` |
| POST | `/index/jobs/clear` | — | `JobClearResponse` |
| DELETE | `/index/{slug}` | — | `DeleteIndexResponse` |

**Search**

| Method | Path | Query | Returns |
|--------|------|-------|---------|
| GET | `/search/structural` | `q` (Cypher), `repo`, `limit?=20` | `{ nodes, relationships }` |
| GET | `/search/semantic` | `q`, `repo`, `k?=10`, `rerank?=false` | `SemanticSearchResponse` |
| GET | `/search/symbol` | `fqn`, `repo` | `{ source, file, line_start, line_end }` |
| POST | `/context-bundle` | `ContextBundleRequest` | `ContextBundleResponse` |
| GET | `/symbols/*` | varies | symbol-graph navigation |

**Operational**

| Method | Path | Use |
|--------|------|-----|
| GET | `/explorer/*` | file-tree introspection |
| GET | `/disk/*` | disk-usage diagnostics |
| ANY | `/github/*` | GitHub integration helpers (PR status, branches) |
| WS | `/ws` | live indexing progress + activity feed |

### 1.4 Key request / response shapes

> Authoritative TypeScript types live in
> `TheForge/src/services/code-indexer-client.ts` and
> `TheForge/web/src/components/code-indexer/types.ts`.
> The shapes below are the durable contract.

```ts
// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------
type LMStudioHealth = {
  configured: boolean;     // LM_STUDIO_URL is set
  reachable: boolean;      // a /v1/models call succeeded
  embed_model: string | null;
  rerank_model: string | null;
  can_embed: boolean;      // embed_model is loaded AND matches strict CodeRankEmbed
  can_rerank: boolean;     // rerank_model is loaded
};

type RepoHealth = {
  name: string;            // slug
  path: string;            // absolute path on disk
  node_count: number;      // graph nodes in LadybugDB (≈ symbols + files)
  embedding_count: number; // rows in DuckDB embeddings table
  last_indexed_at: string | null;  // ISO 8601
};

type HealthResponse = {
  status: 'ok' | 'degraded';
  db_path: string;
  indexed_repos: string[];   // slugs
  repos: RepoHealth[];       // detail per repo
  lm_studio?: LMStudioHealth;
};

// ---------------------------------------------------------------------------
// Index
// ---------------------------------------------------------------------------
type IndexRequest = {
  repo_path: string;
  force_reindex?: boolean;
};

type IndexAccepted = { job_id: string };

type IndexStatus = {
  job_id: string;
  status: 'queued' | 'running' | 'done' | 'failed' | 'cancelled';
  progress_pct: number;
  phase: string;                // e.g. "parse", "embed"
  files_total: number;
  files_done: number;
  node_count: number;
  rel_count: number;
  embedding_count: number;
  started_at: string;
  finished_at: string | null;
  error: string | null;
};

// ---------------------------------------------------------------------------
// Semantic search
// ---------------------------------------------------------------------------
type SemanticSearchResult = {
  qualified_name: string;
  symbol_type: string;          // "Function" | "Method" | "Class" | …
  file_path: string;
  start_line: number;
  end_line: number;
  score: number;                // [-1, 1]; higher is better
  source_snippet?: string;
  pagerank?: number;
};

type SemanticSearchResponse = {
  query: string;
  repo: string;
  results: SemanticSearchResult[];
  reranked: boolean;             // whether stage-2 LLM rerank ran
  search_intent?: 'semantic' | 'fqn' | null;  // "fqn" when bare-FQN regex pinned
};

// ---------------------------------------------------------------------------
// Context bundle
// ---------------------------------------------------------------------------
type ContextBundleRequest = {
  repo_path: string;             // single-repo (absolute path or slug)
                                 // OR "*" for cross-repo fan-out
  task_description: string;
  depth?: number;                // graph traversal depth, default 3
  k?: number;                    // top-k seed symbols, default 12
  rerank?: boolean;              // opt into stage-2 (default false)
};

type ContextBundleResponse = {
  symbols: SemanticSearchResult[];
  source_snippets: Record<string, string>;  // FQN → code
  call_graph: {
    nodes: Array<{ id: string; type: string; label: string }>;
    edges: Array<{ from: string; to: string; type: string }>;
  };
  total_tokens: number;
  reranked: boolean;
};
```

### 1.5 Frontend integration cheat-sheet

```ts
// ---- Health probe (use in Chat header + settings) -------------------------
const health = await fetch('/api/code-indexer/health').then(r => r.json());
const canRerank = !!health.lm_studio?.can_rerank;

// ---- Semantic search ------------------------------------------------------
const res = await fetch(
  `/api/code-indexer/search/semantic?q=${encodeURIComponent(q)}&repo=${slug}&k=10&rerank=${canRerank}`
).then(r => r.json() as Promise<SemanticSearchResponse>);

if (res.search_intent === 'fqn') {
  // Show "Exact match" badge — user typed a bare FQN
}

// ---- Context bundle (orchestrator does this server-side; this is for UI tools)
const bundle = await fetch('/api/code-indexer/context-bundle', {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify({
    repo_path: slug,            // or "*" for fan-out
    task_description: taskDescription,
    depth: 3,
    k: 12,
    rerank: canRerank && symbolCount >= 500,  // matches server-side gate
  }),
}).then(r => r.json() as Promise<ContextBundleResponse>);

// ---- WebSocket — index progress ------------------------------------------
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (e) => {
  const evt = JSON.parse(e.data);
  if (evt.type === 'index_progress') updateProgressBar(evt.payload);
};
```

### 1.6 Rate-limit & token-cost notes for the FE

- `/search/semantic` with `rerank=false` is cheap (linear cosine on 768-dim,
  typically ≪ 200 ms p95 even at 50k symbols). Safe to wire to a search-as-
  you-type input behind a 250 ms debounce.
- `/search/semantic?rerank=true` and `/context-bundle?rerank=true` invoke
  LM Studio listwise rerank. Wall-clock is **~100 s for 5 candidates with
  reasoning** on the qwen3.6-27b dense model. Always show a spinner. Never
  rerank on keystroke.
- `/index` is fire-and-forget (returns 202). Show a non-blocking toast and
  poll `/index/{job}/status` every 2 s until terminal.
- `/health` is cheap (<10 ms) and safe to poll every 10 s for the
  Chat-header model indicator.

### 1.7 What the Chat page needs (from this plan onward)

After the chat-wiring agent finishes (running now):

- `Chat.tsx` invokes the orchestrator → which fans out to Code Indexer →
  produces a context bundle → assembles the system prompt → streams
  via the LM Studio adapter against `qwen/qwen3.6-27b`.
- A subdued "Model: qwen/qwen3.6-27b (local)" indicator renders in the
  Chat header, sourced from `health.lm_studio` or `.forge/config.yaml`.
- A future settings UI (ADR-0004 — to be authored) lets users pick
  the model. Until then, model selection is a one-line config edit.

---

## 2. Phased plan in order

> Phases are strictly sequenced — each depends on the previous shipping
> green.  Within a phase, sub-tasks may be parallelised across agents.

| # | Phase | Why first | Approx |
|---|-------|-----------|-------:|
| 1 | **M365 OAuth + identity middleware** | Everything else assumes "we know who is calling" — auth has to land before persistent multi-user state, deployment, or metrics. | 1.5 d |
| 2 | **Persistent multi-user job store** | Replace in-memory `_jobs` dict with SQLite (or Redis if scale demands). User-scoped job listing + cancellation. Depends on Phase 1 for `actor_id`. | 1 d |
| 3 | **Container image + compose stack** | Reproducible deployment artifact. `Dockerfile` for the FastAPI service, `docker-compose.yml` that brings up code-indexer + TheForge + a Postgres for TheForge state. LM Studio stays on the host (GPU-pinned). | 1 d |
| 4 | **Grafana-ready observability** | Prometheus metrics endpoint (`/metrics`), counters/histograms/gauges across the hot paths, plus a checked-in Grafana dashboard JSON. Depends on Phase 3 so the metrics endpoint is reachable from a Prom scrape config. | 1 d |
| 5 | **Realtime updater wiring** | File-watch → incremental reindex via the fork's existing `realtime_updater.py`. Per-repo enable/disable through `/repos/{slug}/watch`. WebSocket events for "index updated". Depends on Phase 2 (so update jobs are user-attributable) and Phase 4 (so we can observe it). | 0.5 d |
| 6 | **Codebase cleanup** | (a) audit-event unit test for `orchestration.rerank.applied`. (b) collapse stale ROADMAP/SKILL_API_PLAN drafts. (c) remove `.cgr/` test artifacts from repos. (d) lint pass with ruff + ty. | 0.5 d |
| 7 | **Documentation rewrite & compaction** | Inventory in §5. Lands LAST so docs reflect the final state, not an in-flight one. | 1 d |
| 8 | **HNSW / VSS index (promoted from ADR-0001)** | A team of 5 indexing real repos will cross the 50k-symbol trigger almost immediately. DuckDB's official VSS extension makes this nearly free. Builds on Phase 4 metrics so we can verify recall and p95 stay healthy. | 0.5 d |
| 9a | **Cross-repo retrieval eval harness** | Before we build cross-repo ranking we need ground truth — without measurement we'd guess at which merge strategy is correct. Synthetic queries × known-good answers across 3+ repos; reports recall@k and MRR per strategy. | 1 d |
| 9b | **Cross-repo unified ranking (promoted from ADR-0003)** | Implements the strategy that wins the eval (naive merge / z-score / listwise rerank-over-merged). | 0.5 d |

**ADR-0002 (CodeRankLLM proper) stays deferred** — externally blocked on
Nomic publishing an LM-Studio-friendly GGUF.  Phase 6 will add a one-line
config seam so the flip is trivial the day a GGUF appears.

**Total:** ≈ 8.5 working days serial; ≈ 5 days with the inside-phase
parallelism the agent harness gives us.

---

## 3. Phase-by-phase detail

### Phase 1 — M365 OAuth + identity middleware

**Goal:** every non-`/health` HTTP request to `:8000` is authenticated against
Microsoft Entra ID (formerly Azure AD), and every WebSocket connection is
upgraded with a verified bearer.  TheForge frontend uses MSAL.js to acquire
tokens; backend services validate them.

**Stack choice:**
- Frontend: `@azure/msal-browser` + `@azure/msal-react` (both MIT, official).
- Backend: `msal` (Python, MIT) for the dev login helper + `python-jose` or
  `pyjwt[crypto]` for production token validation against Entra's JWKS.
- Service-to-service: client-credentials flow with a service principal
  (TheForge backend → Code Indexer).

**Deliverables:**
1. `code-indexer-service/app/auth.py` — FastAPI `Depends(verify_bearer)`
   that validates JWT (issuer, audience, signature, expiry, optional
   roles/groups claims).  Cached JWKS with periodic refresh.
2. Apply the dependency to every router except `/health` (which stays open
   so liveness probes don't need tokens).
3. WebSocket upgrade: read the bearer from the `Sec-WebSocket-Protocol`
   subprotocol header (since browsers can't set `Authorization` on WS).
4. `code-indexer-service/.env.example` adds:
   ```
   AZURE_TENANT_ID=
   AZURE_CLIENT_ID=          # this service's app registration
   AZURE_AUDIENCE=api://code-indexer
   AZURE_REQUIRED_ROLES=     # comma-separated, optional
   ```
5. `TheForge/web/src/auth/msal.ts` — MSAL config + provider wrapping the app.
6. `TheForge/src/services/code-indexer-client.ts` reads a token-acquirer
   callback (passed in or pulled from a context) and attaches `Authorization:
   Bearer …` to every outbound call.
7. `TheForge/src/services/api-server.ts` middleware — same pattern as the
   Code Indexer side, applied to every `/api/*` route.

**Acceptance:**
- `curl http://localhost:8000/repos` → `401`.
- `curl -H "Authorization: Bearer <valid>" http://localhost:8000/repos` → `200`.
- TheForge UI redirects unauthenticated users to Microsoft login, returns to
  the originating page after consent.
- A test user with a different `oid` cannot see another user's job in
  `/index/jobs/list`.

### Phase 2 — Persistent multi-user job store

**Goal:** replace `_jobs: dict[str, _Job]` with a SQLite-backed store keyed
by `(actor_oid, job_id)` so jobs survive a restart and are isolated per user.

**Schema (SQLite, `./.cgr/jobs.db`):**
```sql
CREATE TABLE IF NOT EXISTS jobs (
  job_id          TEXT PRIMARY KEY,
  actor_oid       TEXT NOT NULL,
  actor_email     TEXT NOT NULL,
  repo_slug       TEXT NOT NULL,
  repo_path       TEXT NOT NULL,
  status          TEXT NOT NULL,
  progress_pct    REAL NOT NULL DEFAULT 0,
  phase           TEXT,
  files_total     INTEGER NOT NULL DEFAULT 0,
  files_done      INTEGER NOT NULL DEFAULT 0,
  node_count      INTEGER NOT NULL DEFAULT 0,
  rel_count       INTEGER NOT NULL DEFAULT 0,
  embedding_count INTEGER NOT NULL DEFAULT 0,
  error           TEXT,
  started_at      TEXT NOT NULL,
  finished_at     TEXT
);
CREATE INDEX idx_jobs_actor ON jobs(actor_oid);
CREATE INDEX idx_jobs_status ON jobs(status);
```

**Deliverables:**
- `app/services/job_store.py` — DAO module with `create / update / get /
  list_for_actor / cancel / clear_terminal`.
- Refactor `app/routers/index.py` to use it.  Remove the global `_jobs` dict.
- Add a `/index/jobs/list?scope=mine|all` parameter; `all` requires an
  admin role (configured by `AZURE_REQUIRED_ROLES`).
- Migration: on first boot post-Phase-2, if `_jobs` was empty in memory the
  file is created fresh; no live state to migrate.

**Acceptance:**
- Restart the service mid-job → status of an in-flight job is `running` then
  flips to `failed` with `error: "service restart"` (we can't recover the
  Python thread, but we can mark stale rows on boot).
- User A cannot see User B's jobs in default scope.
- `pytest tests/test_index.py` → green; new tests for cross-user isolation.

### Phase 3 — Container image + compose stack

**Goal:** one-command deploy, `docker compose up -d`, brings up the whole
stack except LM Studio (which stays on the host for GPU access).

**Files added:**
- `code-indexer-service/Dockerfile` — multi-stage:
  - Stage 1: `python:3.12-slim` + `uv` + clone code-graph-rag at a pinned
    commit, install with `[arrow,treesitter-full,semantic]` extras.
  - Stage 2: copy `.venv` + `app/` + `scripts/`, expose `:8000`,
    `CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0"]`.
- `code-indexer-service/.dockerignore` — exclude `.venv`, `.cgr/`, `__pycache__`.
- `docker-compose.yml` (lives at `~/code-indexer-service/`):
  ```yaml
  services:
    code-indexer:
      build: .
      ports: ["8000:8000"]
      env_file: .env
      volumes:
        - ./.cgr:/app/.cgr               # persist DBs across restarts
        - /Users/zacharymatthews/repos:/repos:ro  # read-only repo mount
      extra_hosts:
        - "host.docker.internal:host-gateway"  # so container can hit LM Studio
      restart: unless-stopped
    theforge:
      build: ../TheForge
      ports: ["3001:3001", "3000:3000"]
      env_file: ../TheForge/.env
      depends_on: [code-indexer]
      restart: unless-stopped
  ```
- `code-indexer-service/scripts/healthcheck.sh` — used by Docker
  `HEALTHCHECK` directive.

**Acceptance:**
- `docker compose up -d` from a fresh checkout brings up the stack.
- Hitting `https://<host>:8000/health` from another host on the team LAN
  succeeds (with valid bearer).
- Killing the code-indexer container and `docker compose up` recovers all
  jobs as Phase 2 designed.

### Phase 4 — Grafana-ready observability

**Goal:** every hot-path operation emits counters/histograms; a checked-in
Grafana dashboard renders them.  No traces yet (defer to ADR if needed).

**Stack:**
- `prometheus_client` (Python, Apache 2.0) for metrics export.
- `prom-client` (already in TheForge if not, add it — MIT) for the Node side.
- Grafana 10+ is already on the team's monitoring host (per user statement).

**Metrics emitted by code-indexer-service** (all labelled by `repo_slug`
where applicable):

| Metric | Type | Labels |
|--------|------|--------|
| `code_indexer_index_jobs_total` | Counter | `status` |
| `code_indexer_index_duration_seconds` | Histogram | `repo_slug`, `phase` |
| `code_indexer_search_requests_total` | Counter | `endpoint`, `status_code` |
| `code_indexer_search_duration_seconds` | Histogram | `endpoint`, `reranked` |
| `code_indexer_rerank_duration_seconds` | Histogram | `model` |
| `code_indexer_embeddings_count` | Gauge | `repo_slug` |
| `code_indexer_lm_studio_up` | Gauge | (1 / 0 from `/health`) |
| `code_indexer_lm_studio_can_rerank` | Gauge | |
| `code_indexer_bulk_insert_rows_total` | Counter | `path` (arrow/executemany) |
| `code_indexer_bulk_insert_duration_seconds` | Histogram | `path` |

**Metrics emitted by TheForge** (already partially present):

| Metric | Type | Labels |
|--------|------|--------|
| `forge_orchestrator_turns_total` | Counter | `actor_role`, `tier` |
| `forge_orchestrator_rerank_applied_total` | Counter | `repo_slug` |
| `forge_orchestrator_context_bundle_seconds` | Histogram | `repo_slug` |
| `forge_chat_completion_seconds` | Histogram | `model` |

**Deliverables:**
- `code-indexer-service/app/metrics.py` — central Prometheus registry +
  decorators.
- `/metrics` route mounted (Prometheus scrape format).
- `TheForge/src/services/metrics.ts` — analogous Node module.
- `infra/grafana/code-indexer-dashboard.json` — checked into the repo,
  importable via `grafana-cli` or the API.
- `infra/prometheus/prometheus.yml.example` — scrape config snippet.

**Acceptance:**
- `curl http://localhost:8000/metrics` returns Prom-format text.
- A dashboard panel for "p95 search latency" shows live data after a few
  searches.
- An alert rule for `code_indexer_lm_studio_up == 0 for 5m` fires when LM
  Studio is killed.

### Phase 5 — Realtime updater wiring

**Goal:** opt-in per-repo file-watch that triggers incremental reindex on
the fly, surfacing progress over the existing `/ws` channel.

**Source asset:** `code-graph-rag/realtime_updater.py` already exists with
tests; uses watchdog.  Just needs to be invoked from the FastAPI side.

**Deliverables:**
- `code-indexer-service/app/routers/repos.py` adds:
  - `POST /repos/{slug}/watch` → spawn watcher, return `202 { watcher_id }`.
  - `DELETE /repos/{slug}/watch` → stop.
  - `GET /repos/{slug}/watch` → status + last-touched-file timestamp.
- The watcher emits `index_partial_update` events on the `/ws` channel
  (frontend already listens to `index_progress`; add this as a sibling).
- Settings: env var `WATCH_DEBOUNCE_MS` (default 1500) — coalesce rapid
  saves into one reindex pass.

**Acceptance:**
- Edit a tracked file → within ~2 s the affected symbol is re-embedded
  and re-PageRanked → `index_partial_update` lands on the WS.
- Killing the file watcher leaves a clean shutdown (no zombie threads).

### Phase 6 — Codebase cleanup

**Goal:** small, finite cleanup items that don't deserve their own phase
but should land before doc rewrite.

Items:

1. **Audit event unit test** — `tests/unit/services/orchestrator.test.ts`
   asserts `orchestration.rerank.applied` fires when (a) `can_rerank` is
   true AND (b) `symbol_count >= 500`; does NOT fire otherwise.
2. **Multi-repo registration UX** — clarify implicit `POST /index` registration vs.
   the new `/repos/{slug}/watch`.  Either add a `POST /repos/{slug}/register`
   convenience endpoint or document the existing flow more clearly. Decide
   on read of phase-5 outcome.
3. **Test artifact hygiene** — purge `.cgr/repos/test_*.duck` and
   `.cgr/repos/test_*.db` from the working tree if any leak past pytest's
   tmp_path.  Add a `.gitignore` rule to belt-and-suspender it.
4. **Lint pass** — `uv run ruff check .` and `uv run ty .` clean across
   both Python repos. `pnpm lint` clean across TheForge.
5. **Stale planning artifacts** — delete `.planning/SKILL_API_PLAN.md`
   and the per-phase build-plan in `~/.claude/plans/full-digestion-…md`
   that the current ROADMAP supersedes (per the ROADMAP front-matter).
6. **`code-graph-rag/TODO.md`** — read, fold into the relevant ADR if a
   trigger exists, else delete.

### Phase 7 — Documentation rewrite & compaction

See §5 for the full inventory and target state.

### Phase 8 — HNSW / VSS index

**Goal:** sub-50ms p95 cosine search at any repo size, retaining > 99%
recall vs. the linear-scan ground truth.

**Stack:** DuckDB's official VSS extension
(`INSTALL vss; LOAD vss;`) — no new Python deps.

**Deliverables:**
1. `code-graph-rag/codebase_rag/storage/vector_store.py`:
   - On `open_or_create`, attempt to `LOAD vss`; if it fails (offline /
     unsigned binary), set a `_vss_available = False` flag and continue
     with the linear-scan path.
   - Add `build_hnsw_index(conn, m=16, ef_construction=200)` called at
     the end of every successful `bulk_insert`. Idempotent — drops and
     rebuilds when row count grew by > 10%.
   - `search_similar` checks `_vss_available` and uses the `?::FLOAT[768]
     <=> embedding` operator with the index when present.
2. New env vars: `VSS_M`, `VSS_EF_CONSTRUCTION`, `VSS_EF_SEARCH`
   (sensible defaults that match DuckDB upstream guidance).
3. Recall sanity-check test: index 1k random vectors with a known nearest
   neighbour, assert HNSW recall@10 ≥ 99%.
4. Phase 4 dashboard panel for "HNSW recall sample" — periodically run a
   small ground-truth comparison and gauge it.

**Acceptance:**
- Cosine search p95 on a 100k-symbol repo drops below 50 ms.
- Recall@10 vs linear scan ≥ 99% on the synthetic test.
- Linear-scan fallback works when `LOAD vss` fails (verified by a
  monkey-patched test).

### Phase 9a — Cross-repo retrieval eval harness

**Goal:** ground-truth measurement of cross-repo retrieval quality so
Phase 9b can pick the right merge strategy with evidence, not guesswork.

**Deliverables:**
1. `code-indexer-service/eval/cross_repo/` directory.
2. A small fixture: 3+ small open-source repos indexed; ~30 hand-curated
   queries each tied to the FQN of the "correct" answer (preferably 1-2
   FQNs per query).
3. `eval/cross_repo/run.py` — runs each query under each strategy
   (naive top-k merge, z-score-normalised merge, LLM rerank over the
   merged set), reports MRR and recall@{1,3,5,10} per strategy.
4. Output: `eval/cross_repo/RESULTS_<date>.md` — comparison table that
   the Phase 9b decision will cite.

**Acceptance:**
- Harness runs in < 5 minutes (excluding rerank passes).
- Each strategy produces deterministic numbers across two runs (mod
  rerank's LLM noise — record seed where applicable).

### Phase 9b — Cross-repo unified ranking

**Goal:** ship the strategy Phase 9a's harness identified as best.

**Deliverables (assuming z-score-normalised merge wins, which is the
likely outcome — adjust if 9a says otherwise):**
1. `app/services/cross_repo.py` — collects per-repo top-k results,
   computes per-repo mean + std on the score column, normalises, merges,
   sorts, returns global top-k.
2. `/context-bundle` already accepts `repo_path: '*'` — wire this through
   the new merge function.
3. `/search/semantic` extends to accept `repo: '*'` (same fan-out).
4. Document the strategy in a new ADR-0006 that supersedes ADR-0003.

**Acceptance:**
- Cross-repo `/context-bundle` returns symbols from multiple repos in
  one ranked list, with diversity proven via a unit test.
- The Phase 9a harness re-run confirms the chosen strategy beats
  the alternatives on the team's actual repos (re-run with team
  repo set during deployment).

---

## 4. Risk register

| # | Risk | Likelihood | Mitigation |
|---|------|-----------:|------------|
| R1 | M365 token validation breaks for a tenant config we didn't anticipate (B2B guests, conditional access). | Med | Test against the user's tenant during Phase 1; keep a `DEV_AUTH_BYPASS=1` escape hatch (off in prod). |
| R2 | LM Studio is the single point of failure for chat. If it crashes, chat is down. | High | Phase 4 alert on `lm_studio_up == 0`; Phase 7 doc the host-pin requirement; future ADR for an Anthropic fallback. |
| R3 | DuckDB single-writer means concurrent `/index` jobs against the same repo serialise. With a team of 5+ devs all hammering the same repo, this could feel slow. | Med | Per-repo job queueing already exists; document expected behaviour; consider per-repo write-lock metric in Phase 4. |
| R4 | The `.cgr/` directory grows unbounded. At a team of 5 over 6 months, plausibly 5-50 GB. | Med | Phase 4 disk-usage gauge + alert at 80% of host disk. Phase 7 doc the per-repo size formula. |
| R5 | Frontend agent reads pre-Phase-1 docs and writes code that breaks once OAuth lands. | Low (we control the order) | Doc rewrite in Phase 7 documents the FINAL state; this plan's §1 documents the interim state and is dated. |
| R6 | Grafana dashboard JSON drifts from actual metric names. | Med | Phase 4 includes a `tests/test_metrics_contract.py` that scrapes `/metrics` and asserts every metric the dashboard references is present. |
| R7 | Realtime updater races a `POST /index force_reindex=true` and corrupts state. | Low | Phase 5 holds a per-repo write-lock for the duration of any indexing operation, watcher or otherwise. |

---

## 5. Documentation compaction inventory

### Current state

**code-indexer-service/** (3)
- `README.md` — keep (will be rewritten in Phase 7)
- `ROADMAP.md` — DELETE in Phase 7 (superseded by CHANGELOG.md + the live
  ADRs)
- `CLAUDE.md` — keep (project memory; update in Phase 7)

**code-indexer-service/.planning/** (3)
- `ROADMAP.md` — DELETE (duplicate of root)
- `SKILL_API_PLAN.md` — DELETE (skill API is shipped; planning doc obsolete)
- `TEAM_DEPLOYMENT_PLAN.md` — DELETE after Phase 7 lands (this file)

**code-indexer-service/docs/adr/** (4) — keep all
- `README.md`
- `0001-defer-hnsw-vss-indexes.md`
- `0002-defer-coderanklm-proper.md`
- `0003-defer-cross-repo-unified-ranking.md`
- (Phase 1 will add `0004-m365-oauth.md`)
- (Phase 7 will add `0005-model-selector-deferred.md`)

**code-graph-rag/** (5 root)
- `README.md` — keep, rewrite in Phase 7
- `PYPI_README.md` — keep (PyPI publish surface)
- `CONTRIBUTING.md` — keep
- `CODE_OF_CONDUCT.md` — keep
- `SECURITY.md` — keep, refresh contact email in Phase 7
- `TODO.md` — DELETE in Phase 7 (fold into ADRs or ROADMAP)
- `CLAUDE.md` — keep (engine-side memory)

**code-graph-rag/docs/** (16 — currently MkDocs site)
- `index.md` — keep, rewrite Phase 7
- `claude-code-setup.md` — keep, rewrite Phase 7
- `contributing.md` — DELETE (duplicate of root CONTRIBUTING)
- `getting-started/quickstart.md` — keep, rewrite Phase 7
- `architecture/overview.md` — keep, rewrite Phase 7
- `architecture/graph-schema.md` — keep
- `architecture/language-support.md` — keep
- `guide/cli-reference.md` — keep
- `guide/interactive-querying.md` — keep
- `guide/mcp-server.md` — keep
- `guide/realtime-updates.md` — keep, rewrite once Phase 5 ships
- `guide/code-optimization.md` — keep
- `guide/graph-export.md` — keep
- `advanced/adding-languages.md` — keep
- `advanced/ignore-patterns.md` — keep
- `advanced/building-binaries.md` — keep
- `sdk/overview.md` — keep
- `sdk/cypher-generator.md` — keep
- `sdk/semantic-search.md` — keep
- `sdk/graph-loader.md` — keep

**code-graph-rag/scripts/**
- `BENCH_RESULTS_2026-04-27.md` — keep (referenced by ROADMAP and ADRs)

**TheForge/docs/** — already curated; only Phase 7 refresh
- `REHAUL_PLAN.md` — keep
- `TARGET_ARCHITECTURE.md` — keep, rewrite Phase 7 to reflect M365 + Grafana
- `ARCHITECTURE.md` — keep
- `IMPROVEMENTS.md` — keep
- `DEVELOPMENT.md` — keep
- `ERROR_CODES.md` — keep, append Phase-1 auth error codes

### Target state after Phase 7

**code-indexer-service/** (4 files at root)
1. `README.md` — quickstart, env vars, dev commands, link to docs/
2. `CHANGELOG.md` — replaces ROADMAP.md going forward; living history
3. `CLAUDE.md` — keep
4. `OPERATIONS.md` — Grafana dashboard import, alert rules, troubleshooting

**code-indexer-service/docs/** (curated)
- `adr/` — architecture decision records (immutable once accepted)
- `frontend-integration.md` — promoted from §1 of THIS plan, the canonical
  contract for FE consumers
- `deployment.md` — Docker compose, env-var reference, secrets handling

**code-graph-rag/** (5 root) + curated `docs/` MkDocs site

**Total active markdown** post-compaction: ~30 files (down from ~45).
DELETED: ROADMAP duplicates, SKILL_API_PLAN, TODO.md, this plan,
duplicate CONTRIBUTING.

---

## 6. Inputs required from product owner

> These are the only things I cannot fabricate.  Provide them and I can
> kick off Phase 1.  Everything else has a sane default.

| # | Item | Why |
|---|------|-----|
| 1 | **M365 tenant ID** (UUID) | Issuer claim validation + JWKS URL construction. |
| 2 | **Code Indexer app registration client ID** (UUID, created in Entra portal) | Audience claim; service principal identity. |
| 3 | **TheForge SPA client ID** (UUID, separate registration) | MSAL.js public client config. |
| 4 | **Required role / group claims** (or "any signed-in user") | RBAC scope — "anyone in the tenant" is fine if that's the policy. |
| 5 | **Team-environment hostname** (e.g. `forge.<corp>.com`) | Redirect URI for OAuth + CORS allow-list + dashboard URL. |
| 6 | **Grafana endpoint + scrape interval** | So the Phase 4 prometheus.yml example is correct out of the box. |
| 7 | **Disk budget for `.cgr/`** on the team host | Phase 4 alert threshold. |
| 8 | **Repo registry** — list of repo paths the team will index | Pre-warm in Phase 3 compose so the first dev to sign in doesn't wait. |

If any of these are TBD when we reach a phase that needs them, that phase
gates on the answer.  Phases 2, 5, 6, 7 do not need any of these.

---

## 7. Validation gates per phase

| Phase | Gate (must hold true to call the phase done) |
|------:|-----------------------------------------------|
| 1 | All non-`/health` endpoints return 401 without a valid bearer; cross-user job isolation test passes; MSAL login flow works in TheForge UI end-to-end. |
| 2 | Restart-mid-job test passes; cross-user isolation test passes; existing 90 + 64 tests stay green. |
| 3 | `docker compose up -d` from a fresh checkout brings up a healthy stack; `/health` reachable from a second host; jobs persist across container restart. |
| 4 | `/metrics` returns Prom format; dashboard panel renders live data; alert rule fires when LM Studio is killed; metric-contract test passes. |
| 5 | File edit → `index_partial_update` WS event within ~2 s; clean watcher shutdown; concurrent `/index force_reindex` does not corrupt state. |
| 6 | Audit-event unit test green; ruff + ty + pnpm-lint all clean; no test-artifact `.duck` or `.db` files in working tree. |
| 7 | Doc inventory in §5 matches reality on disk; every kept doc has been read and either confirmed accurate or updated; CHANGELOG covers everything from "deferred Memgraph swap" through the current shipping state. |
| 8 | Cosine search p95 < 50 ms on 100k symbols; HNSW recall@10 ≥ 99%; linear-scan fallback verified. |
| 9a | Eval harness produces a table comparing ≥ 3 strategies across MRR and recall@{1,3,5,10}; runs in < 5 min. |
| 9b | `repo='*'` on both `/search/semantic` and `/context-bundle` returns globally-ranked results; ADR-0006 supersedes ADR-0003. |

---

## End of plan
