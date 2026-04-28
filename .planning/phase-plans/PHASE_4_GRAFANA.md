# Phase 4 — Grafana / Prometheus Observability

**Status:** plan, awaiting Phase 3 (compose stack) green
**Owner:** application engineering
**Depends on:** Phase 3 (`/metrics` reachable from Prom scrape; container DNS `code-indexer:8000`, `forge:3001`, `skill-api:8002`)
**Blocks:** Phase 5 (we want the file-watch path observable from day one)

## 1. Goal

Three services expose a Prom-format `/metrics` endpoint. A single Grafana dashboard renders the panels from `DEVOPS_REQUEST.md` §5. Four alert rules notify Slack `#forge-alerts`. Everything gated behind `METRICS_ENABLED=true`.

No traces (OpenTelemetry) yet — deferred to Phase 8+. See §10.

## 2. Current state — what already exists

- TheForge has `prom-client@^15.1.3` in `package.json`; `src/services/metrics.ts` already exports `createMetricsService()` with HTTP request counters, sync/validation/drift counters, histograms; `api-server.ts` already mounts `/metrics`. **Phase 4 work on TheForge is incremental** — add 4 metrics, wire the rerank-audit bridge.
- Code Indexer: no metrics code yet. `.env.example` has `METRICS_ENABLED=true` and `METRICS_PATH=/metrics` placeholders.
- Skill API: FastAPI service, no metrics yet. Has `telemetry.py` (logging only).

## 3. Metrics taxonomy

Naming convention: `<namespace>_<subsystem>_<name>_<unit>`. Namespace `forge_`.

### 3.1 Code Indexer (Python, `prometheus_client`)

| Metric | Type | Labels | Unit |
|---|---|---|---|
| `forge_indexer_search_duration_seconds` | Histogram | `endpoint` (semantic\|structural\|symbol), `reranked` (true\|false) | seconds |
| `forge_indexer_search_requests_total` | Counter | `endpoint`, `status_code` | — |
| `forge_indexer_index_job_duration_seconds` | Histogram | `phase` (parse\|embed\|pagerank\|finalize) | seconds |
| `forge_indexer_index_jobs_total` | Counter | `status` | — |
| `forge_indexer_index_job_progress_seconds` | Gauge | `job_id` | seconds since last progress (alert source) |
| `forge_indexer_lm_studio_up` | Gauge | — | 0/1 |
| `forge_indexer_lm_studio_can_rerank` | Gauge | — | 0/1 |
| `forge_indexer_embeddings_count` | Gauge | `repo_name` | rows |
| `forge_indexer_disk_bytes` | Gauge | `path` (cgr\|jobs\|audit) | bytes |
| `forge_indexer_http_requests_total` | Counter | `method`, `route`, `status_code` | — (5xx alert) |

Histogram buckets:
- search: `[0.005, 0.025, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]` (rerank can hit 100 s; long-tail bucket of 10 s captures p99)
- index job: `[1, 5, 15, 30, 60, 120, 300, 600, 1800, 3600]` (single index can take 45 min on large repo)

### 3.2 TheForge backend (Node, `prom-client`)

Add to existing registry:

| Metric | Type | Labels |
|---|---|---|
| `forge_orchestration_rerank_applied_total` | Counter | `repo_name` |
| `forge_orchestration_context_bundle_seconds` | Histogram | `repo_name`, `reranked` |
| `forge_orchestration_turns_total` | Counter | `tier`, `actor_role` |
| `forge_chat_completion_seconds` | Histogram | `model` |

Existing `forge_http_*`, `forge_sync_*`, `forge_validation_*` stay as-is.

### 3.3 Skill API (Python, `prometheus_client`)

| Metric | Type | Labels |
|---|---|---|
| `forge_skill_api_request_duration_seconds` | Histogram | `endpoint`, `status_code` |
| `forge_skill_api_requests_total` | Counter | `endpoint`, `status_code` |
| `forge_skill_api_composition_duration_seconds` | Histogram | `intent` |
| `forge_skill_api_lm_studio_up` | Gauge | — |

### 3.4 Cardinality control (critical)

- **DO NOT** label by `repo_id` (UUID, unbounded) or `job_id` (unbounded) on counters/histograms. `job_id` appears once on `forge_indexer_index_job_progress_seconds` gauge only because gauges decay (delete label set on terminal status).
- `repo_name` capped — top-N (default 20) + `"other"` bucket once we cross 50 repos. Configurable via `METRICS_TOP_N_REPOS`.
- `route` on HTTP counter is the route template (`/index/{job_id}/status`), never resolved path. FastAPI: `request.scope["route"].path`. Express: `req.route?.path`.
- `endpoint` is small enum (≤ 10 values), always safe.

## 4. FastAPI integration (Code Indexer + Skill API)

**Recommendation: explicit decorators on hot paths + auto-instrument the rest.**

- Library: `prometheus-fastapi-instrumentator` (MIT) for auto HTTP middleware. Drop-in, exposes `/metrics`, route-template aware.
- Hot paths get explicit instrumentation:
  - `/search/semantic` — wraps body in `forge_indexer_search_duration_seconds.labels(endpoint="semantic", reranked=str(bool(rerank))).time()`. The `reranked` label is the request param, not URL, so middleware can't capture it.
  - `/search/structural`, `/search/symbol` — same pattern with `endpoint=` switched.
  - Index-phase histogram updated from `app/services/indexer.py` — wrap each phase in a context manager so we get one observation per phase, not per job.

### 4.1 New file: `code-indexer-service/app/metrics.py`

Exports:
```python
REGISTRY = CollectorRegistry()
search_duration = Histogram(...)
index_job_duration = Histogram(...)
# ... etc
def setup_metrics(app: FastAPI) -> None: ...
```

`setup_metrics(app)`:
1. Reads `settings.METRICS_ENABLED`; if false, return early.
2. Calls `Instrumentator(should_group_status_codes=False, excluded_handlers=["/health", "/metrics"]).instrument(app).expose(app, endpoint=settings.METRICS_PATH, registry=REGISTRY)`.
3. Schedules a 30 s background task that polls `lm_studio_health()` and updates the `lm_studio_up` and `lm_studio_can_rerank` gauges. Same for disk-usage gauge.

### 4.2 Wire-up in `app/main.py`

```python
if settings.METRICS_ENABLED:
    from .metrics import setup_metrics, start_background_collectors
    setup_metrics(app)
    metrics_task = asyncio.create_task(start_background_collectors())
yield
if settings.METRICS_ENABLED:
    metrics_task.cancel()
```

### 4.3 Skill API — same pattern in `skill_api/metrics.py`.

## 5. Express integration (TheForge) — incremental

`/metrics` already mounts. Changes:
1. **Extend `metrics.ts`** with the four new counters/histograms from §3.2. Add `recordRerankApplied(repoName)` and `recordContextBundle(repoName, durationMs, reranked)`.
2. **Top-N cardinality cap**: helper `clampRepoLabel(name, allowList)` returns `name` if in top-N, else `"other"`. Top-N from in-memory window (last 1000 calls) recomputed every 60 s.
3. **Bridge audit-event → counter** (§6).
4. Existing HTTP middleware unchanged.

## 6. Audit-event → Prometheus bridge

The orchestrator emits `auditOrchestrationRerankApplied(...)` via `emitAudit()` at `orchestrator.ts:285` and `:323`. Mirror as Prom counter.

**Recommendation: in-process synchronous bridge (no listener, no poll).**

Tap `emitAudit` directly at `orchestrator.ts:646`:
```ts
async function emitAudit(event: ReturnType<typeof auditChatTurnStarted>): Promise<void> {
  metricsService?.recordAuditEvent(event);   // <-- new
  await auditTrailAsync.append(event);
}
```
Inside `recordAuditEvent`, switch on `event.type`:
- `'orchestration.rerank.applied'` → bump `forge_orchestration_rerank_applied_total{repo_name=clampRepoLabel(event.payload.repo_slug)}`.

Pros: single call site, sync, no event bus. Adding a label increment is < 1 µs.

**Cross-check vs `orchestration-stats.ts`**: that module computes p50/p95 over a rolling in-memory window for chat UI. Prom is the cross-service source of truth. We do **not** unify them — different consumers, different retention. Phase 4 includes a sanity test comparing the two for a 60 s window; warns (does not fail) on > 10% divergence.

## 7. Hit rate panel — formula

"Rerank gate hit rate" = `forge_orchestration_rerank_applied_total / forge_orchestration_context_bundle_total`. Denominator is `forge_orchestration_context_bundle_seconds_count`. PromQL:
```
sum(rate(forge_orchestration_rerank_applied_total[5m]))
  /
sum(rate(forge_orchestration_context_bundle_seconds_count[5m]))
```

## 8. Dashboard JSON

**File (placeholder):** `/Users/zacharymatthews/code-indexer-service/grafana/forge-dashboard.json`

**Structure (rows top-to-bottom):**

1. **Overview** — single-stat panels: stack health (`up{job=~"code-indexer|forge|skill-api"}`), LM Studio reachable, total RPS, error rate.
2. **Search latency** — three side-by-side panels (semantic / structural / symbol), each p50 + p95 line.
3. **Indexing** — index job duration p50/p95, jobs-by-status timeseries, currently-running jobs gauge.
4. **Orchestration** — rerank hit rate, context-bundle duration histogram, chat-turn rate by tier.
5. **LM Studio** — uptime % over 24 h, reachability over time with annotations on flips.
6. **Resources** — CPU / memory / disk per service (PromQL against node-exporter + cAdvisor).

**Sample panel definitions:**

Search p95 (semantic):
```promql
histogram_quantile(
  0.95,
  sum by (le) (rate(forge_indexer_search_duration_seconds_bucket{endpoint="semantic"}[5m]))
)
```

Rerank gate hit rate (single stat, last 1 h):
```promql
sum(increase(forge_orchestration_rerank_applied_total[1h]))
  /
sum(increase(forge_orchestration_context_bundle_seconds_count[1h]))
```

LM Studio uptime % (last 24 h):
```promql
avg_over_time(forge_indexer_lm_studio_up[24h]) * 100
```

Code Indexer 5xx error rate:
```promql
sum(rate(forge_indexer_http_requests_total{status_code=~"5.."}[5m]))
  /
sum(rate(forge_indexer_http_requests_total[5m]))
```

Full JSON generated in a follow-up turn — Grafana's schema is verbose but mostly mechanical.

## 9. Alert rules

**File:** `/Users/zacharymatthews/code-indexer-service/grafana/alerts.yml`

```yaml
groups:
  - name: forge-internal
    interval: 30s
    rules:
      - alert: ForgeIndexerHighErrorRate
        expr: |
          sum(rate(forge_indexer_http_requests_total{status_code=~"5.."}[5m]))
            /
          sum(rate(forge_indexer_http_requests_total[5m])) > 0.05
        for: 5m
        labels: { severity: warning, channel: forge-alerts }
        annotations:
          summary: "Code Indexer 5xx rate above 5% for 5 min"

      - alert: ForgeIndexJobStuck
        expr: max(forge_indexer_index_job_progress_seconds) > 600
        for: 1m
        labels: { severity: warning, channel: forge-alerts }
        annotations:
          summary: "Index job has emitted no progress event in 10 min"

      - alert: ForgeLMStudioUnreachable
        expr: forge_indexer_lm_studio_up == 0
        for: 5m
        labels: { severity: warning, channel: forge-alerts }
        annotations:
          summary: "LM Studio adapter unreachable from Code Indexer for 5 min"

      - alert: ForgeDiskNearFull
        expr: |
          (node_filesystem_avail_bytes{mountpoint="/var/lib/forge"}
            / node_filesystem_size_bytes{mountpoint="/var/lib/forge"}) < 0.20
        for: 10m
        labels: { severity: warning, channel: forge-alerts }
        annotations:
          summary: "/var/lib/forge below 20% free"
```

## 10. Trade-offs called out

- **Scrape interval — 15 s default.** Sensible for team-of-25. Alert windows (5 min) tolerate it; 30 s would let `ForgeIndexJobStuck` miss the 10-min wedge by one tick.
- **Cardinality:** §3.4. Hard rule: never add a label whose value space is unbounded. Top-N + `"other"` once we cross 50 repos.
- **OpenTelemetry traces:** **defer to Phase 8+**. Tracing answers "where in the call stack did this slow turn spend its time" — we don't have that question yet. Adding OTel now is ~2 days of work for a question we aren't asking.
- **Histograms vs Summaries:** Histograms only. Summaries can't be aggregated across instances.

## 11. Test plan

Unit:
- `code-indexer-service/tests/unit/test_metrics.py`
  - hitting `/search/semantic` increments `forge_indexer_search_duration_seconds_count{endpoint="semantic"}` by 1.
  - `METRICS_ENABLED=false` → `/metrics` returns 404.
  - cardinality cap: 100 distinct `repo_name` values produce ≤ 21 label sets.
- `TheForge/tests/unit/services/metrics.test.ts`
  - extend existing tests; cover `recordRerankApplied` and audit-bridge tap.
- `tests/unit/services/orchestrator.test.ts` — assert `orchestration.rerank.applied` event fires and counter increments.

Integration:
- `tests/integration/test_metrics_exposition.py` — boots FastAPI app, hits `/metrics`, asserts response is `text/plain; version=0.0.4` and contains every metric the dashboard JSON references. **Metric-contract test** that catches dashboard drift.
- TheForge equivalent: `tests/integration/api-server.metrics.test.ts`.

Manual smoke:
- `docker compose up -d` → `curl http://localhost:8000/metrics`, etc.
- Run `/index` against 1k-symbol repo, observe `forge_indexer_index_job_duration_seconds` populated.

## 12. Rollout

- **Feature flag:** `METRICS_ENABLED=true` (default in `.env.example`). `=false` collapses `setup_metrics` to no-op.
- **Order of merge:**
  1. Code Indexer `metrics.py` + lifespan wire-up + tests.
  2. Skill API equivalent.
  3. TheForge incremental extension.
  4. Dashboard JSON committed to `code-indexer-service/grafana/`.
  5. Alert rules YAML committed.
  6. DevOps wires Prometheus scrape config and imports dashboard.
- **Rollback:** flip `METRICS_ENABLED=false`, redeploy. No data migration.
- **Validation gate (deployment-plan §7):** `/metrics` returns Prom format on all three services, dashboard panel renders live data, LM Studio kill alert fires within 5 min of `pkill -f lm-studio`, metric-contract test passes.

## 13. Critical Files for Implementation

- `code-indexer-service/app/metrics.py` (new)
- `code-indexer-service/app/main.py` (modify lifespan)
- `TheForge/src/services/metrics.ts` (extend with rerank counter + audit-bridge entry)
- `TheForge/src/services/orchestrator.ts` (tap `emitAudit` at line 646)
- `agentic-services/skill_api/metrics.py` (new) and `app.py` (lifespan wire-up)
- `code-indexer-service/grafana/forge-dashboard.json` (new — placeholder)
- `code-indexer-service/grafana/alerts.yml` (new)
