# E2E Test + Cyclic Optimization Plan — 2026-05-01

> Runs once Wave 4 (Phase 2+4 integration) lands clean. Goal: prove the
> stack works end-to-end and iterate on any metric that misses target.

## Stages

```
[ Stage 1 ]  Set up test corpus
[ Stage 2 ]  Cold-start indexing run (measure)
[ Stage 3 ]  Query suite (semantic / structural / symbol / context-bundle)
[ Stage 4 ]  Score against SLO targets
[ Stage 5 ]  Identify worst metric, propose fix
[ Stage 6 ]  Apply fix + commit
[ Stage 7 ]  Re-run from Stage 2; goto Stage 4
[ Stop ]     All SLOs met for two consecutive cycles
```

## Stage 1 — Test corpus

Three repos, sized to exercise interesting code paths without burning hours:

| Repo | Purpose | Approx symbols |
|------|---------|---------------:|
| `TheForge` itself | Real production code (TS heavy) | 8–12k |
| `code-indexer-service` itself | Python heavy, tight surface | 1–2k |
| `code-graph-rag` itself | Mixed Py + tree-sitter grammars | 4–6k |

Total expected: ~15–20k symbols, well below the Phase 8 HNSW trigger
threshold (>50k), good for steady-state measurement.

## Stage 2 — Indexing run

For each repo:
- `POST /index` with `force_reindex=true` (cold start)
- Stream `index_progress` events, record per-phase wall-clock
- On `done`, capture: total elapsed, node_count, rel_count, embedding_count, jobs DB row size
- Verify `/repos/{slug}/health` returns expected counts

Phase-level metrics to record (Phase 4 timer wraps these):
- `parse` seconds
- `embed` seconds (LM Studio path)
- `pagerank` seconds
- `finalize` seconds

## Stage 3 — Query suite

20 hand-crafted queries per repo, three intents each, totalling 180 queries:

| Intent | Examples |
|--------|----------|
| Semantic | "where is bearer-token validation done", "find the rerank gate logic", "where does index progress get broadcast over websocket" |
| Structural (Cypher) | `MATCH (f:Function)-[:CALLS]->(g:Function {name:"validateBearer"}) RETURN f.qualified_name LIMIT 10` |
| Symbol | `validateBearer`, `setup_metrics`, `bulk_insert_arrow` |

Each query records:
- end-to-end latency (HTTP round-trip)
- result count
- top-1 grade (manual: relevant / partial / wrong) — graded once, persisted
- whether rerank fired (for semantic)

## Stage 4 — Score against SLO targets

| Metric | Target | Source |
|---|---|---|
| Cold-start indexing rate | ≥ 200 symbols/s steady-state | Wave 4 run |
| Search semantic p95 | ≤ 200 ms (no rerank), ≤ 4 s (with rerank) | Phase 4 metrics |
| Search structural p95 | ≤ 100 ms | Phase 4 metrics |
| Search symbol p95 | ≤ 50 ms | Phase 4 metrics |
| Context-bundle p95 | ≤ 1.5 s | Phase 4 metrics |
| Top-1 relevance (semantic) | ≥ 70% on graded set | Manual |
| Top-5 relevance (semantic) | ≥ 90% on graded set | Manual |
| Index job memory | ≤ 4 GB peak RSS | OS metrics |
| LM Studio uptime during run | ≥ 95% | metrics gauge |

Pass criterion: every metric green for two consecutive cycles.

## Stage 5 — Identify + propose fix

Standard improvement levers, ranked by effort:

| Lever | When to apply | Effort |
|-------|---------------|-------:|
| Increase `LM_STUDIO_TIMEOUT` | rerank timeouts under load | S |
| Tune `RERANK_SYMBOL_THRESHOLD` | rerank fires too often / too rarely | S |
| Reduce semantic-search default `k` | latency too high, recall already met | S |
| Bulk insert via Arrow batch size tweak | indexing rate below target | M |
| Add bm25 + dense rerank combo | recall@5 below target | M |
| Activate Phase 8 HNSW | cosine p95 > 200 ms or repo > 50k | L |
| Activate Phase 9 RRF cross-repo | unified-rank complaints | L |

After fix: commit on a `perf/<slug>` branch, push, admin-merge, re-run.

## Stage 6 — Apply fix + commit

Touched files committed with conventional message:
```
perf(<scope>): <one-line summary>

Cycle <n> of E2E optimization. Metric: <metric-name>
Before: <number>
After:  <number>
Delta:  <%>

Refs: .planning/E2E_TEST_OPTIMIZATION_PLAN.md
```

## Stage 7 — Re-run

Re-trigger Stage 2 from clean DB state (`DELETE FROM jobs; rm -rf .cgr/repos/<slug>/`). Same query set, same SLO matrix.

## Stop conditions

- All SLOs green for **two consecutive** cycles → done
- Or: 5 cycles run with no improvement → stop and surface diagnostic dump (the system has a structural limit; promoting Phase 8 / Phase 9 may be needed)

## Token strategy

- **Cycle driver** = haiku (mostly running scripts and recording metrics)
- **Query grader** = sonnet (judges relevance — needs reasoning)
- **Fix-proposer + applier** = sonnet (real code work)
- ScheduleWakeup if stage hits Anthropic limits — each stage's output is
  persisted to `.planning/runs/<timestamp>/` so reruns resume cleanly

## Output artefacts

Each cycle writes a directory:
```
.planning/runs/<UTC-timestamp>/
  RUN_SUMMARY.md      # what happened, key metrics, fix applied
  index_phase_times.json
  query_results.jsonl
  metrics_snapshot.txt    # /metrics output before + after
  decision.md            # which lever was chosen and why
```

Top-level `.planning/runs/INDEX.md` rolls up all cycles in time order.

## Critical files for implementation

- `code-indexer-service/scripts/run_e2e.py` (NEW) — the cycle driver
- `code-indexer-service/scripts/grade_queries.py` (NEW) — relevance grader
- `code-indexer-service/scripts/queries.json` (NEW) — query corpus
- `.planning/runs/` (NEW directory) — outputs

The cycle driver is one Python script that reads `queries.json`, hits the
running service via `httpx`, records timings, calls the grader, writes
artefacts, computes pass/fail, and (if passing) exits 0. Wired so it can
be cron-driven later for continuous regression detection.
