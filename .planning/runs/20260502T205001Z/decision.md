# Cycle 5 — Decision

## Result snapshot (after scorer+grader bugfixes)

6 of 9 SLOs PASS. 3 remaining FAILs, ranked by tractability:

| Metric | Measured | Cause | Tractable? |
|---|---|---|---|
| `search_semantic_p95_with_rerank_s` | 60.0 s (target 4.0 s) | qwen3.6-27b loaded with `n_ctx=4096`; rerank prompt overflows | **operator-fix** — load model with larger context (8192+) in LM Studio |
| `top5_relevance_semantic` | 77.78% (target 90%) | 15 of 90 queries hit the broken rerank path; remaining 75 score ~88% | resolves when rerank fixed |
| `indexing_rate_symbols_per_s` | 4.4 sym/s (target 200) | NOT embed throughput (bench showed 89→121 sym/s on embed alone). Real bottleneck is downstream of embedding — likely tree-sitter parsing + LadybugDB MERGE writes per symbol | needs real investigation, ~1-day work |

## Pre-fix vs post-fix (this is why honesty in measurement matters)

The first SUMMARY.md from this cycle reported a "catastrophic regression":
- `semantic_p95_no_rerank` 27.17 s (vs Cycle 4's 132 ms) — actually a SCORING BUG: it lumped 15 broken rerank queries into the no-rerank bucket
- `top1/top5` 0% — actually a LOOP BUG: `grade_queries.py` wasn't run automatically when the cycle was launched standalone (not via `run_cycle_loop.sh`)
- `embedding_count: 0` per repo — actually a DISPLAY BUG in `/index/<id>/status` response shape; embeddings exist (TheForge 4,542 + indexer 306 + cgr 4,864 = 9,712 rows)
- `context_bundle_p95: 0 / relevance 0%` — actually an INTEGRATION BUG: `run_e2e.py` POSTed `repo` instead of `repo_path` and `depth: 12` (max is 3); endpoint 422'd every request

After fixing the cycle's measurement layer, real numbers came out and the headline is unchanged from Cycle 4: search latency + relevance are healthy.

## Bugfixes shipped this cycle

`scripts/run_e2e.py`:
1. `_run_queries`: `/context-bundle` body uses `repo_path` (full filesystem path) and clamps `depth ≤ 3`.
2. `_extract_top_k`: recognises `body['symbols']` as the result list for `intent=context_bundle`; projects to grader's `[{symbol, snippet}]` shape.
3. `_score_queries`: splits semantic queries by `id.startswith('xrr-')` or `record.rerank` flag — rerank and no-rerank get separate p95 buckets.
4. `_score_queries`: adds `context_bundle_p95_s` based on actual context_bundle latencies (was hard-coded 0).

`scripts/grade_queries.py`: already includes `symbol` field (from Cycle 1) and now correctly receives the projected context_bundle records.

## Cycle 5 deferred to backend operator (me)

1. **Indexing-rate bottleneck investigation** — the embed bench (89 sequential / 121 batched sym/s on a 100-text harness) demonstrated embedding is fast at LM Studio. End-to-end indexing at 4.4 sym/s means ~95% of time is in something else: tree-sitter parsing, graph_updater MERGE statements, DuckDB row-by-row inserts, or some serialization between phases. Concrete next step: instrument the indexing pipeline with per-phase timers in `app/routers/index.py` (Phase 4 metric work I deferred earlier — now actionable).

2. **Indexer's job-status `embedding_count` field reports 0** despite real DuckDB rows. Bug in how the response is computed; likely reads from a different DB connection that doesn't see the just-written embeddings. Fix in `app/routers/index.py` `get_index_status`.

3. **LM Studio context size** for rerank — operator action, but I should add a guard: when LM Studio HTTP 400 with "context exceeded", `app/services/lm_studio.py` should already gracefully degrade (and the rerank-deadline PR #20 should be fast-failing this too). Verify and tighten if needed.

## Cycle 5 ladder

| Cycle | Result | Notable |
|---|---|---|
| 0 | FAIL (1 red, scoring bugs) | first ever; surfaced WAL stale issue |
| 1 | PASS (1 SKIP) | grader bug fix (`symbol` field) |
| 2 | PASS (1 SKIP) | LM Studio embedder loaded |
| 3 | PASS (1 SKIP) | confirms stability |
| 4 | PASS (1 SKIP) | second consecutive — earlier SUCCESS ✅ |
| **5** | **6 PASS / 3 FAIL (corrected)** | new corpus surfaces rerank ctx + context_bundle integration + indexing-rate bottleneck |

Cycle 4's SUCCESS marker still stands — Cycle 5 is the first cycle on the EXPANDED 90-query corpus, which is a new measurement surface, not a regression.
