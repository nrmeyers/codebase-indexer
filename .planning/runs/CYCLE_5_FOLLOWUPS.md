# Cycle 5 — Open follow-up findings

## 1. `forge_indexer_rerank_outcome_total` counter never materialises

**Symptom:** Cycle 5 ran 30 reranked semantic queries (visible in
`forge_indexer_search_duration_seconds_count{reranked="true"}`).
Per `app/services/reranker.py`, every rerank path branch calls
`_metrics.record_rerank_outcome("...")` — yet
`forge_indexer_rerank_outcome_total` doesn't appear in `/metrics` at all.

Same applies to `forge_indexer_query_rewriter_applied_total`.

**Likely cause:** Either (a) the `_rerank_outcome` Counter registration
in `app/metrics.py:319` runs but the labeled increments never fire (some
silent exception path), or (b) the `record_rerank_outcome` exception
swallowing in `routers/search.py:689` (`except Exception: pass`) is
hiding a metric-call AttributeError.

**Tractable check:** add a one-line debug log inside
`record_rerank_outcome` that fires every call. Re-run a single rerank
query. If the log fires but the metric doesn't appear, the Counter
registration is the bug. If the log doesn't fire, the rerank function
isn't reaching its outcome branches.

**Impact:** Operators can't measure live rerank hit-rate. A/B for
rerank flag flip is blind. Severity: medium (functional path works,
visibility is missing).

## 2. Rerank latency outliers exceed 5 s deadline

**Symptom:** 8 of 30 reranked queries in Cycle 5 took >10 s, with the
worst hitting 60+ s. The deadline (default `RERANK_DEADLINE_SECONDS=5`)
should have caught these via
`future.result(timeout=5)` raising `TimeoutError`.

**Likely cause:** Per the reranker code, when the deadline fires the
function returns the candidates immediately but the underlying
`_call_lm_studio` future keeps running in the background. If LM Studio
is single-threaded and the cycle hits 15 rerank queries in rapid
succession, the in-flight futures might serialize and the *next*
deadline timer doesn't start until the previous future completes — so
the second query waits for the first's HTTP timeout.

**Verify:** sequential rerank-query trace with timestamps.

**Workaround in production:** load a smaller chat model in LM Studio
(qwen3.6-27b's `n_ctx=4096` is also too small — the rerank prompt
overflows). Operator action.

## 3. Indexing rate 4.4 sym/s — bottleneck localisation pending

**Status:** Cycle 5 followup PR #22 added sub-phase timers
(`parse_open` / `parse_run` / `parse_metadata` / `parse_close`).
Cycle 6 will tell us which sub-phase dominates. Embed bench already
ruled out embed throughput as the cause (89→121 sym/s standalone vs
4.4 end-to-end).

If `parse_run` is >95% of `parse` total (likely), instrumentation
needs to push upstream into `code-graph-rag/codebase_rag/graph_updater.py`.
If `parse_open` or `parse_close` dominate, the issue is LadybugDB
connection lifecycle.
