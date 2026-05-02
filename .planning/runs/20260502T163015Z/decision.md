# Cycle 1 — Decision

## Result snapshot
- 6 of 9 SLOs PASS, including the two **relevance** metrics
  (top1=90.48%, top5=95.24% — well above the 70%/90% targets).
- 2 FAIL:
  - `search_semantic_p95_no_rerank_s`: 1.12 s (target ≤ 0.2 s)
  - `indexing_rate_symbols_per_s`: 0 (skipped — operator action gate)
- 1 PASS that doesn't yet have a workload: `context_bundle_p95_s`,
  `search_semantic_p95_with_rerank_s` (we don't trigger rerank in
  the current corpus)

## Root cause of remaining red metric
**`search_semantic_p95_no_rerank_s` = 1.12 s.** The latency p50 was
fast for structural (6.5 ms) and symbol (6.1 ms) — both pure DB queries.
Only `semantic` is slow because each query spends ~1 s embedding the
query string on CPU via the in-process `transformers` fallback.
LM Studio reports `can_embed = false` (the configured `CodeRankEmbed`
isn't loaded), so the fast GPU/MPS path is unavailable.

Loading the LM Studio embed model is the same single operator action
that also flips the indexing-rate metric. **One unblock, two SLOs
fixed.**

## Cycle 2 plan
1. **Operator action: load `CodeRankEmbed` in LM Studio.** This is the
   only gate.
2. Re-run with `--force-reindex` (Cycle 1 fix B is on main, so the
   stale-WAL failure mode is resolved).
3. Expect indexing rate to jump from ~3.7 sym/s → ~200–400 sym/s
   per the existing bench.
4. Expect semantic p95 to drop from 1.12 s → ~30–80 ms (LM Studio
   call) or ≤ 200 ms even with network round-trip.

## Driver / grader fixes shipped this cycle
- `scripts/grade_queries.py`: added `symbol` to `_TEXT_FIELDS` so
  the grader inspects the actual API response shape. Pre-fix:
  0% top-1; post-fix: 90.5% top-1 against the same data. The
  underlying ranking was already excellent — the grader was blind.
- `scripts/run_e2e.py`: `--skip-indexing` flag for query-only runs,
  and a pre-flight `can_embed` check that aborts force-reindex with
  a clear error rather than running 30+ minutes uselessly.

## Stop-condition tracking
- Cycles run: 2 (Cycle 0, Cycle 1)
- Consecutive all-green cycles: 0
- Need 2 more all-green cycles to declare done.
- Pending blocker: operator action on LM Studio embed model.
