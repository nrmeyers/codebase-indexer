# E2E Cycle 0 — Run Summary

- timestamp: 2026-05-01T22:54:42Z
- service: http://127.0.0.1:8000
- repos: TheForge, code-indexer-service, code-graph-rag
- queries: 60 (queued, not run — cycle stopped before query phase)
- overall: **FAIL**
- driver state: stopped manually after 31 min (embedding bottleneck identified)

## SLO matrix (partial — cycle did not reach query phase)

| metric | target | measured | status |
|---|---:|---:|:--:|
| indexing_rate_symbols_per_s | ≥ 200.0 sym/s | ~3.7 sym/s | **FAIL** (50× below target) |
| search_*_p95 | ≤ 200 ms / 4 s | (not measured) | SKIP |
| top1_relevance_semantic | ≥ 70% | (not measured) | SKIP |
| lm_studio_uptime | ≥ 95% | 100% | PASS |

## Indexing per repo (ground truth from `/index/jobs`)

| repo | status | nodes | rels | elapsed (s) | rate (sym/s) |
|---|---|---:|---:|---:|---:|
| code-indexer-service | done | 527 | 1141 | 134 | 3.9 |
| code-graph-rag | running (92%, embedding) | 6544 | 16291 | ~1830 | ~3.6 |
| TheForge | **failed** at second 0 | 0 | 0 | 0.4 | — |

### TheForge failure root cause

```
Database ID for temporary file
'/Users/zacharymatthews/code-indexer-service/.cgr/repos/TheForge.db.wal'
does not match the current database.
```

Stale DuckDB WAL from prior interrupted run. The service didn't clean it
on `force_reindex=true`, so the new DB tried to attach a journal from a
different database.

## Root-cause diagnosis (worst metric: `indexing_rate_symbols_per_s`)

Two compounding problems:

1. **Embed model not loaded in LM Studio.**
   `/health` reports `lm_studio.can_embed = false` — the configured
   `LM_STUDIO_EMBED_MODEL=CodeRankEmbed` is not currently loaded.
   Indexing falls back to the in-process `transformers` embedder,
   which is 10–100× slower per symbol on the dev Mac (no GPU offload).

   *Evidence*: code-indexer-service indexed 527 nodes in 134 s (3.9 sym/s).
   With LM Studio's embedder we expect ≥ 200 sym/s on the same hardware
   per the existing bench in `code-graph-rag/scripts/BENCH_RESULTS_2026-04-27.md`.

2. **Stale WAL not cleaned on force-reindex.**
   `force_reindex=true` doesn't currently delete `<repo>.db.wal` /
   `<repo>.duck.wal` artefacts, so a prior crash poisons the next run.

## Recommended Cycle 1 fix levers

| Lever | Effort | Owner | Why |
|---|---|---|---|
| **A. Load CodeRankEmbed in LM Studio** | S (operator action) | dev-machine operator | Unblocks the 50× indexing speedup; addresses worst SLO directly |
| **B. Code: clean WAL on force_reindex** | M (1–2 h) | indexer service | Prevents recurring stale-WAL failures; should ship regardless |
| **C. Skip force-reindex on already-indexed repos** | XS (config) | cycle driver | Avoids re-doing work that's already done; doesn't fix root cause |

**Recommendation: A + B for Cycle 1.** A is the lever that flips the
worst metric. B prevents the poison-pill failure mode. C is a workaround
that hides the real bottleneck and is rejected.

## Decision (apply for Cycle 1)

- Land code change: `app/routers/index.py` cleans stale `*.db.wal`,
  `*.duck.wal`, `*.duck.tmp` files when `force_reindex=true`.
- Ask operator to load `CodeRankEmbed` in LM Studio before Cycle 1.
- Cycle 1 re-runs with cleaned state.

## Cycle 0 artefacts in this directory

- `metrics_snapshot_pre.txt` — `/metrics` body before run (1 file)
- (no `query_results.jsonl` — cycle stopped before query phase)
- (no `metrics_snapshot_post.txt` — cycle stopped before snapshot)
