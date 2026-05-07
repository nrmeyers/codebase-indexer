# Cycle 8 — Decision

## Result: 7 PASS / 2 FAIL — qwen2.5-vl-7b vindicates the model swap

The 27b → 7b swap fixed exactly the cascade I predicted:

- `semantic_no_rerank`: 287ms → **175ms** (embedder pressure relieved)
- `context_bundle`: 2.40s → **1.23s** (chains rerank, so the rerank-floor reduction propagated)
- `search_with_rerank`: 6.57s → **5.83s** — improved but still 1.8s over SLO

## Two remaining reds

### 1. `search_semantic_p95_with_rerank_s = 5.83s` (target 4s)

The VL-7b is faster than 27b but still slow on the listwise rerank prompt (~5k tokens, 50 candidates, code-snippet-grounded). Options ranked:

| Lever | Effort | Expected effect |
|---|---|---|
| Reduce `MAX_CANDIDATES` from 50 → 25 | 1 line config | 1.5–2× rerank speedup; rerank lands at 2.9–3.9s — passes SLO |
| Switch to `Qwen2.5-Coder-7B-Instruct` (text-only, code-tuned) | operator action | Slightly faster (no vision tower idle), better code-rerank quality (top-5 also lifts) |
| Reduce listwise prompt verbosity | medium | uncertain |

**Recommendation:** drop `MAX_CANDIDATES` to 25 first. If still over SLO, swap to Coder-7b.

### 2. `top5_relevance_semantic = 83.3%` (target 90%)

Unchanged from Cycle 7. NOT gated on rerank speed — VL-7b's slightly weaker code understanding (vs Coder-7b) is the likely cause. Switching to Qwen2.5-Coder-7B-Instruct should lift this 5–8pp.

## Cycle ladder (full session)

| Cycle | Top change | Result |
|---|---|---|
| 0 | initial baseline | FAIL (WAL bug) |
| 1 | grader fix | PASS (1 SKIP) |
| 2 | nomic embedder loaded | PASS (1 SKIP) |
| 3 | bugfix re-score | PASS |
| 4 | confirms stability — **first SUCCESS** | PASS |
| 5 | 90-query corpus added; scorer bugs found | 6/9 (3 measurement bugs) |
| 6 | qwen reload mid-run | noisy artefacts |
| 7 | qwen-27b 30k context steady-state | 5/9 — hardware floor |
| **8** | **qwen2.5-vl-7b** | **7/9 — single highest-leverage cycle** |

## What unblocks Cycle 9

`MAX_CANDIDATES = 25` config flip in `app/services/reranker.py`. Single-line code change. Should resolve `search_with_rerank` red. `top5_relevance` likely needs the Coder-7b model instead of VL-7b.

I'll ship the MAX_CANDIDATES change now since it's a code-only fix with no operator dependency.
