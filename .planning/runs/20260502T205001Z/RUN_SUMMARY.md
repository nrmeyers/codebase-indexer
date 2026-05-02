# E2E Cycle Run Summary

- timestamp: 2026-05-02T21:54:41Z
- service: http://127.0.0.1:8000
- repos: TheForge, code-indexer-service, code-graph-rag
- queries: 90
- overall: **FAIL**

## SLO matrix

| metric | target | measured | status |
|---|---:|---:|:--:|
| indexing_rate_symbols_per_s | ge 200.0 sym/s | 4.3987 sym/s | FAIL |
| search_semantic_p95_no_rerank_s | le 0.2 s | 0.1266 s | PASS |
| search_semantic_p95_with_rerank_s | le 4.0 s | 60.0030 s | FAIL |
| search_structural_p95_s | le 0.1 s | 0.0009 s | PASS |
| search_symbol_p95_s | le 0.05 s | 0.0007 s | PASS |
| context_bundle_p95_s | le 1.5 s | 0.3479 s | PASS |
| top1_relevance_semantic | ge 0.7 % | 0.7222 % | PASS |
| top5_relevance_semantic | ge 0.9 % | 0.7778 % | FAIL |
| lm_studio_uptime | ge 0.95 % | 1.0000 % | PASS |

## Indexing per repo

| repo | status | elapsed (s) | nodes | rels | embeddings |
|---|---|---:|---:|---:|---:|
| TheForge | done | 1630.8 | 7418 | 12119 | 0 |
| code-indexer-service | done | 118.7 | 569 | 1213 | 0 |
| code-graph-rag | done | 1569.7 | 6613 | 16466 | 0 |

## Worst metrics (fix-applier candidates)

- **search_semantic_p95_with_rerank_s**: target le 4.0 s, measured 60.0030
- **indexing_rate_symbols_per_s**: target ge 200.0 sym/s, measured 4.3987
- **top5_relevance_semantic**: target ge 0.9 %, measured 0.7778
