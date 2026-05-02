# E2E Cycle Run Summary

- timestamp: 2026-05-02T20:31:31Z
- service: http://127.0.0.1:8000
- repos: TheForge, code-indexer-service, code-graph-rag
- queries: 60
- overall: **PASS**

## SLO matrix

| metric | target | measured | status |
|---|---:|---:|:--:|
| indexing_rate_symbols_per_s | ge 200.0 sym/s | — | SKIP |
| search_semantic_p95_no_rerank_s | le 0.2 s | 0.1325 s | PASS |
| search_semantic_p95_with_rerank_s | le 4.0 s | 0.0000 s | PASS |
| search_structural_p95_s | le 0.1 s | 0.0013 s | PASS |
| search_symbol_p95_s | le 0.05 s | 0.0009 s | PASS |
| context_bundle_p95_s | le 1.5 s | 0.0000 s | PASS |
| top1_relevance_semantic | ge 0.7 % | 0.8571 % | PASS |
| top5_relevance_semantic | ge 0.9 % | 0.9048 % | PASS |
| lm_studio_uptime | ge 0.95 % | 1.0000 % | PASS |

## Indexing per repo

| repo | status | elapsed (s) | nodes | rels | embeddings |
|---|---|---:|---:|---:|---:|
| TheForge | done | 0.0 | 7403 | 0 | 0 |
| code-indexer-service | done | 0.0 | 533 | 0 | 0 |
| code-graph-rag | done | 0.0 | 6544 | 0 | 0 |
