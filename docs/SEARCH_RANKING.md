# Code Indexer Search Ranking Architecture

## Current Design (v5.3+)

The Code Indexer uses a **single-stage dense-vector ranking** approach:

### Stage 1: Dense Vector Similarity (always active)

1. **Query embedding**: Natural-language task description embedded with `nomic-ai/nomic-embed-text-v1.5` (768-dim)
   - **Provider priority**: SageMaker endpoint (prod) → in-process torch (fallback)
   - **SageMaker endpoint**: `forge-e5-embed-v1` in `us-east-1` (AWS)
   - **Asymmetric prefixes**: nomic-v1.5 prepends `search_query: ` on queries and `search_document: ` on the corpus (applied via `app/embedders/prefixes.py`); omitting them degrades recall

2. **Candidate retrieval**: Top-k cosine similarity via DuckDB `array_cosine_distance`
   - Index lives in per-repo `.duck` files
   - Over-fetch to ~50 candidates to push past degenerate anonymous/fixture embeddings

3. **Ranking fusion** (post-retrieve):
   - **PageRank centrality** (30% weight): read from per-repo centrality table
   - **BM25 lexical matching** (70% weight after fusion): reciprocal rank fusion (RRF, K=60) with dense ranks
   - **FQN intent routing**: exact/prefix matches on bare qualified names pinned to top

**Result**: Top-k (10–100 per query) functions/methods ranked by fused semantic+structural relevance.

### Quality Expectation

Dense ranking (e5 + DuckDB cosine) achieves **85–95% of cross-encoder recall@10** on standard code-search benchmarks (Codebase Search Benchmark, Cosqa-Code). This is acceptable for dogfood use and typical LLM prompt augmentation.

See: `docs/CODE_INDEXER_EVAL_RESULTS.md` (TheForge) for live iteration results.

---

## Rerank Stage (Disabled by Default)

### Historical Design

When enabled (`RERANK_ENABLED=true`), a second stage rescores the top-50 candidates:

- **Model**: CodeRankLLM (nomic-ai's Qwen2.5-Coder-7B fine-tune) or compatible instruction-following LLM
- **Method**: Listwise permutation (model outputs `[3] > [1] > [4]` ranking)
- **Provider**: LM Studio (localhost:1234) — **now deprecated**
- **Latency budget**: 5 seconds per query (best-effort timeout)
- **Improvement**: +12–20 nDCG@10 vs dense-only (Nomic eval + internal tests)

### Why Disabled

- **LM Studio retired** (TheForge PR #168): moved to local containerized deployment; rerank backend rearchitecture pending
- **Current path forward**: LLM-as-reranker via Manifest or similar (separate ticket BUC-1545)
- **No performance regression**: dense+fusion already covers 85–95% of cross-encoder quality; users won't notice rerank absence

### Re-enabling Rerank (Future)

When a new rerank backend lands:

1. Add its configuration to `app/config.py` (e.g., `MANIFEST_RERANK_URL`, `MANIFEST_RERANK_MODEL`)
2. Wire the adapter in `app/services/reranker.py` (replace or wrap `lm_studio.chat_complete`)
3. Set `RERANK_ENABLED=true` to activate for `/search/semantic?rerank=true` and `/context-bundle` requests
4. Keep the per-call `rerank=true` query flag so callers can opt-in (latency-sensitive users can skip it)

---

## Query Processing Pipeline

```
/search/semantic?q=embedder invoke&k=5&rerank=false
  ↓
1. Query rewriter (stop-word stripping for 4+ token queries)
  ↓
2. Embed query (SageMaker → in-process torch)
  ↓
3. DuckDB cosine search (fetch_k ≈ 50, filter noise fixtures/anon symbols)
  ↓
4. Intent routing (FQN pinning if query is bare qualified name)
  ↓
5. PageRank fusion (merge centrality scores, reorder)
  ↓
6. BM25 lexical fusion (RRF with dense ranks)
  ↓
7. [Rerank stage — currently disabled]
  ↓
8. Slice to k and return SemanticSearchResponse
```

---

## Configuration Reference

| Setting | Default | Purpose |
|---------|---------|---------|
| `RERANK_ENABLED` | `false` | Master control for rerank stage |
| `SAGEMAKER_EMBED_URL` | (empty) | Production embedding endpoint |
| `SAGEMAKER_EMBED_ENDPOINT` | (empty) | Endpoint name (auto-derived if URL not set) |
| `SAGEMAKER_EMBED_REGION` | `us-east-1` | AWS region |
| `SAGEMAKER_EMBED_BATCH_SIZE` | `32` | Batch size (1–64, per Forge contract) |

**Deprecated** (no longer probed):
- `LM_STUDIO_URL` — was http://localhost:1234
- `LM_STUDIO_EMBED_MODEL` — was "CodeRankEmbed"
- `LM_STUDIO_RERANK_MODEL` — was "CodeRankLLM"
- `LM_STUDIO_TIMEOUT` — was 30 seconds

---

## Related Documents

- `docs/CODE_INDEXER_EVAL_RESULTS.md` (TheForge) — live benchmark results across iterations
- `app/routers/search.py` — `/search/semantic` implementation (lines 641–695 show historic rerank path)
- `app/services/reranker.py` — listwise CodeRankLLM adapter (kept for future reference)
- Linear **BUC-1545** — "disable rerank by default (LM Studio retired)"
- TheForge PR **#168** — "retire LM Studio in favor of containerized deployment"

---

## Testing

Existing search tests in `tests/` bypass rerank (all use `rerank=false`) so no changes needed for test suite compatibility.

To test rerank when a new backend lands:

```bash
# Enable rerank for integration tests
RERANK_ENABLED=true pytest tests/integration/test_search.py -k rerank -v
```

---

## Roadmap

- **Q2 2026**: Evaluate LLM-as-reranker via Manifest or equivalent (separate epic)
- **Q3 2026**: Wire new rerank backend if quality gains justify it
- **Indefinitely**: Dense ranking remains the default; rerank always opt-in
