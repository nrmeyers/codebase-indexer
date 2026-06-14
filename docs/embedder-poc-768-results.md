# Embedder POC — 768 roster results & verdict

**Run:** 2026-06-14. Operationalizes [`embedder-poc-768.md`](embedder-poc-768.md).
Driver: [`scripts/run_poc_768.py`](../scripts/run_poc_768.py). Per-model raw
results in `.planning/runs/768-poc/<tag>/` (recall.json, arms.out, probes.out,
meta.json).

## Verdict: keep e5-base-v2. No migration.

**No 768 candidate beats the e5 baseline on the primary metric (first-stage
recall@25).** Combined with the **1024 roster bust** (sibling repo), the
conclusion is decisive: on this corpus neither higher dimensionality nor
code-specialization improves first-stage retrieval over general e5-base-v2.

## Results

Primary metric = facet-coverage recall over 15 design tasks + 6 probes via raw
`/search/semantic` (no graph/boosts/rerank). Each model indexed fresh under its
own correct query/doc prefix (`app/embedders/prefixes.py`).

| Model | @10 | **@25** | @50 | composed lift | probe regressions vs e5 |
|-------|-----|---------|-----|---------------|--------------------------|
| **intfloat/e5-base-v2** (baseline) | 0.667 | **0.850** | 0.911 | 0.967 | — |
| nomic-ai/CodeRankEmbed | **0.722** | 0.817 | 0.883 | **1.00** | auth 0.67→0.33, hang 1.0→0.50 |
| ibm-granite/granite-embedding-english-r2 | 0.700 | 0.817 | 0.911 | 0.967 | none (collide 0.50→1.00 ↑) |
| Alibaba-NLP/gte-modernbert-base | 0.700 | 0.772 | 0.839 | 0.961 | auth 0.67→0.33 |
| jinaai/jina-embeddings-v2-base-code | — | — | — | — | **excluded — would not load** |

- **e5 wins @25 (0.850)** and @50 (tie 0.911 with granite). No candidate clears the bar.
- **coderank** is sharpest at the very top (@10 0.722) but loses broader recall and regresses two probes.
- **granite-r2** is the most e5-like — ties @50, equal composed, zero probe regressions (even improves the collide probe) — but still trails @25. The only candidate worth remembering if a code-flavored swap is ever wanted; no reason to now.
- **gte-modernbert**, the strongest CoIR-benchmark model in the roster, is the *worst* here at @25 — **CoIR rank does not transfer to this task.**
- **jina-v2-base-code excluded:** its vendored modeling code imports
  `find_pruneable_heads_and_indices` from `transformers.pytorch_utils`, removed
  in the installed transformers. Pinning transformers down would break the
  ModernBERT models, so it was dropped rather than chased.

## Method notes (hard-won)

- **Prefixes matter (the #1 footgun):** each model embedded with its card-correct
  query/doc prefix via `app/embedders/prefixes.py`; e5 was previously run
  prefix-less, understating it.
- **GPU targeting:** embed ran on the RTX 3060 via `EMBED_DEVICE=cuda` +
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` + `CUDA_VISIBLE_DEVICES=0`. Default CUDA order
  makes device 0 the RTX 3090 (full with llama-server) → CUDA OOM.
- **Small-GPU batch:** long-context code models (CodeRankEmbed/jina, 8K ctx) OOM
  the 3060 at the default encode batch of 32 on long files; `LOCAL_ENCODE_BATCH_SIZE=8`
  fixed it.
- **Isolation:** each model indexed into a throwaway `.cgr-poc/<tag>` dir
  (fresh = real re-embed, sidestepping the content-hash skip-re-embed trap).
- Cost/CPU-latency metric was skipped — it was a tiebreaker among quality
  winners, and there are none.

## Cost reference (GPU index time, seconds)

| Model | TheForge | code-indexer-service | code-graph-rag | index size |
|-------|----------|----------------------|----------------|------------|
| e5 | 1192 | 170 | 304 | 141 MB |
| coderank | 1405 | 244 | 330 | 136 MB |
| gte-modernbert | 1183 | 171 | 300 | 141 MB |
| granite-r2 | 1169 | 166 | 300 | 141 MB |
