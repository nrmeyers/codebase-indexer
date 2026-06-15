# Embedder POC — 768 roster results & verdict

**Run:** 2026-06-14 (initial roster) + 2026-06-15 (nomic-v1.5 via llama-server).
Operationalizes [`embedder-poc-768.md`](embedder-poc-768.md).
Driver: [`scripts/run_poc_768.py`](../scripts/run_poc_768.py). Per-model raw
results in `.planning/runs/768-poc/<tag>/` (recall.json, arms.out, probes.out,
meta.json).

## Verdict: switch to nomic-embed-text-v1.5. (2026-06-15)

**Supersedes the 2026-06-14 verdict ("keep e5") below.** Adding the
nomic-embed-text-v1.5 holdout (previously failing via sentence-transformers;
landed via the new `llama_server` backend running the Q8 GGUF) flips the
decision: nomic wins **recall@10 by +3.9pp** and we are choosing the head-of-
list metric over the long-tail metric — bundle UX cares about the first
handful of seeds far more than the @50 tail.

| Model | @10 | @25 | @50 |
|-------|-----|-----|-----|
| **nomic-ai/nomic-embed-text-v1.5** (chosen) | **0.7056** | 0.8333 | 0.8889 |
| intfloat/e5-base-v2 (prior baseline) | 0.6667 | **0.8500** | **0.9111** |

Secondary considerations confirming the call:
- Nomic is **instruction-tuned with Matryoshka** support — future-proof for
  256/512-dim truncation if we ever want a cheaper KNN layer without
  re-indexing.
- Schema is unchanged (FLOAT[768]) — zero migration cost on the LadybugDB /
  DuckDB side.
- Llama-server backend (the path that ran nomic in the POC) is now
  production-viable; local default flipped from e5 → nomic in the same
  commit (sentence-transformers loads nomic via `trust_remote_code=True`,
  auto-enabled for the vetted set).

SageMaker swap (jina-code-v2-serverless → nomic-v1.5) is tracked separately
in [`embedder-sagemaker-swap-nomic-v1.5.md`](embedder-sagemaker-swap-nomic-v1.5.md).

---

## (Superseded) Verdict 2026-06-14: keep e5-base-v2

> Preserved verbatim for audit trail. Conclusion reversed 2026-06-15 once
> the nomic-v1.5 holdout was successfully embedded via llama_server.

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
| **nomic-ai/nomic-embed-text-v1.5** (chosen 2026-06-15) | **0.706** | 0.833 | 0.889 | — | — |
| intfloat/e5-base-v2 (prior baseline) | 0.667 | **0.850** | **0.911** | 0.967 | — |
| nomic-ai/CodeRankEmbed | 0.722 | 0.817 | 0.883 | **1.00** | auth 0.67→0.33, hang 1.0→0.50 |
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
| **nomic-v1.5** (llama-server, RTX 3060) | 1158 | 173 | 299 | 141 MB |
| e5 | 1192 | 170 | 304 | 141 MB |
| coderank | 1405 | 244 | 330 | 136 MB |
| gte-modernbert | 1183 | 171 | 300 | 141 MB |
| granite-r2 | 1169 | 166 | 300 | 141 MB |

nomic-v1.5 ran via `EMBEDDER_BACKEND=llama_server` against a podman llama.cpp
container with the Q8 GGUF on the RTX 3060 (CDI device `nvidia.com/gpu=0`,
`--pooling mean --ubatch-size 2048 --ctx-size 2048 -ngl 99`). Index time is
comparable to the sentence-transformers backends — the HTTP round-trip is
absorbed by per-batch GPU throughput.
