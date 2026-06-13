# Embedder POC — 768-dim candidate roster

**Status:** ready to run (2026-06-13). Operationalizes
[`embedder-eval-plan.md`](embedder-eval-plan.md) with a concrete 768-dim
candidate set. This repo stays at `FLOAT[768]`, so **every candidate here is a
swap-class A/B — no schema change, no migration**. The sibling repo
`codebase-indexer-qwen` runs the 1024-dim roster
([`embedder-poc-1024.md`](../../codebase-indexer-qwen/docs/embedder-poc-1024.md)).

## Goal

Find the best **open-source, <1B-param, CPU-runnable** 768-dim embedder for
first-stage code retrieval on this corpus — and, via the shared harness,
produce numbers directly comparable to the 1024 roster so we can answer: *does
the 1024 separability ceiling actually buy anything on a real codebase, or is a
strong 768 code embedder enough?*

## Hard constraints (locked)

- **License: OSI open-source only** (Apache-2.0 / MIT / BSD). No `cc-by-nc` /
  research-only.
- **<1B params**; smaller is better — CPU inference latency is a first-class
  gate, not a footnote.
- **768-dim** (this repo's schema).
- **Code retrieval is the priority metric**; technical-prose handling secondary.

## Candidate roster (5)

Verify params/dim/license/prefix against each model card before indexing — the
table is from research, not a live fetch.

| # | Model | Params | License | Code evidence | Serving / CPU | Query/doc prefix |
|---|-------|--------|---------|---------------|---------------|------------------|
| 0 (baseline) | `intfloat/e5-base-v2` | 110M | MIT | general-prose-trained (the recall floor) | sentence-transformers / ONNX / TEI | `query: ` / `passage: ` |
| 1 | `nomic-ai/CodeRankEmbed` | 137M | Apache-2.0 | code-specialized (current in-process default, `CODERANK_EMBED_MODEL`) | sentence-transformers (`trust_remote_code`) | query instruction prefix (per card) / doc raw |
| 2 | `jinaai/jina-embeddings-v2-base-code` | 161M | Apache-2.0 | code-specialized, 8k ctx (same family as prod `jina-code-v2`, closest dev↔prod parity) | sentence-transformers / ONNX | none (symmetric) |
| 3 | `Alibaba-NLP/gte-modernbert-base` | 149M | Apache-2.0 | **strongest small open code number found** (CoIR ~79.31), 8k ctx | clean ONNX CPU path | none |
| 4 | `ibm-granite/granite-embedding-english-r2` | ~149M | Apache-2.0 | strong CoIR, ModernBERT-based, recent | sentence-transformers / ONNX | none (verify) |

Excluded by constraints (record so they're not re-litigated): EmbeddingGemma
(768 but Gemma license ≠ OSI open-source); SFR-Embedding-Code-400M & jina-code /
jina-v3 (`cc-by-nc`); anything >1B (bge-code-v1 1.5B, codesage-large 1.3B).

## Protocol (per candidate)

Run the methodology already defined in `embedder-eval-plan.md`; only the model +
prefix change.

1. **Set the embedder.** Swap via `EMBEDDER_BACKEND` + the per-backend model id
   (`CODERANK_EMBED_MODEL` constant / `LM_STUDIO_EMBED_MODEL` / `OLLAMA_EMBED_MODEL`);
   confirm the exact knob for an arbitrary HF model in `codebase_rag/embedder.py`
   + `app/config.py`. **Apply the model's correct query/doc prefix** — a wrong or
   missing prefix invalidates the comparison (this is the #1 footgun). Local
   embedders on the 3060; keep the service CPU-pinned (`CUDA_VISIBLE_DEVICES=""`).
2. **Re-index** the eval corpus (768 schema, no change — cheap).
3. **Primary metric — first-stage recall** (`scripts/run_recall.py` per the
   plan): recall@10 / @25 / @50 + MRR of oracle-span symbols over the 15 design
   tasks + 6 probes, querying raw `/search/semantic` (no graph/boosts/rerank).
4. **Secondary** — composed lift via `scripts/run_arms.py` and
   `scripts/run_probes.py --check` (confirm the new ceiling flows through and
   nothing regresses).
5. **Cost** — record CPU query-embed latency (p50/p95), index build time, and
   index size beside quality.

## Pre-registered ship criteria (from the eval plan §7)

Swap-class (768, no migration): **ship a candidate if mean recall@25 improves
≥ 0.05 over `e5-base-v2`** with **zero** composed regressions (no task drops
below its current lift) and probes stay **6/6**. Low bar to adopt — there's no
migration cost. Report latency/index-cost alongside; a code embedder that
doubles index time for +0.02 recall is a different decision than a free one.

## Cross-repo comparability (the dimension question)

This roster and the 1024 roster share the **same 15 oracle tasks + 6 probes,
the same `run_recall.py` recall@k/MRR metric, and the same raw
`/search/semantic` path**. That makes the 768-repo winner and the 1024-repo
winner directly comparable. If the best 768 code embedder here matches or beats
the best 1024 model there on recall@25 at lower CPU latency, the 1024
separability ceiling is not paying for itself on this corpus — and the cheaper,
no-migration 768 path wins. If 1024 opens a real recall gap at scale, that
vindicates the migration cost. Run both rosters before deciding either.
