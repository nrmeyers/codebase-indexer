# Embedder evaluation plan (measurement-first)

**Status:** scoping (2026-06-13). No code yet. This defines the criteria
*before* building so the decision is made on data, not on a vibe. Owner
sign-off requested on the candidate set and the pre-registered margin
before any re-index runs.

## Why this is the bigger lever

The embedder sets the **first-stage recall ceiling**. Everything
downstream — boost bands, graph expansion, snippet hydration, the
(rejected) bundle reranker — can only re-order or drop what the bi-encoder
already surfaced. If the right symbol never enters the semantic top-k, no
later stage recovers it. So unlike the bundle reranker (a re-ordering layer
that capped out fast and then regressed), a better embedder raises the
ceiling for *every* task at once. That is the thesis to test.

Today's plateau (mean lift 0.9778, 14/15 design tasks at 1.00) is measured
*after* all the deterministic legs. It does **not** tell us how much of
that is the embedder vs the boosting machinery. The eval must separate the
two (see "Primary metric").

## Candidates

| backend | model | dim | schema impact | prod parity |
|---|---|---|---|---|
| `local` (current baseline) | `intfloat/e5-base-v2` | 768 | none | dev only |
| `sagemaker` (current prod) | `jina-code-v2` | 768 | none — same `FLOAT[768]` | **prod runs this** |
| code-specialized (local) | `nomic-ai/CodeRankEmbed` or jina-code-v2 local | 768 | none (swap via `EMBEDDER_BACKEND`) | matches prod family |

Notes:
- `local`/`sagemaker`/`tei` already share the `FLOAT[768]` schema and are
  env-swappable (`EMBEDDER_BACKEND`) — cheap to A/B, no migration.
- A code-specialized embedder (CodeRankEmbed/jina-code) is the most likely
  to actually move recall, since e5-base-v2 is general-prose-trained and
  code identifiers are out-of-distribution for it. This is the candidate I
  expect to win on the primary metric.

## Primary metric: first-stage recall of oracle spans (NOT composed lift)

Composed lift is contaminated by the boosting machinery, so it can't
isolate the embedder. Add a **recall harness** that, per benchmark task,
queries raw `/search/semantic` (no graph, no boosts, no rerank) and checks
whether the oracle-span symbols appear in the top-k.

- **recall@10 / recall@25 / recall@50** of oracle-span symbols, averaged
  over the 15 design tasks + the 6 probes, per embedder.
- Report **mean reciprocal rank** of the first oracle hit too — "reachable
  but at rank 40" vs "rank 5" matters for whether the existing boosts can
  finish the job (cf. §5's "expansion makes things reachable, not top-k").
- Secondary: end-to-end composed lift (the existing `run_arms.py`) and
  probes `--check`, to confirm the new ceiling actually flows through and
  nothing regresses.

The discriminating question this answers: *is e5-base-v2 leaving recall on
the table that a code embedder would capture, or is first-stage recall
already saturated and the remaining gaps are all downstream?*

## Pre-registered ship criteria

Borrowed from the methodology doc's §7 ship gate (beat the
post-previous-stage baseline by a pre-registered margin):

1. **Swap-class candidates (768-dim, no migration):** ship if mean
   recall@25 improves by **≥ 0.05** over e5-base-v2 with **zero** composed
   regressions (no task drops below its current lift) and probes stay 6/6.
   Cheap to try; low bar to adopt because there's no migration cost.
2. Report **token/latency cost** beside quality for each (embed throughput,
   index size, query-embed latency). A code embedder that doubles index
   time for +0.02 recall is a different decision than one that's free.

## Sequence (cheapest discriminator first)

1. **Stand up the recall harness** (`scripts/run_recall.py`): raw
   `/search/semantic` recall@k + MRR of oracle spans. Reusable across
   embedders. ~half a day; no model work.
2. **Baseline e5-base-v2** on the harness — establishes the recall floor and
   tells us immediately whether recall is even the bottleneck.
3. **A/B a 768-dim code embedder** (`EMBEDDER_BACKEND` swap + re-index the
   three repos — cheap, no schema change). This is the high-information,
   low-cost experiment.

## Constraints / traps to carry in

- GPU: local embedders run on the 3060 (the 3090 is reserved); the service
  itself stays CPU-pinned (`CUDA_VISIBLE_DEVICES=""`). Don't co-locate a new
  embedder with the qwen3-reranker if it risks OOM.
- A dim change invalidates every committed snapshot/baseline — re-run
  `run_arms.py` and `run_probes.py` to regenerate, and bump any index
  format/version marker so corpora are auditable (§5, §8).
- Prod runs `sagemaker` jina-code-v2 today; a local code embedder in the
  same family is the closest dev↔prod parity, which is its own argument for
  picking it over e5 regardless of the benchmark.
