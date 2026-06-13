# Spike: cross-encoder reranker as the bundle-truncation tie-break (cgr-001)

Status: spike spec, 2026-06-13. Imported from the AgentAlloy session.
A ~30-minute go/no-go before choosing among decomposition (heavy) /
plateau / overfit for the cgr-001 storage facet.

## The diagnosis this addresses

cgr-001 ("Re-ingest only changed files … instead of a full re-parse")
scores 0.67 because the storage symbols (`_generate_semantic_embeddings`,
`GraphUpdater.run`) are **reachable but mis-ranked**:

- They enter the bundle pre-truncation at depth 1 (the graph reaches
  them fine).
- `compute_bundle_scores` in `app/routers/context_bundle.py` (~L516-555)
  clamps every depth-1 neighbour to `neighbor_ceiling = seed_floor * 0.99`
  — a **flat band**. ~150 depth-1 neighbours land at the identical
  ceiling value.
- Truncation (348 → 63) picks survivors from that flat band; the storage
  method loses an **alphabetical tie-break** and is dropped.
- No lexical or *bi-encoder semantic* term ranks storage above the floor:
  the query embedding alone never encoded "look for storage code."

This is a **ranking problem, not a reachability problem** — the symbol is
in the pool, scored too low. Query decomposition (the documented fix)
solves it by bolting on a *reachability* mechanism (a new sub-query that
re-seeds storage with a high score) and pays an LLM-generation call in the
`/context-bundle` hot path TheForge hits. A reranker solves the ranking
problem *as a ranking problem*, with a scorer instead of a generator.

## Why a cross-encoder can succeed where the bi-encoder failed

The "no semantic term ranks it" finding is a statement about the
**bi-encoder** (e5-base-v2): it embeds the query in isolation, so it can't
encode the two-hop inference "re-ingest changed files → re-generate
embeddings → vector store." A **cross-encoder** scores the (query,
document) pair *jointly*, attending across both at scoring time — it can
model that inference, especially since the storage method's body literally
contains `vector_store` and `duckdb`. This is the textbook bi-encoder vs
cross-encoder distinction and the core reason to expect a different result.

## Candidate model

**qwen3-reranker-0.6b** (cross-encoder, pair scoring). Distinct from the
existing `app/services/reranker.py` (CodeRankLLM, *generative listwise*):

| | existing search reranker | this spike |
|---|---|---|
| model | CodeRankLLM (generative) | qwen3-reranker-0.6b (cross-encoder) |
| mechanism | emits a permutation | per-pair yes/no logit → score |
| stage | Stage-1 search top-k (~50), opt-in | bundle neighbour-band tie-break |
| cost | generation, max_tokens 2048, thinking-mode quirks | ~35 ms/pair, deterministic, no generation |
| "nothing relevant" | n/a | natural: no pair clears threshold |

The cross-encoder is cheap enough to run **in the bundle hot path**, which
the generative reranker is not. (It's also embedder-agnostic — it re-scores
candidates regardless of e5 vs qwen first-stage embeddings, so it composes
with the existing e5 pipeline unchanged.) Serving notes from the AgentAlloy
spike: **llama.cpp's `/v1/rerank` endpoint does NOT work for this GGUF** (it
skips the instruction template, scores everything ~0); score via
`/v1/completions` yes/no logprobs with the official Qwen3-Reranker template.
GGUF already on this host's shared model store; AgentAlloy serves it on
`:60001`.

## The go/no-go test (do this first — ~30 min)

Pull the actual cgr-001 candidate set and score it. Pass criterion is
purely about separation:

1. From a live cgr-001 bundle, capture the **tied neighbour band** — the
   depth-1 symbols clamped at `neighbor_ceiling` (the ~150 that currently
   tie). Include `_generate_semantic_embeddings` and `GraphUpdater.run`
   (the symbols truncation drops) and a representative sample of the noise
   neighbours that currently *survive* alphabetically.
2. For each, build the reranker document from what the bundle would index
   for that symbol — qualified name + signature + body/snippet (mirror
   whatever text the bundle already has; that's what production would
   feed).
3. Score every (cgr-001 query, symbol-doc) pair, 5 reps (verify
   determinism).
4. **PASS** = the storage symbols score clearly above the noise band such
   that a single threshold (or a top-N-within-band cut) would keep them in
   the surviving 63. **FAIL** = storage scores indistinguishable from the
   noise floor (a 0.6B general reranker may not make the domain inference;
   that's the thing we're testing).
5. Report: storage-symbol scores vs noise-band score distribution, the
   separating threshold if one exists, and p50 latency for the band size.

Optional A/B: run the same set through the existing CodeRankLLM (it's
code-specialized — may make the inference better) to quantify the
cheap-cross-encoder vs heavy-code-reranker tradeoff on the exact failing
facet.

## If it passes — integration point

Re-score **within the neighbour band only**, preserving the seed-precedence
invariant `compute_bundle_scores` already guarantees:

- Keep seeds and the `neighbor_ceiling` clamp exactly as-is (neighbours
  still sort strictly below seeds).
- Within the band, replace the current value-tie + alphabetical
  `order_symbols_by_score` tie-break with the reranker's relevance score
  (rescaled into the sub-ceiling band so the invariant holds).
- Bound the cost: only re-score symbols actually *at* the flat ceiling
  (the genuine ties), not the whole pool — that may be far fewer than 150.
  Measure p50 against the hot-path budget; ~150 × 35 ms with `--parallel 4`
  ≈ 1.3 s is the worst case.
- **Fail-open** (same discipline as the existing reranker): reranker
  unreachable / timeout / unparseable → today's truncation, byte-for-byte.
- Flag it (`?rerank_bundle=true` or env), default off, until a bundle-eval
  pass — not just cgr-001 — clears it. cgr-001 alone is one facet; the
  general claim is "rescues any mis-ranked-but-reachable symbol," and that
  needs the full eval, or it's overfitting to one task.

## If it fails

The cheap fix is ruled out for ~30 minutes of spend → fall back to query
decomposition (flag-gated, LLM out of the default hot path) or declare the
0.9778 plateau and move to symbol cards / embedder eval. The spike's value
is settling that empirically rather than building the heavy mechanism on a
guess.

## Spike results (2026-06-13, code-indexer session)

Ran via `scripts/spike_rerank_cgr001.py` against the live reranker on
`:60001`. **One correction to the spec's premise:** the storage targets
are *not* in a flat tie band. With the current `compute_bundle_scores`,
cgr-001's 327 depth-1 neighbours span a real score range; the targets are
genuinely *low-scored* (`run` 0.205, `_generate_semantic_embeddings` ~0),
ranked 59th/77th among neighbours — not alphabetical-tie casualties. So
the integration is **re-score the neighbour pool**, not just the ceiling
band (the band is ~2 symbols, not ~150).

Serving note confirmed: llama.cpp returns OpenAI-style `top_logprobs`
(`{token, logprob}`) on `/completion` with `n_probs`; `/v1/rerank` unused.
Score = `softmax({yes,no}-logprobs)`, aggregating case variants.

Findings (327-neighbour pool, single rep + 5-rep determinism check):
- **Determinism: PASS** — spread `0.0` at `temperature 0` across 5 reps.
- **Latency: ~65 ms/pair**, 21 s for 327 sequential; ≈5 s at `--parallel 4`.
  A bounded re-score (top-N neighbours) sits well inside the hot-path budget.
- **General ranking power: STRONG.** `GraphUpdater.run` moves from
  bi-encoder neighbour-rank **59 → cross-encoder rank 5/325**. The
  mechanism demonstrably rescues a mis-ranked-but-reachable symbol — its
  stated purpose. The score distribution is sharply discriminative (noise
  p50 = 0.0000, p90 ≈ 0.048; relevant symbols 0.8+).
- **cgr-001 facet closure: PLAUSIBLE YES, but marginal.** The grader
  matches `vector_store`/`duckdb` substrings over *surviving snippets*, and
  6 such carrier symbols are in the pool. The reranker promotes one —
  `verify_stored_ids` — to rank **39, inside the ~42 surviving neighbour
  slots**. The two symbols the spec named are a red herring:
  `_generate_semantic_embeddings` scores ~0 (correctly — its body is a
  `skip_embeddings` no-op stub here: "Skipping built-in embedding pass
  (handled by caller)"), and `run` (rank 5) carries no facet string.

**Verdict: GO**, with the caveat that cgr-001 closure is marginal (39/42)
and must be confirmed by the real integration + full arms benchmark, not
the spike's top-N approximation. The general mechanism is validated
independent of cgr-001. The CodeRankLLM A/B is **blocked** — LM Studio is
retired (`LM_STUDIO_URL` empty); running it would need a 7B generative
model stood up on the GPU-contended 3060 (qwen3-reranker + Ollama already
there), so the cheap-vs-code-specialized question is deferred.

## Integration result (2026-06-13) — DID NOT CLEAR THE GATE

Built the full flag-gated integration (`app/services/bundle_reranker.py`,
`BUNDLE_RERANK_*` config, `rerank_bundle` request flag, neighbour
re-scoring folded into `compute_bundle_scores` + a relevance-primary
refill mode in `_truncate_to_budget`). Ran the **full 15-task arms gate**
with the reranker on (all neighbours scored, no cap/deadline cutoff):

| | mean lift | dsg-cgr-001 | dsg-tf-003 |
|---|---|---|---|
| flag OFF (baseline) | **0.9778** | 0.67 | 1.00 |
| flag ON | **0.9611** | 0.67 | **0.75** |

**The only benchmark effect is a regression.** Two findings kill the
hot-path case for this model:

1. **cgr-001 not closed.** With the *real* bundle docs (qname + snippet),
   every storage carrier scores barely above the noise floor —
   `verify_stored_ids` 0.038, `_generate_semantic_embeddings` 0.020. The
   spike's "rank 39 within 42 slots" was a knife-edge artifact of an
   optimistic slot count; the real bundle has ~24 neighbour slots and 0.038
   doesn't make the cut. The cross-encoder, *like the bi-encoder*, does not
   judge the storage code relevant to "re-ingest only changed files." Two
   independent models agreeing is strong evidence the **storage facet is
   over-specified** relative to the query text — closing it needs a
   mechanism that *manufactures* the missing sub-intent (query
   decomposition), which is arguably gaming the benchmark.
2. **tf-003 regressed 1.00 → 0.75.** Relevance-primary refill let the
   reranker drop `requireIdentity` (the auth-facet symbol the PR #6 breadth
   reduction had rescued) in favour of symbols it scored higher (e.g.
   `GraphUpdater.run`, 0.81 — correctly rescued in the general sense, but
   facet-irrelevant). The 0.6B general reranker's relevance ordering is
   *worse* than the existing depth+breadth heuristics for these tasks.

**Decision: not shipped.** Implementation kept on branch
`feat/bundle-reranker-spike` (unmerged; flag default off, fail-open — a
no-op in production), `main` stays at 0.9778. The harness and integration
are preserved so the **CodeRankLLM A/B** (the real open question — does a
*code-specialized* reranker make the inference the general one cannot?) can
reuse them when LM Studio or an equivalent CodeRankLLM endpoint is back.
The general-ranking win (`run` 59→5) is real but not benchmark-visible
(14/15 already at 1.00) and not worth a regression elsewhere.

## Cross-tool note

If this passes, both tripod tools converge on **qwen3-reranker-0.6b** as a
shared scoring layer (AgentAlloy uses it for Stage-B fragment re-rank and
is evaluating it for phase-intent classification). One reranker serving
setup, one model to version — the operational consolidation the tripod
wants, independent of the embedders differing (e5 here, qwen3-embedding in
AgentAlloy).
