# Phase 9 — Cross-Repo Unified Ranking

**Owner:** Zachary Matthews (zmatthews@navistone.com)
**Date:** 2026-04-30
**Status:** plan, awaiting Phase 8 (HNSW/VSS) green
**Depends on:** Phase 4 (`forge_*` Prom metrics) for adoption telemetry; Phase 8 (recall stability under HNSW) so eval baselines are not chasing a moving target
**Supersedes (on ship):** ADR-0003 — to be retired by a new ADR-0006 written at the end of 9b
**Companion docs:** `TEAM_DEPLOYMENT_PLAN.md` §3 Phase 9, `docs/adr/0003-defer-cross-repo-unified-ranking.md`

> The trigger condition in ADR-0003 is met once 5+ repos are indexed AND the
> orchestrator surfaces measurable cross-repo recall complaints. Phase 9
> ships the implementation; sub-phase 9a produces the evidence that picks
> the merge strategy.

---

## 0. Goals and non-goals

**Goals**
1. When TheForge orchestrator runs a `repo: '*'` fan-out, the merged top-N is a globally-relevance-ranked list rather than a per-repo round-robin.
2. The chosen normalization beats the current `concat + interleave` baseline on a hand-labelled holdout — target nDCG@10 ≥ 0.85 vs. human-ideal, OR ≥ +0.05 nDCG@10 absolute over baseline (whichever is more lenient given small-N labelling).
3. Per-result provenance (`repoSlug`, `originalScore`, `originalRank`) is preserved through the merge so downstream re-rankers and the UI can still reason about it.
4. Behaviour is gated behind `CROSS_REPO_RANK_STRATEGY={concat|rrf|zscore}` env flag, default `concat`, flipped to `rrf` after 9a passes.

**Non-goals**
- Building a unified vector index spanning all repos (ADR-0003 Option B). Each repo keeps its own DuckDB; merging stays a TheForge-side concern.
- Cross-encoder re-ranking over the merged set — that's a separate decision (ADR-0007 candidate) and depends on rerank capacity which Phase 4 only just made measurable.
- Changing the Code Indexer wire contract for single-repo searches.
- Relevance feedback / click-through learning to rank — out of scope until we have a UI surface that tracks it.

---

## 1. Background

Today's behaviour, end-to-end (see `src/services/orchestrator.ts:267 fetchCodeContext`):

```
codeIndexerRepo === '*'   →   resolve to string[] of all indexed repos
                          →   N parallel POST /context-bundle calls (perRepoK)
                          →   per-repo seed lists collected
                          →   interleave by rank (rank-0 from each repo, then rank-1, …)
                          →   slice to DEFAULT_SYMBOL_LIMIT (20)
                          →   feed into prompt
```

The interleave is rank-based (round-robin) not score-based. So a repo whose
top-1 is barely above its noise floor occupies the same merged-rank-1 slot
as a repo whose top-1 is 0.92 cosine. Worse, cosine score distributions
differ between repos for reasons unrelated to relevance:

- Corpus size — bigger repos have more candidates competing, mean cosine drifts down.
- Embedding density — repos with lots of near-duplicate symbols compress the score range.
- Symbol-type mix — a repo that's 90% glue code embeds very differently from one with rich domain models.

The Code Indexer side already returns `score: number` (cosine, [-1, 1]) on
each `SemanticResult`, so we have the raw signal — we just don't currently
use it across repos.

---

## 2. Phase split

| Sub-phase | Goal | Approx |
|---|---|---:|
| 9a | Build evaluation harness, label holdout, run 3 candidate strategies, decide. | 1 d |
| 9b | Ship the chosen strategy behind an env flag, wire metrics, update ADR. | 0.5 d |

9b cannot start before 9a completes because the strategy choice is the
input to 9b's implementation. The harness from 9a stays — it's the
regression check in CI for any future merge-strategy change.

---

## 3. Sub-phase 9a — evaluation harness

### 3.1 Holdout design

Lives at `code-indexer-service/eval/cross_repo/`.

- **Repo set:** 3–5 repos already indexed in the team environment. Mix sizes (one small ≤ 5k symbols, one mid 20–50k, one large 100k+) so score-distribution skew is realistic.
- **Query set:** 10–20 cross-repo queries. Source pool:
  - User-submitted queries surfaced from TheForge audit logs that hit `repo: '*'` — privileged source, real distribution.
  - Synthetic supplements covering canonical task shapes: "where is auth handled", "find all migration scripts", "rate-limiter implementation", "websocket reconnect logic".
- **Candidate set per query:** the union of top-100 cosine hits from each repo (so 100×N candidates pre-dedup). Capped at 500 to bound labelling.
- **Labels:** ternary — `2 = ideal answer`, `1 = relevant context`, `0 = irrelevant`. Two human labellers, disagreement-resolution by a third. Labels stored in `eval/cross_repo/labels/<query_id>.yaml`.

**Sample size note.** 10–20 queries × ~100 labels each ≈ 1–2k labels. Small
enough for two humans in a half-day, large enough that nDCG@10 differences
of 0.05 are above the noise floor (variance via 1000-bootstrap on the
query set, reported alongside the point estimate).

### 3.2 Candidate strategies

#### A. Per-repo z-score normalization

```
zᵢⱼ = (sᵢⱼ - μⱼ) / σⱼ          # j = repo index, i = result index within repo
```

Then merge by descending z. Pros: simple, preserves distance information.
Cons: assumes scores within a repo are roughly normal; tail behaviour is
poor when σ is small (one good answer + a flat noise floor blows up the z
of the good answer in a way that doesn't reflect cross-repo confidence).
Needs a minimum-population check (e.g. fall back to raw score when N < 5).

#### B. Min-max normalization

```
nᵢⱼ = (sᵢⱼ - min(sⱼ)) / (max(sⱼ) - min(sⱼ))
```

Pros: bounded [0, 1], easy to reason about. Cons: extremely sensitive to
the min/max of the candidate window — change top-k from 10 to 20 and the
ordering can flip. Also collapses two repos with very different absolute
quality onto the same [0, 1] range, which is the opposite of what we want.

#### C. Reciprocal rank fusion (RRF)

```
score_fused(d) = Σⱼ  1 / (k + rankⱼ(d))     # k = 60 (Cormack et al. default)
```

Where `rankⱼ(d)` is `d`'s 1-indexed rank in repo j (or ∞ if absent). Pros:
- Rank-only — completely sidesteps cross-repo score-distribution issues.
- No calibration / tuning needed; the `k=60` constant is robust across
  text-retrieval domains (well-known result from TREC literature).
- Cheap (O(N log N)) and trivially deterministic.

Cons:
- Throws away the absolute score signal. A reranker downstream that wants
  cosine distance has to be passed the original score alongside the RRF
  rank. (Mitigated in 9b by carrying `originalScore` through the schema.)
- Two repos of wildly different quality contribute equally — but this is
  the *correct* behaviour if we believe the per-repo top-k is itself a
  decent confidence signal, which holds for cosine retrieval at k ≤ 50.

### 3.3 Recommended pre-eval bias

Start the eval with **RRF as the prior favourite**, z-score as backup, min-
max as ablation. Reasoning:

1. RRF's robustness to score-distribution mismatch is exactly the failure
   mode we're trying to fix. Z-score addresses the symptom (calibration);
   RRF addresses the cause (don't trust cross-repo absolute scores).
2. RRF is the documented winner across multiple TREC tracks for combining
   rankings from heterogeneous systems — that's our problem analogue.
3. RRF has no hyperparameters to tune against the holdout (k=60 is a fixed
   prior), so we don't risk overfitting.

If the eval contradicts this, ship the winner. Document the surprise in
ADR-0006 so future readers don't second-guess the choice.

### 3.4 Eval harness structure

```
eval/cross_repo/
  run.py                      # orchestrates: load labels, run strategies, compute metrics
  strategies/
    concat.py                 # baseline: today's interleave behaviour
    zscore.py
    minmax.py
    rrf.py
  labels/
    <query_id>.yaml           # human labels
  RESULTS_<date>.md           # human-readable comparison table
  RESULTS_<date>.json         # machine-readable, for CI regression
```

`run.py` writes both the markdown table AND a JSON file checked into
`tests/fixtures/cross_repo_baseline.json`. CI uses the JSON for regression.

Metrics reported per strategy:

| Metric | Why |
|---|---|
| nDCG@{1,3,5,10} | Primary — accounts for ternary relevance grading |
| recall@{1,3,5,10} | Coverage — did we surface any ideal answer at all |
| MRR | Latency-of-first-good-answer proxy |
| Per-repo coverage | Diversity check — does the merged top-10 still pull from ≥ 2 repos when ideal answers span repos |
| Wall-clock p95 | Each strategy's overhead vs. concat baseline |

### 3.5 Acceptance for 9a

- [ ] 10–20 queries labelled, two-labeller agreement κ ≥ 0.6.
- [ ] All four strategies (concat / zscore / minmax / rrf) run end-to-end against the holdout.
- [ ] `RESULTS_<date>.md` checked in with verdict + bootstrap CI.
- [ ] One strategy is unambiguously chosen (or, if two tie within CI, choose the simpler — RRF).

---

## 4. Sub-phase 9b — implementation

### 4.1 Where the merge happens

Two locations were possible:

| Option | Pros | Cons |
|---|---|---|
| (a) Extend `code-indexer-service` to accept `repo: ['list','of','repos']` and return a unified ranking. | Single round-trip from TheForge. Indexer caches the merge. | Indexer becomes opinionated about which repos are "together". DBs are still independent so there's no real performance win. Cross-service breaking change. |
| (b) Keep per-repo fan-out at the network edge; merge in TheForge. | Indexer stays stateless about repo grouping. No wire-contract change. Easy to swap strategies behind an env flag. | TheForge does N HTTP calls (already does today). |

**Decision: (b).** The indexer's job is "given a repo, return its best
matches". Merging is an orchestration concern. TheForge already does the
fan-out — we're just replacing the merge function.

### 4.2 New module

```
TheForge/src/services/cross-repo-rank.ts
```

API surface:

```ts
export type CrossRepoStrategy = 'concat' | 'rrf' | 'zscore';

export type RankedResult<T> = T & {
  repoSlug: string;
  originalScore: number;     // raw cosine, untouched
  originalRank: number;      // 1-indexed position in the source repo's list
  fusedScore: number;        // strategy output; for 'concat' this equals originalScore
};

export function rrfMerge<T extends { score: number }>(
  perRepo: Array<{ repoSlug: string; results: T[] }>,
  options?: { k?: number; limit?: number },
): RankedResult<T>[];

export function zScoreMerge<T extends { score: number }>(
  perRepo: Array<{ repoSlug: string; results: T[] }>,
  options?: { limit?: number; minPopulation?: number },
): RankedResult<T>[];

export function concatMerge<T extends { score: number }>(
  perRepo: Array<{ repoSlug: string; results: T[] }>,
  options?: { limit?: number },
): RankedResult<T>[];
```

The `concatMerge` reproduces today's interleave so the `concat` flag value
is a true bit-for-bit baseline (regression-friendly).

`rrfMerge` reference impl (illustrative — actual implementation in
`cross-repo-rank.ts`):

```ts
export function rrfMerge<T extends { score: number }>(
  perRepo: Array<{ repoSlug: string; results: T[] }>,
  { k = 60, limit = 20 }: { k?: number; limit?: number } = {},
): RankedResult<T>[] {
  const fused = new Map<string, RankedResult<T>>();
  for (const { repoSlug, results } of perRepo) {
    results.forEach((r, idx) => {
      const rank = idx + 1;
      const key = identityKey(r, repoSlug);  // FQN + repoSlug — see §4.4
      const contribution = 1 / (k + rank);
      const existing = fused.get(key);
      if (existing) {
        existing.fusedScore += contribution;
      } else {
        fused.set(key, {
          ...r,
          repoSlug,
          originalScore: r.score,
          originalRank: rank,
          fusedScore: contribution,
        });
      }
    });
  }
  return [...fused.values()]
    .sort((a, b) => b.fusedScore - a.fusedScore)
    .slice(0, limit);
}
```

### 4.3 Wiring into the orchestrator

Touch points in `src/services/orchestrator.ts:267 fetchCodeContext`:

```diff
   const bundles = await Promise.all(
     repoPath.map((rp) => buildContextBundle(rp, userMessage, perRepoK, ...)),
   );

-  // Interleave by per-repo rank …
+  const strategy = (process.env.CROSS_REPO_RANK_STRATEGY ?? 'concat') as CrossRepoStrategy;
+  const merged = mergeBundles(bundles, repoPath, strategy);
+  return merged.slice(0, DEFAULT_SYMBOL_LIMIT);
```

`mergeBundles` is a small adapter that:
1. Drops `Result.Err` bundles (today's behaviour — non-fatal per repo).
2. Builds the `perRepo: [{ repoSlug, results }]` shape from the surviving bundles.
3. Routes to the chosen strategy.

### 4.4 Identity / dedup

Two repos can both contain a symbol with the same FQN (e.g. `utils.format`
in two unrelated codebases). They are *not* the same symbol. The fusion
key is `${repoSlug}::${fqn}` — collisions only occur within a repo, where
the indexer already guarantees uniqueness.

This means RRF will never fuse "the same idea seen in two repos" into a
single boosted entry. That's correct: cross-repo de-duplication by
embedding similarity is a separate problem (vendored libraries, copied
utility code) and would be its own ADR.

### 4.5 Score schema change

`SymbolSnippet` (consumed by prompt assembly) gets three optional fields:

```ts
type SymbolSnippet = {
  name: string;
  path: string;
  content: string;
  // New — present only on cross-repo fan-out:
  repoSlug?: string;
  originalScore?: number;
  originalRank?: number;
};
```

Optional so single-repo callers stay untouched. Prompt assembly ignores
the new fields; the UI surfaces them (§5).

---

## 5. UI surfacing — SearchPlayground

`web/src/pages/SearchPlayground.tsx` already renders semantic-search
results with a per-result score badge. Cross-repo results add:

- **`repoSlug` chip** next to the file path (existing visual idiom: zinc-700 background, indigo-300 text — matches `RepoBadge.tsx`).
- **Hover tooltip** showing `originalScore` and `originalRank` for transparency. ("Repo X cosine 0.84, rank #3 in repo".)
- **Strategy indicator** at the top of the result list — small grey label, "merged via RRF" / "concatenated" — sourced from the response (the API echoes the active strategy).

Same chips on the Chat page's "Sources" panel when it shows cross-repo
context.

No new UI primitive needed. Lucide `Layers` icon for the strategy
indicator (no emoji).

---

## 6. Test plan

### 6.1 Unit — `tests/unit/services/cross-repo-rank.test.ts`

- RRF math against three hand-computed fixtures (single-repo, 2-repo no-overlap, 2-repo full-overlap).
- z-score normalization with σ=0 fallback (should not produce NaN).
- min-max with empty input.
- Identity key correctness — same FQN in two repos produces two entries.
- `concatMerge` reproduces today's `interleave` byte-for-byte against a captured fixture.

### 6.2 Integration — `tests/integration/orchestrator-cross-repo.test.ts`

- Mock 3-repo `buildContextBundle` returns with hand-crafted score distributions.
- Assert merged top-10 ordering matches expected RRF output.
- Assert per-repo failure isolation still holds (one repo errs → other two still merge).
- Assert env flag `CROSS_REPO_RANK_STRATEGY=concat` reproduces the legacy interleave.

### 6.3 Regression — eval harness in CI

`pnpm test:eval` (new script) runs the 9a harness against the labelled
holdout and compares against the JSON baseline. Fails CI if recall@10
regresses by > 0.02 absolute on any strategy. Runs nightly, not per-PR
(it needs labelled fixtures and a populated index — too expensive for
every push).

### 6.4 Manual smoke

- `CROSS_REPO_RANK_STRATEGY=rrf pnpm dev` → run a known cross-repo query in SearchPlayground → confirm ordering changed and chips render.
- Flip back to `concat` → ordering matches pre-Phase-9.

---

## 7. Phase 4 metrics — adoption telemetry

Add to `forge_orchestration_*` registry (TheForge side, `src/services/metrics.ts`):

| Metric | Type | Labels |
|---|---|---|
| `forge_orchestration_cross_repo_searches_total` | Counter | `strategy` (concat\|rrf\|zscore), `repo_count` |
| `forge_orchestration_cross_repo_merge_seconds` | Histogram | `strategy` |
| `forge_orchestration_cross_repo_diversity` | Histogram | (number of distinct repos in returned top-10) |

The diversity histogram gives us a fast read on "is RRF actually pulling
from multiple repos?" — if the distribution collapses to 1, the merge is
a no-op.

Grafana dashboard panel additions (Phase 4 dashboard JSON, infra/grafana/):

- Stacked-area: cross-repo searches per minute by strategy.
- p95 merge latency by strategy.
- Average diversity (top-10 distinct repos) — sanity gauge.

---

## 8. Trade-offs and decision log

### A. RRF discards distance information

**Symptom.** A downstream cross-encoder reranker (potentially Phase 10+)
wants cosine distance as a feature, not a fused rank.

**Resolution.** `originalScore` is preserved on every `RankedResult` and
flows through `SymbolSnippet`. The reranker (when added) can use either
`fusedScore` (rank-fusion confidence) or `originalScore` (within-repo
similarity) — both are present.

### B. RRF treats all repos as equally good

A 100k-symbol mature codebase and a 500-symbol scratch repo contribute
equally to the fusion. If "scratch repo" outputs noise its noise gets the
same rank-1 weight as the production repo's rank-1.

**Resolution (deferred).** Optional per-repo weight `w_j` would change
the formula to `Σⱼ wⱼ / (k + rankⱼ(d))`. Default all weights to 1.0; a
future ADR can wire weights to a "repo trust" signal (recency of last
index? PageRank centrality? user vote?). Out of scope for 9b.

### C. Min-population fallback for z-score

If 9a picks z-score, we must handle `σ=0` (every result tied) and
`N < 5` (population too small to estimate moments). Fallback: raw cosine.
Documented in `zscore.ts` regardless of which strategy ships, since the
file exists for the eval harness.

### D. RRF k=60 is a magic constant

We don't sweep `k` against the holdout — small N would overfit. The
Cormack et al. default is well-established. If 9a's eval suggests k=60 is
wrong for our domain, the ADR documents the override and we sweep k in a
follow-up.

### E. Cross-repo dedup by embedding similarity is out of scope

A vendored library copy-pasted into two repos will produce two entries.
Acceptable — the prompt assembly tier cap (10k tokens) absorbs duplicate
content, and the UI shows both with their `repoSlug` so the user can see
the duplication. Revisit only if user feedback flags it.

---

## 9. Rollout

| Step | Trigger | Action |
|---|---|---|
| 1 | 9a `RESULTS_<date>.md` checked in | Read verdict; if RRF wins, set default `CROSS_REPO_RANK_STRATEGY=rrf` in `.env.example` (still `concat` in production until §9.4). |
| 2 | 9b PR merges | Production stays on `concat` (env override). Internal devs on `rrf` via local `.env`. |
| 3 | One week of internal usage | Check Phase 4 metrics — `cross_repo_diversity` median ≥ 2, no p95 merge-latency regression > 50 ms. |
| 4 | Metrics green | Flip production env to `rrf`. Document flip in CHANGELOG. |
| 5 | 30 days post-flip | Drop the `concat` code path? **No** — keep as feature flag until ADR-0006 is reviewed. Removal is a separate Phase 10+ cleanup. |

Rollback: set `CROSS_REPO_RANK_STRATEGY=concat`. No data migration, no
schema change, no restart needed beyond the env reload.

---

## 10. Acceptance gate (Phase 9 done when…)

- [ ] 9a holdout labelled, harness checked in, results ADR-citeable.
- [ ] 9b `cross-repo-rank.ts` shipped with all three strategies + unit tests.
- [ ] Orchestrator wired through env flag; default `concat` for safety, internal `rrf`.
- [ ] Phase 4 metrics emitting; Grafana panels render live data.
- [ ] SearchPlayground shows `repoSlug` chip and strategy indicator.
- [ ] ADR-0006 supersedes ADR-0003 with the strategy choice + evidence.
- [ ] CI regression harness wired to nightly run.

---

## 11. Critical files for implementation

**Read first (context):**
- `code-indexer-service/docs/adr/0003-defer-cross-repo-unified-ranking.md` — the deferred ADR, baseline rationale.
- `code-indexer-service/.planning/TEAM_DEPLOYMENT_PLAN.md` §3 Phase 9 — the parent outline.
- `TheForge/src/services/orchestrator.ts:267 fetchCodeContext` — current fan-out + interleave.
- `TheForge/src/services/code-indexer-client.ts` — `buildContextBundle` shape, `SemanticResult` types.
- `code-indexer-service/app/routers/search.py` — per-repo scoring (cosine via DuckDB `array_cosine_distance`).
- `code-indexer-service/app/routers/context_bundle.py` — current `repo_path` resolution.

**New files (9a):**
- `code-indexer-service/eval/cross_repo/run.py`
- `code-indexer-service/eval/cross_repo/strategies/{concat,zscore,minmax,rrf}.py`
- `code-indexer-service/eval/cross_repo/labels/<query_id>.yaml`
- `code-indexer-service/eval/cross_repo/RESULTS_<date>.md`
- `code-indexer-service/eval/cross_repo/RESULTS_<date>.json`
- `TheForge/tests/fixtures/cross_repo_baseline.json`

**New files (9b):**
- `TheForge/src/services/cross-repo-rank.ts`
- `TheForge/tests/unit/services/cross-repo-rank.test.ts`
- `TheForge/tests/integration/orchestrator-cross-repo.test.ts`
- `code-indexer-service/docs/adr/0006-cross-repo-unified-ranking.md` (supersedes 0003)

**Modified (9b):**
- `TheForge/src/services/orchestrator.ts` — replace interleave with `mergeBundles` dispatcher.
- `TheForge/src/services/metrics.ts` — three new metrics from §7.
- `TheForge/src/services/types.ts` (or wherever `SymbolSnippet` lives) — three optional fields.
- `TheForge/web/src/pages/SearchPlayground.tsx` — `repoSlug` chip, strategy indicator.
- `TheForge/web/src/pages/Chat.tsx` (Sources panel) — same chips.
- `TheForge/.env.example` — `CROSS_REPO_RANK_STRATEGY=concat`.
- `infra/grafana/code-indexer-dashboard.json` — three new panels from §7.
- `code-indexer-service/CHANGELOG.md` — Phase 9 entry.

---

## End of plan
