# Retrieval methodology — lessons imported from AgentAlloy

AgentAlloy (`~/dev/claude/agentalloy`) is the skills leg of the tripod
(Skills / Code / Knowledge context). Through 2026-06 it ran a measured
retrieval-improvement campaign that took composed-context quality from
"sometimes worse than no context" to capturing 74% of oracle headroom —
every step benchmark-gated. This doc distills the transferable
methodology for code-indexer-service. It is about *how to decide*, not
about AgentAlloy's architecture. Where a practice has a concrete
artifact worth copying, the path is given.

The two services are architecturally siblings (LadybugDB graph + DuckDB
vectors + hybrid lexical/dense search + FastAPI), so most of this maps
directly: AgentAlloy's "skill/fragment" ≈ this repo's "module/symbol
(or snippet)".

## 1. The four-arm benchmark is the core instrument

Every retrieval claim is measured with paired runs over a fixed task
set, four conditions per task (`eval/run_poc.py` in agentalloy):

- **none** — bare model, no injected context. The floor.
- **composed** — your system's retrieval. The thing being measured.
- **flat (oracle)** — the *known-correct* context injected whole,
  bypassing retrieval. The ceiling.
- **external** — the incumbent practice you're competing with (for
  skills: pasted vendor docs; for code context the analog is "user
  pastes the file / grep output into chat").

The oracle arm is the highest-value idea here. It splits every failure
into "content problem" (oracle is also bad → your corpus/index lacks
the answer) vs "retrieval problem" (oracle is good, composed is bad →
the answer exists and you failed to deliver it). It also gives the
honest headline metric: **lift capture = (composed − none) / (oracle −
none)**. AgentAlloy went 45% → 74% in one day because every fix was
aimed using this decomposition.

For code-indexer: tasks = real questions ("where is X defined / who
calls Y / give me context for *add retry to the HTTP client*") against
pinned repo snapshots, graded by deterministic criteria (did the bundle
contain the defining symbol, the callers, the config constant...).
Oracle = a hand-assembled ideal bundle per task.

Mechanics that make the deltas trustworthy: fixed seeds derived from
`sha256(task:condition:run_index)`, n=5 per cell, paired per-task
deltas (never compare unpaired means), temperature pinned.

## 2. Evaluate retrieval with WEAK models

The same retrieval change measured +0.067 on an 8B-class model and
+0.006 on a 27B-class model. Strong models paper over retrieval
failures with their own knowledge; weak models amplify exactly the
signal you're tuning. Corollaries:

- Use the smallest/cheapest model as the primary iteration vehicle
  (one full leg ≈ 45 min on local hardware → same-day design loops).
- Run one strong-model leg per milestone purely as a no-regression
  check (the expected result there is a statistical tie).
- A retrieval feature that only helps weak models is still a win:
  cheap-model + good-context is the cost story.

## 3. Probes and canaries: name your failures

Two cheap instruments that caught everything the big benchmark
explained:

- **Probe queries** — a handful of named "this should work and
  doesn't" queries (ours: *"I want to build a website that is a
  blog"*). Snapshot the retrieval output (`/compose` response) BEFORE
  any change, re-run after. Yours might be *"find the auth handler"*
  against a repo where the symbol is `verify_jwt_middleware`.
- **Canary tasks** — benchmark tasks tied to a specific diagnosed
  failure mode, watched on every leg (ours: domain_4 = wrong rank-1
  amplified; domain_1 = right document, wrong slice of it).
- **Regression gate in CI** — a deterministic retrieval-metrics check
  against committed baselines (`eval/check_corpus_regression.py`,
  `eval/corpus_baselines.json`). Gold-hit-rate over a fixed query set;
  fails the build on drift.

## 4. Trace the pipeline before reaching for ML

Our biggest day-one mistake was assuming failures were ranking
failures. Tracing one task end-to-end showed the right *document*
ranked #1 while the wrong *slice* of it got selected — a completely
different fix. The discriminating experiment ("is it content or
ranking?") was: inject better content into the corpus and re-run; if
results are bit-identical, retrieval never surfaced it — ranking. Your
analog: symbol-level recall can be perfect while snippet/bundle
assembly delivers the wrong lines of the right file. Instrument both
levels separately before tuning either.

## 5. Document expansion beats a model (do it first)

The single largest measured win (+0.067 weak-model, 0→reachable on the
probe) was deterministic: the index covered only *body text*; names,
tags and descriptions never entered it. Fix: prepend a one-line
identity header to each chunk's INDEXED representation (embedding +
lexical index only — the content returned to callers stays
byte-identical), plus one synthetic "card" document per parent entity
that boosts its rank but is never emitted in results.

Code-indexer analog: a symbol's indexed text should carry more than
the code body — qualified name, docstring, module path, and a
**plain-language description** ("verifies JWT auth tokens on incoming
requests" for `verify_jwt_middleware`). The descriptions are the
bridge from how users ask to how code is named. We batch-generated 325
of them with a local LM in one afternoon (style rule that mattered:
descriptions must use the words a *task* would contain — "login",
"payments", "retry" — not restated identifiers). Propagate behind a
version bump; record the index-format mode in a metadata table so a
corpus is auditable; **bump any CI cache key that holds prebuilt
indexes when the format changes** (content hash alone won't
invalidate).

Expected result shape, so you're not disappointed: expansion makes the
right things *reachable* (rank 7–10 instead of absent); it doesn't by
itself put them in the top-k. That last hop is the re-ranker's job.

## 6. Re-ranking: pair-scoring beats generative pickers (measured)

We bake-off'd three ways to add a small model to the retrieval path,
30 minutes each, BEFORE building anything:

- **350M generative picker**: hard no-go. Emits valid JSON but selects
  everything regardless of query.
- **0.8B generative picker**: discriminates direct matches, but
  over-includes and refuses to answer "nothing helps".
- **0.6B reranker (qwen3-reranker, sibling of the embedder)**:
  clean win. Scores (query, passage) pairs via yes/no token logits —
  no JSON to parse, no chat template, deterministic at temp 0,
  ~35 ms/passage on a 3060, and "return nothing" falls out naturally
  as "no passage above threshold" (measured separation: relevant
  ≥0.51, noise ≤0.001).

Rule of thumb: **selection problems want a scorer, not a generator**.
Reserve generative models for query *enrichment* (adding vocabulary to
underspecified queries), where generation is genuinely required.

The go/no-go gauntlet format is worth copying verbatim: 3 scenarios ×
5 reps — (A) an easy direct match, (B) the ambiguous probe case, (C) a
"correct answer is nothing" case. Most candidates die on B or C, and
it costs half an hour to find out.

## 7. Layering and ship gates

- The deterministic pipeline is the **fail-open floor**: any model
  timeout/error/flag-off degrades to it byte-for-byte. Models are
  refinement layers, not load-bearing walls.
- A new moving part ships only if it beats the current baseline by a
  pre-registered margin (ours: ≥0.05 mean score on the weak-model
  leg) — and it benchmarks against the *post-previous-stage* baseline,
  not the original one. Deterministic improvements first, then make
  the model justify itself against the improved floor.
- Report token cost next to quality. "Same answer, 20–40% fewer
  tokens" is a product claim on its own.

## 8. Operational traps we paid for (so you don't)

- **Small-model thinking modes silently eat output**: several runtimes
  route tokens to a hidden `reasoning` field, returning empty
  `content` while burning the whole budget. Pin thinking OFF
  explicitly per-runtime (`think: false` / `chat_template_kwargs:
  {enable_thinking: false}`); `reasoning_effort` is silently ignored
  by some servers. This bit us three separate times in one day.
- **llama.cpp's `/v1/rerank` endpoint** does not apply
  instruction-template rerankers' prompts (scores everything ~0).
  Score via completions + the model's official template instead.
- **Schema migrations that "can't fail" do**: our ALTER statements had
  used wrong syntax for years, hidden by a blanket
  `suppress(Exception)` and masked by fresh-DB rebuilds. Suppress only
  the specific benign error string; raise the rest; run migrations at
  every entry point that reads new columns.
- **`localhost` ≠ `127.0.0.1`** under httpx (connects to `::1` only,
  no IPv4 fallback). Pin IPv4 in service URLs and health probes — a
  stdlib-urllib health check will lie to you by falling back.
- **CI merge automation must gate on checks *passing*, not
  *finishing*** — and a PR's CI runs against a merge ref snapshotted
  at run start; re-run after the base moves.

## 9. Tripod design notes — knowledge + code context in one store

(From the 2026-06-12 design discussion; the working assumption is a
shared store: LadybugDB graph as the system of record — typed nodes,
dates, status, edges — with DuckDB vectors as a derived index pointing
back by id. That shape is what AgentAlloy converged on. Refinements:)

- **Flip the vector emphasis.** Code needs vectors *least* — it ships
  a free lexical skeleton (symbol names, qualified paths, tree-sitter
  structure) and exact-match handles half its queries. Knowledge
  context is pure prose with no naming conventions: "why don't we
  retry POSTs?" has no symbol to anchor on. That is exactly the
  blog-probe problem (§5), and it took dense retrieval plus card-style
  document expansion to crack. Vectors on both legs, but expect the
  knowledge side to *depend* on them while the code side merely
  benefits. Decisions are natively card-shaped (title, date, status,
  "chose X over Y because Z") — the format proven to retrieve well;
  the body (full ADR, thread) is what the pointer dereferences after
  the card wins.
- **The decision→symbol edges are the actual product.** A code bundle
  for "add retry to the HTTP client" should graph-hop from
  `httpClient.send` to the decision node saying "we deliberately don't
  retry non-idempotent calls (2025-11, incident #42)" — context no
  embedding similarity surfaces, because the task never mentions it.
  Retrieval by *structure* is the thing neither leg does alone.
- **Steal the supersession chain wholesale** (`deprecated` /
  `superseded_by` columns + active-only retrieval filters). Decisions
  get reversed; stale-decision injection is the knowledge leg's
  equivalent of AgentAlloy's intake-boilerplate bug — the failure mode
  that made composed context *worse than nothing*.
- **One embedder family across all legs** (currently
  qwen3-embedding, 1024-dim) so similarity spaces stay comparable if
  results are ever fused cross-store — with RRF doing the fusion,
  since raw cosine scores across heterogeneous corpora aren't
  calibrated against each other.
- **Define the knowledge leg's oracle arm on day one**: "the bundle
  that would have prevented the wrong implementation." For decisions,
  the *none* arm fails silently — the model confidently re-makes the
  mistake the ADR documented — so without the oracle ceiling the
  headroom is invisible.

## 10. The loop, summarized

```
benchmark (4 arms, weak model, paired deltas)
  → trace one named failure end-to-end
  → cheapest deterministic fix first (usually: index more of what you know)
  → re-measure same day (probes → leg → regression gate)
  → only then audition a model, via a 30-min capability gauntlet
  → ship behind a pre-registered gate, fail-open to the deterministic floor
```

Deep-dive artifacts in `~/dev/claude/agentalloy`: `eval/run_poc.py`
(harness), `eval/domain_tasks.py` (task+grader format),
`eval/check_corpus_regression.py` (gate), `docs/lm-assist-design.md`
(the staged design with measurement plan), `src/agentalloy/storage/
card_index.py` (document expansion), `eval/judge.py` (LLM-judge
validation of heuristic graders, Batches API).

## 11. Cross-tool principle: don't make the retriever manufacture intent

Both tools independently hit the same wall and drew the same line: **a
retriever must not invent intent the query never expressed.**

- **Code-context (this repo), 2026-06-13.** The `dsg-cgr-001` benchmark
  task ("re-ingest only changed files") has a *storage* facet that wants
  `vector_store`/`duckdb`. Two independent model architectures — the
  e5-base-v2 bi-encoder and the qwen3-reranker-0.6b cross-encoder — both
  score that storage code as irrelevant (~0.03) to the literal query. The
  inference "re-ingesting implies re-embedding implies touching the vector
  store" is real domain knowledge, but it isn't *in the query*. The only
  retrieval mechanism that closes the facet (query decomposition that
  synthesizes a storage sub-query) does so by manufacturing the missing
  intent — gaming the benchmark, not improving retrieval. We **accepted it
  as benchmark debt** rather than build that. See
  `docs/reranker-bundle-tiebreak-spike.md`.
- **AgentAlloy.** Reached the same conclusion from the SDD-workflow /
  blog-probe finding: an under-specified knowledge query ("why don't we
  retry POSTs?") has no anchor for the retriever to guess from, and
  guessing produced *worse-than-nothing* composed context.

The shared rule: **under-specified queries get resolved at the right
layer, never by the retriever guessing.** On the code side that layer is an
explicit facet / a clarification step; on the AgentAlloy side it is
workflow interrogation. Generation is reserved for query *enrichment* with
vocabulary the user implied (§6's "login/payments/retry"), not for
inventing a concern the user never raised. A corollary from the same spike:
relevance reranking belongs at the **search top-k** stage, where every
candidate is already plausibly on-topic — **not** at bundle truncation,
where the candidates are graph neighbours and structure, not surface
relevance, is the load-bearing signal.
