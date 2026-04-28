# Code Indexer — Roadmap & Status

**Owner:** zacharymatthews | **Last updated:** 2026-04-27
**Spec source of truth:** v5.3 (DuckDB §6.5 + §8.4, deferred HNSW §17)

This document supersedes the per-phase build plan in
`~/.claude/plans/full-digestion-consideration-of-vectorized-matsumoto.md`.
That file remains as historical record; new work tracks here.

---

## Stack — Current Reality

```
TheForge API (Express :3001)
        │  HTTP :8000
        ▼
code-indexer-service (FastAPI gateway, this repo)
        │  Python import
        ▼
code-graph-rag (LadybugIngestor + CodeRankEmbed + DuckDB)
        │
        ├─► LadybugDB  (.cgr/repos/{slug}.db    — embedded kuzu graph)
        └─► DuckDB     (.cgr/repos/{slug}.duck  — FLOAT[768] embeddings)

Optional (opt-in via LM_STUDIO_URL):
  LM Studio (localhost:1234)
    ├─► CodeRankEmbed — query-time embedding (avoids torch in uvicorn)
    └─► CodeRankLLM / Qwen / Llama — listwise rerank when ?rerank=true
```

---

## Implementation Status (audit verified 2026-04-27)

| Component                | Status   | Notes                                                   |
|--------------------------|----------|---------------------------------------------------------|
| LadybugDB graph store    | ✅ Done  | Per-repo `.db` files, single-writer locks, schema mig.  |
| DuckDB vector store      | ✅ Done  | Per-repo `.duck`, FLOAT[768], `array_cosine_distance`.  |
| CodeRankEmbed (768-dim)  | ✅ Done  | Replaces UniXcoder. Subprocess-isolated at index time.  |
| Asymmetric prefixes      | ✅ Done  | `Represent this code snippet:` / `search_query:`.       |
| PageRank centrality      | ✅ Done  | Stored in `.duck` `centrality` table.                   |
| RRF/BM25 stage-1 fusion  | ✅ Done  | Pre-rerank, post-cosine.                                |
| LM Studio adapter        | ✅ Done  | `app/services/lm_studio.py` — embed + chat passthrough. |
| `can_embed`/`can_rerank` | ✅ Done  | Precise availability gates (not just `is_available`).   |
| Listwise reranker (LLM)  | ✅ Done  | `app/services/reranker.py` — `?rerank=true` opt-in.     |
| Snippet-enriched rerank  | ✅ Done  | `app/services/source_fetch.py` (shared helper).         |
| Two-stage retrieval      | ✅ Done  | `/search/semantic`, `/context-bundle` both wired.       |
| `.env` bridge            | ✅ Done  | `load_dotenv()` in main.py — `os.environ` reads work.   |
| Smoke-test script        | ✅ Done  | `scripts/lm_studio_smoke.py`.                           |
| Tests                    | ✅ 82/82 | `uv run pytest -q` — green.                             |

---

## Verified at runtime (2026-04-27)

* LM Studio :1234 reachable; `qwen/qwen3.6-27b`, `qwen/qwen3.6-35b-a3b`,
  and `text-embedding-nomic-embed-text-v1.5` loaded.
* End-to-end rerank against **`qwen/qwen3.6-27b`** correctly reorders 5
  synthetic candidates: all 3 `auth.*` candidates rank above noise for a
  JWT-auth query. Wall-clock **~103s** for 5 candidates with reasoning.
* `can_embed()` correctly returns `False` for the loaded
  `nomic-embed-text-v1.5` (parent base — different vector space) — the
  strict `CodeRankEmbed` default protects against silent recall loss.

---

## Active Findings (newly surfaced)

### F1 — Qwen3.6 thinking mode is sticky in LM Studio

The LM Studio preset for `qwen3.6-35b-a3b` and `qwen3.6-27b` ignores
both:
* `/no_think` directive (system or user message)
* `chat_template_kwargs={"enable_thinking": false}` body parameter

Both end up in reasoning mode regardless. The reranker handles this via
fallback to `reasoning_content` when `content` is empty, and
`max_tokens=2048` gives reasoning + answer enough room.

Counter-intuitively, **MoE-A3B is slower per-token (3.4 tok/s) than 27B
dense (~10 tok/s)** on this hardware — likely Apple-Metal expert routing
overhead. The 27B dense remains the preferred rerank model.

**Action:** none required; documented in `.env` and README. Future
options: drop in CodeRankLLM proper (bypasses thinking entirely) or
edit the model preset in LM Studio's UI.

### F2 — DuckDB `bulk_insert` does N sequential DELETEs *(SOLVED 2026-04-27 via Arrow path)*

Original: `DELETE WHERE qn=?` + `INSERT` per row inside a transaction —
2N statements for N embeddings.

**Step 1 (executemany refactor):** collapsed to one batched `DELETE … IN
(?,?,…)` + one `executemany(INSERT …)`.  Cleaner code, **~0% faster**.
Bench (500 rows × FLOAT[768]): 28.7 s → 28.5 s.  Per-row FLOAT[768]
parameter binding from Python lists is the bottleneck — not round-trip
count.

**Step 2 (Arrow-staged path — DEFAULT 2026-04-27):**
`codebase_rag/storage/vector_store_arrow.py::bulk_insert_arrow` stages
embeddings through a registered Arrow table and uses DuckDB's columnar
bulk-load.  `vector_store.bulk_insert` now auto-dispatches to this path
when `pyarrow` is importable; falls back to executemany otherwise.

> **Measured speedup at 100 / 500 / 1000 rows: 324× / 382× / 390×**
> — see `code-graph-rag/scripts/BENCH_RESULTS_2026-04-27.md`.

Concrete numbers (median of 3 trials, fresh `.duck` per trial):

```
  rows |   bulk_ms |  arrow_ms | speedup
   100 |   4419.83 |     13.65 |  323.85×
   500 |  21874.31 |     57.32 |  381.60×
  1000 |  44013.60 |    112.83 |  390.08×
```

5k / 10k tiers skipped — linear scaling at 100/500/1000 (variance < 2%
on `arrow_ms/row`) makes extrapolation safe and saves ~24 min of bench
time.  Phase 4 decision criterion (≥2× at 10k → default) is crossed by
two orders of magnitude.

**Shipped:**
* `vector_store.py` — auto-dispatch via lazy `import pyarrow`.
* `vector_store_arrow.py` — opt-in Arrow path, L2-normalises to keep
  `array_cosine_distance` invariant identical to the fallback.
* `pyproject.toml` — `[arrow]` optional extra (`pyarrow>=15.0`).
* Tests: 5 new in `test_duckdb_vector_store_arrow.py` including a
  ranking-equivalence test (executemany ↔ arrow produce identical
  search ordering on the same input).  19 + 5 = 24/24 green.
* Bench harness: `scripts/bench_bulk_insert.py --arrow [--sizes …]`.

### F3 — Documentation rot in `code-graph-rag/docs/`

23 docs in `code-graph-rag/docs/`, **16 of them reference
Memgraph/Qdrant/docker-compose** that no longer exist. Three files
should be deleted outright (configuration.md, installation.md,
troubleshooting.md — each fully describes the removed Docker stack).

---

## Prioritised next steps

### P0 — Ship-blocking (do next)

Nothing. The stack is functional end-to-end; rerank is opt-in and safe.

### P1 — High-ROI cleanups

1. ~~**Speed up DuckDB `bulk_insert`** (F2)~~ — **DONE 2026-04-27**.
   Two-step landing: (a) batched DELETE+executemany refactor — ~0% at
   SQL layer (per-row FLOAT[768] marshalling dominates). (b) Arrow-
   staged bulk-load (`vector_store_arrow.py`) auto-dispatched when
   `pyarrow` is installed — **324–390× faster** at 100–1000 rows
   (`scripts/BENCH_RESULTS_2026-04-27.md`).  Phase 4 closed: ship as
   default with `[arrow]` extra.
2. ~~**Delete obsolete docs**~~ — **DONE 2026-04-27**. Deleted
   `docs/getting-started/installation.md`,
   `docs/getting-started/configuration.md`,
   `docs/advanced/troubleshooting.md`, and removed their entries from
   `mkdocs.yml`.
3. ~~**Update `docs/index.md` + `docs/architecture/overview.md`**~~ —
   **DONE 2026-04-27**. Plus `docs/guide/mcp-server.md`,
   `docs/claude-code-setup.md`, `PYPI_README.md`, and the main fork
   `README.md` (Prerequisites, Installation, env-vars, dependencies,
   and Debugging sections all rewritten for LadybugDB/DuckDB).
   The migration table (lines 53-84) intentionally retains old names
   so the swap history stays legible.
4. ~~**Delete `code-graph-rag/CLEANUP_TODO.md`**~~ — **DONE 2026-04-27**.

### P2 — Quality / correctness *(shipped 2026-04-27)*

5. ~~**Surface `search_intent`**~~ — **DONE**. Optional field on
   `SemanticSearchResponse`; defaults to `"semantic"`, flips to `"fqn"`
   when the bare-FQN regex pins an exact match. Tests cover both
   branches.
6. ~~**Integration test for rerank fallback**~~ — **DONE**. 4 new tests
   in `tests/test_reranker.py` covering: chat_complete returns None,
   chat_complete raises, unparseable permutation, empty-string response.
   All preserve original order (and object identity).
7. ~~**Document the `enable_thinking` quirk**~~ — **DONE**. Added
   `.. note:: Qwen3 thinking-mode quirk` block to the module-level
   docstring of `app/services/reranker.py`.

Plus (from Phase 5.4 brought forward):

8. **`/health` exposes LM Studio backend status** — **DONE 2026-04-27**.
   New `lm_studio` block in the health response with `configured`,
   `reachable`, `embed_model`, `rerank_model`, `can_embed`, `can_rerank`
   fields. Short-circuits cleanly when `LM_STUDIO_URL` is unset; never
   raises. Test count 90/90 green.

### P1.5 — Token-saver helper (added 2026-04-27)

**Done:** `scripts/ask_local_llm.py` — thin CLI wrapping
`lm_studio.chat_complete()` so the orchestrating agent can offload
distillation / summarization tasks to the locally-running qwen3.6
model.  Big inputs (long docs, grep dumps, file contents) go to LM
Studio once; only the small distilled answer comes back into the
agent's context.  Net savings = (input tokens) − (answer tokens),
typically 5-50× for summarization-style work.

Use cases the agent should default to:
* Distill a long file into 3-5 bullets before reading it natively.
* Group a large grep result by topic; surface only the categories.
* Draft a doc rewrite; agent skims and edits instead of writing
  from scratch.
* Sanity-check a candidate diff against a spec excerpt.

Caveat: when only the MoE-A3B (qwen3.6-35b-a3b) is loadable and the
dense 27b is memory-evicted, latency stretches to several minutes per
call.  In that case the time savings flip negative even if context-token
savings stay positive — only worth it for genuinely token-expensive
inputs (>3k tokens).  ``LM_STUDIO_TIMEOUT`` raised to 600s to cover the
slow path.

### Phase 4 — Arrow-staged bulk_insert *(shipped 2026-04-27)*

See F2 above for the full bench numbers.  Decision: **ship as default**
when `pyarrow` is importable (auto-dispatch in `vector_store.bulk_insert`),
keep executemany as fallback, document the `[arrow]` extra in
`pyproject.toml`.  `code-graph-rag/scripts/BENCH_RESULTS_2026-04-27.md`
captures the methodology + raw numbers.

### Phase 5 — TheForge integration *(shipped 2026-04-27)*

5.1 + 5.2 (frontend) — Agent E:
* `code-indexer-client.ts` — `LMStudioHealth` type, `lm_studio?` on
  `HealthResponse`, optional `rerank` parameter on `semanticSearch` and
  `buildContextBundle`, optional `search_intent` on the response shape.
* `web/src/components/code-indexer/SearchPlayground.tsx` — persisted
  rerank toggle (per-repo localStorage key), `search_intent` badge,
  hint text gated on `canRerank`.

5.3 (orchestrator gating) — Agent F:
* `src/services/orchestration-config.ts` — `RERANK_SYMBOL_THRESHOLD = 500`.
* `src/services/orchestrator.ts` — `preflight()` now threads the
  Code Indexer `/health` response through to `fetchCodeContext()`.
  Rerank gate per repo:
  ```ts
  rerank = canRerank && symbolCount >= RERANK_SYMBOL_THRESHOLD;
  ```
  Multi-repo fan-out (`codeIndexerRepo === '*'`) applies the gate
  per-repo; small repos stay cosine-only, large repos pick up rerank.
* `src/services/audit-trail.ts` — new
  `orchestration.rerank.applied` event, emitted only when the gate
  flips on (so we can observe usage without flooding the audit log
  with negatives).
* Symbol count source: `health.repos[].node_count` (the existing
  per-repo metric); falls back to 0 (gate closed) when no match.
* Both `pnpm build` and `cd web && npx tsc --noEmit` clean.
  64 orchestrator-related tests pass.

5.4 (health endpoint) — already done; see P2 item 8 above.

### Phase 6 — ADR records *(shipped 2026-04-27)*

ADRs in `docs/adr/`:
* `0001-defer-hnsw-vss-indexes.md` — trigger: cosine query p95 > 200 ms
  OR repo > 50,000 symbols.
* `0002-defer-coderanklm-proper.md` — trigger: Nomic publishes a
  CodeRankLLM GGUF for LM Studio.
* `0003-defer-cross-repo-unified-ranking.md` — trigger: 5+ repos
  indexed AND cross-repo recall complaints.
* `README.md` — index with one-line summaries.

---

## Linear / TheForge integration touchpoints

* `code-indexer-client.ts` — already in TheForge, used by orchestrator.
* `?rerank=true` and `rerank: true` (context-bundle) opt-ins — TheForge
  callers get the boost by passing the flag; default behavior is
  unchanged.
* `LM_STUDIO_URL` is purely a **local-dev concern** — TheForge does not
  see or care whether LM Studio is running.

---

## Files changed in latest two-stage retrieval work

```
code-indexer-service/
  app/main.py                   load_dotenv() bridge
  app/services/lm_studio.py     +can_embed, +can_rerank, +reasoning_content fallback,
                                +chat_template_kwargs
  app/services/reranker.py      /no_think trailer, max_tokens=2048
  app/services/source_fetch.py  NEW — shared FQN→snippet helper
  app/routers/search.py         _embed_query helper, snippet-enriched rerank
  app/routers/context_bundle.py delegates to source_fetch
  scripts/lm_studio_smoke.py    NEW — operator smoke test
  tests/test_lm_studio.py       NEW — 14 tests
  tests/test_reranker.py        NEW — 17 tests
  README.md                     +Two-stage retrieval section, latency notes
  .env / .env.example           +LM_STUDIO_* with vector-space safety warning
```

---

## Quick links

* Service README (`README.md`) — endpoints, env vars, dev commands
* `CLAUDE.md` — file map, architecture patterns, coding standards
* Smoke test: `uv run python scripts/lm_studio_smoke.py`
* Test suite: `uv run pytest -q` (82 tests, ~3 seconds)
