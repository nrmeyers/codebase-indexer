# Qwen3-embedding migration — spec & design (fresh-session handoff)

**Audience:** a fresh agent session working in an **isolated clone** of this
repo. **Goal:** swap the embedder from `e5-base-v2` (768-dim) to
`qwen3-embedding:0.6b` (1024-dim) and measure whether it beats e5 on the
design-retrieval benchmark — without touching the canonical repo or its
baseline index.

> Status when written (2026-06-13): NOT started. This is the plan. The
> `main` repo is clean at lift **0.9778**; do the work in a clone so it
> stays that way.

---

## 1. Why this is isolated (clone, not a branch)

A git branch isolates *code*. This migration's real blast radius is **data**:
the per-repo `.cgr/repos/{slug}.duck` vector stores switch from `FLOAT[768]`
to `FLOAT[1024]` and become incompatible with the e5 baseline (and with
prod's jina/sagemaker 768 schema). Those files are gitignored, so a branch
does **not** protect them.

Use a **separate clone** (or a `git worktree` — a separate working dir gives
a separate `.cgr/`). Benefits: the e5 0.9778 baseline index stays pristine,
`main` stays shippable, and the whole experiment is reversible by deleting
the clone. The clone re-indexes its own repos from scratch (see §6).

---

## 2. Strategic rationale & success criterion

Per the project memory: the code-indexer and the **agentalloy** tool are
intended to eventually ship together sharing **one embedding engine**. qwen3
is the convergence candidate. **Therefore the success criterion is NO
REGRESSION, not a big lift** — if qwen3 matches e5 (within noise) on the
benchmark, that is a *win* because it enables engine convergence. A clear
regression (like jina, §4) means don't migrate.

---

## 3. The baseline to beat (the gate)

Measured on `main` with e5-base-v2, deterministic (verified run-to-run):

| metric | value | how to measure |
|---|---|---|
| composed arms mean lift | **0.9778** (14/15 design tasks at 1.00) | `uv run python scripts/run_arms.py` |
| first-stage recall @10 / @25 / @50 | **0.728 / 0.828 / 0.872** | `uv run python scripts/run_recall.py` |
| probe coverage | **6/6** at 1.00 | `uv run python scripts/run_probes.py --check` |

`run_recall.py` (first-stage facet recall over raw `/search/semantic`)
isolates the embedder from the bundle machinery — **it is the primary metric
for this eval**; composed arms is the no-regression gate. Both need the
service running and CPU-pinned (`CUDA_VISIBLE_DEVICES=""`).

**Decision gate:** adopt qwen3 iff recall ≥ e5 (any consistent lift) AND
composed arms ≥ 0.9778 (no task drops) AND probes stay 6/6. Otherwise record
the finding and keep e5.

---

## 4. Priors from the embedder eval (read before starting)

- **jina-code-v2 (TEI, 768) REGRESSED**: recall @25 0.828→0.761, composed
  0.9778→0.9389. Embeddings verified sane — not a bug. Lesson: jina-code-v2
  is *general code-embedding*, but our task is **NL-query → code search**
  (queries are design-intent sentences, docs are code). General code
  embedding is the wrong sub-task.
- **st-codesearch-distilroberta (768) HELPED cgr** (cgr-004/005 recall@25
  0.67→1.00). It was trained on NL-docstring↔code, i.e. **code search** —
  the right sub-task. (Too old to ship; it's a proxy.)
- **e5-base-v2 is a strong NL→code baseline** and hard to beat here.
- **qwen3-embedding is instruction-tuned and strong on NL+code retrieval**
  — the right *class* (code-search / instruction retriever), so a better
  prior than jina. Still a coin-flip vs e5. Consider prepending the model's
  retrieval **instruction/prompt** to queries (qwen3-embedding supports an
  instruction; the current pipeline embeds queries prefix-free, which may
  under-serve qwen — test both).

---

## 5. Serving qwen3 (no backend exists yet)

`app/embedders/__init__.py` has `VALID_BACKENDS = (local, sagemaker, tei,
openai)` and `get_embedder()` dispatches on `EMBEDDER_BACKEND`. There is **no
ollama backend**. Two options:

- **(Recommended) New `app/embedders/ollama.py`** hitting the Ollama embed
  API. `qwen3-embedding:0.6b` is already pulled and served on
  `http://127.0.0.1:11434` (`POST /api/embed`, `{"model":..., "input":[...]}`
  → `{"embeddings":[[...]]}`). Mirror `tei.py`'s structure (async httpx,
  batch, `dim` attribute, availability probe). Register it in the factory +
  `VALID_BACKENDS`. Add `OLLAMA_URL` / `OLLAMA_EMBED_MODEL` to `config.py`.
  Cheapest path — the model is already running on the 3060.
- (Alt) Serve qwen3-embedding via TEI if TEI's version supports the Qwen3
  embedding architecture (jina worked via TEI/ONNX; verify Qwen3 support
  before committing to this).

Sanity-check the backend first (the jina lesson): embed a related
query/code pair and an unrelated pair, confirm cosine separates (≳0.5 vs
≈0). Only then re-index.

---

## 6. The 1024-dim migration — code touch-points

`EMBEDDING_DIM = 768` is **duplicated in four files** (migration trap — they
are independent definitions, not one import):

- `app/services/embedder.py:73`
- `app/services/neighbors.py:46`
- `app/services/centroid.py:52`
- `app/embedders/base.py:33`

Change all four to 1024 (or, better, centralise to one definition imported
by the rest, to prevent future drift). Then:

- **DuckDB schema**: the `embeddings.embedding` column is `FLOAT[768]`,
  created in the vendored engine `codebase_rag/storage/vector_store.py`
  (also `vector_store_arrow.py`). Change the DDL to `FLOAT[1024]`. The
  `embedding_v2` column in `app/services/embedder.py:326` already uses
  `FLOAT[{EMBEDDING_DIM}]`, so it follows the constant automatically.
- **`app/services/neighbors.py`**: `_coerce…` validates a "768-dim list"
  with a hardcoded check — update to `EMBEDDING_DIM` or it will reject 1024
  vectors.
- **`app/embedders/base.py`** asserts `len(vectors[0]) == 768` in its
  contract docs/checks — update.
- Grep the whole tree for `768` and `FLOAT[` after editing; there may be
  stragglers (tests, models.py docstrings, centroid).

---

## 7. Gotchas learned this session (each cost real time — heed them)

1. **uvicorn has NO `--reload`** here. Code edits do NOT take effect until
   you restart the service. Measuring across a half-applied edit produced a
   confounded result this session. Always restart before measuring.
2. **Content-hash skips re-embed on a model swap.** The incremental
   `content_hash` fingerprints the embed *text*, not the model. Swapping the
   embedder leaves the text unchanged → the embed pass SKIPS everything and
   keeps the old vectors. You MUST wipe the `embeddings` table (or do a full
   re-index that recreates it) to force re-embedding with the new model. In
   a fresh clone you re-index from scratch, so this is moot — but if you
   re-measure e5-vs-qwen in place, remember it.
3. **Card sidecars re-emit.** If `.cgr/repos/{slug}.cards.json` exists,
   `embed_driver` phase 2b emits `{qname}::Symbol::card` chunks. For a clean
   embedder comparison, ensure no sidecars are present (a fresh clone won't
   have them). They are an *unmerged, separate* experiment (see
   `docs/symbol-cards-findings.md`).
4. **Back up `.duck` before wiping; restore on abort.** State-juggling is
   error-prone; keep `*.duck.bak`.
5. **GPU placement.** The 3090 (GPU 1) is RESERVED — never use it. The 3060
   (GPU 0) runs the qwen3-reranker + Ollama (incl. qwen3-embedding). The
   indexer service runs CPU-pinned (`CUDA_VISIBLE_DEVICES=""`). qwen3
   embedding inference happens inside Ollama on the 3060; the service just
   calls it over HTTP.
6. The embed pass may re-embed ~100 extra non-card chunks (hash-format
   drift); `embedded_count` slightly exceeding symbol count is expected and
   deterministic.

---

## 8. Procedure (in the clone)

1. Clone repo; `uv sync`. Point at separate data: default `.cgr/` in the
   clone's working dir is already separate.
2. Write the ollama backend (§5); sanity-check embeddings.
3. Apply the 1024 changes (§6); grep-verify no stray 768.
4. Full re-index the three benchmark repos (TheForge @ SHA 19f45130,
   code-graph-rag @ 2224008, code-indexer-service) so graph + 1024 vectors
   are built fresh. Keep checkouts at the oracle SHAs (TheForge/cgr matter;
   see `eval/oracle/*.json` and the arms manifest).
5. Start service `EMBEDDER_BACKEND=ollama CUDA_VISIBLE_DEVICES="" uvicorn …`.
6. Measure: `run_recall.py`, `run_arms.py`, `run_probes.py --check`.
7. Compare to §3 baselines against the §3 gate. Record verdict in a findings
   doc (mirror `docs/symbol-cards-findings.md`). Try query-instruction
   prefix on/off (§4).
8. If adopt: the migration is real (schema + re-index), coordinate the prod
   path (prod runs sagemaker/jina at 768 — a 1024 cutover is a separate
   prod decision). If reject: delete the clone; `main` is untouched.

---

## 9. Reference

- Embedder architecture: `app/embedders/` (`__init__.py` factory,
  `base.py`, `local.py`, `tei.py`, `sagemaker.py`, `openai.py`),
  `docs/EMBEDDERS.md`.
- Eval harnesses: `scripts/run_arms.py`, `scripts/run_recall.py`,
  `scripts/run_probes.py`, `scripts/grade_queries.py`, `eval/oracle/`,
  `scripts/queries.json`, committed baselines in `eval/probes/snapshots/`.
- Methodology + priors: `docs/embedder-eval-plan.md`,
  `docs/retrieval-methodology-from-agentalloy.md` (§11 cross-tool note),
  `docs/symbol-cards-findings.md`, `docs/reranker-bundle-tiebreak-spike.md`.
