# Symbol cards (§5) — findings

**Status (2026-06-13):** built end-to-end, measured cleanly, **not shipped
in the folded integration** (net composed regression). Generator + indexing
wiring + recall harness committed on `feat/symbol-cards`; `{slug}.cards.json`
sidecars kept on disk. `main` stays clean at lift 0.9778. The displacement
that sinks the folded integration has a clear fix (additive seed leg) —
that is the live follow-up.

## What was built

- **`scripts/generate_symbol_cards.py`** — offline batch generation of a
  one-line **task-vocabulary** description per Function/Method into a
  `{slug}.cards.json` sidecar. Local model **qwen3.5:0.8b** via Ollama.
  - Model A/B: qwen3.5:0.8b vs **LFM2.5-350M**. qwen wins decisively — it
    *translates* identifiers to problem-domain words (`createActivityFeed`
    → "GitHub PR opens/merges/closes/CI"); LFM mostly *restates* the
    identifier (`requireRole` → "role_required") and hallucinated, adding
    little over the name the embedder already has. (Consistent with the
    methodology's §6 350M no-go.)
  - Generated: cis 1647, TheForge 3708, cgr 1166 cards.
- **`embed_driver` phase 2b** — emits a `{qname}::Symbol::card` chunk
  (embedded text = the description) into DuckDB; incremental, no sidecar →
  no cards.
- **search/bundle fold** — `_semantic_search_impl` maps a card hit back to
  its PARENT symbol and de-dups, so a card surfaces the real symbol and the
  card qname is never emitted.
- **`scripts/run_recall.py`** — first-stage facet recall@{10,25,50} over
  raw `/search/semantic`, isolating the bi-encoder from the bundle.

## Clean measurement (correct bundle code, freshly embedded)

First-stage recall — **cards help, small and consistent:**

| cut-off | no cards | with cards |
|---|---|---|
| @10 | 0.728 | **0.750** |
| @25 | 0.828 | **0.850** |
| @50 | 0.872 | **0.889** |

Composed arms (gate = 0.9778, probes 6/6):

| | mean lift | dsg-tf-003 | dsg-tf-005 | probe-tf-auth |
|---|---|---|---|---|
| no cards | **0.9778** | 0.75 | 1.00 | 1.00 |
| with cards | 0.9611 | **1.00** ✅ | **0.75** ❌ | **0.67** ❌ |

**The real result:** cards genuinely surface under-retrieved symbols —
`dsg-tf-003` rose 0.75 → 1.00 (cards closed the auth facet that the
cross-encoder reranker, caller expansion, lexical leg, and breadth
reduction all could not). But folded into the **fixed** semantic top-k, a
card-surfaced symbol can only enter by **displacing** another — and that
cost shows up as `dsg-tf-005` 1.00 → 0.75 and `probe-tf-auth` losing
`session_jwt`. Net composed −0.0167. Same shape as the recall numbers: +at
the margin, but a wash-to-negative once a slot is taken from a good symbol.

## Root cause + the fix (follow-up)

Cards are folded into the main semantic seed list, so they compete *in* the
top-k rather than *extending* it. The fix is an **additive** integration:
keep cards OUT of the main semantic path (no displacement) and add a
**bounded card-seed leg** with a small guaranteed quota — exactly the shape
of the existing lexical seed leg (`_lexical_seed_hits` in
`context_bundle.py`). That would keep the `dsg-tf-003` win without evicting
`dsg-tf-005`/`session_jwt`. Requires a card-only retrieval path
(`WHERE symbol_type='SymbolCard'`) so the leg can rank cards independently.
cgr-001's storage facet stays unreachable either way (two-model
over-specification — cards don't manufacture absent intent, §11).

## Process note (why the first read was wrong)

The initial "with cards" run showed −0.033 and two broken probes, but it was
**confounded**: a bundle-ordering bug (re-sorting seeds by raw cosine
instead of preserving fused order — fixed in `df9d1fb`) was active, and the
card rows had been deleted before the recall read. The numbers above are
from a clean re-run on the fixed code. Lesson: restart the service to load
code changes (uvicorn has no `--reload` here) and never measure across a
half-applied edit.
