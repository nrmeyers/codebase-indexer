# Phase 1.5 — PageRank Persistence Audit

**Date:** 2026-05-08
**Roadmap reference:** TheForge `docs/OPTIMIZATION_ROADMAP.md` §1.5
**Branch:** `feat/pagerank-persistence`

## Question 1 — Is PageRank computed? Where?

**Yes — partially.** Plan J already shipped a `compute_pagerank(repo_db_path)`
function in the sibling `code-graph-rag` package at
`codebase_rag/storage/centrality.py`. It:

- Opens the per-repo LadybugDB
- Pulls all `Function` and `Method` nodes via `MATCH … UNION ALL`
- Pulls all `CALLS` edges across the four label permutations
  (`Function→Function`, `Function→Method`, `Method→Function`, `Method→Method`)
  — required because LadybugDB's parser does not accept `(n:A OR n:B)` label
  disjunction
- Builds a `networkx.DiGraph`, runs `nx.pagerank(g, alpha=0.85)`
- Normalises by dividing by the max score (NOT min-max — the floor is the raw
  smallest score, not zero)

The function is invoked at the end of every full ingest pass in
`app/routers/index.py` (lines 742-766) inside a best-effort `try/except` —
PageRank failures log `pagerank.failed` and continue, so the index is still
useful without centrality.

## Question 2 — Is it persisted? In what column / where?

**Yes.** Per-repo `.duck` (DuckDB) file holds a `centrality` table:

```
centrality(qualified_name VARCHAR PRIMARY KEY, pagerank REAL)
```

Written via `codebase_rag.storage.vector_store.write_centrality()` after a
`clear_centrality()` purge so stale qualified names from previous runs do not
linger.

## Question 3 — Is it served via any endpoint?

**Yes.** `GET /search/centrality?limit=N&repo=...` in
`app/routers/search.py` (lines 803-870) returns the top-N rows ordered by
`pagerank DESC`, enriched with `(file_path, line_range)` looked up from the
LadybugDB `Module -[:DEFINES]-> Function/Method` graph.

A second consumer reads centrality at search time: the semantic search
pipeline reads `pr_scores` via `read_centrality(...)` and applies
`final = 0.7 * cosine + 0.3 * normalised_pagerank` fusion before optional
BM25 RRF and reranker passes (`app/routers/search.py` lines 582-612).

## Question 4 — Gaps to fill

The Plan J work covered the substrate but does not satisfy the explicit
deliverables in the brief:

| Brief deliverable | Plan J state | Gap |
|---|---|---|
| `app/services/pagerank.py` (new, in this repo) | Lives in sibling `code-graph-rag` package | Add a thin local module so this service is testable in isolation and the algorithm is pinned to this repo's CI. |
| `compute_pagerank(conn)` separable from DB I/O | Single function does both | Split into a pure `compute_pagerank(edges, nodes)` core + `compute_pagerank_for_repo(repo_db_path)` wrapper. |
| `normalize_pagerank(scores)` distinct, min-max | Embedded; uses divide-by-max not min-max | Provide stand-alone `normalize_pagerank()` that does true min-max so floor=0, ceiling=1. |
| `networkx>=3.0` declared as direct dep | Already present (`networkx>=3.2`) | None — already satisfied. |
| Persistence schema | Exists (`centrality` table in `.duck`) | None — reuse existing table. No schema change. |
| Compute trigger at end of ingest | Wired (Plan J) | None — leave existing call site untouched. |
| `GET /repos/{repo}/centrality?limit=20` | Existing route is `GET /search/centrality` | Add the per-repo path under `/repos/...` with a simpler response shape `{ symbols: [{ qname, centrality }] }` matching the brief. |
| Tests in this repo | Search-side tests mock `read_centrality` | Add 4 surgical tests against the new module + endpoint. |

## Decision

**Do not duplicate the compute pipeline.** Add a thin shim in
`app/services/pagerank.py` that:

1. Exposes a pure `compute_pagerank(edges, nodes=None)` function — easy to
   unit-test against a 10-node fixture graph, no LadybugDB required.
2. Exposes a separate `normalize_pagerank(scores)` doing true min-max.
3. Provides `compute_pagerank_for_repo(repo_db_path)` that reuses the
   sibling-package implementation when available, but the pure core is the
   testable contract.

Add the brief's preferred endpoint shape `GET /repos/{repo}/centrality` in
`app/routers/repos.py` reading from the same `centrality` table, so the
existing `/search/centrality` endpoint (used by the FE today) keeps its
location-enriched contract while TheForge gets the simpler debug-friendly
shape it asked for.

The TheForge `mergeAndRank` integration is intentionally **deferred** —
PR #224 (Phase 1.1 Tantivy) just changed `mergeAndRank`'s signature and we
want a clean follow-up diff after it lands.
