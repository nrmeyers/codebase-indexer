# ADR-0003: Defer Cross-Repo Unified Ranking

**Status:** Deferred — not implemented as of 2026-04-27
**Trigger:** 5+ repos indexed simultaneously AND orchestrator surfaces measurable cross-repo recall complaints.

## Context

Each indexed repository has its own DuckDB file (`.duck`) with a dedicated vector
space (FLOAT[768] embeddings from CodeRankEmbed). TheForge orchestrator currently
fans out semantic search per-repo and merges results by concatenation, allowing
each repo's top-k to bubble up independently.

This per-repo isolation is correct by design: different codebases have different
symbol distributions, naming conventions, and embedding clusters. Mixing embeddings
across vector spaces is undefined.

Cross-repo unified ranking becomes relevant only if users query multiple repos
simultaneously and report recall loss (e.g., "the best match is in repo B but
repo A's top-10 dominates the result set").

## Decision

Maintain per-repo vector isolation. The orchestrator merges top-k results via
concatenation and optional score-based ranking. No unified vector index is built
until both: (a) 5+ repos are live and indexed, and (b) measurable cross-repo
recall complaints arise.

## Consequences

**What stays simple:** Each repo's embedding index is independent; no schema
changes to `.duck` files; code-indexer-service API remains repo-centric
(`/search/semantic?repo={slug}`).

**What we accept as cost:** Top-k merging across repos depends on score
normalization (cosine distance scaled 0–1 per repo) or other heuristics. Users
see results from all repos but no joint reranking by relevance across repos.

## When triggered

1. Monitor TheForge request logs for cross-repo queries (e.g. `codeIndexerRepo="*"`).
2. Collect user feedback on recall quality across repos.
3. If recall complaints are credible (specific examples, 5+ repos indexed):
   a. **Option A:** Normalize cosine scores per-repo to [0, 1] before concatenation.
      Requires no new index; only backend ranking logic change.
   b. **Option B:** Build a shared cross-repo embeddings index (e.g., separate
      `.duck` file with all symbols from all repos). Requires schema expansion,
      new index-time logic, and careful handling of symbol name collisions.
4. Cost-benefit analysis at trigger time. If Option A is sufficient, ship that.
   If Option B is needed, plan a schema migration in code-graph-rag.
5. Update orchestrator context-bundle merging logic; run end-to-end bench.
