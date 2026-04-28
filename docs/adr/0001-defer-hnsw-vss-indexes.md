# ADR-0001: Defer HNSW / VSS Indexes

**Status:** Deferred — not implemented as of 2026-04-27
**Trigger:** cosine query p95 > 200 ms OR repo > 50,000 symbols.

## Context

Current vector retrieval uses DuckDB's linear scan via `array_cosine_distance` on
768-dimensional FLOAT arrays. This is simple, exact, and presently performant. As
the codebase scales (more symbols per repo, more repos indexed), retrieval latency
may degrade below acceptable thresholds.

HNSW (Hierarchical Navigable Small World) and VSS (Vector Similarity Search) are
approximation-based indices that trade recall precision for sub-linear query time,
suited to scale-out scenarios.

## Decision

Stay on linear scan via DuckDB `array_cosine_distance` until one of the trigger
conditions fires. No HNSW or VSS code paths are built now. Accept that query p95
may grow linearly with symbol count; monitor and measure before investing in
approximation.

## Consequences

**What stays simple:** Vector indexing logic remains a single path; no feature flags,
no accuracy-vs-speed tuning. Schema changes (adding an HNSW index structure to
`.duck` files) are deferred.

**What we accept as cost:** Retrieval latency degrades proportionally with corpus
size. Single-repo searches at 50k+ symbols may hit the 200 ms p95 ceiling. No
approximation means every query is exact; if needed, future migrations will be
a breaking change to the `.duck` schema.

## When triggered

1. Run end-to-end latency bench on current prod (5k–50k symbols, real queries).
2. If p95 > 200 ms OR repo corpus > 50k symbols:
   a. Prototype HNSW branch behind a feature flag (e.g. `ENABLE_HNSW_SEARCH`).
   b. Build golden set of 100–200 synthetic queries with hand-validated top-10 recall.
   c. Validate new path recalls @ top-10 before flipping default.
   d. Measure latency reduction; document in `.duck` schema migration notes.
3. Merge to main only after both latency and recall targets are met.
