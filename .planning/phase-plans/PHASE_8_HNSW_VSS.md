# Phase 8 — HNSW / VSS Indexes

**Status:** plan, dormant — activates only when trigger conditions fire
**Owner:** application engineering (storage layer)
**Depends on:** Phase 3 (image bakes the VSS extension), Phase 4 (`forge_indexer_search_duration_seconds` p95 metric is the trigger sensor)
**Blocks:** nothing — this phase ships only when needed
**Implements:** `code-indexer-service/docs/adr/0001-defer-hnsw-vss-indexes.md`

## 1. Goal and non-goals

### 1.1 Goal

Replace the brute-force linear scan in DuckDB (`array_cosine_distance` over `FLOAT[768]`) with an HNSW index from DuckDB's official VSS extension, **only after** the trigger conditions in §2 fire. Keep the brute-force path as a fallback.

Hard targets:
- **Latency.** Cosine search p95 on a 100k-symbol repo drops below **50 ms** (down from a projected 200–400 ms on linear scan).
- **Recall.** Recall@20 vs. brute-force on a 100-query holdout set is **≥ 98%** with the default tuning (`M=16`, `ef_construction=200`, `ef_search=64`). On 100 randomized query vectors against a 10k-row corpus, the HNSW top-20 must be **identical** to brute-force (recall == 100%) for the ranking-equivalence test to pass; the 98% bar is the production-tolerance bar measured on real query embeddings, not synthetic uniform vectors.
- **Compatibility.** The query API (`search_similar(conn, query_vec, k)`) signature does not change. Callers do not learn that HNSW exists.

### 1.2 Non-goals

- **No schema break.** `.duck` files written before Phase 8 still work after Phase 8. The HNSW index is added on top of an unchanged `embeddings` table.
- **No new Python deps.** DuckDB ships the VSS extension via `INSTALL vss; LOAD vss;` — no `pip install` required.
- **No automatic enablement.** This phase ships behind `HNSW_ENABLED=false` by default. Operations flips one repo at a time.
- **No tuning sweeps in this phase.** We pick the upstream-recommended defaults (`M=16`, `ef_construction=200`) and document the knobs in §11. Tuning is its own follow-up if defaults miss the target.
- **No multi-vector / multi-index work.** One HNSW index per `embeddings` table. Centrality and metadata stay on linear paths (they are not similarity-searched).

## 2. Trigger conditions and runbook

### 2.1 Triggers (either fires)

1. **Latency trigger.** Phase 4 dashboard shows `histogram_quantile(0.95, sum by (le) (rate(forge_indexer_search_duration_seconds_bucket{endpoint="semantic"}[1h])))` **> 200 ms** for 24 h continuously on at least one repo.
2. **Size trigger.** Any repo's `forge_indexer_embeddings_count{repo_name="…"}` (Phase 4 gauge) crosses **50,000 rows**. Read directly from `node_count` in the repo's graph or `row_count(conn)` against the `.duck`.

### 2.2 Activation runbook

When a trigger fires:

1. **Confirm signal.** Pull the offending repo's metrics for the last 7 days. A single-day spike from an embedder rerun is not a trigger — we want a sustained regression.
2. **Open Phase 8 epic.** Linear issue (template in §13 references). Cite the trigger metric and timestamp.
3. **Bake the VSS extension into the image** (§3). Phase 3 image rebuild + redeploy. This is a no-op until the per-repo flag is flipped.
4. **Build the index** for the offending repo (§6). Idempotent migration.
5. **Run the ranking-equivalence test** (§8) against that repo's `.duck` file. If recall@20 < 98% on its actual query distribution, do **not** flip the flag — fall through to §11 tuning.
6. **Flip the per-repo feature flag** (§12). Watch the dashboard for 24 h.
7. **Roll back at the first regression** (§10). The query path supports both modes; rollback is a flag flip, no data migration.

## 3. DuckDB VSS extension

DuckDB ships the VSS (Vector Similarity Search) extension as a first-party community extension. Install path is one-time per database connection.

### 3.1 Install / load

```python
def _ensure_vss(conn: Any) -> bool:
    """Install + LOAD the VSS extension. Return True if available."""
    try:
        conn.execute("INSTALL vss")
        conn.execute("LOAD vss")
        return True
    except Exception:
        return False  # offline / unsigned binary / sandboxed env
```

`INSTALL vss` is a network operation (downloads the extension binary) the first time it runs. Subsequent loads on the same machine hit the local extension cache. In the containerised Phase 3 image, this means:

- The image **must bake the extension into the cache** at build time so cold-start containers do not need network access.
- Add to `code-indexer-service/Dockerfile`:
  ```dockerfile
  RUN python -c "import duckdb; c = duckdb.connect(); c.execute('INSTALL vss'); c.execute('LOAD vss')"
  ```
  This populates `~/.duckdb/extensions/<version>/<platform>/vss.duckdb_extension` inside the image. `LOAD vss` at runtime is then offline.

### 3.2 Version pin

DuckDB's VSS extension version tracks the DuckDB version (extensions are signed per release). The repo currently pins `duckdb>=1.1.0` in `pyproject.toml`. Phase 8 raises this floor to `duckdb>=1.1.3` (first release with stable HNSW persistence — earlier 1.1.x dropped the index on close). Verified at activation time; bump if upstream changes.

### 3.3 Behaviour in containerised env

- **Networkless containers** (production): the bake step in §3.1 is mandatory. Without it, `INSTALL vss` raises `IOException: connection refused` and `_ensure_vss` returns `False` — silent fallback to linear scan.
- **CI runners**: same — bake the extension into the test image, or run with the VSS preload step in CI fixtures.
- **Dev laptops**: `INSTALL vss` works the first time over the network and is cached afterwards.

## 4. Index DDL

```sql
CREATE INDEX IF NOT EXISTS hnsw_function_embed
    ON embeddings
    USING HNSW (embedding)
    WITH (
        metric = 'cosine',
        M = 16,
        ef_construction = 200
    );
```

Tuning knobs (defaults; see §11):
- `M` — graph max-degree per node. 16 is DuckDB's default and the upstream HNSW paper's recommendation for general embedding workloads. Higher `M` raises recall and RAM; lower `M` reduces both.
- `ef_construction` — search width during build. 200 is the default. Higher values improve graph quality (and recall) at index-build cost; lower values are faster to build but produce a worse graph.
- `ef_search` — search width at query time. Set per-connection via `SET hnsw_ef_search = 64` (default 64). Tunable per-query for recall/latency trade-off.

### 4.1 Why `metric = 'cosine'` and not inner product

Embeddings are L2-normalised at write time (`vector_store.py::_l2_normalise`, `vector_store_arrow.py::_normalise_matrix`). With unit vectors, `cosine` and `inner product` produce the same ranking. We pick `'cosine'` explicitly because (a) the storage contract is documented as cosine and (b) it is robust if a future code path forgets to normalise.

## 5. Query rewrite

DuckDB's VSS extension exposes the `<=>` operator for cosine distance. The HNSW index is selected automatically when the query has the shape `ORDER BY <distance_op>(col, $param) LIMIT k`.

### 5.1 Current query

```sql
SELECT qualified_name, file_path, start_line, end_line,
       1.0 - array_cosine_distance(embedding, ?::FLOAT[768]) AS score
FROM embeddings
ORDER BY score DESC
LIMIT ?
```

This always linearly scans (`array_cosine_distance` is a scalar function — the planner can't use the HNSW index even when it exists).

### 5.2 New query (HNSW path)

```sql
SELECT qualified_name, file_path, start_line, end_line,
       1.0 - (embedding <=> ?::FLOAT[768]) AS score
FROM embeddings
ORDER BY embedding <=> ?::FLOAT[768]
LIMIT ?
```

Two binds of the same query vector — DuckDB does not (yet) reuse a single bind across SELECT and ORDER BY when one is wrapped in an arithmetic expression. Bind both; same vector. Negligible cost vs. the search itself.

### 5.3 Routing in `search_similar`

```python
def search_similar(conn, query_vec, k=10):
    normalised = _l2_normalise(query_vec)
    if _hnsw_active(conn):
        sql = """
            SELECT qualified_name, file_path, start_line, end_line,
                   1.0 - (embedding <=> ?::FLOAT[768]) AS score
            FROM embeddings
            ORDER BY embedding <=> ?::FLOAT[768]
            LIMIT ?
        """
        rows = conn.execute(sql, (normalised, normalised, int(k))).fetchall()
    else:
        # existing linear-scan path — unchanged
        rows = conn.execute(_LINEAR_SQL, (normalised, int(k))).fetchall()
    return [SearchResult(...) for r in rows]
```

`_hnsw_active(conn)` checks two conditions: the extension loaded (per §3) AND the index exists on this connection's database (per §6). Cached on the connection to avoid repeating `duckdb_indexes` lookups per query.

## 6. Persistence and migration

DuckDB's HNSW indexes are persisted inside the `.duck` file (since 1.1.3). After `CREATE INDEX`, closing and reopening the database preserves the index — no rebuild on connect.

### 6.1 Idempotent migration

Add `vector_store.ensure_hnsw_index(conn)` — safe to call any number of times:

```python
def ensure_hnsw_index(conn: Any) -> bool:
    """Create the HNSW index if VSS is available. Idempotent."""
    if not _ensure_vss(conn):
        return False
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS hnsw_function_embed
            ON embeddings
            USING HNSW (embedding)
            WITH (metric = 'cosine', M = 16, ef_construction = 200)
        """
    )
    return True
```

Called once during the activation runbook (§2.2) for each repo we are flipping.

### 6.2 Migration script

`scripts/enable_hnsw.py REPO_NAME` (new):
1. Resolve `.duck` path from repo metadata.
2. Open connection.
3. Call `ensure_hnsw_index(conn)`.
4. Log row count, index build time, on-disk size delta.
5. Run a sanity 10-query recall check (§8 mini-version).

Re-running on an already-indexed `.duck` is a no-op (`CREATE INDEX IF NOT EXISTS`).

### 6.3 New repos created post-Phase-8

When `HNSW_ENABLED=true` (global flag), `open_or_create` calls `ensure_hnsw_index` after schema creation. New `.duck` files are HNSW-indexed from row 1. When the flag is off, new files match today's behaviour exactly.

## 7. Bulk-insert path interaction

Open question that the test plan resolves: does `vector_store_arrow.bulk_insert_arrow` cooperate with an existing HNSW index?

### 7.1 Hypothesis

DuckDB maintains the HNSW index incrementally on `INSERT` — but the existing Arrow path goes `register Arrow table → INSERT INTO embeddings SELECT FROM staging`. This is a single columnar copy. The HNSW maintainer reads each new row and updates the graph. Cost is per-row at index-time, not amortised.

For the existing **324–390× Arrow speedup** at FLOAT[768] payload sizes, HNSW maintenance is the new bottleneck. Expected impact: insert throughput drops from ~0.11 ms/row (no index) to roughly 0.5–2 ms/row (with index, depending on graph state).

### 7.2 Decision tree

The bench in §9 measures both. Three plausible outcomes:

| Insert cost with index | Action |
|---|---|
| < 5× slowdown vs. no-index Arrow | Keep `bulk_insert` and HNSW maintenance coupled. Simplest path. |
| 5–20× slowdown | **Build-then-index** strategy: drop HNSW before a large `bulk_insert`, recreate after. Add `vector_store.suspend_hnsw(conn)` / `resume_hnsw(conn)` helpers; ingestor calls them around the bulk write loop. |
| > 20× slowdown | Same as above, escalated: rebuild only after every Nth bulk insert; gate behind `HNSW_REBUILD_THRESHOLD_ROWS`. |

We do not pick the strategy in this plan — the bench picks it. The plan commits to running the bench with the HNSW build path on real ingestor input (a 10k-row representative sample), comparing to the zero-index baseline.

### 7.3 Centrality and metadata writes

Unaffected. Only the `embeddings` table has the HNSW index. `centrality` and `repo_metadata` writes do not touch it.

## 8. Ranking equivalence test

Extend `code-graph-rag/codebase_rag/tests/test_duckdb_vector_store_arrow.py` with a new test class.

### 8.1 Synthetic test (deterministic, in CI)

```python
def test_hnsw_top20_matches_brute_force_on_10k_corpus():
    rng = numpy.random.default_rng(seed=20260430)
    corpus = rng.standard_normal((10_000, 768)).astype("float32")
    queries = rng.standard_normal((100, 768)).astype("float32")

    conn = vector_store.open_or_create(tmp_path / "eq.duck")
    rows = [EmbeddingRow(qualified_name=f"q{i}", embedding=corpus[i].tolist(),
                         file_path="x", start_line=0, end_line=0,
                         symbol_type="function") for i in range(10_000)]
    vector_store.bulk_insert(conn, rows)

    # Brute-force baseline (linear scan path)
    bf_results = [vector_store.search_similar(conn, q.tolist(), k=20) for q in queries]

    # Build HNSW and switch path
    assert vector_store.ensure_hnsw_index(conn) is True
    hnsw_results = [vector_store.search_similar(conn, q.tolist(), k=20) for q in queries]

    for q_idx, (bf, hnsw) in enumerate(zip(bf_results, hnsw_results)):
        assert [r.qualified_name for r in bf] == [r.qualified_name for r in hnsw], \
            f"top-20 ordering diverged on query {q_idx}"
```

**Pass criterion:** identical top-20 on all 100 queries against a 10k-row uniform-Gaussian corpus. (Real embeddings are not uniform-Gaussian — but with default `ef_search=64` and a 10k corpus, HNSW recall@20 is empirically 100% on uniform data. Any divergence flags a bug, not a tuning choice.)

### 8.2 Real-embedding holdout (manual, run during activation)

The 100% bar at synthetic 10k breaks down at 100k+ rows of structured (non-uniform) embeddings. A separate, run-during-activation test in `scripts/recall_holdout.py`:

1. Take an indexed repo's `.duck`.
2. Sample 200 query vectors from the same embedder applied to held-out symbol docstrings.
3. Run brute-force top-20 (force the linear path).
4. Run HNSW top-20.
5. Compute recall@20 = |HNSW ∩ brute-force| / 20, averaged over queries.
6. Assert ≥ 98%.

Output: `code-indexer-service/.planning/RECALL_<REPO>_<DATE>.md` — one row per repo we activate HNSW on.

## 9. Performance bench

Extend the bench harness pattern from `scripts/bench_bulk_insert.py` (per `BENCH_RESULTS_2026-04-27.md`) with query-side measurement.

### 9.1 New script: `code-graph-rag/scripts/bench_hnsw_query.py`

Measures cosine-search p50/p95/p99 latency at 1k / 10k / 100k rows, with and without HNSW. Each combination runs 1000 randomized query vectors after a 50-vector warmup.

```text
Output table:
  rows  |  hnsw  |  p50_ms  |  p95_ms  |  p99_ms  |  qps
--------|--------|----------|----------|----------|-------
  1k    |  off   |    …     |    …     |    …     |   …
  1k    |  on    |    …     |    …     |    …     |   …
  10k   |  off   |    …     |    …     |    …     |   …
  10k   |  on    |    …     |    …     |    …     |   …
  100k  |  off   |    …     |    …     |    …     |   …
  100k  |  on    |    …     |    …     |    …     |   …
```

Also reports:
- Index build time at each size.
- On-disk `.duck` size delta from the index.
- Resident memory peak during build (HNSW is RAM-hungry — see §11).
- Insert throughput delta when HNSW is present (resolves §7's open question).

### 9.2 Output

`code-graph-rag/scripts/BENCH_HNSW_<DATE>.md` (mirrors the format of `BENCH_RESULTS_2026-04-27.md`):
- Headline number(s).
- Raw table.
- Decision (does the 100k-row p95 drop below 50 ms?).
- Insert-cost decision per §7.2.
- Out-of-scope notes.

The bench result is a pre-merge gate: if p95 at 100k rows is not below 50 ms, Phase 8 does not ship — re-tune `M` and `ef_construction` per §11 and re-run.

## 10. Roll-back path

The query layer supports both paths. Roll-back is a one-line flag flip:

1. `HNSW_ENABLED=false` (global) **or** flip the per-repo flag (§12) for the offending repo.
2. Restart the Code Indexer process.
3. `_hnsw_active(conn)` now returns `False`; `search_similar` falls through to the linear-scan path.
4. The HNSW index remains on disk — leave it. `DROP INDEX hnsw_function_embed` is reversible but unnecessary; the linear path simply ignores it.

If the index file is corrupted (rare — DuckDB persists transactionally) and is causing crashes on `LOAD`:

```sql
DROP INDEX IF EXISTS hnsw_function_embed;
```

Then re-run the activation runbook from step 4 once the cause is known. The data table (`embeddings`) is untouched.

## 11. Trade-offs

### 11.1 Index build time

HNSW build is `O(N log N · ef_construction)` with a high constant. Empirical estimate at our embedder's dim (768):
- 10k rows: < 1 s.
- 50k rows: 5–15 s.
- 100k rows: 20–60 s.
- 500k rows: 5–10 minutes.

The build runs **once per repo** at activation, then incremental on insert. Acceptable for an activation-time migration. **Not** acceptable as a per-bulk-insert cost — see §7.

### 11.2 RAM during construction

HNSW holds the entire graph in memory while building. Approximate footprint: `N × M × M_max × 4 bytes` for the graph + `N × dim × 4 bytes` for the vectors themselves.
- 100k rows, `M=16`, `M_max=2M=32`, dim=768: ≈ 100k × 32 × 4 + 100k × 768 × 4 ≈ 13 MB graph + 307 MB vectors ≈ 320 MB peak.
- 500k rows: ≈ 1.6 GB peak.

The Phase 3 container has a default `memory: 2g` limit. Phase 8 raises the indexer container's limit to `4g` for the indexing worker (the API process is unaffected — query-time HNSW load is a small mmap-friendly read, not a peak-RAM event).

### 11.3 Per-query `ef_search`

`ef_search` controls the recall/latency trade-off at query time. Default 64. The query path can override per-call:

```python
conn.execute(f"SET hnsw_ef_search = {ef_search}")
```

If recall@20 on a real repo regresses below 98% at the default, Phase 8's runbook prescribes raising `ef_search` to 128 before re-running the holdout. Latency cost is roughly linear in `ef_search`.

### 11.4 Approximation is approximate

HNSW does not guarantee top-k correctness. Even at `ef_search=∞` it is empirically near-perfect, not provably perfect. The `≥ 98% recall@20` bar is the production tolerance. Anyone reading downstream-ranked results should not assume exactness.

### 11.5 Insert path coupling

If §7.2 lands on the suspend-and-rebuild strategy, two operations now exist that previously were one: bulk insert, then index rebuild. The ingestor pipeline gets a new step. Documented in §13 critical files. Failure handling: if rebuild fails after a successful insert, the table is correct but unindexed — the next `ensure_hnsw_index` call recovers.

## 12. Rollout

### 12.1 Feature flags

Two layers:

1. **Global flag** `HNSW_ENABLED` (default `false`). Controls whether new `.duck` files get an HNSW index at create time and whether the query path attempts the HNSW route.
2. **Per-repo flag** stored in `repo_metadata` under key `hnsw_active`:
   - `'true'` → query path uses HNSW (provided global `HNSW_ENABLED` is also true and the index exists).
   - `'false'` or absent → linear-scan path, even if the index file exists.

Per-repo gating lets us flip one repo, watch it for 24 h, then flip the next. A bad rollout on one repo does not bring down search globally.

### 12.2 Order of merge

1. `vector_store.py`: add `_ensure_vss`, `ensure_hnsw_index`, `_hnsw_active`, branch in `search_similar`.
2. `vector_store_arrow.py`: bench-driven decision per §7.2 — either no change or add `suspend_hnsw`/`resume_hnsw` helpers + ingestor wiring.
3. `tests/test_duckdb_vector_store_arrow.py`: add ranking-equivalence test (§8.1).
4. `scripts/bench_hnsw_query.py` + `scripts/recall_holdout.py` + `scripts/enable_hnsw.py` (new).
5. `code-indexer-service/Dockerfile`: bake the VSS extension (§3.3).
6. `.env.example`: add `HNSW_ENABLED=false`, `HNSW_M=16`, `HNSW_EF_CONSTRUCTION=200`, `HNSW_EF_SEARCH=64`.
7. Phase 4 dashboard: add an "HNSW status" panel sourcing `forge_indexer_hnsw_active` (new gauge, 1 per indexed-repo label).

### 12.3 Validation gate

Before declaring Phase 8 done:
- §9 bench output committed; 100k p95 < 50 ms.
- §8.1 ranking-equivalence test green in CI.
- §8.2 real-embedding holdout run on at least 2 repos; recall@20 ≥ 98%.
- Roll-back drill: enable on a staging repo, flip flag off, observe linear-scan path resumes; re-flip on; observe HNSW path resumes. No process restart between flips beyond the standard service restart.

## 13. Critical Files for Implementation

- `code-graph-rag/codebase_rag/storage/vector_store.py` — add `_ensure_vss`, `ensure_hnsw_index`, `_hnsw_active`, dual-path `search_similar`. Suspend/resume helpers if §7.2 lands there.
- `code-graph-rag/codebase_rag/storage/vector_store_arrow.py` — bench-conditional changes per §7.2.
- `code-graph-rag/codebase_rag/tests/test_duckdb_vector_store_arrow.py` — extend with the ranking-equivalence test (§8.1).
- `code-graph-rag/scripts/bench_hnsw_query.py` (new) — query-latency harness.
- `code-graph-rag/scripts/recall_holdout.py` (new) — real-embedding recall measurement.
- `code-graph-rag/scripts/enable_hnsw.py` (new) — per-repo activation script.
- `code-graph-rag/scripts/BENCH_HNSW_<DATE>.md` (new) — bench output committed alongside the merge.
- `code-graph-rag/pyproject.toml` — bump `duckdb>=1.1.3`.
- `code-indexer-service/Dockerfile` — bake the VSS extension into the image (§3.3).
- `code-indexer-service/.env.example` — add HNSW env vars (§12.2).
- `code-indexer-service/app/services/indexer.py` — wire `suspend_hnsw`/`resume_hnsw` if §7.2 demands it; otherwise no change.
- `code-indexer-service/app/metrics.py` — add `forge_indexer_hnsw_active{repo_name}` gauge for Phase 4 dashboard.
- `code-indexer-service/grafana/forge-dashboard.json` — add "HNSW status" panel.
- `code-indexer-service/docs/adr/0001-defer-hnsw-vss-indexes.md` — flip `Status:` to `Implemented` and link this plan.
- `code-indexer-service/docs/adr/000X-hnsw-vss-indexes.md` (new) — the post-implementation ADR documenting actual measured recall and latency.
