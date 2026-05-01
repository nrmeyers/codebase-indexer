# E2E Cycle Runs

Each subdirectory is one cycle of the E2E indexing+querying loop.

## Layout

```
.planning/runs/
  README.md              ← you are here
  INDEX.md               ← chronological roll-up of all cycles
  <UTC-timestamp>/
    RUN_SUMMARY.md       ← human-readable pass/fail report
    index_phase_times.json
    query_results.jsonl
    query_grades.jsonl   ← if grade_queries.py has been run
    metrics_snapshot_pre.txt
    metrics_snapshot_post.txt
    decision.md          ← which lever was applied this cycle (next-cycle prep)
```

## Running a cycle

```
uv run python scripts/run_e2e.py \
    --service-url http://localhost:8000 \
    --queries scripts/queries.json \
    --out .planning/runs/$(date -u +%Y%m%dT%H%M%SZ) \
    [--force-reindex]

uv run python scripts/grade_queries.py \
    --in .planning/runs/<ts>/query_results.jsonl \
    --out .planning/runs/<ts>/query_grades.jsonl

# re-run scoring against the now-graded file
uv run python scripts/run_e2e.py ... --out <same dir>
```

## Stop criterion

Per `.planning/E2E_TEST_OPTIMIZATION_PLAN.md` §7:

- All SLOs green for two consecutive cycles → done
- Or 5 cycles with no improvement → surface diagnostic dump
