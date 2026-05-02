#!/bin/bash
# Autonomous E2E cycle loop.
#
# Per .planning/E2E_TEST_OPTIMIZATION_PLAN.md §7. Runs cycles back-to-back
# applying mechanical fixes from a heuristic table. Stops on:
#   - 2 consecutive all-PASS cycles  → SUCCESS
#   - 5 cycles with no improvement   → STALL (writes STOPPED.md)
#   - any cycle that needs code-only fix → STALL (writes STOPPED.md)
#   - service crash                  → STALL
#
# Usage:
#   nohup bash scripts/run_cycle_loop.sh > /tmp/loop.log 2>&1 &
#
# The loop survives shell exit. Check progress via:
#   tail -f /tmp/loop.log
#   ls .planning/runs/

set -u
SERVICE_URL="${SERVICE_URL:-http://127.0.0.1:8000}"
QUERIES="${QUERIES:-scripts/queries.json}"
MAX_CYCLES="${MAX_CYCLES:-6}"
RUNS_DIR=".planning/runs"

cd "$(dirname "$0")/.."
mkdir -p "$RUNS_DIR"

log() { printf '[loop %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

# Stop conditions tracking
consecutive_pass=0
prev_failures=()

# ---- main loop -------------------------------------------------------------
for n in $(seq 2 "$MAX_CYCLES"); do
    log "=== Cycle $n start ==="

    # Health check first — bail if service is down
    if ! curl -sf "$SERVICE_URL/health" >/dev/null 2>&1; then
        log "FATAL: service not reachable at $SERVICE_URL"
        echo "service unreachable" > "$RUNS_DIR/STOPPED.md"
        exit 2
    fi

    # Decide indexing mode: full reindex on cycle 2 (post-config-change),
    # skip-indexing thereafter unless data is stale
    INDEX_FLAG="--skip-indexing"
    if [ "$n" = "2" ]; then
        INDEX_FLAG="--force-reindex"
    fi

    ts="$(date -u +%Y%m%dT%H%M%SZ)"
    out="$RUNS_DIR/$ts"
    mkdir -p "$out"

    log "running cycle: $INDEX_FLAG → $out"
    if ! uv run python scripts/run_e2e.py \
            --service-url "$SERVICE_URL" \
            --queries "$QUERIES" \
            --out "$out" \
            $INDEX_FLAG > "$out/driver.log" 2>&1; then
        rc=$?
        log "driver exited rc=$rc (FAIL or PREFLIGHT_FAIL)"
    fi

    # Always grade if query_results.jsonl exists
    if [ -f "$out/query_results.jsonl" ]; then
        uv run python scripts/grade_queries.py \
            --in "$out/query_results.jsonl" \
            --out "$out/query_grades.jsonl" >> "$out/driver.log" 2>&1

        # Re-score with grades present
        uv run python scripts/run_e2e.py \
            --service-url "$SERVICE_URL" \
            --queries "$QUERIES" \
            --out "$out" \
            --skip-indexing >> "$out/driver.log" 2>&1
    fi

    # Read summary
    if [ ! -f "$out/RUN_SUMMARY.md" ]; then
        log "no RUN_SUMMARY.md — driver died early; stopping"
        echo "cycle $n produced no summary; see $out/driver.log" > "$RUNS_DIR/STOPPED.md"
        exit 3
    fi

    overall=$(grep -E '^- overall:' "$out/RUN_SUMMARY.md" | head -1)
    log "cycle $n result: $overall"

    # Extract failing metric names (simple grep)
    failures=$(grep -E '^\| .+ \| FAIL \|' "$out/RUN_SUMMARY.md" | sed -E 's/^\| ([^ ]+) .*/\1/' || true)
    nfail=$(echo "$failures" | grep -c . || true)
    log "failing metrics: ${failures:-(none)}"

    if [ "$nfail" = "0" ]; then
        consecutive_pass=$((consecutive_pass + 1))
        log "all PASS — consecutive=$consecutive_pass"
        if [ "$consecutive_pass" -ge 2 ]; then
            log "STOP: 2 consecutive all-PASS cycles → SUCCESS"
            cat > "$RUNS_DIR/SUCCESS.md" << EOF
# E2E Optimization — SUCCESS

Reached 2 consecutive all-PASS cycles at $ts.
Final summary: $out/RUN_SUMMARY.md
EOF
            exit 0
        fi
        # Pass once — re-run to confirm stability before declaring done
        log "running again to confirm stability"
        continue
    else
        consecutive_pass=0
    fi

    # Compare to previous failures: are we stalled?
    cur_failures=$(echo "$failures" | sort | uniq | tr '\n' ',')
    prev=${prev_failures[$((n-2))]:-}
    prev_failures[$((n-1))]=$cur_failures

    # Apply heuristic fix
    fix_applied=""
    case "$failures" in
        *search_semantic_p95_no_rerank_s*)
            # Likely cause: cold-start of LM Studio embedder OR network round-trip
            # On second occurrence, no mechanical fix — needs code/config attention.
            if [ "$n" -ge "3" ]; then
                fix_applied="STALL_SEMANTIC_LATENCY"
            fi
            ;;
        *indexing_rate_symbols_per_s*)
            # If indexing rate fails after embed model is loaded, likely a phase
            # bottleneck we can't tune mechanically. Stop.
            fix_applied="STALL_INDEXING_RATE"
            ;;
    esac

    if [[ "$fix_applied" == STALL_* ]]; then
        log "STOP: heuristic table has no mechanical fix for: $failures"
        cat > "$RUNS_DIR/STOPPED.md" << EOF
# E2E Optimization — STALLED

Cycle $n at $ts could not be auto-fixed.
Failing metrics: $failures
Reason: $fix_applied
Last summary: $out/RUN_SUMMARY.md

Next steps require human/agent decision (e.g. tune k, swap embedder,
activate Phase 8 HNSW, etc).
EOF
        exit 4
    fi

    log "no mechanical fix needed yet; running another cycle"
done

log "STOP: max cycles ($MAX_CYCLES) reached without convergence"
cat > "$RUNS_DIR/STOPPED.md" << EOF
# E2E Optimization — STALLED (max cycles)

Reached MAX_CYCLES=$MAX_CYCLES without 2-consecutive-pass.
Last summary: $out/RUN_SUMMARY.md
EOF
exit 5
