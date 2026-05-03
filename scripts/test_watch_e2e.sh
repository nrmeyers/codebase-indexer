#!/usr/bin/env bash
# Phase 5 — end-to-end validation gate for the realtime file-watcher.
#
# Requires:
#   * The code-indexer-service running with WATCH_ENABLED=true on PORT (default 8000).
#   * The service must have already indexed the repo at REPO_PATH.
#   * websocat (brew install websocat) or wscat for WS capture.
#
# Usage:
#   WATCH_ENABLED=true PORT=8000 REPO_PATH=. SLUG=code-indexer-service \
#       bash scripts/test_watch_e2e.sh
#
# Exit codes:
#   0  — all assertions passed
#   1  — an assertion failed
#   2  — prerequisite missing

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:${PORT:-8000}}"
SLUG="${SLUG:-code-indexer-service}"
REPO_PATH="${REPO_PATH:-.}"
CANARY="unique-canary-string-$(date +%s)"
WAIT_SECS="${WAIT_SECS:-5}"

info()  { echo "[e2e] $*"; }
fail()  { echo "[e2e] FAIL: $*" >&2; exit 1; }
check() { command -v "$1" &>/dev/null || { echo "[e2e] prerequisite '$1' missing (SKIPPED)" >&2; exit 2; }; }

check curl
check jq

info "Base URL: $BASE_URL"
info "Slug:     $SLUG"
info "Repo:     $REPO_PATH"
info "Canary:   $CANARY"

# ── 1. Verify service is reachable ──────────────────────────────────────────
HEALTH=$(curl -sf "$BASE_URL/health") || fail "Service unreachable at $BASE_URL"
info "Health: $(echo "$HEALTH" | jq -r .status)"

# ── 2. Assert canary not yet in search ──────────────────────────────────────
HITS_BEFORE=$(curl -sf "$BASE_URL/search/semantic?q=${CANARY}&repo=${SLUG}" \
    | jq '.results | length') || fail "Semantic search call failed"
if [ "$HITS_BEFORE" -ne "0" ]; then
    fail "Expected 0 hits for canary before file edit, got $HITS_BEFORE"
fi
info "Step 2 OK — canary not in index yet"

# ── 3. POST /repos/{slug}/watch ─────────────────────────────────────────────
START_RESP=$(curl -sf -X POST "$BASE_URL/repos/$SLUG/watch" \
    -H "Content-Type: application/json" -d '{}') \
    || fail "POST /repos/$SLUG/watch failed"
WATCHER_ID=$(echo "$START_RESP" | jq -r .watcher_id)
DEBOUNCE_MS=$(echo "$START_RESP" | jq -r .debounce_ms)
info "Step 3 OK — watcher started (id=$WATCHER_ID debounce=${DEBOUNCE_MS}ms)"

# ── 4. Append canary to a source file ───────────────────────────────────────
TARGET_FILE="$REPO_PATH/app/services/watch_manager.py"
if [ ! -f "$TARGET_FILE" ]; then
    fail "Target file not found: $TARGET_FILE"
fi
echo "# $CANARY" >> "$TARGET_FILE"
info "Step 4 OK — canary appended to $TARGET_FILE"

# ── 5. Wait for debounce + partial index ────────────────────────────────────
info "Step 5 — waiting ${WAIT_SECS}s for partial index..."
sleep "$WAIT_SECS"

# ── 6. Assert canary now in search ──────────────────────────────────────────
HITS_AFTER=$(curl -sf "$BASE_URL/search/semantic?q=${CANARY}&repo=${SLUG}" \
    | jq '.results | length') || fail "Semantic search call failed (post-edit)"
info "Hits after edit: $HITS_AFTER"

# ── 7. Cleanup: remove appended canary line ──────────────────────────────────
# Portable removal of last line (works on macOS + Linux).
if command -v gsed &>/dev/null; then
    gsed -i '$ d' "$TARGET_FILE"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS sed requires a backup extension with -i
    sed -i '' '$ d' "$TARGET_FILE"
else
    sed -i '$ d' "$TARGET_FILE"
fi
info "Step 7 OK — canary line removed from $TARGET_FILE"

# ── 8. DELETE /repos/{slug}/watch ───────────────────────────────────────────
STOP_RESP=$(curl -sf -X DELETE "$BASE_URL/repos/$SLUG/watch") \
    || fail "DELETE /repos/$SLUG/watch failed"
STOPPED_AT=$(echo "$STOP_RESP" | jq -r .stopped_at)
info "Step 8 OK — watcher stopped at $STOPPED_AT"

# ── 9. Result ────────────────────────────────────────────────────────────────
if [ "$HITS_AFTER" -ge "1" ]; then
    info "PASS — canary found in semantic search within ${WAIT_SECS}s of file save"
    exit 0
else
    fail "Canary not found in semantic search after ${WAIT_SECS}s — partial index may not have completed"
fi
