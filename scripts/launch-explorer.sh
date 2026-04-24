#!/usr/bin/env bash
# launch-explorer.sh — spin up kuzu-explorer against the current LadybugDB.
#
# Usage:
#   ./scripts/launch-explorer.sh                       # uses LADYBUG_DB_PATH env or .env
#   ./scripts/launch-explorer.sh /path/to/graph.db     # override DB path
#   PORT=9000 ./scripts/launch-explorer.sh             # override host port
#
# Why this script exists:
#   * Queries http://localhost:${INDEXER_PORT:-8000}/explorer/info if running, so the
#     exact launch command stays authoritative (single source of truth in explorer.py).
#   * Falls back to a direct `docker run` if the Code Indexer Service is not up.
#   * Never modifies the DB — kuzu-explorer runs read-only against a mounted volume.
#
# Requirements:
#   * Docker (only required for visualisation — core indexing is Docker-free)
#   * curl + jq (for the hot-path lookup; optional)

set -euo pipefail

PORT="${PORT:-7000}"
INDEXER_PORT="${INDEXER_PORT:-8000}"
DB_PATH="${1:-${LADYBUG_DB_PATH:-.cgr/graph.db}}"

# Try the live service first — the endpoint returns the authoritative command.
if command -v curl >/dev/null && command -v jq >/dev/null; then
  if info="$(curl -sf --max-time 2 "http://localhost:${INDEXER_PORT}/explorer/info")"; then
    available="$(echo "$info" | jq -r '.available')"
    cmd="$(echo "$info" | jq -r '.launch_command')"
    url="$(echo "$info" | jq -r '.viewer_url')"
    if [[ "$available" != "true" ]]; then
      echo "⚠ Code Indexer reports no indexed repos at $(echo "$info" | jq -r '.db_path')"
      echo "  Run 'curl -XPOST http://localhost:${INDEXER_PORT}/index ...' first."
      exit 1
    fi
    echo "▸ Launching kuzu-explorer (from /explorer/info) — open $url when ready"
    echo "  $cmd"
    exec $cmd
  fi
fi

# Offline path — construct the command locally.
if [[ ! -e "$DB_PATH" ]]; then
  echo "✗ DB path does not exist: $DB_PATH"
  echo "  Set LADYBUG_DB_PATH or pass the path as arg 1."
  exit 1
fi

ABS_DB="$(cd "$(dirname "$DB_PATH")" && pwd)/$(basename "$DB_PATH")"
MOUNT_SRC="$(dirname "$ABS_DB")"
DB_NAME="$(basename "$ABS_DB")"

echo "▸ Launching kuzu-explorer (offline mode) against $ABS_DB"
echo "  Open http://localhost:${PORT} when the container reports ready."

exec docker run --rm -p "${PORT}:8000" \
  -v "${MOUNT_SRC}:/database" \
  -e KUZU_PATH="/database/${DB_NAME}" \
  kuzudb/explorer:latest
