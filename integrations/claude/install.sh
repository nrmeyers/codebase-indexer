#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SCRIPT_DIR}/codebase-indexer"
DEST="${CLAUDE_SKILLS_DIR:-${HOME}/.claude/skills}/codebase-indexer"

mkdir -p "$(dirname "${DEST}")"
ln -sfn "${SRC}" "${DEST}"
echo "Linked ${DEST} -> ${SRC}"
