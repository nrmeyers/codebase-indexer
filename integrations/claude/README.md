# Claude Code harness integration

This directory holds the Claude Code skill for the `codebase-indexer` service.

## What the skill does

The `codebase-indexer` skill (`codebase-indexer/SKILL.md`) lets a Claude Code agent drive the running indexer service to:

- Search code by meaning (`code-indexer --json search`)
- Look up symbols and their source (`code-indexer --json symbol`)
- Trace caller/callee call graphs (`code-indexer --json callers` / `callees`)
- Assemble grounded multi-file context bundles before editing (`code-indexer --json bundle`)

## Why a skill and not MCP

MCP loads every tool schema up front, adding token overhead for every session even when the tools are never used. A Claude Code skill uses **progressive disclosure**: only the name and one-line description load by default (~0 tokens). The full instructions in `SKILL.md` are only injected into context when the agent explicitly invokes the skill.

## Install

```bash
./integrations/claude/install.sh
```

This symlinks `integrations/claude/codebase-indexer/` into `~/.claude/skills/codebase-indexer/` (or `$CLAUDE_SKILLS_DIR/codebase-indexer/` if set). After that, any Claude Code session can invoke the skill by name.

Alternatively, copy or symlink `codebase-indexer/` into `~/.claude/skills/` manually.

## Extend

Other harnesses get their own `integrations/<harness>/` directory alongside this one.
