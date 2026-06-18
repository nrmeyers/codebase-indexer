# AGENTS.md — Code Indexer Service

This repo's `CLAUDE.md` is intentionally gitignored (per-developer local
notes). The agent-skills index lives here so the whole team sees it.

## Agent skills

### Backlog

GitHub Issues on `nrmeyers/codebase-indexer` via the `gh` CLI. See `docs/agents/backlog.md`.

### Triage labels

Default canonical names (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). Created lazily on first use. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout. `CONTEXT.md` lives at the repo root (lazy — produced by `/grill-with-docs` when terms get resolved). ADRs live at `docs/adr/` and currently hold three deferred-decision records (HNSW, CodeRankLLM proper, cross-repo unified rank). See `docs/agents/domain.md`.

## Top-level docs

- `README.md` — service overview, local-dev setup
- `ROADMAP.md` — phased milestone tracker (v5.3 spec)
- `docs/adr/` — architectural decision records
- `.planning/` — team deployment artefacts (`TEAM_DEPLOYMENT_PLAN`, `FRONTEND_AGENT_BRIEF`, `DEVOPS_REQUEST`, `phase-plans/`)
