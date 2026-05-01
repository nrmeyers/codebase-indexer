# `.planning/` — Active Documents

Canonical planning artefacts for the Code Indexer + TheForge team
deployment.  Anything not in this list lives under `_archive/` and is
historical only.

| File | Audience | Purpose |
|------|----------|---------|
| `TEAM_DEPLOYMENT_PLAN.md` | Engineering lead | 9-phase ordered roadmap to ship the team-wide deployment.  Source of truth for what gets built next. |
| `FRONTEND_AGENT_BRIEF.md` | Frontend agent | Self-contained reference: stack, base URLs, endpoints, TS types, OAuth-forward auth wrapping, validation commands. |
| `DEVOPS_REQUEST.md` | Platform / DevOps team | Exact list of asks to unblock Phase 1: Entra app registrations, DNS, TLS, VM, persistent volumes, secrets, Grafana scrape config. |

## `phase-plans/` — detailed phase implementation plans

Each plan covers goals, file inventory, sample code, test plan, rollout, and trade-offs.

| Phase | File | One-line scope |
|-------|------|----------------|
| 1 | `phase-plans/PHASE_1_M365_OAUTH.md` | Two-app Entra ID (SPA + resource API), MSAL PKCE on FE, JWKS validation on both backends, group-OID → role mapping. |
| 2 | `phase-plans/PHASE_2_PERSISTENT_JOBS.md` | SQLite-backed job state (WAL mode), restart recovery via `interrupted` status, dedupe-409 on concurrent `POST /index`. |
| 3 | `phase-plans/PHASE_3_DOCKER.md` | Multi-stage Dockerfiles, docker-compose stack at `forge.navistone.com`, Caddy reverse proxy with Let's Encrypt + WS upgrade, GHCR CI/CD. |
| 4 | `phase-plans/PHASE_4_GRAFANA.md` | `prometheus_client` (Python) + `prom-client` (Node) `/metrics` endpoints, audit-event → counter bridge, Grafana dashboard + 4 alerts. |
| 5 | `phase-plans/PHASE_5_REALTIME_UPDATER.md` | Watchdog file-watcher per repo, 1.5 s asyncio debounce, partial re-index via Phase 2 `kind='watch_partial'` jobs, opt-in `POST /repos/{slug}/watch`. |
| 8 | `phase-plans/PHASE_8_HNSW_VSS.md` | DuckDB VSS extension HNSW index, trigger-gated activation runbook (p95 > 200 ms or > 50k symbols), per-repo flag rollout, recall@20 ≥ 98% gate. |
| 9 | `phase-plans/PHASE_9_CROSS_REPO_RANK.md` | RRF (rank-based) cross-repo result merging in TheForge, two-stage 9a-eval / 9b-ship, env-flag rollout, ADR-0006 supersedes ADR-0003 on ship. |

> Phases 6 (codebase cleanup) and 7 (doc rewrite) are meta-phases ongoing
> through routine engineering work; they have no separate plan file.

## Top-level project docs (not in this folder)

- [`../ROADMAP.md`](../ROADMAP.md) — high-level milestone tracker (v5.3 spec)
- [`../README.md`](../README.md) — service overview, local-dev setup
- [`../CLAUDE.md`](../CLAUDE.md) — agent operating rules for this repo
- [`../docs/adr/`](../docs/adr/) — deferred-decision ADRs (HNSW, CodeRankLLM, cross-repo)

## Archive

`_archive/` holds superseded plans kept for historical context only.
Not referenced by active work.  Do not edit; if you find yourself
wanting to, write a new doc instead.
