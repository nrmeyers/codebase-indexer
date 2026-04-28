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

## Top-level project docs (not in this folder)

- [`../ROADMAP.md`](../ROADMAP.md) — high-level milestone tracker (v5.3 spec)
- [`../README.md`](../README.md) — service overview, local-dev setup
- [`../CLAUDE.md`](../CLAUDE.md) — agent operating rules for this repo
- [`../docs/adr/`](../docs/adr/) — deferred-decision ADRs (HNSW, CodeRankLLM, cross-repo)

## Archive

`_archive/` holds superseded plans kept for historical context only.
Not referenced by active work.  Do not edit; if you find yourself
wanting to, write a new doc instead.
