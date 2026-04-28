# Phase 3 — Containerisation & docker-compose deployment

**Target:** `forge.navistone.com` — single VM, ~10–25 internal users
**Status:** plan ready for review (drafted 2026-04-28)
**Companion docs:** `TEAM_DEPLOYMENT_PLAN.md` §3, `DEVOPS_REQUEST.md` §3, §6, §7
**Predecessors:** Phase 1 (M365 OAuth) and Phase 2 (persistent job store) must be merged first
**Successor:** Phase 4 (Grafana observability) consumes `/metrics` endpoints exposed here

---

## 1. Goals & non-goals

**Goals**
1. One reproducible artifact per service: `ghcr.io/navistone/forge-app:<sha>` and `ghcr.io/navistone/forge-code-indexer:<sha>`.
2. `docker compose up -d` on `forge.navistone.com` brings the stack up green from cold, with persistent state on `/var/lib/forge`.
3. Caddy terminates TLS for `forge.navistone.com`, routes `/api/code-indexer/*` and `/api/skill-api/*` to sidecars, upgrades `/ws` to WebSocket, forwards `Authorization` untouched.
4. CI builds and pushes images on every merge to `main`; deployment is `docker compose pull && docker compose up -d`.
5. Validation gate: kill the host, restore `/var/lib/forge` from snapshot on a new VM, `docker compose up -d`, hit `/health` green within 5 minutes.

**Non-goals**
- LM Studio containerisation (DEVOPS_REQUEST §6 — host-pinned for GPU; revisit Phase 9 with vLLM).
- Skill API runtime maturity — stubbed in compose for completeness.
- Kubernetes — explicitly deferred (see §10).
- Blue/green, canary, autoscaling — out of scope for 10–25 users.

---

## 2. Architectural decisions (call-outs)

### 2.1 Single VM, not Kubernetes (this phase)

- 10–25 internal users do not justify the operational tax of k8s.
- Stateful surface is one directory (`/var/lib/forge`); k8s PVs add complexity without benefit at this size.
- Navistone platform team does not run a managed k8s today.

**Migration path to k8s when needed (>50 users, multi-region, or platform team adopts k8s):**
1. Each Dockerfile already produces non-root, single-process image suitable for a `Deployment`.
2. Compose volumes map 1:1 to `PersistentVolumeClaim`s.
3. Caddy is replaced by an `Ingress` + cert-manager.
4. Healthchecks port to `livenessProbe` / `readinessProbe`.
5. Secrets move from host-mounted `.env` to a `Secret` (or external-secrets-operator).
6. LM Studio bottleneck — Phase 9 (vLLM in GPU pod, or fully outsourced chat).

### 2.2 Reverse proxy: Caddy in compose, not platform-managed ingress

- Navistone platform team does not have a standard ingress (DEVOPS_REQUEST §10 #1 unresolved).
- WebSocket upgrade + 60 s read timeout + Authorization passthrough are app-specific quirks.
- TLS via Caddy's built-in Let's Encrypt is one config line vs. running cert-manager out of band.

If platform later mandates corporate ingress in front: Caddy stays as L7 demux for the in-network path.

### 2.3 Secrets injection: read-only `.env` mount (this phase)

**Option B (`/etc/forge/forge.env` mounted RO)** for Phase 3:
- Zero new infrastructure dependencies.
- Compatible with the validation gate "kill VM, restore snapshot, bring up green".
- 1Password Connect adds a sidecar, network egress, service token to manage; cost not justified for 7 secrets.

File spec: path `/etc/forge/forge.env`, owner `forge:forge`, mode `0400`.

Migration to 1Password Connect when: any secret needs <90 day rotation, or count grows past ~15.

### 2.4 pyarrow: hard dependency in the Code Indexer image

**Recommendation:** pyarrow as a non-optional dep in the Dockerfile (`uv sync --extra arrow` baked in).
- 380× speedup on bulk insert is proven (`code-graph-rag/scripts/BENCH_RESULTS_2026-04-27.md`).
- `vector_store.py` already auto-detects pyarrow and falls back if missing.
- Image size impact is ~30 MB on a base of ~400 MB; immaterial.
- Indexing is the most painful UX path; making it slower in production than dev is bad polish.

The `[arrow]` extra stays in `pyproject.toml` for users who pip-install the library outside our deployment.

### 2.5 `code-graph-rag` path dep: workspace COPY in build stage

**Workspace COPY (chosen):** the GitHub Actions workflow checks out both repos to sibling directories before `docker build`. The Dockerfile uses build context `..` and copies both. Matches `pyproject.toml`'s `code-graph-rag = { path = "../code-graph-rag", editable = true }`.

### 2.6 Local-dev parity: keep `pnpm dev` + `uv run`

Compose-up is **not** the dev loop:
- TheForge dev loop is `tsx watch` + Vite HMR — losing this kills frontend iteration speed.
- `code-indexer-service` uses `uvicorn --reload`; container restart cycles are 100× slower.
- A `docker compose up` smoke test exists in CI; that's enough parity.

`pnpm dev` + `uv run` stays the dev loop. `docker-compose.dev.yml` only exists for full-stack integration smoke tests.

---

## 3. File inventory — to create / modify

### 3.1 Create

| Path | Purpose |
|---|---|
| `TheForge/Dockerfile` | multi-stage Node build → slim runtime |
| `TheForge/.dockerignore` | exclude `node_modules`, `dist`, `.forge`, `tests` |
| `code-indexer-service/Dockerfile` | multi-stage uv build → slim runtime, includes `code-graph-rag` |
| `code-indexer-service/.dockerignore` | exclude `.venv`, `.cgr`, `__pycache__`, `tests`, `service.log` |
| `deploy/forge/docker-compose.yml` | production stack (forge, code-indexer, skill-api stub, caddy) |
| `deploy/forge/Caddyfile` | TLS, routing, websocket upgrade, security headers |
| `deploy/forge/.env.example` | what `/etc/forge/forge.env` shadows |
| `deploy/forge/runbook.md` | bootstrap, update, restore procedures |
| `.github/workflows/build-and-push.yml` | GHCR build + push on `main` |

### 3.2 Modify

| Path | Change |
|---|---|
| `code-indexer-service/pyproject.toml` | document that prod image installs `[arrow]` extra |
| `TheForge/package.json` | add `start:prod` script that runs the built CLI without `tsx watch` |

---

## 4. Sample `TheForge/Dockerfile`

```dockerfile
# syntax=docker/dockerfile:1.7

FROM node:20-bookworm-slim AS builder
RUN corepack enable && corepack prepare pnpm@10.31.0 --activate
WORKDIR /app
COPY package.json pnpm-lock.yaml ./
COPY prisma ./prisma
RUN --mount=type=cache,id=pnpm,target=/root/.local/share/pnpm/store \
    pnpm install --frozen-lockfile
COPY tsconfig.json vite.config.ts eslint.config.js ./
COPY src ./src
COPY web ./web
COPY scripts ./scripts
COPY index.yaml ./
RUN pnpm run db:generate && pnpm run build && pnpm run build:web
RUN pnpm prune --prod

FROM node:20-bookworm-slim AS runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl tini && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --uid 10001 forge
WORKDIR /app
COPY --from=builder --chown=forge:forge /app/node_modules ./node_modules
COPY --from=builder --chown=forge:forge /app/dist ./dist
COPY --from=builder --chown=forge:forge /app/web/dist ./web/dist
COPY --from=builder --chown=forge:forge /app/prisma ./prisma
COPY --from=builder --chown=forge:forge /app/package.json ./package.json
USER forge
EXPOSE 3001
ENV NODE_ENV=production PORT=3001 HOST=0.0.0.0
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:3001/health || exit 1
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["node", "dist/cli/index.js", "serve", "--port", "3001", "--host", "0.0.0.0"]
```

Frontend served from `web/dist/` by Express static-file middleware. `tini` reaps Express's child processes.

---

## 5. Sample `code-indexer-service/Dockerfile`

```dockerfile
# syntax=docker/dockerfile:1.7
# Build context is the parent directory: docker build -f code-indexer-service/Dockerfile ..

FROM ghcr.io/astral-sh/uv:0.10 AS uv

FROM python:3.12-slim AS builder
COPY --from=uv /uv /uvx /bin/
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential cmake git libssl-dev zlib1g-dev libzstd-dev && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY code-graph-rag ./code-graph-rag
COPY code-indexer-service/pyproject.toml code-indexer-service/uv.lock ./code-indexer-service/
WORKDIR /workspace/code-indexer-service
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra arrow
COPY code-indexer-service/app ./app
COPY code-indexer-service/main.py ./main.py
COPY code-indexer-service/scripts ./scripts
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra arrow

FROM python:3.12-slim AS runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl tini libssl3 zlib1g libzstd1 ripgrep && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --uid 10001 forge
WORKDIR /app
COPY --from=builder --chown=forge:forge /workspace/code-indexer-service/.venv /app/.venv
COPY --from=builder --chown=forge:forge /workspace/code-indexer-service/app /app/app
COPY --from=builder --chown=forge:forge /workspace/code-indexer-service/scripts /app/scripts
COPY --from=builder --chown=forge:forge /workspace/code-indexer-service/main.py /app/main.py
COPY --from=builder --chown=forge:forge /workspace/code-graph-rag /code-graph-rag
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    LM_STUDIO_URL=http://host.docker.internal:1234 \
    CGR_DATA_DIR=/var/lib/forge/cgr \
    JOBS_DB_PATH=/var/lib/forge/jobs/jobs.sqlite
USER forge
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Notes:
- Build context is `..` so `code-graph-rag` is reachable as a sibling. CI workflow checks both repos out side-by-side.
- The editable install records the source path; we copy `code-graph-rag` into runtime at the same path. **Recommendation:** non-editable install via `uv pip install --no-deps ./code-graph-rag` to avoid path coupling.

---

## 6. Sample `deploy/forge/docker-compose.yml`

```yaml
services:
  caddy:
    image: caddy:2.8-alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy_data:/data
      - caddy_config:/config
    depends_on:
      forge-app:
        condition: service_healthy
    networks: [forge-net]

  forge-app:
    image: ghcr.io/navistone/forge-app:${FORGE_APP_TAG:-latest}
    restart: unless-stopped
    env_file: /etc/forge/forge.env
    environment:
      CODE_INDEXER_BASE_URL: http://code-indexer:8000
      SKILL_API_BASE_URL: http://skill-api:8002
      FORGE_DATABASE_URL: file:/var/lib/forge/forge/forge.db
      FORGE_AUDIT_DIR: /var/lib/forge/audit
      FORGE_UPLOAD_DIR: /var/lib/forge/uploads
    volumes:
      - /var/lib/forge/forge:/var/lib/forge/forge
      - /var/lib/forge/audit:/var/lib/forge/audit
      - /var/lib/forge/uploads:/var/lib/forge/uploads
    depends_on:
      code-indexer:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://127.0.0.1:3001/health"]
      interval: 15s
      timeout: 3s
      retries: 3
      start_period: 20s
    networks: [forge-net]

  code-indexer:
    image: ghcr.io/navistone/forge-code-indexer:${CODE_INDEXER_TAG:-latest}
    restart: unless-stopped
    env_file: /etc/forge/forge.env
    environment:
      LM_STUDIO_URL: ${LM_STUDIO_URL:-http://host.docker.internal:1234}
      CGR_DATA_DIR: /var/lib/forge/cgr
      JOBS_DB_PATH: /var/lib/forge/jobs/jobs.sqlite
    volumes:
      - /var/lib/forge/cgr:/var/lib/forge/cgr
      - /var/lib/forge/jobs:/var/lib/forge/jobs
      - ${FORGE_REPOS_DIR:-/srv/repos}:/repos:ro
    extra_hosts:
      - "host.docker.internal:host-gateway"
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://127.0.0.1:8000/health"]
      interval: 15s
      timeout: 3s
      retries: 3
      start_period: 30s
    networks: [forge-net]

  skill-api:
    image: ghcr.io/navistone/forge-skill-api:${SKILL_API_TAG:-stub}
    restart: unless-stopped
    env_file: /etc/forge/forge.env
    environment:
      LM_STUDIO_URL: ${LM_STUDIO_URL:-http://host.docker.internal:1234}
    volumes:
      - /var/lib/forge/skill:/var/lib/forge/skill
    extra_hosts:
      - "host.docker.internal:host-gateway"
    networks: [forge-net]

networks:
  forge-net:
    driver: bridge

volumes:
  caddy_data:
  caddy_config:
```

**Bind-mount strategy:** all stateful app data lives at `/var/lib/forge/{cgr,jobs,audit,uploads,forge,skill}` so the daily snapshot is one rsync target. Caddy ACME state uses named volumes.

---

## 7. Sample `deploy/forge/Caddyfile`

```caddy
{
    email zmatthews@navistone.com
}

forge.navistone.com {
    encode zstd gzip

    header {
        Strict-Transport-Security "max-age=31536000; includeSubDomains"
        X-Frame-Options "DENY"
        X-Content-Type-Options "nosniff"
        Referrer-Policy "strict-origin-when-cross-origin"
        Permissions-Policy "camera=(), microphone=(), geolocation=()"
        -Server
    }

    handle_path /api/code-indexer/* {
        reverse_proxy code-indexer:8000 {
            header_up X-Forwarded-Proto {scheme}
            header_up X-Real-IP {remote_host}
            transport http {
                read_timeout 60s
                keepalive 30s
            }
        }
    }

    handle_path /api/skill-api/* {
        reverse_proxy skill-api:8002 {
            header_up X-Forwarded-Proto {scheme}
            header_up X-Real-IP {remote_host}
            transport http { read_timeout 60s }
        }
    }

    @ws {
        path /ws /ws/*
        header Connection *Upgrade*
        header Upgrade websocket
    }
    reverse_proxy @ws forge-app:3001

    reverse_proxy forge-app:3001 {
        header_up X-Forwarded-Proto {scheme}
        header_up X-Real-IP {remote_host}
        transport http { read_timeout 60s }
    }

    log {
        output stdout
        format json
    }
}
```

`Authorization` header forwarded by default; bearer tokens reach upstream untouched.

---

## 8. CI/CD — `.github/workflows/build-and-push.yml`

```yaml
name: build-and-push
on:
  push:
    branches: [main]
  workflow_dispatch:
env:
  REGISTRY: ghcr.io
  ORG: navistone

jobs:
  forge-app:
    runs-on: ubuntu-latest
    permissions: { contents: read, packages: write }
    steps:
      - uses: actions/checkout@v4
        with: { path: TheForge }
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.ORG }}/forge-app
          tags: |
            type=sha,prefix=,format=short
            type=raw,value=latest,enable={{is_default_branch}}
      - uses: docker/build-push-action@v5
        with:
          context: TheForge
          file: TheForge/Dockerfile
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  code-indexer:
    runs-on: ubuntu-latest
    permissions: { contents: read, packages: write }
    steps:
      - uses: actions/checkout@v4
        with: { repository: navistone/code-indexer-service, path: code-indexer-service }
      - uses: actions/checkout@v4
        with:
          repository: navistone/code-graph-rag
          path: code-graph-rag
          ref: ${{ vars.CODE_GRAPH_RAG_REF || 'main' }}
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.ORG }}/forge-code-indexer
          tags: |
            type=sha,prefix=,format=short
            type=raw,value=latest,enable={{is_default_branch}}
      - uses: docker/build-push-action@v5
        with:
          context: .
          file: code-indexer-service/Dockerfile
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

**Image immutability:** every push tags both `<sha>` and `latest`. Production pulls `<sha>` (pinned in `/etc/forge/forge.env`); `latest` exists for ad-hoc smoke-tests, never wired into a running deploy.

---

## 9. Deployment runbook (`deploy/forge/runbook.md`)

### 9.1 First-time bootstrap (fresh VM)

1. Platform team provisions VM per DEVOPS_REQUEST §3a.
2. `apt-get install docker-ce docker-compose-plugin`.
3. Create data tree: `sudo install -d -o 10001 -g 10001 /var/lib/forge/{cgr,jobs,audit,uploads,forge,skill}`.
4. Drop `forge.env` at `/etc/forge/forge.env` (mode 0400, owned by uid 10001).
5. `sudo install -d /srv/repos` and clone team repo registry into it.
6. `git clone github.com/navistone/forge-deploy /opt/forge`.
7. `docker login ghcr.io -u <gh-deploy-bot> -p <token>`.
8. `cd /opt/forge && docker compose pull && docker compose up -d`.
9. Validate: `curl -sf https://forge.navistone.com/api/code-indexer/health` returns 401, then with bearer returns 200.

### 9.2 Rolling update (image bump)

```
cd /opt/forge
git pull
docker compose pull
docker compose up -d
docker image prune -f
```

Healthchecks gate the new container — Caddy keeps routing to the old until new is healthy.

### 9.3 Restore from snapshot

1. New VM, repeat §9.1 steps 1–3, 5–7. Skip step 4 (comes from snapshot).
2. Restore `/var/lib/forge` and `/etc/forge/forge.env` from latest snapshot.
3. `docker compose up -d`.
4. Validate as in §9.1 step 9.

**Validation gate for Phase 3:** §9.3 must complete to a green `/health` within **5 minutes** on a fresh VM.

---

## 10. Trade-offs explicitly captured

| Trade-off | Decision | Defer/migrate trigger |
|---|---|---|
| Single VM vs k8s | Single VM via compose | >50 users, multi-region, or platform team adopts k8s |
| Caddy-in-compose vs platform ingress | Caddy in compose | Platform mandates corporate ingress |
| Mounted `.env` vs 1Password Connect | Mounted RO `.env` | Any secret <90d rotation, or count exceeds ~15 |
| pyarrow extra vs hard dep | Hard dep in image | Never — speedup too large to leave optional in prod |
| `code-graph-rag` clone-in-build vs workspace COPY | Workspace COPY | If repos diverge to separate organisations |
| Compose for local dev vs `pnpm dev` + `uv run` | `pnpm dev` + `uv run` | Never — HMR and uvicorn-reload non-negotiable |
| Single forge-app container vs separate nginx | Single container | Scaling frontend reads independently |
| LM Studio in compose vs host | Host (`host.docker.internal:1234`) | Phase 9 — vLLM in GPU container |

---

## 11. Validation gates (Phase 3 done = all of these pass)

1. `docker compose up -d` from fresh checkout brings stack up green within 5 minutes.
2. `curl https://forge.navistone.com/api/code-indexer/health` returns 401 without bearer (Phase 1) and 200 with one.
3. WebSocket: `wscat -c wss://forge.navistone.com/ws -H "Authorization: Bearer <t>"` connects, receives `index_progress` events.
4. Snapshot/restore drill: kill VM, restore `/var/lib/forge` to new VM, `docker compose up -d`, jobs from dead host appear in `/index/jobs/list?scope=mine` with status `interrupted` (Phase 2 boot-cleanup).
5. CI: PR to `main` produces `<sha>`-tagged image in GHCR for both within 8 minutes.
6. Image size: `forge-app` < 600 MB compressed, `forge-code-indexer` < 1.4 GB compressed.
7. `docker compose logs caddy | grep -i error` empty after 10 minutes of normal traffic.

---

## 12. Critical Files for Implementation

- `code-indexer-service/Dockerfile`
- `TheForge/Dockerfile`
- `deploy/forge/docker-compose.yml`
- `deploy/forge/Caddyfile`
- `.github/workflows/build-and-push.yml`

### Reference files

- `code-indexer-service/pyproject.toml` — confirms `code-graph-rag` path dep
- `code-graph-rag/Dockerfile` — existing pattern for uv multi-stage
- `code-graph-rag/codebase_rag/storage/vector_store_arrow.py` — confirms pyarrow auto-detect
- `TheForge/deploy/staging/docker-compose.staging.yaml` — prior-art layout
- `code-indexer-service/.planning/DEVOPS_REQUEST.md` — §3 host sizing, §3b volume layout, §6 LM Studio
