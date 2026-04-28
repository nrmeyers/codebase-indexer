# DevOps / Platform Team Request — Code Indexer + TheForge Team Deployment

**Owner:** Zachary Matthews (zmatthews@navistone.com)
**Date:** 2026-04-27
**Target host:** `forge.navistone.com` (single shared environment, ~10–25 internal users)
**Companion docs:** `.planning/TEAM_DEPLOYMENT_PLAN.md`, `.planning/FRONTEND_AGENT_BRIEF.md`

This is the exact list of asks for the platform / DevOps team. Engineering will
own application code and Docker images; this document captures everything the
app team **cannot** do without elevated tenancy/network/DNS access.

> Anywhere you see `<DEVOPS-PROVIDED>` in this document, that value is what the
> platform team needs to supply back to engineering (over a secure channel —
> 1Password vault `forge-prod` is preferred). The same placeholders appear in
> `.env.example` files in both repos.

---

## 1. Microsoft Entra ID (Azure AD) — two app registrations

We are using M365 OAuth2 / OIDC for sign-in. Two app registrations are needed
because the SPA (TheForge frontend) and the resource API (Code Indexer + Forge
backend) have different trust models. Single-tenant — Navistone tenant only.

### 1a. SPA app: `forge-frontend`

| Field | Value |
|---|---|
| Name | `forge-frontend` |
| Supported account types | Single tenant (Navistone only) |
| Platform | **Single-page application** |
| Redirect URIs | `https://forge.navistone.com/auth/m365/callback`<br>`http://localhost:3000/auth/m365/callback` (dev) |
| Front-channel logout URL | `https://forge.navistone.com/auth/m365/logout` |
| Implicit grant | **Disabled** (PKCE only — @azure/msal-browser handles this) |
| API permissions (delegated) | `openid`, `profile`, `email`, `User.Read`, `GroupMember.Read.All` |
| Admin consent | Granted by tenant admin for `GroupMember.Read.All` |

**Return to engineering:**
- `AZURE_FRONTEND_CLIENT_ID` = `<DEVOPS-PROVIDED>`
- `AZURE_TENANT_ID` = `<DEVOPS-PROVIDED>`

### 1b. Resource API app: `forge-api`

| Field | Value |
|---|---|
| Name | `forge-api` |
| Supported account types | Single tenant |
| Application ID URI | `api://forge-api` (or whatever Entra assigns — record it) |
| Exposed scope | `code-indexer.access` (scope name; admin + user consent) |
| Pre-authorised client | `forge-frontend` (the SPA above) for that scope |
| App roles | `Architect`, `ProductManager`, `TechLead`, `Developer`, `Reader` (used for backend RBAC) |

**Return to engineering:**
- `AZURE_API_CLIENT_ID` = `<DEVOPS-PROVIDED>`
- `AZURE_API_AUDIENCE` = `api://forge-api` (the App ID URI)
- JWKS URI = `https://login.microsoftonline.com/<TENANT_ID>/discovery/v2.0/keys` (derived)

### 1c. Entra security groups → Forge roles

Create these groups in the Navistone tenant and map them to Forge roles. The
backend reads the group OIDs from the access token's `groups` claim (the SPA
must request the `GroupMember.Read.All` scope; alternatively use group claims
in the token configuration). **Send the OIDs back, not just the names** — names
are not stable identifiers.

| Group name | Forge role | Group OID (return to eng) |
|---|---|---|
| `forge-admin` | `admin` | `<DEVOPS-PROVIDED>` |
| `forge-architect` | `architect` | `<DEVOPS-PROVIDED>` |
| `forge-pm` | `product_manager` | `<DEVOPS-PROVIDED>` |
| `forge-techlead` | `tech_lead` | `<DEVOPS-PROVIDED>` |
| `forge-dev` | `developer` | `<DEVOPS-PROVIDED>` |
| `forge-reader` | `reader` (read-only) | `<DEVOPS-PROVIDED>` |

Initial membership: at minimum, add Zachary Matthews to `forge-admin` and the
core engineering team (~6 people) to `forge-dev`. PM/architect/techlead roles
are populated by the relevant managers as they onboard.

### 1d. Token configuration

Configure the API app to emit:
- `groups` claim with **Group ID** (OID), not sAMAccountName
- `roles` claim from app role assignments (if used)
- `email`, `name`, `oid`, `preferred_username` in ID token

If the user is in more than ~150 groups the `groups` claim is replaced with a
`_claim_names` overage indicator, in which case the backend falls back to a
Microsoft Graph call. **Confirm whether anyone in the org will hit the
overage limit** — most internal users will not.

---

## 2. DNS, TLS, and reverse proxy

### 2a. DNS

A single record:

| Type | Host | Target |
|---|---|---|
| A or CNAME | `forge.navistone.com` | the VM / Kubernetes ingress public IP |

If we land on Kubernetes later, `*.forge.navistone.com` would let us peel
services off subpaths. For now, single host is fine.

### 2b. TLS

- Wildcard or per-host cert for `forge.navistone.com`
- Auto-renewal (Let's Encrypt via cert-manager, or whatever Navistone standard is)
- TLS 1.2+ only, HSTS enabled, `X-Frame-Options: DENY`

### 2c. Reverse proxy

The reverse proxy terminates TLS and routes all traffic to a single TheForge
container, which itself proxies `/api/code-indexer/*` and `/api/skill-api/*` to
the local sidecar containers. Engineering will provide a working Caddyfile or
nginx.conf in the deployment repo — DevOps just needs to host the proxy.

Required behaviour:
- Forward `Authorization` header untouched (bearer token must reach backend)
- WebSocket upgrade for `/ws` (TheForge real-time activity feed)
- 60 s read timeout (background indexing jobs stream progress events)
- Strip `X-Forwarded-*` from client and re-inject from proxy

Inbound ports needed open from the corporate network: **443 only**.

---

## 3. Container host and persistent storage

### 3a. Host

A single Linux VM (Ubuntu 22.04 LTS or RHEL 9) is sufficient for the initial
team-of-25 footprint. Recommended size: **8 vCPU, 32 GB RAM, 500 GB SSD**.
Reasoning:
- LM Studio + qwen3.6-27b model needs ~20 GB RAM resident
- DuckDB embeddings file grows ~6 KB per indexed symbol; 500k symbols ≈ 3 GB
- LadybugDB graph file grows ~2 KB per symbol; 500k symbols ≈ 1 GB
- Plus Docker overhead, OS, headroom for `pnpm dev`-style rebuilds during ops

If LM Studio runs on a separate workstation (current dev setup), drop RAM
requirement to 16 GB and CPU to 4 vCPU.

### 3b. Persistent volumes

Mount these into the containers (engineering will define exact paths in
`docker-compose.yml`):

| Mount point inside container | Purpose | Backup priority |
|---|---|---|
| `/var/lib/forge/cgr` | LadybugDB graph + DuckDB embeddings (one subdir per repo) | **HIGH** — re-indexing 500k symbols takes ~45 min |
| `/var/lib/forge/jobs` | SQLite job-state DB (Phase 2 of plan) | **HIGH** — in-flight indexing jobs |
| `/var/lib/forge/audit` | Forge audit trail JSONL | **HIGH** — compliance |
| `/var/lib/forge/uploads` | Spec attachments | medium |
| `/var/log/forge` | Structured logs (if not piping to stdout) | low |

Daily snapshot of `/var/lib/forge` to off-host storage is sufficient. RPO 24h,
RTO 2h is acceptable for an internal dev tool.

### 3c. Network access from the host

- **Outbound HTTPS to login.microsoftonline.com** (OAuth)
- **Outbound HTTPS to api.github.com** (spec sync, PR creation)
- **Outbound HTTPS to api.linear.app** (Linear hub feature)
- **Outbound HTTPS to api.anthropic.com** (Claude — primary LLM)
- **Outbound to LM Studio host** — see §6 below if LM Studio is on a separate machine

No inbound from the public internet other than 443 → reverse proxy.

---

## 4. Secrets storage and rotation

We do not want secrets baked into the Docker image or committed to the
deployment repo. Two patterns work:

**Option A (preferred): 1Password Connect / Doppler / Vault**
- Engineering uses the platform team's standard secret-injection mechanism
- Secrets are pulled into the container at start-up, never written to disk

**Option B: encrypted `.env` mounted read-only**
- Platform team puts the `.env` on the host at `/etc/forge/forge.env`
- Mounted as `:ro` into the container
- File mode `0400`, owned by the forge service user

**Secrets that need to land in the container:**

| Variable | Source | Rotation cadence |
|---|---|---|
| `AZURE_API_CLIENT_SECRET` | Entra app `forge-api` → Certificates & secrets | every 180 days |
| `AZURE_FRONTEND_CLIENT_ID` | Entra (not secret, but tenant-specific) | n/a |
| `AZURE_TENANT_ID` | Entra | n/a |
| `GITHUB_TOKEN` | Org-level PAT or GitHub App private key | every 90 days |
| `LINEAR_API_KEY` | Linear admin → API keys | every 90 days |
| `ANTHROPIC_API_KEY` | console.anthropic.com (corp account) | every 90 days |
| `FORGE_SERVICE_AUTH_TOKEN` | `openssl rand -base64 32` | every 90 days, rolling |

The `FORGE_SERVICE_AUTH_TOKEN` is shared between TheForge and the Code Indexer
container so the sidecars trust each other. It must be **identical in both
containers**.

---

## 5. Grafana / Prometheus

We use Grafana already (per Zach). Engineering will expose Prometheus metrics
on:

| Service | Endpoint |
|---|---|
| Code Indexer | `http://code-indexer:8000/metrics` |
| TheForge backend | `http://forge:3001/metrics` |
| Skill API | `http://skill-api:8002/metrics` |

**Ask of platform team:** add a Prometheus scrape target (or update the central
Grafana Agent / Alloy config) that polls these three endpoints every 15 s.
Engineering will provide the dashboard JSON in the deployment repo
(`grafana/forge-dashboard.json`) covering:
- Index job duration p50/p95
- Search latency p50/p95 (semantic / structural / symbol)
- Rerank gate hit rate
- LM Studio adapter availability
- Container CPU / memory / disk per service

Alerts (low priority, internal tool — page nobody, just notify Slack
`#forge-alerts`):
- Code Indexer 5xx rate > 5% over 5 min
- Index job stuck (no progress event > 10 min)
- LM Studio adapter unreachable > 5 min
- Disk usage on `/var/lib/forge` > 80%

---

## 6. LM Studio host (the awkward bit)

LM Studio is the local LLM runner that hosts CodeRankEmbed (query-time
embedder) and a chat model (currently qwen3.6-35b-a3b — see caveat below). It
is **not yet container-friendly** — it runs as a desktop app with a GPU
attached.

**Two viable topologies:**

**A. LM Studio on the same VM (simplest, GPU-bound):**
- Requires the VM to have a GPU (24 GB+ VRAM for qwen3.6-27b, ~64 GB unified
  memory on Apple Silicon for qwen3.6-35b-a3b MoE)
- LM Studio listens on `:1234` on `localhost`
- Container reaches it via `host.docker.internal:1234` (Docker Desktop) or
  the host bridge IP on Linux Docker

**B. LM Studio on a workstation, container on the VM:**
- Whichever workstation hosts LM Studio must be reachable from the VM on TCP 1234
- Firewall rule: VM → workstation:1234 (TCP, allow)
- Engineering sets `LM_STUDIO_URL=http://<workstation-host>:1234` in the env
- Less reliable (workstation has to stay on); useful as bridge while we
  productionise the LLM stack

**Memory caveat — please pass to whoever loads the model:** the dev Mac (M-class,
~36 GB unified memory) cannot fit `qwen/qwen3.6-27b` (requires ~18.84 GB free
working set on top of the OS + LM Studio runtime); it must run
`qwen/qwen3.6-35b-a3b` instead (MoE, fits because only ~3 B params are active
per token). On a beefier host, prefer 27 b. The Code Indexer adapter resolves
whichever model is loaded by substring match — engineering does not need to
hard-code the model id.

This is the **one piece of the deployment that is not yet first-class
container infra.** Phase 9 of the deployment plan is to either (a) replace LM
Studio with vLLM in a GPU container, or (b) move all chat to Anthropic and
keep only embedding local. Decision deferred until we see real load.

---

## 7. CI/CD and image registry

If Navistone has a standard container registry (likely GHCR or ACR):
- Create a repo: `navistone/forge-app` and `navistone/forge-code-indexer`
- Service account / deploy key with push access for the GitHub Actions runner
- Pull access for the deployment host (registry credential mounted into the
  Docker daemon)

If we deploy via plain `docker compose pull && docker compose up -d`, no CI/CD
work is needed — engineering will SSH-deploy initially and migrate to a proper
GitOps flow in Phase 7.

---

## 8. Backup and restore

Daily off-host snapshot of `/var/lib/forge`. That is the entire stateful
surface — kill the VM, spin up a fresh one, restore the volume, run
`docker compose up -d`, and the system is back. No external database to
restore.

Test the restore quarterly. Engineering will own the restore runbook.

---

## 9. Summary — what engineering needs back from you

In a 1Password vault item titled `forge-prod`, please supply:

```
AZURE_TENANT_ID=<DEVOPS-PROVIDED>
AZURE_FRONTEND_CLIENT_ID=<DEVOPS-PROVIDED>
AZURE_API_CLIENT_ID=<DEVOPS-PROVIDED>
AZURE_API_AUDIENCE=api://forge-api  (or whatever Entra assigns)
AZURE_API_CLIENT_SECRET=<DEVOPS-PROVIDED>

# Group OIDs (NOT names — OIDs are stable, names aren't)
AZURE_GROUP_ADMIN_OID=<DEVOPS-PROVIDED>
AZURE_GROUP_ARCHITECT_OID=<DEVOPS-PROVIDED>
AZURE_GROUP_PM_OID=<DEVOPS-PROVIDED>
AZURE_GROUP_TECHLEAD_OID=<DEVOPS-PROVIDED>
AZURE_GROUP_DEV_OID=<DEVOPS-PROVIDED>
AZURE_GROUP_READER_OID=<DEVOPS-PROVIDED>

# Confirmation that:
#   - DNS A record forge.navistone.com → <VM IP> is live
#   - TLS cert is provisioned
#   - VM has 8 vCPU / 32 GB RAM / 500 GB SSD (or whatever you provisioned)
#   - /var/lib/forge volume is mounted with daily snapshot enabled
#   - Prometheus is scraping the three /metrics endpoints
```

That's everything app team needs to flip Phase 1 of the deployment plan from
"blocked" to "in flight."

---

## 10. Open questions to confirm with platform team

1. Is there a Navistone-standard reverse proxy preference (nginx vs Caddy vs
   Traefik)? Engineering will adapt.
2. Is there a preferred secret-injection mechanism (1Password Connect, Doppler,
   Vault, plain mounted `.env`)?
3. Will the LM Studio host be the same VM, or a separate workstation? (See §6.)
4. Confirm the ~150-group overage threshold isn't a problem for any user
   (alternative: switch to app roles instead of groups, slightly more work).
5. Backup retention policy — 7 days, 30 days, 90 days?
6. Any existing Grafana folder convention we should drop the dashboard into?

Answers to these unblock the Phase 1 production cutover.
