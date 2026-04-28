# Phase 1 — M365 / Entra ID OAuth Integration Plan

**Owner:** Zachary Matthews (zmatthews@navistone.com)
**Date:** 2026-04-28
**Target host:** `forge.navistone.com`
**Status:** ready to execute once DevOps returns the values flagged in `.planning/DEVOPS_REQUEST.md` §9
**Companion docs:** `TEAM_DEPLOYMENT_PLAN.md` §3 Phase 1, `DEVOPS_REQUEST.md` §1, `FRONTEND_AGENT_BRIEF.md` §6

> NOTE: this plan supersedes the looser Phase-1 sketch in `TEAM_DEPLOYMENT_PLAN.md` §3.
> The two-app-registration model from `DEVOPS_REQUEST.md` §1 is now canonical;
> the existing `TheForge/src/services/m365-auth.ts` (single-app, Graph-based)
> stays as the popup/legacy path but is no longer the production identity source.

---

## 0. Goals and non-goals

**Goals**
1. Every non-`/health` request to TheForge backend (`:3001`) and the Code Indexer (`:8000`) is authenticated against Microsoft Entra ID.
2. The SPA at `forge.navistone.com` performs interactive PKCE login via `@azure/msal-browser` and acquires a token whose audience is `api://forge-api`.
3. Backend services validate the JWT signature against the tenant JWKS, check audience / issuer / expiry, and resolve the caller's Forge role from the `groups` claim using the `AZURE_GROUP_*_OID` env vars.
4. WebSocket upgrades carry the bearer in the `Sec-WebSocket-Protocol` subprotocol header (browser limitation).
5. `FORGE_ENV=development` keeps the existing dev identity flow working end-to-end with no MSAL round-trip required.

**Non-goals**
- Service-principal client-credentials flow for TheForge → Code Indexer (we keep the pre-existing `FORGE_SERVICE_AUTH_TOKEN` shared-secret pattern from `service-auth.ts`; both services validate the *user* JWT independently).
- App-roles RBAC inside the API token (we use group OIDs; app roles deferred unless §1d overage forces a switch).
- oauth2-proxy at the reverse proxy (see Trade-off A below — rejected for this phase).
- Rotating signing keys for the shared service token (Phase 7 cleanup).

---

## 1. Architecture overview

```
Browser (SPA, MSAL)
  │  redirect-flow PKCE login  ──►  login.microsoftonline.com/<tenant>
  │  ◄──  id_token + access_token (aud=api://forge-api)
  │
  ▼  Authorization: Bearer <access_token>
forge.navistone.com  (reverse proxy, TLS, forwards Authorization)
  │
  ▼
TheForge Express :3001
  ├─ requireBearer middleware  (validates JWT, attaches req.principal)
  ├─ /api/code-indexer/*  ──►  Code Indexer :8000
  │     forwards user bearer + adds X-Forge-Service-Auth shared secret
  │
  ▼
Code Indexer FastAPI :8000
  └─ Depends(verify_bearer)  (revalidates the same JWT against JWKS)
```

Both services validate independently. The Code Indexer does NOT trust TheForge to vouch for the user — it re-checks the bearer. The shared `FORGE_SERVICE_AUTH_TOKEN` is layered on top to protect the indexer from being called from anywhere except a known TheForge instance (defence-in-depth).

---

## 2. Frontend SPA login flow (TheForge web)

### 2.1 Files to create

- `web/src/auth/msal.ts` — MSAL config + `PublicClientApplication` singleton.
- `web/src/auth/MsalAuthProvider.tsx` — wraps app root with `MsalProvider`, performs `handleRedirectPromise` on mount, exposes `useToken()` hook.
- `web/src/auth/useToken.ts` — `acquireTokenSilent` → fallback `acquireTokenRedirect` for the `api://forge-api/.default` audience.
- `web/src/pages/AuthCallback.tsx` — route component for `/auth/m365/callback` (just renders a loading spinner; MSAL processes the redirect in the provider).
- `web/src/pages/Login.tsx` — minimal "Sign in with Microsoft" button (only shown when MSAL has no account).

### 2.2 Files to modify

- `web/src/main.tsx` — wrap `<App />` in `<MsalAuthProvider>` (gated by `VITE_AUTH_MODE`).
- `web/src/adapters/identity/azuread-provider.ts` — replace stub with a real implementation that pulls `account.idTokenClaims` from MSAL and maps `groups` → role using the same OIDs the backend uses (frontend just for UX gating; backend remains source of truth).
- `web/src/components/code-indexer/api.ts` (and any other fetch wrappers) — accept an optional `tokenAcquirer: () => Promise<string | null>` and inject `Authorization: Bearer …`. The §6 brief already mandates this shape.
- `web/src/hooks/useIdentity.ts` — when `VITE_AUTH_MODE === 'msal'`, derive `actor.id`, `actor.email`, `role` from MSAL claims; treat `roleIsLocked = true`.

### 2.3 MSAL config (production values)

```ts
// web/src/auth/msal.ts
import { PublicClientApplication, type Configuration } from '@azure/msal-browser';

export const msalConfig: Configuration = {
  auth: {
    clientId: import.meta.env.VITE_AZURE_FRONTEND_CLIENT_ID,
    authority: `https://login.microsoftonline.com/${import.meta.env.VITE_AZURE_TENANT_ID}`,
    redirectUri: import.meta.env.VITE_AZURE_REDIRECT_URI,
    postLogoutRedirectUri: '/',
    navigateToLoginRequestUrl: true,
  },
  cache: {
    cacheLocation: 'sessionStorage',  // sessionStorage > localStorage for XSS resistance
    storeAuthStateInCookie: false,
  },
};

export const apiTokenRequest = {
  scopes: [`${import.meta.env.VITE_AZURE_API_AUDIENCE}/.default`],  // api://forge-api/.default
};

export const graphTokenRequest = {
  scopes: ['User.Read', 'GroupMember.Read.All'],  // ONLY when we need Graph for overage
};

export const pca = new PublicClientApplication(msalConfig);
```

### 2.4 Token acquisition pattern

```ts
// web/src/auth/useToken.ts
export function useApiToken(): () => Promise<string | null> {
  const { instance, accounts } = useMsal();
  return useCallback(async () => {
    if (import.meta.env.VITE_AUTH_MODE !== 'msal') return null;
    if (!accounts[0]) return null;
    try {
      const r = await instance.acquireTokenSilent({ ...apiTokenRequest, account: accounts[0] });
      return r.accessToken;
    } catch (e) {
      if (e instanceof InteractionRequiredAuthError) {
        await instance.acquireTokenRedirect(apiTokenRequest);
      }
      return null;
    }
  }, [instance, accounts]);
}
```

Silent refresh is automatic — MSAL uses the hidden iframe + refresh token; we only fall back to interactive on `InteractionRequiredAuthError` (CA policy change, MFA prompt, etc.).

### 2.5 Trade-off (c) — separate audiences for API vs. Graph

The SPA needs **two different access tokens**:
- `api://forge-api/.default` — for every call to TheForge / Code Indexer.
- `User.Read` / `GroupMember.Read.All` (Graph audience `https://graph.microsoft.com`) — only used as the overage fallback in §4.3.

MSAL handles this natively: each `acquireTokenSilent({ scopes })` request returns a token correctly scoped to the implied resource. **Do NOT mix scopes from different resources in a single request** — that fails with `AADSTS28000`. The `useApiToken()` hook above only ever requests the API scope; the (rarely-used) Graph token is acquired in a separate hook only when the backend signals overage.

### 2.6 WebSocket auth

```ts
// useWebSocket helper change
const token = await getToken();
const ws = new WebSocket(url, token ? ['bearer', token] : undefined);
```

Server-side, the FastAPI `/ws` route reads `websocket.headers["sec-websocket-protocol"]` (already split on comma) and treats the second token as the bearer.

---

## 3. Backend JWT validation — TheForge (Express)

### 3.1 New file: `src/services/jwt-validator.ts`

A tiny module owning JWKS fetch + caching, audience/issuer check, and group-OID → role mapping. Returns `Result<Principal, ErrorEnvelope>` per the project's no-throws-from-services rule.

```ts
import { jwtVerify, createRemoteJWKSet } from 'jose';  // already MIT, in pnpm

interface Principal {
  oid: string;          // user object ID (stable)
  email: string;        // upn / email claim
  name: string;
  tenantId: string;
  groups: string[];     // group OIDs
  forgeRole: string;    // resolved from groups
  hasGroupOverage: boolean;
}

const JWKS = createRemoteJWKSet(
  new URL(`https://login.microsoftonline.com/${process.env.AZURE_TENANT_ID}/discovery/v2.0/keys`),
  { cooldownDuration: 30_000, cacheMaxAge: 600_000 },  // 10 min cache, 30 s cooldown on miss
);

export async function validateBearer(token: string): Promise<Result<Principal, ErrorEnvelope>> {
  const expectedAud = process.env.AZURE_API_AUDIENCE ?? 'api://forge-api';
  const expectedIss = `https://sts.windows.net/${process.env.AZURE_TENANT_ID}/`;  // v1
  // Accept BOTH v1 and v2 issuers — Entra emits v2 by default but legacy clients may get v1.

  try {
    const { payload } = await jwtVerify(token, JWKS, {
      audience: expectedAud,
      issuer: [expectedIss, `https://login.microsoftonline.com/${process.env.AZURE_TENANT_ID}/v2.0`],
      algorithms: ['RS256'],
    });
    const groups = (payload.groups as string[]) ?? [];
    const overage = !!(payload as any)._claim_names?.groups;
    const role = resolveRole(groups);
    return Ok({
      oid: payload.oid as string,
      email: (payload.email ?? payload.preferred_username) as string,
      name: payload.name as string,
      tenantId: payload.tid as string,
      groups,
      forgeRole: role,
      hasGroupOverage: overage,
    });
  } catch (e) {
    return Err(createErrorEnvelope('invalid_bearer', e instanceof Error ? e.message : String(e)));
  }
}
```

### 3.2 New file: `src/services/group-role-map.ts`

```ts
const GROUP_OID_ROLE: Record<string, string> = {
  [process.env.AZURE_GROUP_ADMIN_OID ?? '']: 'admin',
  [process.env.AZURE_GROUP_ARCHITECT_OID ?? '']: 'architect',
  [process.env.AZURE_GROUP_PM_OID ?? '']: 'product_manager',
  [process.env.AZURE_GROUP_TECHLEAD_OID ?? '']: 'tech_lead',
  [process.env.AZURE_GROUP_DEV_OID ?? '']: 'developer',
  [process.env.AZURE_GROUP_READER_OID ?? '']: 'reader',
};
const ROLE_PRIORITY = ['admin', 'architect', 'tech_lead', 'product_manager', 'developer', 'reader'];

export function resolveRole(groupOids: string[]): string {
  const matched = groupOids.map(g => GROUP_OID_ROLE[g]).filter(Boolean);
  return ROLE_PRIORITY.find(r => matched.includes(r)) ?? 'reader';
}
```

### 3.3 New file: `src/services/routes/require-bearer.ts`

Express middleware that:
1. Skips `/health`, `/metrics`, `/api/auth/m365/*`.
2. In dev (`FORGE_ENV !== 'production'` and no `Authorization` header) seeds a synthetic `req.principal` from `VITE_DEV_USER` / `VITE_DEV_ROLE`.
3. Otherwise validates and attaches `req.principal`. On failure returns `401` with the `ErrorEnvelope`.

### 3.4 Modify `src/services/api-server.ts`

- Mount `requireBearer` after the existing `/metrics` and `/api/auth` routes (line ~350) and before `code-indexer` proxy / other `/api/*` routes.
- The proxy code already forwards `Authorization`; verify with a unit test.

---

## 4. Backend JWT validation — Code Indexer (FastAPI)

### 4.1 New file: `app/auth.py`

```python
import os, time
from typing import Annotated, Any
from fastapi import Depends, HTTPException, Request, status
from jose import jwt
from jose.exceptions import JWTError
import httpx

_JWKS_CACHE: dict[str, Any] = {"keys": None, "fetched_at": 0.0}
_JWKS_TTL = 600  # 10 min

async def _jwks() -> dict:
    now = time.time()
    if _JWKS_CACHE["keys"] is None or now - _JWKS_CACHE["fetched_at"] > _JWKS_TTL:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(
                f"https://login.microsoftonline.com/{os.environ['AZURE_TENANT_ID']}/discovery/v2.0/keys"
            )
            r.raise_for_status()
            _JWKS_CACHE["keys"] = r.json()
            _JWKS_CACHE["fetched_at"] = now
    return _JWKS_CACHE["keys"]

async def verify_bearer(request: Request) -> dict:
    if os.getenv("FORGE_ENV", "development") != "production":
        if not request.headers.get("authorization"):
            return {"oid": "dev", "email": "dev@local", "forge_role": "developer", "groups": []}

    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail={"error": "missing_bearer"})
    token = auth.split(" ", 1)[1]
    try:
        unverified_header = jwt.get_unverified_header(token)
        keys = (await _jwks())["keys"]
        rsa_key = next((k for k in keys if k["kid"] == unverified_header["kid"]), None)
        if not rsa_key:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail={"error": "unknown_kid"})
        payload = jwt.decode(
            token, rsa_key, algorithms=["RS256"],
            audience=os.environ["AZURE_API_AUDIENCE"],
            issuer=[
                f"https://sts.windows.net/{os.environ['AZURE_TENANT_ID']}/",
                f"https://login.microsoftonline.com/{os.environ['AZURE_TENANT_ID']}/v2.0",
            ],
        )
    except JWTError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail={"error": "invalid_bearer", "message": str(e)})

    groups = payload.get("groups", [])
    return {
        "oid": payload["oid"],
        "email": payload.get("email") or payload.get("preferred_username"),
        "name": payload.get("name"),
        "groups": groups,
        "has_group_overage": bool(payload.get("_claim_names", {}).get("groups")),
        "forge_role": _resolve_role(groups),
    }

PrincipalDep = Annotated[dict, Depends(verify_bearer)]
```

### 4.2 New file: `app/group_role_map.py`

Mirror of the TS module. Reads the same `AZURE_GROUP_*_OID` env vars; deliberate duplication called out in Trade-off (3).

### 4.3 Trade-off (b) — group overage handling

Three options:

| Option | Effort | Pros | Cons |
|--------|-------:|------|------|
| **Switch to App Roles** | Med | No Graph round-trip, role-based claim is always present and small | Group-of-groups inheritance is lost; admins must assign each user to roles directly |
| **Graph fallback in backend** | Low-med | Reuses groups; stays self-service for tenant admins | Requires the API app to hold a delegated token; Graph latency per overage user (~200 ms cached) |
| **Reject overage tokens** | Low | Trivial code | Breaks for any user in >150 groups |

**Recommendation:** ship "reject" first behind a feature flag (`FORGE_OVERAGE_POLICY=reject|graph|app-roles`), and add the Graph fallback only if DevOps confirms a real user hits it.

If we must implement Graph fallback: backend uses **on-behalf-of (OBO) flow** with `AZURE_API_CLIENT_SECRET` to exchange the user token for a Graph token, then calls `GET /users/{oid}/getMemberObjects` (returns OIDs only). Cache by `oid` for 30 min in-process.

### 4.4 Modify `app/main.py`

```python
app.include_router(health.router)  # public
app.include_router(repos.router, dependencies=[Depends(verify_bearer)])
app.include_router(index.router, dependencies=[Depends(verify_bearer)])
app.include_router(search.router, dependencies=[Depends(verify_bearer)])
app.include_router(context_bundle.router, dependencies=[Depends(verify_bearer)])
app.include_router(symbols.router, dependencies=[Depends(verify_bearer)])
app.include_router(explorer.router, dependencies=[Depends(verify_bearer)])
app.include_router(disk.router, dependencies=[Depends(verify_bearer)])
app.include_router(github.router, dependencies=[Depends(verify_bearer)])
app.include_router(websocket.router)  # WS handles auth in-handler (subprotocol)
```

Modify `app/routers/websocket.py` to read the bearer from `websocket.headers["sec-websocket-protocol"]` (split, take the second item), then call `verify_bearer` manually before `await websocket.accept(subprotocol="bearer")`.

### 4.5 Trade-off (3) — duplicate vs. shared module

**Recommendation: copy** (one Python file, one TS file). Different runtimes, ~80 lines per side; the maintenance burden of two copies is lower than introducing a new shared library. The contract that *must* stay in sync is the env-var schema, documented in both `.env.example` files.

---

## 5. Migration plan for the existing dev identity

1. **Add `VITE_AUTH_MODE` env var** — values `dev` (default), `msal`. Selects which `IdentityProvider` to mount.
2. **Implement `azuread-provider.ts`** — replace the stub with a provider that reads from MSAL accounts; returns `roleIsLocked=true`.
3. **Keep `manual-provider.ts` unchanged** — the dev flow continues to work for `VITE_AUTH_MODE=dev` and `FORGE_ENV=development`.
4. **Backend dev mode** — both `requireBearer` (TS) and `verify_bearer` (Py) check `FORGE_ENV !== 'production'` and synthesise a principal from a header (`X-Dev-User`, `X-Dev-Role`) or env vars.
5. **Production refusal** — when `FORGE_ENV=production`, the backend MUST refuse to start if `AZURE_TENANT_ID` / `AZURE_API_AUDIENCE` / `AZURE_GROUP_*_OID` are unset.
6. **Deprecate `m365-auth.ts` Graph-popup flow** — leave the file in place for now; add a deprecation comment. Phase 7 doc rewrite removes it once the redirect-only MSAL flow is proven for 30 days.
7. **`useActor` shim** — no changes needed; it already delegates to `useIdentity`.

---

## 6. Test plan

### 6.1 Unit tests — token validation

**TheForge** (`tests/unit/services/jwt-validator.test.ts`):
- Valid token → returns Principal with correct `oid`, `email`, `forgeRole`.
- Wrong audience / issuer / expired `exp` → `invalid_bearer` envelope.
- Unknown `kid` → `invalid_bearer`; second call after JWKS refresh succeeds.
- Group OID `AZURE_GROUP_ADMIN_OID` in claims → role `admin`.
- Group overage (`_claim_names.groups` present) → `hasGroupOverage=true`.
- Mock JWKS via `nock`; mint test tokens with `jose` using a test RSA keypair.

**Code Indexer** (`tests/unit/test_auth.py`):
- Mirror the same eight cases via `pytest` + `pytest-httpx`.
- `verify_bearer` with `FORGE_ENV=development` and no header → returns synthetic principal.
- WebSocket subprotocol header parsing — bearer extracted correctly with and without leading `bearer,` prefix.

### 6.2 Integration tests

- `tests/integration/auth-flow.test.ts` — spin up Express with `requireBearer`, mint a test token, call `/api/repos`, expect 200 and `req.principal` reflected in audit log.
- Cross-user isolation: two minted tokens with different `oid`s — `/api/jobs/list?scope=mine` (Phase 2) returns disjoint sets.

### 6.3 e2e smoke against the real Entra tenant

1. `pnpm dev` with `VITE_AUTH_MODE=msal`, `FORGE_ENV=development`.
2. Open `http://localhost:3000`, click "Sign in with Microsoft", consent, return.
3. Verify Network tab shows `Authorization: Bearer eyJ…` on every `/api/*` call.
4. `curl http://localhost:8000/repos` (no token) → 401.
5. `curl -H "Authorization: Bearer <copied>" http://localhost:8000/repos` → 200.
6. Decode at jwt.ms, verify `aud=api://forge-api`, `iss` matches tenant, `groups` contains expected OID.
7. Add user to a different group → re-login → `forgeRole` updates.
8. Sign-out via MSAL `logoutRedirect` → next API call 401.

Scripted version of steps 4–6 lands in `scripts/smoke-auth.sh`.

---

## 7. Rollout sequence with feature flags

| Day | Step | Flags | Acceptance |
|----:|------|-------|------------|
| 1 | Land JWT validators in both services behind `FORGE_AUTH_MODE=optional`. Land MSAL provider in SPA behind `VITE_AUTH_MODE=dev` (still off). | optional / dev | All tests green; `curl /repos` works without token. |
| 2 | Enable MSAL in SPA dev build (`VITE_AUTH_MODE=msal`). Backend stays optional. | optional / msal (dev) | `curl` w/o token still 200; SPA shows MSAL flow. |
| 3a | Deploy to `forge.navistone.com` with `FORGE_AUTH_MODE=required`, `FORGE_ENV=production`, `VITE_AUTH_MODE=msal`. | required everywhere | Smoke test §6.3 passes. |
| 3b | Flip dev defaults: `optional` stays the dev-side default; production hard-codes `required`. | — | `pnpm dev` gets synthesised dev principal automatically. |

Rollback: setting `FORGE_AUTH_MODE=optional` on the prod host re-opens everything (emergency only).

### Trade-off (a) — proxy-level vs. in-app validation

**Recommendation: in-app**. The Code Indexer is reachable via TheForge's proxy *and* directly inside the container network — proxy-only validation would leave the direct path open. In-app is the only correct boundary.

---

## 8. Acceptance gates (mirrors `TEAM_DEPLOYMENT_PLAN.md` §7 row 1)

- [ ] `curl https://forge.navistone.com/api/code-indexer/repos` → 401 without bearer.
- [ ] Same call with valid bearer → 200.
- [ ] `/api/code-indexer/health` → 200 without bearer (liveness probe stays open).
- [ ] SPA login → callback → authenticated session in <5 s.
- [ ] Silent token refresh works for at least one full hour without re-auth prompt.
- [ ] `verify_bearer` unit tests green on both sides.
- [ ] Cross-user isolation integration test green.
- [ ] `FORGE_ENV=development` dev flow still functional with no MSAL config present.
- [ ] e2e smoke (§6.3) passes against the real Navistone tenant.

---

## 9. Open questions / blockers

1. DevOps must return the values listed in `DEVOPS_REQUEST.md` §9 — the entire phase blocks on these.
2. Confirm `FORGE_OVERAGE_POLICY` default — `reject` is safer initially; DevOps to confirm via §10 Q4.
3. Confirm v1 vs. v2 token format — both validators accept both as belt-and-suspenders.
4. Decide before Phase 7 doc rewrite: keep existing `m365-auth.ts` Graph-popup flow as fallback, or delete.

---

## Critical Files for Implementation

- `TheForge/web/src/auth/msal.ts` (new — MSAL singleton + scope config)
- `TheForge/src/services/jwt-validator.ts` (new — JWKS-validated middleware factory)
- `code-indexer-service/app/auth.py` (new — FastAPI `verify_bearer` dependency)
- `TheForge/src/services/api-server.ts` (modify — mount `requireBearer`, line ~350)
- `TheForge/web/src/adapters/identity/azuread-provider.ts` (modify — replace stub with MSAL-backed provider)
