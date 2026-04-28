# Frontend Agent Brief — Code Indexer + TheForge Integration

**Read this whole document before writing any code.**  It is self-
contained: a frontend agent (or human) joining cold can implement
against the Code Indexer service from TheForge using only this brief
and the linked TypeScript types.

**Status of the system:** indexing, semantic search, structural search,
context-bundle, two-stage retrieval (cosine top-50 → optional listwise
rerank), and chat → context-bundle wiring all ship.  M365 OAuth is
**Phase 1 — not yet shipped**; today the service is unauthenticated on
localhost.  When OAuth lands, every endpoint except `/health` requires
`Authorization: Bearer <m365 access token>`.  Section 6 below covers the
forward-compatible patterns to use NOW so OAuth lands as a no-op for
the frontend.

---

## 1. Stack & where things live

| Surface | Repo | Path |
|---------|------|------|
| Code Indexer FastAPI service | `code-indexer-service` | `app/main.py`, `app/routers/*.py` |
| Code Indexer engine (parsers, embedder, graph) | `code-graph-rag` (sibling repo) | imported as a package |
| TheForge Express API gateway | `TheForge` | `src/services/api-server.ts` |
| TheForge frontend | `TheForge/web/` | `web/src/pages/*`, `web/src/components/*` |
| Existing TS client for the indexer | TheForge | `src/services/code-indexer-client.ts` (backend), `web/src/components/code-indexer/api.ts` (frontend) |

**Always go through `code-indexer-client.ts`** — never call the indexer
directly from frontend components.  The client is the typed contract;
keeping it as the single integration point means the OAuth swap in
Phase 1 changes one file.

---

## 2. Base URL conventions

| Environment | TheForge → Code Indexer base URL |
|-------------|----------------------------------|
| Local dev | `http://localhost:8000` |
| Team / staging | `https://forge.navistone.com/api/code-indexer` (reverse-proxied) |
| Production | same as team |

**The frontend never hits `:8000` directly.**  All UI requests go through
TheForge's Express API at `:3001` (or `forge.navistone.com` in prod),
which proxies to the Code Indexer.  This keeps CORS scope to a single
origin and lets us attach the M365 token at the gateway.

```
Browser ──► forge.navistone.com (TheForge gateway, M365 OAuth)
              └──► /api/code-indexer/*  (proxied internally; service-to-service auth)
              └──► /api/skill-api/*
              └──► /api/model-router/*
```

---

## 3. Endpoint catalog (current shipping shape)

> All paths below are AS THEY EXIST ON THE CODE INDEXER SERVICE.  When
> calling from a TheForge frontend component, prefix with
> `/api/code-indexer/`.

### Health & metadata

```
GET  /health                       → HealthResponse
GET  /repos                        → { repos: RepoSummary[] }
GET  /repos/{slug}                 → RepoStatsResponse
GET  /repos/{slug}/watch           → watcher status (Phase 5+)
```

### Indexing (jobs are user-scoped from Phase 2 onward)

```
POST   /index                      → 202 IndexAccepted
GET    /index/{job_id}/status      → IndexStatus
POST   /index/{job_id}/cancel      → IndexStatus
GET    /index/jobs/list?scope=mine → JobListResponse  (Phase 2: scope=all admin-only)
POST   /index/jobs/clear           → JobClearResponse
DELETE /index/{slug}               → DeleteIndexResponse
```

### Search

```
GET  /search/structural?q=…&repo=…&limit=20    → { nodes, relationships }
GET  /search/semantic?q=…&repo=…&k=10&rerank=  → SemanticSearchResponse
GET  /search/symbol?fqn=…&repo=…               → { source, file, line_start, line_end }
POST /context-bundle                            → ContextBundleResponse
GET  /symbols/*                                 → symbol-graph navigation
```

### WebSocket

```
WS   /ws    multiplexed channel; events shaped { type, payload, ts }
            current event types:
              "index_progress"          — per-job % + phase + counters
              "index_partial_update"    — Phase 5: realtime updater touched a symbol
              "activity_event"          — orchestrator turns + audit events
```

---

## 4. TypeScript types (authoritative — don't reshape these)

```ts
type ErrorEnvelope = {
  error: string;       // stable machine-readable code
  message: string;     // human-readable
  timestamp: string;   // ISO 8601
  traceId?: string;
};

type LMStudioHealth = {
  configured: boolean;
  reachable: boolean;
  embed_model: string | null;
  rerank_model: string | null;
  can_embed: boolean;
  can_rerank: boolean;
};

type RepoHealth = {
  name: string;             // slug
  path: string;
  node_count: number;       // graph nodes ≈ symbols + files
  embedding_count: number;
  last_indexed_at: string | null;
};

type HealthResponse = {
  status: 'ok' | 'degraded';
  db_path: string;
  indexed_repos: string[];
  repos: RepoHealth[];
  lm_studio?: LMStudioHealth;
};

type IndexRequest = { repo_path: string; force_reindex?: boolean };
type IndexAccepted = { job_id: string };

type IndexStatus = {
  job_id: string;
  status: 'queued' | 'running' | 'done' | 'failed' | 'cancelled';
  progress_pct: number;
  phase: string;
  files_total: number;
  files_done: number;
  node_count: number;
  rel_count: number;
  embedding_count: number;
  started_at: string;
  finished_at: string | null;
  error: string | null;
};

type SemanticSearchResult = {
  qualified_name: string;
  symbol_type: string;        // "Function" | "Method" | "Class" | …
  file_path: string;
  start_line: number;
  end_line: number;
  score: number;              // [-1, 1]
  source_snippet?: string;
  pagerank?: number;
};

type SemanticSearchResponse = {
  query: string;
  repo: string;
  results: SemanticSearchResult[];
  reranked: boolean;
  search_intent?: 'semantic' | 'fqn' | null;
};

type ContextBundleRequest = {
  repo_path: string;          // slug, absolute path, or "*" for fan-out
  task_description: string;
  depth?: number;             // graph traversal depth (default 3)
  k?: number;                 // top-k seed symbols (default 12)
  rerank?: boolean;
};

type ContextBundleResponse = {
  symbols: SemanticSearchResult[];
  source_snippets: Record<string, string>;
  call_graph: {
    nodes: Array<{ id: string; type: string; label: string }>;
    edges: Array<{ from: string; to: string; type: string }>;
  };
  total_tokens: number;
  reranked: boolean;
};
```

**Single source of truth:**
- Backend: `TheForge/src/services/code-indexer-client.ts`
- Frontend: `TheForge/web/src/components/code-indexer/types.ts`

If you find yourself redefining one of these in a new component, stop
— import from the client.

---

## 5. Common UI patterns

### 5.1 Chat header model indicator

```tsx
// Already shipped: web/src/pages/Chat.tsx
const { data: settings } = useQuery(['settings'], fetchSettings);
return (
  <div className="chat-model-indicator">
    Model: <code>{settings?.llm_model ?? 'qwen/qwen3.6-27b'}</code> (local)
  </div>
);
```

### 5.2 Search-as-you-type (NEVER rerank on keystroke)

```tsx
const debouncedQ = useDebounce(query, 250);

const { data } = useQuery({
  queryKey: ['search', debouncedQ, repo],
  queryFn: () => searchSemanticFull(debouncedQ, repo, { k: 10, rerank: false }),
  enabled: debouncedQ.length >= 2,
});

// Optional explicit "Improve with rerank" button:
<button onClick={() => setRerank(true)} disabled={!health?.lm_studio?.can_rerank}>
  Improve with rerank (~100s)
</button>
```

### 5.3 Index job progress

```tsx
// 1. POST /index, get job_id
const { job_id } = await startIndex(repoPath);

// 2. Subscribe to WS for live updates (preferred)
useWebSocket((evt) => {
  if (evt.type === 'index_progress' && evt.payload.job_id === job_id) {
    setProgress(evt.payload);
  }
});

// 3. Or poll /index/{job_id}/status every 2s as fallback
useInterval(async () => {
  const st = await getIndexStatus(job_id);
  setProgress(st);
  if (['done', 'failed', 'cancelled'].includes(st.status)) stopPolling();
}, 2000);
```

### 5.4 Context bundle (orchestrator already does this server-side)

The chat path now fans out automatically. You only need the
context-bundle endpoint directly when **building a debug / explorer UI**
that lets users inspect what the LLM saw:

```tsx
const bundle = await fetchContextBundle({
  repo_path: slug,             // or "*" for cross-repo
  task_description: taskDescription,
  depth: 3,
  k: 12,
  rerank: health?.lm_studio?.can_rerank && repoSymbolCount >= 500,
});
```

### 5.5 Error handling

The Result<T, E> pattern is enforced server-side; the frontend gets
either a 2xx body or a 4xx/5xx with an `ErrorEnvelope`.  Standard
treatment:

```tsx
try {
  const res = await searchSemantic(q, repo);
  // success path
} catch (err) {
  // err is { status, envelope: ErrorEnvelope }
  const code = err.envelope.error;     // e.g. "repo_not_indexed"
  const msg  = err.envelope.message;
  toast.error(msg, {
    description: code === 'lm_studio_unavailable'
      ? 'Rerank temporarily disabled — using cosine-only.'
      : undefined,
  });
}
```

**Never display `traceId` to end users.** Surface it only in dev tools
or a "Copy diagnostics" button.

---

## 6. Auth — what to build NOW so Phase 1 OAuth lands as a no-op

The service is unauthenticated today.  Phase 1 lights up M365 OAuth.
To avoid rework:

1. **Wrap every fetch in a single helper** that already takes a token
   parameter:

   ```ts
   async function indexerFetch<T>(
     path: string,
     opts: RequestInit = {},
     token?: string,
   ): Promise<T> {
     const headers = new Headers(opts.headers);
     if (token) headers.set('Authorization', `Bearer ${token}`);
     const res = await fetch(`/api/code-indexer${path}`, { ...opts, headers });
     if (!res.ok) throw await parseEnvelope(res);
     return res.json();
   }
   ```

   In dev, `token` is `undefined`.  Phase 1 swaps in MSAL.

2. **Add an MSAL provider stub** at the app root using a feature flag:

   ```tsx
   // web/src/auth/msal.ts
   export const msalConfig = {
     auth: {
       clientId: import.meta.env.VITE_AZURE_CLIENT_ID,
       authority: `https://login.microsoftonline.com/${import.meta.env.VITE_AZURE_TENANT_ID}`,
       redirectUri: import.meta.env.VITE_API_BASE_URL + '/api/auth/m365/callback',
     },
     cache: { cacheLocation: 'sessionStorage' },
   };

   export const tokenScopes = ['api://code-indexer/.default'];
   ```

   Until Phase 1, the env vars are dev placeholders and MSAL is
   short-circuited by `useEffect(() => { if (!enabled) seedDevUser(); })`.

3. **Identity context for components**:

   ```tsx
   const { user, getToken } = useIdentity();   // dev: synthetic; prod: MSAL
   const onSearch = async () => {
     const token = await getToken();           // no-op in dev
     await indexerFetch('/search/semantic?…', {}, token);
   };
   ```

4. **WebSocket auth** (Phase 1+) — bearer goes in the
   `Sec-WebSocket-Protocol` subprotocol because browsers refuse
   `Authorization` headers on WS.  The hook should already accept a
   token-acquirer callback today even if it ignores it:

   ```ts
   useWebSocket((evt) => …, { protocols: token ? ['bearer', token] : undefined });
   ```

---

## 7. Performance & UX expectations

| Operation | Latency budget | Notes |
|-----------|---------------:|-------|
| `GET /health` | < 50 ms | Safe to poll every 10 s for the model indicator. |
| `GET /search/semantic` (no rerank) | p95 < 200 ms | Search-as-you-type with 250 ms debounce. |
| `GET /search/semantic?rerank=true` | ≈ 100 s | Always behind an explicit "Improve" button or progress UI. |
| `GET /search/structural` | p95 < 500 ms | Cypher-like graph queries; depends on query selectivity. |
| `POST /index` | < 50 ms (returns 202) | Real work runs in background. |
| `GET /index/{job}/status` | < 50 ms | Use WS for live updates; this is a fallback. |
| `POST /context-bundle` (no rerank) | p95 < 1 s | Orchestrator builds these on every chat turn. |
| `POST /context-bundle?rerank=true` | ≈ 100 s | Server-gated at ≥ 500 symbols + can_rerank=true. |

**Never block first paint on indexer calls.**  All endpoints can fail
or be slow; render skeletons immediately and patch in data as it
arrives.

---

## 8. Things you should NOT do

- **Don't bypass `code-indexer-client.ts`.** Every fetch goes through it.
- **Don't redefine the types in §4.** Import them.
- **Don't call the indexer at `:8000` directly from the browser.** Always
  through TheForge's gateway.
- **Don't poll `/index/{job}/status` faster than every 2 s** — burns tokens
  and blocks the event loop.  Use the WebSocket.
- **Don't construct ErrorEnvelopes inline.** They come from the server.
- **Don't add new endpoints to the indexer from the frontend side.** If
  you need data, propose a new endpoint and let the backend agent ship
  it.
- **Don't trust `traceId` to be present.** It's added by the orchestrator
  but absent on direct indexer calls.

---

## 9. Validation — how to know your work is correct

Before opening a PR:

```bash
# TheForge
pnpm build                        # backend TS
cd web && npx tsc --noEmit        # frontend TS (REQUIRED — pnpm build misses this)
pnpm vitest run                   # all tests

# Code Indexer (unaffected by FE work but spot-check it's still up)
curl http://localhost:8000/health | jq .
```

A green run on all four commands is the floor.  PRs that touch
`code-indexer-client.ts` should also include a `pnpm vitest run
tests/unit/services/code-indexer-client.test.ts` line in the
description.

---

## 10. Phases on the roadmap that affect the frontend

| Phase | Frontend impact |
|-------|-----------------|
| 1 — M365 OAuth | MSAL provider goes live; `getToken()` becomes real; login button replaces dev-user banner. |
| 2 — Persistent jobs | `/index/jobs/list` adds `scope` filter; show "your jobs" vs "all jobs (admin)". |
| 3 — Container deploy | Base URLs flip to `forge.navistone.com`; no code change if §6 helpers were used. |
| 4 — Grafana metrics | New `/metrics` endpoint (server-only); FE may add a "View dashboard" link. |
| 5 — Realtime updater | New WS event `index_partial_update`; toast or badge on affected files. |
| 6 — Cleanup | None visible. |
| 7 — Doc rewrite | None visible. |
| 8 — HNSW index | Search latency drops; no API change. |
| 9 — Cross-repo unified ranking | `repo='*'` becomes the recommended default; multi-repo result list with per-repo grouping. |

Each phase will arrive with a tight changelog.  Read those before
upgrading.
