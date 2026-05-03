# Consolidated Next Steps — 2026-05-03

> Synthesis of Wave 1 (cleanup, code sweep, docs rehaul, audit refresh)
> + Wave 2 (real-browser E2E UI audit). Drives the next 1–2 sessions of
> work for both BE and FE operators.

## What just landed (this session)

| PR | Repo | Scope |
|----|------|-------|
| #69 | TheForge | docs rehaul-2: -16 net files, 2 major consolidations |
| #28 | indexer | docs rehaul-2: 2 files archived |
| #70 | TheForge | code sweep #3: 7 prose, 10 console.log, types |
| #9 | cgr | code sweep #3: ruff fixes |
| #71 | TheForge | TESTING_GAPS_2 + CONFIG_AUDIT_2 |
| #29 | indexer | TESTING_GAPS_2 + CONFIG_AUDIT_2 |
| #72 | TheForge | playwright E2E UI suite + per-page audit |
| #73 | TheForge | m365/config 503 → 200 enabled:false (E2E #2) |

Plus 57 stale local branches deleted, 1 orphan worktree pruned.

## E2E UI audit — top 5 issues (severity-ranked)

From `docs/E2E_UI_AUDIT.md` on TheForge main:

| # | Layer | Issue | Effort |
|---|-------|-------|--------|
| 1 | FE | No `<Route path="*">` catch-all — unknown URLs render fully black | 1 h |
| 2 | BE | ~~`/api/auth/m365/config` returns 503 on every page~~ → **fixed in #73** | done |
| 3 | BE | `/api/state/health` and `/api/llm/test-connection` return 404 (Chat needs both) | 2 h |
| 4 | BE+FE | Code-Indexer proxy port mismatch (proxy → :8003, service on :8000); FE display string hardcodes :8003 | 1 h |
| 5 | FE | Double-fetch on `/tasks/:id` and `/planner/:id` (StrictMode useEffect) | 30 m |

Plus secondary items from the audit:
- Settings page calls 4 routes that 404 (`git-host/orgs`, `code-indexer/disk-usage`, etc.)
- `/architecture-review`, `/deploy-readiness`, `/spec-review` error tiles have no recovery CTA (dead-ends)
- Governance page top compliance tile shows generic "Something went wrong" with no service context
- Chat sessions panel stuck in permanent "Loading…" with no fallback for failed fetch

## Backend operator queue (mine)

In priority order:

1. **Add missing routes** — `/api/state/health`, `/api/llm/test-connection`, `/api/git-host/orgs`, `/api/code-indexer/disk-usage`. Either implement properly or stub with `200 { available: false }` so the FE renders graceful offline state instead of 404. **2 h.**
2. **Fix code-indexer proxy port** — `src/services/routes/code-indexer-routes.ts` proxies to :8003 but the service runs on :8000 in this environment. Reconcile via env var (`CODE_INDEXER_PROXY_PORT`). **30 m.**
3. **`source_fetch.py` test coverage** — highest-risk untested file project-wide per W1.D. Pure function over `tmp_path`; ~1 h to write 6 cases. **1 h.**
4. **Unblock `jwt-validator.test.ts` suite** — the entire test suite is `.skip`'d in production-shipped code (Phase 1 M6). Per W1.D, the fix is `vitest.config.ts` adding `resolve.conditions: ['node']`. Worth a single attempt; if not, document the actual blocker. **1–2 h.**
5. **`.env.example` port-clarity comments** — both repos document conflicting ports (TheForge :8003 vs indexer :8000). 5-min comment fix in each. **10 m.**
6. **Governance error context** — the top-tile "Something went wrong" is the BE returning a generic `ErrorEnvelope` without service identity. Plumb the service name through. **45 m.**

Total estimate: 5–7 h.

## Frontend operator queue (FE agent)

In priority order:

1. **404 catch-all route** — `<Route path="*" element={<NotFound />} />` in `web/src/App.tsx`. New `<NotFound>` component using existing `<EmptyState>` primitive + a "Go home" link. **1 h.**
2. **Double-fetch fix** — `web/src/pages/TaskDetail.tsx` and `web/src/pages/PlannerProject.tsx` have unguarded `useEffect`s that fire twice under StrictMode. Add a `useRef` guard or move the fetch into a TanStack Query hook (preferred — query layer has dedupe built in). **30 m.**
3. **Hardcoded port string** — `web/src/pages/CodeIndexer.tsx` has a literal `localhost:8003` in the offline banner. Read from env / config. **15 m.**
4. **Recovery CTAs on dead-end error states** — `/architecture-review`, `/deploy-readiness`, `/spec-review` render an `<ErrorState>` with no nav. Add a "Back to Tasks" link or an explicit "Refresh" CTA. Use the existing `<EmptyState>` primitive variants. **1 h.**
5. **Chat sessions loading-spinner fallback** — when `/api/chat/sessions` (or wherever) fails, the loading spinner runs forever. Add an error tile + retry button. **30 m.**
6. **axe-core baseline cleanup** (carryover from the earlier brief) — read the report from PR #66's first run on main, fix violations, flip `continue-on-error: true → false`. **2–4 h depending on findings.**
7. **`<FieldError>` primitive** (carryover) — for `pp-form-error` and `settings-error` per-field annotations. ≤ 40 LOC. **45 m.**

Total estimate: 5–8 h.

## Coordination

- BE owns `src/services/**` and the indexer/cgr backends. FE owns `web/**`.
- For BE issue #4 (port mismatch) the FE side is the hardcoded display string — it's listed in the FE queue. Coordinate at PR boundary, not at file boundary.
- The `.env.example` port-clarity item touches both repos but only needs comments — BE handles since BE already owns both `.env.example` files.

## What's deferred / non-blocking

- **`top5_relevance_semantic` ≥ 90%** target unmet (currently 83%) — gated on Phase 8 HNSW activation OR a stricter grader switch. Both have plans; neither has fired triggers.
- **`indexing_rate_symbols_per_s` ≥ 200 sym/s** target unmet (currently 6 sym/s) — gated on per-symbol → batch refactor in `code-graph-rag/codebase_rag/storage/vector_store_arrow.py`. ~50 LOC, dedicated PR; not blocking SUCCESS_90Q.
- **Phase 5 watcher activation** — `WATCH_ENABLED=true` flip + manual smoke. Operator action; no code work needed.

## Next session kickoff

When you're ready, start with BE #1 (missing routes — biggest console-noise reduction across the FE) in parallel with FE #1 (404 catch-all — biggest UX bug).
