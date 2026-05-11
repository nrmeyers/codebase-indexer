# code-indexer-service — Claude Instructions

## Cross-Repo Agent Dispatch — Worktree Isolation (READ FIRST if you arrived here from another repo's session)

If this repo is **not** the session-rooted repo (i.e. you started in `~/TheForge`
or `~/code-graph-rag` and were directed here by a prompt), DO NOT touch
`~/code-indexer-service` directly. The harness's `isolation: worktree` only
isolates the session-rooted repo; foreign-repo work in this clone collides with
other agents and corrupts the working tree (confirmed via reflog 2026-05-11;
see `~/TheForge/docs/_diagnostics/worktree-isolation-investigation-2026-05-11.md`).

Required preamble before any code work:

```bash
cd ~/code-indexer-service
git fetch --all
git worktree add /tmp/agent-${RANDOM}-buc-NNNN -b feat/buc-NNNN main
cd /tmp/agent-...   # all work happens here
# Commit + push frequently. Remote branch is canonical.
```

NEVER skip steps 3–4. Concurrent agents WILL be operating in this repo.

---

## What This Repo Is

A **thin FastAPI HTTP gateway** over the `code-graph-rag` engine. TheForge
(Express `:3001`) calls this service over HTTP `:8000` to index repos and
search their symbol graphs.

```
TheForge API (Express :3001)
        │  HTTP :8000
        ▼
Code Indexer Service (FastAPI — this repo)
        │  Python import + pluggable embedder backend
        ▼
code-graph-rag (LadybugIngestor + DuckDB vector store)
        │
        ├─► LadybugDB (.cgr/repos/{slug}.db — embedded kuzu, no Docker)
        └─► DuckDB    (.cgr/repos/{slug}.duck — FLOAT[768] e5-base-v2 vectors)

  Embedder (BUC-1605): pluggable via EMBEDDER_BACKEND={local|sagemaker|tei}.
  All three produce 768-dim intfloat/e5-base-v2 vectors. `local` is the default
  (sentence-transformers in-process); `sagemaker` is the Navistone prod default;
  `tei` is a Hugging Face TEI sidecar. See README §Embedder backends.
```

---

## Tech Stack

| Layer | Choice |
|-------|--------|
| Framework | FastAPI 0.136 + uvicorn |
| Dep manager | uv (workspace path dep on `../code-graph-rag`) |
| Config | pydantic-settings (`.env`) |
| Tests | pytest + httpx (35 tests) |
| Port | 8000 (default) |

---

## Key Files

| File | Purpose |
|------|---------|
| `app/main.py` | FastAPI app, lifespan, startup checks |
| `app/config.py` | `Settings` — pydantic-settings |
| `app/models.py` | Request/response Pydantic models |
| `app/routers/health.py` | `GET /health` |
| `app/routers/index.py` | `POST /index`, `GET /index/{job_id}/status` |
| `app/routers/search.py` | `GET /search/structural|semantic|symbol`, `POST /context-bundle` |
| `app/routers/stats.py` | `GET /stats/{repo}` |
| `app/routers/explorer.py` | `GET /explorer/info` — LadybugDB Explorer launcher |
| `app/embedders/` | Pluggable embedder backends (`local`, `sagemaker`, `tei`). Use `get_embedder()` factory (async) or `app.embedders.sync_bridge.embed_text_sync` / `get_embedder_or_none` for sync callers. The legacy `app/services/sagemaker_embedder.py` shim was removed in BUC-1608 (PR #58). |
| `app/services/lm_studio.py` | OpenAI-compatible adapter for legacy LM Studio path (retired in TheForge PR #168) |
| `app/services/reranker.py` | Listwise rerank via `nomic-ai/CodeRankLLM` (legacy; future rerank backends TBD) |

---

## All 20 Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness + indexed repo list |
| `POST` | `/index` | Start background indexing job (202) |
| `GET` | `/index/{job_id}/status` | Poll job progress |
| `GET` | `/search/structural` | Cypher passthrough |
| `GET` | `/search/semantic` | Vector cosine similarity |
| `GET` | `/search/symbol` | Exact FQN lookup |
| `POST` | `/context-bundle` | Grounded code context for dev-agent |
| `GET` | `/stats/{repo}` | Node/rel counts, embedding count |
| `GET` | `/repos` | List all indexed repos |
| `GET` | `/explorer/info` | LadybugDB Explorer launch command |
| `DELETE` | `/repos/{repo}` | Remove a repo from the index |
| `GET` | `/jobs` | List all background jobs |
| `DELETE` | `/jobs/{job_id}` | Cancel a running job |
| `GET` | `/search/browse` | Structured browse (package tree, file list) |
| `GET` | `/search/callers` | Upstream callers of a symbol |
| `GET` | `/search/callees` | Downstream callees of a symbol |
| `GET` | `/graph/neighborhood` | N-hop subgraph around a symbol |
| `GET` | `/graph/schema` | Repo schema (node labels, rel types, counts) |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/openapi.json` | OpenAPI spec |

---

## Dev Commands

```bash
uv sync                                              # Install deps
uv run uvicorn app.main:app --port 8000              # Start service
uv run pytest tests/ -v                              # Run 35 tests
```

Auto-started by TheForge when `pnpm dev` runs (via `scripts/start-indexer.sh`).
Set `CODE_INDEXER_PATH` env var if the service is not at `~/code-indexer-service`.

---

## Environment Variables

See `.env.example`. Critical ones:

| Variable | Default | Notes |
|----------|---------|-------|
| `LADYBUG_DB_PATH` | `.cgr/graph.db` | Shared with code-graph-rag |
| `LADYBUG_BATCH_SIZE` | `1000` | Ingestor flush batch |
| `TARGET_REPO_PATH` | `.` | Default repo when request omits `repo_path` |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Bind port |
| `EMBEDDER_BACKEND` | `local` | Selects embedder backend (`local`/`sagemaker`/`tei`). See README §Embedder backends. Navistone prod sets `sagemaker`. |

---

## Architecture Patterns

### Background jobs
`POST /index` returns `202 { job_id }` immediately and runs indexing in a
`asyncio.Task`. Each repo has an `asyncio.Lock` — concurrent index of the same
repo returns `409 Conflict` (LadybugDB single-writer constraint).

### Error responses
Use FastAPI `HTTPException` with descriptive `detail` strings. No custom
error envelope — the gateway is thin.

### Result pattern
This service does **not** use TheForge's `Result<T, E>` pattern — it uses
standard FastAPI exception handling. TheForge's `code-indexer-client.ts`
wraps responses in `Result<T, ErrorEnvelope>` on the Node side.

---

## Coding Standards

- Python 3.12+, strict type hints
- `from __future__ import annotations` at top of every file
- Pydantic models in `app/models.py` — no inline TypedDicts in routers
- All config via `app/config.py Settings` (pydantic-settings) — no `os.environ` direct reads
- No `print()` — use `logging.getLogger(__name__)`
- Tests use `httpx.AsyncClient` via `httpx.ASGITransport` (no live server needed)

---

## Connection to TheForge

TheForge proxies requests through `/api/code-indexer/*` → `http://localhost:8000/*`.
The client is at `src/services/code-indexer-client.ts`.

The `CODE_INDEXER_BASE_URL` env var in TheForge's `.env` controls the target
(default `http://localhost:8000`).
