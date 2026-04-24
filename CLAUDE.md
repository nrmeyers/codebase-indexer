# code-indexer-service — Claude Instructions

## What This Repo Is

A **thin FastAPI HTTP gateway** over the `code-graph-rag` engine. TheForge
(Express `:3001`) calls this service over HTTP `:8000` to index repos and
search their symbol graphs.

```
TheForge API (Express :3001)
        │  HTTP :8000
        ▼
Code Indexer Service (FastAPI — this repo)
        │  Python import
        ▼
code-graph-rag (LadybugIngestor + numpy embeddings + MCP)
        │
        ▼
LadybugDB (.cgr/graph.db — embedded kuzu, no Docker)
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
