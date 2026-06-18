# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

A **FastAPI HTTP gateway** over the `code-graph-rag` engine. TheForge
(Express `:3001`) calls this service over HTTP `:8000` to index repos and
search their symbol graphs.

```
TheForge API (Express :3001)
        │  HTTP :8000
        ▼
Code Indexer Service (FastAPI — this repo)
        │  Python import + pluggable embedder backend
        ▼
code-graph-rag (LadybugIngestor)
        │
        ├─► LadybugDB (.cgr/repos/{slug}.db — embedded kuzu, no Docker)
        ├─► DuckDB    (.cgr/repos/{slug}.duck — FLOAT[768] vectors)
        └─► Tantivy   (.cgr/repos/{slug}.tantivy/ — BM25 lexical index)
```

Per-repo storage keyed by canonical slug `{org}__{repo}` (derived from git
remote origin — `app/services/slug.py`). Job state persists in SQLite WAL at
`.cgr/jobs.sqlite` and survives restarts.

---

## Dev Commands

```bash
uv sync                                              # Install deps (uv workspace; path dep on ../code-graph-rag)
uv run uvicorn app.main:app --port 8000              # Start service
uv run pytest tests/ -v                              # Run tests (~60 files, 500+ tests)
uv run pytest tests/test_search.py::test_name -v     # Single test
```

There is also a standalone CLI (`app/cli/`): `code-indexer setup|serve|start|stop|status|index|search|symbol|callers|callees|bundle|explore|remove`. CLI-launched daemon defaults to port 8003.

Auto-started by TheForge when `pnpm dev` runs (via `scripts/start-indexer.sh`).
Set `CODE_INDEXER_PATH` env var if the service is not at `~/code-indexer-service`.

---

## Endpoints by Router (app/routers/)

| Router | Routes |
|--------|--------|
| `health.py` | `GET /health` — liveness, indexed repos, embedder status |
| `index.py` | `POST /index` (202), job status/cancel/list/clear, `POST /index/embed` (standalone embed pass), `GET /stats/{repo}`, `GET /jobs/{job_id}/events` |
| `search.py` | `GET /search/structural|semantic|symbol|lexical|centrality|files|types`, `GET /search/graph/overview` |
| `context_bundle.py` | `POST /context-bundle` — grounded context for dev-agent |
| `repos.py` | `GET /repos`, per-repo stats/reindex/watch/centrality/centroid/neighbors/clusters, `DELETE /repos/{name}` |
| `symbols.py` | `GET /symbols/{fqn}`, `/symbols/{fqn}/callers|callees` |
| `embed.py` | `POST /embed` — embed arbitrary text (query vectors for TheForge) |
| `github.py` | PAT-authed org/repo listing + `POST /github/index` (clone + index) |
| `admin.py` | `/admin/s3/*` snapshot/restore/health, `/admin/migrate-slugs`, `/admin/resolve-cross-repo-imports` |
| `explorer.py` | `GET /explorer/info` — LadybugDB Explorer launch command |
| `disk.py` | `GET /disk-usage` |
| `websocket.py` | `WS /ws` — index progress event stream |

Plus `GET /metrics` (Prometheus) and `GET /openapi.json`.

---

## Key Services (app/services/)

| Module | Purpose |
|--------|---------|
| `ladybug_ingestor.py` | Graph ingestor (context manager + batch flush) |
| `ladybug_pool.py` / `ladybug_buffer_pool.py` | Connection helpers (RO/RW locks); bounded kuzu buffer-pool sizing |
| `jobs_store.py` | Durable SQLite job store; worker-token orphan detection on restart |
| `chunk_strategies.py` | Hierarchical chunking (file/class/module summaries) for semantic index |
| `bm25_index.py` / `tantivy_index.py` | Lexical search; BM25 fused with semantic via RRF |
| `pagerank.py` / `centroid.py` / `neighbors.py` | Centrality, per-repo topic centroid, KNN/cluster layers |
| `markdown_indexer.py` | Docs discovery + chunking (`docs/`, `.planning/`, root docs) |
| `s3_store.py` / `s3_restore.py` | S3-primary persistence: restore on startup, snapshot on re-index/shutdown + periodic (10 min) |
| `watch_manager.py` | Per-repo file watcher (feature-flagged `WATCH_ENABLED`, off by default); debounced partial re-index |
| `cross_repo_imports.py` | Rewires external imports to `{target_slug}::{qname}` (flagged, off by default) |
| `slug.py` | Canonical `{org}__{repo}` slug derivation |
| `source_fetch.py` | Source snippet resolution for FQNs (context-bundle + reranker) |
| `reranker.py` | Opt-in (`?rerank=true`) listwise rerank; fail-open — always falls back to bi-encoder order |
| `lm_studio.py` | Only rerank backend currently wired (used by `reranker.py`); also a dev-only embed fallback in `search.py:_embed_query`. BUC-1545 tracks replacing it. |

## Embedders (app/embedders/)

Pluggable via `EMBEDDER_BACKEND` env var: `local` (default; sentence-transformers
in-process, 768-dim nomic-embed-text-v1.5), `sagemaker` (production; jina-code-v2
serverless endpoint since LE-129), `tei` (HF TEI sidecar), `openai`
(text-embedding-3-small, 1536-dim — **requires re-index**; the other three share
the FLOAT[768] schema and are env-var swappable). Use the async `get_embedder()`
factory, or `app.embedders.sync_bridge` for sync callers. The actual embed pass
runs in a subprocess (`app/scripts/embed_driver.py`).

---

## Environment Variables

See `.env.example` and `app/config.py` (all config via `Settings`). Critical ones:

| Variable | Default | Notes |
|----------|---------|-------|
| `LADYBUG_DB_DIR` | `.cgr/repos` | Per-repo `{slug}.db` files (`LADYBUG_DB_PATH` is legacy fallback) |
| `JOBS_DB_PATH` | `.cgr/jobs.sqlite` | Durable job store |
| `EMBEDDER_BACKEND` | `local` | `local`/`sagemaker`/`tei`/`openai`; prod = `sagemaker` |
| `S3_INDEX_BUCKET` | `` (empty) | Set to an S3 bucket to enable sync; empty disables it |
| `WATCH_ENABLED` | `false` | File-watcher master switch |
| `GITHUB_TOKEN` / `GITHUB_ALLOWED_OWNERS` | — / `` (empty) | Enables `/github/*` routes; empty allowlist = all owners (set per deploy) |
| `TARGET_REPO_PATH` | `.` | Default repo when request omits `repo_path` |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Loopback-only by default; `0.0.0.0` for containers |

---

## Architecture Patterns

### Background jobs
`POST /index` returns `202 { job_id }`; heavy work runs in a thread-pool worker
(`_blocking_index` in `app/routers/index.py`): passes 1–3 (parse → LadybugDB)
then pass 4 (embed subprocess). Progress flushes to `jobs_store` and broadcasts
over WebSocket `/ws`. Per-repo `asyncio.Lock` — concurrent index of the same
slug returns `409 Conflict` (LadybugDB single-writer constraint). A heartbeat
watchdog fails jobs with no progress for `JOB_STALENESS_THRESHOLD_SECONDS`
(300s; 600s in the `writing` phase).

### Startup lifespan (app/main.py)
DB corruption self-heal → jobs-store init + orphan sweep → repo reconciliation
from on-disk files → metrics → embedder probe/pre-warm → periodic S3 snapshot →
Tantivy mmap warmup → heartbeat reconciler. Shutdown pushes changed index files
to S3.

### Error responses
FastAPI `HTTPException` with descriptive `detail` strings. No custom error
envelope. This service does **not** use TheForge's `Result<T, E>` pattern —
TheForge's `code-indexer-client.ts` wraps responses on the Node side.

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
