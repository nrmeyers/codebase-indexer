# Code Indexer Service

FastAPI HTTP gateway over [code-graph-rag](../code-graph-rag). Indexes
repositories into LadybugDB (embedded kuzu graph, no Docker) with numpy
sidecar embeddings for semantic search. Exposes structural, semantic, symbol,
and context-bundle search to TheForge's dev-agent.

**Default Port:** 8000

---

## Endpoints (core)

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check + indexed repo list |
| `POST` | `/index` | Start a background indexing job (202 Accepted) |
| `GET` | `/index/{job_id}/status` | Poll job progress |
| `GET` | `/search/structural` | Cypher passthrough against LadybugDB |
| `GET` | `/search/semantic` | Vector-similarity search over numpy embeddings |
| `GET` | `/search/symbol` | Exact FQN lookup returning source + location |
| `GET` | `/search/browse` | Package tree / file list browser |
| `GET` | `/search/callers` | Upstream callers of a symbol |
| `GET` | `/search/callees` | Downstream callees of a symbol |
| `POST` | `/context-bundle` | Grounded code context for the dev-agent |
| `GET` | `/stats/{repo}` | Node/rel counts + embedding count |
| `GET` | `/repos` | List all indexed repos |
| `DELETE` | `/repos/{repo}` | Remove a repo from the index |
| `GET` | `/graph/neighborhood` | N-hop subgraph around a symbol |
| `GET` | `/graph/schema` | Node labels, rel types, counts |
| `GET` | `/jobs` | List background jobs |
| `DELETE` | `/jobs/{job_id}` | Cancel a job |
| `GET` | `/explorer/info` | LadybugDB Explorer launch command (optional) |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/openapi.json` | OpenAPI spec |

---

## Run

```bash
# From the code-indexer-service directory:
uv run uvicorn app.main:app --port 8000 --log-level info
```

TheForge auto-starts this service when you run `pnpm dev` (via
`scripts/start-indexer.sh`). Set `CODE_INDEXER_PATH` if the service lives
somewhere other than `~/code-indexer-service`.

## Test

```bash
uv run pytest tests/ -v   # 35 tests
```

## Install (first run)

The service depends on the local `code-graph-rag` fork as a path dependency.
Run `uv sync` once to pull everything:

```bash
cd ~/code-indexer-service
uv sync
```

`real-ladybug>=0.15.3` (LadybugDB) and `numpy` are installed automatically.

---

## Config

| Env var | Default | Notes |
|---|---|---|
| `LADYBUG_DB_PATH` | `.cgr/graph.db` | Shared with code-graph-rag |
| `LADYBUG_BATCH_SIZE` | `1000` | Ingestor flush batch size |
| `TARGET_REPO_PATH` | `.` | Default repo when request omits `repo_path` |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server bind port |

Copy `.env.example` → `.env` and adjust paths for your machine.

---

## Endpoint Reference

### `GET /health`

```json
{
  "status": "ok",
  "db_path": ".cgr/graph.db",
  "indexed_repos": ["myproject", "other-repo"]
}
```

### `POST /index`

```json
// Request
{ "repo_path": "/absolute/path/to/repo", "force_reindex": false }

// Response 202
{ "job_id": "3f2a…", "status": "running" }
```

Returns `409 Conflict` if an index job for the same repo is already running
(LadybugDB is single-writer; concurrent jobs serialize via `asyncio.Lock`).

### `GET /index/{job_id}/status`

```json
{
  "job_id": "3f2a…",
  "status": "done",
  "progress_pct": 100,
  "node_count": 1842,
  "rel_count": 3107,
  "embedding_count": 892
}
```

`status` is one of `running | done | failed`.

### `GET /search/structural?q={cypher}&limit=20`

Runs arbitrary Cypher against LadybugDB. A `LIMIT` clause is appended
automatically if the query does not already include one.

```json
{
  "row_count": 2,
  "nodes": [
    { "name": "process_file", "qualified_name": "myproject.parser.process_file" }
  ]
}
```

### `GET /search/semantic?q={text}&k=10`

Vector-similarity search using UniXcoder embeddings stored as numpy sidecars.

```json
{
  "results": [
    { "symbol": "myproject.parser.process_file", "score": 0.94,
      "node_id": "myproject.parser.process_file", "name": "process_file", "type": "Function" }
  ]
}
```

### `GET /search/symbol?fqn={qualified_name}`

```json
{
  "qualified_name": "myproject.parser.process_file",
  "source": "def process_file(path: Path) -> ...",
  "file": "/abs/path/to/parser.py",
  "start_line": 42,
  "end_line": 71
}
```

### `POST /context-bundle`

```json
// Request
{
  "repo_path": "/abs/path/to/repo",
  "task_description": "add retry logic to the HTTP client",
  "depth": 2
}

// Response
{
  "symbols": ["myproject.http.get", "myproject.http.post"],
  "source_snippets": { "myproject.http.get": "def get(url): ..." },
  "call_graph": { "myproject.http.get": ["myproject.http._send"] },
  "total_tokens": 1840
}
```

---

## Architecture

```
TheForge API (Express :3001)
        │
        │  HTTP :8000
        ▼
Code Indexer Service (FastAPI — this repo)
        │
        │  Python import
        ▼
code-graph-rag (LadybugIngestor + numpy embeddings)
        │
        ├─► LadybugDB (.cgr/graph.db — embedded kuzu)
        └─► numpy sidecars (.cgr/{repo}.embeddings.npy + .json)
```

The service imports `code-graph-rag` as a local `uv` workspace path dependency.
Both share the same `LADYBUG_DB_PATH` so indexed data is immediately visible
to search.

---

## Visualising the graph (optional)

LadybugDB is kuzu-compatible on disk. The Kuzu Explorer Docker image
opens the `.db` file read-only for interactive node/edge browsing, a Cypher
console, and schema introspection.

### Ask the running service for the command

```bash
curl -s http://localhost:8000/explorer/info | jq
```

```json
{
  "available": true,
  "db_path": "/abs/path/to/graph.db",
  "indexed_repos": ["myproject"],
  "launch_command": "docker run --rm -p 7001:8000 -v /abs/path:/database -e LADYBUG_PATH=/database/graph.db ghcr.io/ladybugdb/explorer:latest",
  "viewer_url": "http://localhost:7001",
  "docs_url": "https://docs.ladybugdb.com/visualization/explorer/"
}
```

Paste `launch_command` into a terminal and open `viewer_url` once ready.

### Notes

- **Docker is only required for the viewer.** All indexing and search work
  without Docker — the viewer is purely opt-in.
- **Single-writer safety.** The Explorer mounts the DB read-only so it's safe
  to browse while TheForge queries, but do **not** browse during a live
  `/index` job.
- `available: false` means no repos are indexed yet — run `POST /index` first.
