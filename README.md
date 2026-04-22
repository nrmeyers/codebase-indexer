# Code Indexer Service

FastAPI HTTP gateway over [code-graph-rag](../code-graph-rag). Indexes
repositories into LadybugDB (embedded kuzu graph + vector, no Docker) and
exposes structural, semantic, and symbol search to TheForge's dev-agent.

**Default Port:** 8000 (TheForge config uses `8003` via `code_indexer.base_url`)

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check + indexed repo list |
| `POST` | `/index` | Start a background indexing job (202 Accepted) |
| `GET` | `/index/{job_id}/status` | Poll job progress |
| `GET` | `/search/structural` | Cypher passthrough against LadybugDB |
| `GET` | `/search/semantic` | Vector-similarity search over embeddings |
| `GET` | `/search/symbol` | Exact FQN lookup returning source + location |
| `POST` | `/context-bundle` | Grounded code context for the dev-agent |
| `GET` | `/explorer/info` | Kuzu Explorer viewer availability + launch command (optional) |

---

## Run

```bash
# From the code-indexer-service directory:
uv run uvicorn app.main:app --port 8000 --log-level info
```

## Test

```bash
uv run pytest tests/ -v   # 23 tests
```

## Install (first run)

The service depends on the local `code-graph-rag` fork as a path dependency.
Run `uv sync` once to pull everything:

```bash
cd code-indexer-service
uv sync
```

`real-ladybug>=0.15.3` (LadybugDB) is installed automatically.

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

### `GET /index/{job_id}/status`

```json
{
  "job_id": "3f2a…",
  "status": "done",
  "progress_pct": 100,
  "node_count": 1842,
  "rel_count": 3107
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

Vector-similarity search using UniXcoder embeddings stored in LadybugDB.

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
Code Indexer Service (FastAPI)
        │
        │  Python import
        ▼
code-graph-rag (LadybugIngestor + semantic search)
        │
        ▼
LadybugDB (.cgr/graph.db — embedded kuzu, no Docker)
```

The service imports `code-graph-rag` as a local `uv` workspace path
dependency (`code-graph-rag @ file:///…`). Both share the same
`LADYBUG_DB_PATH` file so indexed data is immediately visible to search.

---

## Visualising the graph (optional)

LadybugDB is kuzu-compatible on disk, so the official
[**Kuzu Explorer**](https://docs.kuzudb.com/visualization/kuzu-explorer/)
opens our `.db` file read-only and gives you an interactive node/edge
browser, a Cypher console, and schema introspection.

### Option A — ask the running service for the command

When the Code Indexer is up and the graph has at least one indexed repo:

```bash
curl -s http://localhost:8000/explorer/info | jq
```

```json
{
  "available": true,
  "db_path": "/abs/path/to/graph.db",
  "indexed_repos": ["myproject"],
  "launch_command": "docker run --rm -p 7000:8000 -v /abs/path:/database -e KUZU_PATH=/database/graph.db kuzudb/explorer:latest",
  "viewer_url": "http://localhost:7000",
  "docs_url": "https://docs.kuzudb.com/visualization/kuzu-explorer/"
}
```

Paste `launch_command` into a terminal and open `viewer_url` once the
container reports ready.

### Option B — one-shot helper script

```bash
./scripts/launch-explorer.sh                           # uses LADYBUG_DB_PATH env / .env
./scripts/launch-explorer.sh /path/to/graph.db         # explicit DB
PORT=9000 ./scripts/launch-explorer.sh                 # custom host port
```

The script queries `/explorer/info` for the authoritative command when the
service is running; otherwise it constructs the command offline.

### Notes

- **Docker is only required for the viewer.** All indexing and search work
  without Docker — the viewer is purely opt-in.
- **Single-writer safety.** kuzu-explorer mounts the DB read-only, so it's
  safe to open while TheForge continues to query — but do **not** run it
  during a live `/index` job (LadybugDB serialises writers).
- `available: false` means the viewer would open on an empty graph — index
  a repo first.
