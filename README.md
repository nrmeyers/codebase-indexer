# code-indexer-service

> Ask your codebase questions. Index any repo into a typed symbol graph plus a vector store, then query it from the shell or over HTTP.

[![Python](https://img.shields.io/badge/python-3.12%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-d7ff64?logo=ruff)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](#license)

`code-indexer-service` is a FastAPI gateway and CLI that indexes source repositories into a [tree-sitter](https://tree-sitter.github.io/)–parsed symbol graph (LadybugDB, an embedded kuzu fork — no Docker) backed by a DuckDB vector store. It is powered by the [`code-graph-rag`](https://github.com/navistone/code-graph-rag) engine and supports 11 languages out of the box.

Use it standalone from your shell, or embed it as a sidecar — the same HTTP surface drives both.

You can ask it things like:

- "Where is `process_file` defined and who calls it?"
- "Find the auth handler" — semantic search over symbols.
- "Show me every function downstream of `httpClient.send`."
- "Give me a grounded context bundle for _add retry logic to the HTTP client_."
- "What does this Cypher query return against the repo graph?"

---

## Contents

- [Quickstart](#quickstart)
- [Demo](#demo)
- [Why not grep](#why-not-grep)
- [Two ways to use it](#two-ways-to-use-it)
- [Standalone CLI](#standalone-cli)
- [REST API](#rest-api)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Development](#development)
- [License](#license)

---

## Quickstart

```bash
pipx install git+https://github.com/navistone/code-indexer-service.git
code-indexer setup                       # one-time interactive wizard
code-indexer index ~/path/to/your/repo   # indexes in the background, polls to completion
code-indexer search "where is the auth code"
code-indexer callers myproject.parser.process_file
```

> **STOP — semantic search needs an embedder backend.**
>
> The default `EMBEDDER_BACKEND=local` requires the optional
> `[local-embed]` extras group. If you skip the install, the indexer will
> boot but every `/search/semantic` call returns 503 with
> `in-process embedder not initialised`, and `GET /health` will show
> `embedder.available: false` (look for the loud startup banner).
>
> Pick **one** of the four paths before you start indexing:
>
> ```bash
> # 1. Local dev (recommended for new contributors)
> uv sync --group local-embed
>
> # 2. Navistone production (AWS SageMaker Serverless Inference)
> uv sync && export AWS_PROFILE=... EMBEDDER_BACKEND=sagemaker \
>            SAGEMAKER_ENDPOINT_NAME=jina-code-v2-serverless
>
> # 3. GPU box without AWS (Hugging Face TEI sidecar)
> uv sync && export EMBEDDER_BACKEND=tei TEI_URL=http://localhost:8080
>
> # 4. BYO OpenAI key
> uv sync --extra byo && export EMBEDDER_BACKEND=openai OPENAI_API_KEY=sk-...
> ```
>
> Verify after `uvicorn app.main:app` starts:
>
> ```bash
> curl -s http://localhost:8003/health | jq .embedder
> # Expected: { "backend": "local", "available": true, "configured": true, "dim": 768, ... }
> ```
>
> If `available` is `false`, look for the startup banner — it prints the
> exact `last_error` (e.g. `ModuleNotFoundError: No module named
> 'sentence_transformers'`) and the fix. See
> [`docs/EMBEDDERS.md`](docs/EMBEDDERS.md) for full troubleshooting.

The CLI auto-starts the FastAPI service in the background on first use. To run the HTTP gateway directly, see [Standalone CLI § serve](#cli-reference).

---

## Demo

> _Screencast coming — drop an asciinema or `vhs`-generated GIF here when one is recorded._

---

## Why not grep

`grep` and your IDE's "Find Usages" both top out where this tool starts:

- **Semantic, not lexical.** Query "the auth handler" without knowing the function is called `verify_session_token`. Backed by 768-dim embeddings (SageMaker: [`jinaai/jina-code-embeddings-v2`](https://huggingface.co/jinaai/jina-code-embeddings-v2); local/TEI: [`intfloat/e5-base-v2`](https://huggingface.co/intfloat/e5-base-v2)) stored in DuckDB.
- **Cross-file and cross-repo by default.** Imports, calls, inheritance, and references are first-class graph edges. Ask "who calls X" across the entire indexed corpus, not just the current file.
- **Structural Cypher queries.** The symbol graph is queryable directly — return every `Function` that imports from a given module, list all `Class` nodes with more than 20 methods, etc.
- **Grounded context bundles for LLMs.** `/context-bundle` returns the symbols, snippets, and call graph relevant to a task — primary use case is feeding a coding agent without dumping whole files.

---

## Two ways to use it

### Standalone tool

`pipx install` (or `uv sync` from a clone), `code-indexer setup`, point it at a directory. The CLI manages the service daemon, indexes repos, and wraps every endpoint. This is the path you want for personal use, evaluation, or scripting against your own repos.

### Embedded in TheForge

[TheForge](https://github.com/navistone/TheForge) auto-starts this service when you run `pnpm dev` (via `scripts/start-indexer.sh`) and proxies it under `/api/code-indexer/*`. Set `CODE_INDEXER_PATH` if the service lives somewhere other than `~/code-indexer-service`. The CLI is a parallel, optional surface — it does not change any HTTP contract.

---

## Standalone CLI

### Install

```bash
# Option 1 — pipx (recommended for users)
pipx install git+https://github.com/navistone/code-indexer-service.git

# Option 2 — from a clone (recommended for contributors)
git clone https://github.com/navistone/code-indexer-service.git
cd code-indexer-service
uv sync
# Then prefix all commands with: uv run code-indexer ...
```

### First run

```bash
code-indexer setup        # writes ~/.code-indexer/config.toml
code-indexer index ~/proj # auto-starts daemon, polls until done
code-indexer status       # shows daemon + indexed repos
```

### CLI reference

| Command                               | Description                                               |
| ------------------------------------- | --------------------------------------------------------- |
| `setup`                               | Interactive wizard. Writes `~/.code-indexer/config.toml`. |
| `serve [--port 8003]`                 | Run the FastAPI service in the foreground.                |
| `start`                               | Spawn the service in the background.                      |
| `stop`                                | Stop the background daemon.                               |
| `status`                              | Show daemon liveness + indexed repos.                     |
| `index <path> [--watch] [--force]`    | Index a directory; polls until done.                      |
| `reindex <slug>`                      | Force a clean re-index of an indexed repo.                |
| `list`                                | List every indexed repo.                                  |
| `search "<query>" [-k 10] [--repo X]` | Semantic search over symbols.                             |
| `symbol <fqn> [--repo X]`             | Look up a symbol's source + location.                     |
| `callers <fqn> [--repo X]`            | Upstream callers of a symbol.                             |
| `callees <fqn> [--repo X]`            | Downstream callees of a symbol.                           |
| `bundle "<task>" --repo <path>`       | Build a grounded context bundle for an LLM.               |
| `explore`                             | Print the LadybugDB Explorer launch command + URL.        |
| `remove <slug> [-y]`                  | Delete a repo's index (cascade).                          |

Pass `--base-url` to any command to talk to a remote service; the CLI otherwise reads `[server].base_url` from the config.

### Config file

The setup wizard writes `~/.code-indexer/config.toml`:

```toml
[server]
base_url = "http://localhost:8003"
port = 8003

[embedder]
backend = "local"      # local | sagemaker | tei

[paths]
data_dir = "/Users/jane/.code-indexer"
```

Override the base URL at any time with `--base-url` or `CODE_INDEXER_BASE_URL`.

---

## REST API

The HTTP service listens on port `8000` by default when run directly (`uv run uvicorn app.main:app`), or `8003` when launched by the CLI to match TheForge's proxy.

Full schema lives at `GET /openapi.json`. The most-used endpoints:

| Method   | Path                     | Description                                             |
| -------- | ------------------------ | ------------------------------------------------------- |
| `GET`    | `/health`                | Liveness + list of indexed repos.                       |
| `POST`   | `/index`                 | Start a background index job. Returns `202` + `job_id`. |
| `GET`    | `/index/{job_id}/status` | Poll job progress.                                      |
| `POST`   | `/index/{job_id}/cancel` | Cancel a running job.                                   |
| `GET`    | `/index/jobs`            | List background jobs.                                   |
| `GET`    | `/search/semantic?q=&k=` | Vector-similarity search.                               |
| `GET`    | `/search/structural?q=`  | Arbitrary Cypher against LadybugDB.                     |
| `GET`    | `/search/lexical?q=`     | BM25 search via Tantivy.                                |
| `GET`    | `/search/symbol?fqn=`    | Exact FQN lookup.                                       |
| `GET`    | `/search/callers?fqn=`   | Upstream callers.                                       |
| `GET`    | `/search/callees?fqn=`   | Downstream callees.                                     |
| `GET`    | `/search/centrality`     | PageRank scores over the symbol graph.                  |
| `GET`    | `/search/files`          | Browse the package/file tree.                           |
| `GET`    | `/search/types`          | Enumerate node labels + counts.                         |
| `POST`   | `/context-bundle`        | Grounded context for a task description.                |
| `GET`    | `/repos`                 | List all indexed repos.                                 |
| `GET`    | `/repos/{name}/stats`    | Node, relationship, and embedding counts.               |
| `POST`   | `/repos/{name}/reindex`  | Force re-index.                                         |
| `DELETE` | `/repos/{name}`          | Remove a repo from the index.                           |
| `GET`    | `/explorer/info`         | LadybugDB Explorer launch command.                      |
| `GET`    | `/metrics`               | Prometheus metrics.                                     |

### Examples

Start an index job:

```bash
curl -sX POST http://localhost:8000/index \
     -H 'content-type: application/json' \
     -d '{"repo_path": "/abs/path/to/repo"}'
# → {"job_id": "3f2a…", "status": "running"}
```

Poll until done:

```bash
curl -s http://localhost:8000/index/3f2a…/status
# → {"status": "done", "progress_pct": 100, "node_count": 1842, "rel_count": 3107, "embedding_count": 892}
```

Semantic search:

```bash
curl -s 'http://localhost:8000/search/semantic?q=session+token+validation&k=5'
```

Cypher passthrough:

```bash
curl -s --data-urlencode 'q=MATCH (f:Function)-[:CALLS]->(g:Function) RETURN f.qualified_name, g.qualified_name LIMIT 10' \
     http://localhost:8000/search/structural
```

Grounded context bundle:

```bash
curl -sX POST http://localhost:8000/context-bundle \
     -H 'content-type: application/json' \
     -d '{"repo_path": "/abs/path/to/repo", "task_description": "add retry logic to the HTTP client", "depth": 2}'
```

---

## Startup Performance Optimizations

Three latency taxes are paid once at `uvicorn` startup so they are never charged to user-facing requests.

**PageRank centrality precompute.** At the end of every index job the Plan J block runs a single PageRank pass over the LadybugDB CALLS graph and persists the normalized scores into the `centrality` table of the per-repo `.duck` DuckDB file. `GET /repos/{name}/centrality` is therefore a pure `SELECT … ORDER BY pagerank DESC LIMIT ?` — sub-millisecond regardless of repo size. The `last_computed_at` field in the response tells TheForge's orchestrator whether its 5-minute in-memory cache is still fresh relative to the last index run (BUC-1577).

**Embedder model warmup (Phase 5).** On startup, a daemon thread issues a single `"warmup"` embed call through whichever backend is configured (`local` / `sagemaker` / `tei`). This absorbs the SageMaker Serverless cold-start (~4–5 s observed) and the `sentence-transformers` model-load time before any search request arrives. The warmup is non-fatal: a broken or unconfigured embedder is logged at DEBUG level and startup continues normally.

**Tantivy segment warm-up (Phase 9).** A second daemon thread iterates all existing `<slug>.tantivy/` directories under `LADYBUG_DB_DIR` and calls `Index.reload()` on each, paging the segment mmaps into the OS buffer cache. Without this, the first BM25 lexical search on a large repo pays 200–800 ms of page faults. The warmup is non-fatal: missing tantivy bindings or corrupt segment directories are silently skipped.

---

## Architecture

```mermaid
flowchart LR
    User([CLI / curl / TheForge UI]) -->|HTTP| API[FastAPI service<br/>app/routers/*]
    API -->|index, search| Engine[code-graph-rag engine<br/>LadybugIngestor + DuckDB store]
    Engine -->|parse| TS[tree-sitter<br/>11 languages]
    Engine -->|embed| EMB{Embedder<br/>backend}
    EMB -->|local| Local[sentence-transformers<br/>in-process]
    EMB -->|sagemaker| SM[AWS SageMaker<br/>Serverless Inference]
    EMB -->|tei| TEI[Text-Embeddings-Inference<br/>HTTP sidecar]
    Engine -->|graph writes| LB[(LadybugDB<br/>.cgr/repos/&lt;slug&gt;.db)]
    Engine -->|vectors| DD[(DuckDB<br/>.cgr/repos/&lt;slug&gt;.duck)]
    API -.->|read-only| LB
    API -.->|read-only| DD
```

- **Parsing.** [tree-sitter](https://tree-sitter.github.io/) grammars for Python, JavaScript, TypeScript, Go, Java, C#, Rust, and more. Symbols (functions, classes, methods), files, modules, and references become typed graph nodes.
- **Graph store.** [LadybugDB](https://docs.ladybugdb.com/) — an embedded, file-backed [kuzu](https://kuzudb.com/) fork. One `.db` file per repo at `.cgr/repos/<slug>.db`. Cypher-queryable. No Docker.
- **Vector store.** DuckDB `FLOAT[768]` column plus `array_cosine_distance` for similarity search. One `.duck` file per repo at `.cgr/repos/<slug>.duck`.
- **Embedders.** Four interchangeable backends behind a single `EmbedderBackend` protocol. The three "native" backends produce 768-dim vectors (`sagemaker`: `jinaai/jina-code-embeddings-v2` post-LE-129; `local` + `tei`: `intfloat/e5-base-v2`) so the on-disk index shape is portable across them — but indexes built under one model are NOT interchangeable with another. A fourth, `openai`, is the bring-your-own path (1536 or 3072 dim — needs a re-index). Switch with `EMBEDDER_BACKEND={local|sagemaker|tei|openai}` and restart. See [`docs/EMBEDDERS.md`](docs/EMBEDDERS.md).
- **Imports across repos.** Cross-repo `IMPORTS` edges link symbols indexed under different slugs (BUC-1598).
- **Centrality.** PageRank scores over the call/import graph are precomputed and exposed at `/search/centrality` (BUC-1577, BUC-1599 persistence).

---

## Configuration

### Environment variables

| Variable                  | Default                 | Description                                                      |
| ------------------------- | ----------------------- | ---------------------------------------------------------------- |
| `LADYBUG_DB_DIR`          | `.cgr/repos`            | Per-repo `.db` + `.duck` storage root.                           |
| `LADYBUG_DB_PATH`         | _(legacy)_              | Single-DB fallback for direct `code-graph-rag` callers.          |
| `LADYBUG_BATCH_SIZE`      | `1000`                  | Ingestor flush batch size.                                       |
| `TARGET_REPO_PATH`        | `.`                     | Default repo when a request omits `repo_path`.                   |
| `HOST`                    | `0.0.0.0`               | HTTP bind address.                                               |
| `PORT`                    | `8000`                  | HTTP bind port.                                                  |
| `EMBEDDER_BACKEND`        | `local`                 | One of `local`, `sagemaker`, `tei`, `openai`.                    |
| `LOCAL_EMBED_MODEL`       | `intfloat/e5-base-v2`   | Override only after re-indexing.                                 |
| `SAGEMAKER_ENDPOINT_NAME` | —                       | Required when `EMBEDDER_BACKEND=sagemaker`.                      |
| `SAGEMAKER_EMBED_REGION`  | `us-east-1`             | AWS region for the SageMaker endpoint.                           |
| `TEI_URL`                 | `http://localhost:8080` | Endpoint for the TEI sidecar.                                    |
| `TEI_TIMEOUT_MS`          | `30000`                 | Per-request TEI timeout.                                         |
| `OPENAI_API_KEY`          | —                       | Required when `EMBEDDER_BACKEND=openai`.                         |
| `OPENAI_EMBED_MODEL`      | `text-embedding-3-small`| `text-embedding-3-small` (1536) or `text-embedding-3-large` (3072). |
| `OPENAI_EMBED_DIM`        | —                       | Matryoshka truncation (3-series only); blank → native dim.       |
| `OPENAI_BASE_URL`         | —                       | Override for Azure / vLLM / LiteLLM gateways.                    |
| `RERANK_ENABLED`          | `false`                 | Opt into the future rerank stage (see `docs/SEARCH_RANKING.md`). |
| `GITHUB_TOKEN`            | —                       | Fine-scoped PAT for `/github/*` routes.                          |

Copy [`.env.example`](.env.example) to `.env` and adjust paths for your machine. The example file documents every variable inline.

### Embedder backends

| Backend     | Default model              | Dim   | When to use                                | Tradeoffs                                                        |
| ----------- | -------------------------- | ----- | ------------------------------------------ | ---------------------------------------------------------------- |
| `local`     | `intfloat/e5-base-v2`      | 768   | Standalone / laptop / no AWS. **Default.** | ~440 MB model download on first run; CPU-bound; zero infra cost. |
| `sagemaker` | `jinaai/jina-code-embeddings-v2` | 768 | Navistone production. Was E5, swapped 2026-05-26 LE-129. | GPU-backed batching; requires AWS creds; per-invocation cost.    |
| `tei`       | `intfloat/e5-base-v2`      | 768   | Self-hosted GPU box.                       | Highest throughput; one extra container; no AWS coupling.        |
| `openai`    | `text-embedding-3-small`   | 1536  | Bring your own — no AWS, no GPU.           | $0.02 / 1M tokens; needs `OPENAI_API_KEY`; **re-index required** (1536 ≠ 768). |

Switching among `local` / `sagemaker` / `tei` is a pure env-var flip — the DuckDB `FLOAT[768]` schema is shared. Switching to `openai` (or anywhere the dim changes) needs a fresh index because the column type doesn't match; see [`docs/EMBEDDERS.md`](docs/EMBEDDERS.md) for the recipe, cost/quality comparison, and protocol contract for plugging in your own backend.

Install the BYO extra alongside the default deps:

```bash
uv sync --extra byo              # installs openai>=1.0 for the openai backend
```

---

## Development

```bash
git clone https://github.com/navistone/code-indexer-service.git
cd code-indexer-service
uv sync                          # installs all deps incl. code-graph-rag path dep
uv sync --group local-embed      # add sentence-transformers for the local backend
uv run uvicorn app.main:app --reload --port 8000
uv run pytest tests/ -v          # 51+ tests
uv run ruff check .              # lint
```

Tests live under `tests/` and cover routers (`tests/routers/`), services, embedders, and CLI command parsing.

---

## License

MIT. See [LICENSE](LICENSE) if present, otherwise this project is released under the [MIT License](https://opensource.org/licenses/MIT).

## Contributing

Issues and PRs welcome on [GitHub](https://github.com/navistone/code-indexer-service/issues). See [`AGENTS.md`](AGENTS.md) for the conventions agents follow when working in this repo.
