---
name: codebase-indexer
description: Search and navigate an already-indexed codebase by meaning rather than text — semantic/lexical search, symbol lookup, caller/callee call-graph tracing, and grounded context bundles — via the local code-indexer service. Use when grep or reading files would be slow, or when you need to find code by intent, trace who-calls-what, or assemble cross-file context for a task.
---

## What this is

An HTTP code-intelligence service backed by a graph DB (LadybugDB/kuzu) and a vector store (DuckDB). The agent drives it through the `code-indexer` CLI binary using the global `--json` flag for machine-readable output. No MCP required.

---

## Prerequisite — ensure the daemon is up

```bash
code-indexer --json status
```

Returns health JSON including `status`, `alive`, `daemon_pid`, and `indexed_repos`. If the service is unreachable:

```bash
code-indexer start        # backgrounds a daemon on port 8003
code-indexer stop         # stop the daemon
```

---

## Core query commands

**The global `--json` flag MUST come before the subcommand:**

```
code-indexer --json <subcommand> [args]
```

Stdout is exactly one JSON document. Human-facing chatter goes to stderr. Non-zero exit on error — always check the exit code.

### list

```bash
code-indexer --json list
```

Returns:
```json
{"repos": [{"slug": "...", "status": "fresh|stale|indexed|unindexed", "repo_path": "...", "last_indexed_at": "..."}]}
```

Use this to discover available repos and their exact slugs. The `slug` field is the canonical identifier (`{org}__{repo}`) used by all other commands.

### search

```bash
code-indexer --json search "<natural language query>" -k 10 [--repo <slug>]
```

Returns:
```json
{"results": [{"symbol": "fully.qualified.name", "score": 0.87, "type": "Function"}]}
```

`--repo` scopes to one repo slug (optional — omit to search the first indexed repo). `-k` defaults to 10.

### symbol

```bash
code-indexer --json symbol <fully.qualified.name> [--repo <slug>]
```

Returns the symbol record: `qualified_name`, `file`, `line_start`, `line_end`, `source` (code snippet read from disk), `docstring`.

### callers

```bash
code-indexer --json callers <fqn> [--repo <slug>]
```

Returns:
```json
{"results": [{"qualified_name": "...", "file_path": "...", "line_number": 42}]}
```

### callees

```bash
code-indexer --json callees <fqn> [--repo <slug>]
```

Same shape as callers. Lists every symbol that `fqn` calls.

### bundle

```bash
code-indexer --json bundle "<task description>" --repo <PATH-to-repo-on-disk> [--k 10] [--depth 2]
```

Returns:
```json
{
  "symbols": ["ordered.by.relevance", "..."],
  "source_snippets": {"fqn": "source code..."},
  "call_graph": {"caller": ["callee1", "callee2"]},
  "total_tokens": 3200,
  "scores": {"fqn": 0.91}
}
```

**IMPORTANT:** For `bundle`, `--repo` is a **filesystem PATH** to the repo on disk, not a slug. For all other commands (`search`, `symbol`, `callers`, `callees`), `--repo` is a slug.

`--k` controls seed symbols from semantic search (default 10, max 50). `--depth` controls call-graph hop expansion (default 2, max 4).

---

## Indexing

```bash
# Index a repo for the first time (polls to completion)
code-indexer --json index <path-to-repo> [--force]

# Force a clean re-index of a known repo by slug
code-indexer --json reindex <slug>

# Drop an index
code-indexer remove <slug> --yes
```

`index` polls until the job finishes and emits final job-status JSON on completion with fields including `status`, `node_count`, `rel_count`.

---

## Slugs

A repo's canonical slug is `{org}__{repo}` derived from its git remote origin (e.g. `nrmeyers__TheForge`). Local repos indexed by path get a slug from the directory name. Always discover the exact slug with `list` before using it in other commands.

---

## HTTP API fallback

When the CLI is not available, the service is reachable directly over HTTP. The CLI daemon defaults to `http://127.0.0.1:8003`; the TheForge deployment uses `http://127.0.0.1:8000`.

| Method | Path | Key params / body | Returns |
|--------|------|-------------------|---------|
| `GET` | `/health` | — | `{status, indexed_repos[], repos[], running_jobs, embedder}` |
| `GET` | `/repos` | — | `{repos: [{slug, full_name, status, repo_path, last_indexed_at, last_indexed_sha, indexed}]}` |
| `GET` | `/repos/{name}/stats` | path: repo slug | size, fragment count, node/edge counts, last_indexed_at |
| `POST` | `/index` | body: `{repo_path, force_reindex?, exclude_paths?}` | `202 {job_id}` |
| `GET` | `/index/{job_id}/status` | path: job_id | `{job_id, status, phase, progress_pct, node_count, rel_count, error?}` |
| `GET` | `/search/semantic` | `?q=<query>&k=10&repo=<slug>` | `{results: [{symbol, score, type}]}` |
| `GET` | `/search/symbol` | `?fqn=<name>&repo=<slug>` | `{qualified_name, file, line_start, line_end, source, docstring}` |
| `GET` | `/symbols/{fqn}` | path: fqn (URL-encoded); `?repo=<slug>` | same as `/search/symbol` |
| `GET` | `/symbols/{fqn}/callers` | path: fqn; `?repo=<slug>` | `{results: [{qualified_name, file_path, line_number}]}` |
| `GET` | `/symbols/{fqn}/callees` | path: fqn; `?repo=<slug>` | same shape as callers |
| `POST` | `/context-bundle` | body: `{repo_path, task_description, k?, depth?, intent?, rerank?}` | `{symbols, source_snippets, call_graph, total_tokens, scores}` |
| `POST` | `/repos/{name}/reindex` | path: slug; body: `{force: true}` | `202 {job_id}` |
| `DELETE` | `/repos/{name}` | path: slug | deletion confirmation |

The service also serves `GET /openapi.json` for the full schema.

---

## Tips

- Prefer `search` over grep for "find code about X" — it matches by meaning, not text.
- Prefer `callers`/`callees` to trace call graphs — faster than reading files and following imports.
- Prefer `bundle` to assemble grounded multi-file context before editing — it seeds semantically and expands through the call graph, then estimates token cost.
- Keep `-k` small (5–10) to limit token consumption. Use `--depth 1` for tight, focused bundles.
- If a search returns unhelpful results, try rephrasing as a function description rather than a filename or concept: "function that retries HTTP requests" rather than "retry logic".
