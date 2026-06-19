# codebase-indexer — Install Runbook

## What this installs

A local code-intelligence service (`code-indexer`) consisting of:

- A **FastAPI daemon** (port 8003 by default) that exposes HTTP endpoints for indexing and querying repos.
- A **`code-indexer` CLI** that manages the daemon, submits index jobs, and runs searches — works from any directory once installed to PATH.
- Three per-repo stores built on first `index`: a typed symbol graph (**LadybugDB**, embedded kuzu), a vector store (**DuckDB**, FLOAT[768]), and a lexical index (**Tantivy** BM25).
- An **in-process embedder** (nomic-embed-text-v1.5, 768-dim via sentence-transformers) — downloaded once from Hugging Face on first index, then cached locally. No separate embedding server required.

---

## Standalone install (recommended)

Installs the `code-indexer` binary to PATH so it works from any directory.

```bash
# Requires uv (https://docs.astral.sh/uv/getting-started/installation/)
uv tool install git+https://github.com/nrmeyers/codebase-indexer.git

# Or with pipx:
pipx install git+https://github.com/nrmeyers/codebase-indexer.git
```

Verify the binary is on PATH:

```bash
code-indexer --help
```

If `code-indexer` is not found, ensure `~/.local/bin` is in your `$PATH`:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc  # or ~/.zshrc
source ~/.bashrc
```

---

## First-run setup

```bash
code-indexer setup        # writes config; prompts for port, embedder backend, data dir
code-indexer start        # spawns daemon in background on port 8003
code-indexer status       # confirm daemon is live + no repos indexed yet
```

Config is written to `~/.code-indexer/config.toml`. The daemon PID and log are at `~/.code-indexer/server.pid` and `~/.code-indexer/server.log`.

---

## User-scoped data paths

When installed and run from any directory, all state is user-scoped — nothing is scattered in your current working directory:

| What | Default path |
|------|-------------|
| Config | `${XDG_CONFIG_HOME:-~/.config}/codebase-indexer/` (or `~/.code-indexer/config.toml`) |
| Graph/vector/lexical indexes + jobs DB | `${XDG_DATA_HOME:-~/.local/share}/codebase-indexer/` |

**Backward-compatibility exception:** if a `.cgr/` directory already exists in the working directory (e.g. a service-in-a-folder deployment), it is used instead of the user-scoped paths.

---

## Index and search a repo

```bash
# Start the daemon if not already running
code-indexer start

# Index a repo (blocks until the job finishes; shows progress)
code-indexer index /path/to/your/repo

# Semantic search — human output
code-indexer search "where is auth handled" -k 5

# Machine-readable JSON (for agents/harnesses — use the global --json flag)
code-indexer --json search "where is auth handled" -k 5
code-indexer --json list
```

The global `--json` flag routes all human-facing output to stderr and emits a single JSON document to stdout. Use it when driving the CLI from an agent or script.

---

## Claude skill (optional)

Wire the indexer into Claude Code so a Claude agent can drive it directly:

```bash
bash integrations/claude/install.sh
```

This symlinks the harness skill into `~/.claude/skills/`. See `integrations/claude/README.md` for the full skill API and usage notes.

---

## Container option

Build and run with podman (preferred) or docker:

```bash
podman build -t codebase-indexer .
podman run -p 8003:8000 \
  -e HOST=0.0.0.0 \
  -v codebase-indexer-data:/var/lib/forge \
  codebase-indexer
```

The service listens on **8000 inside the container**; map it to whichever host port you prefer (example above uses 8003 to match the CLI default). Set `HOST=0.0.0.0` so the service binds on all interfaces inside the container.

Index data persists in the named volume `/var/lib/forge` across container restarts.

---

## Contributor / editable install

```bash
git clone https://github.com/nrmeyers/codebase-indexer.git
cd codebase-indexer
uv sync
uv tool install --editable .
```

Run tests:

```bash
uv run pytest tests/cli -q          # fast, hermetic (mocked HTTP)
uv run pytest tests/ -v             # full suite (needs a running daemon for some tests)
```

---

## Upgrading

```bash
uv tool install --force git+https://github.com/nrmeyers/codebase-indexer.git
# or:
pipx upgrade codebase-indexer
```

---

## Verify

```bash
code-indexer status        # daemon running + list of indexed repos
code-indexer --json list   # JSON list of indexed repos (empty on fresh install)
code-indexer --json search "authentication" -k 3   # returns hits after first index
```

A healthy `status` shows the daemon pid and the service responding at `http://localhost:8003`. A search returning results (score + symbol + type) confirms the full stack — daemon, embedder, and index — is working.
