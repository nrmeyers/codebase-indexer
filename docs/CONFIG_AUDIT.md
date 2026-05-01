# Code Indexer Service — Config Audit (Wave 3.2)

Read-only analysis of scripts, dependencies, environment variables, and cross-config consistency.

## 1. Scripts Audit

### Shell Scripts

| Script | Purpose | Last Changed | Callers | Status |
|--------|---------|--------------|---------|--------|
| `scripts/launch-explorer.sh` | Start code-graph-rag CLI explorer (interactive analysis) | 2026-04-24 | Manual (dev exploration) | keep |
| `scripts/lm_studio_smoke.py` | Test LM Studio connectivity + embedding + rerank models | 2026-04-27 | Manual (dev diagnostics) | keep |
| `scripts/ask_local_llm.py` | Interactive chat with LM Studio LLM (testing prompt engineering) | 2026-04-27 | Manual (dev exploration) | keep |

### No uv scripts defined in pyproject.toml
The indexer uses direct Python runner or bash wrappers. All test/lint commands run via uv natively:
- `uv run pytest` (test group)
- `uv run ruff check` (linting, not yet integrated into CI)

**Observations:**
- No formal script section in pyproject.toml; all invocation is direct CLI.
- 3 exploratory/diagnostic scripts present; no dead code.
- Recommend documenting in a Makefile or scripts/README.md for team discoverability.

## 2. Dependency Audit

### Workspace / Editable Dependencies
- **`code-graph-rag`** (path: `../code-graph-rag`, editable): Core dependency. Dual sourcing: path in dev, PyPI in prod. Correct.

### Dev Dependency Group
- **`httpx>=0.28.1`**: Test HTTP client (for mocking).
- **`pytest>=9.0.3`, `pytest-asyncio>=1.3.0`**: Test runner + async support. Required.
- **`ruff>=0.15.11`**: Linter (not yet run in CI; candidate for integration).

### Production Dependencies Analysis

| Dep | Version | Footprint | References | Status |
|-----|---------|-----------|------------|--------|
| fastapi | >=0.136.0 | 5.2 MB | `app/main.py`, all routers | required |
| pydantic | >=2.13.3 | 2.8 MB | Settings, route models | required |
| uvicorn | >=0.45.0 | 1.2 MB | ASGI server | required |
| real-ladybug | >=0.15.3 | 600 KB | Graph DB (`app/services/graph_db.py`) | required |
| duckdb | >=1.1.0 | 18 MB | Embeddings + vector search | required |
| torch | >=2.6.0 | 600+ MB | CodeRankEmbed via transformers | optional (inference only) |
| transformers | >=4.0.0 | 1.8 GB | Embedding model loader | optional (inference only) |

### Dependency Chain Risks
- **torch + transformers:** Combined 2.4 GB. Pre-computed embeddings used at search time; torch is conditionally loaded only if `LM_STUDIO_URL` is unset. Acceptable for current phase.
- **code-graph-rag editable:** Critical dependency. Path source works in dev; PyPI fallback in prod (defined in uv.sources).

### Dead Dependency Candidates
**None detected.** All 25+ dependencies are actively used in `app/`.

## 3. `.env.example` Completeness

### Environment Variables Referenced in Code

| Variable | In .env.example | Code References | Status |
|----------|-----------------|-----------------|--------|
| LADYBUG_DB_PATH | ✓ | `app/config.py`, lifespan setup | documented |
| LADYBUG_BATCH_SIZE | ✓ | `app/services/graph_db.py` | documented |
| TARGET_REPO_PATH | ✓ | `app/config.py`, `/index` handler | documented |
| HOST | ✓ | uvicorn startup | documented |
| PORT | ✓ | uvicorn startup, default 8000 | documented |
| GITHUB_TOKEN | ✓ | `app/routers/github.py` | documented |
| GITHUB_ALLOWED_OWNERS | ✓ | `app/routers/github.py` | documented |
| LM_STUDIO_URL | ✓ | `app/services/lm_studio.py` (optional) | documented |
| LM_STUDIO_EMBED_MODEL | ✓ | LM Studio adapter | documented |
| LM_STUDIO_RERANK_MODEL | ✓ | LM Studio adapter | documented |
| LM_STUDIO_TIMEOUT | ✓ | LM Studio adapter | documented |
| AZURE_TENANT_ID | ✓ | Auth (Phase 1 M6) | documented |
| AZURE_API_CLIENT_ID | ✓ | Auth (Phase 1 M6) | documented |
| AZURE_API_AUDIENCE | ✓ | Token validation | documented |
| AZURE_JWKS_URI | ✓ | JWKS fetch for token validation | documented |
| FORGE_ENV | ✓ | Environment gating | documented |
| FORGE_SERVICE_AUTH_TOKEN | ✓ | Service-to-service auth | documented |
| JOBS_DB_PATH | ✓ | Phase 2 jobs store (planned) | documented |
| METRICS_ENABLED | ✓ | `app/metrics.py` | documented |
| METRICS_PATH | ✓ | Prometheus endpoint | documented |

### Missing Variables (in code but not in .env.example)
- **`GH_TOKEN`** (fallback in `app/routers/github.py` if GITHUB_TOKEN unset): Not documented. Recommend adding note to GITHUB_TOKEN section.
- **`METRICS_TOP_N_REPOS`** (optional, `app/metrics.py` line ~47): Not in .env.example. Low priority (debug metric); can be added if metrics are formalized.
- **`CGR_DATA_DIR`** (optional, fallback to LadybugDB parent): Not explicitly documented; used internally. Acceptable.

### Dead Variables (in .env.example but not referenced)
**None detected.** Example is comprehensive and current.

## 4. Cross-Config Consistency

### Code Indexer ↔ TheForge Alignment

| Config Item | Code Indexer | TheForge | Consistency |
|-------------|--------------|----------|-------------|
| **Port (dev)** | PORT=8000 (default) | CODE_INDEXER_PORT=8003 (override) | **DRIFT** |
| **Port (prod)** | PORT=8000 | Containerized, same | ✓ match |
| **FORGE_ENV** | `development` (default) | `development` (default) | ✓ match |
| **FORGE_SERVICE_AUTH_TOKEN** | Optional (dev blank) | Optional (dev blank) | ✓ match |
| **Base URL consumption** | N/A (indexer doesn't consume) | CODE_INDEXER_BASE_URL | ✓ one-way |
| **Auth (Phase 1 M6)** | AZURE_* vars | AZURE_* vars (M6) | ✓ match |

**DRIFT FINDING:** Code Indexer `.env.example` line 22 defaults PORT=8000, but TheForge script `start-indexer.sh` passes `--port 8003`. The prod containerized deployment uses :8000. **Recommendation:** Add a note in .env.example clarifying:
- Dev override: Start with `uvicorn app.main:app --port 8003` (or set PORT=8003 in .env)
- Prod Docker: Uses PORT=8000 (mapped to container)

### Indexer ↔ code-graph-rag Alignment

| Config Item | Code Indexer | code-graph-rag | Consistency |
|-------------|--------------|----------------|-------------|
| **LADYBUG_DB_PATH** | `.cgr/graph.db` (default) | Same default | ✓ match |
| **TARGET_REPO_PATH** | `.` (current) | Same default | ✓ match |
| **LM Studio URL** | `http://localhost:1234` (optional) | Not consumed by cgr; host-level config | ✓ acceptable |

## 5. Recommendations

### Priority 1 (Easy, High Impact)

**1.1 — Clarify dev vs prod port in .env.example**
- **File:** `.env.example` line 22
- **Change:** Add comment:
  ```
  # Dev: override via uvicorn --port 8003 or set PORT=8003 in .env (matches TheForge)
  # Prod: Docker container runs :8000, mapped to host via docker-compose
  PORT=8000
  ```
- **Effort:** S (comment only, ~2 lines)

**1.2 — Document missing GH_TOKEN fallback**
- **File:** `.env.example` section "GitHub integration" line 26
- **Change:** Add:
  ```
  # Fallback: if GITHUB_TOKEN is unset, service tries GH_TOKEN (rare, legacy support)
  ```
- **Effort:** S (1 line)

### Priority 2 (Medium, Polish)

**2.1 — Add scripts/README.md for discoverability**
- **File:** `scripts/README.md` (new)
- **Change:** Document 3 exploratory scripts: usage, when to use, sample output
- **Effort:** M (30 lines, examples)

**2.2 — Formalize test/lint/dev targets**
- **File:** Create `Makefile` (optional) or `pyproject.toml [tool.uv.scripts]`
- **Change:** Add:
  ```
  test = "pytest"
  lint = "ruff check ."
  format = "ruff format ."
  dev = "uvicorn app.main:app --reload --port 8000"
  ```
- **Effort:** M (integrates ruff into CI-ready pipeline)

### Priority 3 (Technical Debt)

**3.1 — Expose METRICS_TOP_N_REPOS in .env.example**
- **File:** `.env.example` after METRICS_PATH
- **Change:**
  ```
  # Max repos to display in /metrics dashboard (debugging)
  # METRICS_TOP_N_REPOS=10
  ```
- **Effort:** S (1 line)

---

## Summary

- **Scripts:** 3 diagnostic/exploratory scripts; no dead code. Missing formal uv scripts section (opportunity for discoverability).
- **Dependencies:** 25 total, all active. torch + transformers are large but conditionally loaded (dev only). Editable code-graph-rag correctly sourced.
- **.env.example:** Complete; covers 19+ active vars. Two minor gaps (GH_TOKEN fallback, METRICS_TOP_N_REPOS).
- **Cross-config:** One clarity issue (port 8000 vs 8003 in dev flow). Prod is correct.

**Action items:** 1 doc fix (port clarification), 1 optional improvement (scripts/README), 1 future (formalize uv scripts).
