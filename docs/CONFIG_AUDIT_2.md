# code-indexer-service — Config Audit (Wave 1.D)

> Pure analysis — no source or config code changed.
> Generated: 2026-05-03

---

## 1. Scripts Audit

### Bash Scripts

| Script | Purpose | Last Touch | Callers | Status | Notes |
|--------|---------|------------|---------|--------|-------|
| `scripts/launch-explorer.sh` | Start code-graph-rag CLI explorer | 2026-04-24 | Manual (dev exploration) | **KEEP** | Interactive symbol browser for debugging. |
| `scripts/lm_studio_smoke.py` | Test LM Studio connectivity + embedding + rerank models | 2026-04-27 | Manual (dev diagnostics) | **KEEP** | Validates LM Studio endpoint, model availability. |
| `scripts/ask_local_llm.py` | Interactive chat with LM Studio LLM | 2026-04-27 | Manual (dev exploration) | **KEEP** | Test prompt engineering, model behavior. |

### uv Scripts in pyproject.toml

**Status:** None defined.

**Observation:** Service uses direct CLI invocation (`uv run pytest`, `uv run ruff check`) instead of named scripts. This works but reduces discoverability.

**Recommendation:** Consider adding to `pyproject.toml [tool.uv.scripts]`:
```toml
[tool.uv.scripts]
test = "pytest"
lint = "ruff check . && ruff format --check ."
format = "ruff format ."
dev = "uvicorn app.main:app --reload --port 8000"
smoke = "python scripts/lm_studio_smoke.py"
```

---

## 2. Dependency Audit

### Production Dependencies (top 15 by integration risk)

| Dep | Version | Footprint | Usage | Risk | Status |
|-----|---------|-----------|-------|------|--------|
| fastapi | >=0.136.0 | 5.2 MB | Web framework (all routers) | **LOW** | ✅ required |
| pydantic | >=2.13.3 | 2.8 MB | Request/response models, Settings | **MEDIUM** | ✅ required |
| uvicorn | >=0.45.0 | 1.2 MB | ASGI server | **LOW** | ✅ required |
| ladybug | ==0.17.1 | 600 KB | Graph DB (`app/services/graph_db.py`) | **MEDIUM** | ✅ required |
| duckdb | >=1.1.0 | 18 MB | Embeddings + vector search | **MEDIUM** | ✅ required |
| torch | >=2.6.0 | 600+ MB | CodeRankEmbed via transformers | **HIGH** | ✅ required (optional; conditional) |
| transformers | >=4.0.0 | 1.8 GB | Embedding model loader | **HIGH** | ✅ required (optional; conditional) |
| httpx | >=0.28.1 | 400 KB | Async HTTP (LM Studio, GitHub) | **LOW** | ✅ required |
| pydantic-settings | >=2.3.0 | 150 KB | Env var parsing (Settings) | **LOW** | ✅ required |
| pydantic-core | >=2.28.0 | 800 KB | Pydantic internals | **LOW** | ✅ required |

### Dev Dependencies

| Dep | Purpose | Status |
|-----|---------|--------|
| httpx | Test HTTP client | ✅ required |
| pytest | Test runner | ✅ required |
| pytest-asyncio | Async test support | ✅ required |
| ruff | Linter + formatter | ✅ required (but not integrated into CI yet) |

### Dead Dependency Candidates

**All 25+ dependencies are actively used.** Verification:
- `torch`, `transformers` — conditionally loaded when `LM_STUDIO_URL` unset (optional embedding mode)
- `code-graph-rag` — editable dependency; correctly sourced via uv.sources
- `httpx` — used in LM Studio adapter + GitHub routes

**No dead dependencies detected.**

---

## 3. `.env.example` Completeness

### Variables Referenced in Code

| Variable | In .env.example | Code References | Status |
|----------|-----------------|-----------------|--------|
| `LADYBUG_DB_PATH` | ✅ L14 | `app/config.py`, lifespan setup | documented |
| `LADYBUG_BATCH_SIZE` | ✅ L15 | `app/services/graph_db.py` | documented |
| `TARGET_REPO_PATH` | ✅ L18 | `/index` handler default | documented |
| `HOST` | ✅ L21 | uvicorn startup | documented |
| `PORT` | ✅ L22 | uvicorn startup (default :8000) | documented |
| `GITHUB_TOKEN` | ✅ L26 | `app/routers/github.py` (list orgs) | documented |
| `GITHUB_ALLOWED_OWNERS` | ✅ L27 | `app/routers/github.py` (org allowlist) | documented |
| `LM_STUDIO_URL` | ✅ L44 (commented) | `app/services/lm_studio.py` (optional) | documented |
| `LM_STUDIO_EMBED_MODEL` | ✅ L51 (commented) | LM Studio adapter | documented |
| `LM_STUDIO_RERANK_MODEL` | ✅ L64 (commented) | LM Studio adapter | documented |
| `LM_STUDIO_TIMEOUT` | ✅ L65 (commented) | LM Studio adapter | documented |
| `AZURE_TENANT_ID` | ✅ L74 | Auth (Phase 1 M6) | documented |
| `AZURE_API_CLIENT_ID` | ✅ L75 | Backend JWT validation | documented |
| `AZURE_API_AUDIENCE` | ✅ L76 | Token validation | documented |
| `AZURE_JWKS_URI` | ✅ L77 | JWKS fetch + token validation | documented |
| `FORGE_ENV` | ✅ L85 | Environment gating (dev vs prod) | documented |
| `FORGE_SERVICE_AUTH_TOKEN` | ✅ L86 | Service-to-service auth | documented |
| `JOBS_DB_PATH` | ✅ L93 | Phase 2 jobs store (planned) | documented |
| `METRICS_ENABLED` | ✅ L98 | `app/metrics.py` | documented |
| `METRICS_PATH` | ✅ L99 | Prometheus endpoint | documented |

### Missing Variables (in code but not in .env.example)

| Variable | Code Reference | Recommendation |
|----------|-----------------|-----------------|
| `GH_TOKEN` | `app/routers/github.py` (fallback if GITHUB_TOKEN unset) | Add note to GITHUB_TOKEN section. Low priority (legacy fallback). |
| `METRICS_TOP_N_REPOS` | `app/metrics.py` (optional, debug metric) | Not required. Can add if metrics dashboard is formalized. |
| `CGR_DATA_DIR` | Optional fallback, used internally | Not exposed; internal detail. |

### Dead Variables (in .env.example but not used)

**None detected.** All 20+ documented vars are active or future-proofed (Phase 2 JOBS_DB_PATH).

---

## 4. Cross-Config Consistency

### Code Indexer ↔ TheForge Alignment

| Config Item | Code Indexer | TheForge | Consistency | Note |
|-------------|--------------|----------|-------------|------|
| **PORT (dev)** | 8000 (default) | CODE_INDEXER_BASE_URL=http://localhost:8003 | **DRIFT** | TheForge script forces :8003; indexer defaults :8000. Both docs mention this, but source of confusion. |
| **PORT (prod)** | 8000 (in Docker) | (code-indexer container) | ✅ match | Containerized deployment uses :8000. |
| **FORGE_ENV** | development (default) | development (default) | ✅ match | Both use for localhost-trust bypass. |
| **FORGE_SERVICE_AUTH_TOKEN** | (blank in dev) | (blank in dev) | ✅ match | Must be sync'd in prod (same value). Rotate every 90 days. |
| **AZURE_TENANT_ID** | L74 (optional) | L67 (required) | ✅ match | Indexer optional (Phase 1 future); TheForge required (prod). |
| **AZURE_API_CLIENT_ID** | L75 (optional) | L79 (required) | ✅ match | Both validate bearer tokens against same app. |
| **AZURE_API_AUDIENCE** | L76 (api://forge-api) | L81 (api://forge-api) | ✅ match | Tokens issued by TheForge; validated by indexer. |
| **AZURE_JWKS_URI** | L77 (indexed) | L202 (NOT indexed) | **GAP** | Indexer explicitly doc's it; TheForge omits it. Both work (derived from TENANT_ID), but inconsistent. |
| **LM_STUDIO_URL** | L44 (optional) | Code Indexer proxy (LM Studio not consumed by TheForge) | ✅ one-way | TheForge sends LM Studio config to indexer; doesn't need it locally. |

### Indexer ↔ code-graph-rag Alignment

| Config Item | Indexer | code-graph-rag | Consistency |
|-------------|---------|----------------|----|
| **LADYBUG_DB_PATH** | L14 (.cgr/graph.db) | Same default | ✅ match |
| **TARGET_REPO_PATH** | L18 (. = current) | Same default | ✅ match |
| **LM_STUDIO_URL** | L44 (optional) | Not consumed by cgr; host-level config | ✅ acceptable |

---

## 5. Recommendations (Max 5)

### 1. Add Explicit Port Clarification in Both `.env.example` Files (EASY) — Effort: S
**Files:** code-indexer-service/.env.example, TheForge/.env.example

**Problem:** Dev confusion about which port to use (8000 vs 8003).

**Action:**

*Code Indexer `.env.example` L22 (after PORT=8000):*
```
# Dev: TheForge script (scripts/start-indexer.sh) overrides to 8003.
#      To run standalone, set PORT=8003 or use `uvicorn --port 8003`.
# Prod: Docker container binds :8000 (mapped to host via docker-compose).
```

*TheForge `.env.example` L132 (after CODE_INDEXER_BASE_URL):*
```
# Dev note: Indexer defaults to :8000 but TheForge script forces :8003.
# If running indexer standalone, ensure PORT=8003 in indexer/.env matches CODE_INDEXER_PORT here.
```

**Effort:** 5 minutes.

---

### 2. Add AZURE_JWKS_URI to TheForge `.env.example` for Clarity (EASY) — Effort: S
**File:** TheForge/.env.example

**Problem:** Indexer explicitly documents AZURE_JWKS_URI; TheForge omits it. Maintenance debt.

**Action:** Add to TheForge/.env.example after AZURE_API_AUDIENCE (L81):
```
# JWKS discovery endpoint for bearer token validation.
# Derived from TENANT_ID if omitted (recommended).
# Manual override: https://login.microsoftonline.com/{TENANT_ID}/discovery/v2.0/keys
# AZURE_JWKS_URI=https://login.microsoftonline.com/...
```

**Effort:** 5 minutes.

---

### 3. Create `scripts/README.md` for Discoverability (EASY) — Effort: M
**File:** code-indexer-service/scripts/README.md (new)

**Problem:** 3 exploratory/diagnostic scripts exist; no documentation for new devs.

**Action:**
```markdown
# Diagnostic Scripts

## launch-explorer.sh
Interactive code-graph-rag symbol browser for debugging the graph DB.
Usage: `bash scripts/launch-explorer.sh`
Output: Opens browser to localhost:8080 with symbol explorer.

## lm_studio_smoke.py
Test LM Studio connectivity, embedding model, and rerank model.
Usage: `uv run python scripts/lm_studio_smoke.py`
Requires: LM Studio running on http://localhost:1234
Validates: embedding + rerank model in-memory loading, latency.

## ask_local_llm.py
Interactive chat interface for testing prompt engineering with LM Studio.
Usage: `uv run python scripts/ask_local_llm.py`
Requires: LM Studio running.
Example: Test new rerank prompts before shipping.
```

**Effort:** 30 minutes.

---

### 4. Add uv Scripts Section to pyproject.toml (MEDIUM) — Effort: M
**File:** code-indexer-service/pyproject.toml

**Problem:** Commands are run directly (`uv run pytest`); no named script group for consistency with other repos.

**Action:** Add after `[tool.pytest.ini_options]`:
```toml
[tool.uv.scripts]
test = "pytest"
lint = "ruff check ."
format = "ruff format ."
dev = "uvicorn app.main:app --reload --port 8000"
smoke = "python scripts/lm_studio_smoke.py"
```

**Benefit:**
- Developers run `uv run test` instead of `uv run pytest`
- Ruff linter now integrated into CI-ready pipeline
- Consistency with TheForge (which uses scripts)

**Effort:** 30 minutes (add section, document in DEVELOPMENT.md).

---

### 5. Document FORGE_SERVICE_AUTH_TOKEN Rotation in Both Repos (MEDIUM) — Effort: M
**Files:** docs/SERVICE_AUTH.md (both repos)

**Problem:** Both `.env.example` files mention "rotate every 90 days" but no procedure exists.

**Action:** Add section to docs/SERVICE_AUTH.md (if exists; create if not):
```markdown
## Service-to-Service Auth Token Rotation (90-day cycle)

Tokens are shared between TheForge and Code Indexer.
Rotation must maintain zero downtime.

### Pre-rotation
- Decide new token: `openssl rand -base64 32`
- Verify vault access to `forge-prod` in 1Password
- Plan a maintenance window (off-peak, <30 min)

### During rotation
1. Update FORGE_SERVICE_AUTH_TOKEN in vault
2. Deploy new TheForge instance (reads from vault)
3. Wait 5 min for all Forge instances to boot
4. Deploy new Code Indexer instance
5. Verify health checks pass on both
6. Monitor logs for auth errors (should be none)

### Rollback (if needed)
- Revert to old token in vault
- Redeploy both services
- Both services trust localhost in dev (FORGE_ENV=development)

### Post-rotation
- Update ticket with deployment timestamp
- Schedule next rotation (90 days from now)
```

**Effort:** 1 hour (write + review).

---

## Summary

**Scripts:** 3 diagnostic/exploratory scripts; all active. Missing formal uv scripts (opportunity).

**Dependencies:** 25+ total; all active. torch + transformers large but conditionally loaded (optional embedding mode). code-graph-rag editable dependency correctly sourced.

**.env.example:** Complete (20+ vars). One clarity gap (PORT 8000 vs 8003 in dev). One cross-repo gap (AZURE_JWKS_URI documented in indexer but not TheForge).

**Cross-config:** Mostly consistent. Auth vars match. Port drift needs clarification in both repos.

**Action items:**
1. Add port clarification to both `.env.example` files (easy, high impact)
2. Add AZURE_JWKS_URI to TheForge (easy, consistency)
3. Create `scripts/README.md` (medium, discoverability)
4. Add uv scripts section (medium, CI integration)
5. Document token rotation procedure (medium, operational safety)

**Effort estimate:** 3–4 hours for all 5 items.

