# code-indexer-service — Testing Gap Analysis (Wave 2.2)

> Pure analysis — no source or test code changed.
> Generated: 2026-04-30

---

## 1. Coverage Map

Top-20 files ranked by logic density. The service has 13 test files covering approximately 9 of 15 source modules.

| # | File | Lines | Status | Notes |
|---|------|------:|--------|-------|
| 1 | `app/routers/repos.py` | 303 | ❌ Untested | `/repos/{name}/stats` and `/repos/{name}/reindex` — no dedicated test file; disk reads + subprocess calls are the riskiest paths |
| 2 | `app/routers/websocket.py` | 219 | ❌ Untested | WebSocket `/ws` progress stream — polling loop, event synthesis, disconnect handling all untested |
| 3 | `app/routers/symbols.py` | 200 | ⚠️ Partially | `test_integration.py:test_symbol_lookup_real_db` is the only coverage; callers/callees endpoints and FQN URL decoding have no dedicated assertions |
| 4 | `app/routers/search.py` | ~180 est. | ✅ Well tested | `test_search.py` covers structural search, semantic search, symbol lookup, fast-fail, 503 paths |
| 5 | `app/services/source_fetch.py` | 127 | ❌ Untested | Central source-snippet resolver used by `/context-bundle` and semantic rerank — no tests at all |
| 6 | `app/routers/index.py` | ~120 est. | ✅ Well tested | `test_index.py` covers POST/GET lifecycle, force-reindex, duplicate-409, blocking-embed read-only |
| 7 | `app/routers/context_bundle.py` | ~100 est. | ✅ Well tested | `test_context_bundle.py` covers results, empty, depth-zero, 503, nonexistent-repo |
| 8 | `app/services/bm25_index.py` | ~100 est. | ✅ Well tested | `test_bm25_service.py` covers tokenisation, empty query, cache-invalidation on mtime change |
| 9 | `app/services/lm_studio.py` | ~200 est. | ✅ Well tested | `test_lm_studio.py` covers base URL, embed, chat, model resolution, HTTP error bodies |
| 10 | `app/services/reranker.py` | ~180 est. | ✅ Well tested | `test_reranker.py` has 18 cases covering parse, prompt build, permutation, fallbacks |
| 11 | `app/services/jobs_store.py` | ~120 est. | ✅ Well tested | `test_jobs_store.py` covers CRUD, partial update, terminal state, isolation, sweep |
| 12 | `app/routers/github.py` | ~80 est. | ✅ Well tested | `test_github_orgs.py` + `test_github_allowlist.py` cover auth, allowlist, sorting, case-insensitive match |
| 13 | `app/routers/explorer.py` | ~80 est. | ✅ Well tested | `test_explorer.py` covers unavailable/empty/available DB states, launch command, viewer URL |
| 14 | `app/routers/health.py` | ~60 est. | ✅ Well tested | `test_health.py` covers ok, indexed repos, DB error, LM Studio block present/absent/failure |
| 15 | `app/services/source_fetch.py` | 127 | ❌ Untested | (same as #5 — no entry in any test file) |
| 16 | `app/routers/disk.py` | 90 | ⚠️ Partially | Covered only indirectly through integration tests; no dedicated route-level assertions on the disk endpoints |
| 17 | `app/metrics.py` | ~80 est. | ✅ Well tested | `test_metrics.py` covers Prometheus format, dashboard contract, disabled 404, label cap |
| 18 | `app/config.py` | ~50 est. | ❌ Untested | Settings class — validation boundaries (env var parsing, defaults, clamps) have no direct tests |
| 19 | `app/models.py` | ~80 est. | ❌ Untested | Pydantic response models — field validators / serialisers untested |
| 20 | `app/main.py` | ~40 est. | ⚠️ Partially | App factory covered by integration smoke; lifespan event and router registration ordering untested |

---

## 2. Skipped / Disabled Tests

No `pytest.mark.skip`, `pytest.mark.xfail`, `@skip`, or `unittest.skip` decorators found in any test file.

No skipped tests in this repo. This is good, but it means flaky or hard-to-run tests have simply been deleted rather than marked — see §3 for the slow-test cases that remain.

---

## 3. Slow / Flaky Tests

### `tests/test_index.py:75–103` — real polling loop (5 s deadline)

```python
deadline = time.time() + 5
while time.time() < deadline:
    status_resp = client.get(f"/index/{job_id}/status")
    if status_resp.json()["status"] == "done":
        break
    time.sleep(0.05)
```

**Why it's slow/flaky:** The test uses `_fake_blocking` to avoid real indexing, but still spins the polling loop waiting for the background thread to update `job.status`. Under slow CI runners the background thread can lag and the test burns 5 real seconds.

**Recommendation:** After calling `_fake_blocking` via `patch`, call `GET /index/{id}/status` once and assert directly — the patched function runs synchronously in the test thread so the status is already `"done"` before the first poll. Remove the polling loop entirely.

---

### `tests/test_bm25_service.py:120` — `time.time() + 5` mtime manipulation

```python
future = time.time() + 5
os.utime(db, (future, future))
```

**Why it's slow:** Not actually slow (no sleep), but sets mtime 5 seconds in the future, which can surprise filesystem-level caches on some macOS AFP/SMB mounts and cause intermittent cache-miss mismatches.

**Recommendation:** Use `time.time() + 1` or a fixed `mtime + 1` delta. The BM25 service checks `mtime != cached_mtime`, not `mtime > now`, so any distinct value works. Lower future offset reduces any edge-case clock interaction.

---

### `tests/test_integration.py:144–150` — polling loop (5 s deadline)

```python
deadline = time.time() + 5
while time.time() < deadline:
    ...
    time.sleep(0.05)
```

**Why it's slow/flaky:** Same pattern as `test_index.py`. The live `TestClient` integration test waits for a background indexer thread — can burn up to 5 s on slow CI.

**Recommendation:** This is a genuine integration test (real indexer, real DuckDB). Acceptable as-is, but should be isolated to an `integration` pytest marker so unit test runs (`pytest -m "not integration"`) stay fast.

---

### `tests/test_explorer.py` — 6 `TestClient(create_app())` instantiations

Each test function in `test_explorer.py` calls `create_app()` directly, constructing a fresh FastAPI app per test. This is correct for isolation but means 6 full application startups.

**Recommendation:** Convert to a module-scoped `@pytest.fixture` (or session-scoped if state isolation allows) that creates the app once and injects the `TestClient`. Reduces startup overhead by ~5x.

---

## 4. Missing High-Value Coverage

Ranked by **risk × likelihood**:

| Rank | File / Area | Risk | Why |
|------|------------|------|-----|
| 1 | `app/services/source_fetch.py` | CRITICAL | Used by both `/context-bundle` and semantic rerank. A regression in file-path resolution or line-range slicing silently returns empty snippets, degrading all AI-grounded context without any error. 127 lines of pure logic — trivial to unit-test with `tmp_path`. |
| 2 | `app/routers/repos.py` — `/repos/{name}/reindex` | HIGH | Wipes LadybugDB + DuckDB then re-schedules 4 indexing passes. A bug in the wipe-then-reindex sequence corrupts the graph DB. Calls `subprocess` (git SHA) with no test coverage. |
| 3 | `app/routers/websocket.py` | HIGH | 219 lines of stateful WebSocket logic: 1 Hz polling loop, event deduplication via `_last_seen`, disconnect cleanup. Untested disconnection path can leak asyncio tasks. |
| 4 | `app/routers/symbols.py` — callers/callees endpoints | MEDIUM | The `/symbols/{fqn}/callers` and `/symbols/{fqn}/callees` endpoints run two-pass UNION ALL Cypher queries. Route-order bug (documented in file header) is a real footgun — a test asserting correct routing would pin it. |
| 5 | `app/config.py` | MEDIUM | `Settings` class parses env vars including `GITHUB_TOKEN`, `LM_STUDIO_BASE_URL`, clamp logic for `REQUEST_TIMEOUT`. A misconfigured env var silently uses the wrong default. A `monkeypatch`-based test takes ~5 min to write. |
| 6 | `app/routers/repos.py` — `/repos/{name}/stats` | MEDIUM | Reads LadybugDB node/edge counts and returns them as dashboard sidebar facts. A wrong label or a DB schema change produces silent zero-counts rather than an error. |
| 7 | `app/models.py` | LOW-MEDIUM | Pydantic v2 models — alias serialisation and `model_validator` logic is untested. A field rename or alias change would not be caught until a real request arrives. |
| 8 | `app/routers/disk.py` | LOW | Disk-management endpoints covered only via integration path. Dedicated route-level test would catch 404-vs-400 shape regressions. |
| 9 | `app/main.py` — lifespan event | LOW | App lifespan startup (DB path setup, settings validation) is never directly tested. Would catch boot-time misconfiguration earlier. |
| 10 | `tests/test_integration.py` — `test_structural_search_real_db` | LOW | This is a test, not a gap — but it has no assertion on *which* nodes are returned, only that `len > 0`. A more precise fixture would catch query regressions. |

---

## 5. Test Infrastructure Suggestions

### 1. Shared `@pytest.fixture` for `TestClient` (repeated 8+ times)

**Pattern observed in:** `test_search.py`, `test_github_orgs.py`, `test_github_allowlist.py`, `test_index.py`, `test_health.py`, `test_context_bundle.py`, `test_metrics.py`, `test_integration.py`

Every file does one of:
```python
client = TestClient(app)   # module-level singleton
# or
with TestClient(app) as client:  # per-test context manager
```

**Proposed path:** `tests/conftest.py`

```python
import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c
```

A single module-scoped client eliminates 8+ duplicated setup blocks and makes test isolation explicit via `scope`.

---

### 2. `pytest.ini` / `pyproject.toml` integration marker

**Pattern observed in:** `test_integration.py` — contains 4 tests that spin up real DuckDB/LadybugDB instances and poll with `time.sleep`.

**Proposed addition to `pyproject.toml`:**

```toml
[tool.pytest.ini_options]
markers = [
    "integration: marks tests that require a full app stack (deselect with '-m not integration')",
    "slow: marks tests with real I/O or sleep loops",
]
```

Then mark the relevant tests:

```python
@pytest.mark.integration
def test_index_lifecycle(...): ...
```

This lets CI run `pytest -m "not integration"` for the fast loop and `pytest` (all) for pre-merge.

---

### 3. Shared `tmp_path`-backed `db_path` fixture (repeated 4+ times)

**Pattern observed in:** `test_integration.py` (fixture at line 62), `test_search.py` (inline at lines 102, 140, 171, 197), `test_bm25_service.py` (inline at line 74+).

All create a `tmp_path / "something.duck"` or `tmp_path / ".cgr/graph.db"` with slightly different naming. A shared fixture in `conftest.py` would centralise this:

```python
@pytest.fixture()
def fresh_duck(tmp_path: Path) -> Path:
    return tmp_path / "test.duck"

@pytest.fixture()
def fresh_db(tmp_path: Path) -> str:
    return str(tmp_path / ".cgr" / "graph.db")
```

---

### 4. `source_fetch` test module (new file needed)

`app/services/source_fetch.py` is the highest-risk untested file (rank #1 above). Its public API is a pure function over filesystem reads — ideal for `tmp_path`-based unit tests with no mocking needed. Proposed path: `tests/test_source_fetch.py`.
