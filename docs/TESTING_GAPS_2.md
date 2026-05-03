# code-indexer-service — Testing Gap Analysis (Wave 1.D)

> Pure analysis — no source or test code changed.
> Generated: 2026-05-03

---

## 1. Coverage Delta vs Wave 2.2 Audit (2026-04-30)

**Baseline:** `docs/TESTING_GAPS.md` (wave 2.2, 3 days old)

**Changes since baseline:**
- **Test files:** 13 → 13 (no new test files)
- **Coverage:** 9/15 source modules → still 9/15 (no regression, no improvement)
- **Skipped tests:** 0 then, 0 now (no change)
- **Slow tests:** 4 polling loops identified, **still present** (no fixes applied)

**Regression:** None. **Stagnation:** Yes — High-risk files remain untested:
- `app/routers/repos.py` (303 lines) — still untested
- `app/routers/websocket.py` (219 lines) — still untested
- `app/services/source_fetch.py` (127 lines) — still untested

**New commits (30 days):** 20 commits. Most are performance/integration fixes; **zero added test coverage** for untested modules.

---

## 2. Top 10 Untested High-Risk Files

Same as wave 2.2 baseline (no changes to coverage status).

| Rank | File | Lines | Risk | Why | Status |
|------|------|------:|------|-----|--------|
| 1 | `app/routers/repos.py` — `/repos/{name}/stats` | 303 | **CRITICAL** | Disk reads + LadybugDB node/edge counts. A schema change silently returns zeros. Used by dashboard sidebar. **Still untested.** | ❌ |
| 2 | `app/routers/repos.py` — `/repos/{name}/reindex` | 303 | **CRITICAL** | Wipes graph DB + DuckDB, reschedules 4 passes. Subprocess calls (git SHA). Untested disconnect/failure paths. **Still untested.** | ❌ |
| 3 | `app/routers/websocket.py` — `/ws` streaming | 219 | **HIGH** | 1 Hz polling loop, event deduplication, 4-event types. Untested disconnect cleanup. Asyncio task leak risk. **Still untested.** | ❌ |
| 4 | `app/services/source_fetch.py` | 127 | **CRITICAL** | Pure-function file-path resolver used by `/context-bundle` + semantic rerank. Empty-snippet regression goes silent. **Still untested.** | ❌ |
| 5 | `app/routers/symbols.py` — `/symbols/{fqn}/callers` | 200 | **MEDIUM** | Two-pass UNION ALL Cypher queries. Route-order bug documented in file. Untested endpoint routing. **Partially tested.** | ⚠️ |
| 6 | `app/config.py` — Settings class | 50 | **MEDIUM** | Env var parsing, defaults, clamps. Misconfigured timeout silently uses wrong default. **Still untested.** | ❌ |
| 7 | `app/routers/disk.py` | 90 | **LOW-MEDIUM** | Disk management endpoints. Covered only via integration path. No route-level 404 vs 400 shape tests. **Still partially tested.** | ⚠️ |
| 8 | `app/models.py` — Pydantic v2 | 80 | **LOW-MEDIUM** | Alias serialization, field validators untested. A rename would not be caught until prod. **Still untested.** | ❌ |
| 9 | `app/main.py` — lifespan event | 40 | **LOW** | App startup (DB path setup, settings validation) untested. Boot-time misconfiguration goes undetected. **Still untested.** | ❌ |
| 10 | `app/routers/disk.py` — disk endpoints | 90 | **LOW** | Route-level assertions missing (404 vs 400). Covered only indirectly. **Still partially tested.** | ⚠️ |

**Trend:** No improvements. Wave 2.2 recommendations were **not acted on** in the past 3 days.

---

## 3. Skipped / Disabled Tests

**Same as wave 2.2:** Zero `pytest.mark.skip`, `pytest.mark.xfail`, `@skip`, or `unittest.skip` in any test file.

**Status:** ✅ No regressions. Good — no hidden flakiness.

---

## 4. Slow / Flaky Tests (Wave 2.2 Findings — Still Present)

### Test 1: `tests/test_index.py:75–103` — Real polling loop (5s deadline)

**Status:** ❌ **Still present** (no fix applied in 3 days)

```python
deadline = time.time() + 5
while time.time() < deadline:
    status_resp = client.get(f"/index/{job_id}/status")
    if status_resp.json()["status"] == "done":
        break
    time.sleep(0.05)
```

**Fix available from wave 2.2:** Remove polling loop after `_fake_blocking()` patch; assert directly on first status call.

---

### Test 2: `tests/test_bm25_service.py:120` — mtime manipulation (future offset)

**Status:** ⚠️ **Still present** (5-second future offset could cause flakiness on SMB mounts)

**Recommendation:** Use `time.time() + 1` instead of `+ 5`.

---

### Test 3: `tests/test_integration.py:144–150` — Integration polling loop (5s deadline)

**Status:** ❌ **Still present** (genuine slow integration test; acceptable but should be marked)

**Recommendation:** Add `@pytest.mark.integration` so CI can skip with `-m "not integration"`.

---

### Test 4: `tests/test_explorer.py` — 6 app startups per test

**Status:** ❌ **Still present** (6 `create_app()` calls, no fixture consolidation)

**Recommendation:** Convert to module-scoped fixture (5x speedup).

---

## 5. New Test Gaps from Recent Merges (last 30 days)

**Commits analyzed:** 20 (2026-04-03 to 2026-05-03)

### New files shipped without tests:

None detected. Recent commits are refactors + performance fixes to existing modules:
- `perf/e2e-cycle-N` — improvements to existing `tests/test_integration.py`
- `refactor/unify-jobs-store` — refactors existing `app/services/jobs_store.py` (already tested)
- `fix/embedding-count-display` — tweak to `app/routers/index.py` (already tested)

**Status:** ✅ No new untested modules shipped in last 30 days. Existing gaps remain.

---

## 6. Recommendations (Max 5)

### 1. Implement `tests/test_source_fetch.py` (CRITICAL) — Effort: S
**File:** `tests/test_source_fetch.py` (new)

**Rank:** #1 untested high-risk file (pure function, trivial to test)

**Why:** 127-line pure function. Used by `/context-bundle` + semantic rerank. A regression returns empty snippets silently.

**What to cover (3–5 test cases):**
- Valid file path → correct snippet extracted
- Out-of-bounds line range → clamp to file bounds or error gracefully
- Missing file → appropriate error (FileNotFoundError)
- Empty file → empty snippet (no crash)
- Large file → performance acceptable (< 100 ms per call)

**Template:**
```python
import pytest
from app.services.source_fetch import fetch_snippet

def test_fetch_snippet_valid_range(tmp_path):
    # Create temp file with 10 lines
    file = tmp_path / "test.py"
    file.write_text("\n".join(f"# Line {i}" for i in range(1, 11)))
    snippet = fetch_snippet(str(file), 2, 5)
    assert "Line 2" in snippet
    assert "Line 5" in snippet

def test_fetch_snippet_out_of_bounds(tmp_path):
    file = tmp_path / "test.py"
    file.write_text("a\nb\nc\n")
    snippet = fetch_snippet(str(file), 5, 100)  # Beyond file
    assert snippet is not None  # Should clamp gracefully

def test_fetch_snippet_missing_file():
    with pytest.raises(FileNotFoundError):
        fetch_snippet("/nonexistent/file.py", 1, 5)
```

**Effort:** 1–2 hours (write + review).

---

### 2. Add `tests/test_repos_py` for `/repos/{name}/reindex` (HIGH) — Effort: M
**File:** `tests/test_repos_py.ts` (new) — actually should be Python

**Rank:** #2 high-risk (wipes DB + spawns subprocess)

**Why:** `/repos/{name}/reindex` wipes graph DB and DuckDB, reschedules indexing. A bug in the wipe-then-reindex sequence corrupts the graph.

**What to cover (4–6 test cases):**
- Successful reindex workflow (wipe → resched)
- Reindex on non-existent repo → 404
- Concurrent reindex attempts → 409 (already indexing)
- Subprocess failure (git SHA fetch fails) → 500
- Database state after reindex cancel → consistent (no orphaned jobs)

**Template:**
```python
def test_reindex_wipes_and_reschedules(client, fresh_db):
    # 1. Index repo
    client.post("/index", json={"repo_path": "/path"})
    # 2. Trigger reindex
    resp = client.post("/repos/test/reindex")
    assert resp.status_code == 202  # Accepted
    # 3. Verify DB was wiped (node count = 0 before resched completes)

def test_reindex_nonexistent_repo(client):
    resp = client.post("/repos/nonexistent/reindex")
    assert resp.status_code == 404
```

**Effort:** 2–3 hours (setup mocks for subprocess, DB queries).

---

### 3. Add `@pytest.mark.integration` to Slow Tests (EASY) — Effort: S
**Files:** `tests/test_integration.py`, `tests/test_index.py` (selected tests)

**Why:** CI can run fast unit tests (`-m "not integration"`) and slow integration tests separately.

**Action:**
```python
# In pyproject.toml
[tool.pytest.ini_options]
markers = [
    "integration: marks tests requiring full app stack",
    "slow: marks tests with real I/O or polling loops",
]

# In test files
@pytest.mark.integration
def test_index_lifecycle(...):
    ...
```

**Effort:** 30 minutes (add marker, update CI).

---

### 4. Consolidate TestClient Fixture in `conftest.py` (EASY) — Effort: S
**File:** `tests/conftest.py` (new)

**Why:** 8+ test files create `TestClient(app)` separately. Shared fixture reduces duplication.

**Template:**
```python
import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c
```

**Effort:** 30 minutes (refactor 8 files).

---

### 5. Fix `test_bm25_service.py:120` mtime offset (EASY) — Effort: S
**File:** `tests/test_bm25_service.py:120`

**Why:** 5-second future offset can cause clock-skew issues on network mounts.

**Change:**
```python
# Before
future = time.time() + 5
os.utime(db, (future, future))

# After
future = time.time() + 1
os.utime(db, (future, future))
```

**Effort:** 5 minutes (change, verify test still passes).

---

## Test Infrastructure Recommendations

### 1. Move slow tests to separate marker
**Pattern:** Polling loops (5s deadline) should be `@pytest.mark.slow`.

**Benefit:** `pytest -m "not slow"` runs unit tests in <10s (CI fast loop).

---

### 2. Add pytest.ini markers section
**File:** `pyproject.toml` (already has `[tool.pytest.ini_options]`)

**Add:**
```toml
markers = [
    "integration: marks tests requiring full app stack",
    "slow: marks tests with real I/O or polling loops",
]
```

---

## Summary

**Current state:** 13 test files, 9/15 modules tested (60% coverage). No regression from wave 2.2, but **zero improvements** in 3 days.

**High-risk gaps remain:**
- `app/services/source_fetch.py` (CRITICAL)
- `app/routers/repos.py` (CRITICAL)
- `app/routers/websocket.py` (HIGH)

**Slow tests present:** 4 polling loops identified; none fixed.

**Priorities:**
1. Implement `test_source_fetch.py` (1–2 hours, highest ROI)
2. Add reindex tests (2–3 hours, critical flow)
3. Mark slow/integration tests (1 hour, unblocks CI optimization)
4. Consolidate TestClient fixture (30 min, reduces duplication)
5. Fix mtime offset (5 min, removes flakiness risk)

**Effort estimate:** 8–10 hours for all 5 recommendations.

