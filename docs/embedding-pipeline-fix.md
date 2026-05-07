# Embedding Pipeline Reliability — Fix Plan

**Date:** 2026-05-06
**Status:** Drafted, not started
**Owner:** TBD
**Related Linear epic:** TBD (created from this doc)

---

## TL;DR

The embed phase of `POST /index` hangs indefinitely on the first SageMaker
HTTPS call when invoked via the inline Python `-c` driver subprocess.
Root cause: `urllib.request.urlopen(timeout=120)` does not reliably fire
when MMS keeps the TCP socket alive but sends no application-layer bytes.
Combined with no per-batch progress logging and no resume support, a
single hung call wastes the entire embedding pass.

This plan replaces the embed driver with a boto3-based, observable, and
resumable implementation.

---

## Symptoms observed (2026-05-06)

| Symptom | Detail |
|---|---|
| Subprocess `STAT: SN`, `CPU: 0%` for 5+ minutes | Sleeping in syscall |
| TCP connection `ESTABLISHED` to SageMaker, no data flowing | `lsof` confirmed |
| Direct test from same machine + venv: 0.5 s latency | Endpoint healthy |
| `urllib` `timeout=120` set in code, did not fire after 5+ min | urllib timeout is per-blocking-syscall, not wall-clock |
| Zero log output during hang (silent loop) | No per-batch logging exists |
| `.duck` file unchanged for 5+ min | Confirms no inserts happening |
| Endpoint responds `200 OK` to single-item curl during the hang | Server is fine |

Reproducing this conclusively requires `py-spy` to dump the running stack —
not currently installed. We can confirm "stuck inside `urlopen` reading
the response body" by adding `py-spy` to the indexer image as part of
this work.

---

## Verified preconditions (do not change)

- New SageMaker endpoint `forge-e5-embed-v2` with custom mean-pool
  `inference.py` returns flat `[batch, 768]` floats (~16.5 KB / item)
- `forge-dev` IAM has `sagemaker:InvokeEndpoint` on
  `arn:aws:sagemaker:us-east-1:944937319166:endpoint/*`
- Indexer `.env` points at the new endpoint
- Single-item / batch-of-8 / batch-of-32 / batch-of-64 calls all succeed
  via direct `SageMakerEmbedder.batch_embed()` from the indexer-service venv
- The graph build phase (parsing → writing) works correctly: 9,617 nodes,
  15,086 relationships in 73 seconds for TheForge

---

## Root cause analysis

The embed driver is an inline Python script defined as an f-string in
`code-indexer-service/app/routers/index.py` and run via:

```python
proc = subprocess.run([sys.executable, "-c", driver], stdout=log_fh, ...)
```

It does the following sequence:

1. Open `*.db` (LadybugDB) read-only, run a single Cypher to materialise
   ALL `(Function|Method)` rows into a Python list (~10k rows for
   TheForge), close the DB.
2. Open the `*.duck` (DuckDB) vector store for writes.
3. Iterate the materialised rows, accumulate `_BATCH=50` items, call
   `embed_code_batch(batch_texts)` (which sub-batches at
   `_SAGEMAKER_BATCH_SIZE` per SageMaker call), then `bulk_insert` the
   resulting vectors.

The hang is inside `embed_code_batch → SageMakerEmbedder._signed_post →
urllib.request.urlopen(prepared, timeout=120)` on the **first uncached
batch**.

Why `urllib.urlopen(timeout=120)` doesn't fire:

- `timeout=` becomes the per-recv socket timeout.
- The MMS Netty server is sending TCP keepalive packets and likely a
  partial HTTP response stream. As long as a single byte arrives within
  120 s of the previous, the timeout does not fire.
- Neither the MMS frontend nor the response stream sets an application
  deadline. We have observed predict times of 30–60 s for batch=32 with
  realistic 1000-char inputs; if the worker is queued behind a previous
  request, the wall-clock can run far longer.

Secondary issues that compound the user pain:

- **Silent embed loop** — only one log line at startup (cache load),
  zero output until the entire pass finishes.
- **No `embedding_count` updates during the loop** — driver only sets
  `_embedded_count` locally and prints it at the very end, so the
  `/index/{id}/status` endpoint always reports `embedding_count=0`
  during the embed phase.
- **No resume** — `force_reindex=true` recreates `*.duck` from scratch
  even when a prior subprocess wrote a partial result, so progress is
  lost on each retry.
- **Subprocess hard timeout** — wraps `subprocess.run(..., timeout=...)`
  which can never give us a graceful "save progress and exit" path.
- **`embed_code_batch` swallowed exceptions** — fixed earlier today
  (loguru `%s` → f-string), but still falls back silently to local torch
  on any failure, hiding the actual problem.

---

## Goals

1. Embedding 9,617 symbols against `forge-e5-embed-v2` finishes in
   < 1 hour wall-clock from a clean `force_reindex=true` start.
2. A single hung HTTPS call cannot stall the entire pass; the call is
   killed within ≤ 2× expected latency and logged.
3. The embed loop reports progress every 100 symbols (or every 10 s,
   whichever is sooner) into both the `/tmp/cis_embed_*.log` file and
   the in-memory `embedding_count` field.
4. Re-running an interrupted embed pass resumes from the last persisted
   `*.duck` row instead of starting over.
5. The embed driver lives in a checked-in `.py` module, not an inline
   `-c` f-string in a router file.

---

## Plan — phased

### Phase 1 — Diagnostics (½ day)
Make the failure mode observable without changing behaviour.

- Add `py-spy` to the indexer-service dev/prod image.
- Wire a SIGTERM handler in the embed driver that dumps the current stack
  to the log before exit.
- Capture one fresh hang and confirm the stack is parked inside
  `http.client.HTTPResponse.read()` (or similar). File the trace.

**Exit criteria:** A captured stack confirms (or refutes) the urllib-stream
hypothesis. Either way, we know what to fix.

---

### Phase 2 — Replace urllib with boto3 SageMaker runtime (1 day)
Boto3's `sagemaker-runtime.invoke_endpoint()` accepts an explicit
`Config(connect_timeout=…, read_timeout=…, retries={…})`. `read_timeout`
is a true total-time-without-bytes timeout enforced by botocore's
endpoint connection. It WILL fire on slow streams.

**Files to change:**

- `code-graph-rag/codebase_rag/embedder.py` — replace `_signed_post()`
  body. Same SigV4 auth, but via `boto3.client('sagemaker-runtime', …,
  config=Config(read_timeout=90, connect_timeout=10, retries={'max_attempts':3,'mode':'standard'}))`.
  Keep the urllib fallback for the LM Studio path unchanged.
- `code-indexer-service/app/services/sagemaker_embedder.py` — same
  treatment. This file is a service-local copy of the same client.
- Drop `_INVOKE_TIMEOUT = 120` constant; replaced by `read_timeout=90`.

**Acceptance:**

- A test that mocks a slow-streaming response (1 byte per 30 s) raises
  `ReadTimeoutError` within 90 s ± 5 s.
- A test against the real endpoint with `inputs=["hello"]*32` returns in
  the same wall-clock as it does today (no regression).

---

### Phase 3 — Per-batch progress + status surface (½ day)
Surface real-time progress to humans and to `/index/{id}/status`.

- Embed driver writes a single line per outer batch:
  `[18:01:42] embedded 50/9617 (cache: 6114, sm: 50, lm: 0, torch: 0) — last batch 1.4s`.
- Driver writes `_embedded_count` to a sidecar JSON file
  (`/tmp/cis_embed_{job_id}.progress.json`) every batch.
- `app/routers/index.py` reads that sidecar in `_get_status()` to
  populate `embedding_count` while the job is running, instead of
  reporting 0 until completion.
- Bump existing log statements to `INFO` so they survive any future
  log-level tightening.

**Acceptance:**

- During an embed pass, hitting `/index/{id}/status` once per second
  returns a monotonically increasing `embedding_count`.
- The `/tmp/cis_embed_*.log` shows one line per batch with timing.

---

### Phase 4 — Resume from partial `*.duck` (½ day)
Re-running an interrupted embed pass should skip rows already vectorised.

- Embed driver, after opening the duck conn, reads
  `SELECT qualified_name FROM embeddings` into a `set`.
- Skip any row whose `qualified_name` is in that set unless
  `force_reindex` is True AND the .duck was just truncated.
- Add a `--resume` CLI flag (or env var) so the indexer can call the
  driver with `resume=true` after a subprocess timeout.

**Acceptance:**

- Kill the embed subprocess at 50% progress, restart the job, observe
  the second run skips the first 50% in < 5 s and finishes the
  remaining 50%.

---

### Phase 5 — Wall-clock watchdog + graceful timeout (½ day)
The current `subprocess.run(timeout=14400)` only kills via SIGKILL with
no chance to save state. Replace with a poll-based watchdog.

- Run the driver via `subprocess.Popen` instead of `subprocess.run`.
- Parent watches the progress sidecar JSON. If `last_update` is older
  than `max_idle_sec` (default 300), send SIGTERM (driver dumps stack +
  flushes duck, then exits 137).
- Total wall-clock ceiling stays at 4 h but is enforced by the parent,
  not by `subprocess.run`.

**Acceptance:**

- A driver that artificially stops emitting progress receives SIGTERM
  within 305 s and exits with a "watchdog: no progress for 300s" line.
- The duck file is consistent (no torn writes) after such a kill.

---

### Phase 6 — Move embed driver out of inline string (½ day)
The current 130-line inline `-c` f-string is unreviewable, untestable,
and forced us to use double-brace escape gymnastics that have already
caused regressions (see comment near `_header_parts.append(f"# {{_stype}}: {{_qname}}")`).

- Create `code-indexer-service/app/services/embed_driver.py` with a
  proper CLI: `python -m app.services.embed_driver --repo-name FOO
  --repo-path /abs/path --vec-db /abs/path.duck --graph-db
  /abs/path.db --batch 16 [--resume]`.
- Add a `tests/test_embed_driver.py` that runs the driver against a
  fixture .db with a mock SageMaker URL.
- Update `index.py` to invoke `[sys.executable, '-m',
  'app.services.embed_driver', ...]` instead of building an f-string.

**Acceptance:**

- `pytest tests/test_embed_driver.py -k slow_stream` reproduces and
  catches the original hang in a unit test.
- No more `f"""..."""` driver block in `index.py`.

---

### Phase 7 — Concurrent SageMaker calls (optional, 1 day)
With Phases 1–6 done, one synchronous batch=16 call ≈ 25 s. Even
perfectly sequential, 9,617 / 16 ≈ 600 calls × 25 s = 4.2 h.

To finish under an hour, we need ~5× concurrency. Options:

- **a)** `concurrent.futures.ThreadPoolExecutor(max_workers=5)` in the
  embed driver, batching writes to duck. Cleanest, no new deps.
- **b)** SageMaker async inference endpoint — write to S3, callback
  when done. More work, but better fit for very large repos.
- **c)** Increase the endpoint's instance count or upgrade
  `ml.m5.large` → `ml.c6i.2xlarge`. Pure infra, no code change.

Recommend (a) first. If still slow, do (c). (b) only for repos
> 100k symbols.

**Acceptance:**

- TheForge re-index embed pass < 30 min wall-clock at concurrency=5.

---

## Out of scope for this plan

- Switching embedding model or moving off SageMaker
- Rewriting the LadybugDB (graph) ingestion pipeline
- Hardening the LM Studio fallback path

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| boto3 has its own keepalive surprises | Low | Phase 1 stack capture + integration test in Phase 2 |
| Resume logic mis-skips rows after truncation | Medium | Stamp `*.duck` with the source `*.db` mtime; clear table if mtime changed |
| Concurrent calls hit MMS worker count = 2 | Medium | Tunable pool size, default 2, observe latency |
| boto3 cold-start adds latency to first call | Low | Cache the client at module level (we already do this for `urllib` session) |

---

## Estimated effort

| Phase | Engineer-days |
|---|---|
| 1 — Diagnostics | 0.5 |
| 2 — boto3 swap | 1.0 |
| 3 — Progress logging | 0.5 |
| 4 — Resume | 0.5 |
| 5 — Watchdog | 0.5 |
| 6 — Driver-as-module | 0.5 |
| 7 — Concurrency (optional) | 1.0 |
| **Total (without 7)** | **3.5 days** |
| **Total (with 7)** | **4.5 days** |

---

## What's already shipped (today, 2026-05-06)

These changes are committed locally and don't need re-doing:

- `forge-e5-embed-v2` SageMaker endpoint with custom mean-pool inference handler
- `ForgePlatformPolicy` IAM policy widened to wildcard SageMaker + inference profiles
- `_SAGEMAKER_BATCH_SIZE = 16` (was 32, then 1)
- `urlopen(timeout=120)` (was 30) — superseded by Phase 2
- Loguru error message fix (`%s` → f-string)
- Unwrap loop dropped in both embedders (response is now flat)
- Indexer `.env` points at v2

These changes get TheForge into a 47%-embedded state and unblock structural
search end-to-end. The remaining 53% requires Phase 1+ above.
