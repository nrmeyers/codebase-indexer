# BUC-1574 — Incremental embed by content hash: audit findings

Phase 1.4 of the optimization roadmap. This document audits the existing
content-hash skip path in `app/routers/index.py` (the `_blocking_embed`
subprocess driver) to confirm the hash flips on every change that should
require re-embedding, and notes any gaps for follow-up bugs.

## What gets hashed

In the embed-job driver template (~lines 1145–1167 of `app/routers/index.py`)
the per-symbol input fed into `embed_code_batch` is composed as:

```
# {_stype}: {_qname}
# Module: {_mod_path}        # only when _mod_path is non-empty
# Callers: {_callers}        # only when caller_count > 0
# ---
{_formatted_doc}              # only when docstring is non-empty
{_src}                        # source lines [start_line - 1 : end_line]
```

The hash is then `sha1(_embed_text.encode("utf-8")).hexdigest()` and is
compared against the stored `content_hash` for the symbol's
`qualified_name` (loaded once via `read_content_hashes()` at the top of
the run).

## Audit per requirement

| Change                | Flips hash? | Reasoning |
| --------------------- | ----------- | --------- |
| Source line range     | YES         | The slice `_lines[start-1:end]` changes when either bound moves *and* the content of the new range differs from the old. Pure renumbering of identical text is intentional (see gap #1). |
| Caller count          | YES         | Header line `# Callers: {N}` is included verbatim; goes from absent → present at the 0→1 boundary, otherwise the integer flips. |
| Symbol kind / qname   | YES         | Header line `# {_stype}: {_qname}` is the first line of every embed input; either field changing flips the hash. |
| Docstring             | YES         | `format_docstring(_doc)` output is appended verbatim before the source. Adding, removing, or editing the docstring changes the hash. |

All four must-flip cases are covered. The path is correct as designed.

## Gaps (filed as follow-up bugs, NOT fixed in this PR)

### Gap #1 — file move under unchanged qualified_name

`_abs` (the resolved file path) is **not** included in `_embed_text`. If a
function is moved between files but keeps the same `qualified_name` and
identical source text, the hash will match the prior run and the embed
call will be skipped. Today this is rare in practice (qname usually
includes the module path), but Python `__init__.py` re-exports and
TypeScript barrel files can produce identical qnames at different paths.

Severity: low. Recommend including `_rel` (relative path) in the header
when we next touch the driver.

### Gap #2 — pure line renumbering with identical content

If a function moves from lines 100–120 to 50–70 with byte-identical
content, the hash stays the same and we keep the previous embedding.
This is **correct** for semantic search (the embedding is a function of
the text, not its address) but means the persisted `start_line`/
`end_line` columns can drift relative to the on-disk reality until the
next genuine source change re-triggers the row write.

Severity: low. The structural index (LadybugDB) is rebuilt from scratch
on every run, so search → context-bundle resolution still works
correctly; only the `.duck` row carries stale line numbers.

### Gap #3 — caller_count 0 vs absent

When `_callers == 0` the `# Callers:` line is omitted entirely (an
optimization to keep the embed input compact for leaf functions). A
symbol going from "no callers" to "one caller" therefore inserts a new
header line, which correctly flips the hash. Going from one caller to
zero (e.g., the only call site is removed) also flips the hash. No bug,
called out for completeness.

## Hash persistence verification

The `.duck` (DuckDB) file is closed cleanly at the end of every embed
pass (`_vec_conn.close()` in the driver, line ~1193). DuckDB persists
all writes within the closed transaction to disk. Re-opening the file
via `open_or_create()` and calling `read_content_hashes()` returns the
same `{qualified_name: content_hash}` mapping that was written, with no
loss across uvicorn restarts. The
`tests/test_incremental_embed.py::test_content_hash_persists_across_reopen`
test exercises this round-trip.

## Diff metrics

The `GET /index/{job_id}/diff_metrics` endpoint reports:

```
total_symbols      = embedded + skipped_unchanged + skipped_filtered
embedded           = symbols sent to SageMaker on this run
skipped_unchanged  = content_hash matched the stored value (cache hit)
skipped_filtered   = file matched a skip pattern (tests/generated/vendored)
hash_match_rate    = skipped_unchanged / (embedded + skipped_unchanged)
wall_clock_seconds = embed_finished_at - embed_started_at  (or now - start for running jobs)
```

`hash_match_rate` is the headline number: a re-index of an unchanged
repo should report ≈1.0; a fresh repo or after `force_reindex=true` will
report 0.0.
