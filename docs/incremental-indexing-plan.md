# Incremental git-aware indexing — implementation plan

**Epic:** BUC-1518
**Sibling epic:** BUC-1510 (embed pipeline reliability)
**Date:** 2026-05-06

See chat session for full reasoning. This doc is the engineering plan.

## TL;DR

Stop re-indexing everything every time. Use `git diff <last_sha> HEAD` to find changed files, re-parse only those, embed only symbols whose `content_hash` changed. For a typical commit (1-5 files): full re-index drops from hours to < 30 s.

## Day-by-day execution

### Day 1: foundation
- BUC-1512: replace `urllib` with `boto3.sagemaker-runtime` (real `read_timeout`)
- A1: switch SageMaker endpoint to Serverless Inference (~$85/mo savings)

### Day 2-3: detection
- A2: `RepoMeta` table + `content_hash` + `source_file_sha` columns
- B1: `git diff` helper with fallback to full reindex on bad SHA
- B2: symbol-level diff for modified files

### Day 4-5: write path
- C1: tombstone deletion of removed symbols
- C2: selective re-embed (cache-first, content_hash-aware)
- C3: atomic SHA stamp on success

### Day 6: warmup + tests
- D1: pre-warm endpoint during parse phase
- E1: integration tests with fixture repo
- E2: migration for existing `.db` files

### Day 7: rollout
- Enable for one repo first, observe a week of incremental updates
- Then enable for all repos
