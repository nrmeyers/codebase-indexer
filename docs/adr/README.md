# Architectural Decision Records (ADRs)

This directory captures deferred design decisions that belong to future phases. Each ADR
is a **trigger-gated stub**: no implementation until the trigger condition fires.

## Active ADRs

| ID | Title | Trigger |
|----|-------|---------|
| [0001](./0001-defer-hnsw-vss-indexes.md) | Defer HNSW / VSS Indexes | cosine query p95 > 200 ms OR repo > 50,000 symbols |
| [0002](./0002-defer-coderanklm-proper.md) | Defer CodeRankLLM Proper | Nomic publishes a CodeRankLLM GGUF for LM Studio |
| [0003](./0003-defer-cross-repo-unified-ranking.md) | Defer Cross-Repo Unified Ranking | 5+ repos indexed AND orchestrator surfaces cross-repo recall complaints |

## Numbering Convention

ADRs use chronological, four-digit, hyphen-prefixed slugs: `NNNN-slug-title.md`.

## Status Markers

- **Deferred** — not started; includes explicit trigger condition. Remove marker when work lands.
- **Active** — under investigation or in flight.
- **Decided** — work is complete and merged.

## How to Use These

1. **Developers:** Check this directory when planning work that touches vector indices,
   reranking, or cross-repo search. If your change might unlock a trigger, note it.
2. **On-call:** If a trigger fires (e.g., latency SLA breach, new GGUF release),
   activate the ADR by changing Status to "Active" and opening a task.
3. **Reviews:** Link to the relevant ADR when discussing deferred decisions in PRs.
