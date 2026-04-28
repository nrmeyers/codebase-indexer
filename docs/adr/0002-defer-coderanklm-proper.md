# ADR-0002: Defer CodeRankLLM Proper

**Status:** Deferred — not implemented as of 2026-04-27
**Trigger:** Nomic publishes a CodeRankLLM GGUF that loads in LM Studio.

## Context

Listwise reranking today uses Qwen 3.6 (27B dense or 35B MoE-A3B) as the rerank
model. Nomic's CodeRankLLM (Qwen 2.5-Coder fine-tune) is the named long-term target
because it bypasses the "sticky thinking mode" quirk present in Qwen 3.6 when run
in LM Studio.

No LM-Studio-friendly GGUF of CodeRankLLM exists today (2026-04-27). The project
maintains escape hatches (`/no_think` directive, `chat_template_kwargs={"enable_thinking": false}`,
and `reasoning_content` fallback) to handle thinking-mode edge cases.

## Decision

Keep Qwen 3.6 as the current rerank model with documented thinking-mode quirks.
No CodeRankLLM code path is built until a GGUF artifact is published and tested.
The escape hatches stay in place to handle any future model that defaults to
reasoning mode.

## Consequences

**What stays simple:** No new model selection logic; rerank config continues to
list Qwen 3.6 as the default, with thinking-mode workarounds baked into
`app/services/reranker.py`.

**What we accept as cost:** Rerank latency includes reasoning overhead (~3–5 seconds
per query). Fallback to `reasoning_content` when `content` is empty adds an
extra parse step but is negligible vs. model latency.

## When triggered

1. Monitor Nomic's repository for a CodeRankLLM GGUF release.
2. Download GGUF; load into LM Studio; run smoke test (`scripts/lm_studio_smoke.py`).
3. Measure end-to-end rerank latency vs. current Qwen 3.6 on the same test set.
4. If latency improvement is > 20% (typical: 1–2 seconds saved per query):
   a. Add environment variable `LM_STUDIO_RERANK_MODEL` override.
   b. Update `.env.example` with new default model name.
   c. Re-run integration tests; verify fallback paths still work.
   d. Document the swap in `README.md` under "Reranker Model".
5. Roll out to users with a note on expected latency change.
