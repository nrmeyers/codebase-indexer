# SageMaker swap: jina-code-v2-serverless → nomic-embed-text-v1.5

**Status:** plan. Code change is staged (local default + llama_server
backend on `feat/embedder-prefixes-768`); this doc covers the prod
endpoint flip, which is a separate rollout.

**Why:** The 768-POC verdict on 2026-06-15 chose
`nomic-ai/nomic-embed-text-v1.5` over both the prior local default
(`intfloat/e5-base-v2`) and the current prod endpoint
(`jina-code-v2-serverless`). The local backend has already been swapped
in the same branch; this brings prod in line so cross-environment
embeddings stay symmetric. See
[`embedder-poc-768-results.md`](embedder-poc-768-results.md) for the
recall numbers driving the call.

**Schema:** unchanged — `FLOAT[768]`, mean-pooled, L2-normalised. No
LadybugDB / DuckDB migration. A re-index of every served repo IS
required because vector spaces are not comparable across models — see
"Rollout" below.

## Endpoint contract diff

| Field | jina-code-v2-serverless (today) | nomic-v1.5 (target) |
|-------|---------------------------------|---------------------|
| Endpoint name | `jina-code-v2-serverless` | `nomic-v1.5-serverless` (proposed) |
| Region | us-east-1 | us-east-1 |
| Dim | 768 | 768 (Matryoshka — also supports 512/256 truncation) |
| Pooling | server-side mean | server-side mean |
| Max tokens | 8192 (server-truncated) | 8192 (server-truncated) |
| Query prefix | none (symmetric) | `search_query: ` (asymmetric, applied client-side) |
| Doc prefix | none (symmetric) | `search_document: ` (asymmetric, applied client-side) |
| Custom modeling code | no | yes (`trust_remote_code=True`) |

Critical: nomic is asymmetric. The client side already applies the
correct query/doc prefix through `app/embedders/prefixes.py` **only when
`backend.name in {"local", "llama_server"}`**. For the SageMaker swap
we must extend that gate to include `"sagemaker"` AND set
`SageMakerEmbedder.prefix_model = "nomic-ai/nomic-embed-text-v1.5"` so
the registry hits. Forgetting this is the #1 silent-regression risk —
it tanks recall without any error.

## Container image

Two viable paths; pick at deploy time based on whichever the platform
team prefers to maintain:

1. **HF Inference Toolkit image** (matches the current jina-code-v2
   endpoint shape). Pull `nomic-ai/nomic-embed-text-v1.5` into the
   image at build time and set `HF_TASK=feature-extraction`. The
   current `SageMakerEmbedder` already mean-pools any `[batch][tok][dim]`
   shape down to `[batch][dim]` and L2-normalises (see
   `app/embedders/sagemaker.py:18-24`), so the existing client code
   works unmodified.

2. **Custom container** wrapping `llama-server` with the same Q8 GGUF
   the POC used (`nomic-embed-text-v1.5.Q8_0.gguf`). Cheaper at idle
   (no GPU keep-alive), faster cold-start than torch. Would require
   either a new `EMBEDDER_BACKEND=sagemaker_llama` or extending the
   existing `sagemaker` backend to speak the OpenAI-shape body.

Recommend **option 1** unless the platform team has a reason to move
off the HF toolkit — minimises code surface area on our side.

## Instance sizing

Current `jina-code-v2-serverless` runs serverless (memory-only sizing,
no warm GPU). nomic-v1.5 is the same parameter count (137M), so the
existing serverless tier should hold — **but bench against the POC index
times before committing**. The POC indexed TheForge in 1158s on an RTX
3060; SageMaker serverless on CPU will be ~5–10× that. If that's
unacceptable for prod re-index latency, move to a provisioned
ml.g4dn.xlarge or similar.

## Config to flip

```diff
- SAGEMAKER_ENDPOINT_NAME=jina-code-v2-serverless
+ SAGEMAKER_ENDPOINT_NAME=nomic-v1.5-serverless
```

Plus (in code, **before** the endpoint flip lands):

1. `app/embedders/prefixes.py:102` — extend the gate:
   ```python
   if not texts or getattr(backend, "name", None) not in ("local", "llama_server", "sagemaker"):
   ```
2. `app/embedders/sagemaker.py` — expose `prefix_model = "nomic-ai/nomic-embed-text-v1.5"`
   as an instance attribute (or read from `SAGEMAKER_PREFIX_MODEL` env so the
   value is config-driven, not hardcoded). Default to empty for backwards
   compatibility — the gate falls through to a no-op when `prefix_model` is
   empty AND `model` doesn't resolve to a PREFIXES entry.
3. Update `app/embedders/sagemaker.py:53` truncation comment — nomic-v1.5 is
   8K tokens, not 512. The current `_MAX_CHARS=1000` cap is conservative but
   correct; bump or leave as-is depending on the chunk-size distribution
   observed during a staging re-index.

## Rollout

1. **Stand up the new endpoint** under a non-prod name. Don't touch
   `jina-code-v2-serverless` yet.
2. **Verify the contract**: hit `/embed` against the new endpoint with
   `SAGEMAKER_ENDPOINT_NAME` overridden, assert 768-dim vector, assert
   query/doc prefixes appear in the request body (log inspection).
3. **Re-index one canary repo** end-to-end (recommend code-indexer-service
   — smallest of the three POC corpora at 173s indexing time). Run the
   recall harness against it. Numbers should match POC ±2pp; if they
   diverge by more, the prefix gate or pooling is wrong.
4. **Atomic flip**: update the prod `SAGEMAKER_ENDPOINT_NAME` env, kick
   a re-index of every served repo (vectors are NOT cross-model
   comparable — running mixed indexes will silently return wrong
   neighbors).
5. **Decommission** `jina-code-v2-serverless` after a soak window (1
   week recommended — long enough that a rollback ask would come in
   loud).

## Rollback

If recall regresses or the new endpoint thrashes:

1. Revert `SAGEMAKER_ENDPOINT_NAME` to `jina-code-v2-serverless`.
2. Revert the `prefixes.py` gate change (or accept the no-op since
   jina-code-v2 is symmetric and the registry has no entry for it).
3. Re-index every served repo against the restored endpoint. There is
   no shortcut — the on-disk DuckDB vectors are model-specific.

Cost of rollback ≈ cost of rollout (one full re-index per served repo).
Plan the rollout window accordingly.

## Open questions for the platform team

- Container image choice (HF toolkit vs custom llama-server wrapper).
- Serverless vs provisioned sizing — do current p50/p99 SLAs hold on
  serverless-CPU for a re-index, or do we need provisioned GPU?
- Endpoint naming — `nomic-v1.5-serverless` follows the current
  pattern but the `-serverless` suffix lies if we end up on a
  provisioned instance. Suggest `nomic-v1.5-embed-768` as a
  pooling/dim-agnostic name.
