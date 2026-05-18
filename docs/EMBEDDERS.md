# Embedders — bring-your-own backend guide

The Code Indexer Service is **embedder-agnostic**. The vector arm of every
search hits a single `EmbedderBackend` protocol that ships with four
interchangeable implementations. Pick the one that matches your
infrastructure footprint and your latency / cost / privacy budget.

```
app/embedders/
├── base.py        EmbedderBackend protocol + EMBEDDING_DIM (768, legacy)
├── __init__.py    get_embedder() factory + EMBEDDER_BACKEND resolution
├── local.py       sentence-transformers in-process (no network)
├── sagemaker.py   AWS SageMaker Serverless Inference (HTTP)
├── tei.py         Hugging Face Text-Embeddings-Inference sidecar (HTTP)
└── openai.py      OpenAI /v1/embeddings (HTTP, BYO)
```

The selected backend is exposed via `GET /health`:

```json
{
  "embedder": {
    "backend": "openai",
    "model": "text-embedding-3-small",
    "dim": 1536,
    "configured": true,
    "error": null
  }
}
```

> Always verify `embedder.dim` matches your DuckDB `FLOAT[dim]` schema
> before issuing a search — a mismatch silently corrupts ranking.

---

## Backend comparison

| Backend     | Default model               | Dim   | Cost / 1M tokens | Network | Setup                                          |
|-------------|-----------------------------|-------|-------------------|---------|------------------------------------------------|
| `local`     | `intfloat/e5-base-v2`       | 768   | $0 (CPU/GPU)      | no      | `uv sync --group local-embed`                  |
| `tei`       | `intfloat/e5-base-v2`       | 768   | $0 (your GPU)     | local   | Run TEI sidecar at `:8080`                     |
| `sagemaker` | `intfloat/e5-base-v2`       | 768   | ~$0.05 (Navistone)| AWS     | Provision SageMaker Serverless endpoint        |
| `openai`    | `text-embedding-3-small`    | 1536  | $0.02             | OpenAI  | `uv sync --extra byo` + `OPENAI_API_KEY`       |
| `openai`    | `text-embedding-3-large`    | 3072  | $0.13             | OpenAI  | `uv sync --extra byo` + `OPENAI_API_KEY`       |

> Cost-per-1M-tokens for SageMaker depends on your endpoint's instance
> type and traffic profile; the ~$0.05 figure is Navistone's observed
> spend on `ml.m5.large` Serverless Inference at production-ish load.

---

## When to pick which

* **`local`** — laptop dev, evaluation, air-gapped deployments. Zero
  network egress; you eat a 440MB model download once. Recommended
  default for standalone installs.
* **`tei`** — you already run a GPU box and want maximum throughput
  without sending source code over the public internet. Drop-in 768-dim
  parity with the `local` and `sagemaker` backends.
* **`sagemaker`** — Navistone production, or any team that wants a
  managed AWS endpoint with SigV4 auth, autoscaling, and CloudWatch
  metrics out of the box.
* **`openai`** — fastest BYO path: paste an API key and go. No model
  download, no AWS account, no GPU. Best quality at the
  `text-embedding-3-large` tier. **Requires re-indexing** because the
  default schema is `FLOAT[768]` and OpenAI's models produce 1536 or
  3072 dim vectors (or whatever you truncate to via Matryoshka).

---

## Concrete `.env` examples

### `local` (default — no AWS, no key)

```bash
# .env
EMBEDDER_BACKEND=local
# Optional — defaults to intfloat/e5-base-v2 (768-dim)
# LOCAL_EMBED_MODEL=sentence-transformers/all-mpnet-base-v2
# LOCAL_EMBED_DIM=768
```

Install: `uv sync --group local-embed`

### `tei` (Hugging Face sidecar)

```bash
# .env
EMBEDDER_BACKEND=tei
TEI_URL=http://localhost:8080
TEI_TIMEOUT_MS=30000
TEI_BATCH_SIZE=32
```

Start the sidecar:

```bash
docker run -d --name tei -p 8080:80 --gpus all \
  ghcr.io/huggingface/text-embeddings-inference:1.5 \
  --model-id intfloat/e5-base-v2
```

### `sagemaker` (AWS)

```bash
# .env
EMBEDDER_BACKEND=sagemaker
SAGEMAKER_ENDPOINT_NAME=forge-e5-embed-v2
SAGEMAKER_EMBED_REGION=us-east-1
SAGEMAKER_EMBED_BATCH_SIZE=32
# AWS credentials resolved via the default boto3 chain
# (env vars > ~/.aws/credentials > IAM role > IMDS).
```

The endpoint must accept `{"inputs": [...]}` and return
`[[float, ...], ...]` of 768-dim L2-normalised vectors.

### `openai` (BYO)

```bash
# .env
EMBEDDER_BACKEND=openai
OPENAI_API_KEY=sk-...
OPENAI_EMBED_MODEL=text-embedding-3-small     # or text-embedding-3-large
# Optional — Matryoshka truncation to fit the legacy 768-dim schema
# OPENAI_EMBED_DIM=768
# Optional — for Azure OpenAI / vLLM / LiteLLM proxies
# OPENAI_BASE_URL=https://gateway.example.com/v1
OPENAI_EMBED_BATCH_SIZE=96
OPENAI_TIMEOUT_S=30.0
```

Install: `uv sync --extra byo`

---

## Switching backends — re-index recipe

Backends with the same `dim` (`local` ↔ `sagemaker` ↔ `tei`, all 768) are
drop-in: stop the service, swap env vars, restart. The DuckDB files stay.

Switching to a backend with a **different** `dim` (e.g. `local` → `openai`
`text-embedding-3-large`, 768 → 3072) requires a fresh index because
DuckDB's `FLOAT[768]` column rejects the larger vectors:

```bash
# 1. stop the service
# 2. delete per-repo vector stores (NOT the .db graph files)
rm -f .cgr/repos/*.duck

# 3. update .env to the new backend
echo "EMBEDDER_BACKEND=openai" >> .env
echo "OPENAI_API_KEY=sk-..." >> .env
echo "OPENAI_EMBED_MODEL=text-embedding-3-large" >> .env

# 4. restart and re-index each repo
uv run uvicorn app.main:app
# then POST /index for each repo
```

If you'd rather keep the existing 768-dim schema, use Matryoshka
truncation: set `OPENAI_EMBED_DIM=768`. Only 3-series OpenAI models
support this.

---

## Protocol contract — for backend authors

If you want to plug in a new backend (Cohere, Voyage, Mistral Embed, a
private Bedrock endpoint, whatever), implement the protocol in
`app/embedders/base.py`:

```python
class EmbedderBackend(Protocol):
    name: str           # stable identifier, surfaced in /health
    model: str          # human-readable model id
    dim: int            # output vector dimensionality

    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...
```

Then:

1. Drop the implementation in `app/embedders/<your-backend>.py`.
2. Add the name to `VALID_BACKENDS` in `app/embedders/__init__.py` and
   wire a branch in `get_embedder()`.
3. Add a per-backend test at `tests/test_embedder_<name>.py` covering
   construction, dim reporting, happy-path embed, dim-mismatch error,
   empty-input short-circuit.
4. Update the comparison table above.

The contract is intentionally tiny:

* **Async-only** — keeps the FastAPI hot path non-blocking. For
  synchronous SDKs (boto3, openai), wrap the call in
  `asyncio.to_thread`.
* **Batched** — single-string embedding is just `embed([text])[0]`.
  Forcing the batched shape eliminates an N+1 footgun and keeps GPU /
  cloud utilisation high.
* **Fail loud** — raise `EmbedderError` on every error path. Returning
  `None` and falling through is how silent index corruption happens.

---

## Health visibility

```bash
curl -s http://localhost:8000/health | jq '.embedder'
```

```json
{
  "backend": "openai",
  "model": "text-embedding-3-small",
  "dim": 1536,
  "configured": true,
  "error": null
}
```

`configured: false` means construction raised — for instance
`EMBEDDER_BACKEND=openai` with no `OPENAI_API_KEY`. The `error` field
carries the message verbatim. The service stays up either way; structural
search and cached results keep working.
