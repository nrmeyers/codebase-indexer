"""Unit tests for :class:`app.embedders.llama_server.LlamaServerEmbedder`.

The HTTP layer is mocked with ``respx`` so the suite runs offline against a
synthetic ``/v1/embeddings`` endpoint. The HuggingFace tokenizer is replaced
with a deterministic whitespace stub so tests do not download
``nomic-ai/nomic-embed-text-v1.5`` from HF.

Coverage targets the load-bearing surface of the backend:

* protocol satisfaction + ``name``/``dim`` + ``prefix_model`` advertising
  (the GGUF-vs-HF disambiguation the prefix registry depends on)
* ``from_env`` happy path and override path
* empty-input short-circuit (no tokenizer load, no HTTP)
* embed happy path: OpenAI-shape request body, ``data[*].embedding`` parsing
* dim-mismatch protection (a misconfigured ``--pooling`` would surface
  here before bad vectors hit DuckDB)
* 5xx batch fallback to per-string (one bad input must not sink the run)
* non-2xx error propagation as :class:`EmbedderError`
* tokenizer-driven truncation when input exceeds ``max_tokens``
"""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from app.embedders import EMBEDDING_DIM, EmbedderError
from app.embedders.base import EmbedderBackend
from app.embedders.llama_server import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_MS,
    DEFAULT_TOKENIZER,
    DEFAULT_URL,
    LlamaServerEmbedder,
)


# ---------------------------------------------------------------------------
# Tokenizer stub — keeps tests offline + deterministic.
# ---------------------------------------------------------------------------


class _WhitespaceTokenizer:
    """Stand-in for an HF fast tokenizer that splits on whitespace.

    ``encode`` returns one fake token id per whitespace-delimited word so
    test assertions about truncation are predictable. ``decode`` rejoins
    the words it was given (the backend feeds it the truncated id slice it
    just got from ``encode``, so the reverse map is a list comprehension on
    the original text).
    """

    def __init__(self, text_corpus: dict[int, str] | None = None) -> None:
        self._next_id = 0
        self._by_id: dict[int, str] = {}
        self._last_words: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        words = text.split()
        self._last_words = words
        ids: list[int] = []
        for w in words:
            self._by_id[self._next_id] = w
            ids.append(self._next_id)
            self._next_id += 1
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(self._by_id[i] for i in ids if i in self._by_id)


def _embedder(**kwargs: Any) -> LlamaServerEmbedder:
    """Build a backend pre-loaded with the whitespace tokenizer stub.

    Bypasses ``_get_tokenizer``'s HF download by setting ``_tokenizer``
    directly. Every test uses this so no test ever talks to HuggingFace.
    """
    eb = LlamaServerEmbedder(**kwargs)
    eb._tokenizer = _WhitespaceTokenizer()
    return eb


def _ok_embeddings_response(n: int, *, dim: int = EMBEDDING_DIM) -> dict[str, Any]:
    """Build a minimal OpenAI-shape ``/v1/embeddings`` 200 body for ``n`` inputs."""
    return {
        "model": DEFAULT_MODEL,
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": [0.001 * i] * dim}
            for i in range(n)
        ],
    }


# ---------------------------------------------------------------------------
# Protocol + attribute surface
# ---------------------------------------------------------------------------


def test_llama_server_embedder_satisfies_protocol() -> None:
    """Backend conforms to :class:`EmbedderBackend` (duck typed via Protocol)."""
    assert isinstance(LlamaServerEmbedder(), EmbedderBackend)


def test_llama_server_embedder_advertises_dim_and_name() -> None:
    """``dim`` is 768 (Matryoshka full-quality); ``name`` matches factory key."""
    eb = LlamaServerEmbedder()
    assert eb.dim == EMBEDDING_DIM
    assert eb.name == "llama_server"


def test_prefix_model_advertises_hf_id_distinct_from_model() -> None:
    """``model`` (GGUF filename) differs from ``prefix_model`` (HF id) so the
    prefix registry can still resolve the asymmetric query/doc prefixes for
    a quantised checkpoint.
    """
    eb = LlamaServerEmbedder(
        model="custom-q4.gguf",
        tokenizer_id="nomic-ai/nomic-embed-text-v1.5",
    )
    assert eb.model == "custom-q4.gguf"
    assert eb.prefix_model == "nomic-ai/nomic-embed-text-v1.5"


# ---------------------------------------------------------------------------
# from_env
# ---------------------------------------------------------------------------


def test_from_env_uses_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """No LLAMA_SERVER_* env → all defaults from the module constants."""
    for var in (
        "LLAMA_SERVER_URL",
        "LLAMA_SERVER_MODEL",
        "LLAMA_SERVER_TOKENIZER",
        "LLAMA_SERVER_MAX_TOKENS",
        "LLAMA_SERVER_TIMEOUT_MS",
        "LLAMA_SERVER_BATCH_SIZE",
    ):
        monkeypatch.delenv(var, raising=False)

    eb = LlamaServerEmbedder.from_env()
    assert eb.base_url == DEFAULT_URL
    assert eb.model == DEFAULT_MODEL
    assert eb.prefix_model == DEFAULT_TOKENIZER
    assert eb.max_tokens == DEFAULT_MAX_TOKENS
    assert eb.timeout_s == DEFAULT_TIMEOUT_MS / 1000.0
    assert eb.batch_size == DEFAULT_BATCH_SIZE


def test_from_env_reads_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """All LLAMA_SERVER_* env vars are read and applied."""
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://gpu1:9999/")  # trailing slash stripped
    monkeypatch.setenv("LLAMA_SERVER_MODEL", "custom.gguf")
    monkeypatch.setenv("LLAMA_SERVER_TOKENIZER", "some-org/some-model")
    monkeypatch.setenv("LLAMA_SERVER_MAX_TOKENS", "512")
    monkeypatch.setenv("LLAMA_SERVER_TIMEOUT_MS", "5000")
    monkeypatch.setenv("LLAMA_SERVER_BATCH_SIZE", "8")

    eb = LlamaServerEmbedder.from_env()
    assert eb.base_url == "http://gpu1:9999"
    assert eb.model == "custom.gguf"
    assert eb.prefix_model == "some-org/some-model"
    assert eb.max_tokens == 512
    assert eb.timeout_s == 5.0
    assert eb.batch_size == 8


def test_from_env_falls_back_on_garbage_numeric_envs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-int env values do not crash startup; defaults are used instead."""
    monkeypatch.setenv("LLAMA_SERVER_MAX_TOKENS", "not-a-number")
    monkeypatch.setenv("LLAMA_SERVER_TIMEOUT_MS", "")
    monkeypatch.setenv("LLAMA_SERVER_BATCH_SIZE", "oops")

    eb = LlamaServerEmbedder.from_env()
    assert eb.max_tokens == DEFAULT_MAX_TOKENS
    assert eb.timeout_s == DEFAULT_TIMEOUT_MS / 1000.0
    assert eb.batch_size == DEFAULT_BATCH_SIZE


# ---------------------------------------------------------------------------
# embed()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_input_short_circuits_without_http() -> None:
    """``embed([])`` returns ``[]`` and never touches HTTP or the tokenizer."""
    eb = LlamaServerEmbedder()
    # Deliberately do NOT seed _tokenizer — confirming no tokenizer load
    # happens on the empty path.
    assert eb._tokenizer is None
    with respx.mock(assert_all_called=False) as router:
        route = router.post(f"{DEFAULT_URL}/v1/embeddings")
        result = await eb.embed([])
    assert result == []
    assert not route.called
    assert eb._tokenizer is None


@pytest.mark.asyncio
async def test_embed_posts_openai_shape_body() -> None:
    """Request body is ``{"model", "input": [...]}`` against /v1/embeddings."""
    eb = _embedder(base_url="http://lh:8090")
    with respx.mock(assert_all_called=True) as router:
        route = router.post("http://lh:8090/v1/embeddings").mock(
            return_value=httpx.Response(200, json=_ok_embeddings_response(2)),
        )
        vectors = await eb.embed(["hello world", "another input"])

    assert len(vectors) == 2
    assert len(vectors[0]) == EMBEDDING_DIM
    assert route.call_count == 1
    sent_body = json.loads(route.calls.last.request.content)
    assert sent_body == {
        "model": DEFAULT_MODEL,
        "input": ["hello world", "another input"],
    }


@pytest.mark.asyncio
async def test_embed_batches_inputs_per_batch_size() -> None:
    """3 inputs at batch_size=2 hit /v1/embeddings exactly twice."""
    eb = _embedder(base_url="http://lh:8090", batch_size=2)

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        n = len(body["input"])
        return httpx.Response(200, json=_ok_embeddings_response(n))

    with respx.mock(assert_all_called=True) as router:
        route = router.post("http://lh:8090/v1/embeddings").mock(side_effect=_handler)
        vectors = await eb.embed(["a", "b", "c"])

    assert len(vectors) == 3
    assert route.call_count == 2
    sizes = [
        len(json.loads(c.request.content)["input"]) for c in route.calls
    ]
    assert sizes == [2, 1]


@pytest.mark.asyncio
async def test_embed_raises_on_dim_mismatch() -> None:
    """A 256-dim vector in the response surfaces as :class:`EmbedderError`
    before the wrong-shape vector can leak into DuckDB.
    """
    eb = _embedder(base_url="http://lh:8090")
    bad_body = {
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.0] * 256}],
    }
    with respx.mock(assert_all_called=True) as router:
        router.post("http://lh:8090/v1/embeddings").mock(
            return_value=httpx.Response(200, json=bad_body),
        )
        with pytest.raises(EmbedderError, match="256-dim vector"):
            await eb.embed(["text"])


@pytest.mark.asyncio
async def test_embed_5xx_falls_back_to_per_string() -> None:
    """A 500 on a batch retries each input one-by-one; happy singletons return.

    Mirrors the real-world case where one oversize input wedges a whole
    batch — we want to salvage the rest of the batch, not abort the run.
    """
    eb = _embedder(base_url="http://lh:8090", batch_size=3)

    call_idx = {"n": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        call_idx["n"] += 1
        # First call is the batch of 3 → 500 to trigger fallback.
        if call_idx["n"] == 1 and isinstance(body["input"], list):
            return httpx.Response(500, text="Worker died.")
        # Subsequent calls are per-string fallbacks → 200.
        # Single-input fallback sends ``"input": str`` (not a list).
        return httpx.Response(200, json=_ok_embeddings_response(1))

    with respx.mock(assert_all_called=True) as router:
        route = router.post("http://lh:8090/v1/embeddings").mock(side_effect=_handler)
        vectors = await eb.embed(["a", "b", "c"])

    assert len(vectors) == 3
    # 1 batch attempt + 3 per-string retries.
    assert route.call_count == 4


@pytest.mark.asyncio
async def test_embed_non_5xx_error_propagates() -> None:
    """A 400 is NOT retried; it propagates as :class:`EmbedderError`."""
    eb = _embedder(base_url="http://lh:8090")
    with respx.mock(assert_all_called=True) as router:
        router.post("http://lh:8090/v1/embeddings").mock(
            return_value=httpx.Response(400, text="bad input"),
        )
        with pytest.raises(EmbedderError, match="HTTP 400"):
            await eb.embed(["x"])


@pytest.mark.asyncio
async def test_embed_truncates_inputs_above_max_tokens() -> None:
    """Inputs longer than ``max_tokens`` are truncated client-side BEFORE the
    HTTP call. The whitespace tokenizer makes this assertion deterministic:
    ``max_tokens=3`` keeps only the first 3 words.
    """
    eb = _embedder(base_url="http://lh:8090", max_tokens=3)
    captured: dict[str, Any] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_ok_embeddings_response(1))

    with respx.mock(assert_all_called=True) as router:
        router.post("http://lh:8090/v1/embeddings").mock(side_effect=_handler)
        vectors = await eb.embed(["one two three four five six"])

    assert len(vectors) == 1
    assert captured["body"]["input"] == ["one two three"]


# ---------------------------------------------------------------------------
# _extract_vectors edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_rejects_wrong_response_count() -> None:
    """Server returning fewer entries than inputs is treated as a protocol
    violation, not silently misaligned with the input order.
    """
    eb = _embedder(base_url="http://lh:8090")
    with respx.mock(assert_all_called=True) as router:
        router.post("http://lh:8090/v1/embeddings").mock(
            return_value=httpx.Response(200, json=_ok_embeddings_response(1)),
        )
        with pytest.raises(EmbedderError, match="entries for 2 inputs"):
            await eb.embed(["a", "b"])
