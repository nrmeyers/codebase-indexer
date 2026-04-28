"""LM Studio adapter — thin HTTP client for the OpenAI-compatible local API.

LM Studio (https://lmstudio.ai) hosts GGUF/MLX models on a localhost server
that speaks a subset of the OpenAI API.  When the user has it running with
a code-embedding and/or code-reranking model loaded, we prefer it over the
in-process ``transformers`` path because:

* The model stays warm in LM Studio's process, eliminating the ~3-5s cold
  start uvicorn pays on the first query.
* uvicorn's resident set stays lean (~500 MB instead of ~5 GB once a
  reranker is added).
* OOM kills land in LM Studio, not in the indexer service.

Both backends remain in tree.  The adapter is **opt-in via env var**:

    LM_STUDIO_URL=http://localhost:1234   # empty/unset = disabled
    LM_STUDIO_EMBED_MODEL=CodeRankEmbed   # substring match against /v1/models
    LM_STUDIO_RERANK_MODEL=CodeRankLLM    # substring match against /v1/models

Health probing is best-effort and *non-fatal*: if LM Studio is unreachable
or the requested model isn't loaded, callers transparently fall back to
the in-process embedder (``codebase_rag.embedder``) and skip the optional
rerank stage.

This module has zero hard dependencies on FastAPI, pydantic-settings, or
any other infra — it reads the env vars directly so the same module can be
imported from a subprocess without dragging in the whole settings stack.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def base_url() -> str:
    """Return the configured LM Studio base URL, or empty string when disabled.

    Empty string is the canonical "disabled" signal — every other function
    in this module short-circuits when ``base_url() == ""``.
    """
    return _env("LM_STUDIO_URL").rstrip("/")


def embed_model_hint() -> str:
    """Substring hint used to pick the embedding model from /v1/models.

    Defaults to ``"CodeRankEmbed"`` to match Nomic's official model id.

    .. warning::
        The query-time embedding model **must** be the same model the
        index was built with.  CodeRankEmbed and ``nomic-embed-text-v1``
        (the parent base) are both 768-dim but live in different vector
        spaces — using the parent at query time silently destroys
        cosine recall (~50–70% precision drop on Nomic's eval) without
        any error.  We default the hint to the strict CodeRankEmbed id
        so an accidentally-loaded base model never resolves.  Override
        only when you've also rebuilt the index with the same backend.
    """
    return _env("LM_STUDIO_EMBED_MODEL", "CodeRankEmbed")


def rerank_model_hint() -> str:
    """Substring hint used to pick the reranker model from /v1/models.

    Defaults to ``"CodeRankLLM"`` (Nomic's Qwen2.5-Coder-7B fine-tune)
    but the rerank prompt format works with any instruction-following
    LLM — Qwen3, Llama 3, Mistral, etc.  Set this to whatever model id
    you have loaded; the model only needs to follow the bracketed-
    permutation output format the system prompt asks for.
    """
    return _env("LM_STUDIO_RERANK_MODEL", "CodeRankLLM")


def request_timeout_s() -> float:
    """Total timeout for any single HTTP call to LM Studio."""
    try:
        return max(1.0, float(_env("LM_STUDIO_TIMEOUT", "30")))
    except ValueError:
        return 30.0


# ---------------------------------------------------------------------------
# Low-level HTTP
# ---------------------------------------------------------------------------


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """POST ``payload`` as JSON, return the parsed JSON response.

    Raises ``RuntimeError`` for any non-200, network failure, or parse error.
    Callers should catch broadly and treat any exception as "LM Studio is
    not available, fall back".
    """
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"LM Studio HTTP {exc.code}: {exc.reason}") from exc
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"LM Studio request failed: {exc}") from exc


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"LM Studio request failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Health + model resolution
# ---------------------------------------------------------------------------


_PROBE_TTL_S = 30.0  # cache /v1/models for 30s so repeated calls are cheap
_probe_cache: dict[str, tuple[float, list[str]]] = {}


def list_models() -> list[str]:
    """Return loaded model ids reported by LM Studio's /v1/models endpoint.

    Cached for ``_PROBE_TTL_S`` seconds.  Returns ``[]`` when LM Studio is
    unreachable or disabled.
    """
    url = base_url()
    if not url:
        return []
    cached = _probe_cache.get(url)
    if cached and (time.monotonic() - cached[0]) < _PROBE_TTL_S:
        return cached[1]
    try:
        data = _get_json(f"{url}/v1/models", request_timeout_s())
        models = [m["id"] for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
    except Exception as exc:
        logger.debug("LM Studio probe failed: %s", exc)
        models = []
    _probe_cache[url] = (time.monotonic(), models)
    return models


def is_available() -> bool:
    """Return True when LM Studio is reachable and at least one model is loaded."""
    return bool(base_url()) and bool(list_models())


def can_embed() -> bool:
    """Return True when LM Studio can actually serve an embedding right now.

    More precise than ``is_available()`` — LM Studio can be up with only a
    chat model loaded (no embedder), in which case ``is_available()`` is
    True but every ``embed()`` call will return ``None`` and downstream
    callers will fall through to a slower error path.  Use this helper at
    fast-fail decision points where you need to know whether the embed
    backend specifically is usable.
    """
    if not is_available():
        return False
    return resolve_model(embed_model_hint()) is not None


def can_rerank() -> bool:
    """Return True when LM Studio can actually serve a rerank chat completion.

    Mirror of ``can_embed()`` for the rerank path — ``is_available()`` only
    tells you the server is up, not whether the named rerank model is
    loaded.  Used by route handlers to short-circuit ``?rerank=true``
    cleanly when the model isn't loaded.
    """
    if not is_available():
        return False
    return resolve_model(rerank_model_hint()) is not None


def resolve_model(hint: str) -> str | None:
    """Resolve a substring hint to a concrete loaded-model id.

    LM Studio model ids are typically full HF paths or local-folder paths
    such as ``"nomic-ai/CodeRankEmbed-GGUF/CodeRankEmbed-Q4_K_M.gguf"``,
    so a case-insensitive substring match is the most ergonomic API.

    Returns ``None`` when no loaded model matches the hint.
    """
    if not hint:
        return None
    hint_lc = hint.lower()
    for model in list_models():
        if hint_lc in model.lower():
            return model
    return None


@lru_cache(maxsize=1)
def _log_backend_once() -> None:
    """Emit a one-line backend summary the first time anyone asks."""
    if is_available():
        logger.info(
            "LM Studio: %s — embed=%s rerank=%s",
            base_url(),
            resolve_model(embed_model_hint()) or "(none)",
            resolve_model(rerank_model_hint()) or "(none)",
        )
    else:
        logger.info("LM Studio: disabled or unreachable — using in-process fallback")


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def embed(text: str, *, prefix: str = "") -> list[float] | None:
    """Embed ``prefix + text`` via LM Studio.

    Returns the float vector on success, ``None`` on any failure (caller
    should fall back to the in-process embedder).  Does NOT raise — the
    LM Studio path is always best-effort.

    Args:
        text: The raw text to embed (code snippet or query string).
        prefix: Asymmetric Nomic prefix.  Use ``"Represent this code
            snippet: "`` for code at index time and ``"search_query: "`` for
            queries at search time.  Mixing these silently degrades recall
            ~15-20% so callers should pass it explicitly rather than
            defaulting.
    """
    _log_backend_once()
    url = base_url()
    if not url:
        return None
    model = resolve_model(embed_model_hint())
    if not model:
        return None
    payload = {"model": model, "input": prefix + text}
    try:
        data = _post_json(f"{url}/v1/embeddings", payload, request_timeout_s())
        rows = data.get("data") or []
        if not rows:
            return None
        embedding = rows[0].get("embedding")
        if not isinstance(embedding, list):
            return None
        return [float(x) for x in embedding]
    except Exception as exc:
        logger.warning("LM Studio embed failed (%s) — falling back", exc)
        return None


# ---------------------------------------------------------------------------
# Chat completion (used by the reranker)
# ---------------------------------------------------------------------------


def chat_complete(
    messages: list[dict[str, str]],
    *,
    model_hint: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str | None:
    """Send a chat-completion request, return the assistant message content.

    Returns ``None`` on any failure.  Used by ``reranker.py`` for listwise
    permutation generation; not exposed as a general-purpose chat API.

    Args:
        messages: OpenAI-format chat messages.
        model_hint: Substring of a loaded LM Studio model id; falls back
            to ``rerank_model_hint()`` when ``None``.
        max_tokens: Token budget for the assistant's reply.  Be generous
            for thinking-mode models (Qwen3, DeepSeek-R1) — the reasoning
            channel consumes from the same budget as ``content``.
        temperature: Sampling temperature.  ``0.0`` for deterministic
            permutations.
        chat_template_kwargs: Pass-through to LM Studio's chat-template
            renderer.  Used to disable Qwen3's reasoning mode via
            ``{"enable_thinking": False}`` — saves the 200-500 reasoning
            tokens that would otherwise consume the budget.  Models that
            don't recognise the kwarg ignore it harmlessly.
    """
    _log_backend_once()
    url = base_url()
    if not url:
        return None
    model = resolve_model(model_hint or rerank_model_hint())
    if not model:
        return None
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    try:
        data = _post_json(f"{url}/v1/chat/completions", payload, request_timeout_s())
        choices = data.get("choices") or []
        if not choices:
            return None
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        # Thinking models (Qwen3, DeepSeek-R1, …) emit a separate
        # ``reasoning_content`` channel and ``content`` may be empty if the
        # model spent its whole token budget reasoning.  Fall back to the
        # reasoning channel so the bracketed permutation is still parseable
        # downstream — the rerank parser is tolerant of surrounding prose.
        if isinstance(content, str) and content.strip():
            return content
        reasoning = msg.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning
        return None
    except Exception as exc:
        logger.warning("LM Studio chat_complete failed (%s) — skipping rerank", exc)
        return None
