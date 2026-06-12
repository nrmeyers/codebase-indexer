# MIT License — Copyright (c) 2026 Navistone, contributors to code-graph-rag
"""Contextual prefix generator (Anthropic Contextual Retrieval).

Implements the technique described in
https://www.anthropic.com/news/contextual-retrieval:

> Before embedding a chunk, prepend a 50-100 token summary of *how this chunk
> relates to its parent file and project*.  The embedder and the reranker then
> see chunk text WITH that context — Anthropic reports a 67% reduction in
> retrieval failures vs. naive chunked embeddings.

This module is intentionally dependency-light:

* No new heavy deps — uses ``httpx`` (already transitive via mcp/pydantic-ai).
* Anthropic Messages API is called directly with a single short prompt.
* Behind ``CONTEXTUAL_RETRIEVAL_ENABLED`` — disabled by default because
  enabling it on an existing repo triggers a one-time re-index pass that
  costs roughly $100 per 100k chunks via Claude Haiku 3.5 at current
  published rates (see ``estimate_cost`` below for the live formula).
* Cache is keyed on ``(file_hash, qualified_name)``.  An unchanged chunk
  in an unchanged file is *never* re-generated — the cache is consulted
  before any LLM call.
* Fallback is **silent and graceful**: if the LLM is unreachable, mis-configured,
  or rate-limited, the prefix degrades to ``[from {file_path}]`` (better than
  nothing — still anchors the embedding to its source file).

The CACHE FILE LIVES NEXT TO THE LadybugDB FILE (under ``EMBEDDING_CACHE_DIR``)
so that a ``rm -rf .cgr`` resets prefixes alongside embeddings.

USAGE
-----

>>> from codebase_rag.services.contextual_prefix import ContextualPrefixGenerator
>>> gen = ContextualPrefixGenerator()
>>> prefix = gen.generate(
...     file_path="src/api/users.py",
...     qualified_name="myapp.api.users.create_user",
...     chunk_text="def create_user(req): ...",
...     file_hash="abc123",
...     sibling_chunks=["def get_user(...): ...", "def delete_user(...): ..."],
... )
>>> embed_text = f"{prefix}\\n\\n{chunk_text}"  # ← this is what gets embedded

The contract is: ``generate()`` NEVER raises.  Worst case it returns the
minimal fallback string.

ENV VARS
--------

``CONTEXTUAL_RETRIEVAL_ENABLED``  — ``true`` to enable LLM calls.
``CONTEXTUAL_RETRIEVAL_MODEL``    — Anthropic model id (default ``claude-haiku-4-5``).
``CONTEXTUAL_RETRIEVAL_MAX_TOKENS`` — cap on generated tokens (default 150).
``CONTEXTUAL_RETRIEVAL_TIMEOUT_S`` — per-call timeout (default 10).
``ANTHROPIC_API_KEY``             — credentials (already used elsewhere in
                                    code-graph-rag).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import httpx  # transitive via mcp / pydantic-ai
except ImportError:  # pragma: no cover - defensive
    httpx = None  # type: ignore[assignment]


_DEFAULT_MODEL = "claude-haiku-4-5"
_DEFAULT_MAX_TOKENS = 150
_DEFAULT_TIMEOUT_S = 10.0
_DEFAULT_API_BASE = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"

# Haiku 3.5 published rates (USD per 1M tokens) as of 2026-05.  Used only
# for the cost estimator surfaced in CLI help.  Update when Anthropic
# revises pricing; nothing in the runtime path depends on these numbers.
_HAIKU_INPUT_USD_PER_1M = 1.00
_HAIKU_OUTPUT_USD_PER_1M = 5.00

_PROMPT_TEMPLATE = """\
You are summarising a code chunk so a downstream retrieval system can match it \
to natural-language questions about the codebase.

<file_path>{file_path}</file_path>
<qualified_name>{qualified_name}</qualified_name>
{siblings_block}
<chunk>
{chunk_text}
</chunk>

In 1-3 sentences (target 50-100 tokens) describe:
1. What this chunk does, in plain English.
2. The role it plays in {file_path}.
3. Any concrete domain terms a user might search for to find it.

Respond with the summary only — no preamble, no markdown, no quotes."""


@dataclass(frozen=True)
class ContextualPrefixConfig:
    """Runtime configuration sourced from environment variables."""

    enabled: bool
    model: str
    max_tokens: int
    timeout_s: float
    api_key: str | None
    api_base: str

    @classmethod
    def from_env(cls) -> ContextualPrefixConfig:
        return cls(
            enabled=os.environ.get("CONTEXTUAL_RETRIEVAL_ENABLED", "false").lower()
            in {"true", "1", "yes", "on"},
            model=os.environ.get("CONTEXTUAL_RETRIEVAL_MODEL", _DEFAULT_MODEL),
            max_tokens=int(
                os.environ.get("CONTEXTUAL_RETRIEVAL_MAX_TOKENS", _DEFAULT_MAX_TOKENS)
            ),
            timeout_s=float(
                os.environ.get("CONTEXTUAL_RETRIEVAL_TIMEOUT_S", _DEFAULT_TIMEOUT_S)
            ),
            api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
            api_base=os.environ.get(
                "CONTEXTUAL_RETRIEVAL_API_BASE", _DEFAULT_API_BASE
            ),
        )


class _PrefixCache:
    """Disk-backed JSON cache keyed by ``sha256(file_hash + qualified_name)``.

    Lives at ``{cache_dir}/contextual_prefixes.json``.  Loaded once at
    construction, written-through on every ``set()``.  Thread-safe.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._path = cache_dir / "contextual_prefixes.json"
        self._lock = threading.Lock()
        self._data: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                # Coerce keys/values to str — the cache is best-effort, any
                # malformed entry just becomes a cache miss.
                self._data = {str(k): str(v) for k, v in raw.items()}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "contextual_prefix.cache.load_failed path={} error={}",
                self._path,
                exc,
            )
            self._data = {}

    def _flush(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(self._data, separators=(",", ":")), encoding="utf-8"
            )
            tmp.replace(self._path)
        except OSError as exc:  # pragma: no cover - disk IO best-effort
            logger.warning(
                "contextual_prefix.cache.flush_failed path={} error={}",
                self._path,
                exc,
            )

    @staticmethod
    def _key(file_hash: str, qualified_name: str) -> str:
        return hashlib.sha256(
            f"{file_hash}:{qualified_name}".encode()
        ).hexdigest()

    def get(self, file_hash: str, qualified_name: str) -> str | None:
        return self._data.get(self._key(file_hash, qualified_name))

    def set(self, file_hash: str, qualified_name: str, prefix: str) -> None:
        with self._lock:
            self._data[self._key(file_hash, qualified_name)] = prefix
            self._flush()

    def __len__(self) -> int:  # used by tests + telemetry
        return len(self._data)


class ContextualPrefixGenerator:
    """Generates and caches contextual prefixes for code chunks.

    The generator is safe to construct in any environment — when
    ``CONTEXTUAL_RETRIEVAL_ENABLED`` is false (the default), every
    ``generate()`` call short-circuits to the minimal ``[from <path>]``
    fallback, which still helps the embedder anchor the chunk to its file.
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        *,
        config: ContextualPrefixConfig | None = None,
        http_client: Any | None = None,
    ) -> None:
        self.config = config or ContextualPrefixConfig.from_env()
        if cache_dir is None:
            cache_dir = Path(os.environ.get("EMBEDDING_CACHE_DIR", ".cgr"))
        self._cache = _PrefixCache(Path(cache_dir))
        self._http = http_client  # injected for tests; lazy-built otherwise
        self._stats = {"hits": 0, "misses": 0, "llm_calls": 0, "llm_failures": 0}

    # ------------------------------------------------------------------ API

    @staticmethod
    def fallback_prefix(file_path: str) -> str:
        """Minimal prefix used when the LLM is unavailable or disabled."""
        return f"[from {file_path}]"

    def generate(
        self,
        *,
        file_path: str,
        qualified_name: str,
        chunk_text: str,
        file_hash: str,
        sibling_chunks: list[str] | None = None,
    ) -> str:
        """Return a contextual prefix for ``chunk_text``.

        Never raises.  Returns either the cached prefix, a freshly-generated
        prefix from the LLM, or the file-path fallback.
        """
        cached = self._cache.get(file_hash, qualified_name)
        if cached:
            self._stats["hits"] += 1
            return cached

        self._stats["misses"] += 1

        if not self.config.enabled:
            return self.fallback_prefix(file_path)

        if not self.config.api_key:
            logger.debug(
                "contextual_prefix.no_api_key — falling back to minimal prefix"
            )
            return self.fallback_prefix(file_path)

        prefix = self._call_llm(
            file_path=file_path,
            qualified_name=qualified_name,
            chunk_text=chunk_text,
            sibling_chunks=sibling_chunks or [],
        )

        if prefix is None:
            return self.fallback_prefix(file_path)

        self._cache.set(file_hash, qualified_name, prefix)
        return prefix

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    # ----------------------------------------------------------- internals

    def _build_prompt(
        self,
        *,
        file_path: str,
        qualified_name: str,
        chunk_text: str,
        sibling_chunks: list[str],
    ) -> str:
        if sibling_chunks:
            joined = "\n---\n".join(s[:400] for s in sibling_chunks[:3])
            siblings_block = f"<siblings>\n{joined}\n</siblings>"
        else:
            siblings_block = ""
        # Truncate chunk_text aggressively — the prompt's job is summarisation,
        # not faithful reproduction.  4kB cap keeps Haiku input cost predictable.
        return _PROMPT_TEMPLATE.format(
            file_path=file_path,
            qualified_name=qualified_name,
            siblings_block=siblings_block,
            chunk_text=chunk_text[:4000],
        )

    def _client(self) -> Any:
        if self._http is not None:
            return self._http
        if httpx is None:  # pragma: no cover
            raise RuntimeError("httpx is not installed")
        self._http = httpx.Client(timeout=self.config.timeout_s)
        return self._http

    def _call_llm(
        self,
        *,
        file_path: str,
        qualified_name: str,
        chunk_text: str,
        sibling_chunks: list[str],
    ) -> str | None:
        prompt = self._build_prompt(
            file_path=file_path,
            qualified_name=qualified_name,
            chunk_text=chunk_text,
            sibling_chunks=sibling_chunks,
        )

        body = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": self.config.api_key or "",
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        started = time.monotonic()
        self._stats["llm_calls"] += 1
        try:
            client = self._client()
            response = client.post(self.config.api_base, json=body, headers=headers)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            self._stats["llm_failures"] += 1
            # NEVER log the api key.  ``exc`` from httpx never includes
            # headers in its string form, but we still guard against future
            # exception types by stringifying explicitly.
            logger.warning(
                "contextual_prefix.llm_failed model={} elapsed_ms={:.0f} error_type={} qualified_name={}",
                self.config.model,
                (time.monotonic() - started) * 1000.0,
                type(exc).__name__,
                qualified_name,
            )
            return None

        text = self._extract_text(payload)
        if not text:
            self._stats["llm_failures"] += 1
            logger.debug(
                "contextual_prefix.empty_response qualified_name={}", qualified_name
            )
            return None
        return text.strip()

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        """Pull the first text block out of an Anthropic Messages response."""
        content = payload.get("content")
        if not isinstance(content, list):
            return ""
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                value = block.get("text")
                if isinstance(value, str):
                    return value
        return ""


# ---------------------------------------------------------------------------
# Cost estimator — surfaced through the CLI and docs.
# ---------------------------------------------------------------------------


def estimate_cost(
    *,
    chunks: int,
    avg_input_tokens: int = 600,
    avg_output_tokens: int = 90,
) -> dict[str, float]:
    """Estimate USD cost for contextual-prefix generation across ``chunks``.

    Defaults reflect typical observed sizes (≈600 in / 90 out for a 25-line
    function summarised in 1-3 sentences) on Haiku 3.5 pricing.
    """
    input_cost = (chunks * avg_input_tokens / 1_000_000.0) * _HAIKU_INPUT_USD_PER_1M
    output_cost = (chunks * avg_output_tokens / 1_000_000.0) * _HAIKU_OUTPUT_USD_PER_1M
    return {
        "chunks": float(chunks),
        "input_tokens_total": float(chunks * avg_input_tokens),
        "output_tokens_total": float(chunks * avg_output_tokens),
        "input_usd": round(input_cost, 4),
        "output_usd": round(output_cost, 4),
        "total_usd": round(input_cost + output_cost, 4),
    }
