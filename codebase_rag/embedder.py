from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path

from loguru import logger

from . import constants as cs
from . import exceptions as ex
from . import logs as ls
from .config import settings
from .utils.dependencies import has_torch, has_transformers


class EmbeddingCache:
    __slots__ = ("_cache", "_path")

    def __init__(self, path: Path | None = None) -> None:
        self._cache: dict[str, list[float]] = {}
        self._path = path

    @staticmethod
    def _content_hash(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, content: str) -> list[float] | None:
        return self._cache.get(self._content_hash(content))

    def put(self, content: str, embedding: list[float]) -> None:
        self._cache[self._content_hash(content)] = embedding

    def get_many(self, snippets: list[str]) -> dict[int, list[float]]:
        results: dict[int, list[float]] = {}
        for i, snippet in enumerate(snippets):
            if (cached := self.get(snippet)) is not None:
                results[i] = cached
        return results

    def put_many(self, snippets: list[str], embeddings: list[list[float]]) -> None:
        for snippet, embedding in zip(snippets, embeddings):
            self.put(snippet, embedding)

    def save(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("w", encoding="utf-8") as f:
                json.dump(self._cache, f)
        except Exception as e:
            logger.warning(ls.EMBEDDING_CACHE_SAVE_FAILED, path=self._path, error=e)

    def load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            with self._path.open("r", encoding="utf-8") as f:
                self._cache = json.load(f)
            logger.debug(
                ls.EMBEDDING_CACHE_LOADED, count=len(self._cache), path=self._path
            )
        except Exception as e:
            logger.warning(ls.EMBEDDING_CACHE_LOAD_FAILED, path=self._path, error=e)
            self._cache = {}

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


_embedding_cache: EmbeddingCache | None = None


def get_embedding_cache() -> EmbeddingCache:
    global _embedding_cache
    if _embedding_cache is None:
        cache_path = Path(settings.EMBEDDING_CACHE_DIR) / cs.EMBEDDING_CACHE_FILENAME
        _embedding_cache = EmbeddingCache(path=cache_path)
        _embedding_cache.load()
    return _embedding_cache


def clear_embedding_cache() -> None:
    global _embedding_cache
    if _embedding_cache is not None:
        _embedding_cache.clear()
        _embedding_cache = None


# ---------------------------------------------------------------------------
# LM Studio HTTP embedder
# ---------------------------------------------------------------------------


class LMStudioEmbedder:
    """Thin HTTP client for LM Studio's OpenAI-compatible /v1/embeddings endpoint.

    LM Studio supports batched input (``input: ["text1", "text2", ...]``).
    A single HTTP round-trip for N=64 symbols replaces N sequential calls and
    yields the ~50x indexing-rate improvement identified in SUCCESS.md §Outstanding-1.

    Usage:
        embedder = LMStudioEmbedder.from_env()
        if embedder is not None:
            vecs = embedder.batch_embed(texts, prefix=cs.CODERANK_CODE_PREFIX)

    The single-call ``embed()`` method is preserved for callers that process
    texts one at a time (e.g. query-time embedding).  Both methods fall back
    gracefully — ``None`` return signals "LM Studio unavailable; use in-process
    torch fallback".
    """

    __slots__ = ("_base_url", "_model", "_timeout")

    def __init__(self, base_url: str, model: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    @classmethod
    def from_env(cls) -> LMStudioEmbedder | None:
        """Construct from ``LM_STUDIO_URL`` / ``LM_STUDIO_EMBED_MODEL`` env vars.

        Returns ``None`` when ``LM_STUDIO_URL`` is unset/empty so callers can
        use a simple ``if embedder:`` guard.
        """
        base_url = (os.environ.get("LM_STUDIO_URL") or "").strip().rstrip("/")
        if not base_url:
            return None
        model_hint = (os.environ.get("LM_STUDIO_EMBED_MODEL") or "CodeRankEmbed").strip()
        # Resolve substring hint → concrete model id via /v1/models (best-effort).
        resolved = cls._resolve_model(base_url, model_hint)
        if resolved is None:
            logger.debug(
                "LMStudioEmbedder: no model matching %r found — falling back to in-process",
                model_hint,
            )
            return None
        try:
            timeout = max(1.0, float((os.environ.get("LM_STUDIO_TIMEOUT") or "30").strip()))
        except ValueError:
            timeout = 30.0
        return cls(base_url=base_url, model=resolved, timeout=timeout)

    @staticmethod
    def _resolve_model(base_url: str, hint: str) -> str | None:
        """Return the first loaded model whose id contains ``hint`` (case-insensitive).

        Returns ``None`` on any network / parse error so callers fall through to
        the in-process embedder without hard-failing.
        """
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5.0) as resp:
                data: dict = json.loads(resp.read().decode("utf-8"))
            hint_lc = hint.lower()
            for item in data.get("data", []):
                model_id = item.get("id", "")
                if hint_lc in model_id.lower():
                    return model_id
        except Exception as exc:
            logger.debug("LMStudioEmbedder._resolve_model failed: %s", exc)
        return None

    def _post(self, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/v1/embeddings",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                err_body = exc.read().decode("utf-8", errors="replace")[:400]
            except Exception:
                err_body = ""
            raise RuntimeError(
                f"LM Studio HTTP {exc.code}: {exc.reason}" + (f" — {err_body}" if err_body else "")
            ) from exc
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LM Studio request failed: {exc}") from exc

    def embed(self, text: str, *, prefix: str = "") -> list[float] | None:
        """Embed a single text.  Returns ``None`` on any failure (caller falls back)."""
        result = self.batch_embed([text], prefix=prefix)
        if result is None:
            return None
        return result[0] if result else None

    def batch_embed(
        self,
        texts: list[str],
        *,
        prefix: str = "",
        batch_size: int = cs.LM_STUDIO_EMBED_BATCH_SIZE,
    ) -> list[list[float]] | None:
        """Embed a list of texts in one or more HTTP requests.

        Sends at most ``batch_size`` texts per request (LM Studio may silently
        truncate very large batches depending on the context window of the
        loaded model).  Returns ``None`` on any failure so callers can fall
        through to the in-process torch embedder.

        Args:
            texts: Raw text strings to embed (without prefix).
            prefix: Asymmetric Nomic prefix — ``CODERANK_CODE_PREFIX`` at
                index time, ``CODERANK_QUERY_PREFIX`` at query time.
            batch_size: Max texts per HTTP request (default ``LM_STUDIO_EMBED_BATCH_SIZE``).

        Returns:
            A list of float vectors in the same order as ``texts``, or ``None``
            on network / server error.
        """
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)

        try:
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                prefixed = [prefix + t for t in chunk]
                payload = {"model": self._model, "input": prefixed}
                data = self._post(payload)
                rows = data.get("data") or []
                # LM Studio returns rows sorted by "index"; sort defensively.
                rows_sorted = sorted(rows, key=lambda r: r.get("index", 0))
                if len(rows_sorted) != len(chunk):
                    raise RuntimeError(
                        f"LM Studio returned {len(rows_sorted)} embeddings for "
                        f"{len(chunk)} inputs"
                    )
                for offset, row in enumerate(rows_sorted):
                    embedding = row.get("embedding")
                    if not isinstance(embedding, list):
                        raise RuntimeError(
                            f"LM Studio: unexpected embedding type {type(embedding)}"
                        )
                    results[start + offset] = [float(x) for x in embedding]
        except Exception as exc:
            logger.warning("LMStudioEmbedder.batch_embed failed (%s) — falling back", exc)
            return None

        # Type narrowing: all slots must be filled at this point.
        filled: list[list[float]] = []
        for v in results:
            if v is None:
                logger.warning("LMStudioEmbedder.batch_embed: missing result slot — falling back")
                return None
            filled.append(v)
        return filled


@lru_cache(maxsize=1)
def get_lm_studio_embedder() -> LMStudioEmbedder | None:
    """Return a module-level singleton ``LMStudioEmbedder``, or ``None``.

    Cached so the /v1/models probe only fires once per process lifetime.
    Call ``get_lm_studio_embedder.cache_clear()`` in tests to reset.
    """
    return LMStudioEmbedder.from_env()


if has_torch() and has_transformers():
    import numpy as np
    import torch
    from numpy.typing import NDArray
    from transformers import AutoModel, AutoTokenizer

    @lru_cache(maxsize=1)
    def get_model() -> tuple[AutoTokenizer, AutoModel]:
        tokenizer = AutoTokenizer.from_pretrained(
            cs.CODERANK_EMBED_MODEL, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            cs.CODERANK_EMBED_MODEL,
            trust_remote_code=True,
            safe_serialization=True,
        )
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        return tokenizer, model

    def _mean_pool(
        token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(
            mask_expanded.sum(1), min=1e-9
        )

    def _embed_texts(texts: list[str], max_length: int) -> list[list[float]]:
        tokenizer, model = get_model()
        device = next(model.parameters()).device
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)
        embeddings = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        result: NDArray[np.float32] = embeddings.cpu().numpy()
        return result.tolist()

    def embed_code(code: str, max_length: int | None = None) -> list[float]:
        cache = get_embedding_cache()
        if (cached := cache.get(code)) is not None:
            return cached

        if max_length is None:
            max_length = settings.EMBEDDING_MAX_LENGTH
        result = _embed_texts([cs.CODERANK_CODE_PREFIX + code], max_length)[0]
        cache.put(code, result)
        return result

    def embed_query(query: str, max_length: int | None = None) -> list[float]:
        if max_length is None:
            max_length = settings.EMBEDDING_MAX_LENGTH
        return _embed_texts([cs.CODERANK_QUERY_PREFIX + query], max_length)[0]

    def embed_code_batch(
        snippets: list[str],
        max_length: int | None = None,
        batch_size: int = cs.EMBEDDING_DEFAULT_BATCH_SIZE,
    ) -> list[list[float]]:
        if not snippets:
            return []

        if max_length is None:
            max_length = settings.EMBEDDING_MAX_LENGTH

        cache = get_embedding_cache()
        cached_results = cache.get_many(snippets)

        if len(cached_results) == len(snippets):
            logger.debug(ls.EMBEDDING_CACHE_HIT, count=len(snippets))
            return [cached_results[i] for i in range(len(snippets))]

        uncached_indices = [i for i in range(len(snippets)) if i not in cached_results]
        uncached_snippets = [snippets[i] for i in uncached_indices]

        all_new_embeddings: list[list[float]] = []
        for start in range(0, len(uncached_snippets), batch_size):
            batch = uncached_snippets[start : start + batch_size]
            prefixed = [cs.CODERANK_CODE_PREFIX + s for s in batch]
            all_new_embeddings.extend(_embed_texts(prefixed, max_length))

        cache.put_many(uncached_snippets, all_new_embeddings)

        results: list[list[float]] = [[] for _ in snippets]
        for i, emb in cached_results.items():
            results[i] = emb
        for idx, orig_i in enumerate(uncached_indices):
            results[orig_i] = all_new_embeddings[idx]

        return results

else:

    def embed_code(code: str, max_length: int | None = None) -> list[float]:
        raise RuntimeError(ex.SEMANTIC_EXTRA)

    def embed_query(query: str, max_length: int | None = None) -> list[float]:
        raise RuntimeError(ex.SEMANTIC_EXTRA)

    def embed_code_batch(
        snippets: list[str],
        max_length: int | None = None,
        batch_size: int = cs.EMBEDDING_DEFAULT_BATCH_SIZE,
    ) -> list[list[float]]:
        raise RuntimeError(ex.SEMANTIC_EXTRA)
