"""In-memory BM25 lexical index per repo, fused with semantic via RRF.

Rebuilds when the underlying ``.duck`` file mtime changes; otherwise serves
from cache.  Tokenisation: split ``qualified_name`` on dots and underscores,
plus the ``symbol_type`` label.  Pure-Python ``bm25s`` — zero native deps.

Cache key is the absolute ``.duck`` filesystem path so multiple repos
co-exist safely; entries are invalidated when mtime advances (new index run).

Public API:
    bm25_service.search(vec_db_path, query, k=50)
        -> list[(qualified_name, score)]

Best-effort by design: missing files, empty corpora, tokenisation failures,
and bm25s import errors all degrade to an empty list rather than raising.
"""
from __future__ import annotations

import os
import re
import threading
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]+")


def _tokenise(text: str) -> list[str]:
    """Split a qualified name / free-text query into BM25 tokens.

    Dots and underscores are converted to whitespace so ``foo.bar_baz`` yields
    ``[foo, bar, baz]``.  Tokens are lower-cased; pure-numeric runs and single
    characters are dropped (they carry no symbol-recall signal and inflate the
    term-document matrix).
    """
    if not text:
        return []
    normalised = text.replace(".", " ").replace("_", " ")
    return [t.lower() for t in _TOKEN_RE.findall(normalised)]


class BM25Service:
    """Lazy, per-repo BM25 retriever with mtime-based cache invalidation."""

    def __init__(self) -> None:
        # path -> (retriever, doc_ids, mtime)
        self._cache: dict[str, tuple[Any, list[str], float]] = {}
        self._lock = threading.RLock()

    def _build(self, vec_db_path: str) -> tuple[Any, list[str]]:
        """Construct a fresh BM25 index from the embeddings table.

        Returns ``(None, [])`` when the table is empty so callers can short-
        circuit without a useless retriever object in the cache.

        Reads from the per-repo ``.duck`` file via DuckDB (v5.3 §6.5 + §8.4).
        The ``embeddings`` table holds ``qualified_name`` + ``symbol_type``
        alongside the FLOAT[768] vector; BM25 only needs the text columns.
        """
        import bm25s
        from codebase_rag.storage.vector_store import open_or_create

        conn = open_or_create(vec_db_path)
        try:
            rows = conn.execute(
                "SELECT qualified_name, symbol_type FROM embeddings"
            ).fetchall()
        finally:
            try:
                conn.close()
            except Exception:
                pass

        if not rows:
            return None, []

        doc_ids: list[str] = [r[0] for r in rows]
        corpus_tokens: list[list[str]] = [
            _tokenise(f"{r[0]} {r[1] or ''}") for r in rows
        ]

        # Note: do NOT pass `corpus=` to the constructor.  bm25s ties that
        # argument to a parallel "documents" store used by `retrieve()` to
        # return original doc objects — when the shape doesn't match exactly,
        # retrieve() crashes with an inhomogeneous-array reshape error.  We
        # already maintain `doc_ids` ourselves; just index the tokens.
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        return retriever, doc_ids

    def search(
        self,
        vec_db_path: str,
        query: str,
        k: int = 50,
    ) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(qualified_name, score)`` BM25 matches.

        Args:
            vec_db_path: Filesystem path to the per-repo ``.duck`` file.
            query: Free-text or identifier query — will be tokenised the same
                way as the corpus.
            k: Maximum results to return.

        Returns:
            list[tuple[str, float]]: ranked matches. Empty when the corpus is
            empty, the file is missing, the query yields no tokens, or any
            internal failure occurs (best-effort contract).
        """
        if not query or not query.strip():
            return []

        try:
            mtime = os.path.getmtime(vec_db_path)
        except OSError:
            return []

        with self._lock:
            cached = self._cache.get(vec_db_path)
            if cached is None or cached[2] != mtime:
                try:
                    retriever, doc_ids = self._build(vec_db_path)
                except Exception:
                    return []
                if retriever is None:
                    # Empty corpus — don't cache; next index run will populate.
                    return []
                self._cache[vec_db_path] = (retriever, doc_ids, mtime)
            else:
                retriever, doc_ids, _ = cached

        try:
            import bm25s

            tokens = _tokenise(query)
            if not tokens:
                return []
            query_tokens = bm25s.tokenize(" ".join(tokens), stopwords=None)
            results, scores = retriever.retrieve(
                query_tokens,
                k=min(k, len(doc_ids)),
            )
            # bm25s returns shape (n_queries, k) — we issue exactly one query.
            return [
                (doc_ids[int(idx)], float(score))
                for idx, score in zip(results[0], scores[0])
            ]
        except Exception:
            return []


# Module-level singleton — safe to import from routers.
bm25_service = BM25Service()
