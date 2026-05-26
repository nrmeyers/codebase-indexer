"""Code-specific embedding model A/B path (Phase 1.3).

This module introduces an abstraction layer over the embedding provider so
the indexer can run two models in parallel during evaluation:

    e5-base-v2        Generic-text bi-encoder.  Legacy persisted model id
                      retained for back-compat with rows written before
                      the LE-129 Jina swap (2026-05-26). SageMaker is now
                      jina-code-v2-serverless; this module's MODEL_E5_BASE_V2
                      constant continues to name the legacy column.
                      Wraps the upstream codebase_rag SageMakerEmbedder
                      verbatim (no behaviour change).
    bge-code-v1       Code-specific bi-encoder (768-dim drop-in).  Active
                      only when SAGEMAKER_BGE_CODE_ENDPOINT is set;
                      otherwise embed() returns None and callers fall
                      back to the e5 path.

A/B strategy
------------
The DuckDB ``embeddings`` table gains two additive columns at first open:

    embedding_v2     FLOAT[768]   — embedding produced by the v2 model
    embedding_model  TEXT         — model id of the active embedding (e.g.
                                    'e5-base-v2' or 'bge-code-v1')

Default behaviour is preserved: ``EMBEDDING_MODEL_ACTIVE`` defaults to
``e5-base-v2`` and the existing ``embedding`` column is the source of truth.
When the operator flips ``EMBEDDING_MODEL_ACTIVE=bge-code-v1`` and configures
``SAGEMAKER_BGE_CODE_ENDPOINT``, ingestion ALSO writes ``embedding_v2`` and
sets ``embedding_model='bge-code-v1'``.  Search reads ``embedding_v2`` (with a
graceful fallback to ``embedding`` for rows where v2 is still NULL during a
partial migration).

The eval harness compares cosine recall between the two columns.  When the
v2 model wins by the roadmap-projected +5–20% nDCG@10, operators flip the
default and run ``scripts/embed-v2-backfill.py`` to populate v2 for legacy
rows.

Env vars
--------
    EMBEDDING_MODEL_ACTIVE         'e5-base-v2' (default) | 'bge-code-v1'
    SAGEMAKER_BGE_CODE_ENDPOINT    SageMaker endpoint name for v2 model
    SAGEMAKER_BGE_CODE_URL         Full invocation URL (preferred over name)
    SAGEMAKER_BGE_CODE_REGION      AWS region (default 'us-east-1')

Cost tracking
-------------
Every embed() call updates the per-instance ``cost_calls`` and
``cost_tokens`` counters so operators can attribute SageMaker spend to the
A/B path.  The backfill script reads these to enforce a hard $50 cap.
"""
from __future__ import annotations

import logging
import os
import threading
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# Module-level once-flag so the "v2 endpoint absent" warning only fires once
# per process rather than spamming the log on every embed call.
_v2_warning_emitted = False
_v2_warning_lock = threading.Lock()

# Canonical model identifiers — written verbatim into ``embedding_model``.
MODEL_E5_BASE_V2 = "e5-base-v2"
MODEL_BGE_CODE_V1 = "bge-code-v1"

DEFAULT_ACTIVE_MODEL = MODEL_E5_BASE_V2
EMBEDDING_DIM = 768


def active_model_name() -> str:
    """Return the currently-active embedding model id from env.

    Defaults to ``e5-base-v2`` so unconfigured deployments preserve the
    existing behaviour exactly.  Unknown values fall back to the default
    with a one-time warning.
    """
    raw = (os.environ.get("EMBEDDING_MODEL_ACTIVE") or DEFAULT_ACTIVE_MODEL).strip()
    if raw not in (MODEL_E5_BASE_V2, MODEL_BGE_CODE_V1):
        logger.warning(
            "EMBEDDING_MODEL_ACTIVE=%r is not a recognised model id; "
            "falling back to %s.  Valid values: %s, %s",
            raw,
            DEFAULT_ACTIVE_MODEL,
            MODEL_E5_BASE_V2,
            MODEL_BGE_CODE_V1,
        )
        return DEFAULT_ACTIVE_MODEL
    return raw


def is_v2_active() -> bool:
    """True when the active model writes/reads the ``embedding_v2`` column."""
    return active_model_name() == MODEL_BGE_CODE_V1


class CodeEmbedder(ABC):
    """Common interface for the A/B embedding providers.

    Implementations expose the same shape as the upstream SageMakerEmbedder
    (``embed(text) -> list[float] | None``) so callers can switch without
    knowing the concrete class.  ``embed()`` returns ``None`` when the
    backend is unavailable — callers MUST fall back to the e5 path rather
    than raising.
    """

    model_name: str

    def __init__(self) -> None:
        self.cost_calls: int = 0
        self.cost_tokens: int = 0

    @abstractmethod
    def embed(self, text: str) -> list[float] | None:
        """Return a 768-dim embedding for ``text``, or ``None`` on miss."""
        raise NotImplementedError

    def _record_cost(self, text: str) -> None:
        """Update cost counters.  Token estimate is conservative (chars / 4)."""
        self.cost_calls += 1
        # Cheap proxy — exact token counts require the tokenizer of each model.
        # 4 chars/token is a stable rule-of-thumb for English + identifiers.
        self.cost_tokens += max(1, len(text) // 4)


class E5BaseV2Embedder(CodeEmbedder):
    """Wraps the configured embedder backend (``e5-base-v2``).

    Originally delegated to ``app.services.sagemaker_embedder``; after the
    BUC-1605 pluggable-backend migration this delegates to the
    :mod:`app.embedders` factory so the A/B harness picks up whatever
    backend the operator has configured (``local`` for laptops,
    ``sagemaker`` for the Navistone prod deploy). The production path is
    identical to the pre-Phase-1.3 behaviour when ``EMBEDDER_BACKEND=sagemaker``
    — only the cost counters are added on top.
    """

    model_name = MODEL_E5_BASE_V2

    def embed(self, text: str) -> list[float] | None:
        # Local import — keeps cold-start cheap and avoids a circular ref
        # if app.embedders ever pulls from this module.
        from app.embedders.sync_bridge import embed_text_sync

        vec = embed_text_sync(text)
        if vec:
            self._record_cost(text)
            return vec
        return None


class BgeCodeV1Embedder(CodeEmbedder):
    """SageMaker-backed BGE-Code-v1 embedder (code-specific, 768-dim).

    Activates ONLY when ``SAGEMAKER_BGE_CODE_ENDPOINT`` (or
    ``SAGEMAKER_BGE_CODE_URL``) is set.  When absent, ``embed()`` returns
    ``None`` and emits a one-time WARN — callers then fall back to E5.

    The endpoint must serve a 768-dim model so the vector is a drop-in
    replacement for E5 in the existing ``FLOAT[768]`` schema.  Both
    ``bge-code-v1`` and ``CodeRankEmbed`` qualify (MIT, 768-dim).
    """

    model_name = MODEL_BGE_CODE_V1

    def __init__(self) -> None:
        super().__init__()
        self._inner: Any | None = None
        self._tried_init = False

    def _build_inner(self) -> Any | None:
        """Construct a SageMaker backend pointing at the v2 endpoint.

        Reads ``SAGEMAKER_BGE_CODE_*`` env vars rather than the standard
        ``SAGEMAKER_EMBED_*`` ones so the v1 (E5) and v2 (BGE) endpoints can
        coexist.  Returns ``None`` when the endpoint is not configured.

        Uses :class:`app.embedders.sagemaker.SageMakerEmbedder` directly
        (not the env-var-driven factory) so the v2 endpoint stays isolated
        from the production E5 backend selection.
        """
        url = (os.environ.get("SAGEMAKER_BGE_CODE_URL") or "").strip()
        endpoint = (os.environ.get("SAGEMAKER_BGE_CODE_ENDPOINT") or "").strip()
        if not url and not endpoint:
            self._warn_once()
            return None

        region = (os.environ.get("SAGEMAKER_BGE_CODE_REGION") or "us-east-1").strip()
        try:
            from app.embedders.sagemaker import SageMakerEmbedder
        except ImportError as exc:  # pragma: no cover — surfaced clearly
            logger.warning("BGE-Code v2 embedder unavailable: %s", exc)
            return None

        # Construct directly with the v2 endpoint name (don't reuse from_env(),
        # which reads the v1 env vars).  When only URL is set, extract name.
        if not endpoint and url:
            endpoint = SageMakerEmbedder._extract_endpoint_name(url)  # noqa: SLF001
        if not endpoint:
            return None

        try:
            batch_size = int(os.environ.get("SAGEMAKER_BGE_CODE_BATCH_SIZE") or 16)
        except (TypeError, ValueError):
            batch_size = 16

        try:
            return SageMakerEmbedder(
                endpoint_name=endpoint,
                region=region,
                batch_size=batch_size,
            )
        except Exception as exc:  # noqa: BLE001
            # SageMakerEmbedder raises EmbedderError on misconfig; treat
            # that as "unavailable" so the caller falls back to E5.
            logger.warning("BGE-Code v2 embedder unavailable: %s", exc)
            return None

    @staticmethod
    def _warn_once() -> None:
        """Emit the missing-endpoint warning at most once per process."""
        global _v2_warning_emitted  # noqa: PLW0603
        with _v2_warning_lock:
            if _v2_warning_emitted:
                return
            _v2_warning_emitted = True
        logger.warning(
            "EMBEDDING_MODEL_ACTIVE=bge-code-v1 but SAGEMAKER_BGE_CODE_ENDPOINT "
            "(and SAGEMAKER_BGE_CODE_URL) are unset — v2 embeddings will not "
            "be written.  Falling back to e5-base-v2 for ingestion and search.  "
            "Configure the v2 SageMaker endpoint to enable the A/B path."
        )

    def embed(self, text: str) -> list[float] | None:
        if not self._tried_init:
            self._tried_init = True
            self._inner = self._build_inner()
        if self._inner is None:
            return None
        # ``app.embedders.sagemaker.SageMakerEmbedder.embed`` is async and
        # batched. Run it on a fresh event loop and unwrap the single
        # result so the legacy sync ``list[float] | None`` contract is
        # preserved for this A/B harness.
        import asyncio

        try:
            vectors = asyncio.run(self._inner.embed([text]))
        except Exception as exc:  # noqa: BLE001 — preserve fail-soft contract
            logger.warning("BGE-Code v2 embed failed: %s", exc)
            return None

        if vectors and vectors[0]:
            self._record_cost(text)
            return vectors[0]
        return None


@lru_cache(maxsize=4)
def get_embedder(model_name: str) -> CodeEmbedder:
    """Factory returning the embedder for ``model_name``.

    Cached so repeated lookups in a hot path don't reconstruct the wrapper.
    Use ``get_embedder.cache_clear()`` in tests when env vars change between
    cases.

    Raises:
        ValueError: When ``model_name`` is not a recognised model id.
    """
    if model_name == MODEL_E5_BASE_V2:
        return E5BaseV2Embedder()
    if model_name == MODEL_BGE_CODE_V1:
        return BgeCodeV1Embedder()
    raise ValueError(
        f"Unknown embedding model: {model_name!r}.  "
        f"Valid values: {MODEL_E5_BASE_V2!r}, {MODEL_BGE_CODE_V1!r}"
    )


def reset_v2_warning_for_tests() -> None:
    """Test-only: re-arm the once-warning so each test sees a fresh emit."""
    global _v2_warning_emitted  # noqa: PLW0603
    with _v2_warning_lock:
        _v2_warning_emitted = False


# ---------------------------------------------------------------------------
# Schema migration — additive ALTER TABLE for embedding_v2 + embedding_model
# ---------------------------------------------------------------------------


def ensure_v2_schema(conn: Any) -> None:
    """Add ``embedding_v2`` and ``embedding_model`` columns idempotently.

    Mirrors the BUC-1518 C2 ``content_hash`` migration pattern in upstream
    ``codebase_rag.storage.vector_store.open_or_create``: DuckDB doesn't
    support ``ADD COLUMN IF NOT EXISTS``, so we guard with a presence check
    against ``information_schema.columns`` and swallow any error so a
    transient DB hiccup never breaks ingestion.

    Safe to call multiple times.  No-op when both columns are already present.

    Args:
        conn: Open DuckDB connection (from
            ``codebase_rag.storage.vector_store.open_or_create``).
    """
    try:
        existing = {
            r[0]
            for r in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'embeddings'"
            ).fetchall()
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug("ensure_v2_schema: column probe failed: %s", exc)
        return

    if "embedding_v2" not in existing:
        try:
            conn.execute(
                f"ALTER TABLE embeddings ADD COLUMN embedding_v2 FLOAT[{EMBEDDING_DIM}]"
            )
            logger.info("Added embedding_v2 FLOAT[%d] column to embeddings", EMBEDDING_DIM)
        except Exception as exc:  # noqa: BLE001
            logger.debug("ensure_v2_schema: ADD embedding_v2 failed: %s", exc)

    if "embedding_model" not in existing:
        try:
            conn.execute(
                "ALTER TABLE embeddings ADD COLUMN embedding_model TEXT"
            )
            logger.info("Added embedding_model TEXT column to embeddings")
        except Exception as exc:  # noqa: BLE001
            logger.debug("ensure_v2_schema: ADD embedding_model failed: %s", exc)


def _l2_normalise(vec: list[float]) -> list[float]:
    """Local L2-normalise (avoids reaching into upstream privates)."""
    import math
    mag = math.sqrt(sum(x * x for x in vec))
    if mag == 0.0:
        return list(vec)
    return [x / mag for x in vec]


def search_similar_v2(conn: Any, query_vec: list[float], k: int = 10) -> list[Any]:
    """Cosine search against ``embedding_v2`` with graceful fallback.

    During a partial v2 migration some rows have a populated ``embedding_v2``
    while older rows do not.  We COALESCE so the query can still rank every
    row using whichever column is populated.  When the v2 column is missing
    entirely (legacy ``.duck`` files), we delegate to upstream
    ``search_similar`` which uses ``embedding``.

    Returns a list of upstream ``SearchResult`` instances so callers don't
    need to know which path was taken.
    """
    from codebase_rag.storage.vector_store import SearchResult, search_similar

    if not has_v2_column(conn):
        return search_similar(conn, query_vec, k=k)

    normalised = _l2_normalise(query_vec)
    rows = conn.execute(
        f"""
        SELECT qualified_name, file_path, start_line, end_line,
               1.0 - array_cosine_distance(
                   COALESCE(embedding_v2, embedding),
                   ?::FLOAT[{EMBEDDING_DIM}]
               ) AS score
        FROM embeddings
        WHERE embedding_v2 IS NOT NULL OR embedding IS NOT NULL
        ORDER BY score DESC
        LIMIT ?
        """,
        (normalised, int(k)),
    ).fetchall()

    return [
        SearchResult(
            qualified_name=r[0],
            file_path=r[1] or "",
            start_line=int(r[2]) if r[2] is not None else 0,
            end_line=int(r[3]) if r[3] is not None else 0,
            score=float(r[4]),
        )
        for r in rows
    ]


def has_v2_column(conn: Any) -> bool:
    """Return True when the ``embedding_v2`` column exists on ``embeddings``."""
    try:
        existing = {
            r[0]
            for r in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'embeddings'"
            ).fetchall()
        }
    except Exception:
        return False
    return "embedding_v2" in existing
