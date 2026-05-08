"""Per-repo topic centroid — mean-pool of top-k centrality embeddings (BUC-1581).

Phase 2.5 v1 prerequisite for TheForge's ``getRepoAffinity`` (BUC-1575).
TheForge's affinity module shipped as a v0 stub returning uniform weights
because the indexer didn't expose per-symbol embeddings client-side. This
module precomputes a single 768-dim "topic centroid" per repo by:

    1. Reading the top-k qualified names from the per-repo ``centrality``
       table (populated by Plan J / BUC-1577 PageRank persistence).
    2. Joining them against the per-repo ``embeddings`` table (BUC-1573
       ``embedding_v2`` column with fallback to legacy ``embedding``).
    3. Mean-pooling the resulting (k, 768) matrix to a single 768-dim vector.

The caller (TheForge) decides whether to L2-normalise — we do not normalise
at compute time so consumers can preserve magnitude information when useful.

Caching contract:
    In-process LRU keyed by ``(slug, k, duck_mtime)`` with a 1h soft TTL.
    The ``duck_mtime`` component invalidates automatically on re-index, so
    the TTL is belt-and-braces (catches edge cases like manual file mutation
    during dev). Centroids are stable across queries within an index window
    so a cache hit is the common case.

Failure surface:
    * ``CentroidUnavailableError`` — slug exists but neither the centrality
      table nor the embedding column is populated yet. The HTTP layer maps
      this to 503 so TheForge can poll a freshly-indexed repo without
      treating it as a hard failure.
    * ``CentroidNotFoundError`` — the ``.duck`` file is absent. Maps to 404.

This module **never raises** for arbitrary errors in the hot path: any
unexpected DuckDB / sibling-package failure is logged and re-raised as
``CentroidUnavailableError`` so the caller serves 503 (graceful degradation)
rather than leaking a 5xx.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


# 768 dims — must match ``EMBEDDING_DIM`` in ``app.services.embedder``.
# Hard-coded here (not imported) so this module has no upstream import cycle
# with the embedder; the dim is a stable cross-service contract anyway.
EMBEDDING_DIM = 768

# Cache TTL — 1h. Centroids are stable across queries within a single index
# window. Re-indexing bumps the .duck file mtime which invalidates the entry
# regardless of TTL via the ``duck_mtime`` cache key component.
_CACHE_TTL_SECONDS = 3600

# Default top-k centrality symbols to pool. Matches the brief and the
# downstream affinity module's expectations.
_DEFAULT_K = 20

# Hard ceiling on k — beyond ~200 the centroid loses topical specificity and
# starts approximating the repo-wide mean (which is uninformative for
# affinity routing).
_MAX_K = 200


class CentroidNotFoundError(Exception):
    """Repo has not been indexed at all (no ``.duck`` file). Maps to HTTP 404."""


class CentroidUnavailableError(Exception):
    """Repo indexed but centroid cannot be computed yet. Maps to HTTP 503.

    Reasons include:
        * ``centrality`` table empty (PageRank not yet computed).
        * ``embeddings`` table has no rows for the centrality qnames.
        * Sibling ``codebase_rag`` package unavailable in this environment.
    """


@dataclass(frozen=True)
class CentroidResult:
    """One materialised centroid for a repo.

    Attributes:
        repo: Repo slug — same key the rest of the API uses.
        centroid: 768-dim ``list[float]``, mean-pooled (NOT L2-normalised).
            Caller normalises if it wants cosine-friendly arithmetic.
        computed_at: UTC datetime the underlying compute ran. On a cache hit
            this reflects the cached entry's compute time, not the request
            time — so callers can detect a stale-but-served centroid.
        cache_age_seconds: Wall-clock seconds since ``computed_at``.
        k: Effective k used for the mean-pool. May be smaller than the
            requested k when the centrality table has fewer rows than asked.
    """

    repo: str
    centroid: list[float]
    computed_at: datetime
    cache_age_seconds: int
    k: int


# ---------------------------------------------------------------------------
# Cache — process-local, keyed on (slug, k, duck_mtime)
# ---------------------------------------------------------------------------
#
# We don't use ``functools.lru_cache`` because we need to:
#   1. Track a TTL (lru_cache has no expiry).
#   2. Recompute ``cache_age_seconds`` on every read.
#   3. Invalidate on .duck mtime change (re-index).


@dataclass(frozen=True)
class _CacheEntry:
    centroid: list[float]
    computed_at: float  # epoch seconds
    k: int


_cache: dict[tuple[str, int, float], _CacheEntry] = {}
_cache_lock = Lock()


def clear_cache() -> None:
    """Drop every cached centroid. Test-only; not exposed via HTTP.

    The production path relies on TTL + mtime invalidation. This helper
    exists so tests can assert deterministic compute behaviour without
    waiting for the 1h TTL to lapse between cases.
    """
    with _cache_lock:
        _cache.clear()


def _read_top_k_qnames(conn: Any, k: int) -> list[str]:
    """Fetch the top-k qnames from the ``centrality`` table.

    Pure read — does not require the LadybugDB. Returns an empty list if
    the table doesn't exist or is empty (PageRank not yet computed). Any
    other DB error is logged and re-raised so the caller can decide
    whether to degrade or surface.
    """
    try:
        rows = conn.execute(
            "SELECT qualified_name FROM centrality "
            "ORDER BY pagerank DESC LIMIT ?",
            (int(k),),
        ).fetchall()
    except Exception as exc:
        # Table missing → empty list (graceful). DuckDB raises CatalogException
        # which we catch generically; production environments may surface
        # the underlying duckdb.* exception or a wrapped variant.
        logger.warning("centroid.centrality_read_failed err=%s", exc)
        return []

    return [str(r[0]) for r in rows if r and r[0]]


def _read_embeddings_for_qnames(
    conn: Any, qnames: list[str]
) -> list[list[float]]:
    """Fetch embedding vectors for ``qnames``, preferring ``embedding_v2``.

    Returns the list of 768-dim vectors in the order DuckDB returns them
    (which is unordered with respect to the input list — that's fine,
    mean-pool is commutative). Skips rows whose vector column is NULL on
    both ``embedding_v2`` and ``embedding``.
    """
    if not qnames:
        return []

    placeholders = ",".join(["?"] * len(qnames))

    # Detect which columns exist. ``embedding_v2`` is added by
    # app/services/embedder.py via additive ALTER TABLE, so older .duck
    # files will only have ``embedding``. We fall back gracefully.
    try:
        existing_cols = {
            r[0]
            for r in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'embeddings'"
            ).fetchall()
        }
    except Exception as exc:
        logger.warning("centroid.schema_probe_failed err=%s", exc)
        return []

    if "embedding_v2" in existing_cols:
        select_expr = "COALESCE(embedding_v2, embedding)"
        where_expr = "(embedding_v2 IS NOT NULL OR embedding IS NOT NULL)"
    elif "embedding" in existing_cols:
        select_expr = "embedding"
        where_expr = "embedding IS NOT NULL"
    else:
        # No embedding columns at all — pre-BUC-1573 schema.
        return []

    try:
        rows = conn.execute(
            f"SELECT {select_expr} "
            f"FROM embeddings "
            f"WHERE qualified_name IN ({placeholders}) AND {where_expr}",
            qnames,
        ).fetchall()
    except Exception as exc:
        logger.warning("centroid.embedding_read_failed err=%s", exc)
        return []

    out: list[list[float]] = []
    for r in rows:
        if not r or r[0] is None:
            continue
        try:
            vec = list(r[0])
        except TypeError:
            # Defensive: DuckDB FLOAT[N] should always be iterable.
            continue
        if len(vec) != EMBEDDING_DIM:
            # Skip dimension mismatches (mixed-model repos in flight); we'd
            # rather under-report k than poison the centroid with garbage.
            logger.warning(
                "centroid.dim_mismatch expected=%d got=%d",
                EMBEDDING_DIM,
                len(vec),
            )
            continue
        out.append([float(x) for x in vec])

    return out


def _mean_pool(vectors: list[list[float]]) -> list[float]:
    """Mean-pool an (n, 768) matrix to a single 768-dim vector.

    Pure compute. Caller has already validated that every vector is exactly
    768-dim, so we can drop the inner length check from the hot loop.
    """
    n = len(vectors)
    if n == 0:
        # Caller must guard against this — mean of zero vectors is undefined.
        # We surface as an empty list so the type stays ``list[float]``;
        # the upstream service raises CentroidUnavailableError before this
        # branch can be reached in production.
        return []

    accum = [0.0] * EMBEDDING_DIM
    for vec in vectors:
        for i in range(EMBEDDING_DIM):
            accum[i] += vec[i]

    inv_n = 1.0 / n
    return [x * inv_n for x in accum]


def compute_repo_centroid(
    repo: str,
    duck_path: str | Path,
    k: int = _DEFAULT_K,
) -> CentroidResult:
    """Compute (or fetch from cache) the topic centroid for ``repo``.

    Args:
        repo: Repo slug — used as the response label and cache key.
        duck_path: Path to the per-repo ``.duck`` file. The caller resolves
            this via ``settings.vec_db_path_for_repo(repo)`` so this module
            stays decoupled from FastAPI request state.
        k: Number of top-centrality symbols to mean-pool. Clamped to
            ``[1, _MAX_K]``; defaults to ``_DEFAULT_K`` (20).

    Returns:
        CentroidResult: The materialised centroid plus cache metadata.

    Raises:
        CentroidNotFoundError: ``duck_path`` does not exist.
        CentroidUnavailableError: The .duck file exists but the centrality
            table or embedding column is empty / unavailable.
    """
    # Clamp k to the supported range. We don't reject — silently clamping
    # gives the HTTP layer a clean contract (Query(ge=1, le=200)) without
    # forcing every caller to validate.
    k = max(1, min(int(k), _MAX_K))

    duck_path = Path(duck_path)
    if not duck_path.exists():
        raise CentroidNotFoundError(
            f"No .duck file for repo '{repo}' at {duck_path}"
        )

    # Cache key includes mtime so a re-index automatically invalidates the
    # cached centroid even before the TTL expires.
    try:
        duck_mtime = duck_path.stat().st_mtime
    except OSError as exc:
        # Can't stat — treat as not-found (race with deletion).
        raise CentroidNotFoundError(
            f"Cannot stat .duck file for repo '{repo}': {exc}"
        ) from exc

    cache_key = (repo, k, duck_mtime)
    now = time.time()

    with _cache_lock:
        entry = _cache.get(cache_key)
        if entry is not None and (now - entry.computed_at) < _CACHE_TTL_SECONDS:
            return CentroidResult(
                repo=repo,
                centroid=entry.centroid,
                computed_at=datetime.fromtimestamp(entry.computed_at, tz=timezone.utc),
                cache_age_seconds=int(now - entry.computed_at),
                k=entry.k,
            )

    # Cold path — open the .duck file and compute. We open + close inside
    # this function so we don't pin a connection across the cache lifetime.
    try:
        from codebase_rag.storage.vector_store import open_or_create  # type: ignore[import-untyped]
    except ImportError as exc:
        logger.warning("centroid.no_sibling_pkg repo=%s", repo)
        raise CentroidUnavailableError(
            f"codebase_rag sibling package not importable; cannot compute "
            f"centroid for '{repo}'"
        ) from exc

    try:
        conn = open_or_create(str(duck_path))
    except Exception as exc:
        logger.warning("centroid.open_failed repo=%s err=%s", repo, exc)
        raise CentroidUnavailableError(
            f"Failed to open .duck file for '{repo}': {exc}"
        ) from exc

    try:
        qnames = _read_top_k_qnames(conn, k)
        if not qnames:
            raise CentroidUnavailableError(
                f"Centrality table empty for '{repo}' — PageRank may not "
                "have run yet. Try POST /repos/{name}/reindex."
            )

        vectors = _read_embeddings_for_qnames(conn, qnames)
        if not vectors:
            raise CentroidUnavailableError(
                f"No embeddings found for top-{k} centrality symbols of "
                f"'{repo}'. The repo may have been indexed without the "
                "v2 embedding model — check SAGEMAKER_BGE_E5_URL."
            )

        centroid = _mean_pool(vectors)
        if not centroid:
            # Defensive — _mean_pool returns [] only on empty input, which
            # we already guarded above. Belt-and-braces.
            raise CentroidUnavailableError(
                f"Mean-pool produced empty vector for '{repo}'"
            )
    finally:
        try:
            conn.close()
        except Exception:
            # Best-effort close; an already-closed handle is harmless.
            pass

    # Write-through cache. Re-acquire the lock; another request may have
    # populated the same key in the meantime — last writer wins (safe
    # because the result is deterministic for a given (repo, k, mtime)).
    with _cache_lock:
        _cache[cache_key] = _CacheEntry(
            centroid=centroid,
            computed_at=now,
            k=len(vectors),
        )

    return CentroidResult(
        repo=repo,
        centroid=centroid,
        computed_at=datetime.fromtimestamp(now, tz=timezone.utc),
        cache_age_seconds=0,
        k=len(vectors),
    )
