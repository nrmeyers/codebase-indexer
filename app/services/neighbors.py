"""Per-repo embedding neighbors + semantic clustering (LE-158 Stream C).

Powers TheForge's knowledge-graph viewer: ``similar_to`` edges (nearest
neighbors of a focused symbol) and semantic *layers* (clusters of
topically-related symbols). Both operate purely on the per-repo DuckDB
``.duck`` vector store the indexer already builds — no re-embedding, no
graph traversal.

Design mirrors :mod:`app.services.centroid` (BUC-1581):

    * Read the per-repo ``embeddings`` table via the sibling
      ``codebase_rag.storage.vector_store.open_or_create`` connection,
      preferring the ``embedding_v2`` column with a fallback to legacy
      ``embedding`` (additive ALTER on older ``.duck`` files).
    * Open + close the connection inside each call so we never pin a
      handle across the cache lifetime.
    * Hard caps on the number of vectors processed so a giant repo can't
      stall the event loop (these are precompute-friendly, not hot-path,
      endpoints — callers are expected to materialise layers offline).

Failure surface (matches centroid):
    * :class:`NeighborsNotFoundError` — the ``.duck`` file is absent →
      HTTP 404.
    * :class:`NeighborsUnavailableError` — file present but no usable
      embeddings (pre-embedding schema, empty table, sibling package
      missing) → HTTP 503.

Cosine + clustering are NumPy-only (NumPy ships transitively with the
hard ``scipy`` dependency — no new package). K-means is a small,
deterministic Lloyd's-iteration implementation seeded by k-means++ so we
avoid pulling in scikit-learn.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# Must match ``EMBEDDING_DIM`` in app.services.centroid / app.services.embedder.
EMBEDDING_DIM = 768

# Hard ceiling on vectors loaded into memory for a single neighbors/clusters
# call. ~50k * 768 * 4 bytes ≈ 150 MB as float32 — bounded and safe. Beyond
# this we truncate (deterministic ORDER BY) and log; these endpoints are
# precompute-friendly, not latency-critical.
_MAX_SYMBOLS = 50_000

# Default / ceiling for neighbor count.
_DEFAULT_K = 10
_MAX_K = 100

# Cluster-count bounds.
_DEFAULT_N_CLUSTERS = 8
_MAX_N_CLUSTERS = 64

# K-means iteration cap — Lloyd's converges fast on normalised embeddings;
# 25 iterations is plenty and keeps the worst-case bounded.
_KMEANS_MAX_ITERS = 25


class NeighborsNotFoundError(Exception):
    """Repo has not been indexed (no ``.duck`` file). Maps to HTTP 404."""


class NeighborsUnavailableError(Exception):
    """Repo indexed but embeddings unavailable. Maps to HTTP 503."""


@dataclass(frozen=True)
class Neighbor:
    """One nearest-neighbor result."""

    fqn: str
    score: float  # cosine similarity in [-1, 1]; higher = more similar.


@dataclass(frozen=True)
class Cluster:
    """One semantic cluster (a "layer" in the graph viewer)."""

    cluster_id: int
    label: str  # representative fqn (closest to the cluster centroid).
    fqns: list[str]


# ---------------------------------------------------------------------------
# DuckDB read helpers (shared shape with app.services.centroid)
# ---------------------------------------------------------------------------


def _open_conn(repo: str, duck_path: Path) -> Any:
    """Open the per-repo ``.duck`` store, translating failures to typed errors.

    Raises:
        NeighborsNotFoundError: file absent.
        NeighborsUnavailableError: sibling package missing or open failed.
    """
    if not duck_path.exists():
        raise NeighborsNotFoundError(
            f"No .duck file for repo '{repo}' at {duck_path}"
        )

    try:
        from codebase_rag.storage.vector_store import (  # type: ignore[import-untyped]
            open_or_create,
        )
    except ImportError as exc:
        logger.warning("neighbors.no_sibling_pkg repo=%s", repo)
        raise NeighborsUnavailableError(
            f"codebase_rag sibling package not importable; cannot read "
            f"embeddings for '{repo}'"
        ) from exc

    try:
        return open_or_create(str(duck_path))
    except Exception as exc:  # noqa: BLE001 — translate any open failure to 503
        logger.warning("neighbors.open_failed repo=%s err=%s", repo, exc)
        raise NeighborsUnavailableError(
            f"Failed to open .duck file for '{repo}': {exc}"
        ) from exc


def _embedding_select_clause(conn: Any) -> tuple[str, str]:
    """Resolve the (select_expr, where_expr) for the embedding column.

    Prefers ``embedding_v2`` with a COALESCE fallback to legacy ``embedding``;
    falls back to ``embedding`` alone on older schemas. Raises
    :class:`NeighborsUnavailableError` when no embedding column exists.
    """
    try:
        existing_cols = {
            r[0]
            for r in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'embeddings'"
            ).fetchall()
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("neighbors.schema_probe_failed err=%s", exc)
        raise NeighborsUnavailableError(
            f"Cannot probe embeddings schema: {exc}"
        ) from exc

    if "embedding_v2" in existing_cols:
        return (
            "COALESCE(embedding_v2, embedding)",
            "(embedding_v2 IS NOT NULL OR embedding IS NOT NULL)",
        )
    if "embedding" in existing_cols:
        return "embedding", "embedding IS NOT NULL"
    raise NeighborsUnavailableError(
        "No embedding column on 'embeddings' table (pre-embedding schema)."
    )


def _coerce_vec(raw: Any) -> list[float] | None:
    """Coerce a DuckDB FLOAT[N] row value into a validated 768-dim list."""
    if raw is None:
        return None
    try:
        vec = list(raw)
    except TypeError:
        return None
    if len(vec) != EMBEDDING_DIM:
        return None
    return [float(x) for x in vec]


def _load_all_vectors(
    conn: Any, select_expr: str, where_expr: str
) -> tuple[list[str], np.ndarray]:
    """Load up to ``_MAX_SYMBOLS`` (qname, vector) rows into a NumPy matrix.

    Returns ``(fqns, matrix)`` where ``matrix`` is ``(n, 768)`` float32.
    Deterministic ordering (``ORDER BY qualified_name``) so truncation on
    huge repos is stable across calls.
    """
    try:
        rows = conn.execute(
            f"SELECT qualified_name, {select_expr} AS vec "
            f"FROM embeddings WHERE {where_expr} "
            f"ORDER BY qualified_name LIMIT ?",
            (_MAX_SYMBOLS,),
        ).fetchall()
    except Exception as exc:  # noqa: BLE001
        logger.warning("neighbors.bulk_read_failed err=%s", exc)
        raise NeighborsUnavailableError(
            f"Failed to read embeddings: {exc}"
        ) from exc

    fqns: list[str] = []
    vecs: list[list[float]] = []
    for r in rows:
        if not r or r[0] is None:
            continue
        vec = _coerce_vec(r[1])
        if vec is None:
            continue
        fqns.append(str(r[0]))
        vecs.append(vec)

    if not vecs:
        raise NeighborsUnavailableError(
            "No usable embedding vectors found for repo."
        )

    matrix = np.asarray(vecs, dtype=np.float32)
    return fqns, matrix


def _read_one_vector(
    conn: Any, fqn: str, select_expr: str, where_expr: str
) -> list[float] | None:
    """Read a single symbol's embedding vector by exact qualified name."""
    try:
        rows = conn.execute(
            f"SELECT {select_expr} AS vec FROM embeddings "
            f"WHERE qualified_name = ? AND {where_expr} LIMIT 1",
            (fqn,),
        ).fetchall()
    except Exception as exc:  # noqa: BLE001
        logger.warning("neighbors.single_read_failed fqn=%s err=%s", fqn, exc)
        return None
    if not rows:
        return None
    return _coerce_vec(rows[0][0])


# ---------------------------------------------------------------------------
# Pure compute — cosine + k-means (NumPy only)
# ---------------------------------------------------------------------------


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2-normalise. Zero-norm rows are left as zeros (cosine 0)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _kmeans(
    matrix: np.ndarray, n_clusters: int, *, seed: int = 42
) -> np.ndarray:
    """Deterministic k-means++ seeded Lloyd's iteration on unit vectors.

    Operates on L2-normalised rows so squared-euclidean distance is a
    monotone proxy for cosine distance. Returns a ``(n,)`` int array of
    cluster assignments. Empty clusters are re-seeded to the point farthest
    from its current centroid to avoid degenerate collapse.
    """
    n = matrix.shape[0]
    k = min(n_clusters, n)
    rng = np.random.default_rng(seed)

    # k-means++ initialisation.
    centers = np.empty((k, matrix.shape[1]), dtype=np.float32)
    first = int(rng.integers(n))
    centers[0] = matrix[first]
    closest_sq = np.sum((matrix - centers[0]) ** 2, axis=1)
    for c in range(1, k):
        total = float(closest_sq.sum())
        if total <= 0.0:
            centers[c] = matrix[int(rng.integers(n))]
        else:
            probs = closest_sq / total
            nxt = int(rng.choice(n, p=probs))
            centers[c] = matrix[nxt]
        new_sq = np.sum((matrix - centers[c]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, new_sq)

    assignments = np.zeros(n, dtype=np.int64)
    for _ in range(_KMEANS_MAX_ITERS):
        # (n, k) squared distances via expansion ||a-b||^2.
        dists = (
            np.sum(matrix**2, axis=1, keepdims=True)
            - 2.0 * matrix @ centers.T
            + np.sum(centers**2, axis=1)
        )
        new_assignments = np.argmin(dists, axis=1)
        if np.array_equal(new_assignments, assignments):
            assignments = new_assignments
            break
        assignments = new_assignments

        for c in range(k):
            members = matrix[assignments == c]
            if members.shape[0] == 0:
                # Re-seed empty cluster to the globally worst-fit point.
                worst = int(np.argmax(np.min(dists, axis=1)))
                centers[c] = matrix[worst]
            else:
                centers[c] = members.mean(axis=0)

    return assignments


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_neighbors(
    repo: str, duck_path: str | Path, fqn: str, k: int = _DEFAULT_K
) -> list[Neighbor]:
    """Return the top-``k`` cosine-nearest symbols to ``fqn`` in ``repo``.

    Args:
        repo: Repo slug (response label only).
        duck_path: Path to the per-repo ``.duck`` store.
        fqn: Exact qualified name of the seed symbol.
        k: Number of neighbors to return (clamped to ``[1, 100]``).

    Returns:
        Neighbors sorted by descending cosine similarity, excluding the
        seed symbol itself. Empty list when ``fqn`` has no embedding (the
        HTTP layer renders this as an empty body, not a 404 — the repo
        *is* indexed, the symbol just isn't embedded).

    Raises:
        NeighborsNotFoundError: ``.duck`` file absent (HTTP 404).
        NeighborsUnavailableError: file present but no usable embeddings
            (HTTP 503).
    """
    k = max(1, min(int(k), _MAX_K))
    duck_path = Path(duck_path)

    conn = _open_conn(repo, duck_path)
    try:
        select_expr, where_expr = _embedding_select_clause(conn)
        seed = _read_one_vector(conn, fqn, select_expr, where_expr)
        if seed is None:
            # Repo indexed, embeddings present, but this symbol isn't one of
            # them (unknown / unembedded fqn). Empty result, not an error.
            return []
        fqns, matrix = _load_all_vectors(conn, select_expr, where_expr)
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001 — best-effort close
            pass

    seed_vec = np.asarray(seed, dtype=np.float32)
    seed_norm = float(np.linalg.norm(seed_vec))
    if seed_norm == 0.0:
        return []

    unit_matrix = _l2_normalize(matrix)
    sims = unit_matrix @ (seed_vec / seed_norm)  # (n,) cosine similarities.

    # argsort descending; over-fetch by one to drop the seed self-match.
    order = np.argsort(-sims)
    out: list[Neighbor] = []
    for idx in order:
        cand = fqns[int(idx)]
        if cand == fqn:
            continue
        out.append(Neighbor(fqn=cand, score=float(sims[int(idx)])))
        if len(out) >= k:
            break
    return out


def compute_clusters(
    repo: str, duck_path: str | Path, n_clusters: int = _DEFAULT_N_CLUSTERS
) -> list[Cluster]:
    """Partition the repo's symbol embeddings into ``n_clusters`` groups.

    Every loaded symbol is assigned to exactly one cluster (k-means is a
    hard partition). Each cluster's ``label`` is the member fqn closest to
    the cluster centroid (a cheap, deterministic representative).

    Args:
        repo: Repo slug.
        duck_path: Path to the per-repo ``.duck`` store.
        n_clusters: Requested cluster count (clamped to ``[1, 64]``).
            Reduced automatically when the repo has fewer symbols than
            requested clusters.

    Returns:
        Clusters sorted by descending member count. Empty clusters are
        omitted. The union of all ``fqns`` equals the set of embedded
        symbols loaded (capped at ``_MAX_SYMBOLS``).

    Raises:
        NeighborsNotFoundError: ``.duck`` file absent (HTTP 404).
        NeighborsUnavailableError: file present but no usable embeddings
            (HTTP 503).
    """
    n_clusters = max(1, min(int(n_clusters), _MAX_N_CLUSTERS))
    duck_path = Path(duck_path)

    conn = _open_conn(repo, duck_path)
    try:
        select_expr, where_expr = _embedding_select_clause(conn)
        fqns, matrix = _load_all_vectors(conn, select_expr, where_expr)
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001 — best-effort close
            pass

    unit_matrix = _l2_normalize(matrix)
    effective_k = min(n_clusters, unit_matrix.shape[0])
    assignments = _kmeans(unit_matrix, effective_k)

    clusters: list[Cluster] = []
    for c in range(effective_k):
        member_idx = np.flatnonzero(assignments == c)
        if member_idx.size == 0:
            continue
        members = unit_matrix[member_idx]
        centroid = members.mean(axis=0)
        # Representative = member closest to centroid (max cosine on units).
        sims = members @ centroid
        rep_local = int(np.argmax(sims))
        rep_global = int(member_idx[rep_local])
        member_fqns = [fqns[int(i)] for i in member_idx]
        clusters.append(
            Cluster(
                cluster_id=c,
                label=fqns[rep_global],
                fqns=member_fqns,
            )
        )

    clusters.sort(key=lambda cl: len(cl.fqns), reverse=True)
    # Re-id sequentially after the size sort so cluster_id is stable/contiguous.
    return [
        Cluster(cluster_id=i, label=cl.label, fqns=cl.fqns)
        for i, cl in enumerate(clusters)
    ]
