"""POST /embed — embed an arbitrary text string via the configured embedder
backend (typically SageMaker in production; ``local`` for standalone installs).

Returns a 768-dim e5-base-v2 vector ready for KNN / cosine ops against the
indexer's per-repo centroids (``GET /repos/{name}/centroid``, BUC-1581).

BUC-1592: unblocks TheForge's cross-repo affinity weighting in real chat —
``weightReposByCentroidAffinity`` already exists on the orchestrator side,
it was the *query vector* (not the per-repo centroids) that was missing.

Design notes:
    * The embedder factory (:mod:`app.embedders`, BUC-1605) raises
      :class:`EmbedderError` on hard failures (no endpoint, protocol
      mismatch). We translate those to 503 so callers can fail-open at
      the orchestrator layer (uniform repo weights on failure, audit row,
      chat continues).
    * Auth middleware (BUC-1431 bearer) applies automatically — no extra
      wiring at the router level.
    * Length cap of 4000 chars is generous compared to the ~512-token /
      ~2000-char effective window of e5-base-v2 (the model truncates
      internally) but keeps the request body bounded for sanity.
"""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..embedders import EmbedderError, get_embedder

logger = logging.getLogger(__name__)

router = APIRouter()

# Pinned model identifier reported by this router. The SageMaker endpoint
# was swapped to jina-code-v2-serverless on 2026-05-26 (LE-129); SageMakerEmbedder
# reports the new model name itself when used. This constant remains "e5-base-v2"
# for the local/TEI backends until they are migrated too — keep in sync with
# whichever backend the deployment actually uses.
_MODEL_NAME = "e5-base-v2"


class EmbedRequest(BaseModel):
    """POST body — single text string to embed.

    Length bounds:
        * min 1 — reject empty strings via 422 rather than wasting a
          SageMaker invocation on a no-op.
        * max 4000 — generous compared to the effective ~2000-char window
          of e5-base-v2. Keeps request bodies bounded.
    """

    text: str = Field(min_length=1, max_length=4000)


class EmbedResponse(BaseModel):
    """Successful response — 768-dim vector + provenance metadata."""

    embedding: list[float]
    dims: int
    model: str


@router.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
    """Embed ``req.text`` via the configured embedder backend.

    Returns:
        EmbedResponse: ``embedding`` (768-dim float list), ``dims`` (== 768
        in practice — populated from the actual vector length so a
        downstream model swap is observable), and ``model`` (pinned to
        ``"e5-base-v2"``).

    Raises:
        HTTPException: 503 when the embedder backend is unconfigured,
            unreachable, or returns no vector. Callers (TheForge
            orchestrator) treat this as "fall back to uniform weights",
            audit the failure, and let the chat turn complete normally.
    """
    # Resolve the backend lazily — EmbedderError surfaces both
    # "no backend configured" and "backend misconfigured" as a single
    # 503 with a descriptive detail. The legacy shim conflated these by
    # returning ``None``; the new factory raises, which is strictly more
    # informative.
    try:
        backend = get_embedder()
    except EmbedderError as exc:
        logger.warning("embed: backend unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"embed unavailable: {exc}",
        ) from exc

    t0 = time.time()
    try:
        vectors = await backend.embed([req.text])
    except EmbedderError as exc:
        logger.warning("embed: backend %s call failed: %s", backend.name, exc)
        raise HTTPException(
            status_code=503,
            detail=f"embed failed: {exc}",
        ) from exc
    except Exception as exc:  # noqa: BLE001 — translate any backend error to 503
        logger.warning("embed: backend %s unexpected error: %s", backend.name, exc)
        raise HTTPException(
            status_code=503,
            detail=f"embed failed: {exc}",
        ) from exc

    if not vectors or not vectors[0]:
        # Backend returned no vector — typically a transient cold-start
        # failure or a downstream HTTP timeout already swallowed inside
        # the backend. 503 keeps the contract simple for the orchestrator.
        logger.warning(
            "embed: backend %s returned no vector for %d-char input",
            backend.name, len(req.text),
        )
        raise HTTPException(
            status_code=503,
            detail="embed failed: empty vector from backend",
        )

    vec = vectors[0]
    logger.info(
        "embed: backend=%s model=%s dims=%d latency_ms=%d",
        backend.name, _MODEL_NAME, len(vec), int((time.time() - t0) * 1000),
    )
    return EmbedResponse(embedding=vec, dims=len(vec), model=_MODEL_NAME)
