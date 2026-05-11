"""POST /embed — embed an arbitrary text string via the same SageMaker
endpoint used for symbol ingestion.

Returns a 768-dim e5-base-v2 vector ready for KNN / cosine ops against the
indexer's per-repo centroids (``GET /repos/{name}/centroid``, BUC-1581).

BUC-1592: unblocks TheForge's cross-repo affinity weighting in real chat —
``weightReposByCentroidAffinity`` already exists on the orchestrator side,
it was the *query vector* (not the per-repo centroids) that was missing.

Design notes:
    * The SageMaker embedder is synchronous and returns ``list[float] | None``
      on failure. We translate ``None`` to 503 so callers can fail-open at
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

from ..services.sagemaker_embedder import get_sagemaker_embedder

logger = logging.getLogger(__name__)

router = APIRouter()

# Pinned model identifier. The SageMaker endpoint is configured to serve
# e5-base-v2; if the team swaps endpoints, this constant should follow.
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
def embed(req: EmbedRequest) -> EmbedResponse:
    """Embed ``req.text`` via the shared SageMaker e5-base-v2 endpoint.

    Returns:
        EmbedResponse: ``embedding`` (768-dim float list), ``dims`` (== 768
        in practice — populated from the actual vector length so a
        downstream model swap is observable), and ``model`` (pinned to
        ``"e5-base-v2"``).

    Raises:
        HTTPException: 503 when the SageMaker endpoint is unconfigured,
            unreachable, or returns no vector. Callers (TheForge
            orchestrator) treat this as "fall back to uniform weights",
            audit the failure, and let the chat turn complete normally.
    """
    sm = get_sagemaker_embedder()
    if sm is None:
        # No endpoint configured — return 503 so the orchestrator can
        # fail-open. We never want to 500 here because this endpoint is
        # explicitly an optimisation, not a hard dependency.
        raise HTTPException(
            status_code=503,
            detail="embed unavailable: SageMaker endpoint not configured",
        )

    t0 = time.time()
    try:
        vec = sm.embed(req.text)
    except Exception as exc:  # noqa: BLE001 — translate any backend error to 503
        logger.warning("embed: SageMaker call failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"embed failed: {exc}",
        ) from exc

    if vec is None or len(vec) == 0:
        # Backend returned no vector — typically a transient SageMaker
        # cold-start failure or a downstream HTTP timeout already swallowed
        # inside ``SageMakerEmbedder.embed``. 503 keeps the contract simple
        # for the orchestrator.
        logger.warning("embed: SageMaker returned no vector for %d-char input", len(req.text))
        raise HTTPException(
            status_code=503,
            detail="embed failed: empty vector from backend",
        )

    logger.info(
        "embed: model=%s dims=%d latency_ms=%d",
        _MODEL_NAME, len(vec), int((time.time() - t0) * 1000),
    )
    return EmbedResponse(embedding=vec, dims=len(vec), model=_MODEL_NAME)
