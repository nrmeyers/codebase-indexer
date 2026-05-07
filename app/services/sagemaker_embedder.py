"""Re-export of the canonical ``SageMakerEmbedder`` from code-graph-rag.

Historically this module contained a near-copy of the embedder class.  After
BUC-1512 fixed the urllib hang in code-graph-rag, the two implementations
drifted and the local copy was never updated, so production calls went
through the upstream module instead.  We now re-export from there so there
is exactly one source of truth.

Public API (compatible with the previous version):
    SageMakerEmbedder       — class, ``from_env()`` factory and ``embed()`` method
    get_sagemaker_embedder  — module-level singleton accessor (LRU-cached)

Env vars (read by ``from_env()``, unchanged):
    SAGEMAKER_EMBED_URL        Full invocation URL (preferred).
    SAGEMAKER_EMBED_ENDPOINT   Endpoint name; URL derived automatically.
    SAGEMAKER_EMBED_REGION     AWS region (default: us-east-1).
    SAGEMAKER_EMBED_BATCH_SIZE Inputs per request (1–64, default 16).
"""
from __future__ import annotations

from codebase_rag.embedder import SageMakerEmbedder, get_sagemaker_embedder

__all__ = ["SageMakerEmbedder", "get_sagemaker_embedder"]
