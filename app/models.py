"""Pydantic request/response models for the Code Indexer Service."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    db_path: str
    indexed_repos: list[str]


# ---------------------------------------------------------------------------
# /index
# ---------------------------------------------------------------------------


class IndexRequest(BaseModel):
    repo_path: str = Field(
        description="Absolute or relative path to the repository to index."
    )
    force_reindex: bool = Field(
        default=False,
        description="When true, clean the graph before re-indexing.",
    )


class IndexAccepted(BaseModel):
    job_id: str
    message: str = "Indexing job accepted"


class IndexStatus(BaseModel):
    job_id: str
    status: Literal["running", "done", "failed"]
    progress_pct: float = Field(ge=0.0, le=100.0)
    node_count: int = 0
    rel_count: int = 0
    error: str | None = None
