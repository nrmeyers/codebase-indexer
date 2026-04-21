"""Pydantic request/response models for the Code Indexer Service.

Each section corresponds to a single endpoint group (health, index, search).
Models are deliberately narrow — payloads shared with TheForge's TypeScript
client (``code-indexer-client.ts``) must stay byte-compatible, so field names
use snake_case to match JSON wire format and are not renamed lightly.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response for ``GET /health``.

    Attributes:
        status: ``ok`` when the service can reach LadybugDB, ``degraded``
            otherwise. Callers use this for readiness probes.
        db_path: Resolved LadybugDB path (useful for multi-env debugging).
        indexed_repos: Deduplicated list of project names currently present
            in the graph.
    """

    status: Literal["ok", "degraded"]
    db_path: str
    indexed_repos: list[str]


# ---------------------------------------------------------------------------
# /index
# ---------------------------------------------------------------------------


class IndexRequest(BaseModel):
    """Request body for ``POST /index``."""

    repo_path: str = Field(
        description="Absolute or relative path to the repository to index."
    )
    force_reindex: bool = Field(
        default=False,
        description="When true, clean the graph before re-indexing.",
    )


class IndexAccepted(BaseModel):
    """202 response from ``POST /index`` — hand-off identifier for polling."""

    job_id: str
    message: str = "Indexing job accepted"


class IndexStatus(BaseModel):
    """Response for ``GET /index/{job_id}/status``.

    Attributes:
        job_id: Identifier returned from ``POST /index``.
        status: Current execution state.
        progress_pct: Bounded to [0, 100]; progress is best-effort and jumps
            at milestones rather than tracking every file.
        node_count: Final count once the job completes; 0 while running.
        rel_count: Final count once the job completes; 0 while running.
        error: Populated only on ``failed`` status.
    """

    job_id: str
    status: Literal["running", "done", "failed"]
    progress_pct: float = Field(ge=0.0, le=100.0)
    node_count: int = 0
    rel_count: int = 0
    error: str | None = None


# ---------------------------------------------------------------------------
# /search/structural
# ---------------------------------------------------------------------------


class StructuralSearchResponse(BaseModel):
    """Response for ``GET /search/structural``.

    Nodes and relationships are separated to make it easy for clients to
    render graph views without extra parsing. When a query returns only
    scalar columns, those appear in ``nodes`` as plain dicts.
    """

    nodes: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    row_count: int


# ---------------------------------------------------------------------------
# /search/semantic
# ---------------------------------------------------------------------------


class SemanticResult(BaseModel):
    """Single row in a semantic search result set."""

    symbol: str
    score: float
    type: str = ""


class SemanticSearchResponse(BaseModel):
    """Top-k semantic search results, ordered by score descending."""

    results: list[SemanticResult]


# ---------------------------------------------------------------------------
# /search/symbol
# ---------------------------------------------------------------------------


class SymbolResponse(BaseModel):
    """Response for ``GET /search/symbol`` — qualified-name → source.

    Attributes:
        qualified_name: The fully-qualified name requested.
        file: Absolute file path the symbol was defined in.
        line_start: 1-indexed start line; ``None`` when unknown.
        line_end: 1-indexed inclusive end line; ``None`` when unknown.
        source: The exact source snippet between start/end lines. Empty
            string when the file cannot be read.
    """

    qualified_name: str
    file: str
    line_start: int | None
    line_end: int | None
    source: str
