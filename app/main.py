"""Code Indexer Service — FastAPI application factory.

This module owns the top-level FastAPI app construction for the Code Indexer
Service. It wires routers for health checks, repository indexing, structural
and semantic search, and the context-bundle endpoint used by TheForge's
dev-agent.

Key design decisions:
    * A single ``create_app`` factory is exposed so the service can be
      instantiated under tests with a fresh state and so ASGI servers can
      import ``app`` directly (``app = create_app()`` at module scope).
    * The application lifespan eagerly warms the LadybugDB schema so the first
      ``/index`` call does not pay the migration cost.
    * A generic exception handler converts any uncaught ``Exception`` into a
      structured 500 JSON response so the service never leaks an HTML error
      page.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .config import settings
from .routers import context_bundle, explorer, health, index, search


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan hook — runs startup tasks then yields control.

    Runs the LadybugDB schema migration on process start so that the first
    ``/index`` request does not have to pay for DDL. Schema migration is
    idempotent (``IF NOT EXISTS`` guards), so repeat startups are safe.

    Args:
        app: The FastAPI application being started. Unused, but required by
            the lifespan contract.

    Yields:
        None: Control is yielded to FastAPI once startup finishes.
    """
    # Ensure the parent directory for the DB file exists before migration.
    # Without this, LadybugDB raises "No such file or directory" on a clean
    # install where `.cgr/` has not been created yet.
    from pathlib import Path as _Path

    _Path(settings.LADYBUG_DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    # Warm the LadybugDB schema on startup so the first /index call is faster.
    try:
        from codebase_rag.services.ladybug_schema import migrate

        migrate(settings.LADYBUG_DB_PATH)
    except Exception as exc:
        # Non-fatal — the schema may already exist or the shared package may
        # be unavailable in some deployment contexts. Log and continue so
        # /health still returns ok.
        import logging

        logging.getLogger(__name__).warning(
            "Schema migration warning on startup: %s", exc
        )
    yield


def create_app() -> FastAPI:
    """Construct and return a fully-wired FastAPI application.

    Returns:
        FastAPI: An app with all routers registered and a catch-all exception
        handler installed.
    """
    app = FastAPI(
        title="Code Indexer Service",
        description=(
            "HTTP gateway for code-graph-rag — indexes repositories into LadybugDB "
            "and exposes structural + semantic search to TheForge."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(health.router, tags=["health"])
    app.include_router(index.router, tags=["index"])
    app.include_router(search.router, tags=["search"])
    app.include_router(context_bundle.router, tags=["context"])
    app.include_router(explorer.router, tags=["explorer"])

    @app.exception_handler(Exception)
    async def _generic_error(request, exc):  # type: ignore[override]
        # Catch-all fallback so any unhandled error surfaces as JSON rather
        # than a default HTML error page. Specific handlers/HTTPException
        # cases are still honored by FastAPI's own exception pipeline.
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
        )

    return app


# Module-level app instance used by ASGI servers (e.g. `uvicorn app.main:app`).
app = create_app()
