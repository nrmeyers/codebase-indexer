"""Code Indexer Service — FastAPI application factory."""
from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .config import settings
from .routers import health, index


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan — run startup checks, then yield."""
    # Warm the LadybugDB schema on startup so the first /index call is faster.
    try:
        from codebase_rag.services.ladybug_schema import migrate

        migrate(settings.LADYBUG_DB_PATH)
    except Exception as exc:
        # Non-fatal — service still works; schema may already exist.
        import logging

        logging.getLogger(__name__).warning(
            "Schema migration warning on startup: %s", exc
        )
    yield


def create_app() -> FastAPI:
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

    @app.exception_handler(Exception)
    async def _generic_error(request, exc):  # type: ignore[override]
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
        )

    return app


app = create_app()
