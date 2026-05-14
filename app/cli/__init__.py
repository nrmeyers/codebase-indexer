"""Standalone command-line interface for the Code Indexer Service.

Exposes a ``code-indexer`` Typer app that wraps the FastAPI HTTP surface
so the service can be used as an independent developer tool without
requiring TheForge or any other orchestrator.

Public entry point: :data:`app.cli.main.app`.
"""

from .main import app

__all__ = ["app"]
