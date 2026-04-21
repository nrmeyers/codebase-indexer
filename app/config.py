"""Configuration for the Code Indexer Service.

Reads from environment variables (or a ``.env`` file at the working directory).
All settings mirror the code-graph-rag settings where applicable so that a
single ``.env`` file can drive both services without drift.

Key design decisions:
    * Uses ``pydantic-settings`` so env var names map 1:1 to attributes and
      type coercion is automatic.
    * ``extra="ignore"`` lets this module coexist with a shared ``.env`` that
      defines additional variables for code-graph-rag.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment / ``.env``.

    Attributes:
        LADYBUG_DB_PATH: Filesystem path to the shared LadybugDB file. Must
            point to the same file as code-graph-rag for indexed data to be
            visible across services.
        LADYBUG_BATCH_SIZE: Batch size used by the underlying ingestor when
            flushing nodes/relationships. Larger batches = fewer round-trips
            but higher peak memory.
        TARGET_REPO_PATH: Default repository to index when a request does not
            specify ``repo_path``.
        HOST: Bind host for the HTTP server.
        PORT: Bind port for the HTTP server.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LadybugDB (shared with code-graph-rag) ---
    LADYBUG_DB_PATH: str = ".cgr/graph.db"
    LADYBUG_BATCH_SIZE: int = 1000

    # --- Default repo to index when none is provided ---
    TARGET_REPO_PATH: str = "."

    # --- Server ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000


# Module-level singleton — import this rather than re-instantiating Settings.
settings = Settings()
