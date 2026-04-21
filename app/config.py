"""Configuration for the Code Indexer Service.

Reads from environment variables (or a .env file at the working directory).
All settings mirror the code-graph-rag settings where applicable so that
a single .env file can drive both services.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LadybugDB (shared with code-graph-rag) ---
    LADYBUG_DB_PATH: str = ".cgr/graph.db"
    LADYBUG_BATCH_SIZE: int = 1000

    # --- Default repo to index when none is provided ---
    TARGET_REPO_PATH: str = "."

    # --- Server ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000


settings = Settings()
