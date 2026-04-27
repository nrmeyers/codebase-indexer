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

import re
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def slugify_repo(name: str) -> str:
    """Normalise a repo name for use as a filename.

    Replaces anything that's not alphanumeric/dash/underscore with ``_`` so
    the result is safe to embed in a filesystem path. Collapses runs of
    underscores and strips leading/trailing separators.

    Args:
        name: Raw repo name (typically ``Path(repo_path).name``).

    Returns:
        str: A filesystem-safe slug. Never empty; falls back to ``repo``.
    """
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return s or "repo"


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
    # Per-repo DB files live under ``LADYBUG_DB_DIR`` as ``{slug}.db``.  Each
    # indexed repository gets its own isolated graph so the LadybugDB Explorer
    # can open one index at a time, WAL corruption in one doesn't blast others,
    # and re-indexing is a simple unlink.  ``LADYBUG_DB_PATH`` is retained as
    # a fallback / legacy pointer for code paths that don't yet know a repo.
    LADYBUG_DB_DIR: str = ".cgr/repos"
    LADYBUG_DB_PATH: str = ".cgr/graph.db"
    LADYBUG_BATCH_SIZE: int = 1000

    def db_path_for_repo(self, repo_name: str) -> str:
        """Return the per-repo LadybugDB file path for ``repo_name``.

        Args:
            repo_name: Human-readable repo name — usually ``Path(repo_path).name``.

        Returns:
            str: Absolute or relative path to the repo's ``.db`` file.  Does
            not create the file; the ingestor (or migrate) does that.
        """
        return str(Path(self.LADYBUG_DB_DIR) / f"{slugify_repo(repo_name)}.db")

    def vec_db_path_for_repo(self, repo_name: str) -> str:
        """Return the per-repo DuckDB vector-store file path for ``repo_name``.

        The vector store lives next to the LadybugDB file with a ``.duck``
        suffix (v5.3 §6.5), holding ``FLOAT[768]`` embeddings, per-repo
        metadata, and PageRank centrality scores.

        Args:
            repo_name: Human-readable repo name — usually ``Path(repo_path).name``.

        Returns:
            str: Absolute or relative path to the repo's ``.duck`` file.
        """
        return str(Path(self.LADYBUG_DB_DIR) / f"{slugify_repo(repo_name)}.duck")

    # --- Default repo to index when none is provided ---
    TARGET_REPO_PATH: str = "."

    # --- Server ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # --- GitHub integration ---
    # Personal access token used by the /github/* routes to list the
    # authenticated user's repos and clone private repos.  Read from
    # ``GITHUB_TOKEN`` first (TheForge convention), ``GH_TOKEN`` second
    # (``gh`` CLI convention).  Empty string = unauthenticated mode.
    GITHUB_TOKEN: str = ""
    GH_TOKEN: str = ""

    # Comma-separated list of owners (users or orgs) that are permitted to
    # be cloned and indexed.  Empty list disables the guard (all owners
    # allowed) — set this to e.g. ``"navistone"`` in production to make
    # sure the indexer can never be tricked into cloning random public
    # repos.  Owners are compared case-insensitively.
    GITHUB_ALLOWED_OWNERS: str = "navistone"

    @property
    def github_allowed_owners(self) -> list[str]:
        """Parsed allowlist as a lower-cased list. Empty = no restriction."""
        raw = (self.GITHUB_ALLOWED_OWNERS or "").strip()
        if not raw:
            return []
        return [o.strip().lower() for o in raw.split(",") if o.strip()]


# Module-level singleton — import this rather than re-instantiating Settings.
settings = Settings()
