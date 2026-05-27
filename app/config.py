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

    # --- Persistent job store (Phase 2) ---
    # SQLite file backing the persistent job store. Created on first startup.
    # Use ``:memory:`` in tests via environment override.
    JOBS_DB_PATH: str = ".cgr/jobs.sqlite"
    # LE-143: heartbeat reconciliation of orphaned running jobs
    JOB_HEARTBEAT_INTERVAL_SECONDS: int = 60
    JOB_STALENESS_THRESHOLD_SECONDS: int = 300

    # --- Prometheus metrics (Phase 4) ---
    METRICS_ENABLED: bool = True
    METRICS_PATH: str = "/metrics"
    # Top-level data dir for disk-usage gauges (defaults to .cgr).
    CGR_DATA_DIR: str = ".cgr"

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

    # --- Phase 5: Realtime file-watcher ---
    # Master switch — false by default until validated in production (§12).
    WATCH_ENABLED: bool = False
    # Debounce window in milliseconds.  Env var name ends with _MS (not _S)
    # to match the plan §2 table and the WatchAccepted payload field.
    WATCH_DEBOUNCE_MS: int = 1500
    # Hard timeout (seconds) for joining all Observer threads on shutdown.
    WATCH_SHUTDOWN_TIMEOUT_S: float = 5.0
    # Rows older than this are swept by clear_terminal on startup.
    WATCH_PARTIAL_RETENTION_HOURS: int = 24
    # Maximum concurrent watchers (inotify budget guard).
    WATCH_MAX_REPOS: int = 32

    # --- Context-bundle seed ranking (LE-180) ---
    # Multiplicative penalty applied to a seed candidate's semantic score when
    # its qualified_name / file path looks like a test or script file (e.g.
    # ``.test.``, ``.spec.``, ``scripts.``, ``tests.``). Mirrors TheForge's
    # orchestrator-side ~0.4x test-path multiplier so the implementation
    # symbol leads over scripts that merely *mention* the query terms.
    # Range (0, 1]; values >1 or <=0 fall back to the default. Set to 1.0 to
    # disable the down-weight entirely.
    CONTEXT_BUNDLE_TEST_PATH_PENALTY: float = 0.4

    # --- Rerank pipeline (disabled by default) ---
    # Master control for the two-stage retrieval rerank path.  When False (default),
    # dense vector ranking (e5-base-v2 + DuckDB cosine) is the only ranker.
    # When True, semantic search top-50 are reranked via CodeRankLLM.
    # LM Studio was retired (TheForge PR #168); future implementations will wire
    # LLM-as-reranker via a different backend (e.g. Manifest) — see BUC-1545.
    RERANK_ENABLED: bool = False

    # --- LM Studio adapter (deprecated; LM Studio retired in TheForge PR #168) ---
    # These settings are retained for backward compatibility and reference.
    # LM_STUDIO_URL is no longer probed at startup.  To re-enable rerank in a
    # future release, set RERANK_ENABLED=true and configure a non-LM-Studio
    # backend (TBD; see docs/SEARCH_RANKING.md for options).
    # These are READ DIRECTLY by app.services.lm_studio (which has no
    # pydantic-settings dependency), so duplicating them here is purely
    # for documentation / IDE discoverability.  Keep names in sync with
    # lm_studio._env() default values.
    LM_STUDIO_URL: str = ""                       # e.g. http://localhost:1234 (deprecated)
    LM_STUDIO_EMBED_MODEL: str = "CodeRankEmbed"  # substring hint (deprecated)
    LM_STUDIO_RERANK_MODEL: str = "CodeRankLLM"   # substring hint (deprecated)
    LM_STUDIO_TIMEOUT: float = 30.0

    # --- S3 snapshot / restore (BUC-1499) ---
    # When S3_INDEX_BUCKET is set the service pulls index files from S3 on
    # startup and pushes changed files back on clean shutdown.  This lets
    # containers inherit the last committed graph without a bind-mount.
    # Leave blank in local dev to disable S3 sync entirely.
    S3_INDEX_BUCKET: str = "navistone-forge-data"   # set to "" to disable
    S3_INDEX_PREFIX: str = "code-indexer/indexes"   # key prefix inside the bucket
    S3_INDEX_REGION: str = "us-east-1"

    # --- Pluggable embedder backend (BUC-1605) ---
    # Selects which embedder implementation `app.embedders.get_embedder()`
    # returns at startup. Read directly by `app.embedders` (no pydantic
    # dep); values below exist purely for IDE discoverability + .env
    # documentation. Keep names in sync with `app.embedders.VALID_BACKENDS`.
    #
    #   local      sentence-transformers in-process (no AWS, no sidecar).
    #              Default for standalone installs.
    #   sagemaker  Navistone's AWS SageMaker jina-code-v2-serverless endpoint (was E5, swapped 2026-05-26 LE-129).
    #              Default for the Navistone production deploy.
    #   tei        Hugging Face Text-Embeddings-Inference HTTP sidecar
    #              (http://localhost:8080 by default).
    EMBEDDER_BACKEND: str = "local"

    # --- SageMaker backend config (used when EMBEDDER_BACKEND=sagemaker) ---
    # Priority: SAGEMAKER_ENDPOINT_NAME > SAGEMAKER_EMBED_ENDPOINT > derived from URL.
    # Requires AWS credentials with sagemaker:InvokeEndpoint on the endpoint.
    # Read directly by app.embedders.sagemaker.SageMakerEmbedder.from_env().
    SAGEMAKER_ENDPOINT_NAME: str = ""             # BUC-1605 preferred name (e.g. jina-code-v2-serverless; was forge-e5-embed-v2, swapped 2026-05-26 LE-129)
    SAGEMAKER_EMBED_URL: str = ""                 # legacy alias: https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/forge-e5-embed-v1/invocations
    SAGEMAKER_EMBED_ENDPOINT: str = ""            # legacy alias: forge-e5-embed-v1
    SAGEMAKER_EMBED_REGION: str = "us-east-1"
    SAGEMAKER_EMBED_BATCH_SIZE: int = 32          # 16–64 per Forge contract

    # --- OpenAI backend config (used when EMBEDDER_BACKEND=openai) ---
    # OpenAI's /v1/embeddings — the recommended "bring your own embedder"
    # path. Read directly by app.embedders.openai.OpenAIEmbedder.from_env().
    # The defaults below produce 1536-dim vectors; switch to
    # text-embedding-3-large for 3072-dim. Either way you'll need to
    # re-create the per-repo .duck files because the legacy schema is
    # FLOAT[768] — see docs/EMBEDDERS.md.
    OPENAI_API_KEY: str = ""
    OPENAI_EMBED_MODEL: str = "text-embedding-3-small"
    OPENAI_EMBED_DIM: str = ""                    # blank → use model's native dim
    OPENAI_BASE_URL: str = ""                     # for Azure / vLLM / LiteLLM gateways
    OPENAI_EMBED_BATCH_SIZE: int = 96
    OPENAI_TIMEOUT_S: float = 30.0

    # --- Embed device (used by the codebase_rag local embedder subprocess) ---
    # Set to "cuda" to allow GPU acceleration. Defaults to "cpu" so the embed
    # subprocess does not compete for VRAM on shared AI servers.
    EMBED_DEVICE: str = "cpu"

    # --- TEI backend config (used when EMBEDDER_BACKEND=tei) ---
    # Hugging Face Text-Embeddings-Inference HTTP sidecar. Bring up via:
    #   docker run -d -p 8080:80 --gpus all \
    #     ghcr.io/huggingface/text-embeddings-inference:1.5 \
    #     --model-id intfloat/e5-base-v2
    # Read directly by app.embedders.tei.TEIEmbedder.from_env().
    TEI_URL: str = "http://localhost:8080"
    TEI_TIMEOUT_MS: int = 30000
    TEI_BATCH_SIZE: int = 32

    # --- BUC-1598: cross-repo IMPORTS resolution ---
    # Master switch for the cross-repo IMPORTS resolution pass that runs
    # after every successful /index job (and on-demand via
    # POST /admin/resolve-cross-repo-imports).  When False (default), the
    # pass is a no-op — external Module nodes stay as leaves, exactly
    # mirroring pre-BUC-1598 behaviour.  When True, external Modules that
    # match another indexed repo's package.json / pyproject.toml identity
    # are rewired to the canonical ``{target_slug}::{qname}`` form.
    #
    # Read directly by app.services.cross_repo_imports.is_enabled() at
    # call time (not import time) so tests can toggle via monkey-patching
    # os.environ without reloading the module.  The duplicated declaration
    # here is purely for IDE discoverability + .env.template documentation.
    CROSS_REPO_IMPORTS_ENABLED: bool = False

    # --- Phase 1.3: code-specific embedding A/B path ---
    # Default 'e5-base-v2' preserves the pre-Phase-1.3 behaviour exactly.
    # Set to 'bge-code-v1' to write to the parallel embedding_v2 column and
    # have search read from it (with graceful fallback to embedding when v2
    # is NULL during partial migration).  See app/services/embedder.py.
    # Read directly by app.services.embedder (no pydantic dep) — the values
    # below are documentation / IDE discoverability only.
    EMBEDDING_MODEL_ACTIVE: str = "e5-base-v2"     # 'e5-base-v2' | 'bge-code-v1'
    SAGEMAKER_BGE_CODE_URL: str = ""               # full invocation URL (preferred)
    SAGEMAKER_BGE_CODE_ENDPOINT: str = ""          # fallback if URL not set
    SAGEMAKER_BGE_CODE_REGION: str = "us-east-1"
    SAGEMAKER_BGE_CODE_BATCH_SIZE: int = 16


# Module-level singleton — import this rather than re-instantiating Settings.
settings = Settings()
