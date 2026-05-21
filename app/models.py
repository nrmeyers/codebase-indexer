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


class RepoHealth(BaseModel):
    """Per-repo health probe entry in ``GET /health``.

    Attributes:
        name: Project slug (matches ``indexed_repos``).
        db_path: Filesystem path to the per-repo LadybugDB file.
        size_bytes: Current file size; 0 when the DB has not been written.
        node_count: Total node count across all node tables; None when the
            probe could not open the DB (corrupted WAL, missing file).
        readable: True iff the DB opened cleanly. A False here is a strong
            signal that a restart self-heal cycle is needed.
        last_indexed_at: Unix timestamp of the last successful index job for
            this repo; None when the repo has never been indexed in this
            service instance.  Persisted on the Project node so it survives
            restarts.
        indexing: True when a job is currently writing to this repo — UIs
            should disable the re-index button and show a spinner.
    """

    name: str
    db_path: str
    size_bytes: int
    node_count: int | None
    readable: bool
    last_indexed_at: float | None = None
    indexing: bool = False
    repo_path: str | None = Field(
        default=None,
        description=(
            "Absolute filesystem path to the source repo — consumers like the "
            "orchestrator need this to call endpoints (e.g. /context-bundle) "
            "that validate repo_path on disk. Null when the path wasn't captured "
            "(e.g. DB indexed before this field existed — triggers a re-index to fix)."
        ),
    )


class LmStudioStatus(BaseModel):
    """LM Studio backend status block surfaced in ``GET /health``.

    Lets TheForge render a backend-availability badge without making a
    separate probe call. All boolean fields default to ``False`` and model
    fields to ``None`` so a "not configured" or "unreachable" payload is
    unambiguous to the client.

    Attributes:
        configured: True when ``LM_STUDIO_URL`` is set (non-empty
            ``base_url()``). When False, every other field is False/None
            and the handler short-circuits without making a network call.
        reachable: True when LM Studio responded to /v1/models with at
            least one loaded model. Cached for 30s by the adapter.
        embed_model: Resolved loaded-model id matching
            ``LM_STUDIO_EMBED_MODEL``; None when no match is found.
        rerank_model: Resolved loaded-model id matching
            ``LM_STUDIO_RERANK_MODEL``; None when no match is found.
        can_embed: True when the named embed model is actually loaded —
            stricter than ``reachable``.
        can_rerank: True when the named rerank model is actually loaded —
            stricter than ``reachable``.
    """

    configured: bool = False
    reachable: bool = False
    embed_model: str | None = None
    rerank_model: str | None = None
    can_embed: bool = False
    can_rerank: bool = False


class S3SyncStatus(BaseModel):
    """Snapshot of where the S3-backed index store stands.

    Surfaces in /health so operators (and the frontend) can tell at a glance
    whether the local indexes are backed by S3 and when they were last synced.
    """

    enabled: bool = False
    bucket: str | None = None
    prefix: str | None = None
    region: str | None = None
    # Last time the indexer pushed indexes to S3 in this process.  None until
    # the first successful snapshot happens.  Unix timestamp.
    last_snapshot_at: float | None = None
    # Number of files pushed in the last snapshot.  0 means "no changes
    # since last sync"; absent means we haven't snapshotted yet.
    last_snapshot_count: int | None = None
    # Error message from the most recent snapshot attempt, if any.  None when
    # the last snapshot succeeded or hasn't run yet.
    last_error: str | None = None


class EmbedderStatus(BaseModel):
    """Active embedder backend description.

    Surfaced by /health so TheForge (and any external caller) can verify the
    index/embedder pair before issuing a search. A dim mismatch between this
    block and the DuckDB schema means the index needs to be rebuilt before
    vectors line up.

    Attributes:
        backend: Selected backend name (``local`` | ``sagemaker`` | ``tei``
            | ``openai``).
        model: Underlying model identifier the backend talks to.
        dim: Output vector dimensionality. MUST match the per-repo DuckDB
            ``FLOAT[dim]`` schema. Default 768 (e5-base-v2).
        configured: True when ``get_embedder()`` returned a backend; False
            when construction failed (e.g. ``EMBEDDER_BACKEND=openai`` with
            no API key set). ``error`` carries the message in that case.
        error: Construction error message when ``configured`` is False.
        available: True iff the startup probe both constructed the backend
            AND verified its heavy dependency (e.g. ``sentence_transformers``
            for ``local``, ``boto3`` + endpoint env for ``sagemaker``). A
            ``configured=True`` / ``available=False`` split means the
            factory returned a backend object but ``embed()`` would still
            fail at the first call — exactly the silent-503 mode this
            field exists to surface.
        last_error: Most recent probe error message, walked through
            ``__cause__`` so a missing optional dep surfaces as the
            original ``ModuleNotFoundError`` rather than the wrapper.
            ``None`` on a healthy probe. Distinct from ``error``: ``error``
            is the legacy construction-only failure string (kept for
            backward compat with PR #69 callers); ``last_error`` is the
            full probe outcome including dep-validation failures.
        fallback_lm_studio: True when LM Studio is configured AND has the
            named embed model loaded. Informational — does NOT flip
            ``available`` to True, because the primary backend is still
            the authoritative path for the index pipeline.
        last_check_at: ISO 8601 UTC timestamp of the most recent probe.
            ``None`` until the lifespan startup probe completes.
        check_latency_ms: Wall-clock milliseconds the probe took. Useful
            for alerting if construction starts dragging (e.g. SageMaker
            cold start exceeds an SLO).
    """

    backend: str = "unknown"
    model: str = ""
    dim: int = 0
    configured: bool = False
    error: str | None = None
    # Loud-availability extensions (this PR). All additive — existing
    # callers reading ``configured`` / ``error`` keep working.
    available: bool = False
    last_error: str | None = None
    fallback_lm_studio: bool = False
    last_check_at: str | None = None
    check_latency_ms: float | None = None


class HealthResponse(BaseModel):
    """Response for ``GET /health``.

    Attributes:
        status: ``ok`` when every per-repo DB is readable, ``degraded`` when
            one or more repos are unreadable. Callers use this for readiness
            probes and to decide whether to prompt the user to re-index.
        db_path: Resolved LadybugDB directory (useful for multi-env debugging).
        indexed_repos: Deduplicated list of project names currently present
            on disk.
        repos: Detailed per-repo probe results (size, node count, readability).
        running_jobs: Count of currently-running index jobs across all repos.
        lm_studio: LM Studio backend status block — always present so the
            UI can render a backend badge without a separate probe call.
    """

    status: Literal["ok", "degraded"]
    db_path: str
    indexed_repos: list[str]
    repos: list[RepoHealth] = []
    running_jobs: int = 0
    lm_studio: "LmStudioStatus" = Field(default_factory=lambda: LmStudioStatus())
    # S3 sync status: shows whether per-repo .db / .duck files are backed
    # up to S3, and when the most recent push happened.  Lets the frontend
    # display a "synced" / "local-only" badge per indexer instance.
    s3_sync: "S3SyncStatus" = Field(default_factory=lambda: S3SyncStatus())
    # BYO embedder visibility (BYO-embedder config pass): which backend is
    # actively configured, which model it talks to, and what dim it
    # produces. Lets TheForge verify the index/embedder pair match before
    # searching.
    embedder: "EmbedderStatus" = Field(default_factory=lambda: EmbedderStatus())


# ---------------------------------------------------------------------------
# /index
# ---------------------------------------------------------------------------


class IndexRequest(BaseModel):
    """Request body for ``POST /index``.

    Two modes:

    1. **Local path mode** (backward compatible) — supply ``repo_path`` only.
       The service indexes the directory in place. Auth for any out-of-band
       git operations comes from the local git config (SSH key, PAT, etc).

    2. **App-authenticated clone mode** (BUC-1561b) — supply ``github_token``
       and ``full_name`` (and optionally ``branch``). The service clones the
       remote repo using the token-bearing URL
       ``https://x-access-token:{token}@github.com/{owner}/{repo}.git`` and
       then indexes the freshly-cloned working tree. ``repo_path`` is
       ignored / optional in this mode — the destination is derived from
       ``full_name``.

       The token is treated as an ephemeral secret: it is **never** logged,
       **never** persisted in the jobs_store payload, and is scrubbed (``***``)
       from any error messages echoed back to the caller. GitHub App
       installation tokens are valid for ~1 hour, so the clone must complete
       within that window — large repos that exceed the window will fail
       with a 502 ``git failed`` and the caller must request a fresh token
       and retry.
    """

    repo_path: str = Field(
        default="",
        description=(
            "Absolute or relative path to the repository to index. Required "
            "in local-path mode; ignored when ``github_token`` + ``full_name`` "
            "are supplied (the cloned path is used instead)."
        ),
    )
    force_reindex: bool = Field(
        default=False,
        description="When true, clean the graph before re-indexing.",
    )
    exclude_paths: list[str] = Field(
        default_factory=list,
        description=(
            "Repo-relative path prefixes to skip during indexing (e.g. 'tests/fixtures'). "
            "Defaults to an empty list; common fixture directories are excluded automatically."
        ),
    )
    github_token: str | None = Field(
        default=None,
        description=(
            "Optional GitHub App installation token (or PAT) used to clone "
            "private repos without relying on local git config / SSH keys. "
            "When set, ``full_name`` is required. Treated as an ephemeral "
            "secret — never logged, never persisted; only the masked form "
            "``***`` appears in any persisted state."
        ),
    )
    full_name: str | None = Field(
        default=None,
        description=(
            "``owner/repo`` identifier — required when ``github_token`` is set. "
            "Used to build the clone URL."
        ),
    )
    branch: str | None = Field(
        default=None,
        description=(
            "Branch to check out when cloning. Defaults to the repo's HEAD "
            "(typically the default branch). Only used in clone mode."
        ),
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
        phase: Current work phase — monotonically advances through the
            pipeline. ``queued`` while the job waits for a repo lock;
            ``discovering`` during filesystem walk; ``parsing`` during
            tree-sitter pass; ``writing`` during LadybugDB flush;
            ``embedding`` during the CodeRankEmbed model pass; ``finalizing``
            for final metadata writes; ``done`` on success.
            Set to ``"cancelled"`` when a cancel request is honoured.
        progress_pct: Bounded to [0, 100]; monotonically non-decreasing.
            Computed from phase + per-file counters so the bar moves
            smoothly rather than jumping at milestones.
        files_total: Total eligible files discovered in the repo (available
            after the discovering phase; 0 before).
        files_done: Files fully parsed so far (advances during parsing).
        current_file: Relative path of the file being parsed right now;
            None outside the parsing phase or when embedding.
        node_count: Live graph node count during the run; final value on
            completion.
        rel_count: Live relationship count; final on completion.
        started_at: Unix epoch seconds when the job was accepted.
        elapsed_sec: Wall-clock seconds since the job started (computed
            at response time, not stored).
        eta_sec: Estimated seconds remaining; None until progress_pct > 10
            (too early for a reliable estimate).
        error: Populated only on ``failed`` status.
    """

    job_id: str
    status: Literal["pending", "running", "done", "failed", "interrupted"]
    phase: Literal[
        "queued", "discovering", "parsing", "writing",
        "embedding", "finalizing", "done", "cancelled",
    ] = "queued"
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    files_total: int = 0
    files_done: int = 0
    current_file: str | None = None
    node_count: int = 0
    rel_count: int = 0
    embedding_count: int = 0
    # BUC-1518: live embed-pass breakdown.  Populated from PROGRESS lines
    # the embed driver writes to its log file, so the frontend can show
    # how the work is being divided across actual SageMaker calls vs
    # cache skips vs filtered-out files.
    embeddings_skipped_unchanged: int = 0
    embeddings_filtered_out: int = 0
    # BUC-1601 (Fix A) — read-failure count from the embed subprocess.
    # 0 on healthy runs; non-zero means files referenced by the graph
    # were unreadable at embed time (each emitted a WARN line).
    embeddings_dropped_unreadable: int = 0
    started_at: float = 0.0
    elapsed_sec: float = 0.0
    eta_sec: float | None = None
    error: str | None = None


class DiffMetrics(BaseModel):
    """Response for ``GET /index/{job_id}/diff_metrics``.

    BUC-1574 (Phase 1.4) — surfaces the incremental-embed audit shape
    for a single index run.  Lets the operator confirm that re-indexing
    an unchanged repo produces a high ``hash_match_rate`` (i.e. the
    content-hash skip is firing as designed) without grepping the
    embed subprocess log.

    Attributes:
        total_symbols: ``embedded + skipped_unchanged + skipped_filtered``.
            For running jobs this is the running total at the latest
            ``PROGRESS`` line.
        embedded: Symbols actually sent to SageMaker on this run.
        skipped_unchanged: Symbols where the stored ``content_hash``
            matched the freshly-computed hash and we skipped the embed
            call entirely.
        skipped_filtered: Symbols whose source file matched a skip
            pattern (tests / generated / vendored).
        dropped_unreadable: BUC-1601 — symbols whose source file could
            not be read off disk at embed time (file deleted out from
            under the indexer, permission denied, etc.).  Each drop also
            emitted a WARN line to the per-job embed log.  Should be 0
            on a clean run.
        hash_match_rate: ``skipped_unchanged / (embedded +
            skipped_unchanged)`` — the fraction of in-scope symbols that
            were unchanged since the previous run.  Returns 0.0 when the
            denominator is zero (very small repos or first-ever index).
        wall_clock_seconds: Wall-clock seconds spent in the embed phase.
            For running jobs this is ``now - embed_started_at``; for
            completed jobs it is ``embed_finished_at - embed_started_at``.
    """

    total_symbols: int
    embedded: int
    skipped_unchanged: int
    skipped_filtered: int
    dropped_unreadable: int = 0
    hash_match_rate: float
    wall_clock_seconds: float


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
    """Top-k semantic search results, ordered by score descending.

    Attributes:
        results: Ranked semantic matches (descending score).
        search_intent: Internal routing label describing how the query
            was interpreted by the retrieval pipeline (e.g. ``"fqn"``
            when a bare qualified-name was detected and exact/prefix
            matches were pinned, or ``"semantic"`` for the default
            natural-language path). ``None`` when intent classification
            was not performed. Surfaced for observability — callers
            should not branch on it.
    """

    results: list[SemanticResult]
    search_intent: str | None = None


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
    docstring: str | None = None


# ---------------------------------------------------------------------------
# /search/files
# ---------------------------------------------------------------------------


class FileEntry(BaseModel):
    """Single row in ``GET /search/files``."""

    path: str
    name: str
    extension: str = ""


class FileListResponse(BaseModel):
    """Response for ``GET /search/files`` — paginated file listing."""

    files: list[FileEntry]
    total: int


# ---------------------------------------------------------------------------
# /search/lexical (Tantivy BM25 — Phase 1.1)
# ---------------------------------------------------------------------------


class LexicalHit(BaseModel):
    """One hit from ``GET /search/lexical`` — Tantivy BM25 ranked match."""

    symbol_qname: str
    file_path: str
    symbol_kind: str
    score: float
    start_line: int = 0
    end_line: int = 0


class LexicalSearchResponse(BaseModel):
    """Response for ``GET /search/lexical`` — top-k lexical hits."""

    results: list[LexicalHit]


# ---------------------------------------------------------------------------
# /search/types
# ---------------------------------------------------------------------------


class NodeTypeStat(BaseModel):
    """One row of ``GET /search/types`` — node label + count."""

    label: str
    count: int


class NodeTypesResponse(BaseModel):
    """Response for ``GET /search/types`` — lets UIs discover Browse tabs."""

    types: list[NodeTypeStat]


# ---------------------------------------------------------------------------
# /stats/{repo}
# ---------------------------------------------------------------------------


class RepoStatsResponse(BaseModel):
    """Response for ``GET /stats/{repo}`` — per-repo graph breakdown.

    Attributes:
        repo: Project slug the stats belong to.
        node_count: Total nodes across every node label.
        rel_count: Total relationships across every rel type.
        node_breakdown: Per-label node counts.
        rel_breakdown: Per-type relationship counts.
        db_size_bytes: DB file size on disk; 0 when the file is missing.
        last_modified: Unix timestamp of the DB file's mtime; None when
            the file is missing.
        last_indexed_at: Unix timestamp of the last successful index job.
            Persisted on the Project node via Cypher ``SET`` so it survives
            service restarts — more authoritative than ``last_modified``
            which can drift on any write.
        root_path: Absolute filesystem path the repo was indexed from.
            Empty when unknown (legacy DB indexed before this field).
        has_embeddings: True when the numpy embedding file exists and is
            non-empty, indicating semantic search is available.
        embedding_count: Number of vectors in the embedding store; None
            when embeddings are absent or the count can't be determined.
        indexing: True when a job is currently writing to this repo.
    """

    repo: str
    node_count: int
    rel_count: int
    node_breakdown: list[NodeTypeStat]
    rel_breakdown: list[NodeTypeStat]
    db_size_bytes: int
    last_modified: float | None
    last_indexed_at: float | None = None
    root_path: str = ""
    has_embeddings: bool
    embedding_count: int | None = None
    indexing: bool = False


# ---------------------------------------------------------------------------
# /repos/{name}/stats — frontend-shape index facts (BACKEND_HANDOVER §2.1)
# ---------------------------------------------------------------------------


class RepoIndexStatsResponse(BaseModel):
    """Frontend-facing per-repo index stats.

    Shape mirrors the ``RepoIndexStats`` interface in TheForge's
    ``web/src/components/code-indexer/types.ts`` so the IndexRunDashboard
    sidebar can render directly off this payload. All fields are nullable
    by design — the dashboard renders ``—`` placeholders when a field is
    missing (e.g. embedder hasn't run yet → ``fragment_count: null``).

    Attributes:
        db_size_bytes: LadybugDB ``.db`` file size; null when missing.
        duck_size_bytes: DuckDB ``.duck`` vector store size; null when
            missing or pre-CodeRankEmbed (vector store not yet populated).
        last_indexed_at: ISO 8601 UTC timestamp of the last successful
            index. Returned as a string (not unix epoch) per FE contract.
        indexed_commit_sha: Git commit SHA that was indexed; null when the
            repo wasn't a clean git checkout at index time.
        fragment_count: Number of embedding rows in the ``.duck`` file —
            "fragments" in the FE copy.
        edge_count: Total relationship count across all rel types.
        node_count_by_label: Per-label node count breakdown for the
            sidebar facts list.
    """

    db_size_bytes: int | None = None
    duck_size_bytes: int | None = None
    last_indexed_at: str | None = None
    indexed_commit_sha: str | None = None
    fragment_count: int | None = None
    edge_count: int | None = None
    node_count_by_label: dict[str, int] = Field(default_factory=dict)


class ReindexRequest(BaseModel):
    """Request body for ``POST /repos/{name}/reindex`` — force re-index."""

    force: bool = Field(
        default=True,
        description=(
            "When true (default), wipes both the LadybugDB ``.db`` and the "
            "DuckDB ``.duck`` files before kicking off all 4 indexing passes. "
            "Currently the only supported mode."
        ),
    )


class ReindexAccepted(BaseModel):
    """202 response from ``POST /repos/{name}/reindex``."""

    job_id: str


# ---------------------------------------------------------------------------
# /disk-usage — capacity gauge (BACKEND_HANDOVER §2.11)
# ---------------------------------------------------------------------------


class DiskUsageResponse(BaseModel):
    """Disk usage under ``LADYBUG_DB_DIR``.

    Frontend computes ``total = used + free``, percentage, and color-codes
    the gauge (≤75% green, 75-90% amber, >90% red).

    Attributes:
        used_bytes: Bytes occupied by ``.cgr/repos/`` and its children.
        free_bytes: Free bytes available on the filesystem hosting that
            directory (``shutil.disk_usage`` ``free`` field).
    """

    used_bytes: int
    free_bytes: int


# ---------------------------------------------------------------------------
# /search/centrality — PageRank top-N (BACKEND_HANDOVER §2.8)
# ---------------------------------------------------------------------------


class CentralityResult(BaseModel):
    """Single row in a centrality result set."""

    qualified_name: str
    pagerank: float
    file_path: str = ""
    line_range: tuple[int, int] = (0, 0)


class CentralityResponse(BaseModel):
    """Top-N most-central symbols ordered by PageRank descending."""

    results: list[CentralityResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# /symbols/{fqn}/callers + /symbols/{fqn}/callees (BACKEND_HANDOVER §2.9)
# ---------------------------------------------------------------------------


class CallSiteResult(BaseModel):
    """One caller (or callee) entry for the symbol detail panel."""

    qualified_name: str
    file_path: str = ""
    line_number: int = 0


class CallSiteResponse(BaseModel):
    """Response wrapper for /callers and /callees."""

    results: list[CallSiteResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# /index/{repo} — admin
# ---------------------------------------------------------------------------


class DeleteIndexResponse(BaseModel):
    """Response for ``DELETE /index/{repo}`` — admin wipe."""

    repo: str
    removed_files: list[str]
    ok: bool
    cleanup: dict[str, str] = {}  # Resource type -> status message


# ---------------------------------------------------------------------------
# /index/jobs — job history management
# ---------------------------------------------------------------------------


class JobSummary(BaseModel):
    """Compact representation of a job record for list endpoints."""

    job_id: str
    repo_path: str
    repo_name: str
    status: Literal["running", "done", "failed", "interrupted", "cancelled", "queued"]
    progress_pct: float
    phase: str
    node_count: int
    rel_count: int
    error: str | None
    started_at: float
    finished_at: float | None


class JobListResponse(BaseModel):
    """Response for ``GET /index/jobs`` — newest-first history."""

    jobs: list[JobSummary]
    total: int
    running: int


class JobClearResponse(BaseModel):
    """Response for ``POST /index/jobs/clear`` and ``DELETE /index/jobs/{id}``."""

    cleared: int
    remaining: int


# ---------------------------------------------------------------------------
# /github/status
# ---------------------------------------------------------------------------


class GitHubRateLimit(BaseModel):
    """Core GitHub REST API rate-limit snapshot."""

    limit: int
    remaining: int
    reset_at: float | None  # unix ts when the window rolls over


class GitHubStatusResponse(BaseModel):
    """Response for ``GET /github/status`` — connection readiness probe.

    Attributes:
        connected: True iff a token is present AND GitHub accepted it.
        token_source: Where the token came from (``settings``, ``env``,
            or ``none``).  Helps developers diagnose env-var vs .env issues.
        token_type: Detected GitHub credential family — ``pat`` (classic or
            fine-grained PAT), ``github_app`` (App installation token,
            ``ghs_*``), ``unknown`` (token present but unrecognised prefix),
            or ``none`` (no token configured).  Drives endpoint routing
            (``/user/repos`` vs ``/installation/repositories``) and the
            recovery hint shown to operators on 401.
        user: Authenticated GitHub login, or None when unauthenticated.
        scopes: OAuth scopes the token carries (best-effort — GitHub
            exposes these on the ``X-OAuth-Scopes`` header).
        rate_limit: Core API rate-limit snapshot.
        message: Human-readable status line the UI can show directly.
    """

    connected: bool
    token_source: Literal["settings", "env", "none"]
    token_type: Literal["pat", "github_app", "unknown", "none"] = "none"
    user: str | None
    scopes: list[str]
    rate_limit: GitHubRateLimit | None
    message: str


# ---------------------------------------------------------------------------
# /explorer/info
# ---------------------------------------------------------------------------


class ExplorerInfoResponse(BaseModel):
    """Response for ``GET /explorer/info`` — graph viewer availability.

    Callers (TheForge UI, developer CLIs) poll this endpoint to decide
    whether to surface a "Visualise graph" button.  The endpoint itself
    never launches a viewer — it only reports what is possible and returns
    the shell command the caller can execute locally.

    Attributes:
        available: True iff the LadybugDB file exists **and** contains at
            least one indexed project.  False means the viewer would open
            on an empty graph, so the UI should hide/disable the button.
        db_path: Resolved LadybugDB path (same as ``/health.db_path``).
        indexed_repos: Project names that would be visible in the viewer.
        launch_command: Ready-to-paste shell command that spins up the
            official ``kuzudb/explorer`` Docker container pointed at the
            current DB file.  Docker is **only** required for visualisation;
            all structural and semantic search still work without it.
        viewer_url: HTTP URL to open once the launch command is running.
        docs_url: Upstream kuzu-explorer documentation for the UI itself.
    """

    available: bool
    db_path: str
    indexed_repos: list[str]
    launch_command: str
    viewer_url: str
    docs_url: str



# ---------------------------------------------------------------------------
# Graph overview
# ---------------------------------------------------------------------------


class GraphNode(BaseModel):
    """A node in the repo-wide graph overview.

    Attributes:
        id: Stable identifier derived from qname or path+name.
        label: Node type label (Function, Class, File, etc.).
        name: Human-readable short name.
        qname: Fully-qualified name, if the node has one.
        path: Source file path, if applicable.
    """

    id: str
    label: str
    name: str
    qname: str | None = None
    path: str | None = None


class GraphEdge(BaseModel):
    """A directed edge in the repo-wide graph overview.

    Attributes:
        source: Source node ID (matches GraphNode.id).
        target: Target node ID (matches GraphNode.id).
        type: Relationship type label (CALLS, CONTAINS, IMPORTS, etc.).
    """

    source: str
    target: str
    type: str


class GraphOverviewResponse(BaseModel):
    """Response for ``GET /graph/overview``.

    Attributes:
        nodes: Up to ``max_nodes`` graph nodes.
        edges: Relationships between nodes in the result set.
        node_count: Total nodes returned.
        edge_count: Total edges returned.
    """

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    node_count: int
    edge_count: int


# ---------------------------------------------------------------------------
# Phase 5 — Realtime watcher
# ---------------------------------------------------------------------------


class WatchStatus(BaseModel):
    """Response for ``GET /repos/{slug}/watch``.

    Attributes:
        repo_slug: Repo identifier (same as the URL slug).
        repo_path: Absolute filesystem path being watched.
        actor_oid: OID of the actor who started the watch.
        actor_email: Email of the actor who started the watch.
        started_at: Unix epoch when the watcher was started.
        last_event_at: Unix epoch of the last FS event; None when no
            event has fired since the watcher started.
        last_partial_job_id: Job ID of the most recent ``watch_partial``
            run; None when no partial has run yet.
        debounce_ms: Configured debounce window in milliseconds.
        pending_paths_count: Number of paths accumulated in the debouncer
            but not yet dispatched (0 outside the debounce window).
        state: Lifecycle state of the watcher entry.
    """

    repo_slug: str
    repo_path: str
    actor_oid: str
    actor_email: str
    started_at: float
    last_event_at: float | None
    last_partial_job_id: str | None
    debounce_ms: int
    pending_paths_count: int
    state: Literal["starting", "active", "stopping", "stopped", "errored"]


class WatchAccepted(BaseModel):
    """202 response from ``POST /repos/{slug}/watch``."""

    watcher_id: str
    started_at: float
    debounce_ms: int


class PartialIndexEvent(BaseModel):
    """WebSocket payload for ``index_partial_update`` events.

    Attributes:
        repo_slug: Repo the partial index ran against.
        job_id: Job ID of the ``watch_partial`` row.
        status: Terminal status of the run.
        changed_paths: Repo-relative paths included in this batch.
        files_done: Files actually re-processed (after hash-diff skip).
        files_total: Total paths in the batch before hash-diff.
        embedding_count: Symbols re-embedded.
        node_count: Graph nodes touched.
        rel_count: Graph relationships touched.
        duration_ms: Wall-clock milliseconds for the partial run.
        noop: True when hash-diff showed no content change — no graph
            work was done.
        cancelled: True when the partial was cancelled (e.g. superseded
            by a full re-index).
    """

    repo_slug: str
    job_id: str
    status: Literal["running", "done", "failed", "cancelled"]
    changed_paths: list[str] = Field(default_factory=list)
    files_done: int = 0
    files_total: int = 0
    embedding_count: int = 0
    node_count: int = 0
    rel_count: int = 0
    duration_ms: int = 0
    noop: bool = False
    cancelled: bool = False
