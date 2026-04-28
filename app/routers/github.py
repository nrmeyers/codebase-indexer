"""GET /github/repos and POST /github/index — GitHub repo discovery & ingest.

Lets the TheForge UI pick a GitHub repository from a dropdown and kick off an
indexing job without the user having to clone the repo by hand.

The flow is:

1. ``GET /github/repos`` → fetches the authenticated user's personal repos
   plus every org they belong to via the GitHub REST API.  Auth is a PAT
   read from the ``GITHUB_TOKEN`` env var (same token TheForge uses).
2. ``POST /github/index`` → clones (or fast-forward fetches) the repo into
   ``.cgr/clones/{owner}__{name}`` and reuses the existing background
   ``_run_ingestion`` worker from the ``index`` router.  The per-repo DB
   file ends up at ``.cgr/repos/{slug}.db`` with the repo name as the slug,
   so the explorer / search / browse endpoints find it automatically.

No tokens are returned to the browser; the service acts as a proxy.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from ..config import settings, slugify_repo
from ..models import GitHubRateLimit, GitHubStatusResponse, IndexAccepted
from .index import _Job, _jobs, _run_ingestion

router = APIRouter(prefix="/github")


# ---------------------------------------------------------------------------
# Models (kept inline — only used by this router)
# ---------------------------------------------------------------------------


class GitHubRepo(BaseModel):
    """Minimal repo record returned by ``GET /github/repos``.

    Mirrors the fields the UI needs to render a picker: owner/name for
    display and uniqueness, clone URL for ingestion, privacy + stars for
    sorting/filtering, and default branch for the clone.
    """

    full_name: str = Field(description="owner/name (e.g. 'navistone/legacy-api')")
    name: str = Field(description="Short repo name (e.g. 'legacy-api')")
    owner: str = Field(description="Account or org that owns the repo")
    private: bool = Field(description="True for private repos")
    description: str | None = None
    default_branch: str = Field(default="main", description="Branch to clone by default")
    clone_url: str = Field(description="HTTPS clone URL")
    ssh_url: str | None = None
    stars: int = Field(default=0, alias="stargazers_count")
    updated_at: str | None = None


class GitHubReposResponse(BaseModel):
    """Envelope for ``GET /github/repos`` — list + total count."""

    repos: list[GitHubRepo]
    total: int


class GitHubOrg(BaseModel):
    """Minimal org record returned by ``GET /github/orgs``.

    The Settings UI uses these to render the allowlist editor as a
    checkbox list with the user's actual orgs (instead of forcing
    them to type ``navistone`` from memory).
    """

    login: str = Field(description="Org slug, e.g. 'navistone'")
    description: str | None = None
    avatar_url: str | None = None
    allowlisted: bool = Field(
        default=False,
        description="True when this org is currently in GITHUB_ALLOWED_OWNERS",
    )


class GitHubOrgsResponse(BaseModel):
    """Envelope for ``GET /github/orgs``."""

    orgs: list[GitHubOrg]
    total: int
    allowlist: list[str] = Field(
        description="Currently configured allowlist (lower-cased) for cross-checking",
    )


class GitHubIndexRequest(BaseModel):
    """Body for ``POST /github/index``."""

    full_name: str = Field(
        description="owner/repo to clone and index",
        min_length=3,
    )
    branch: str | None = Field(
        default=None,
        description="Branch to check out — defaults to the repo's default branch",
    )
    force_reindex: bool = Field(
        default=False,
        description="Clear existing graph before re-indexing",
    )


# ---------------------------------------------------------------------------
# GitHub API client
# ---------------------------------------------------------------------------


_GITHUB_API = "https://api.github.com"


def _enforce_owner_allowlist(full_name: str) -> None:
    """Reject ``full_name`` whose owner isn't in the configured allowlist.

    The allowlist defends against an attacker (or a stray UI bug) that
    POSTs ``full_name="random/repo"`` and tricks the indexer into cloning
    arbitrary public repos onto local disk.  When ``GITHUB_ALLOWED_OWNERS``
    is empty the guard is a no-op so dev installs aren't gated.

    Args:
        full_name: ``owner/repo`` identifier from the request body.

    Raises:
        HTTPException: 422 when ``full_name`` is malformed; 403 when the
        owner is not in the allowlist.
    """
    owner, _, name = full_name.partition("/")
    if not owner or not name:
        raise HTTPException(status_code=422, detail=f"Invalid full_name: {full_name}")

    allowed = settings.github_allowed_owners
    if allowed and owner.lower() not in allowed:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Owner '{owner}' is not in GITHUB_ALLOWED_OWNERS "
                f"({', '.join(allowed)}). Refusing to clone."
            ),
        )


def _github_token() -> str | None:
    """Return the GitHub PAT for upstream API calls, or None.

    Resolves in order: Settings (from ``.env``) → process env.  Falls back
    across both ``GITHUB_TOKEN`` and ``GH_TOKEN`` names so users who only
    have the ``gh`` CLI configured don't have to duplicate anything.
    """
    token = settings.GITHUB_TOKEN or settings.GH_TOKEN
    if token:
        return token
    return os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")


def _token_source() -> str:
    """Classify where the active token came from so the UI can explain it."""
    if settings.GITHUB_TOKEN or settings.GH_TOKEN:
        return "settings"
    if os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN"):
        return "env"
    return "none"


async def _gh_get(
    client: httpx.AsyncClient, path: str, params: dict[str, str | int] | None = None
) -> list[dict] | dict:
    """Authenticated GET against the GitHub REST API.

    Args:
        client: A shared async HTTP client (so auth + retries stay consistent).
        path: Path component starting with ``/`` (e.g. ``/user/repos``).
        params: Query params passed straight through.

    Returns:
        Parsed JSON body.

    Raises:
        HTTPException: 401 when the token is missing or rejected, 502 for
            any other upstream failure so the UI surfaces a retryable error.
    """
    token = _github_token()
    if not token:
        raise HTTPException(
            status_code=401,
            detail="GITHUB_TOKEN not set — add it to .env to browse GitHub repos.",
        )
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    r = await client.get(f"{_GITHUB_API}{path}", headers=headers, params=params)
    if r.status_code == 401:
        raise HTTPException(status_code=401, detail="GitHub rejected the token.")
    if r.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"GitHub API error ({r.status_code}): {r.text[:200]}",
        )
    return r.json()


# ---------------------------------------------------------------------------
# GET /github/status
# ---------------------------------------------------------------------------


@router.get("/status", response_model=GitHubStatusResponse)
async def github_status() -> GitHubStatusResponse:
    """Readiness probe for the GitHub integration.

    Returns enough information for the UI to render a "Connected as
    @username · 4998/5000" badge without a second round-trip.  Never
    raises on missing/invalid tokens — the response itself encodes the
    failure state via ``connected=False`` and a human-readable ``message``.

    Returns:
        GitHubStatusResponse: connection readiness, token source, user,
        scopes, rate-limit snapshot, and a user-facing message.
    """
    source = _token_source()
    token = _github_token()

    if not token:
        return GitHubStatusResponse(
            connected=False,
            token_source="none",
            user=None,
            scopes=[],
            rate_limit=None,
            message="GITHUB_TOKEN not set. Add it to .env or export it to browse GitHub repos.",
        )

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # Best-effort probe.  Both calls are wrapped so a network hiccup doesn't
    # turn the readiness endpoint into a 500 — the UI needs a clean response
    # to decide whether to show "connected" or "tap to retry".
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            user_resp = await client.get(f"{_GITHUB_API}/user", headers=headers)
            if user_resp.status_code == 401:
                return GitHubStatusResponse(
                    connected=False,
                    token_source=source,  # type: ignore[arg-type]
                    user=None,
                    scopes=[],
                    rate_limit=None,
                    message="GitHub rejected the token (401). Regenerate the PAT or refresh its scopes.",
                )
            if user_resp.status_code >= 400:
                return GitHubStatusResponse(
                    connected=False,
                    token_source=source,  # type: ignore[arg-type]
                    user=None,
                    scopes=[],
                    rate_limit=None,
                    message=f"GitHub API returned {user_resp.status_code}. Check connectivity.",
                )

            user_body = user_resp.json()
            login = user_body.get("login")
            scopes_header = user_resp.headers.get("X-OAuth-Scopes") or ""
            scopes = [s.strip() for s in scopes_header.split(",") if s.strip()]

            # Pull the current rate-limit snapshot for the primary (core) pool.
            rl_resp = await client.get(f"{_GITHUB_API}/rate_limit", headers=headers)
            rl: GitHubRateLimit | None = None
            if rl_resp.status_code == 200:
                core = (rl_resp.json().get("resources") or {}).get("core") or {}
                rl = GitHubRateLimit(
                    limit=int(core.get("limit", 0)),
                    remaining=int(core.get("remaining", 0)),
                    reset_at=float(core.get("reset")) if core.get("reset") else None,
                )

            msg = f"Connected as @{login}" if login else "Connected"
            if rl is not None:
                msg += f" · {rl.remaining}/{rl.limit} rate-limit"

            return GitHubStatusResponse(
                connected=True,
                token_source=source,  # type: ignore[arg-type]
                user=login,
                scopes=scopes,
                rate_limit=rl,
                message=msg,
            )
    except Exception as exc:
        return GitHubStatusResponse(
            connected=False,
            token_source=source,  # type: ignore[arg-type]
            user=None,
            scopes=[],
            rate_limit=None,
            message=f"Could not reach api.github.com: {exc}",
        )


# ---------------------------------------------------------------------------
# GET /github/orgs
# ---------------------------------------------------------------------------


@router.get("/orgs", response_model=GitHubOrgsResponse)
async def list_orgs() -> GitHubOrgsResponse:
    """List orgs the authenticated user belongs to.

    Powers the Settings allowlist editor: instead of asking the user to
    type org names into a CSV string, the UI fetches this list and
    renders checkboxes pre-checked against the current
    ``GITHUB_ALLOWED_OWNERS`` setting.

    Returns:
        GitHubOrgsResponse: Each org includes an ``allowlisted`` flag so
        the UI doesn't have to do its own case-insensitive comparison.

    Raises:
        HTTPException: 401 when no token is configured (the response
        already has a richer status path via ``/github/status``, but a
        plain 401 here keeps the contract obvious for direct callers).
    """
    allowed = settings.github_allowed_owners

    async with httpx.AsyncClient(timeout=10.0) as client:
        # ``/user/orgs`` requires the ``read:org`` scope; without it
        # GitHub returns an empty list rather than a 403, which is fine
        # — the UI just shows "no orgs found, add read:org scope".
        raw = await _gh_get(client, "/user/orgs", params={"per_page": 100})

    orgs_in: list[dict] = list(raw) if isinstance(raw, list) else []

    # Build the response with a stable (alphabetised) ordering and an
    # explicit allowlisted flag so the UI doesn't have to do the
    # case-insensitive comparison itself.
    orgs: list[GitHubOrg] = []
    for o in orgs_in:
        login = o.get("login")
        if not login:
            continue
        orgs.append(
            GitHubOrg(
                login=login,
                description=o.get("description"),
                avatar_url=o.get("avatar_url"),
                allowlisted=login.lower() in allowed,
            )
        )

    orgs.sort(key=lambda o: o.login.lower())
    return GitHubOrgsResponse(orgs=orgs, total=len(orgs), allowlist=allowed)


# ---------------------------------------------------------------------------
# GET /github/repos
# ---------------------------------------------------------------------------


@router.get("/repos", response_model=GitHubReposResponse)
async def list_repos(
    q: str | None = None,
    limit: int = 100,
) -> GitHubReposResponse:
    """List the authenticated user's repos + every org repo they can access.

    Args:
        q: Optional case-insensitive substring filter on ``full_name``.
        limit: Max number of repos to return after sorting by last update.

    Returns:
        GitHubReposResponse: Sorted by ``updated_at`` desc so freshly-touched
        repos bubble to the top of the picker.
    """
    if limit < 1:
        limit = 1
    if limit > 500:
        limit = 500

    async with httpx.AsyncClient(timeout=15.0) as client:
        # Personal repos — include private ones the token has access to.
        user_repos = await _gh_get(
            client,
            "/user/repos",
            params={"per_page": 100, "sort": "updated", "affiliation": "owner,collaborator,organization_member"},
        )

    raw = list(user_repos) if isinstance(user_repos, list) else []

    # Normalise and dedupe by full_name (a repo can appear twice if the user
    # is both a direct collaborator and an org member).
    seen: set[str] = set()
    repos: list[GitHubRepo] = []
    for r in raw:
        fn = r.get("full_name")
        if not fn or fn in seen:
            continue
        seen.add(fn)
        repos.append(
            GitHubRepo(
                full_name=fn,
                name=r.get("name", ""),
                owner=(r.get("owner") or {}).get("login", ""),
                private=bool(r.get("private", False)),
                description=r.get("description"),
                default_branch=r.get("default_branch", "main"),
                clone_url=r.get("clone_url", ""),
                ssh_url=r.get("ssh_url"),
                stargazers_count=int(r.get("stargazers_count", 0)),
                updated_at=r.get("updated_at"),
            )
        )

    # Apply owner allowlist before any text filter — UI never sees repos it
    # couldn't clone anyway, and the response shrinks for big org lists.
    allowed = settings.github_allowed_owners
    if allowed:
        repos = [r for r in repos if r.owner.lower() in allowed]

    if q:
        needle = q.lower()
        repos = [r for r in repos if needle in r.full_name.lower()]

    repos.sort(key=lambda r: r.updated_at or "", reverse=True)
    return GitHubReposResponse(repos=repos[:limit], total=len(repos))


# ---------------------------------------------------------------------------
# POST /github/index
# ---------------------------------------------------------------------------


_CLONES_DIR = ".cgr/clones"


def _clone_or_update(full_name: str, branch: str | None, token: str | None) -> Path:
    """Clone the repo into ``.cgr/clones`` or fast-forward an existing clone.

    The auth token is injected into the URL only for private repos to avoid
    leaking it into git's credential helper for public ones.  The clone
    happens with ``--depth 1`` because the indexer only needs the tree at
    HEAD — full history would bloat disk for large monorepos.

    Args:
        full_name: ``owner/repo`` identifier.
        branch: Branch to check out, or None to stick with the default.
        token: GitHub PAT for private repos.  When None, the clone is
            anonymous and only public repos will succeed.

    Returns:
        Path: Filesystem path to the freshly-prepared working tree.

    Raises:
        HTTPException: 502 when ``git`` returns a non-zero exit, so the UI
            can surface the git error message to the user instead of
            silently stalling the job.
    """
    owner, _, name = full_name.partition("/")
    if not owner or not name:
        raise HTTPException(status_code=422, detail=f"Invalid full_name: {full_name}")

    dest = Path(_CLONES_DIR) / f"{owner}__{name}"
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Build auth URL when a token is set so private repos work.  Token
    # embedded in the URL is the simplest, CI-friendly approach; ``git
    # clone`` + ``gh auth git-credential`` would be cleaner in production
    # but adds a runtime dependency for local dev.
    if token:
        url = f"https://x-access-token:{token}@github.com/{owner}/{name}.git"
    else:
        url = f"https://github.com/{owner}/{name}.git"

    try:
        if dest.exists() and (dest / ".git").exists():
            # Fast-forward an existing clone.
            subprocess.run(
                ["git", "-C", str(dest), "fetch", "--depth=1", "origin", branch or "HEAD"],
                check=True, capture_output=True, text=True, timeout=120,
            )
            subprocess.run(
                ["git", "-C", str(dest), "reset", "--hard", "FETCH_HEAD"],
                check=True, capture_output=True, text=True, timeout=60,
            )
        else:
            cmd = ["git", "clone", "--depth=1"]
            if branch:
                cmd += ["--branch", branch]
            cmd += [url, str(dest)]
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
    except subprocess.CalledProcessError as exc:
        # Scrub the token from the error message before surfacing.
        msg = (exc.stderr or exc.stdout or str(exc))
        if token:
            msg = msg.replace(token, "***")
        raise HTTPException(status_code=502, detail=f"git failed: {msg[:500]}") from exc

    return dest.resolve()


@router.post("/index", response_model=IndexAccepted, status_code=202)
async def index_github_repo(
    req: GitHubIndexRequest,
    background_tasks: BackgroundTasks,
) -> IndexAccepted:
    """Clone + index a GitHub repo asynchronously.

    Args:
        req: Body specifying ``full_name`` and optional branch / force flag.
        background_tasks: FastAPI-injected task registry; the clone runs in
            a worker thread and the ingestion is scheduled via the same
            ``_run_ingestion`` coroutine the local ``/index`` endpoint uses,
            so job polling works unchanged.

    Returns:
        IndexAccepted: Job id the UI polls via ``/index/{job_id}/status``.
    """
    # Reject disallowed owners before doing any work — saves a network round-trip
    # and prevents the clone token from being passed to a URL we don't trust.
    _enforce_owner_allowlist(req.full_name)

    # Clone runs off-loop because subprocess.run blocks; keep the HTTP handler
    # snappy so the UI doesn't see a multi-second hang before the job id comes back.
    token = _github_token()
    loop = asyncio.get_running_loop()
    try:
        repo_path = await loop.run_in_executor(
            None, _clone_or_update, req.full_name, req.branch, token
        )
    except HTTPException:
        raise

    # The per-repo DB name is derived from the working-tree folder name,
    # which includes the owner prefix to prevent collisions between repos
    # with the same short name (e.g. acme/legacy-api vs foo/legacy-api).
    slug = slugify_repo(repo_path.name)

    job_id = str(uuid.uuid4())
    job = _Job(job_id=job_id, repo_path=str(repo_path))
    _jobs[job_id] = job

    # Sanity check — the slug drives the DB filename below, so if it's empty
    # we'd collide with every other repo.  Shouldn't be reachable (slugify
    # never returns empty), but an explicit assert beats a silent corruption.
    assert slug, "slug derivation returned empty string"

    background_tasks.add_task(_run_ingestion, job, req.force_reindex)
    return IndexAccepted(job_id=job_id)
