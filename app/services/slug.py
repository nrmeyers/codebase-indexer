"""Canonical repo slug derivation (BUC-1580).

Resolves a stable ``{org}__{repo}`` slug for an indexed working tree by
inspecting ``git remote get-url origin`` instead of relying on the
working-tree directory's basename.

Why this exists:
    The indexer historically derived its per-repo slug from
    ``Path(repo_path).name``.  That works for the App-clone path (which
    writes clones into ``.cgr/clones/{owner}__{repo}``), but breaks when
    the same repo is also indexed from a developer's local checkout —
    e.g. ``~/TheForge`` lands under slug ``TheForge`` while the
    App-clone lands under ``navistone__TheForge``, and the two indexes
    diverge silently.

Resolution rules:
    1. Run ``git -C <local_path> remote get-url origin`` (5s timeout).
    2. If exactly one remote is configured AND its URL parses as a
       GitHub HTTPS or SSH URL, return ``{org}__{repo}``.
    3. Otherwise return None — the caller falls back to the basename.

The function is pure-python with the only side effect being a bounded
subprocess call.  Failures are silent (log + None) so that ingestion
never blocks on an environmental quirk like a missing ``git`` binary.
"""
from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from ..config import slugify_repo

logger = logging.getLogger(__name__)


# https://github.com/<org>/<repo>(.git)?  — also accept the trailing slash form.
# git@github.com:<org>/<repo>(.git)?
# ssh://git@github.com/<org>/<repo>(.git)?
_GITHUB_URL_RE = re.compile(
    r"""^
    (?:
        (?:https?://)(?:[^@/]+@)?github\.com/
        |
        git@github\.com:
        |
        ssh://git@github\.com/
    )
    (?P<org>[A-Za-z0-9][A-Za-z0-9._-]*)/
    (?P<repo>[A-Za-z0-9][A-Za-z0-9._-]*?)
    (?:\.git)?/?
    $""",
    re.VERBOSE,
)

# Hard cap on subprocess wall-clock so a hung filesystem can't stall ingest.
_GIT_TIMEOUT_S = 5.0


def parse_github_remote(url: str) -> Optional[Tuple[str, str]]:
    """Parse a GitHub remote URL into ``(org, repo)``.

    Args:
        url: A remote URL string from ``git remote get-url``.  Accepts
            HTTPS (``https://github.com/org/repo(.git)?``), SSH
            (``git@github.com:org/repo(.git)?``), and the rarer
            ``ssh://git@github.com/...`` form.

    Returns:
        Tuple of ``(org, repo)`` when the URL is a recognisable GitHub
        URL, else None.  ``.git`` and trailing-slash suffixes are stripped.
        Non-GitHub hosts (gitlab, bitbucket, self-hosted) return None so
        the caller falls back to the basename.

    Examples:
        >>> parse_github_remote("git@github.com:navistone/TheForge.git")
        ('navistone', 'TheForge')
        >>> parse_github_remote("https://github.com/navistone/TheForge")
        ('navistone', 'TheForge')
        >>> parse_github_remote("https://gitlab.com/foo/bar.git") is None
        True
    """
    if not isinstance(url, str):
        return None
    candidate = url.strip()
    if not candidate:
        return None
    match = _GITHUB_URL_RE.match(candidate)
    if not match:
        return None
    org = match.group("org")
    repo = match.group("repo")
    if not org or not repo:
        return None
    return (org, repo)


def canonical_slug_for_path(local_path: Path) -> Optional[str]:
    """Return ``{org}__{repo}`` when ``local_path`` has a single GitHub remote.

    Runs ``git -C <local_path> remote`` to enumerate remotes (we refuse
    to guess when more than one exists — origin may not be the canonical
    one), then ``git -C <local_path> remote get-url origin`` to fetch the
    URL.  All subprocess calls are capped at 5 seconds.

    Args:
        local_path: Resolved working-tree directory.

    Returns:
        Canonical slug ``"{org}__{repo}"`` (already passed through
        ``slugify_repo`` to enforce the filesystem-safe charset), or
        None when:
            * ``local_path`` is not a git checkout
            * the repo has no ``origin`` remote
            * the repo has multiple remotes (ambiguous — refuse to guess)
            * the origin URL is not a GitHub URL
            * the ``git`` binary is missing or times out
    """
    path = Path(local_path)
    if not path.is_dir():
        return None

    try:
        # First check the remote count — if there are multiple, we refuse
        # to guess which one is canonical (some monorepos have an origin
        # pointing at a fork plus an ``upstream`` pointing at the canonical
        # repo; picking origin would silently route the slug to the wrong
        # GitHub project).
        remotes_proc = subprocess.run(
            ["git", "-C", str(path), "remote"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
        if remotes_proc.returncode != 0:
            return None
        remotes = [r.strip() for r in remotes_proc.stdout.splitlines() if r.strip()]
        if len(remotes) != 1 or remotes[0] != "origin":
            # Either zero remotes, multiple remotes, or origin missing —
            # all three are ambiguous, fall back to basename.
            return None

        url_proc = subprocess.run(
            ["git", "-C", str(path), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
        if url_proc.returncode != 0:
            return None
        url = (url_proc.stdout or "").strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.debug("slug.canonical: git probe failed for %s — %s", path, exc)
        return None

    parsed = parse_github_remote(url)
    if parsed is None:
        return None
    org, repo = parsed
    return slugify_repo(f"{org}__{repo}")


def derive_slug(local_path: Path, fallback_basename: str) -> str:
    """Derive the canonical filesystem slug for an indexed repo.

    Tries :func:`canonical_slug_for_path` first; falls back to
    ``fallback_basename`` (passed through :func:`slugify_repo` for safety)
    when no canonical slug can be determined.

    Args:
        local_path: Working-tree path on disk.  Need not exist — non-existent
            paths fall through to ``fallback_basename`` cleanly.
        fallback_basename: The name to slug-encode when no canonical slug
            can be derived.  Typically ``Path(repo_path).name``.

    Returns:
        A filesystem-safe slug.  Never empty (``slugify_repo`` falls back
        to ``"repo"`` for pathological inputs).
    """
    canonical = canonical_slug_for_path(Path(local_path))
    if canonical:
        return canonical
    return slugify_repo(fallback_basename or "repo")
