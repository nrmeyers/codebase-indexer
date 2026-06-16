"""Shared helpers for the retrieval-eval harness scripts.

Three sibling scripts (run_recall.py, run_probes.py, run_arms.py) all map
checkout-dir names to canonical service slugs via the /health endpoint. The
implementations had diverged just enough that a fix to one didn't reach the
others. This module is the single source.
"""
from __future__ import annotations

from pathlib import Path

import httpx


def resolve_slugs_sync(
    base: str,
    repos: set[str],
    repo_paths: dict[str, str] | None = None,
) -> dict[str, str]:
    """Synchronous variant for harnesses that don't use ``httpx.AsyncClient``.

    Args:
        base: indexer service base URL (e.g. ``http://127.0.0.1:8000``).
        repos: query repo names from ``queries.json`` / probe specs.
        repo_paths: optional name -> checkout-path map so the matcher can
            fall back to the directory name when the raw name does not appear
            in /health. ``run_recall`` calls with ``None``; ``run_probes`` /
            ``run_arms`` pass ``_DEFAULT_REPO_PATHS``.

    Returns:
        Mapping ``{requested_name: service_slug}``. Falls back to the input
        name (or directory name) when nothing matches so callers always get a
        usable string.
    """
    health = httpx.get(f"{base}/health", timeout=10).json()
    return _match(health, repos, repo_paths)


async def resolve_slugs_async(
    client: httpx.AsyncClient,
    base: str,
    repos: set[str],
    repo_paths: dict[str, str] | None = None,
) -> dict[str, str]:
    """Async variant — shares an ``AsyncClient`` with the harness's main loop."""
    health = (await client.get(f"{base}/health", timeout=10.0)).json()
    return _match(health, repos, repo_paths)


def _match(
    health: dict,
    repos: set[str],
    repo_paths: dict[str, str] | None,
) -> dict[str, str]:
    slugs = [r["name"] for r in health.get("repos", [])]
    out: dict[str, str] = {}
    for name in repos:
        target = Path(repo_paths.get(name, name)).name if repo_paths else name
        match = next((s for s in slugs if s == target), None) or next(
            (s for s in slugs if target.lower() in s.lower()), None
        )
        out[name] = match or target
    return out
