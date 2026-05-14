"""Thin synchronous HTTP client for the Code Indexer FastAPI service.

The CLI subcommands are deliberately stateless — they construct a
:class:`IndexerClient`, issue one or two HTTP calls, and render the
result. Connection failures surface as :class:`ServiceUnavailable` so
callers can offer to auto-start the daemon.
"""
from __future__ import annotations

from typing import Any

import httpx


class ServiceUnavailable(RuntimeError):
    """Raised when the FastAPI service refuses a TCP connection.

    Distinct from generic ``httpx.HTTPError`` so the CLI can offer to
    auto-start the daemon without swallowing real protocol-level errors.
    """


class IndexerClient:
    """Tiny ``httpx``-backed client mirroring the public REST surface.

    Args:
        base_url: Base URL of the running FastAPI service. Stripped of
            trailing slashes so endpoint paths can be concatenated
            naively.
        timeout: Per-request timeout in seconds. Index polling uses a
            higher value via :meth:`get`.
    """

    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={"User-Agent": "code-indexer-cli/0.1"},
        )

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        """Close the underlying connection pool."""
        self._client.close()

    def __enter__(self) -> "IndexerClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- generic helpers --------------------------------------------------

    def _wrap(self, fn):  # type: ignore[no-untyped-def]
        try:
            return fn()
        except httpx.ConnectError as exc:
            raise ServiceUnavailable(
                f"Could not reach Code Indexer at {self.base_url}: {exc}"
            ) from exc

    def get(self, path: str, params: dict[str, Any] | None = None) -> httpx.Response:
        """Issue a GET request relative to :attr:`base_url`."""
        return self._wrap(lambda: self._client.get(path, params=params))

    def post(
        self,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Issue a POST request relative to :attr:`base_url`."""
        return self._wrap(lambda: self._client.post(path, json=json, params=params))

    def delete(self, path: str) -> httpx.Response:
        """Issue a DELETE request relative to :attr:`base_url`."""
        return self._wrap(lambda: self._client.delete(path))

    # -- typed endpoint helpers -------------------------------------------

    def health(self) -> dict[str, Any]:
        """Fetch ``GET /health``."""
        r = self.get("/health")
        r.raise_for_status()
        return r.json()

    def list_repos(self) -> dict[str, Any]:
        """Fetch ``GET /repos``."""
        r = self.get("/repos")
        r.raise_for_status()
        return r.json()

    def start_index(self, repo_path: str, *, force_reindex: bool = False) -> dict[str, Any]:
        """Start an indexing job via ``POST /index``."""
        r = self.post(
            "/index",
            json={"repo_path": repo_path, "force_reindex": force_reindex},
        )
        r.raise_for_status()
        return r.json()

    def job_status(self, job_id: str) -> dict[str, Any]:
        """Poll ``GET /index/{job_id}/status``."""
        r = self.get(f"/index/{job_id}/status")
        r.raise_for_status()
        return r.json()

    def semantic_search(
        self, query: str, *, k: int = 10, repo: str | None = None
    ) -> dict[str, Any]:
        """Fetch ``GET /search/semantic``."""
        params: dict[str, Any] = {"q": query, "k": k}
        if repo:
            params["repo"] = repo
        r = self.get("/search/semantic", params=params)
        r.raise_for_status()
        return r.json()

    def symbol(self, fqn: str, *, repo: str | None = None) -> dict[str, Any]:
        """Fetch ``GET /symbols/{fqn}``."""
        params: dict[str, Any] = {}
        if repo:
            params["repo"] = repo
        r = self.get(f"/symbols/{fqn}", params=params)
        r.raise_for_status()
        return r.json()

    def callers(self, fqn: str, *, repo: str | None = None) -> dict[str, Any]:
        """Fetch ``GET /symbols/{fqn}/callers``."""
        params: dict[str, Any] = {}
        if repo:
            params["repo"] = repo
        r = self.get(f"/symbols/{fqn}/callers", params=params)
        r.raise_for_status()
        return r.json()

    def callees(self, fqn: str, *, repo: str | None = None) -> dict[str, Any]:
        """Fetch ``GET /symbols/{fqn}/callees``."""
        params: dict[str, Any] = {}
        if repo:
            params["repo"] = repo
        r = self.get(f"/symbols/{fqn}/callees", params=params)
        r.raise_for_status()
        return r.json()

    def context_bundle(
        self,
        repo_path: str,
        task: str,
        *,
        k: int = 10,
        depth: int = 2,
    ) -> dict[str, Any]:
        """Fetch ``POST /context-bundle``."""
        r = self.post(
            "/context-bundle",
            json={
                "repo_path": repo_path,
                "task_description": task,
                "k": k,
                "depth": depth,
            },
        )
        r.raise_for_status()
        return r.json()

    def explorer_info(self) -> dict[str, Any]:
        """Fetch ``GET /explorer/info``."""
        r = self.get("/explorer/info")
        r.raise_for_status()
        return r.json()

    def delete_repo(self, slug: str) -> dict[str, Any]:
        """Issue ``DELETE /index/{slug}``."""
        r = self.delete(f"/index/{slug}")
        r.raise_for_status()
        return r.json()
