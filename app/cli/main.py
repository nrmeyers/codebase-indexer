"""``code-indexer`` Typer application — standalone CLI entry point.

Each subcommand is a thin wrapper around an existing FastAPI endpoint
(see :mod:`app.cli.client`). The CLI is intentionally I/O-thin: it
parses flags, calls one or two HTTP endpoints, and renders the result
with Rich for human-readable output. No business logic lives here.

Commands::

    code-indexer setup
    code-indexer serve [--port 8003]
    code-indexer start
    code-indexer stop
    code-indexer status
    code-indexer index <path> [--watch] [--force]
    code-indexer reindex <slug>
    code-indexer list
    code-indexer search "<query>" [-k 10] [--repo X]
    code-indexer symbol <fqn>
    code-indexer callers <fqn>
    code-indexer callees <fqn>
    code-indexer bundle "<task>" --repo <path> [--k N] [--depth N]
    code-indexer explore
    code-indexer remove <slug>
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .client import IndexerClient, ServiceUnavailable
from .config import (
    DEFAULT_LOG_PATH,
    DEFAULT_PID_PATH,
    CliConfig,
    EmbedderBackend,
    load_config,
    write_config,
)
from .daemon import is_port_open, pid_alive, read_pid, spawn_server, stop_server

app = typer.Typer(
    name="code-indexer",
    help=(
        "Standalone CLI for the Code Indexer Service. Manage the local "
        "FastAPI daemon, index repositories, and search code without "
        "needing TheForge. Pass the global --json flag before any "
        "subcommand for machine-readable output (for harnesses/agents)."
    ),
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
# Diagnostics (auto-start notices, job-started lines) go here so that in
# ``--json`` mode stdout carries *only* the JSON document a harness parses.
err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Shared context wiring
# ---------------------------------------------------------------------------


def _json_mode(ctx: typer.Context) -> bool:
    """Return ``True`` when the global ``--json`` flag was supplied.

    JSON mode is the contract for non-human callers (agents/harnesses):
    each command writes exactly one JSON document to stdout and routes any
    human-facing chatter to stderr.
    """
    return bool((ctx.obj or {}).get("json_output"))


def _emit_json(data: object) -> None:
    """Write a single JSON document to stdout for machine consumption.

    Uses the builtin ``print`` rather than ``console.print`` so Rich never
    soft-wraps long lines to the terminal width (which would corrupt the
    JSON when stdout is a pipe).
    """
    print(json.dumps(data, indent=2, default=str))


def _get_config(ctx: typer.Context) -> CliConfig:
    """Return the resolved :class:`CliConfig` for the current invocation."""
    obj = ctx.ensure_object(dict)
    cfg = obj.get("config")
    if cfg is None:
        cfg = load_config(base_url_override=obj.get("base_url_override"))
        obj["config"] = cfg
    return cfg


def _client(ctx: typer.Context, *, timeout: float = 30.0) -> IndexerClient:
    """Return an :class:`IndexerClient` bound to the resolved base URL."""
    cfg = _get_config(ctx)
    return IndexerClient(cfg.base_url, timeout=timeout)


def _bail(message: str, code: int = 1) -> "typer.Exit":
    """Print an error message in red and raise ``typer.Exit(code)``."""
    console.print(f"[red]{message}[/red]")
    return typer.Exit(code)


# ---------------------------------------------------------------------------
# Root callback — wires --base-url before any subcommand runs
# ---------------------------------------------------------------------------


@app.callback()
def _root(
    ctx: typer.Context,
    base_url: Annotated[
        str | None,
        typer.Option(
            "--base-url",
            envvar="CODE_INDEXER_BASE_URL",
            help="Override the FastAPI service base URL.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help=(
                "Emit machine-readable JSON to stdout (human chatter goes to "
                "stderr). Use when driving the CLI from a harness/agent."
            ),
        ),
    ] = False,
) -> None:
    """Resolve global flags before dispatching to subcommands."""
    obj = ctx.ensure_object(dict)
    if base_url:
        obj["base_url_override"] = base_url
    obj["json_output"] = json_output


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------


@app.command()
def setup(
    ctx: typer.Context,
    non_interactive: Annotated[
        bool,
        typer.Option(
            "--non-interactive",
            help="Accept defaults instead of prompting (useful in CI/tests).",
        ),
    ] = False,
) -> None:
    """Interactive first-run wizard. Writes ``~/.code-indexer/config.toml``."""
    cfg = _get_config(ctx)

    if non_interactive:
        backend: EmbedderBackend = cfg.embedder_backend
        data_dir = cfg.data_dir
        port = cfg.port
    else:
        console.print("[bold]code-indexer setup[/bold]\n")
        backend = typer.prompt(  # type: ignore[assignment]
            "Embedder backend (local / sagemaker / tei)",
            default=cfg.embedder_backend,
        )
        if backend not in ("local", "sagemaker", "tei"):
            raise _bail(
                f"Invalid embedder backend '{backend}'. "
                "Choose one of: local, sagemaker, tei."
            )
        data_dir_raw = typer.prompt(
            "Data directory", default=str(cfg.data_dir)
        )
        data_dir = Path(data_dir_raw).expanduser()
        port = typer.prompt("Default port", default=cfg.port, type=int)
        if is_port_open("127.0.0.1", port):
            console.print(
                f"[yellow]Warning:[/yellow] port {port} is already in use."
            )

    base_url = f"http://localhost:{port}"
    path = write_config(
        base_url=base_url,
        port=port,
        embedder_backend=backend,
        data_dir=data_dir,
    )
    console.print(f"[green]Wrote[/green] {path}")
    console.print(
        f"  base_url = {base_url}\n  port = {port}\n  embedder = {backend}\n"
        f"  data_dir = {data_dir}"
    )


# ---------------------------------------------------------------------------
# serve / start / stop / status
# ---------------------------------------------------------------------------


@app.command()
def serve(
    ctx: typer.Context,
    port: Annotated[int | None, typer.Option(help="Override the bind port.")] = None,
) -> None:
    """Run the FastAPI service in the foreground (``uvicorn app.main:app``)."""
    cfg = _get_config(ctx)
    bind_port = port or cfg.port
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(bind_port),
        "--log-level",
        "info",
    ]
    console.print(f"[bold]Starting[/bold] uvicorn on port {bind_port}")
    env = os.environ.copy()
    env.setdefault("PORT", str(bind_port))
    try:
        raise typer.Exit(subprocess.call(cmd, env=env))  # noqa: S603
    except KeyboardInterrupt:
        raise typer.Exit(0) from None


@app.command()
def start(ctx: typer.Context) -> None:
    """Spawn the FastAPI service in the background; records PID file."""
    cfg = _get_config(ctx)
    existing = read_pid(DEFAULT_PID_PATH)
    if existing and pid_alive(existing):
        console.print(
            f"[yellow]Already running[/yellow] (pid={existing}). "
            f"Stop with `code-indexer stop` first."
        )
        return
    try:
        pid = spawn_server(
            port=cfg.port, pid_path=DEFAULT_PID_PATH, log_path=DEFAULT_LOG_PATH
        )
    except RuntimeError as exc:
        raise _bail(str(exc)) from exc
    console.print(
        f"[green]Started[/green] pid={pid} on http://127.0.0.1:{cfg.port} "
        f"(log: {DEFAULT_LOG_PATH})"
    )


@app.command()
def stop() -> None:
    """Stop the background FastAPI daemon."""
    if stop_server(DEFAULT_PID_PATH):
        console.print("[green]Stopped.[/green]")
    else:
        console.print("[yellow]No running daemon recorded.[/yellow]")


@app.command()
def status(ctx: typer.Context) -> None:
    """Show daemon liveness and the indexed-repos table."""
    cfg = _get_config(ctx)
    json_mode = _json_mode(ctx)
    pid = read_pid(DEFAULT_PID_PATH)
    alive = bool(pid and pid_alive(pid))
    if not json_mode:
        if alive:
            console.print(f"[green]Daemon[/green] pid={pid}")
        else:
            console.print("[yellow]No managed daemon process recorded.[/yellow]")

    with _client(ctx, timeout=5.0) as client:
        try:
            health = client.health()
        except ServiceUnavailable as exc:
            raise _bail(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise _bail(f"Health check failed: {exc}") from exc

    if json_mode:
        payload: dict[str, object] = {"daemon_pid": pid, "alive": alive}
        if isinstance(health, dict):
            payload.update(health)
        _emit_json(payload)
        return

    repos = health.get("indexed_repos") or []
    table = Table(title=f"Code Indexer @ {cfg.base_url}")
    table.add_column("status")
    table.add_column("indexed repos")
    table.add_row(
        str(health.get("status", "?")),
        ", ".join(str(r) for r in repos) if repos else "(none)",
    )
    console.print(table)


# ---------------------------------------------------------------------------
# index / reindex
# ---------------------------------------------------------------------------


def _ensure_running(ctx: typer.Context) -> None:
    """Auto-start the daemon if ``/health`` is unreachable."""
    cfg = _get_config(ctx)
    if is_port_open("127.0.0.1", _port_from_url(cfg.base_url) or cfg.port):
        return
    out = err_console if _json_mode(ctx) else console
    out.print(
        f"[yellow]Code Indexer not reachable at {cfg.base_url}; "
        f"auto-starting...[/yellow]"
    )
    try:
        pid = spawn_server(
            port=cfg.port, pid_path=DEFAULT_PID_PATH, log_path=DEFAULT_LOG_PATH
        )
    except RuntimeError as exc:
        raise _bail(str(exc)) from exc
    out.print(f"[green]Auto-started[/green] pid={pid}")


def _port_from_url(url: str) -> int | None:
    """Extract the port from a ``http://host:port`` URL, if present."""
    try:
        host_part = url.split("://", 1)[1]
        if ":" in host_part:
            return int(host_part.split(":", 1)[1].split("/", 1)[0])
    except (IndexError, ValueError):
        return None
    return None


def _poll_index(client: IndexerClient, job_id: str) -> dict[str, object]:
    """Poll ``/index/{job_id}/status`` until terminal, rendering progress."""
    terminal = {"done", "failed", "interrupted", "cancelled"}
    last: dict[str, object] = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}[/bold]"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("indexing", total=100.0)
        while True:
            try:
                last = client.job_status(job_id)
            except httpx.HTTPError as exc:
                progress.update(task, description=f"poll error: {exc}")
                time.sleep(1.0)
                continue
            phase = str(last.get("phase") or last.get("status") or "...")
            pct_raw = last.get("progress_pct", 0.0)
            try:
                pct = float(pct_raw)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                pct = 0.0
            progress.update(task, completed=pct, description=phase)
            if last.get("status") in terminal:
                break
            time.sleep(0.5)
    return last


def _poll_index_quiet(client: IndexerClient, job_id: str) -> dict[str, object]:
    """Poll ``/index/{job_id}/status`` until terminal with no progress UI.

    Used in ``--json`` mode so stdout stays free of Rich progress output —
    only the final terminal status dict is emitted by the caller.
    """
    terminal = {"done", "failed", "interrupted", "cancelled"}
    last: dict[str, object] = {}
    while True:
        try:
            last = client.job_status(job_id)
        except httpx.HTTPError:
            time.sleep(1.0)
            continue
        if last.get("status") in terminal:
            return last
        time.sleep(0.5)


@app.command()
def index(
    ctx: typer.Context,
    path: Annotated[Path, typer.Argument(help="Path to repo to index.")],
    watch: Annotated[
        bool,
        typer.Option("--watch", help="Poll status until the job finishes."),
    ] = True,
    force: Annotated[
        bool,
        typer.Option("--force", help="Force a clean re-index."),
    ] = False,
) -> None:
    """Index a repository at the given path."""
    repo_path = path.expanduser().resolve()
    if not repo_path.is_dir():
        raise _bail(f"Not a directory: {repo_path}")
    _ensure_running(ctx)
    with _client(ctx) as client:
        try:
            accepted = client.start_index(str(repo_path), force_reindex=force)
        except ServiceUnavailable as exc:
            raise _bail(str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            raise _bail(
                f"Index request rejected ({exc.response.status_code}): "
                f"{exc.response.text}"
            ) from exc
        job_id = str(accepted.get("job_id") or "")
        if not job_id:
            raise _bail(f"Service returned no job_id: {accepted!r}")
        json_mode = _json_mode(ctx)
        (err_console if json_mode else console).print(
            f"[green]Job started[/green] {job_id} → {repo_path}"
        )
        if not watch:
            if json_mode:
                _emit_json(
                    {"job_id": job_id, "status": "accepted", "repo_path": str(repo_path)}
                )
            return
        result = (
            _poll_index_quiet(client, job_id)
            if json_mode
            else _poll_index(client, job_id)
        )
        status_val = str(result.get("status"))
        if json_mode:
            _emit_json(result)
            if status_val != "done":
                raise typer.Exit(1)
            return
        if status_val == "done":
            console.print(
                f"[green]Done[/green] nodes={result.get('node_count')} "
                f"relationships={result.get('rel_count', '?')}"
            )
        else:
            raise _bail(
                f"Job ended with status={status_val}: {result.get('error', '')}"
            )


@app.command()
def reindex(
    ctx: typer.Context,
    slug: Annotated[str, typer.Argument(help="Repo slug to re-index.")],
) -> None:
    """Force a clean re-index of an already-indexed repo."""
    _ensure_running(ctx)
    with _client(ctx) as client:
        # Resolve slug → repo_path via /repos
        try:
            listing = client.list_repos()
        except ServiceUnavailable as exc:
            raise _bail(str(exc)) from exc
        repos = listing.get("repos") if isinstance(listing, dict) else []
        repo_path: str | None = None
        for entry in repos or []:
            if not isinstance(entry, dict):
                continue
            if entry.get("name") == slug or entry.get("slug") == slug:
                repo_path = entry.get("repo_path") or entry.get("path")
                break
        if not repo_path:
            raise _bail(f"Unknown repo slug: {slug}")
        try:
            accepted = client.start_index(repo_path, force_reindex=True)
        except httpx.HTTPStatusError as exc:
            raise _bail(
                f"Re-index rejected ({exc.response.status_code}): "
                f"{exc.response.text}"
            ) from exc
        job_id = str(accepted.get("job_id") or "")
        json_mode = _json_mode(ctx)
        (err_console if json_mode else console).print(
            f"[green]Re-index started[/green] {job_id} → {repo_path}"
        )
        if json_mode:
            result = _poll_index_quiet(client, job_id)
            _emit_json(result)
            if str(result.get("status")) != "done":
                raise typer.Exit(1)
            return
        _poll_index(client, job_id)


# ---------------------------------------------------------------------------
# list / search / symbol / callers / callees
# ---------------------------------------------------------------------------


@app.command("list")
def list_repos_cmd(ctx: typer.Context) -> None:
    """List every repo this indexer knows about."""
    with _client(ctx) as client:
        try:
            data = client.list_repos()
        except ServiceUnavailable as exc:
            raise _bail(str(exc)) from exc
    if _json_mode(ctx):
        _emit_json(data)
        return
    repos = data.get("repos") if isinstance(data, dict) else None
    if not repos:
        console.print("[yellow]No repos indexed yet.[/yellow]")
        return
    table = Table(title="Indexed repos")
    table.add_column("slug")
    table.add_column("status")
    table.add_column("path")
    table.add_column("last_indexed")
    for entry in repos:
        if not isinstance(entry, dict):
            continue
        table.add_row(
            str(entry.get("name") or entry.get("slug") or "?"),
            str(entry.get("status", "?")),
            str(entry.get("repo_path") or entry.get("path") or ""),
            str(entry.get("last_indexed_at") or ""),
        )
    console.print(table)


@app.command()
def search(
    ctx: typer.Context,
    query: Annotated[str, typer.Argument(help="Natural-language query.")],
    k: Annotated[int, typer.Option("-k", "--k", help="Number of results.")] = 10,
    repo: Annotated[
        str | None,
        typer.Option("--repo", help="Repo slug to scope to."),
    ] = None,
) -> None:
    """Semantic search via ``GET /search/semantic``."""
    with _client(ctx) as client:
        try:
            data = client.semantic_search(query, k=k, repo=repo)
        except ServiceUnavailable as exc:
            raise _bail(str(exc)) from exc
    if _json_mode(ctx):
        _emit_json(data)
        return
    results = data.get("results", []) if isinstance(data, dict) else []
    if not results:
        console.print("[yellow]No results.[/yellow]")
        return
    table = Table(title=f"semantic: {query}")
    table.add_column("score", justify="right")
    table.add_column("symbol")
    table.add_column("type")
    for r in results:
        if not isinstance(r, dict):
            continue
        score = r.get("score")
        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "?"
        # /search/semantic returns {symbol, score, type}; fall back to the
        # older qualified_name key in case an older service is on the line.
        table.add_row(
            score_str,
            str(r.get("symbol") or r.get("qualified_name", "?")),
            str(r.get("type", "")),
        )
    console.print(table)


@app.command()
def symbol(
    ctx: typer.Context,
    fqn: Annotated[str, typer.Argument(help="Fully qualified name.")],
    repo: Annotated[str | None, typer.Option("--repo")] = None,
) -> None:
    """Look up a symbol's source via ``GET /symbols/{fqn}``."""
    with _client(ctx) as client:
        try:
            data = client.symbol(fqn, repo=repo)
        except ServiceUnavailable as exc:
            raise _bail(str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise _bail(f"Symbol not found: {fqn}") from exc
            raise
    _emit_json(data)


def _render_call_sites(title: str, data: dict[str, object]) -> None:
    """Render a callers/callees response as a Rich table."""
    results = data.get("results") if isinstance(data, dict) else None
    if not results:
        console.print(f"[yellow]No {title}.[/yellow]")
        return
    table = Table(title=title)
    table.add_column("qualified_name")
    table.add_column("file")
    table.add_column("line", justify="right")
    for r in results:  # type: ignore[union-attr]
        if not isinstance(r, dict):
            continue
        table.add_row(
            str(r.get("qualified_name", "?")),
            str(r.get("file_path", "")),
            str(r.get("line_number", "")),
        )
    console.print(table)


@app.command()
def callers(
    ctx: typer.Context,
    fqn: Annotated[str, typer.Argument()],
    repo: Annotated[str | None, typer.Option("--repo")] = None,
) -> None:
    """List symbols that call ``fqn``."""
    with _client(ctx) as client:
        try:
            data = client.callers(fqn, repo=repo)
        except ServiceUnavailable as exc:
            raise _bail(str(exc)) from exc
    if _json_mode(ctx):
        _emit_json(data)
        return
    _render_call_sites(f"callers of {fqn}", data)


@app.command()
def callees(
    ctx: typer.Context,
    fqn: Annotated[str, typer.Argument()],
    repo: Annotated[str | None, typer.Option("--repo")] = None,
) -> None:
    """List symbols that ``fqn`` calls."""
    with _client(ctx) as client:
        try:
            data = client.callees(fqn, repo=repo)
        except ServiceUnavailable as exc:
            raise _bail(str(exc)) from exc
    if _json_mode(ctx):
        _emit_json(data)
        return
    _render_call_sites(f"callees of {fqn}", data)


# ---------------------------------------------------------------------------
# bundle / explore / remove
# ---------------------------------------------------------------------------


@app.command()
def bundle(
    ctx: typer.Context,
    task: Annotated[str, typer.Argument(help="Natural-language task description.")],
    repo: Annotated[
        Path,
        typer.Option("--repo", help="Path to the indexed repo."),
    ],
    k: Annotated[int, typer.Option("--k", help="Seed symbols.")] = 10,
    depth: Annotated[int, typer.Option("--depth", help="Call-graph hop depth.")] = 2,
) -> None:
    """Build a grounded code context bundle via ``POST /context-bundle``."""
    repo_path = repo.expanduser().resolve()
    if not repo_path.is_dir():
        raise _bail(f"Not a directory: {repo_path}")
    with _client(ctx, timeout=120.0) as client:
        try:
            data = client.context_bundle(str(repo_path), task, k=k, depth=depth)
        except ServiceUnavailable as exc:
            raise _bail(str(exc)) from exc
    _emit_json(data)


@app.command()
def explore(ctx: typer.Context) -> None:
    """Open the LadybugDB Explorer URL in a browser."""
    with _client(ctx) as client:
        try:
            data = client.explorer_info()
        except ServiceUnavailable as exc:
            raise _bail(str(exc)) from exc
    if _json_mode(ctx):
        _emit_json(data)
        return
    url = (
        data.get("url") if isinstance(data, dict) else None
    ) or data.get("explorer_url") if isinstance(data, dict) else None
    if not url:
        console.print_json(data=data)
        return
    console.print(f"[bold]Opening[/bold] {url}")
    try:
        webbrowser.open(str(url))
    except Exception:  # pragma: no cover - webbrowser is platform-specific
        pass


@app.command()
def remove(
    ctx: typer.Context,
    slug: Annotated[str, typer.Argument(help="Repo slug to delete.")],
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip confirmation prompt.")
    ] = False,
) -> None:
    """Delete a repo's index and all related artefacts."""
    if not yes:
        typer.confirm(f"Delete the index for '{slug}'?", abort=True)
    with _client(ctx) as client:
        try:
            data = client.delete_repo(slug)
        except ServiceUnavailable as exc:
            raise _bail(str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise _bail(f"Unknown repo slug: {slug}") from exc
            raise
    if _json_mode(ctx):
        _emit_json(data if isinstance(data, dict) else {"deleted": True, "slug": slug})
        return
    console.print(f"[green]Deleted[/green] {slug}")
    if isinstance(data, dict) and data:
        console.print_json(data=data)


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    app()
