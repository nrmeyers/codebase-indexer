"""Unit tests for the standalone ``code-indexer`` CLI.

Each test uses Typer's :class:`~typer.testing.CliRunner` to invoke a
subcommand and ``respx`` to mock the underlying FastAPI service. These
tests exercise the Typer command surface end-to-end (flag parsing,
rendering, exit codes) without spawning a real uvicorn process.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import httpx
import pytest
import respx
from typer.testing import CliRunner

from app.cli.main import app
from app.cli import config as cli_config
from app.cli import main as cli_main

BASE_URL = "http://localhost:8003"


@pytest.fixture
def runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[CliRunner]:
    """Yield a CliRunner with isolated config + PID files.

    Patches the CLI's data-dir / PID-file globals so tests never touch
    the developer's real ``~/.code-indexer/`` directory.
    """
    data_dir = tmp_path / ".code-indexer"
    data_dir.mkdir()
    cfg_path = data_dir / "config.toml"
    cfg_path.write_text(
        '[server]\nbase_url = "http://localhost:8003"\nport = 8003\n'
        '\n[embedder]\nbackend = "local"\n'
        f'\n[paths]\ndata_dir = "{data_dir}"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_config, "DEFAULT_DATA_DIR", data_dir)
    monkeypatch.setattr(cli_config, "DEFAULT_CONFIG_PATH", cfg_path)
    monkeypatch.setattr(cli_config, "DEFAULT_PID_PATH", data_dir / "server.pid")
    monkeypatch.setattr(cli_config, "DEFAULT_LOG_PATH", data_dir / "server.log")
    monkeypatch.setattr(cli_main, "DEFAULT_PID_PATH", data_dir / "server.pid")
    monkeypatch.setattr(cli_main, "DEFAULT_LOG_PATH", data_dir / "server.log")
    # Pretend the port is always open so _ensure_running short-circuits.
    monkeypatch.setattr(cli_main, "is_port_open", lambda *_a, **_kw: True)
    yield CliRunner()


# ---------------------------------------------------------------------------
# Help + setup
# ---------------------------------------------------------------------------


def test_help_lists_all_subcommands_when_invoked_without_args(runner: CliRunner) -> None:
    """Running ``code-indexer`` with no args should print full help."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in (
        "setup",
        "serve",
        "start",
        "stop",
        "status",
        "index",
        "reindex",
        "list",
        "search",
        "symbol",
        "callers",
        "callees",
        "bundle",
        "explore",
        "remove",
    ):
        assert cmd in result.stdout


def test_setup_writes_config_when_run_non_interactively(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``setup --non-interactive`` should write a TOML file with defaults."""
    target = tmp_path / "alt" / "config.toml"
    monkeypatch.setattr(cli_config, "DEFAULT_CONFIG_PATH", target)
    monkeypatch.setattr(cli_config, "DEFAULT_DATA_DIR", target.parent)
    result = runner.invoke(app, ["setup", "--non-interactive"])
    assert result.exit_code == 0, result.stdout
    assert target.is_file()
    body = target.read_text(encoding="utf-8")
    assert "[server]" in body
    assert 'backend = "local"' in body


# ---------------------------------------------------------------------------
# status / list / search
# ---------------------------------------------------------------------------


@respx.mock(base_url=BASE_URL)
def test_status_prints_table_when_health_returns_ok(
    respx_mock: respx.MockRouter, runner: CliRunner
) -> None:
    """``status`` should render the indexed-repos table on healthy /health."""
    respx_mock.get("/health").mock(
        return_value=httpx.Response(200, json={"status": "ok", "indexed_repos": ["forge"]})
    )
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.stdout
    assert "ok" in result.stdout
    assert "forge" in result.stdout


@respx.mock(base_url=BASE_URL)
def test_status_exits_nonzero_when_service_unreachable(
    respx_mock: respx.MockRouter, runner: CliRunner
) -> None:
    """``status`` should exit non-zero when the service is unreachable."""
    respx_mock.get("/health").mock(side_effect=httpx.ConnectError("refused"))
    result = runner.invoke(app, ["status"])
    assert result.exit_code != 0


@respx.mock(base_url=BASE_URL)
def test_list_renders_repo_table_when_repos_are_indexed(
    respx_mock: respx.MockRouter, runner: CliRunner
) -> None:
    """``list`` should render slug, status, path for each repo."""
    respx_mock.get("/repos").mock(
        return_value=httpx.Response(
            200,
            json={
                "repos": [
                    {
                        "slug": "forge",
                        "status": "fresh",
                        "repo_path": "/repos/forge",
                        "last_indexed_at": "2026-05-14T00:00:00Z",
                    }
                ]
            },
        )
    )
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0, result.stdout
    assert "forge" in result.stdout
    assert "fresh" in result.stdout


@respx.mock(base_url=BASE_URL)
def test_search_renders_results_when_semantic_search_returns_hits(
    respx_mock: respx.MockRouter, runner: CliRunner
) -> None:
    """``search`` should render a score/symbol row for each hit.

    The real ``GET /search/semantic`` returns ``{symbol, score, type}`` per
    hit (NOT qualified_name/file_path/line_number — those are what the
    /symbols/* endpoints return). Keep this mock faithful to that contract
    so it actually guards the CLI's column rendering against regressions.
    """
    respx_mock.get("/search/semantic").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "score": 0.91,
                        "symbol": "app.auth.login",
                        "type": "function",
                    }
                ]
            },
        )
    )
    result = runner.invoke(app, ["search", "auth", "-k", "5"])
    assert result.exit_code == 0, result.stdout
    assert "app.auth.login" in result.stdout
    assert "0.910" in result.stdout


# ---------------------------------------------------------------------------
# --json machine-readable contract (global flag, must precede the subcommand)
# ---------------------------------------------------------------------------


@respx.mock(base_url=BASE_URL)
def test_search_emits_parseable_json_when_json_flag_set(
    respx_mock: respx.MockRouter, runner: CliRunner
) -> None:
    """``--json search`` should write exactly one JSON document to stdout.

    This is the harness contract: stdout is machine-parseable, no Rich table.
    """
    respx_mock.get("/search/semantic").mock(
        return_value=httpx.Response(
            200,
            json={"results": [{"score": 0.91, "symbol": "app.auth.login", "type": "function"}]},
        )
    )
    result = runner.invoke(app, ["--json", "search", "auth", "-k", "5"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["results"][0]["symbol"] == "app.auth.login"
    assert payload["results"][0]["score"] == 0.91


@respx.mock(base_url=BASE_URL)
def test_list_emits_parseable_json_when_json_flag_set(
    respx_mock: respx.MockRouter, runner: CliRunner
) -> None:
    """``--json list`` should emit the raw /repos payload as JSON."""
    respx_mock.get("/repos").mock(
        return_value=httpx.Response(
            200,
            json={"repos": [{"slug": "forge", "status": "fresh", "repo_path": "/repos/forge"}]},
        )
    )
    result = runner.invoke(app, ["--json", "list"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["repos"][0]["slug"] == "forge"


@respx.mock(base_url=BASE_URL)
def test_status_emits_parseable_json_with_daemon_keys_when_json_flag_set(
    respx_mock: respx.MockRouter, runner: CliRunner
) -> None:
    """``--json status`` folds daemon liveness into the health JSON on stdout."""
    respx_mock.get("/health").mock(
        return_value=httpx.Response(200, json={"status": "ok", "indexed_repos": ["forge"]})
    )
    result = runner.invoke(app, ["--json", "status"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["indexed_repos"] == ["forge"]
    assert "alive" in payload and "daemon_pid" in payload


# ---------------------------------------------------------------------------
# symbol / callers / callees
# ---------------------------------------------------------------------------


@respx.mock(base_url=BASE_URL)
def test_callers_renders_call_sites_when_endpoint_returns_results(
    respx_mock: respx.MockRouter, runner: CliRunner
) -> None:
    """``callers`` should render every call-site row in the response."""
    respx_mock.get("/symbols/app.auth.login/callers").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "qualified_name": "app.api.handler",
                        "file_path": "app/api.py",
                        "line_number": 10,
                    }
                ]
            },
        )
    )
    result = runner.invoke(app, ["callers", "app.auth.login"])
    assert result.exit_code == 0, result.stdout
    assert "app.api.handler" in result.stdout


@respx.mock(base_url=BASE_URL)
def test_symbol_exits_with_message_when_endpoint_returns_404(
    respx_mock: respx.MockRouter, runner: CliRunner
) -> None:
    """``symbol`` should bail with a friendly message on 404."""
    respx_mock.get("/symbols/missing.fn").mock(
        return_value=httpx.Response(404, json={"detail": "not found"})
    )
    result = runner.invoke(app, ["symbol", "missing.fn"])
    assert result.exit_code != 0
    assert "Symbol not found" in result.stdout


# ---------------------------------------------------------------------------
# index — happy path with poll loop
# ---------------------------------------------------------------------------


@respx.mock(base_url=BASE_URL)
def test_index_polls_until_done_when_job_completes(
    respx_mock: respx.MockRouter, runner: CliRunner, tmp_path: Path
) -> None:
    """``index`` should POST /index then poll /status until terminal."""
    target = tmp_path / "repo"
    target.mkdir()
    respx_mock.post("/index").mock(
        return_value=httpx.Response(202, json={"job_id": "job-1"})
    )
    statuses = [
        httpx.Response(
            200,
            json={
                "status": "running",
                "phase": "parsing",
                "progress_pct": 30.0,
                "node_count": 0,
            },
        ),
        httpx.Response(
            200,
            json={
                "status": "done",
                "phase": "done",
                "progress_pct": 100.0,
                "node_count": 100,
                "rel_count": 50,
            },
        ),
    ]
    respx_mock.get("/index/job-1/status").mock(side_effect=statuses)
    result = runner.invoke(app, ["index", str(target)])
    assert result.exit_code == 0, result.stdout
    assert "Job started" in result.stdout
    assert "Done" in result.stdout
    # Job status reports the relationship count as ``rel_count`` (not
    # ``relationship_count``) — guard the human render against that regression.
    assert "relationships=50" in result.stdout


@respx.mock(base_url=BASE_URL)
def test_index_exits_nonzero_when_path_is_not_a_directory(
    respx_mock: respx.MockRouter, runner: CliRunner, tmp_path: Path
) -> None:
    """``index`` should reject paths that don't exist on disk."""
    missing = tmp_path / "ghost"
    result = runner.invoke(app, ["index", str(missing)])
    assert result.exit_code != 0
    assert "Not a directory" in result.stdout


# ---------------------------------------------------------------------------
# bundle / remove
# ---------------------------------------------------------------------------


@respx.mock(base_url=BASE_URL)
def test_bundle_prints_json_when_endpoint_returns_payload(
    respx_mock: respx.MockRouter, runner: CliRunner, tmp_path: Path
) -> None:
    """``bundle`` should print the JSON payload to stdout."""
    repo = tmp_path / "repo"
    repo.mkdir()
    payload = {
        "symbols": ["app.foo"],
        "source_snippets": {"app.foo": "def foo(): ..."},
        "call_graph": {},
        "total_tokens": 10,
    }
    respx_mock.post("/context-bundle").mock(
        return_value=httpx.Response(200, json=payload)
    )
    result = runner.invoke(app, ["bundle", "add login", "--repo", str(repo)])
    assert result.exit_code == 0, result.stdout
    assert '"symbols"' in result.stdout
    assert "app.foo" in result.stdout


@respx.mock(base_url=BASE_URL)
def test_remove_invokes_delete_when_yes_flag_supplied(
    respx_mock: respx.MockRouter, runner: CliRunner
) -> None:
    """``remove --yes`` should call DELETE /index/<slug> without prompting."""
    route = respx_mock.delete("/index/forge").mock(
        return_value=httpx.Response(200, json={"deleted": True})
    )
    result = runner.invoke(app, ["remove", "forge", "--yes"])
    assert result.exit_code == 0, result.stdout
    assert route.called
    assert "Deleted" in result.stdout
