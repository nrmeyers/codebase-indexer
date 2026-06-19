"""User-level CLI configuration for the standalone ``code-indexer`` tool.

Stores per-user defaults at ``${XDG_CONFIG_HOME:-~/.config}/codebase-indexer/
config.toml`` (a legacy ``~/.code-indexer/config.toml`` is still read when
present) so subsequent invocations don't need flags. The on-disk schema is
intentionally minimal:

.. code-block:: toml

    [server]
    base_url = "http://localhost:8003"
    port = 8003

    [embedder]
    backend = "local"

    [paths]
    data_dir = "/home/jane/.local/share/codebase-indexer"

Resolution precedence for the server base URL (highest first):

1. ``--base-url`` CLI flag (handled by callers in :mod:`app.cli.main`)
2. ``CODE_INDEXER_BASE_URL`` env var
3. ``server.base_url`` from the TOML file
4. Hard default ``http://localhost:8003``
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - project pins >=3.12
    import tomli as tomllib  # type: ignore[no-redef]


DEFAULT_BASE_URL = "http://localhost:8003"
DEFAULT_PORT = 8003


def _xdg_config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")


def _xdg_data_home() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")


# User-scoped layout (matches the agentalloy install convention): the TOML
# config lives under ``XDG_CONFIG_HOME``; runtime artefacts (pid, log) live
# under ``XDG_DATA_HOME``. A ``pipx`` / ``uv tool`` install therefore behaves
# like an installed tool from any directory instead of dropping dotfiles into
# the caller's cwd.
DEFAULT_DATA_DIR = _xdg_data_home() / "codebase-indexer"
DEFAULT_CONFIG_PATH = _xdg_config_home() / "codebase-indexer" / "config.toml"
DEFAULT_PID_PATH = DEFAULT_DATA_DIR / "server.pid"
DEFAULT_LOG_PATH = DEFAULT_DATA_DIR / "server.log"

# Backward-compat: a pre-XDG install kept everything under ~/.code-indexer/.
# ``load_config`` falls back to it when no XDG config exists yet.
_LEGACY_CONFIG_PATH = Path.home() / ".code-indexer" / "config.toml"


EmbedderBackend = Literal["local", "sagemaker", "tei"]


@dataclass(slots=True)
class CliConfig:
    """Resolved CLI configuration for a single invocation.

    Attributes:
        base_url: HTTP base URL for the running FastAPI service.
        port: Bind port used by ``serve`` / ``start``.
        embedder_backend: Selected embedder backend identifier.
        data_dir: Root directory for CLI artefacts (pid file, logs).
        config_path: TOML file the config was loaded from (may not exist).
    """

    base_url: str
    port: int
    embedder_backend: EmbedderBackend
    data_dir: Path
    config_path: Path


def load_config(
    config_path: Path | None = None,
    *,
    base_url_override: str | None = None,
) -> CliConfig:
    """Load CLI config from disk, applying env + CLI overrides.

    Args:
        config_path: Optional path to a TOML config file. Defaults to
            ``~/.code-indexer/config.toml``.
        base_url_override: Highest-precedence base URL override, typically
            supplied via the ``--base-url`` flag.

    Returns:
        A populated :class:`CliConfig`. Missing files are tolerated — the
        function falls back to documented defaults when the TOML file does
        not exist.
    """
    if config_path is not None:
        cfg_path = config_path
    elif DEFAULT_CONFIG_PATH.is_file():
        cfg_path = DEFAULT_CONFIG_PATH
    elif _LEGACY_CONFIG_PATH.is_file():
        cfg_path = _LEGACY_CONFIG_PATH
    else:
        cfg_path = DEFAULT_CONFIG_PATH
    data_dir = DEFAULT_DATA_DIR
    base_url = DEFAULT_BASE_URL
    port = DEFAULT_PORT
    backend: EmbedderBackend = "local"

    if cfg_path.is_file():
        with cfg_path.open("rb") as fp:
            raw = tomllib.load(fp)
        server = raw.get("server", {}) or {}
        embedder = raw.get("embedder", {}) or {}
        paths = raw.get("paths", {}) or {}
        if isinstance(server.get("base_url"), str):
            base_url = server["base_url"]
        if isinstance(server.get("port"), int):
            port = server["port"]
        if embedder.get("backend") in ("local", "sagemaker", "tei"):
            backend = embedder["backend"]  # type: ignore[assignment]
        if isinstance(paths.get("data_dir"), str):
            data_dir = Path(paths["data_dir"]).expanduser()

    env_url = os.environ.get("CODE_INDEXER_BASE_URL")
    if env_url:
        base_url = env_url
    if base_url_override:
        base_url = base_url_override

    return CliConfig(
        base_url=base_url.rstrip("/"),
        port=port,
        embedder_backend=backend,
        data_dir=data_dir,
        config_path=cfg_path,
    )


def write_config(
    *,
    base_url: str,
    port: int,
    embedder_backend: EmbedderBackend,
    data_dir: Path,
    config_path: Path | None = None,
) -> Path:
    """Persist the supplied CLI defaults to a TOML file.

    Args:
        base_url: Server base URL to record.
        port: Bind port to record.
        embedder_backend: Embedder backend selection.
        data_dir: Root directory for CLI artefacts.
        config_path: Override target path (defaults to
            ``~/.code-indexer/config.toml``).

    Returns:
        The path the configuration was written to.
    """
    cfg_path = config_path or DEFAULT_CONFIG_PATH
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    body = (
        "# Generated by `code-indexer setup`. Hand-edit at will.\n"
        "\n"
        "[server]\n"
        f'base_url = "{base_url.rstrip("/")}"\n'
        f"port = {port}\n"
        "\n"
        "[embedder]\n"
        f'backend = "{embedder_backend}"\n'
        "\n"
        "[paths]\n"
        f'data_dir = "{data_dir}"\n'
    )
    cfg_path.write_text(body, encoding="utf-8")
    return cfg_path
