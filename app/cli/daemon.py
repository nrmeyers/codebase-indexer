"""Lightweight background-daemon helpers for the standalone CLI.

The ``start`` / ``stop`` / ``status`` subcommands need to manage a
backgrounded ``uvicorn`` process without pulling in a full process
supervisor. This module stores a PID file at ``~/.code-indexer/server.pid``
and exposes helpers to spawn, signal, and inspect the daemon.

We deliberately avoid double-forking — the helpers rely on
``subprocess.Popen`` with ``start_new_session=True`` which detaches the
child from the terminal session, which is sufficient for a developer
tool.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def is_port_open(host: str, port: int, *, timeout: float = 0.25) -> bool:
    """Return True when *host:port* accepts a TCP connection.

    Useful as a cheap "is the service up?" probe before falling back to
    parsing ``/health``.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def read_pid(pid_path: Path) -> int | None:
    """Return the PID stored at *pid_path* or ``None`` when absent/invalid."""
    if not pid_path.is_file():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def pid_alive(pid: int) -> bool:
    """Return True when a process with *pid* is still running.

    Uses ``os.kill(pid, 0)`` which is a no-op signal probe — raises
    ``ProcessLookupError`` when the PID is gone.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def spawn_server(
    *,
    port: int,
    pid_path: Path,
    log_path: Path,
    extra_env: dict[str, str] | None = None,
) -> int:
    """Spawn ``uvicorn app.main:app`` in the background.

    Args:
        port: Bind port for uvicorn.
        pid_path: File to write the spawned PID into.
        log_path: File to redirect stdout/stderr into.
        extra_env: Optional environment overrides merged on top of the
            current process environment.

    Returns:
        The PID of the spawned uvicorn process.

    Raises:
        RuntimeError: When the spawned process exits before the port
            becomes reachable (a 5-second window).
    """
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    env.setdefault("PORT", str(port))

    log_fd = log_path.open("ab")
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "info",
    ]
    proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
        cmd,
        stdout=log_fd,
        stderr=log_fd,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        env=env,
    )
    pid_path.write_text(str(proc.pid), encoding="utf-8")

    # Wait up to 5 s for the port to become reachable.
    for _ in range(50):
        if proc.poll() is not None:
            raise RuntimeError(
                f"uvicorn exited early with code {proc.returncode}; see {log_path}"
            )
        if is_port_open("127.0.0.1", port):
            return proc.pid
        time.sleep(0.1)
    return proc.pid


def stop_server(pid_path: Path, *, grace_seconds: float = 5.0) -> bool:
    """Signal the daemon to terminate, escalating to SIGKILL if needed.

    Args:
        pid_path: Path to the PID file written by :func:`spawn_server`.
        grace_seconds: How long to wait for graceful shutdown before
            escalating to SIGKILL.

    Returns:
        True when the daemon was stopped (or was already gone), False
        when the PID file was missing or unparseable.
    """
    pid = read_pid(pid_path)
    if pid is None:
        return False
    if not pid_alive(pid):
        pid_path.unlink(missing_ok=True)
        return True
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pid_path.unlink(missing_ok=True)
        return True

    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            pid_path.unlink(missing_ok=True)
            return True
        time.sleep(0.1)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    pid_path.unlink(missing_ok=True)
    return True
