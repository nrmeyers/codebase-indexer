"""Env-var parsing helpers shared by all embedder backends.

Every backend's ``from_env`` classmethod previously open-coded the same
try/except ``int(os.environ.get(...))`` pattern five times over; one
helper keeps the shape consistent and the failure mode (fall back to the
default, never raise) uniform across backends.
"""
from __future__ import annotations

import os


def env_int(name: str, default: int) -> int:
    """Read ``os.environ[name]`` as an int; fall back to ``default``.

    Unset, empty, or non-integer values silently fall back so a bad env
    var never blocks startup.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    """Read ``os.environ[name]`` as a float; fall back to ``default``.

    Same fail-soft contract as :func:`env_int`.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


__all__ = ["env_float", "env_int"]
