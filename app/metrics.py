"""Metrics shim — Phase 2 placeholder for Phase 4 instrumentation.

The Phase 4 dashboard plan calls for Prometheus counters / gauges /
histograms attached to job-store transitions. Wiring those calls into
``app/services/jobs_store.py`` and ``app/routers/index.py`` *now*
(while the call sites are open in this PR) makes Phase 4 a body-only
change — flip every ``pass`` below to a real metric and we're done.

All functions are intentionally no-ops. They accept the keyword
arguments the eventual implementation will need so the call sites
stay byte-identical when Phase 4 lands.

Metric names mirror Phase 2 plan §7 verbatim:
    code_indexer_jobs_total                 (Counter)
    code_indexer_jobs_active                (Gauge)
    code_indexer_job_duration_seconds       (Histogram)
    code_indexer_jobs_interrupted_total     (Counter)
    code_indexer_jobs_dedupe_409_total      (Counter)
    code_indexer_jobs_store_write_seconds   (Histogram)
"""
from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator


def jobs_total_inc(*, status: str, kind: str = "index") -> None:
    """Increment ``code_indexer_jobs_total{status,kind}`` (Counter)."""
    _ = (status, kind)


def jobs_active_set(value: int, *, kind: str = "index") -> None:
    """Set ``code_indexer_jobs_active{kind}`` (Gauge)."""
    _ = (value, kind)


def job_duration_observe(
    seconds: float, *, kind: str = "index", terminal_status: str = "done"
) -> None:
    """Observe ``code_indexer_job_duration_seconds{kind,terminal_status}``."""
    _ = (seconds, kind, terminal_status)


def jobs_interrupted_inc(count: int = 1) -> None:
    """Increment ``code_indexer_jobs_interrupted_total`` (Counter)."""
    _ = count


def jobs_dedupe_409_inc() -> None:
    """Increment ``code_indexer_jobs_dedupe_409_total`` (Counter)."""


@contextmanager
def store_write_timer(op: str) -> Iterator[None]:
    """Time a jobs_store write op into ``code_indexer_jobs_store_write_seconds{op}``.

    Use as a context manager around store writes:

        with store_write_timer("create_job"):
            jobs_store.create_job(...)
    """
    _ = op
    yield
