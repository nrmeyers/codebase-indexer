"""Prometheus instrumentation for the Code Indexer Service.

Per ``.planning/phase-plans/PHASE_4_GRAFANA.md`` §3.1 (metric taxonomy)
and §4 (FastAPI integration). Exposes ``/metrics`` in Prometheus
exposition format when ``METRICS_ENABLED=true``.

Cardinality control (plan §3.4):
    * ``repo_name`` is capped to a top-N sliding window via
      ``_clamp_repo_label`` — repos beyond the window collapse to
      ``"other"``. ``METRICS_TOP_N_REPOS`` configures N (default 20).
    * ``job_id`` never appears as a counter / histogram label (would be
      unbounded). It is only attached to gauges that get explicitly
      cleaned up on terminal status.

The module degrades gracefully: when ``METRICS_ENABLED=false`` the
exporter calls become no-ops and ``/metrics`` is not registered. This
keeps the test suite fast and lets us roll out per-environment.
"""
from __future__ import annotations

import asyncio
import logging
import os
from collections import deque
from dataclasses import dataclass
from threading import Lock

from fastapi import FastAPI

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy / optional imports — fail-soft so unit tests can import this module
# without prometheus_client installed.
# ---------------------------------------------------------------------------
try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
    )

    _HAS_PROM = True
except Exception:  # pragma: no cover - dep should always be present in prod
    _HAS_PROM = False


# ---------------------------------------------------------------------------
# Histogram buckets (plan §3.1)
# ---------------------------------------------------------------------------

# Search latency: spans 5 ms (cache hit) → 10 s (long tail rerank).
_SEARCH_BUCKETS = (0.005, 0.025, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

# Index job duration: spans 1 s (tiny repo) → 1 h (large repo full reindex).
_INDEX_BUCKETS = (1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0)


# ---------------------------------------------------------------------------
# Singleton metrics — populated by ``setup_metrics`` so the import path is
# safe when prometheus_client is unavailable.
# ---------------------------------------------------------------------------
_REGISTRY: "CollectorRegistry | None" = None
_search_duration: "Histogram | None" = None
_search_requests: "Counter | None" = None
_index_job_duration: "Histogram | None" = None
_index_jobs_total: "Counter | None" = None
_index_job_progress: "Gauge | None" = None
_lm_studio_up: "Gauge | None" = None
_lm_studio_can_rerank: "Gauge | None" = None
_embeddings_count: "Gauge | None" = None
_disk_bytes: "Gauge | None" = None
_jobs_active: "Gauge | None" = None
_jobs_dedupe_409: "Counter | None" = None
_query_rewriter_applied: "Counter | None" = None
_rerank_outcome: "Counter | None" = None

# Phase 5 — realtime watcher metrics
_watch_active_repos: "Gauge | None" = None
_watch_events_total: "Counter | None" = None
_watch_partial_duration: "Histogram | None" = None
_watch_partial_files: "Histogram | None" = None
_watch_inotify_failures: "Counter | None" = None

_state_lock = Lock()
_initialised = False


# ---------------------------------------------------------------------------
# Top-N repo label cap (plan §3.4 / §5)
# ---------------------------------------------------------------------------
@dataclass
class _TopNCap:
    n: int
    window: deque[str]
    seen: dict[str, int]

    def observe(self, name: str) -> str:
        self.window.append(name)
        if len(self.window) > 1000:
            old = self.window.popleft()
            self.seen[old] = max(0, self.seen.get(old, 0) - 1)
            if self.seen[old] == 0:
                self.seen.pop(old, None)
        self.seen[name] = self.seen.get(name, 0) + 1
        # Top-N by frequency
        top = sorted(self.seen.items(), key=lambda kv: kv[1], reverse=True)[: self.n]
        top_names = {n for n, _ in top}
        return name if name in top_names else "other"


_repo_cap: _TopNCap | None = None


def _clamp_repo_label(name: str) -> str:
    """Return ``name`` if it's in the top-N hot set, else ``"other"``."""
    global _repo_cap
    if _repo_cap is None:
        n = int(os.environ.get("METRICS_TOP_N_REPOS", "20"))
        _repo_cap = _TopNCap(n=n, window=deque(), seen={})
    return _repo_cap.observe(name)


# ---------------------------------------------------------------------------
# Public API — fail-soft no-ops when metrics are disabled.
# ---------------------------------------------------------------------------
def is_enabled() -> bool:
    return _initialised and _HAS_PROM


def record_search(endpoint: str, *, reranked: bool, duration_seconds: float, status_code: int) -> None:
    """Wrapper called from search routers."""
    if not is_enabled():
        return
    _search_duration.labels(endpoint=endpoint, reranked=str(reranked).lower()).observe(duration_seconds)  # type: ignore[union-attr]
    _search_requests.labels(endpoint=endpoint, status_code=str(status_code)).inc()  # type: ignore[union-attr]


def record_index_phase(phase: str, duration_seconds: float) -> None:
    if not is_enabled():
        return
    _index_job_duration.labels(phase=phase).observe(duration_seconds)  # type: ignore[union-attr]


def record_index_terminal(status: str, kind: str = "index") -> None:
    if not is_enabled():
        return
    _index_jobs_total.labels(status=status, kind=kind).inc()  # type: ignore[union-attr]


def record_dedupe_409() -> None:
    if not is_enabled():
        return
    _jobs_dedupe_409.inc()  # type: ignore[union-attr]


def record_query_rewriter(intent: str, outcome: str) -> None:
    """Record one call into ``/search/semantic``'s query-rewriter stage.

    Args:
        intent: The query category — typically ``"semantic"``. Reserved
            label dimension for future routing variants (e.g. context-bundle
            queries that hit a different rewriter shape).
        outcome: One of ``"applied"``, ``"skip-short"``,
            ``"skip-symbol-like"``, ``"skip-overstrip"``. See
            ``app/routers/search.py:_rewrite_descriptive_query``.

    A/B observability lets us see the rewriter's hit-rate live without
    changing traffic. ``rate(forge_indexer_query_rewriter_applied_total
    {outcome="applied"}[5m]) / rate(...{}[5m])`` is the headline metric.
    """
    if not is_enabled():
        return
    _query_rewriter_applied.labels(intent=intent, outcome=outcome).inc()  # type: ignore[union-attr]


def record_rerank_outcome(outcome: str) -> None:
    """Record one observation against the LLM-rerank stage.

    Args:
        outcome: One of ``"applied"``, ``"skip-empty-input"``,
            ``"skip-unavailable"``, ``"skip-deadline"``,
            ``"skip-empty-response"``, ``"skip-parse-error"``. See
            ``app/services/reranker.py:rerank``.

    Headline metric for "is rerank actually helping users?":
        rate(forge_indexer_rerank_outcome_total{outcome="applied"}[5m])
        / rate(forge_indexer_rerank_outcome_total[5m])

    Cardinality is bounded at 6 outcomes — small label set safe for
    high-cardinality scrape intervals.
    """
    if not is_enabled():
        return
    _rerank_outcome.labels(outcome=outcome).inc()  # type: ignore[union-attr]


def set_jobs_active(kind: str, value: int) -> None:
    if not is_enabled():
        return
    _jobs_active.labels(kind=kind).set(value)  # type: ignore[union-attr]


def set_lm_studio(up: bool, can_rerank: bool) -> None:
    if not is_enabled():
        return
    _lm_studio_up.set(1 if up else 0)  # type: ignore[union-attr]
    _lm_studio_can_rerank.set(1 if can_rerank else 0)  # type: ignore[union-attr]


def set_embeddings_count(repo_name: str, count: int) -> None:
    if not is_enabled():
        return
    _embeddings_count.labels(repo_name=_clamp_repo_label(repo_name)).set(count)  # type: ignore[union-attr]


def set_disk_bytes(path_label: str, bytes_value: int) -> None:
    if not is_enabled():
        return
    _disk_bytes.labels(path=path_label).set(bytes_value)  # type: ignore[union-attr]


def update_index_progress_gauge(job_id: str, seconds_since_progress: float) -> None:
    if not is_enabled():
        return
    _index_job_progress.labels(job_id=job_id).set(seconds_since_progress)  # type: ignore[union-attr]


def clear_index_progress_gauge(job_id: str) -> None:
    if not is_enabled():
        return
    try:
        _index_job_progress.remove(job_id)  # type: ignore[union-attr]
    except KeyError:
        pass


# Phase 5 — realtime watcher helpers
def set_watch_active_repos(count: int) -> None:
    """Update the gauge tracking the number of active watchers."""
    if not is_enabled():
        return
    _watch_active_repos.set(count)  # type: ignore[union-attr]


def record_watch_event(result: str) -> None:
    """Increment the watch-event counter.

    Args:
        result: One of ``dispatched``, ``filtered``, or ``coalesced``.
    """
    if not is_enabled():
        return
    _watch_events_total.labels(result=result).inc()  # type: ignore[union-attr]


def record_watch_partial_duration(duration_seconds: float, terminal_status: str) -> None:
    """Observe a partial-index run duration.

    Args:
        duration_seconds: Wall-clock seconds for the full partial run.
        terminal_status: ``done`` | ``failed`` | ``cancelled`` | ``noop``.
    """
    if not is_enabled():
        return
    _watch_partial_duration.labels(terminal_status=terminal_status).observe(duration_seconds)  # type: ignore[union-attr]


def record_watch_partial_files(dirty_count: int) -> None:
    """Observe the number of dirty files in one debounced batch."""
    if not is_enabled():
        return
    _watch_partial_files.observe(dirty_count)  # type: ignore[union-attr]


def record_watch_inotify_failure(reason: str) -> None:
    """Increment the inotify-failure counter.

    Args:
        reason: One of ``max_watches``, ``permission``, or ``other``.
    """
    if not is_enabled():
        return
    _watch_inotify_failures.labels(reason=reason).inc()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Setup / shutdown
# ---------------------------------------------------------------------------
def setup_metrics(app: FastAPI) -> None:
    """Mount /metrics and start auto-instrumentation. Idempotent."""
    global _REGISTRY, _initialised
    global _search_duration, _search_requests
    global _index_job_duration, _index_jobs_total, _index_job_progress
    global _lm_studio_up, _lm_studio_can_rerank
    global _embeddings_count, _disk_bytes
    global _jobs_active, _jobs_dedupe_409
    global _query_rewriter_applied, _rerank_outcome
    global _watch_active_repos, _watch_events_total
    global _watch_partial_duration, _watch_partial_files, _watch_inotify_failures

    enabled = os.environ.get("METRICS_ENABLED", "true").lower() in ("1", "true", "yes", "on")
    if not enabled:
        logger.info("metrics disabled via METRICS_ENABLED env var; /metrics not mounted")
        return
    if not _HAS_PROM:
        logger.warning("prometheus_client not installed; metrics disabled")
        return

    with _state_lock:
        if _initialised:
            return

        _REGISTRY = CollectorRegistry()

        _search_duration = Histogram(
            "forge_indexer_search_duration_seconds",
            "Wall-clock time per /search/* request, by endpoint and rerank status.",
            labelnames=("endpoint", "reranked"),
            buckets=_SEARCH_BUCKETS,
        )
        _search_requests = Counter(
            "forge_indexer_search_requests_total",
            "Total /search/* requests by endpoint and HTTP status code.",
            labelnames=("endpoint", "status_code"),
        )
        _index_job_duration = Histogram(
            "forge_indexer_index_job_duration_seconds",
            "Per-phase duration of an index job (parse, embed, pagerank, finalize).",
            labelnames=("phase",),
            buckets=_INDEX_BUCKETS,
        )
        _index_jobs_total = Counter(
            "forge_indexer_index_jobs_total",
            "Index job terminal-state counter.",
            labelnames=("status", "kind"),
        )
        _index_job_progress = Gauge(
            "forge_indexer_index_job_progress_seconds",
            "Seconds since the last progress event for an active index job.",
            labelnames=("job_id",),
        )
        _lm_studio_up = Gauge(
            "forge_indexer_lm_studio_up",
            "1 if LM Studio adapter responded to its last health probe, else 0.",
        )
        _lm_studio_can_rerank = Gauge(
            "forge_indexer_lm_studio_can_rerank",
            "1 if LM Studio has a chat model loaded for rerank, else 0.",
        )
        _embeddings_count = Gauge(
            "forge_indexer_embeddings_count",
            "Indexed-symbol count per repo (sampled from DuckDB).",
            labelnames=("repo_name",),
        )
        _disk_bytes = Gauge(
            "forge_indexer_disk_bytes",
            "Bytes used per persistent path (cgr|jobs|audit).",
            labelnames=("path",),
        )
        _jobs_active = Gauge(
            "forge_indexer_jobs_active",
            "Currently-active job count by kind.",
            labelnames=("kind",),
        )
        _jobs_dedupe_409 = Counter(
            "forge_indexer_jobs_dedupe_409_total",
            "POST /index requests rejected as duplicates of an active job.",
        )
        _query_rewriter_applied = Counter(
            "forge_indexer_query_rewriter_applied_total",
            "Query-rewriter outcomes per /search/semantic call. "
            "outcome ∈ {applied, skip-short, skip-symbol-like, skip-overstrip}.",
            labelnames=("intent", "outcome"),
        )
        _rerank_outcome = Counter(
            "forge_indexer_rerank_outcome_total",
            "LLM-rerank outcomes per ?rerank=true call. outcome ∈ "
            "{applied, skip-empty-input, skip-unavailable, skip-deadline, "
            "skip-empty-response, skip-parse-error}.",
            labelnames=("outcome",),
        )

        # Phase 5 — realtime watcher metrics (plan §9)
        _WATCH_PARTIAL_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        _watch_active_repos = Gauge(
            "forge_indexer_watch_active_repos",
            "Number of repos currently being watched by the file-watcher.",
        )
        _watch_events_total = Counter(
            "forge_indexer_watch_events_total",
            "FS events received by the watcher; result ∈ {dispatched, filtered, coalesced}.",
            labelnames=("result",),
        )
        _watch_partial_duration = Histogram(
            "forge_indexer_watch_partial_duration_seconds",
            "Wall-clock seconds per partial-index run.",
            labelnames=("terminal_status",),
            buckets=_WATCH_PARTIAL_BUCKETS,
        )
        _watch_partial_files = Histogram(
            "forge_indexer_watch_partial_files",
            "Number of dirty files per debounced batch.",
            buckets=(1, 2, 5, 10, 20, 50, 100),
        )
        _watch_inotify_failures = Counter(
            "forge_indexer_watch_inotify_failures_total",
            "Observer.schedule failures; reason ∈ {max_watches, permission, other}.",
            labelnames=("reason",),
        )

        # HTTP middleware via prometheus-fastapi-instrumentator + the
        # default REGISTRY so our explicit metrics above ride along.
        metrics_path = os.environ.get("METRICS_PATH", "/metrics")
        try:
            from prometheus_fastapi_instrumentator import Instrumentator

            (
                Instrumentator(
                    should_group_status_codes=False,
                    excluded_handlers=["/health", metrics_path],
                )
                .instrument(app, metric_namespace="forge_indexer")
                .expose(app, endpoint=metrics_path, include_in_schema=False)
            )
            logger.info("metrics: mounted %s (with HTTP auto-instrumentation)", metrics_path)
        except Exception as e:  # pragma: no cover - environmental
            # Fallback: expose the default REGISTRY without HTTP middleware.
            logger.warning("metrics: instrumentator unavailable, mounting bare /metrics: %s", e)
            try:
                from prometheus_client import (
                    CONTENT_TYPE_LATEST,
                    generate_latest,
                )
                from fastapi import Response

                @app.get(metrics_path, include_in_schema=False)
                def _metrics_endpoint() -> Response:
                    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
            except Exception as e2:  # pragma: no cover
                logger.error("metrics: bare /metrics fallback also failed: %s", e2)

        _initialised = True


async def start_background_collectors(*, lm_studio_health_fn=None, cgr_data_dir: str | None = None) -> None:
    """Polls LM Studio + disk usage every 30 s.

    Parameters are injected so this module stays decoupled from the
    rest of the service. Pass ``lm_studio_health_fn`` returning
    ``(up: bool, can_rerank: bool)``.
    """
    if not is_enabled():
        return
    cgr_data_dir = cgr_data_dir or settings.CGR_DATA_DIR
    while True:
        try:
            if lm_studio_health_fn is not None:
                try:
                    result = lm_studio_health_fn()
                    if asyncio.iscoroutine(result):
                        result = await result
                    up, can_rerank = result if isinstance(result, tuple) else (bool(result), bool(result))
                    set_lm_studio(up, can_rerank)
                except Exception as e:
                    logger.debug("lm-studio health probe failed: %s", e)
                    set_lm_studio(False, False)

            try:
                if os.path.isdir(cgr_data_dir):
                    total = sum(
                        os.path.getsize(os.path.join(dp, f))
                        for dp, _, files in os.walk(cgr_data_dir)
                        for f in files
                        if os.path.isfile(os.path.join(dp, f))
                    )
                    set_disk_bytes("cgr", total)
            except Exception as e:
                logger.debug("disk-usage probe failed: %s", e)
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.warning("metrics background collector error: %s", e)
        await asyncio.sleep(30)
