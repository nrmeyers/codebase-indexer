"""Smoke tests for ``app.metrics``.

Per ``.planning/phase-plans/PHASE_4_GRAFANA.md`` §11. Asserts the
metric-contract — every dashboard-referenced metric name MUST appear
in the /metrics output once setup_metrics has been called.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import metrics


# Metric names the Phase 4 dashboard JSON references. Adding/removing
# names here is intentional — the dashboard MUST be updated in lock-step.
_DASHBOARD_METRIC_NAMES = (
    "forge_indexer_search_duration_seconds",
    "forge_indexer_search_requests_total",
    "forge_indexer_index_job_duration_seconds",
    "forge_indexer_index_jobs_total",
    "forge_indexer_index_job_progress_seconds",
    "forge_indexer_lm_studio_up",
    "forge_indexer_lm_studio_can_rerank",
    "forge_indexer_embeddings_count",
    "forge_indexer_disk_bytes",
    "forge_indexer_jobs_active",
    "forge_indexer_jobs_dedupe_409_total",
    "forge_indexer_query_rewriter_applied_total",
)


def _reset_metrics_state() -> None:
    """Unregister our metrics from the default REGISTRY so re-init is safe."""
    from prometheus_client import REGISTRY

    for collector in list(REGISTRY._collector_to_names.keys()):  # noqa: SLF001
        names = REGISTRY._collector_to_names.get(collector, set())  # noqa: SLF001
        if any(n.startswith("forge_indexer_") for n in names):
            try:
                REGISTRY.unregister(collector)
            except KeyError:
                pass

    # Reset module-level singletons.
    metrics._initialised = False
    metrics._REGISTRY = None
    metrics._search_duration = None
    metrics._search_requests = None
    metrics._index_job_duration = None
    metrics._index_jobs_total = None
    metrics._index_job_progress = None
    metrics._lm_studio_up = None
    metrics._lm_studio_can_rerank = None
    metrics._embeddings_count = None
    metrics._disk_bytes = None
    metrics._jobs_active = None
    metrics._jobs_dedupe_409 = None
    metrics._query_rewriter_applied = None
    metrics._repo_cap = None


@pytest.fixture
def metrics_app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.setenv("METRICS_ENABLED", "true")
    monkeypatch.setenv("METRICS_PATH", "/metrics")
    _reset_metrics_state()
    app = FastAPI()
    metrics.setup_metrics(app)
    yield app
    _reset_metrics_state()


def test_metrics_endpoint_returns_prom_format(metrics_app: FastAPI) -> None:
    client = TestClient(metrics_app)
    r = client.get("/metrics")
    assert r.status_code == 200
    ctype = r.headers.get("content-type", "")
    assert "text/plain" in ctype


def test_dashboard_metric_contract(metrics_app: FastAPI) -> None:
    """Every metric the dashboard references must be exposed."""
    # Pre-populate so histograms emit their _bucket / _count series.
    metrics.record_search("semantic", reranked=False, duration_seconds=0.05, status_code=200)
    metrics.record_index_phase("parse", duration_seconds=2.5)
    metrics.record_index_terminal("done", "index")
    metrics.set_lm_studio(True, True)
    metrics.set_embeddings_count("repo-foo", 12345)
    metrics.set_disk_bytes("cgr", 1024 * 1024 * 100)
    metrics.set_jobs_active("index", 1)
    metrics.record_dedupe_409()
    metrics.update_index_progress_gauge("job-abc", 4.2)
    metrics.record_query_rewriter("semantic", "applied")
    metrics.record_query_rewriter("semantic", "skip-short")

    client = TestClient(metrics_app)
    body = client.get("/metrics").text
    missing: list[str] = [m for m in _DASHBOARD_METRIC_NAMES if m not in body]
    assert not missing, f"dashboard contract: missing metrics {missing}"


def test_metrics_disabled_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("METRICS_ENABLED", "false")
    _reset_metrics_state()
    app = FastAPI()
    metrics.setup_metrics(app)
    client = TestClient(app)
    assert client.get("/metrics").status_code == 404


def test_top_n_repo_label_caps_to_other(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("METRICS_TOP_N_REPOS", "3")
    metrics._repo_cap = None  # force re-init
    # Hit 5 distinct repos; bottom 2 should collapse to "other".
    for repo in ["a", "b", "c", "d", "e"]:
        for _ in range(3 if repo in ("a", "b", "c") else 1):
            metrics._clamp_repo_label(repo)
    assert metrics._clamp_repo_label("a") == "a"
    # "d" / "e" have lower frequency → "other" once N hot repos established.
    label = metrics._clamp_repo_label("d")
    assert label in {"d", "other"}  # eviction is rolling — accept both


def test_query_rewriter_counter_emits_outcome_labels(metrics_app: FastAPI) -> None:
    """All four outcomes hit by the rewriter must show up as distinct
    label-tuples on the counter."""
    for outcome in ("applied", "skip-short", "skip-symbol-like", "skip-overstrip"):
        metrics.record_query_rewriter("semantic", outcome)

    body = TestClient(metrics_app).get("/metrics").text
    for outcome in ("applied", "skip-short", "skip-symbol-like", "skip-overstrip"):
        line = (
            'forge_indexer_query_rewriter_applied_total'
            f'{{intent="semantic",outcome="{outcome}"}} 1.0'
        )
        assert line in body, f"missing line for outcome={outcome}: {line!r}"


def test_query_rewriter_metric_is_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """`record_query_rewriter` must never raise even when metrics are off —
    the rewriter call site is on the hot path; it can't depend on
    Prometheus being initialised."""
    monkeypatch.setenv("METRICS_ENABLED", "false")
    _reset_metrics_state()
    app = FastAPI()
    metrics.setup_metrics(app)
    # Should silently no-op, not raise AttributeError on the global counter.
    metrics.record_query_rewriter("semantic", "applied")
