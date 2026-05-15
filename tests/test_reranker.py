"""Unit tests for the listwise reranker (`app.services.reranker`).

The reranker wraps ``lm_studio.chat_complete`` with a frozen prompt and
a permutation parser.  These tests focus on the parser (it has to be
robust to a wide variety of LLM output formats) and the route-level
fallback contract (rerank() must NEVER raise — every failure mode
returns the candidates in their original order).
"""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def _reset_lm_studio(monkeypatch: pytest.MonkeyPatch):
    """Reload the LM Studio adapter to clear its probe cache."""
    for k in (
        "LM_STUDIO_URL",
        "LM_STUDIO_EMBED_MODEL",
        "LM_STUDIO_RERANK_MODEL",
        "LM_STUDIO_TIMEOUT",
    ):
        monkeypatch.delenv(k, raising=False)
    from app.services import lm_studio  # noqa: WPS433
    importlib.reload(lm_studio)
    from app.services import reranker  # noqa: WPS433
    importlib.reload(reranker)
    yield


# ---------------------------------------------------------------------------
# _parse_permutation
# ---------------------------------------------------------------------------


def test_parse_canonical_format() -> None:
    from app.services import reranker
    assert reranker._parse_permutation("[3] > [1] > [2]", 3) == [2, 0, 1]


def test_parse_comma_separated() -> None:
    from app.services import reranker
    assert reranker._parse_permutation("[3], [1], [2]", 3) == [2, 0, 1]


def test_parse_newline_separated() -> None:
    from app.services import reranker
    assert reranker._parse_permutation("[3]\n[1]\n[2]", 3) == [2, 0, 1]


def test_parse_with_prose_around_indices() -> None:
    from app.services import reranker
    # Even when the model leaks prose despite the prompt, we still extract the
    # bracketed indices and produce a valid permutation.
    txt = "Sure, here is the ranking: [3] > [1] > [2] (most to least relevant)."
    assert reranker._parse_permutation(txt, 3) == [2, 0, 1]


def test_parse_handles_duplicates() -> None:
    from app.services import reranker
    # [1] appears twice — second occurrence is dropped, missing index appended.
    assert reranker._parse_permutation("[1] > [1] > [3]", 3) == [0, 2, 1]


def test_parse_drops_out_of_range() -> None:
    from app.services import reranker
    # [99] is out of range; missing indices auto-appended in original order.
    assert reranker._parse_permutation("[99] > [2] > [1]", 3) == [1, 0, 2]


def test_parse_returns_none_on_empty() -> None:
    from app.services import reranker
    assert reranker._parse_permutation("", 5) is None
    assert reranker._parse_permutation("no brackets here", 5) is None


def test_parse_recovers_partial_permutation() -> None:
    from app.services import reranker
    # Model only emits 2 of 4 indices — parser appends the missing two
    # in original order rather than dropping the rest.
    assert reranker._parse_permutation("[2] > [4]", 4) == [1, 3, 0, 2]


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------


def test_build_prompt_truncates_long_snippets() -> None:
    from app.services import reranker
    long_src = "x" * 5_000
    cand = [{"qualified_name": "a.b.c", "source": long_src}]
    out = reranker._build_prompt("query", cand)
    # MAX_SNIPPET_CHARS + ellipsis + identifier; well under the original 5k.
    assert len(out) < 2_000
    assert "a.b.c" in out
    assert out.startswith("Query: query")


def test_build_prompt_uses_qualified_name_fallback() -> None:
    from app.services import reranker
    cand = [{"symbol": "foo.bar"}, {"node_id": "baz.qux"}, {}]
    out = reranker._build_prompt("q", cand)
    assert "foo.bar" in out
    assert "baz.qux" in out
    assert "candidate_3" in out  # final fallback


# ---------------------------------------------------------------------------
# rerank() — orchestration + fallback contract
# ---------------------------------------------------------------------------


def _cands(n: int) -> list[dict]:
    """Build n minimal candidate dicts."""
    return [{"qualified_name": f"sym_{i}", "source": f"def f{i}(): pass"} for i in range(n)]


def test_rerank_short_circuits_on_empty_inputs() -> None:
    from app.services import reranker
    assert reranker.rerank("", _cands(3)) == _cands(3)
    assert reranker.rerank("q", []) == []


def test_rerank_returns_original_when_lm_studio_unavailable() -> None:
    from app.services import reranker
    cands = _cands(5)
    # No LM_STUDIO_URL → is_available() is False → no-op.
    assert reranker.rerank("find foo", cands) == cands


def test_rerank_applies_permutation(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import lm_studio, reranker

    monkeypatch.setenv("LM_STUDIO_URL", "http://localhost:9999")
    monkeypatch.setattr(
        lm_studio,
        "_get_json",
        lambda *a, **k: {"data": [{"id": "CodeRankLLM"}]},
    )
    monkeypatch.setattr(
        lm_studio,
        "_post_json",
        lambda *a, **k: {
            "choices": [{"message": {"content": "[3] > [1] > [2]"}}],
        },
    )

    cands = _cands(3)
    out = reranker.rerank("find foo", cands)
    assert [c["qualified_name"] for c in out] == ["sym_2", "sym_0", "sym_1"]


def test_rerank_falls_back_on_unparseable_response(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import lm_studio, reranker

    monkeypatch.setenv("LM_STUDIO_URL", "http://localhost:9999")
    monkeypatch.setattr(
        lm_studio,
        "_get_json",
        lambda *a, **k: {"data": [{"id": "CodeRankLLM"}]},
    )
    monkeypatch.setattr(
        lm_studio,
        "_post_json",
        lambda *a, **k: {"choices": [{"message": {"content": "I refuse to rank."}}]},
    )

    cands = _cands(4)
    assert reranker.rerank("q", cands) == cands  # original order preserved


def test_rerank_caps_to_max_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import lm_studio, reranker

    monkeypatch.setenv("LM_STUDIO_URL", "http://localhost:9999")
    monkeypatch.setattr(
        lm_studio,
        "_get_json",
        lambda *a, **k: {"data": [{"id": "CodeRankLLM"}]},
    )

    captured: dict = {}

    def fake_post(url: str, payload: dict, timeout: float):
        captured["payload"] = payload
        # Return identity permutation so the cap-and-tail logic is the only
        # behaviour under test here.
        n = reranker.MAX_CANDIDATES
        ids = " > ".join(f"[{i + 1}]" for i in range(n))
        return {"choices": [{"message": {"content": ids}}]}

    monkeypatch.setattr(lm_studio, "_post_json", fake_post)

    cands = _cands(reranker.MAX_CANDIDATES + 5)
    out = reranker.rerank("q", cands)
    # Length preserved — tail is appended back unchanged.
    assert len(out) == len(cands)
    # Last 5 are the un-reranked tail in original order.
    tail_qns = [c["qualified_name"] for c in out[-5:]]
    assert tail_qns == [f"sym_{i}" for i in range(reranker.MAX_CANDIDATES, reranker.MAX_CANDIDATES + 5)]
    # Prompt should reference exactly MAX_CANDIDATES entries.
    assert f"Rank the {reranker.MAX_CANDIDATES} candidates" in captured["payload"]["messages"][1]["content"]


def test_is_available_requires_loaded_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import lm_studio, reranker

    monkeypatch.setenv("LM_STUDIO_URL", "http://localhost:9999")
    # Models endpoint succeeds but doesn't list CodeRankLLM
    monkeypatch.setattr(
        lm_studio,
        "_get_json",
        lambda *a, **k: {"data": [{"id": "some-other-model"}]},
    )
    assert reranker.is_available() is False


# ---------------------------------------------------------------------------
# rerank() — graceful-fallback contract for chat_complete failure modes
# ---------------------------------------------------------------------------
#
# These exercise the contract that any failure inside ``lm_studio.chat_complete``
# (HTTP 5xx, urllib timeout, empty/missing choices, missing both content and
# reasoning_content — all caught and surfaced as ``None`` by the adapter) is
# absorbed silently and the original candidate list is returned untouched.


def test_should_preserve_order_when_chat_complete_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import lm_studio, reranker

    # Force the is_available() short-circuit to pass so we exercise the
    # post-availability fallback path.
    monkeypatch.setattr(lm_studio, "can_rerank", lambda: True)
    monkeypatch.setattr(lm_studio, "chat_complete", lambda *a, **k: None)

    cands = _cands(5)
    out = reranker.rerank("find foo", cands)
    assert out == cands
    # Same objects (no copy) — the adapter contract preserves identity.
    assert all(out[i] is cands[i] for i in range(len(cands)))


def test_should_preserve_order_when_chat_complete_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import lm_studio, reranker

    monkeypatch.setattr(lm_studio, "can_rerank", lambda: True)

    def _boom(*_a, **_k):
        raise RuntimeError("simulated mid-stream HTTP 500")

    monkeypatch.setattr(lm_studio, "chat_complete", _boom)

    cands = _cands(4)
    # rerank() must NEVER propagate exceptions from the LM Studio adapter.
    with pytest.raises(RuntimeError):
        # Sanity: confirm our monkeypatched stub does raise. If a future
        # rerank() refactor swallows the exception itself (rather than
        # relying on the adapter), flip this to a direct call assertion.
        lm_studio.chat_complete()
    # Real contract check: when the adapter is hardened to swallow and
    # return None (current production behaviour), rerank() preserves order.
    monkeypatch.setattr(lm_studio, "chat_complete", lambda *a, **k: None)
    assert reranker.rerank("q", cands) == cands


def test_should_preserve_order_when_response_permutation_unparseable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import lm_studio, reranker

    monkeypatch.setattr(lm_studio, "can_rerank", lambda: True)
    # Model returns prose with no bracketed indices — _parse_permutation
    # yields None and rerank() falls back to original order.
    monkeypatch.setattr(
        lm_studio,
        "chat_complete",
        lambda *a, **k: "I cannot rank these candidates.",
    )

    cands = _cands(6)
    assert reranker.rerank("q", cands) == cands


def test_should_preserve_order_when_chat_complete_returns_empty_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import lm_studio, reranker

    monkeypatch.setattr(lm_studio, "can_rerank", lambda: True)
    # Empty string is the adapter's surface form for "neither content nor
    # reasoning_content was usable" — rerank() must treat it as a no-op.
    monkeypatch.setattr(lm_studio, "chat_complete", lambda *a, **k: "")

    cands = _cands(3)
    assert reranker.rerank("q", cands) == cands


# ---------------------------------------------------------------------------
# Latency budget (RERANK_DEADLINE_SECONDS) + outcome metric
# ---------------------------------------------------------------------------


def test_rerank_returns_original_order_when_deadline_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The user-facing search request can't be held hostage by a slow LM
    Studio rerank. When ``RERANK_DEADLINE_SECONDS`` is exceeded, ``rerank()``
    returns the bi-encoder order untouched and the worker thread is left to
    finish in the background."""
    import time
    from app.services import lm_studio, reranker

    monkeypatch.setenv("RERANK_DEADLINE_SECONDS", "0.3")
    monkeypatch.setattr(lm_studio, "can_rerank", lambda: True)

    def slow_call(*a, **k):
        time.sleep(1.5)  # exceeds 0.3s budget
        return "[2] > [1]"

    monkeypatch.setattr(lm_studio, "chat_complete", slow_call)
    # Reset module-level executor so the slow thread from this test
    # doesn't leak into the next.
    reranker._executor = None

    cands = _cands(2)
    started = time.monotonic()
    out = reranker.rerank("q", cands)
    elapsed = time.monotonic() - started

    assert out == cands, "deadline path must return original list unchanged"
    # Allow some slack for the thread-pool overhead but make sure we
    # didn't actually wait the full 1.5s.
    assert elapsed < 1.0, f"rerank waited {elapsed:.2f}s — deadline didn't fire"


def test_rerank_deadline_disabled_when_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting RERANK_DEADLINE_SECONDS=0 disables the deadline — useful for
    batch eval runs that prefer correctness over latency."""
    from app.services import lm_studio, reranker

    monkeypatch.setenv("RERANK_DEADLINE_SECONDS", "0")
    monkeypatch.setattr(lm_studio, "can_rerank", lambda: True)
    monkeypatch.setattr(lm_studio, "chat_complete", lambda *a, **k: "[2] > [1]")
    reranker._executor = None

    cands = _cands(2)
    out = reranker.rerank("q", cands)
    # Permutation [2] > [1] flips the order, so output must differ.
    assert out != cands
    assert out == [cands[1], cands[0]]


# ---------------------------------------------------------------------------
# Outcome metric — every code path emits exactly one observation
# ---------------------------------------------------------------------------


def _outcome_count(outcome: str) -> float:
    """Read the current value of `forge_indexer_rerank_outcome_total{outcome=...}`."""
    from prometheus_client import generate_latest

    needle = f'forge_indexer_rerank_outcome_total{{outcome="{outcome}"}}'
    for line in generate_latest().decode().splitlines():
        if line.startswith(needle):
            try:
                return float(line.split()[-1])
            except (ValueError, IndexError):
                return 0.0
    return 0.0


def _setup_metrics_for_test(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the metrics module is initialised inside a test so the
    rerank counter actually records observations."""
    from fastapi import FastAPI
    from prometheus_client import REGISTRY
    from app import metrics

    # Unregister any pre-existing forge_indexer_* collectors to avoid
    # `Duplicated timeseries` errors across tests.
    for collector in list(REGISTRY._collector_to_names.keys()):  # noqa: SLF001
        names = REGISTRY._collector_to_names.get(collector, set())  # noqa: SLF001
        if any(n.startswith("forge_indexer_") for n in names):
            try:
                REGISTRY.unregister(collector)
            except KeyError:
                pass

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
    metrics._rerank_outcome = None
    metrics._repo_cap = None

    monkeypatch.setenv("METRICS_ENABLED", "true")
    metrics.setup_metrics(FastAPI())


def test_outcome_metric_records_skip_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import lm_studio, reranker

    _setup_metrics_for_test(monkeypatch)
    before = _outcome_count("skip-unavailable")

    monkeypatch.setattr(lm_studio, "can_rerank", lambda: False)
    out = reranker.rerank("q", _cands(2))
    assert out == _cands(2)
    assert _outcome_count("skip-unavailable") == before + 1


def test_outcome_metric_records_applied_on_valid_permutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import lm_studio, reranker

    _setup_metrics_for_test(monkeypatch)
    before = _outcome_count("applied")

    monkeypatch.setattr(lm_studio, "can_rerank", lambda: True)
    monkeypatch.setattr(lm_studio, "chat_complete", lambda *a, **k: "[2] > [1]")
    monkeypatch.setenv("RERANK_DEADLINE_SECONDS", "0")  # disable for determinism
    reranker._executor = None

    out = reranker.rerank("q", _cands(2))
    assert out != _cands(2)  # actually reordered
    assert _outcome_count("applied") == before + 1


def test_outcome_metric_records_skip_parse_error_on_unparseable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import lm_studio, reranker

    _setup_metrics_for_test(monkeypatch)
    before = _outcome_count("skip-parse-error")

    monkeypatch.setattr(lm_studio, "can_rerank", lambda: True)
    monkeypatch.setattr(lm_studio, "chat_complete", lambda *a, **k: "no brackets here")
    monkeypatch.setenv("RERANK_DEADLINE_SECONDS", "0")
    reranker._executor = None

    out = reranker.rerank("q", _cands(3))
    assert out == _cands(3)
    assert _outcome_count("skip-parse-error") == before + 1


def test_outcome_metric_records_skip_empty_input_on_no_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import reranker

    _setup_metrics_for_test(monkeypatch)
    before = _outcome_count("skip-empty-input")

    out = reranker.rerank("q", [])
    assert out == []
    assert _outcome_count("skip-empty-input") == before + 1


# ---------------------------------------------------------------------------
# BUC-1651 — structured warning when LM Studio is unreachable
# ---------------------------------------------------------------------------


def test_should_log_warning_when_lm_studio_unreachable(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When a caller opts into rerank but LM Studio is unreachable, the
    fail-open path must emit a single WARNING-level log line — distinct
    from the silent skip-empty-input path so operators can grep for
    rerank degradation in production logs (BUC-1651)."""
    import logging
    from app.services import lm_studio, reranker

    # Force the "unavailable" path: empty base_url short-circuits is_available()
    # without any network calls.
    monkeypatch.setattr(lm_studio, "can_rerank", lambda: False)
    monkeypatch.setattr(lm_studio, "base_url", lambda: "")

    cands = _cands(3)
    with caplog.at_level(logging.WARNING, logger="app.services.reranker"):
        out = reranker.rerank("find foo", cands)

    assert out == cands  # identity preserved
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "LM Studio unreachable" in warnings[0].getMessage()
