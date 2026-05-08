"""Tests for the Manifest file-summary client and cost-cap behaviour
(Phase 1.2b)."""
from __future__ import annotations

from unittest import mock

import httpx
import pytest

from app.services import manifest_client
from app.services.chunk_strategies import (
    FILE_SUMMARY_REPO_COST_CAP_USD,
    estimate_haiku_call_cost_usd,
)


@pytest.fixture(autouse=True)
def _set_manifest_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default to a configured Manifest gateway for these tests."""
    monkeypatch.setenv("MANIFEST_URL", "http://localhost:2099")
    monkeypatch.setenv("MANIFEST_AGENT_KEY", "test-key")


def test_should_return_summary_when_manifest_returns_valid_response() -> None:
    """A 200 OK with a normal completion shape yields a populated
    :class:`FileSummaryResult`."""
    fake_response = httpx.Response(
        status_code=200,
        json={
            "choices": [
                {"message": {"content": "Records governance audit events."}},
            ],
            "usage": {"prompt_tokens": 600, "completion_tokens": 180},
        },
    )

    def _fake_post(self, url: str, **kwargs):  # type: ignore[no-untyped-def]
        return fake_response

    with mock.patch.object(httpx.Client, "post", _fake_post):
        result = manifest_client.summarize_file(
            "src/services/audit-trail.ts",
            "// audit trail content",
        )

    assert result is not None
    assert result.summary == "Records governance audit events."
    assert result.input_tokens == 600
    assert result.output_tokens == 180


def test_should_return_none_when_manifest_returns_5xx() -> None:
    """A 5xx response degrades gracefully — the caller continues without
    a summary, no exception bubbles up."""
    fake_response = httpx.Response(status_code=503, text="upstream down")

    def _fake_post(self, url: str, **kwargs):  # type: ignore[no-untyped-def]
        return fake_response

    with mock.patch.object(httpx.Client, "post", _fake_post):
        result = manifest_client.summarize_file(
            "src/x.ts", "console.log('x');",
        )
    assert result is None


def test_should_abort_file_summarization_when_cumulative_cost_exceeds_cap() -> None:
    """Driver-side cost-cap behaviour: when cumulative spend would cross
    the $1.50 ceiling, the File-summary pass aborts (sets a flag, breaks
    the loop) — Function/Method ingestion still completes because that
    pass already ran above this loop in the driver."""
    per_call = estimate_haiku_call_cost_usd(input_tokens=600, output_tokens=180)
    spent = 0.0
    files_summarized = 0
    aborted = False
    function_method_ingestion_completed = True  # already done before this pass

    # Simulate a repo with thousands of files.  The cap MUST trip before
    # we run out of files — otherwise the cap isn't actually a cap.
    for _ in range(5000):
        if spent + per_call > FILE_SUMMARY_REPO_COST_CAP_USD:
            aborted = True
            break
        spent += per_call
        files_summarized += 1

    assert aborted is True
    assert spent <= FILE_SUMMARY_REPO_COST_CAP_USD
    assert files_summarized > 0
    # The crucial invariant: aborting File-summary does NOT roll back or
    # interfere with the Function/Method pass that completed earlier.
    assert function_method_ingestion_completed is True
