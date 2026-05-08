"""Manifest LLM client wrapper — File summary path (Phase 1.2b).

Thin OpenAI-compatible chat-completions client used by the embedding
subprocess driver to generate per-file summaries (the "File" chunk kind
in :mod:`app.services.chunk_strategies`).

Design constraints:
  * Stateless module-level function — the embed driver runs as a
    subprocess f-string (see :func:`_blocking_embed` in
    ``app/routers/index.py``) so a class with self-state would be awkward
    to reuse there.
  * Returns ``None`` on every failure mode (timeout, network,
    non-2xx, JSON-shape mismatch).  The caller proceeds without the
    summary — file summaries are additive; ingestion of
    Function/Method/Class/Module continues unaffected.
  * Hard 15s timeout per call.  Manifest's verified Haiku latency is
    400-1200 ms p95; 15s catches a wedged egress without stalling the
    indexer.
  * Token usage is returned alongside the summary so the caller can
    enforce the per-repo cost cap defined in
    :data:`chunk_strategies.FILE_SUMMARY_REPO_COST_CAP_USD`.

Environment:
  * ``MANIFEST_URL`` — base URL of the Manifest gateway (e.g.
    ``http://localhost:2099``).  Required.
  * ``MANIFEST_AGENT_KEY`` — bearer token for the Manifest agent.
    Required.

If either env var is missing, :func:`summarize_file` returns ``None``
without making a network call.  This is the documented graceful-degrade
path for environments where Manifest isn't reachable (CI, sandbox,
fully-offline regression runs).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import httpx

from .chunk_strategies import build_file_summary_input

logger = logging.getLogger(__name__)

# Per-call wall-clock cap.  Verified p95 for Haiku via Manifest is
# ~1.2s; 15s is generous slack for a cold cache or transient backoff.
_MANIFEST_TIMEOUT_S: float = 15.0

# Default summarizer model — Haiku is the cheapest Anthropic model with
# acceptable summarization quality.  Verified pricing in
# ``chunk_strategies.HAIKU_INPUT_USD_PER_TOKEN``.
DEFAULT_SUMMARY_MODEL: str = "claude-haiku-4-5"


@dataclass(frozen=True)
class FileSummaryResult:
    """Outcome of a single Manifest summary call.

    Attributes:
        summary: The model's summary text (already trimmed).
        input_tokens: Prompt tokens billed.
        output_tokens: Completion tokens billed.
    """

    summary: str
    input_tokens: int
    output_tokens: int


def summarize_file(
    path: str,
    content: str,
    model: str = DEFAULT_SUMMARY_MODEL,
) -> Optional[FileSummaryResult]:
    """Request a one-shot file summary from Manifest.

    Returns ``None`` on any failure — caller continues without a summary.

    Args:
        path: Repo-relative file path; used in the prompt to ground the
            model.
        content: File body.  Already byte-capped at 8 KB inside
            :func:`chunk_strategies.build_file_summary_input`.
        model: Manifest model id.  Defaults to ``claude-haiku-4-5``.

    Returns:
        :class:`FileSummaryResult` on success, ``None`` on any error.
    """
    base_url = os.environ.get("MANIFEST_URL")
    api_key = os.environ.get("MANIFEST_AGENT_KEY")
    if not base_url or not api_key:
        logger.debug(
            "manifest.skip_unconfigured "
            "MANIFEST_URL=%s MANIFEST_AGENT_KEY=%s",
            bool(base_url), bool(api_key),
        )
        return None

    prompt = build_file_summary_input(path, content)
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 220,
        "temperature": 0.2,
    }

    try:
        with httpx.Client(timeout=_MANIFEST_TIMEOUT_S) as client:
            resp = client.post(
                url,
                json=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
        if resp.status_code >= 400:
            logger.warning(
                "manifest.summarize_http_error path=%s status=%d",
                path, resp.status_code,
            )
            return None
        data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("manifest.summarize_failed path=%s err=%s", path, exc)
        return None

    try:
        summary = (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError):
        logger.warning("manifest.summarize_bad_shape path=%s", path)
        return None
    if not summary:
        return None

    usage = data.get("usage") or {}
    return FileSummaryResult(
        summary=summary,
        input_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("completion_tokens") or 0),
    )
