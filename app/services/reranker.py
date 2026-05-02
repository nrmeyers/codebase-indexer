"""Listwise rerank stage using ``nomic-ai/CodeRankLLM`` via LM Studio.

This is the **second stage** of the two-stage retrieval design (v5.3 §17
revisit, ADR pending).  Stage 1 (the bi-encoder) widens DuckDB's
``array_cosine_distance`` top-k from N to ~50 candidates; this stage runs
those candidates through CodeRankLLM as a single listwise prompt and
returns the model's permutation, which the caller slices back to N.

Why listwise (vs cross-encoder pairs)?
    Nomic's CodeRankLLM is a *generative* reranker fine-tuned to emit a
    permutation of bracketed indices given a query and a list of
    candidates.  A single listwise call beats N pairwise calls on both
    latency and quality (the model can reason about candidates relative
    to each other instead of in isolation).

Failure mode:
    The reranker is *strictly opt-in* (`?rerank=true`) and *non-fatal*.
    If LM Studio is unreachable, the requested model is not loaded, the
    HTTP call times out, or the LLM emits an unparseable response, we
    return the candidates **in their original order** — the bi-encoder
    results are already good; rerank is a precision boost, not a
    correctness requirement.

Prompt template:
    A small adaptation of the Nomic-recommended format, optimised for
    code candidates (qualified_name as the salient identifier, snippet
    truncated to keep the prompt within sensible context bounds).

.. note:: Qwen3 thinking-mode quirk
    The LM Studio preset for ``qwen3.6-*`` ignores both the ``/no_think``
    user-message directive and ``chat_template_kwargs={"enable_thinking":
    false}`` — both end up in reasoning mode regardless. We send both
    anyway as belt-and-suspenders (no-op for non-Qwen models). The
    LM Studio adapter falls back to ``reasoning_content`` when
    ``content`` is empty, and ``max_tokens=2048`` gives the reasoning +
    bracketed answer enough room. Counter-intuitive: 27B dense beats
    MoE-A3B per-token on Apple-Metal hardware (expert-routing overhead).
"""
from __future__ import annotations

import concurrent.futures
import logging
import os
import re
import time
from typing import Any

from . import lm_studio
from .. import metrics as _metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


# Per-candidate snippet cap.  CodeRankLLM is a 7B model; tokenizer overhead
# at ~4 chars/token means a 4000-char snippet ≈ 1000 tokens.  With 50
# candidates we'd blow past the model's context window — cap aggressively
# and rely on the qualified_name + first lines for ranking signal.
MAX_SNIPPET_CHARS = 800

# Hard cap on candidates we send to a single rerank call.  Beyond ~30 the
# model's accuracy degrades (per Nomic's own evals) and prompt cost
# grows linearly.
MAX_CANDIDATES = 30


def _rerank_deadline_seconds() -> float:
    """Resolve the per-call rerank latency budget from the environment.

    Read at call time (not module load) so tests can ``monkeypatch.setenv``
    without needing to reimport the module. Default 5.0 s on the assumption
    that the rerank endpoint serves interactive search — slow runs degrade
    to bi-encoder ordering rather than blocking the user.

    Set ``RERANK_DEADLINE_SECONDS=0`` (or any non-positive value) to disable
    the deadline entirely, e.g. for batch eval runs that prefer correctness
    over latency.
    """
    raw = os.environ.get("RERANK_DEADLINE_SECONDS", "5.0")
    try:
        return float(raw)
    except ValueError:
        return 5.0


# Module-level executor — one worker is enough; we serialise rerank calls
# through LM Studio anyway and concurrent calls would just contend for the
# same loaded model. Lazy-initialised on first use.
_executor: concurrent.futures.ThreadPoolExecutor | None = None


def _get_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="rerank"
        )
    return _executor

# System prompt — frozen so production behaviour is reproducible.
_SYSTEM_PROMPT = (
    "You are a code-search relevance ranker. Given a query and a numbered "
    "list of code candidates, output ONLY a permutation of the candidate "
    "numbers from most to least relevant, in the format: [3] > [1] > [4]. "
    "Do not include explanations, prose, or any text outside the brackets."
)


# Trailer appended to the END of the user message.  Qwen3 only honors the
# ``/no_think`` directive when it appears in the user role (not system) —
# this disables the model's reasoning channel so the permutation lands
# in ``content`` rather than burning the token budget on chain-of-thought.
# Other model families (CodeRankLLM, Llama 3, Mistral) treat it as
# trailing whitespace and ignore it harmlessly.
_NO_THINK_TRAILER = "\n\n/no_think"

# Generous token budget so a *thinking* model (Qwen3, DeepSeek-R1) that
# ignores the ``/no_think`` directive still has room to finish reasoning
# AND emit the bracketed permutation.  Plain models cap their output well
# before this and pay no penalty for the headroom — LM Studio bills only
# for tokens actually generated.
_MAX_TOKENS = 2048


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


Candidate = dict[str, Any]
"""A retrieval result.  Must include at least ``qualified_name`` and either
``source`` or ``snippet``; other keys are passed through unchanged."""


def is_available() -> bool:
    """Return True when LM Studio is reachable and the rerank model is loaded.

    Used by route handlers to short-circuit ``?rerank=true`` requests
    cleanly when the backend isn't available, instead of paying the
    timeout latency to discover it.

    Thin wrapper around :func:`lm_studio.can_rerank` — defined here so
    callers (search/router, context_bundle/router) only need to import
    the ``reranker`` module, not both.
    """
    return lm_studio.can_rerank()


def rerank(query: str, candidates: list[Candidate]) -> list[Candidate]:
    """Return ``candidates`` reordered by CodeRankLLM relevance to ``query``.

    Best-effort + latency-budgeted: returns the original list unchanged on
    any failure (LM Studio unreachable, model not loaded, deadline exceeded,
    empty/unparseable response).  The returned list contains the *same*
    objects (no copy) — this preserves every metadata field the bi-encoder
    attached (similarity score, file path, line range, etc.) while only
    changing order.

    Each call records exactly one observation against
    ``forge_indexer_rerank_outcome_total{outcome}`` with outcome ∈
    {applied, skip-empty-input, skip-unavailable, skip-deadline,
    skip-empty-response, skip-parse-error}.  Operators consume this metric
    to decide when the rerank flag is safe to flip on in production
    (rate(applied)/rate(total) ≥ some confidence threshold).

    The latency budget is read at call time from
    ``RERANK_DEADLINE_SECONDS`` (default 5.0 s).  A non-positive value
    disables the deadline — useful for batch eval runs.

    Args:
        query: The natural-language query (no Nomic prefix needed; the
            reranker model uses its own internal prompt format).
        candidates: List of retrieval results from stage 1.  Capped to
            ``MAX_CANDIDATES`` before the rerank call; trailing
            candidates are appended back in their original order.
    """
    if not query or not candidates:
        _metrics.record_rerank_outcome("skip-empty-input")
        return candidates
    if not is_available():
        _metrics.record_rerank_outcome("skip-unavailable")
        return candidates

    head = candidates[:MAX_CANDIDATES]
    tail = candidates[MAX_CANDIDATES:]

    prompt = _build_prompt(query, head)

    # Run the LM Studio call in a worker thread so we can enforce a
    # deadline. ``concurrent.futures.Future.result(timeout=)`` raises
    # TimeoutError when the budget is exceeded; the underlying HTTP call
    # keeps running until LM Studio releases it, but we no longer block
    # the search request on it.
    def _call_lm_studio() -> str | None:
        return lm_studio.chat_complete(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt + _NO_THINK_TRAILER},
            ],
            max_tokens=_MAX_TOKENS,
            temperature=0.0,
            # Belt-and-suspenders: pass ``enable_thinking=False`` via the
            # chat-template renderer (some LM Studio model presets honor it
            # there) AND keep the ``/no_think`` user-message trailer (the
            # documented Qwen3 escape hatch).  At least one of these two
            # paths takes effect on every Qwen3 quant we've tested; both
            # are no-ops for non-Qwen models.
            chat_template_kwargs={"enable_thinking": False},
        )

    deadline = _rerank_deadline_seconds()
    started = time.monotonic()
    try:
        future = _get_executor().submit(_call_lm_studio)
        response = future.result(timeout=deadline if deadline > 0 else None)
    except concurrent.futures.TimeoutError:
        elapsed = time.monotonic() - started
        logger.info(
            "rerank deadline exceeded after %.2fs (budget=%.2fs); "
            "returning bi-encoder order",
            elapsed, deadline,
        )
        _metrics.record_rerank_outcome("skip-deadline")
        return candidates
    except Exception as exc:  # pragma: no cover — defensive; LM Studio
        # adapter already swallows expected errors and returns None
        logger.warning("rerank call raised: %s; keeping bi-encoder order", exc)
        _metrics.record_rerank_outcome("skip-unavailable")
        return candidates

    if not response:
        _metrics.record_rerank_outcome("skip-empty-response")
        return candidates

    permutation = _parse_permutation(response, len(head))
    if permutation is None:
        logger.debug("Rerank response unparseable; keeping bi-encoder order")
        _metrics.record_rerank_outcome("skip-parse-error")
        return candidates

    reordered = [head[i] for i in permutation]
    _metrics.record_rerank_outcome("applied")
    return reordered + tail


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_prompt(query: str, candidates: list[Candidate]) -> str:
    """Render the listwise rerank prompt.

    Each candidate gets a 1-indexed bracket label (matching the model's
    expected output format), the qualified name as the salient identifier,
    and a truncated snippet so the LLM has enough signal to rank without
    blowing the context window.
    """
    lines = [f"Query: {query}", "", "Candidates:"]
    for i, c in enumerate(candidates, start=1):
        qn = c.get("qualified_name") or c.get("symbol") or c.get("node_id") or f"candidate_{i}"
        snippet = c.get("source") or c.get("snippet") or ""
        if isinstance(snippet, str) and len(snippet) > MAX_SNIPPET_CHARS:
            snippet = snippet[:MAX_SNIPPET_CHARS] + "…"
        # Strip leading whitespace so multi-line snippets stay legible
        # without bloating the prompt with redundant indentation.
        snippet = "\n".join(line.rstrip() for line in str(snippet).splitlines()).strip()
        lines.append(f"[{i}] {qn}")
        if snippet:
            lines.append(snippet)
        lines.append("")
    lines.append(
        f"Rank the {len(candidates)} candidates from most to least relevant. "
        "Output only a permutation like [3] > [1] > [2]."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


_BRACKET_RE = re.compile(r"\[(\d+)\]")


def _parse_permutation(response: str, n: int) -> list[int] | None:
    """Parse a model response into a 0-indexed permutation of ``range(n)``.

    Accepts any text containing bracketed integers; ``[3] > [1] > [4]``,
    ``"[3], [1], [4]"``, even ``"[3]\\n[1]\\n[4]"`` all parse the same.

    Returns ``None`` when the response doesn't yield a valid permutation
    (missing indices, duplicates, out-of-range values).  Callers should
    keep the original order on ``None`` rather than partially applying.
    """
    matches = _BRACKET_RE.findall(response or "")
    if not matches:
        return None
    seen: set[int] = set()
    perm: list[int] = []
    for m in matches:
        try:
            idx = int(m) - 1  # 1-indexed in prompt → 0-indexed for slicing
        except ValueError:
            continue
        if idx < 0 or idx >= n or idx in seen:
            continue
        seen.add(idx)
        perm.append(idx)
    if len(perm) != n:
        # Append any candidates the model omitted, preserving their
        # original order — better than dropping them silently.
        for i in range(n):
            if i not in seen:
                perm.append(i)
    return perm if len(perm) == n else None
