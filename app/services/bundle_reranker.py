"""Cross-encoder reranker for /context-bundle call-graph neighbours.

This is the **bundle-truncation** reranker — distinct from
``app/services/reranker.py`` (the generative listwise CodeRankLLM that
re-orders search top-k). See ``docs/reranker-bundle-tiebreak-spike.md``.

The problem it solves
---------------------
The bi-encoder (e5-base-v2) embeds the query in isolation, so it cannot
encode a two-hop design inference like "re-ingest changed files → re-embed
→ vector store". Symbols carrying that downstream concern are *reachable*
in the call-graph but score too low to survive the token-budget
truncation. A **cross-encoder** scores the (query, symbol-doc) pair
jointly — attending across both at scoring time — and can recover the
inference, especially when the symbol's body literally names the concern.

Mechanism
---------
``qwen3-reranker-0.6b`` served via llama.cpp ``/completion``. We score
each pair by the first-token yes/no logprobs under the official
Qwen3-Reranker template (the ``/v1/rerank`` endpoint is broken for this
GGUF — it skips the instruction template). The relevance score is
``softmax({yes, no})`` over the aggregated case variants. Deterministic at
``temperature 0`` (verified: spread 0.0 over 5 reps).

Discipline (mirrors the existing reranker)
------------------------------------------
* **Opt-in**: gated by ``BUNDLE_RERANK_ENABLED`` / the request flag.
* **Fail-open**: endpoint unreachable, timeout, or unparseable response →
  return ``None``; the caller keeps today's bi-encoder ordering, so a
  bundle is byte-for-byte identical to the flag-off path.
* **Bounded**: caller caps the candidate set; scoring is concurrent with a
  hard wall-clock deadline.
"""
from __future__ import annotations

import concurrent.futures
import logging
import math
import time

import httpx

from ..config import settings

logger = logging.getLogger(__name__)

# Official Qwen3-Reranker yes/no judgement template.
_SYS = (
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)
_INSTRUCT = (
    "Given a software design or feature task, retrieve code symbols whose "
    "implementation is relevant to carrying out the task."
)
# Case/spacing variants the GGUF tokenizer emits for the two answers.
_YES = {"yes", "Yes", "YES"}
_NO = {"no", "No", "NO", "not", "Not"}

# Doc text cap — mirror the design snippet budget; keeps each prompt short
# enough for low per-pair latency.
_DOC_CAP_CHARS = 1800
_NEG_INF = -50.0


def _prompt(query: str, doc: str) -> str:
    user = f"<Instruct>: {_INSTRUCT}\n<Query>: {query}\n<Document>: {doc}"
    return (
        f"<|im_start|>system\n{_SYS}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _logsumexp(xs: list[float]) -> float:
    if not xs:
        return _NEG_INF
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def build_doc(qualified_name: str, snippet: str) -> str:
    """Build the reranker document for a symbol — the salient identifier
    (qname tail, stripped of any ``::summary`` marker) plus its snippet,
    capped. Mirrors what the bundle already indexes for that symbol."""
    tail = qualified_name.split("::")[0].split(".")[-1] or qualified_name
    body = (snippet or "")[:_DOC_CAP_CHARS]
    return f"{tail}\n{body}" if body else tail


def is_available(timeout_s: float = 1.5) -> bool:
    """Fast health probe so the caller can skip the work cleanly when the
    reranker isn't serving (e.g. production, where :60001 doesn't exist)."""
    try:
        r = httpx.get(f"{settings.BUNDLE_RERANK_URL}/health", timeout=timeout_s)
        return r.status_code == 200
    except Exception:
        return False


def _score_pair(client: httpx.Client, query: str, doc: str) -> float:
    """P(yes) over {yes,no} from the first-token logprob distribution.

    Returns NaN on any per-pair failure so the caller can treat it as
    "no signal" (score 0) rather than aborting the whole batch.
    """
    body = {
        "prompt": _prompt(query, doc),
        "n_predict": 1,
        "temperature": 0,
        "n_probs": 25,
        "top_k": 0,
        "cache_prompt": False,
    }
    try:
        r = client.post(f"{settings.BUNDLE_RERANK_URL}/completion", json=body)
        cp = r.json().get("completion_probabilities") or []
    except Exception:
        return float("nan")
    if not cp:
        return float("nan")
    # llama.cpp returns OpenAI-style top_logprobs: [{token, logprob, ...}].
    top = cp[0].get("top_logprobs") or cp[0].get("probs") or []

    def _tok(p: dict) -> str:
        return (p.get("token") or p.get("tok_str") or "").strip()

    yes_lps = [p["logprob"] for p in top if _tok(p) in _YES]
    no_lps = [p["logprob"] for p in top if _tok(p) in _NO]
    if not yes_lps and not no_lps:
        return float("nan")
    ly, ln = _logsumexp(yes_lps), _logsumexp(no_lps)
    return math.exp(ly) / (math.exp(ly) + math.exp(ln))


def rerank_scores(
    query: str, candidates: list[tuple[str, str]]
) -> dict[str, float] | None:
    """Return ``{qualified_name → relevance}`` for ``candidates``.

    Args:
        query: The task description.
        candidates: ``[(qualified_name, doc_text), ...]`` — already capped
            and de-duplicated by the caller. ``doc_text`` should come from
            :func:`build_doc`.

    Returns:
        A score map (relevance in ``[0, 1]``; NaN-scored pairs map to 0.0),
        or ``None`` to signal fail-open — the caller must keep its existing
        ordering unchanged. ``None`` is returned when the endpoint is
        unreachable, the deadline is exceeded before any pair scores, or an
        unexpected error occurs.
    """
    if not query or not candidates:
        return None
    if not is_available():
        logger.info(
            "bundle rerank requested but endpoint unreachable (url=%r); "
            "keeping bi-encoder order",
            settings.BUNDLE_RERANK_URL,
        )
        return None

    deadline = settings.BUNDLE_RERANK_DEADLINE_S
    workers = max(1, int(settings.BUNDLE_RERANK_CONCURRENCY))
    started = time.monotonic()
    out: dict[str, float] = {}

    def _work(item: tuple[str, str]) -> tuple[str, float]:
        qn, doc = item
        with httpx.Client(timeout=httpx.Timeout(deadline)) as client:
            return qn, _score_pair(client, query, doc)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_work, c): c[0] for c in candidates}
            for fut in concurrent.futures.as_completed(futures):
                remaining = deadline - (time.monotonic() - started)
                if remaining <= 0:
                    break
                try:
                    qn, sc = fut.result(timeout=max(remaining, 0.01))
                except Exception:
                    continue
                out[qn] = 0.0 if math.isnan(sc) else sc
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("bundle rerank batch raised: %s; keeping order", exc)
        return None

    if not out:
        logger.info(
            "bundle rerank produced no scores within %.1fs; keeping order",
            deadline,
        )
        return None
    logger.info(
        "bundle rerank scored %d/%d neighbours in %.2fs",
        len(out), len(candidates), time.monotonic() - started,
    )
    return out
