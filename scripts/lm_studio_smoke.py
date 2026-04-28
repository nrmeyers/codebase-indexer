#!/usr/bin/env python3
"""LM Studio rerank smoke test — end-to-end probe of the two-stage pipeline.

Exercises the integration without standing up the full FastAPI service:

    1. Resolves the LM Studio base URL from the environment (with .env).
    2. Lists loaded models and prints which ones the embed/rerank hints
       resolve to.
    3. Sends a small listwise rerank request through the production
       :func:`app.services.reranker.rerank` path against a fixed set of
       synthetic candidates.
    4. Asserts the resulting permutation puts the relevant ``auth.*``
       candidates above the irrelevant ones.

Exits 0 on success, 1 on any failure.  Wall-clock time is printed so
operators can spot a slow model (>30s for a 5-candidate prompt usually
means rerank is too slow to ship interactively — switch to a
faster/MoE variant).

Usage::

    cd ~/code-indexer-service
    uv run python scripts/lm_studio_smoke.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Bridge .env → os.environ so direct env reads in lm_studio.py see
# locally-configured LM_STUDIO_* values.  Same trick the FastAPI app
# uses at process start.
from dotenv import load_dotenv

# Locate the repo root regardless of where the script is invoked from
# so ``uv run`` and ``python scripts/...`` both work.
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")
sys.path.insert(0, str(_REPO_ROOT))


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    from app.services import lm_studio, reranker

    _section("Configuration")
    print(f"  LM_STUDIO_URL       = {lm_studio.base_url() or '(unset)'}")
    print(f"  embed_model_hint    = {lm_studio.embed_model_hint()}")
    print(f"  rerank_model_hint   = {lm_studio.rerank_model_hint()}")
    print(f"  request_timeout_s   = {lm_studio.request_timeout_s()}")

    _section("Server probe")
    if not lm_studio.base_url():
        print("  FAIL — LM_STUDIO_URL is not set; nothing to probe.")
        return 1
    models = lm_studio.list_models()
    if not models:
        print("  FAIL — no models loaded (or LM Studio unreachable).")
        return 1
    for m in models:
        print(f"  • {m}")

    _section("Hint resolution")
    embed = lm_studio.resolve_model(lm_studio.embed_model_hint())
    rerank_m = lm_studio.resolve_model(lm_studio.rerank_model_hint())
    print(f"  embed  hint → {embed or '(no match)'}")
    print(f"  rerank hint → {rerank_m or '(no match)'}")

    if not rerank_m:
        print(
            "\n  FAIL — rerank model not loaded.  Set LM_STUDIO_RERANK_MODEL\n"
            "  to a substring of one of the loaded model ids above."
        )
        return 1

    _section("Listwise rerank E2E")
    cands = [
        {
            "qualified_name": "utils.format_currency",
            "source": (
                "def format_currency(amount: float) -> str:\n"
                "    return f\"${amount:,.2f}\""
            ),
        },
        {
            "qualified_name": "auth.verify_jwt",
            "source": (
                "def verify_jwt(token: str) -> dict:\n"
                "    return jwt.decode(token, SECRET, algorithms=[\"HS256\"])"
            ),
        },
        {
            "qualified_name": "auth.create_session_token",
            "source": (
                "def create_session_token(user_id: int) -> str:\n"
                "    return jwt.encode({\"sub\": user_id}, SECRET)"
            ),
        },
        {
            "qualified_name": "database.connect_pool",
            "source": (
                "def connect_pool() -> Pool:\n"
                "    return Pool(dsn=DSN, max=10)"
            ),
        },
        {
            "qualified_name": "auth.refresh_token",
            "source": (
                "def refresh_token(rt: str) -> str:\n"
                "    return jwt.encode(decode_refresh(rt), SECRET)"
            ),
        },
    ]
    query = "How does JWT authentication work in this codebase?"
    print("  Pre-rerank order:")
    for i, c in enumerate(cands, 1):
        print(f"    [{i}] {c['qualified_name']}")

    t0 = time.monotonic()
    result = reranker.rerank(query, cands)
    dt = time.monotonic() - t0
    print(f"\n  Reranked in {dt:.1f}s")

    print("  Post-rerank order:")
    for i, c in enumerate(result, 1):
        print(f"    [{i}] {c['qualified_name']}")

    # Quality gate — at least 2 of the top-3 should be auth.*
    top3_auth = sum(
        1 for c in result[:3] if "auth" in c["qualified_name"]
    )
    if top3_auth >= 2:
        print(
            f"\n  OK — {top3_auth}/3 top results are auth.* "
            f"(rerank quality acceptable)."
        )
    else:
        print(
            f"\n  WARN — only {top3_auth}/3 top results are auth.*. "
            "Inspect manually; the model may be ignoring snippet body."
        )

    # Latency gate — anything over 30s for 5 candidates is too slow for
    # interactive use.  Recommend MoE variants for speed.
    if dt > 30.0:
        print(
            f"\n  WARN — rerank took {dt:.0f}s.  For interactive search,\n"
            "  switch to an MoE-A3B model (e.g. qwen3.6-35b-a3b) or\n"
            "  raise LM_STUDIO_TIMEOUT to avoid uvicorn-side timeouts."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
