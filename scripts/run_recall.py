"""First-stage recall harness — isolates the bi-encoder from the bundle.

The four-arm benchmark (run_arms.py) measures the COMPOSED bundle, so a win
there is contaminated by graph expansion, boosts, and snippet hydration.
This harness queries RAW ``/search/semantic`` (no graph, no boosts, no
bundle) and grades facet coverage over the returned qualified names at
several cut-offs. It answers a different question: *does first-stage
retrieval surface the right symbols at all, before any downstream stage?*

Two uses:
  * **Symbol cards (§5):** cards make under-indexed-but-relevant symbols
    reachable; the effect shows here (coverage@k up) even when composed
    lift is already saturated at 1.00. Run before/after a card re-index.
  * **Embedder eval (#11):** the primary metric for comparing embedders
    (see docs/embedder-eval-plan.md).

Usage:
    uv run python scripts/run_recall.py [--service-url URL] [--k 50]
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grade_queries import _grade_design  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("recall")

REPO_ROOT = Path(__file__).resolve().parent.parent
_CUTOFFS = (10, 25, 50)


def _resolve_slugs(base: str, repos: set[str]) -> dict[str, str]:
    health = httpx.get(f"{base}/health", timeout=10).json()
    slugs = [r["name"] for r in health.get("repos", [])]
    out: dict[str, str] = {}
    for name in repos:
        match = next((s for s in slugs if s == name), None) or next(
            (s for s in slugs if name.lower() in s.lower()), None
        )
        out[name] = match or name
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--service-url", default="http://127.0.0.1:8000")
    ap.add_argument("--k", type=int, default=max(_CUTOFFS))
    args = ap.parse_args()

    queries = json.loads((REPO_ROOT / "scripts" / "queries.json").read_text())
    if isinstance(queries, dict):
        queries = queries["queries"]
    tasks = [q for q in queries if q.get("intent") == "design"]
    if not tasks:
        log.error("no design queries")
        return 1

    slug_map = _resolve_slugs(args.service_url, {q["repo"] for q in tasks})
    per_cutoff: dict[int, list[float]] = {c: [] for c in _CUTOFFS}
    rows = []

    with httpx.Client() as client:
        for q in tasks:
            r = client.get(
                f"{args.service_url}/search/semantic",
                params={"q": q["q"], "repo": slug_map[q["repo"]], "k": args.k},
                timeout=60,
            )
            if not r.is_success:
                log.error("%s: /search/semantic %s", q["id"], r.status_code)
                continue
            results = r.json().get("results", [])
            row = {"id": q["id"]}
            for c in _CUTOFFS:
                top = [{"symbol": x.get("symbol", "")} for x in results[:c]]
                cov = _grade_design({**q, "top_k": top})["facet_coverage"]
                per_cutoff[c].append(cov)
                row[f"cov@{c}"] = round(cov, 3)
            rows.append(row)
            log.info(
                "%-14s %s",
                q["id"],
                "  ".join(f"@{c}={row[f'cov@{c}']:.2f}" for c in _CUTOFFS),
            )

    means = {c: round(statistics.mean(v), 4) if v else 0.0 for c, v in per_cutoff.items()}
    log.info("MEAN facet recall: %s", "  ".join(f"@{c}={means[c]}" for c in _CUTOFFS))
    print(json.dumps({"mean_recall_by_cutoff": means, "tasks": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
