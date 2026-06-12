"""Substring-based relevance grader for E2E query results.

Reads `query_results.jsonl` produced by `run_e2e.py`, evaluates each
query against its `expected_topk_substrings`, writes
`query_grades.jsonl` with `top1_relevant`, `top5_relevant`, and
`topk_relevant` boolean fields.

Pure deterministic grading — no LLM call. LLM-as-judge is a deferred
upgrade; the substring approach is good enough to detect glaring
regressions and gives reproducible numbers across cycles.

Usage:
    uv run python scripts/grade_queries.py \
        --in .planning/runs/<ts>/query_results.jsonl \
        --out .planning/runs/<ts>/query_grades.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("grade")

# Fields where we look for substring matches in a result item.
# Add new keys here whenever the indexer surfaces a different shape; the
# metric-contract should match real API responses.
_TEXT_FIELDS = (
    "symbol",   # current /search/semantic + /search/symbol shape
    "qualified_name",
    "fqn",
    "name",
    "file_path",
    "file",
    "path",
    "snippet",
    "source_snippet",
    "summary",
    "doc",
)


def _flatten_text(item: Any) -> str:
    """Concatenate all interesting text fields from a result item."""
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return ""
    parts: list[str] = []
    for key in _TEXT_FIELDS:
        v = item.get(key)
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, (list, tuple)):
            parts.extend(str(x) for x in v if x is not None)
    return " | ".join(parts)


def _matches_any(text: str, needles: Iterable[str]) -> bool:
    text_lc = text.lower()
    return any(n and n.lower() in text_lc for n in needles)


# Minimum fraction of facets a design bundle must cover to pass. Calibrate
# after the first baseline run; 0.7 means "most components present" without
# requiring every facet (some facets are intentionally hard).
DESIGN_PASS_THRESHOLD = 0.7


def _grade_design(record: dict[str, Any]) -> dict[str, Any]:
    """Facet-coverage grading for design-intent queries.

    A facet is `{"name": str, "any": [substr, ...]}` — covered when any of
    its substrings appears anywhere in the bundle text. Coverage measures
    whether the bundle contains ALL the components an agent needs to design
    the feature, not whether any single hit matches (the lookup grader).
    """
    facets = record.get("expected_facets", []) or []
    top_k = record.get("top_k", []) or []
    bundle_text = " | ".join(_flatten_text(item) for item in top_k).lower()

    covered: list[str] = []
    missed: list[str] = []
    for facet in facets:
        name = facet.get("name", "?")
        if any(n and n.lower() in bundle_text for n in facet.get("any", [])):
            covered.append(name)
        else:
            missed.append(name)

    coverage = len(covered) / len(facets) if facets else 0.0
    design_pass = bool(facets) and coverage >= DESIGN_PASS_THRESHOLD
    return {
        **record,
        "facet_coverage": round(coverage, 4),
        "facets_covered": covered,
        "facets_missed": missed,
        "design_pass": design_pass,
        # Back-compat with aggregate tooling that reads the boolean fields.
        "top1_relevant": design_pass,
        "top5_relevant": design_pass,
        "topk_relevant": design_pass,
    }


def _grade(record: dict[str, Any]) -> dict[str, Any]:
    intent = record["intent"]
    if intent == "design":
        return _grade_design(record)
    needles = record.get("expected_topk_substrings", []) or []
    top_k = record.get("top_k", []) or []
    expected_min = record.get("expected_min_results")

    # Per-position relevance
    positions = [_matches_any(_flatten_text(item), needles) for item in top_k]

    if intent == "structural":
        # Structural queries grade pass/fail purely on count, since the
        # query IS the structural pattern and substrings are fuzzy.
        passes = (expected_min or 1) <= len(top_k)
        return {
            **record,
            "top1_relevant": passes,
            "top5_relevant": passes,
            "topk_relevant": passes,
            "match_positions": positions,
        }

    return {
        **record,
        "top1_relevant": positions[0] if positions else False,
        "top5_relevant": any(positions[:5]),
        "topk_relevant": any(positions),
        "match_positions": positions,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    args = p.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    if not in_path.exists():
        log.error("input not found: %s", in_path)
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fout.write(json.dumps(_grade(rec)) + "\n")
            n += 1
    log.info("graded %d queries → %s", n, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
