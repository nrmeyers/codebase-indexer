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
_TEXT_FIELDS = (
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


def _grade(record: dict[str, Any]) -> dict[str, Any]:
    intent = record["intent"]
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
