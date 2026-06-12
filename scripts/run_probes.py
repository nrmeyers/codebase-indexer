"""Probe-query runner — named "this should work" retrieval snapshots.

Probes are plain-language design queries whose intent vocabulary does NOT
match the symbol vocabulary of the answer (the "find the auth handler"
shape from docs/retrieval-methodology-from-agentalloy.md §3). Each run
POSTs /context-bundle per probe, grades facet coverage with the same
logic as grade_queries.py, and writes:

- ``eval/probes/snapshots/<id>.response.json`` — full bundle response,
  committed so retrieval changes show up as reviewable diffs.
- ``eval/probes/snapshots/summary.json`` — per-probe facet coverage.

``--check`` compares coverage against the committed summary and exits 1
on any regression (a probe covering fewer facets than baseline).

Usage:
    uv run python scripts/run_probes.py                # snapshot
    uv run python scripts/run_probes.py --check        # regression gate
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grade_queries import _grade_design  # noqa: E402
from run_e2e import _DEFAULT_REPO_PATHS, _extract_top_k  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("probes")

SNAPSHOT_DIR = Path(__file__).resolve().parents[1] / "eval" / "probes" / "snapshots"


async def _resolve_slugs(
    client: httpx.AsyncClient, base: str, repos: set[str]
) -> dict[str, str]:
    """Map checkout dir names to canonical service slugs via /health."""
    slug_map: dict[str, str] = {}
    health = (await client.get(f"{base}/health", timeout=10.0)).json()
    slugs = [r["name"] for r in health.get("repos", [])]
    for name in repos:
        dir_name = Path(_DEFAULT_REPO_PATHS.get(name, name)).name
        match = next((s for s in slugs if s == dir_name), None) or next(
            (s for s in slugs if dir_name.lower() in s.lower()), None
        )
        slug_map[name] = match or dir_name
    return slug_map


async def run_probes(base: str, probes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    async with httpx.AsyncClient() as client:
        slug_map = await _resolve_slugs(client, base, {p["repo"] for p in probes})
        for probe in probes:
            r = await client.post(
                f"{base}/context-bundle",
                json={
                    "repo_path": _DEFAULT_REPO_PATHS.get(probe["repo"], probe["repo"]),
                    "repo": slug_map[probe["repo"]],
                    "task_description": probe["q"],
                    "depth": min(probe.get("k", 3), 3),
                },
                timeout=60.0,
            )
            body = r.json() if r.is_success else {"error": r.status_code, "detail": r.text[:500]}
            top_k = _extract_top_k("design", body) if r.is_success else []
            graded = _grade_design({**probe, "top_k": top_k})
            summary = {
                "id": probe["id"],
                "repo": probe["repo"],
                "q": probe["q"],
                "ok": r.is_success,
                "facet_coverage": graded["facet_coverage"],
                "facets_covered": graded["facets_covered"],
                "facets_missed": graded["facets_missed"],
                "result_count": len(top_k),
            }
            summaries.append(summary)
            SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            (SNAPSHOT_DIR / f"{probe['id']}.response.json").write_text(
                json.dumps(body, indent=2, sort_keys=True) + "\n"
            )
            log.info(
                "%s coverage=%.2f covered=%s missed=%s",
                probe["id"],
                summary["facet_coverage"],
                summary["facets_covered"],
                summary["facets_missed"],
            )
    return summaries


def check_regression(summaries: list[dict[str, Any]]) -> int:
    baseline_path = SNAPSHOT_DIR / "summary.json"
    if not baseline_path.exists():
        log.error("no committed baseline at %s — run without --check first", baseline_path)
        return 2
    baseline = {s["id"]: s for s in json.loads(baseline_path.read_text())}
    failures = 0
    for s in summaries:
        b = baseline.get(s["id"])
        if b is None:
            log.info("%s: new probe (no baseline) — coverage %.2f", s["id"], s["facet_coverage"])
            continue
        if s["facet_coverage"] < b["facet_coverage"]:
            log.error(
                "REGRESSION %s: coverage %.2f < baseline %.2f (now missing %s)",
                s["id"],
                s["facet_coverage"],
                b["facet_coverage"],
                s["facets_missed"],
            )
            failures += 1
        else:
            log.info(
                "%s: %.2f vs baseline %.2f OK", s["id"], s["facet_coverage"], b["facet_coverage"]
            )
    return 1 if failures else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--service-url", default="http://127.0.0.1:8000")
    p.add_argument("--probes", default=str(Path(__file__).resolve().parent / "probes.json"))
    p.add_argument(
        "--check",
        action="store_true",
        help="compare against committed summary.json; exit 1 on coverage regression",
    )
    args = p.parse_args()

    probes = json.loads(Path(args.probes).read_text())
    summaries = asyncio.run(run_probes(args.service_url, probes))

    if args.check:
        return check_regression(summaries)

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    (SNAPSHOT_DIR / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    log.info("wrote baseline %s", SNAPSHOT_DIR / "summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
