"""Four-arm retrieval benchmark — Level A (no model), design queries only.

Arms (docs/retrieval-methodology-from-agentalloy.md §2):

- ``none``      — empty context. Floor; facet coverage is 0 by construction.
- ``composed``  — POST /context-bundle (the system under test).
- ``oracle``    — hand-assembled ideal bundle hydrated from span lists in
  ``eval/oracle/<task_id>.json``. Ceiling: what retrieval *could* return.
- ``external``  — incumbent practice: deterministic ``git grep -n -C3`` over
  per-task keywords, capped at ``EXTERNAL_MAX_LINES`` lines.

All arms are graded with the same facet grader (``_grade_design``), so the
numbers decompose failures: oracle low → content problem; oracle high +
composed low → retrieval problem. Headline metric per task is lift capture::

    lift_capture = (composed - none) / (oracle - none)

Oracle span files::

    {
      "task_id": "dsg-cis-001",
      "repo": "code-indexer-service",
      "spans": [{"file_path": "app/routers/search.py",
                 "start_line": 800, "end_line": 840}],
      "external_keywords": ["rrf", "bm25"]
    }

Spans hydrate from the live checkouts in ``_DEFAULT_REPO_PATHS`` at run
time. The manifest records each repo's HEAD SHA; a dirty worktree aborts
the run unless ``--allow-dirty`` is passed, because span line numbers are
only meaningful against a known SHA.

Usage:
    uv run python scripts/run_arms.py                       # all design queries
    uv run python scripts/run_arms.py --task dsg-tf-003     # one task
    uv run python scripts/run_arms.py --allow-dirty
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grade_queries import _grade_design  # noqa: E402
from run_e2e import _DEFAULT_REPO_PATHS, _extract_top_k  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("arms")

REPO_ROOT = Path(__file__).resolve().parents[1]
ORACLE_DIR = REPO_ROOT / "eval" / "oracle"
RUNS_DIR = REPO_ROOT / ".planning" / "runs"

EXTERNAL_MAX_LINES = 150
ARMS = ("none", "composed", "oracle", "external")


# ---------------------------------------------------------------------------
# Repo state
# ---------------------------------------------------------------------------
def _git(repo_path: str, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", repo_path, *args],
        capture_output=True,
        text=True,
        check=False,
    ).stdout


def repo_manifest(repos: set[str], allow_dirty: bool) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for name in sorted(repos):
        path = _DEFAULT_REPO_PATHS.get(name, name)
        sha = _git(path, "rev-parse", "HEAD").strip()
        dirty = bool(_git(path, "status", "--porcelain").strip())
        manifest[name] = {"path": path, "sha": sha, "dirty": dirty}
        if dirty and not allow_dirty:
            raise SystemExit(
                f"{name} worktree at {path} is dirty — oracle span line numbers "
                f"are pinned to a SHA. Commit/stash, or pass --allow-dirty."
            )
        if dirty:
            log.warning("%s is dirty; span line numbers may be stale", name)
    return manifest


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
def hydrate_oracle(task_id: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Read span list and hydrate text from the checkout."""
    spec_path = ORACLE_DIR / f"{task_id}.json"
    if not spec_path.exists():
        return None, []
    spec = json.loads(spec_path.read_text())
    repo_path = Path(_DEFAULT_REPO_PATHS.get(spec["repo"], spec["repo"]))
    items: list[dict[str, Any]] = []
    for span in spec.get("spans", []):
        fp = repo_path / span["file_path"]
        if not fp.exists():
            log.warning("%s: span file missing: %s", task_id, fp)
            continue
        lines = fp.read_text(errors="replace").splitlines()
        start, end = span["start_line"], span["end_line"]
        text = "\n".join(lines[start - 1 : end])
        items.append({"symbol": f"{span['file_path']}:{start}-{end}", "snippet": text})
    return spec, items


def run_external(spec: dict[str, Any] | None, repo: str) -> list[dict[str, Any]]:
    """Deterministic incumbent arm: git grep -n -C3 per keyword, capped."""
    keywords = (spec or {}).get("external_keywords", [])
    if not keywords:
        return []
    repo_path = _DEFAULT_REPO_PATHS.get(repo, repo)
    out_lines: list[str] = []
    for kw in keywords:
        if len(out_lines) >= EXTERNAL_MAX_LINES:
            break
        res = subprocess.run(
            ["git", "-C", repo_path, "grep", "-n", "-C3", "-i", "--", kw],
            capture_output=True,
            text=True,
            check=False,
        )
        chunk = res.stdout.splitlines()[: EXTERNAL_MAX_LINES - len(out_lines)]
        if chunk:
            out_lines.append(f"### git grep -n -C3 -i {kw}")
            out_lines.extend(chunk)
    return [{"symbol": "git-grep", "snippet": "\n".join(out_lines)}] if out_lines else []


async def run_composed(
    client: httpx.AsyncClient, base: str, slug_map: dict[str, str], q: dict[str, Any]
) -> list[dict[str, Any]]:
    r = await client.post(
        f"{base}/context-bundle",
        json={
            "repo_path": _DEFAULT_REPO_PATHS.get(q["repo"], q["repo"]),
            "repo": slug_map[q["repo"]],
            "task_description": q["q"],
            "depth": min(q.get("k", 12), 3),
        },
        timeout=60.0,
    )
    if not r.is_success:
        log.error("%s: /context-bundle %s: %s", q["id"], r.status_code, r.text[:200])
        return []
    return _extract_top_k("design", r.json())


async def _resolve_slugs(
    client: httpx.AsyncClient, base: str, repos: set[str]
) -> dict[str, str]:
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


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _coverage(q: dict[str, Any], top_k: list[dict[str, Any]]) -> dict[str, Any]:
    g = _grade_design({**q, "top_k": top_k})
    return {
        "facet_coverage": g["facet_coverage"],
        "facets_covered": g["facets_covered"],
        "facets_missed": g["facets_missed"],
        "design_pass": g["design_pass"],
        "result_count": len(top_k),
    }


async def main_async(args: argparse.Namespace) -> int:
    queries = json.loads((REPO_ROOT / "scripts" / "queries.json").read_text())
    if isinstance(queries, dict):
        queries = queries["queries"]
    tasks = [q for q in queries if q.get("intent") == "design"]
    if args.task:
        tasks = [q for q in tasks if q["id"] in args.task]
    if not tasks:
        log.error("no matching design queries")
        return 2
    # Grader expects `expected_facets`; queries.json already uses that key.

    repos = {q["repo"] for q in tasks}
    manifest = repo_manifest(repos, args.allow_dirty)

    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient() as client:
        slug_map = await _resolve_slugs(client, args.service_url, repos)
        for q in tasks:
            spec, oracle_items = hydrate_oracle(q["id"])
            if spec is None:
                log.warning("%s: no oracle spec at eval/oracle/%s.json", q["id"], q["id"])
            arm_items: dict[str, list[dict[str, Any]]] = {
                "none": [],
                "composed": await run_composed(client, args.service_url, slug_map, q),
                "oracle": oracle_items,
                "external": run_external(spec, q["repo"]),
            }
            arms = {arm: _coverage(q, items) for arm, items in arm_items.items()}
            oracle_cov = arms["oracle"]["facet_coverage"]
            composed_cov = arms["composed"]["facet_coverage"]
            none_cov = arms["none"]["facet_coverage"]
            lift = (
                round((composed_cov - none_cov) / (oracle_cov - none_cov), 4)
                if oracle_cov > none_cov
                else None
            )
            rec = {
                "id": q["id"],
                "repo": q["repo"],
                "q": q["q"],
                "arms": arms,
                "lift_capture": lift,
            }
            results.append(rec)
            log.info(
                "%s  none=%.2f composed=%.2f oracle=%.2f external=%.2f  lift=%s",
                q["id"],
                none_cov,
                composed_cov,
                oracle_cov,
                arms["external"]["facet_coverage"],
                f"{lift:.2f}" if lift is not None else "n/a",
            )

    lifts = [r["lift_capture"] for r in results if r["lift_capture"] is not None]
    aggregate = {
        "tasks": len(results),
        "tasks_with_oracle": len(lifts),
        "mean_lift_capture": round(sum(lifts) / len(lifts), 4) if lifts else None,
        "mean_coverage_by_arm": {
            arm: round(
                sum(r["arms"][arm]["facet_coverage"] for r in results) / len(results), 4
            )
            for arm in ARMS
        },
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = RUNS_DIR / f"{ts}-arms"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(
            {"manifest": manifest, "aggregate": aggregate, "results": results}, indent=2
        )
        + "\n"
    )
    log.info("aggregate: %s", json.dumps(aggregate))
    log.info("wrote %s", out_dir / "results.json")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--service-url", default="http://127.0.0.1:8000")
    p.add_argument("--task", action="append", help="run only these task ids (repeatable)")
    p.add_argument("--allow-dirty", action="store_true")
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
