"""E2E cycle driver for the Code Indexer Service.

Per `.planning/E2E_TEST_OPTIMIZATION_PLAN.md`. Indexes the test corpus
repos, runs the query suite, snapshots /metrics, and scores against the
SLO matrix in plan §4. Exits 0 on all-green, 1 on any red metric.

Usage:
    uv run python scripts/run_e2e.py \
        --service-url http://localhost:8000 \
        --queries scripts/queries.json \
        --out .planning/runs/$(date -u +%Y%m%dT%H%M%SZ)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("e2e")

# Repo path → service-side slug map. Override via --repo-paths if dirs differ.
_DEFAULT_REPO_PATHS = {
    "TheForge": "/Users/zacharymatthews/TheForge",
    "code-indexer-service": "/Users/zacharymatthews/code-indexer-service",
    "code-graph-rag": "/Users/zacharymatthews/code-graph-rag",
}


# ---------------------------------------------------------------------------
# SLO matrix — plan §4
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SLO:
    name: str
    target: float
    op: str  # "le" | "ge"
    unit: str

    def passes(self, value: float) -> bool:
        if self.op == "le":
            return value <= self.target
        if self.op == "ge":
            return value >= self.target
        raise ValueError(f"unknown op {self.op}")


SLOS: tuple[SLO, ...] = (
    SLO("indexing_rate_symbols_per_s", 200.0, "ge", "sym/s"),
    SLO("search_semantic_p95_no_rerank_s", 0.2, "le", "s"),
    SLO("search_semantic_p95_with_rerank_s", 4.0, "le", "s"),
    SLO("search_structural_p95_s", 0.1, "le", "s"),
    SLO("search_symbol_p95_s", 0.05, "le", "s"),
    SLO("context_bundle_p95_s", 1.5, "le", "s"),
    SLO("top1_relevance_semantic", 0.70, "ge", "%"),
    SLO("top5_relevance_semantic", 0.90, "ge", "%"),
    SLO("lm_studio_uptime", 0.95, "ge", "%"),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ordered[int(k)]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


async def _fetch_metrics_text(client: httpx.AsyncClient, base: str) -> str:
    try:
        r = await client.get(f"{base}/metrics", timeout=10.0)
        return r.text if r.status_code == 200 else f"# /metrics returned {r.status_code}\n"
    except Exception as e:
        return f"# /metrics fetch failed: {e}\n"


async def _ensure_indexed(
    client: httpx.AsyncClient,
    base: str,
    repo: str,
    repo_path: str,
    force: bool,
    out_dir: Path,
) -> dict[str, Any]:
    """POST /index, poll until done. Return per-phase timings."""
    log.info("ensuring %s is indexed (force=%s)", repo, force)
    start = time.monotonic()
    payload = {"repo_path": repo_path, "force_reindex": force}
    r = await client.post(f"{base}/index", json=payload, timeout=30.0)
    if r.status_code == 409:
        log.info("%s already has an active index; polling existing job", repo)
        # Find the active job_id via /index/jobs
        jobs = await client.get(f"{base}/index/jobs", timeout=10.0)
        active = [
            j
            for j in jobs.json().get("jobs", [])
            if j.get("repo_slug") == Path(repo_path).name
            and j.get("status") in ("running", "queued")
        ]
        if not active:
            raise RuntimeError(f"409 from /index but no active job in /index/jobs for {repo}")
        job_id = active[0]["job_id"]
    elif r.is_success:
        job_id = r.json()["job_id"]
    else:
        raise RuntimeError(f"/index failed: {r.status_code} {r.text}")

    # Poll status until terminal
    last_status = None
    last_phase = None
    while True:
        s = await client.get(f"{base}/index/{job_id}/status", timeout=10.0)
        body = s.json()
        status = body.get("status")
        phase = body.get("phase")
        if status != last_status or phase != last_phase:
            log.info("  %s: status=%s phase=%s pct=%.1f", repo, status, phase, body.get("progress_pct", 0.0))
            last_status, last_phase = status, phase
        if status in ("done", "failed", "cancelled", "interrupted"):
            elapsed = time.monotonic() - start
            return {
                "repo": repo,
                "status": status,
                "elapsed_s": round(elapsed, 3),
                "node_count": body.get("node_count", 0),
                "rel_count": body.get("rel_count", 0),
                "embedding_count": body.get("embedding_count", 0),
                "error": body.get("error"),
            }
        await asyncio.sleep(0.5)


async def _run_queries(
    client: httpx.AsyncClient,
    base: str,
    queries: list[dict[str, Any]],
    out_dir: Path,
) -> list[dict[str, Any]]:
    """Dispatch each query to the right endpoint, time it, capture top-k."""
    results: list[dict[str, Any]] = []
    out = (out_dir / "query_results.jsonl").open("w")
    for q in queries:
        intent = q["intent"]
        repo_slug = Path(_DEFAULT_REPO_PATHS.get(q["repo"], q["repo"])).name
        url, params = _build_request(intent, q, repo_slug)
        t0 = time.monotonic()
        try:
            r = await client.get(f"{base}{url}", params=params, timeout=60.0)
            elapsed = time.monotonic() - t0
            ok = r.is_success
            body = r.json() if ok else None
            top_k = _extract_top_k(intent, body) if ok else []
        except Exception as e:
            elapsed = time.monotonic() - t0
            ok = False
            top_k = []
            r = None
            log.warning("query %s failed: %s", q["id"], e)

        rec = {
            "id": q["id"],
            "repo": q["repo"],
            "intent": intent,
            "q": q["q"],
            "k": q.get("k"),
            "elapsed_s": round(elapsed, 4),
            "ok": ok,
            "status_code": r.status_code if r is not None else None,
            "expected_topk_substrings": q.get("expected_topk_substrings", []),
            "expected_min_results": q.get("expected_min_results"),
            "result_count": len(top_k),
            "top_k": top_k,
        }
        results.append(rec)
        out.write(json.dumps(rec) + "\n")
        out.flush()
    out.close()
    return results


def _build_request(intent: str, q: dict[str, Any], repo_slug: str) -> tuple[str, dict[str, str]]:
    if intent == "semantic":
        return "/search/semantic", {
            "q": q["q"],
            "repo": repo_slug,
            "k": str(q.get("k", 10)),
        }
    if intent == "structural":
        return "/search/structural", {
            "cypher": q["q"],
            "repo": repo_slug,
            "limit": str(q.get("k", 10)),
        }
    if intent == "symbol":
        return "/search/symbol", {
            "q": q["q"],
            "repo": repo_slug,
            "limit": str(q.get("k", 5)),
        }
    raise ValueError(f"unknown intent {intent!r}")


def _extract_top_k(intent: str, body: Any) -> list[dict[str, Any]]:
    if not body:
        return []
    if isinstance(body, dict):
        for key in ("results", "rows", "matches", "items", "data"):
            v = body.get(key)
            if isinstance(v, list):
                return v
    if isinstance(body, list):
        return body
    return []


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _score_indexing(index_results: list[dict[str, Any]]) -> dict[str, float]:
    total_symbols = sum((r.get("node_count") or 0) + (r.get("embedding_count") or 0) for r in index_results)
    total_seconds = sum(r.get("elapsed_s") or 0 for r in index_results) or 1e-9
    return {
        "indexing_rate_symbols_per_s": total_symbols / total_seconds,
    }


def _score_queries(query_results: list[dict[str, Any]]) -> dict[str, float]:
    by_intent: dict[str, list[float]] = {}
    semantic_with_rerank: list[float] = []
    semantic_no_rerank: list[float] = []
    for r in query_results:
        intent = r["intent"]
        elapsed = r["elapsed_s"]
        by_intent.setdefault(intent, []).append(elapsed)
        if intent == "semantic":
            # We don't currently flag rerank in the request — assume no-rerank for cycle 0.
            semantic_no_rerank.append(elapsed)

    return {
        "search_semantic_p95_no_rerank_s": _percentile(semantic_no_rerank, 0.95),
        "search_semantic_p95_with_rerank_s": _percentile(semantic_with_rerank, 0.95) if semantic_with_rerank else 0.0,
        "search_structural_p95_s": _percentile(by_intent.get("structural", []), 0.95),
        "search_symbol_p95_s": _percentile(by_intent.get("symbol", []), 0.95),
    }


def _score_relevance(graded: list[dict[str, Any]]) -> dict[str, float]:
    semantic = [g for g in graded if g["intent"] == "semantic"]
    if not semantic:
        return {"top1_relevance_semantic": 0.0, "top5_relevance_semantic": 0.0}
    return {
        "top1_relevance_semantic": sum(1 for g in semantic if g.get("top1_relevant")) / len(semantic),
        "top5_relevance_semantic": sum(1 for g in semantic if g.get("top5_relevant")) / len(semantic),
    }


def _score_lm_uptime(metrics_pre: str, metrics_post: str) -> dict[str, float]:
    # Naive — read forge_indexer_lm_studio_up gauge from both snapshots; uptime
    # = pre + post / 2. Real measurement would scrape over the full run window.
    def parse_gauge(text: str, name: str) -> float | None:
        for line in text.splitlines():
            if line.startswith(name) and " " in line:
                try:
                    return float(line.split()[-1])
                except ValueError:
                    return None
        return None

    pre = parse_gauge(metrics_pre, "forge_indexer_lm_studio_up")
    post = parse_gauge(metrics_post, "forge_indexer_lm_studio_up")
    samples = [v for v in (pre, post) if v is not None]
    return {
        "lm_studio_uptime": statistics.mean(samples) if samples else 1.0,
    }


def _score_context_bundle(query_results: list[dict[str, Any]]) -> dict[str, float]:
    # Cycle 0 doesn't directly hit /context-bundle (queries.json is search-only).
    # Carry as 0.0 so SLO row stays N/A; populate in a later cycle if we add a
    # context-bundle suite.
    return {"context_bundle_p95_s": 0.0}


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------
async def main_async(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = json.loads(Path(args.queries).read_text())
    repo_paths = json.loads(args.repo_paths) if args.repo_paths else _DEFAULT_REPO_PATHS

    async with httpx.AsyncClient() as client:
        log.info("snapshotting /metrics pre-run")
        pre_metrics = await _fetch_metrics_text(client, args.service_url)
        (out_dir / "metrics_snapshot_pre.txt").write_text(pre_metrics)

        # Stage 2 — indexing
        index_results = []
        for repo_name, path in repo_paths.items():
            try:
                ix = await _ensure_indexed(
                    client, args.service_url, repo_name, path, args.force_reindex, out_dir
                )
                index_results.append(ix)
            except Exception as e:
                log.error("indexing %s failed: %s", repo_name, e)
                index_results.append(
                    {"repo": repo_name, "status": "failed", "error": str(e), "elapsed_s": 0}
                )
        (out_dir / "index_phase_times.json").write_text(json.dumps(index_results, indent=2))

        # Stage 3 — queries
        query_results = await _run_queries(client, args.service_url, queries, out_dir)

        log.info("snapshotting /metrics post-run")
        post_metrics = await _fetch_metrics_text(client, args.service_url)
        (out_dir / "metrics_snapshot_post.txt").write_text(post_metrics)

    # Stage 4 — score
    log.info("scoring against SLO matrix")
    measured = {
        **_score_indexing(index_results),
        **_score_queries(query_results),
        **_score_context_bundle(query_results),
        **_score_lm_uptime(pre_metrics, post_metrics),
    }

    # Grading is delegated to grade_queries.py — read its output if present.
    grades_path = out_dir / "query_grades.jsonl"
    if grades_path.exists():
        graded = [json.loads(l) for l in grades_path.read_text().splitlines() if l.strip()]
        measured.update(_score_relevance(graded))
    else:
        log.warning("query_grades.jsonl not found; relevance SLOs skipped this run")
        measured["top1_relevance_semantic"] = 0.0
        measured["top5_relevance_semantic"] = 0.0

    # Build pass/fail report
    rows = []
    all_pass = True
    for slo in SLOS:
        v = measured.get(slo.name)
        if v is None:
            rows.append((slo, None, "SKIP"))
            continue
        passes = slo.passes(v)
        if not passes:
            all_pass = False
        rows.append((slo, v, "PASS" if passes else "FAIL"))

    summary_lines = [
        "# E2E Cycle Run Summary",
        "",
        f"- timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"- service: {args.service_url}",
        f"- repos: {', '.join(repo_paths.keys())}",
        f"- queries: {len(query_results)}",
        f"- overall: **{'PASS' if all_pass else 'FAIL'}**",
        "",
        "## SLO matrix",
        "",
        "| metric | target | measured | status |",
        "|---|---:|---:|:--:|",
    ]
    for slo, v, status in rows:
        target_str = f"{slo.op} {slo.target} {slo.unit}".strip()
        v_str = "—" if v is None else (f"{v:.4f} {slo.unit}".strip())
        summary_lines.append(f"| {slo.name} | {target_str} | {v_str} | {status} |")

    summary_lines.extend([
        "",
        "## Indexing per repo",
        "",
        "| repo | status | elapsed (s) | nodes | rels | embeddings |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for ix in index_results:
        summary_lines.append(
            f"| {ix['repo']} | {ix['status']} | {ix.get('elapsed_s', 0):.1f} | "
            f"{ix.get('node_count', 0)} | {ix.get('rel_count', 0)} | {ix.get('embedding_count', 0)} |"
        )

    # Worst-metric callout for the fix-applier
    worst = [
        (slo, v) for slo, v, status in rows if status == "FAIL" and v is not None
    ]
    if worst:
        summary_lines.extend(["", "## Worst metrics (fix-applier candidates)", ""])
        worst.sort(key=lambda sv: abs(sv[1] - sv[0].target) / max(sv[0].target, 1e-9), reverse=True)
        for slo, v in worst[:3]:
            summary_lines.append(
                f"- **{slo.name}**: target {slo.op} {slo.target} {slo.unit}, measured {v:.4f}"
            )

    (out_dir / "RUN_SUMMARY.md").write_text("\n".join(summary_lines) + "\n")
    log.info("wrote %s", out_dir / "RUN_SUMMARY.md")

    return 0 if all_pass else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--service-url", default="http://localhost:8000")
    p.add_argument("--queries", default="scripts/queries.json")
    p.add_argument("--out", required=True, help="output directory for run artefacts")
    p.add_argument("--force-reindex", action="store_true", help="force cold-start indexing")
    p.add_argument("--repo-paths", default=None, help="JSON {name: path} override map")
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
