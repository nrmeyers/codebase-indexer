"""Go/no-go spike: can qwen3-reranker-0.6b separate the cgr-001 storage
symbols from the tied neighbour band?

See docs/reranker-bundle-tiebreak-spike.md. Captures the live cgr-001
pre-truncation candidate set, isolates the depth-1 neighbour band that
currently ties at ``neighbor_ceiling``, scores every (query, symbol-doc)
pair with the cross-encoder via llama.cpp ``/completion`` yes/no logprobs,
and reports whether the storage symbols (``_generate_semantic_embeddings``,
``GraphUpdater.run``) separate cleanly above the noise floor.

Run with the indexer service NOT required (builds the bundle in-process),
but the reranker must be serving on :60001 (AgentAlloy). CPU-pin as usual.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_e2e import _DEFAULT_REPO_PATHS  # noqa: E402

import app.routers.context_bundle as cb  # noqa: E402
from app.routers.context_bundle import ContextBundleRequest  # noqa: E402

RERANK_URL = "http://127.0.0.1:60001/completion"
QUERY = "Re-ingest only changed files into the graph instead of a full re-parse"
REPO = "code-graph-rag"
TARGETS = ("_generate_semantic_embeddings", "GraphUpdater.run")
DOC_CAP = 1800  # chars; mirror the design snippet_cap budget

_SYS = (
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)
_INSTRUCT = (
    "Given a software design task, retrieve code symbols whose "
    "implementation is relevant to carrying out the task."
)
_YES = {"yes", "Yes", "YES", " yes", " Yes"}
_NO = {"no", "No", "NO", "not", "Not", " no", " No"}


def _prompt(query: str, doc: str) -> str:
    user = f"<Instruct>: {_INSTRUCT}\n<Query>: {query}\n<Document>: {doc}"
    return (
        f"<|im_start|>system\n{_SYS}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _logsumexp(xs: list[float]) -> float:
    if not xs:
        return -50.0
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def _score(query: str, doc: str, client: httpx.Client) -> float:
    """P(yes) over {yes,no} from the first-token logprob distribution."""
    body = {
        "prompt": _prompt(query, doc),
        "n_predict": 1,
        "temperature": 0,
        "n_probs": 25,
        "top_k": 0,
        "cache_prompt": False,
    }
    r = client.post(RERANK_URL, json=body, timeout=60)
    cp = r.json().get("completion_probabilities") or []
    if not cp:
        return float("nan")
    # llama.cpp returns OpenAI-style top_logprobs: [{token, logprob, bytes}].
    top = cp[0].get("top_logprobs") or cp[0].get("probs") or []

    def _tok(p: dict) -> str:
        return p.get("token") or p.get("tok_str") or ""

    yes_lps = [p["logprob"] for p in top if _tok(p).strip() in {t.strip() for t in _YES}]
    no_lps = [p["logprob"] for p in top if _tok(p).strip() in {t.strip() for t in _NO}]
    if not yes_lps and not no_lps:
        return float("nan")
    ly, ln = _logsumexp(yes_lps), _logsumexp(no_lps)
    return math.exp(ly) / (math.exp(ly) + math.exp(ln))


def capture_band() -> dict:
    """Build the cgr-001 bundle in-process, capturing the pre-truncation
    inputs to _truncate_to_budget."""
    cap: dict = {}
    orig = cb._truncate_to_budget

    def wrap(*, all_symbols, source_snippets, call_graph, symbol_depth, budget, scores=None):  # type: ignore[no-untyped-def]
        cap["all_symbols"] = set(all_symbols)
        cap["snippets"] = dict(source_snippets)
        cap["depth"] = dict(symbol_depth)
        cap["scores"] = dict(scores or {})
        return orig(
            all_symbols=all_symbols,
            source_snippets=source_snippets,
            call_graph=call_graph,
            symbol_depth=symbol_depth,
            budget=budget,
            scores=scores,
        )

    cb._truncate_to_budget = wrap
    try:
        rp = _DEFAULT_REPO_PATHS.get(REPO, REPO)
        resp = cb.build_context_bundle(
            ContextBundleRequest(repo_path=rp, repo=REPO, task_description=QUERY, depth=3)
        )
    finally:
        cb._truncate_to_budget = orig
    cap["survivors"] = list(resp.symbols)
    return cap


def main() -> int:
    cap = capture_band()
    scores = cap["scores"]
    depth = cap["depth"]
    snippets = cap["snippets"]
    survivors = set(cap["survivors"])

    # The targets are not in a flat tie band — they are genuinely
    # low-scored neighbours (bi-encoder never ranked storage). The real
    # test: re-score the FULL neighbour pool with the cross-encoder and
    # check whether storage rises into the surviving neighbour slots.
    neigh = {s: scores.get(s, 0.0) for s, d in depth.items() if d >= 1}
    if not neigh:
        print("no depth>=1 neighbours captured")
        return 1
    n_seeds = sum(1 for d in depth.values() if d == 0)
    n_keep = max(1, len(survivors) - n_seeds)  # surviving neighbour slots
    print(f"captured: {len(cap['all_symbols'])} symbols, "
          f"{len(neigh)} neighbours, {n_seeds} seeds, survivors={len(survivors)} "
          f"({n_keep} neighbour slots)")

    targets = [s for s in cap["all_symbols"] if any(t in s for t in TARGETS)]
    print("target storage symbols:", [t.split('.')[-1] for t in targets])
    for t in targets:
        cur_rank = sorted(neigh, key=lambda s: -neigh[s]).index(t) + 1 if t in neigh else -1
        print(f"  {t.split('.')[-1]}: depth={depth.get(t)} biE_score={scores.get(t):.4f} "
              f"biE_neighbour_rank={cur_rank}/{len(neigh)} survived={t in survivors} "
              f"snip_chars={len(snippets.get(t,''))}")

    # Score the full neighbour pool with the cross-encoder.
    score_set = list(dict.fromkeys(list(neigh) + targets))

    def doc_for(sym: str) -> str:
        tail = sym.split("::")[0].split(".")[-1]
        snip = snippets.get(sym, "")[:DOC_CAP]
        return f"{tail}\n{snip}" if snip else tail

    client = httpx.Client()
    # Determinism: 5 reps on the two targets + 3 sample noise.
    sample = targets + [s for s in neigh if s not in targets][:3]
    print("\n--- determinism (5 reps) ---")
    for s in sample:
        reps = [_score(QUERY, doc_for(s), client) for _ in range(5)]
        spread = max(reps) - min(reps)
        print(f"  {s.split('.')[-1][:40]:40} {reps[0]:.4f}  spread={spread:.2e}")

    # Full band scoring (1 rep) + latency.
    print(f"\n--- scoring full band ({len(score_set)} symbols) ---")
    t0 = time.monotonic()
    scored = []
    for s in score_set:
        sc = _score(QUERY, doc_for(s), client)
        scored.append((s, sc))
    dt = time.monotonic() - t0
    client.close()

    scored.sort(key=lambda kv: -kv[1])
    target_scores = {s: sc for s, sc in scored if s in targets}
    noise_scores = [sc for s, sc in scored if s not in targets and not math.isnan(sc)]

    print(f"latency: {dt:.1f}s total, {dt/len(score_set)*1000:.0f} ms/pair "
          f"(p50 over band, sequential)")
    print(f"noise band (n={len(noise_scores)}): "
          f"min={min(noise_scores):.4f} p50={statistics.median(noise_scores):.4f} "
          f"p90={sorted(noise_scores)[int(len(noise_scores)*0.9)]:.4f} "
          f"max={max(noise_scores):.4f}")
    print("storage targets:")
    for s, sc in sorted(target_scores.items(), key=lambda kv: -kv[1]):
        rank = [i for i, (sym, _) in enumerate(scored) if sym == s][0] + 1
        above = sum(1 for n in noise_scores if n >= sc)
        print(f"  {s.split('.')[-1]:40} score={sc:.4f} rank={rank}/{len(scored)} "
              f"noise>=it: {above}")

    # The facet grader matches substrings over surviving snippets, not just
    # the two named symbols. The real closure test: does ANY symbol carrying
    # "vector_store"/"duckdb" land in the reranked surviving slots?
    facet_terms = ("vector_store", "duckdb")
    carriers = [s for s in score_set
                if any(t in snippets.get(s, "").lower() for t in facet_terms)]
    print(f"\nfacet-string carriers in neighbour pool: {len(carriers)}")
    top_keep_list = [s for s, _ in scored[:n_keep]]
    top_keep = set(top_keep_list)
    carrier_kept = [c for c in carriers if c in top_keep]
    for c in carriers:
        rr = [i for i, (sym, _) in enumerate(scored) if sym == c][0] + 1
        print(f"  carrier {c.split('.')[-1][:36]:36} rerank={rr}/{len(scored)} "
              f"kept={c in top_keep}")
    print(f"FACET CLOSURE under reranker: {'YES' if carrier_kept else 'NO'} "
          f"({len(carrier_kept)} carrier(s) survive)")

    # Verdict: do both targets land in the top-n_keep by reranker score?
    hit = [t for t in targets if t in top_keep]
    for t in targets:
        rr = [i for i, (sym, _) in enumerate(scored) if sym == t][0] + 1
        print(f"  reranker rank of {t.split('.')[-1]}: {rr}/{len(scored)} "
              f"(need <= {n_keep})")
    print(f"\nVERDICT: cross-encoder re-rank of {len(neigh)} neighbours would keep "
          f"{len(hit)}/{len(targets)} storage targets in top-{n_keep}: "
          f"{[h.split('.')[-1] for h in hit]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
