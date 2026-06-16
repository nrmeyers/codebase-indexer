"""POST /context-bundle — build a grounded code context for the dev-agent.

The dev-agent (TheForge) needs a small, high-signal slice of the repository
to feed into an LLM prompt when implementing a task. This endpoint assembles
that slice by:

    1. Semantic search over the task description to find relevant seed
       functions/methods.
    2. Expansion through the CALLS graph up to ``depth`` hops so callees are
       included (so the LLM sees what the seed symbols actually do).
    3. Source snippet retrieval for every reached symbol.
    4. A rough token estimate so the caller can budget the prompt window.

Models are defined locally (not in ``app/models.py``) because they are
specific to this router and not reused elsewhere.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from ..config import settings
# Source-fetch helpers live in ``app.services.source_fetch`` so the
# /search/semantic rerank path can share the same FQN→snippet pipeline.
# Aliased to its old private name so existing call sites in this module
# (``_fetch_source_for_symbols``) keep working.
from ..services.source_fetch import (
    fetch_sources_for_symbols as _fetch_source_for_symbols,
)

router = APIRouter()

# Rough token estimate: ~4 characters per token for typical English/code
# mixes. Good enough for prompt-window budgeting; exact tokenization varies
# per model and is not worth pulling in a tokenizer dependency for.
_CHARS_PER_TOKEN = 4

# Soft cap on the bundle's total source-snippet token estimate.  Matches
# TheForge's tier-2 prompt cap (10k tokens) with headroom for the
# system prompt + skill fragments + user message that get prepended on
# the orchestrator side.  When the BFS expansion produces more, we
# truncate from the deepest hop inward — see ``_truncate_to_budget``.
# Set to 12_000 (vs 10_000) so the LLM gets useful context even when
# the orchestrator's other inputs are minimal; the orchestrator tier
# cap itself does the final hard clip.
_TOKEN_BUDGET = 12_000


# ---------------------------------------------------------------------------
# Test/script path down-weighting (LE-180)
# ---------------------------------------------------------------------------
#
# The reproduction bug: for an NL query like "Where is the zero-retrieval
# refusal gate implemented and what threshold does it use?", the exact-name
# and module-keyword boosts below matched calibration *scripts* (which mention
# "refusal", "gate", "threshold") and flooded the seed cap, evicting the real
# implementation that /search/semantic ranks at the top (~0.83). To match
# /search/semantic quality we (a) seed primarily from the semantic ranking
# and (b) push test/script paths *below* implementation symbols by applying a
# multiplicative score penalty — mirroring TheForge's orchestrator-side ~0.4x
# test-path multiplier.

# A qualified_name segment (or substring) that marks a symbol as living in a
# test or script file. The Code Indexer stores dotted FQNs (e.g.
# ``TheForge.scripts.calibrate-refusal.pct``,
# ``TheForge.src.foo.bar.test.helper``), so we match against the lowercased
# dotted FQN rather than a filesystem path.
_TEST_SCRIPT_MARKERS = (
    ".test.",
    ".spec.",
    ".tests.",
    "tests.",
    ".scripts.",
    "scripts.",
    "__tests__",
    "__mocks__",
    ".stories.",
    ".bench.",
    ".benchmark.",
    "conftest.",
)


def _is_test_or_script_path(qualified_name: str) -> bool:
    """Return True when a symbol's qualified name looks like a test or script.

    Conservative substring match against the lowercased dotted FQN. We bound
    the match to the start-of-FQN or a dotted boundary so we don't penalise an
    implementation symbol that merely *contains* the word "test" inside a
    longer identifier (e.g. ``runTestSuite`` in production code).

    Args:
        qualified_name: Dotted symbol FQN (e.g. ``Repo.scripts.bench.run``).

    Returns:
        True if the FQN matches any test/script marker.
    """
    lower = qualified_name.lower()
    for marker in _TEST_SCRIPT_MARKERS:
        if marker.startswith("."):
            # Dotted-boundary markers: must appear mid-FQN, not as a bare
            # substring of an identifier.
            if marker in lower:
                return True
        else:
            # Prefix-style markers (``scripts.``, ``tests.``): match at the
            # start of the FQN OR after a package boundary (``.scripts.`` is
            # already covered above; this catches ``TheForge.scripts.…`` via
            # the leading-segment form).
            if lower.startswith(marker) or ("." + marker) in lower:
                return True
    return False


# ---------------------------------------------------------------------------
# Models (local to this router — not shared in models.py)
# ---------------------------------------------------------------------------


class ContextBundleRequest(BaseModel):
    """Request body for ``POST /context-bundle``."""

    repo_path: str = Field(description="Absolute or relative path to the indexed repo")
    task_description: str = Field(description="Natural-language description of the dev task")
    k: int = Field(default=10, ge=1, le=50, description="Number of seed symbols from semantic search")
    depth: int = Field(default=2, ge=0, le=4, description="Call-graph hop depth")
    intent: str | None = Field(
        default=None,
        description=(
            "Optional retrieval intent hint. 'symbol' favours exact-name + "
            "semantic seeds with deep call-graph expansion (best for "
            "'what does X do?' queries). 'conceptual' widens seeds and "
            "prepends all functions in matched modules (best for 'how does "
            "X work?' / architectural queries). 'howto' adds router/CLI "
            "entry-point boosting on top of conceptual. When omitted the "
            "endpoint classifies from task_description. Invalid values fall "
            "through to the default (symbol) rather than erroring."
        ),
    )
    rerank: bool = Field(
        default=False,
        description=(
            "When true, run the merged seed list through CodeRankLLM "
            "(via LM Studio) before BFS expansion. Sharpens the seed "
            "set's relevance ordering so the BFS spends its hop budget "
            "on the highest-signal symbols. Best-effort — falls back "
            "silently to the bi-encoder + boost order when LM Studio "
            "isn't running."
        ),
    )

    @field_validator("repo_path")
    @classmethod
    def repo_path_must_exist(cls, v: str) -> str:
        """Reject paths that do not exist on disk early (422) rather than
        propagating a confusing 503 from the semantic search layer.
        """
        if not Path(v).exists():
            raise ValueError(f"repo_path does not exist: {v}")
        return v


class ContextBundleResponse(BaseModel):
    """Response body for ``POST /context-bundle``.

    Attributes:
        symbols: Every qualified name in the bundle, in RELEVANCE order
            (LE-182). Semantic/boost seeds come first ranked by their merged
            score (descending), then call-graph-expansion neighbours ranked by
            hop distance and originating-seed score. A consumer that truncates
            this list to a token budget by taking the front therefore keeps
            the highest-signal symbols. (Was alphabetical pre-LE-182, which
            caused the orchestrator to truncate the true top hits out of the
            prompt.)
        source_snippets: Map of qualified name → source code. Empty string
            when the symbol's file could not be read.
        call_graph: Adjacency list ``caller → [callees]`` limited to edges
            discovered during BFS expansion.
        total_tokens: Rough estimate of token cost if every snippet were
            concatenated into a prompt.
        scores: Map of qualified name → relevance score (LE-182, additive /
            backward-compatible). Higher = more relevant. Seeds carry their
            merged semantic+boost score (test/script-penalised); neighbours
            carry a derived score < their lowest seed so a downstream consumer
            can truncate or re-rank with full fidelity. Parallel to
            ``symbols``: ``symbols`` is exactly ``sorted(scores, by value
            desc)``.
    """

    symbols: list[str]
    source_snippets: dict[str, str]
    call_graph: dict[str, list[str]]
    total_tokens: int
    scores: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_conn(repo: str | None = None):  # type: ignore[override]
    """Open a LadybugDB connection with the VECTOR extension lazily loaded.

    Args:
        repo: Optional repo slug — when given, routes to that repo's
            per-repo DB file.  Falls back to the first indexed DB on disk
            or the legacy combined path.

    Returns:
        lb.Connection: A connection usable for Cypher queries.
    """
    from ..services.ladybug_pool import open_read_conn

    from pathlib import Path as _Path
    if repo:
        db_path = settings.db_path_for_repo(repo)
    else:
        db_dir = _Path(settings.LADYBUG_DB_DIR)
        dbs = sorted(db_dir.glob("*.db")) if db_dir.is_dir() else []
        db_path = str(dbs[0]) if dbs else settings.LADYBUG_DB_PATH

    # BUC-1571: read-only — context-bundle is a pure read path, no need
    # to contend with the indexer's exclusive lock.
    _db, conn = open_read_conn(db_path)
    return conn


def _result_to_rows(result: object) -> list[dict]:
    """Consume a LadybugDB result iterator into a list of column-keyed dicts."""
    rows = []
    col_names = result.get_column_names()  # type: ignore[attr-defined]
    while result.has_next():  # type: ignore[attr-defined]
        raw = result.get_next()  # type: ignore[attr-defined]
        rows.append(dict(zip(col_names, raw)))
    return rows


# Source-fetch helpers are imported at the top of the module — see
# ``..services.source_fetch`` for the implementation shared with the
# /search/semantic rerank path.


_TEST_PATH_MARKERS = (".tests.", ".test.", ".spec.")


def _is_test_symbol(qname: str) -> bool:
    """True when a qualified name points into test code."""
    return any(m in qname for m in _TEST_PATH_MARKERS) or qname.endswith(".test")


def _expand_call_graph(
    conn: object,
    seed_symbols: list[str],
    depth: int,
    *,
    caller_cap: int = 0,
    exclude_test_callees: bool = False,
) -> tuple[set[str], dict[str, list[str]], dict[str, int]]:
    """BFS over the CALLS graph up to ``depth`` hops from the seed symbols.

    Args:
        conn: An open LadybugDB connection.
        seed_symbols: Qualified names to start BFS from (seed set).
        depth: Maximum number of hops to traverse. 0 returns only seeds.
        caller_cap: When > 0, also expand up to this many *inbound* callers
            per seed (depth 1, test code excluded) and add them to the BFS
            frontier so their callees are reachable on later hops. This
            captures wiring context — e.g. the route-mounting function that
            applies auth middleware around a seeded route handler — that a
            callee-only walk can never reach.
        exclude_test_callees: When True, callees whose qname matches
            ``_is_test_symbol`` (paths containing ``.tests.`` / ``.test.`` /
            ``.spec.``, or ending in ``.test``) are dropped from the BFS
            frontier so production wiring is not crowded out by test code.

    Returns:
        tuple:
            * all_symbols: every symbol encountered (seeds + reachable).
            * call_graph: ``{caller → [callee, ...]}`` for every edge
              traversed during BFS.
            * symbol_depth: ``{symbol → depth}`` where seeds = 0,
              direct callees = 1, etc.  Lets the caller apply a
              token-budget truncation that drops the deepest hops first.
    """
    call_graph: dict[str, list[str]] = {}
    all_symbols: set[str] = set(seed_symbols)
    symbol_depth: dict[str, int] = {s: 0 for s in seed_symbols}
    frontier: set[str] = set(seed_symbols)

    # Inbound caller expansion: one hop *up* from each seed before the
    # callee walk. Callers join the frontier at depth 1, so their own
    # callees (the middleware / wiring siblings of the seed) land at
    # depth 2 within the normal budgeted walk. Capped per seed — hot
    # utilities can have hundreds of callers.
    if caller_cap > 0 and depth > 0:
        for sym in list(seed_symbols):
            if _SUMMARY_QNAME_MARKER in sym:
                continue  # summary chunks are not graph nodes
            try:
                rows = _result_to_rows(
                    conn.execute(  # type: ignore[attr-defined]
                        "MATCH (m)-[:CALLS]->(n {qualified_name: $qn}) "
                        "RETURN m.qualified_name AS caller",
                        {"qn": sym},
                    )
                )
            except Exception:
                continue
            callers = [
                r["caller"]
                for r in rows
                if r.get("caller") and not _is_test_symbol(r["caller"])
            ][:caller_cap]
            for c in callers:
                call_graph.setdefault(c, []).append(sym)
                if c not in all_symbols:
                    all_symbols.add(c)
                    symbol_depth[c] = 1
                    frontier.add(c)

    # Standard BFS: expand one hop per iteration, tracking only newly-reached
    # symbols in next_frontier to avoid revisiting.
    for hop in range(depth):
        if not frontier:
            break
        next_frontier: set[str] = set()
        for sym in frontier:
            try:
                # Parameterised Cypher: look up outgoing CALLS edges from the
                # current symbol and return callee qualified names.
                rows = _result_to_rows(
                    conn.execute(  # type: ignore[attr-defined]
                        "MATCH (n {qualified_name: $qn})-[:CALLS]->(m) "
                        "RETURN m.qualified_name AS callee",
                        {"qn": sym},
                    )
                )
            except Exception:
                # A broken symbol node should not abort the whole expansion.
                continue
            callees = [r["callee"] for r in rows if r.get("callee")]
            if exclude_test_callees:
                callees = [c for c in callees if not _is_test_symbol(c)]
            if callees:
                call_graph[sym] = callees
            for c in callees:
                if c not in all_symbols:
                    all_symbols.add(c)
                    symbol_depth[c] = hop + 1
                    next_frontier.add(c)
        frontier = next_frontier

    return all_symbols, call_graph, symbol_depth


def _truncate_to_budget(
    *,
    all_symbols: set[str],
    source_snippets: dict[str, str],
    call_graph: dict[str, list[str]],
    symbol_depth: dict[str, int],
    budget: int,
    scores: dict[str, float] | None = None,
) -> tuple[set[str], dict[str, str], dict[str, list[str]], int]:
    """Drop symbols from deepest BFS hops first until the snippet token
    estimate fits inside ``budget``.

    Algorithm:
        1. Group symbols by depth (0 = seed, 1 = direct callee, …).
        2. Starting from the maximum depth, drop entire layers until the
           token estimate is under budget OR only seeds remain.
        3. If seeds alone still exceed the budget, drop seeds one at a
           time in reverse-rank order (the merged seed list is already
           ranked, so trailing seeds are the lowest-priority).

    Args:
        all_symbols: Full symbol set from BFS.
        source_snippets: ``{symbol → source}`` for every member.
        call_graph: ``{caller → [callees]}`` from BFS.
        symbol_depth: ``{symbol → depth}`` from BFS.
        budget: Max permitted token estimate.

    Returns:
        ``(symbols, snippets, call_graph, total_tokens)`` after truncation.
        ``symbols`` is a set; the caller is responsible for sorting if
        deterministic output is desired.
    """
    if not symbol_depth:
        total = sum(len(s) for s in source_snippets.values()) // _CHARS_PER_TOKEN
        return all_symbols, source_snippets, call_graph, total

    # Working copies — we mutate these in the loop.
    kept = set(all_symbols)
    snippets = dict(source_snippets)

    # A symbol's prompt cost is its snippet PLUS its qualified name (the
    # name appears in ``symbols`` and again in call-graph edges). Without
    # the name term, empty-snippet symbols are free riders: the refill
    # pass below re-admits ALL of them at cost 0, and design bundles
    # ballooned to ~250 symbols of which ~210 were name-only noise.
    def _cost(sym: str, snip: str) -> int:
        return (len(snip) + len(sym)) // _CHARS_PER_TOKEN + 1

    # Drop deepest layers until under budget or only seeds remain.
    max_depth = max(symbol_depth.get(s, 0) for s in kept)
    current = sum(_cost(s, snippets.get(s, "")) for s in kept)

    while current > budget and max_depth > 0:
        layer = {s for s in kept if symbol_depth.get(s, 0) == max_depth}
        kept -= layer
        for s in layer:
            snippets.pop(s, None)
        current = sum(_cost(s, snippets.get(s, "")) for s in kept)
        max_depth -= 1

    # If seeds alone still bust the budget, drop trailing seeds.  Seeds
    # are kept in ``kept`` but we need a deterministic order to drop —
    # sort by qualified_name length descending (longer = more specific
    # / often less central) as a cheap heuristic.
    if current > budget:
        seeds = sorted(
            (s for s in kept if symbol_depth.get(s, 0) == 0),
            key=lambda s: (-len(s), s),
        )
        for s in seeds:
            if current <= budget:
                break
            kept.discard(s)
            removed = snippets.pop(s, "")
            current -= _cost(s, removed)

    # Refill. Layer-dropping is all-or-nothing per hop, which leaves a
    # budget cliff: going one token over budget evicts an entire depth-1
    # layer even when thousands of tokens of headroom remain afterwards.
    # Re-admit dropped symbols — shallowest hop first, then highest
    # relevance — while their snippets still fit in the remaining budget.
    if current < budget:
        ranked_scores = scores or {}
        dropped = sorted(
            all_symbols - kept,
            key=lambda s: (
                symbol_depth.get(s, 0),
                -ranked_scores.get(s, 0.0),
                s,
            ),
        )
        for s in dropped:
            snip = source_snippets.get(s, "")
            cost = _cost(s, snip)
            if current + cost <= budget:
                kept.add(s)
                snippets[s] = snip
                current += cost

    # Prune call_graph entries that reference dropped symbols.  Keep an
    # edge only when BOTH endpoints survived; otherwise the LLM would
    # see an arrow pointing into a black hole.
    pruned_graph: dict[str, list[str]] = {}
    for caller, callees in call_graph.items():
        if caller not in kept:
            continue
        kept_callees = [c for c in callees if c in kept]
        if kept_callees:
            pruned_graph[caller] = kept_callees

    return kept, snippets, pruned_graph, current


# ---------------------------------------------------------------------------
# Relevance ordering (LE-182)
# ---------------------------------------------------------------------------
#
# The endpoint historically returned ``symbols=sorted(all_symbols)`` —
# alphabetical by qualified name. The merged seed ranking computed in the
# route (semantic cosine + lexical boosts, test/script-penalised) was used
# only to pick WHICH seeds to expand, then discarded. A consumer that
# truncates the bundle to a token budget by taking it in array order would
# keep ``api-server`` / ``audit-trail`` / ``errors`` (alphabetically first)
# and drop the true top hits (e.g. ``…zero-retrieval-refusal.*``) before they
# reach the model. ``compute_symbol_scores`` rebuilds a per-symbol relevance
# score so the response can be ordered correctly AND surface scores to
# downstream consumers.

# Per-hop decay applied to a neighbour's inherited seed score. A neighbour one
# hop from a seed inherits seed_score * NEIGHBOR_HOP_DECAY; two hops, that
# squared; and so on. Chosen so even a direct callee of the strongest seed
# sorts strictly below the weakest *seed* (the seed floor is pinned below):
# pure neighbours never outrank a real seed.
_NEIGHBOR_HOP_DECAY = 0.5


def compute_symbol_scores(
    *,
    all_symbols: set[str],
    seed_scores: dict[str, float],
    call_graph: dict[str, list[str]],
    symbol_depth: dict[str, int],
) -> dict[str, float]:
    """Assign every symbol in the bundle a relevance score for ordering.

    Seeds (``symbol_depth == 0``) keep their merged semantic+boost score from
    the route (already test/script-penalised). Pure call-graph neighbours
    inherit a decayed fraction of the best score among the seeds that reach
    them, clamped strictly below the lowest seed score so seeds always precede
    neighbours regardless of their absolute decayed value.

    Args:
        all_symbols: Full symbol set (seeds + BFS neighbours).
        seed_scores: ``{seed_qn → merged_score}`` from the route's ranking.
        call_graph: ``{caller → [callees]}`` from BFS expansion.
        symbol_depth: ``{symbol → hop_distance}`` (0 = seed).

    Returns:
        ``{symbol → score}`` covering every member of ``all_symbols``. Higher
        is more relevant; seeds rank above all pure neighbours.
    """
    scores: dict[str, float] = {}

    # 1. Seeds keep their merged score. A seed missing from seed_scores
    #    (defensive — shouldn't happen) falls back to a small positive value.
    seeds = [s for s in all_symbols if symbol_depth.get(s, 0) == 0]
    for s in seeds:
        scores[s] = float(seed_scores.get(s, 0.01))

    # Floor that all neighbours must sort below. When there are no seed
    # scores (degenerate), use a small constant so neighbours still order
    # deterministically among themselves.
    seed_floor = min((scores[s] for s in seeds), default=0.01)
    # Reserve a band strictly below the lowest seed for neighbours.
    neighbor_ceiling = seed_floor * 0.99 if seed_floor > 0 else 0.0

    # 2. Neighbours: propagate the best inbound score along call-graph edges,
    #    BFS-ordered by hop depth so a parent's score is settled before its
    #    children consume it. Reverse-map callee → [callers] for lookup.
    callers_of: dict[str, list[str]] = {}
    for caller, callees in call_graph.items():
        for c in callees:
            callers_of.setdefault(c, []).append(caller)

    neighbours = sorted(
        (s for s in all_symbols if symbol_depth.get(s, 0) > 0),
        key=lambda s: symbol_depth.get(s, 0),
    )
    for sym in neighbours:
        parents = callers_of.get(sym, [])
        best_parent = max(
            (scores.get(p, 0.0) for p in parents),
            default=seed_floor,
        )
        decayed = best_parent * _NEIGHBOR_HOP_DECAY
        # Clamp into the neighbour band so no neighbour can equal/exceed a seed.
        scores[sym] = min(decayed, neighbor_ceiling) if neighbor_ceiling > 0 else decayed

    # 3. Any symbol not covered above (defensive) gets the floor of the band.
    for s in all_symbols:
        scores.setdefault(s, 0.0)

    return scores


def order_symbols_by_score(
    symbols: set[str], scores: dict[str, float]
) -> list[str]:
    """Return ``symbols`` ordered by score descending, ties broken by FQN.

    Tie-break on qualified name keeps the output deterministic (important for
    test stability and reproducible bundles).
    """
    return sorted(symbols, key=lambda s: (-scores.get(s, 0.0), s))


# ---------------------------------------------------------------------------
# Intent classification + module-level retrieval
# ---------------------------------------------------------------------------

import re as _re_intent


# Keyword triggers for conceptual queries — "how does it work", "explain",
# "overview", etc.  Match on lowercase word boundaries so we don't fire on
# substrings like "explainer" inside a symbol name.
_CONCEPTUAL_PATTERNS = (
    _re_intent.compile(r"\bhow\s+(does|do|is|are)\b", _re_intent.IGNORECASE),
    _re_intent.compile(r"\bexplain\b", _re_intent.IGNORECASE),
    _re_intent.compile(r"\boverview\s+of\b", _re_intent.IGNORECASE),
    _re_intent.compile(r"\barchitecture\b", _re_intent.IGNORECASE),
    _re_intent.compile(r"\bwhat\s+is\s+the\s+(flow|architecture|structure|design)\b", _re_intent.IGNORECASE),
    _re_intent.compile(r"\bwalk\s+me\s+through\b", _re_intent.IGNORECASE),
    # Broad-surface questions — "what tools/libraries/dependencies/modules
    # does it use", "what features does X have".  These are conceptual-
    # scope questions whose answer lives across many symbols, not one.
    # Without this pattern they hit the default symbol intent, k=12
    # semantic seeds, and the bundle misses the actual tool/module
    # definitions the user is asking about.
    _re_intent.compile(
        r"\bwhat\s+(tools|libraries|libs|dependencies|packages|modules|features|apis|endpoints|routes|parsers|adapters|services|components|tests)\b",
        _re_intent.IGNORECASE,
    ),
    _re_intent.compile(
        r"\bwhat\s+does\s+it\s+(use|leverage|depend\s+on|call|import|expose)\b",
        _re_intent.IGNORECASE,
    ),
    _re_intent.compile(r"\blist\s+(all\s+)?(the\s+)?(tools|modules|endpoints|routes|services|components|parsers|features)\b", _re_intent.IGNORECASE),
)

# Imperative feature-work phrasing — "Add X", "Extend Y", "Enforce Z on W".
# These are *design* tasks: the caller is about to modify the subsystem the
# query names, so the bundle should include the named modules' surface
# (module-keyword boost) and any indexed summary chunks, not just the
# tightest-matching function bodies. Anchored at the start of the query so
# we don't fire on incidental verbs mid-sentence ("the gate must send…").
_DESIGN_PATTERNS = (
    _re_intent.compile(
        r"^\s*(?:re-?)?(add|implement|extend|support|enforce|send|show|"
        r"resolve|build|create|wire|integrate|expose|migrate|refactor|"
        r"introduce|enable|update|allow|persist|delete|remove|stream|"
        r"ingest|compute|prioriti[sz]e|record|replace|rename|cache|batch)\b",
        _re_intent.IGNORECASE,
    ),
)

# Keyword triggers for how-to / tracing queries — these want entry-point
# handlers (HTTP routes, CLI commands) included.
_HOWTO_PATTERNS = (
    _re_intent.compile(r"\bwalk\s+me\s+through\b", _re_intent.IGNORECASE),
    _re_intent.compile(r"\bend[-\s]to[-\s]end\b", _re_intent.IGNORECASE),
    _re_intent.compile(r"\btrace\s+(the|a|an)?\b", _re_intent.IGNORECASE),
    _re_intent.compile(r"\bstep[-\s]by[-\s]step\b", _re_intent.IGNORECASE),
    _re_intent.compile(r"\bshow\s+me\s+the\s+(flow|flows?)\b", _re_intent.IGNORECASE),
)


def _classify_intent(task_description: str) -> str:
    """Classify a task description into one of three retrieval intents.

    Order matters: ``howto`` patterns are a subset of conceptual phrasing
    (e.g. "walk me through") so we check the more specific bucket first.

    Args:
        task_description: Raw natural-language query from the caller.

    Returns:
        One of ``"symbol"`` (default), ``"conceptual"``, or ``"howto"``.
    """
    if any(p.search(task_description) for p in _HOWTO_PATTERNS):
        return "howto"
    # Design before conceptual: design patterns are anchored at the start
    # of the query (imperative verb), so they are the more specific signal.
    # "Add an endpoint returning an architecture overview" is feature work
    # that mentions architecture, not an architecture question.
    if any(p.search(task_description) for p in _DESIGN_PATTERNS):
        return "design"
    if any(p.search(task_description) for p in _CONCEPTUAL_PATTERNS):
        return "conceptual"
    return "symbol"


# Words too short or too generic to make good module-name lookups.  Adding
# them would explode the boost set with irrelevant modules (every codebase
# has a "config" module, for example) and the LLM's token budget suffers.
_STOPWORDS_FOR_MODULE_LOOKUP = frozenset({
    "how", "does", "work", "the", "and", "for", "with", "from", "this",
    "that", "what", "which", "when", "where", "explain", "overview",
    "architecture", "flow", "flows", "file", "files", "code", "used",
    "uses", "using", "made", "make", "makes", "work", "works", "system",
    "leverage", "leverages", "leveraging", "forge", "repo", "repos",
    "short", "concise", "chart", "charts", "give", "tell", "show",
    "walk", "through", "trace", "step", "steps",
})


def _stem_keyword(word: str) -> str | None:
    """Return a root-form variant of a keyword when it ends in a common
    suffix, else None.

    Keeps the CONTAINS query honest when the user asks about
    "indexing" but the module is named "index.py" — without stemming,
    "indexing" CONTAINS "index" is true only in the reverse direction
    (which LadybugDB doesn't match here).  We generate BOTH forms so
    either-direction substring is covered.

    Handles the most common English suffixes empirically seen in real
    queries: -ing, -ed, -s, -er, -ion.  Deliberately skipped: -ly (too
    generic, strips meaning), -tion (rare in code vocab).

    Args:
        word: Lowercase input token.

    Returns:
        Stem form (≥3 chars) or None when no suffix applies or the
        stem would be too short to be useful.
    """
    if len(word) < 5:
        return None
    for suffix in ("ing", "ed", "er", "es", "ion", "s"):
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            # Undo single-consonant doubling ("indexing" → "index", but
            # don't produce "indexx") — conservative: only strip the
            # trailing double if the stem is still ≥3 chars after.
            if len(stem) >= 2 and stem[-1] == stem[-2] and len(stem) > 3:
                stem = stem[:-1]
            if len(stem) >= 3 and stem != word:
                return stem
    return None


def _extract_module_keywords(task_description: str) -> list[str]:
    """Pull candidate module-name keywords from a natural-language query.

    A module in LadybugDB has a ``name`` property like ``"index.py"``,
    ``"gate-state-machine.ts"``, ``"chat.ts"``.  This function extracts
    lowercase word tokens and their stem forms (so "indexing" ALSO
    queries for "index", letting us hit ``index.py`` modules).

    Deduplicates while preserving order — important for stable Cypher
    parameterisation and to keep the token budget bounded.

    Args:
        task_description: Raw query text.

    Returns:
        List of candidate keywords + stems, capped at 12.
    """
    tokens = _re_intent.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", task_description)
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        lower = t.lower()
        if lower in _STOPWORDS_FOR_MODULE_LOOKUP or lower in seen:
            continue
        seen.add(lower)
        out.append(lower)
        # Add the stem too so "indexing" reaches "index.py", "chatted"
        # reaches "chat.ts", etc.  Stem-matches are the main way to
        # resolve verb-form queries against noun-form module names.
        stem = _stem_keyword(lower)
        if stem and stem not in seen and stem not in _STOPWORDS_FOR_MODULE_LOOKUP:
            seen.add(stem)
            out.append(stem)
    # Cap at 12 (up from 10) because stems expand the list.  Kuzu's
    # UNWIND + MATCH is cheap; 12 keyword lookups still finish in
    # single-digit milliseconds against a <10k-node graph.
    return out[:12]


def _score_modules_by_keyword_coverage(
    conn: object,
    keywords: list[str],
) -> list[tuple[str, int]]:
    """Return ``[(module_qn, hit_count)]`` ranked by how many distinct query
    keywords each module's qualified_name contains.

    A module that mentions ``orchestrator`` AND ``compose`` AND ``prompt``
    is far more likely the answer to "how does the orchestrator compose
    prompts?" than one that only mentions ``orchestrator`` (which alone
    matches ``cleanup-orchestrator`` too).  This boost rescues precision
    when a single keyword is over-broad.

    Implementation: pull every module whose qualified_name contains AT
    LEAST ONE keyword (cheap UNWIND over a <10k-row table), then count
    matches in Python.  We score in Python rather than building a
    multi-keyword Cypher because LadybugDB doesn't have a native
    "count of true predicates" reducer and chained ``OR`` matches
    would re-explode the row count.

    Args:
        conn: Open LadybugDB connection.
        keywords: Lowercase keyword tokens (already de-stopworded /
            stemmed by ``_extract_module_keywords``).

    Returns:
        List of ``(module_qn, hit_count)`` sorted by hit_count descending.
        Empty on query failure or empty keyword list.
    """
    if not keywords or len(keywords) < 2:
        # Single-keyword queries can't benefit from co-occurrence scoring;
        # the caller handles them via the standard one-pass path.
        return []
    try:
        cypher = (
            "UNWIND $kws AS kw "
            "MATCH (m:Module) "
            "WHERE toLower(m.name) CONTAINS kw "
            "   OR toLower(m.qualified_name) CONTAINS ('.' + kw + '.') "
            "   OR toLower(m.qualified_name) CONTAINS ('.' + kw) "
            "RETURN DISTINCT m.qualified_name AS qn"
        )
        rows = _result_to_rows(conn.execute(cypher, {"kws": keywords}))  # type: ignore[attr-defined]
    except Exception:
        return []
    scored: list[tuple[str, int]] = []
    for r in rows:
        qn = r.get("qn") or ""
        if not qn:
            continue
        lower_qn = qn.lower()
        # Apply the same noise filters used elsewhere — there's no point
        # ranking test/fixture modules even if they match many keywords.
        if ".test" in lower_qn or "tests." in lower_qn or ".web." in lower_qn:
            continue
        hits = sum(1 for kw in keywords if kw in lower_qn)
        if hits >= 2:
            scored.append((qn, hits))
    # Highest-hit-count first; ties broken by shorter qualified_name
    # (heuristic: shallower modules are usually more authoritative than
    # deeply-nested helpers).
    scored.sort(key=lambda t: (-t[1], len(t[0])))
    return scored


def _rank_and_cap_module_hits(
    hits: list[str],
    cooccur_modules: list[str],
    per_module_cap: int,
    limit: int,
) -> list[str]:
    """Rank module-function hits by co-occurrence + apply per-module cap.

    Two-step transform:
        1. Bucket each hit by its source module (everything before the
           trailing function/method segment of its qualified_name).
        2. Order buckets so co-occurring-keyword modules come first,
           preserving original order otherwise.
        3. Round-robin through buckets emitting up to ``per_module_cap``
           hits per module until ``limit`` is reached.

    The round-robin step is what enforces breadth — without it, a single
    module that happened to be enumerated first would consume the entire
    ``limit`` even if 10 other equally-relevant modules also matched.

    Args:
        hits: Module-function qualified names from the Cypher query.
        cooccur_modules: Module qualified names ranked by multi-keyword
            co-occurrence; empty when the query has only one keyword.
        per_module_cap: Max functions emitted per source module.
        limit: Total cap on returned items.

    Returns:
        A re-ranked, capped list of qualified names.
    """
    if not hits:
        return []

    # Bucket by source module.  ``module_qn`` is everything up to the
    # last dot in the function qualified_name (parser convention).
    buckets: dict[str, list[str]] = {}
    bucket_order: list[str] = []  # original first-seen order
    for qn in hits:
        idx = qn.rfind(".")
        module_qn = qn[:idx] if idx > 0 else qn
        if module_qn not in buckets:
            buckets[module_qn] = []
            bucket_order.append(module_qn)
        buckets[module_qn].append(qn)

    # Move co-occurrence-ranked modules to the front of bucket_order
    # while preserving their relative ranking.  Anything not in
    # cooccur_modules keeps its original first-seen position.
    if cooccur_modules:
        cooccur_set = set(cooccur_modules)
        ranked_front = [m for m in cooccur_modules if m in buckets]
        rest = [m for m in bucket_order if m not in cooccur_set]
        bucket_order = ranked_front + rest

    # Round-robin emit up to per_module_cap from each bucket.  Each
    # iteration of the outer loop yields at most one item per bucket;
    # we run min(per_module_cap, max_bucket_len) outer passes.
    out: list[str] = []
    for slot in range(per_module_cap):
        for module_qn in bucket_order:
            if len(out) >= limit:
                return out
            funcs = buckets[module_qn]
            if slot < len(funcs):
                out.append(funcs[slot])
    return out


def _module_level_symbols(
    conn: object,
    keywords: list[str],
    limit: int = 25,
) -> list[str]:
    """Return function/method qualified names defined in modules whose
    name contains any of the given keywords.

    This is the core of conceptual-query retrieval — when the user asks
    "how does indexing work", semantic seeds tend to land on utility
    functions.  We want the actual router handlers defined in any module
    whose name mentions "index".  Those handlers are long (low embedding
    signal) but conceptually dominant.

    Args:
        conn: Open LadybugDB connection.
        keywords: Lowercase module-name fragments to match via CONTAINS.
        limit: Max qualified names to return (default 25 — balances
            coverage vs token budget).

    Returns:
        List of qualified names.  Empty on query failure (non-fatal —
        the caller falls back to pure semantic retrieval).
    """
    if not keywords:
        return []

    # Per-module fairness cap: keep at most this many functions from any
    # single source module.  Without this, a query whose top keyword
    # appears in one giant module (e.g. ``cleanup-orchestrator`` has 25+
    # ``buildXCategory`` helpers) consumes the entire ``limit`` and
    # starves other relevant modules from the bundle.
    PER_MODULE_CAP = 5

    # When the query has 2+ keywords, surface modules that contain
    # MULTIPLE of them first.  This is the precision booster — it
    # promotes ``src.services.orchestrator`` (matches "orchestrator" +
    # "compose" + "prompt") above ``src.services.cleanup.cleanup-orchestrator``
    # (matches only "orchestrator").
    cooccur_modules: list[str] = []
    if len(keywords) >= 2:
        scored = _score_modules_by_keyword_coverage(conn, keywords)
        cooccur_modules = [qn for qn, _ in scored]

    try:
        # LadybugDB/kuzu quirks this query has to sidestep:
        #   - `any(kw IN $list WHERE ...)` is rejected (LIST_CONTAINS
        #     binder error on string args) — use UNWIND instead.
        #   - `labels(f)` returns a single STRING, not a LIST, so
        #     `'Function' IN labels(f)` fails for the same reason.
        #     Use the multi-label `:Function|Method` pattern which
        #     kuzu supports natively.
        #
        # Filter noise INSIDE the query so the LIMIT doesn't get
        # consumed by anonymous/iife_arrow inline closures or test
        # modules.  Without the test filter, keywords like "chat" get
        # their slots consumed by fixture helpers in
        # `chat-persistence.test` etc., starving the real router.
        # We run TWO queries (backend-first, then everything else) so
        # backend code is always present in the bundle even when the
        # frontend has many more matching files — important because
        # backend code carries the business logic the LLM needs to
        # ground answers, while frontend matches tend to be shallow
        # UI wrappers.
        # Match keyword against BOTH the file basename (m.name) AND the
        # fully-qualified path (m.qualified_name).  Without the path
        # fallback, queries like "what tools does it use" miss modules
        # whose path contains the keyword as a directory segment (e.g.
        # `codebase_rag.tools.semantic_search` — the file is
        # "semantic_search.py" but it sits in a `tools/` package).
        # Over-fetch beyond ``limit`` — the fairness cap + co-occurrence
        # ranking below do the real selection.  With LIMIT == limit, a
        # single keyword whose modules enumerate first (e.g. "gate" →
        # gate-state-machine + gate-policy) consumes every slot inside the
        # query and rows from other matched modules (notification-service)
        # never reach the ranking stage at all.
        fetch_cap = int(limit) * 4
        backend_cypher = (
            "UNWIND $kws AS kw "
            "MATCH (m:Module) "
            "WHERE toLower(m.name) CONTAINS kw "
            "   OR toLower(m.qualified_name) CONTAINS ('.' + kw + '.') "
            "   OR toLower(m.qualified_name) CONTAINS ('.' + kw) "
            "MATCH (m)-[:DEFINES]->(f:Function|Method) "
            "WHERE NOT f.name STARTS WITH 'anonymous_' "
            "  AND NOT f.name STARTS WITH 'iife_arrow_' "
            "  AND NOT m.qualified_name CONTAINS '.test' "
            "  AND NOT m.qualified_name CONTAINS 'tests.' "
            "  AND NOT m.qualified_name CONTAINS '.web.' "
            "RETURN DISTINCT f.qualified_name AS qn "
            f"LIMIT {fetch_cap}"
        )
        # Second pass picks up frontend + tests ONLY if the backend
        # query left slots unfilled.  Ensures we never hit LIMIT before
        # backend routers are enumerated.
        fallback_cypher = (
            "UNWIND $kws AS kw "
            "MATCH (m:Module) "
            "WHERE toLower(m.name) CONTAINS kw "
            "   OR toLower(m.qualified_name) CONTAINS ('.' + kw + '.') "
            "   OR toLower(m.qualified_name) CONTAINS ('.' + kw) "
            "MATCH (m)-[:DEFINES]->(f:Function|Method) "
            "WHERE NOT f.name STARTS WITH 'anonymous_' "
            "  AND NOT f.name STARTS WITH 'iife_arrow_' "
            "  AND NOT m.qualified_name CONTAINS '.test' "
            "  AND NOT m.qualified_name CONTAINS 'tests.' "
            "RETURN DISTINCT f.qualified_name AS qn "
            f"LIMIT {fetch_cap}"
        )
        backend_rows = _result_to_rows(
            conn.execute(backend_cypher, {"kws": keywords})  # type: ignore[attr-defined]
        )
        backend_hits = [r["qn"] for r in backend_rows if r.get("qn")]

        # Apply both fairness cap AND co-occurrence ranking before
        # truncating to ``limit``.  The cap spreads coverage across more
        # source modules; the ranking promotes co-occurring-keyword
        # modules to the top of each bucket.
        backend_hits = _rank_and_cap_module_hits(
            backend_hits,
            cooccur_modules=cooccur_modules,
            per_module_cap=PER_MODULE_CAP,
            limit=limit,
        )

        # Short-circuit when backend alone filled the limit.
        if len(backend_hits) >= limit:
            return backend_hits[:limit]

        fallback_rows = _result_to_rows(
            conn.execute(fallback_cypher, {"kws": keywords})  # type: ignore[attr-defined]
        )
        fallback_hits = [r["qn"] for r in fallback_rows if r.get("qn")]
        # Reapply both ranking + cap to the fallback rows, then merge —
        # otherwise frontend/test slots dominate even with the backend
        # check above (e.g. when backend yielded only 3 hits, we still
        # want fairness in the fallback set).
        fallback_hits = _rank_and_cap_module_hits(
            fallback_hits,
            cooccur_modules=cooccur_modules,
            per_module_cap=PER_MODULE_CAP,
            limit=limit,
        )
        seen = set(backend_hits)
        for qn in fallback_hits:
            if qn and qn not in seen:
                seen.add(qn)
                backend_hits.append(qn)
                if len(backend_hits) >= limit:
                    break
        return backend_hits
    except Exception:
        # Non-fatal — fall through to the semantic-only path.
        return []


def _entrypoint_symbols(conn: object, limit: int = 15) -> list[str]:
    """Return likely HTTP/CLI entry-point function names.

    Heuristic — no explicit ``:EntryPoint`` label in the graph, so we look
    for functions whose name or parent module suggests they're a route
    handler or CLI command:
      - function name matches common route verbs (``get_*``, ``post_*``,
        ``create_*_router``, ``main``, ``cli``, ``handle_*``, ``run_*``)
      - OR the enclosing module has ``router``, ``cli``, ``main``, or
        ``handler`` in its name

    Used by the ``howto`` intent to ensure the flow's entry points are
    always in the context bundle, regardless of semantic rank.

    Args:
        conn: Open LadybugDB connection.
        limit: Row cap.

    Returns:
        List of qualified names; empty on query failure.
    """
    try:
        # Use the `:Function|Method` multi-label syntax (kuzu-native) so
        # we don't hit the `labels(f)` LIST_CONTAINS binder error that
        # standard-Cypher-style label checks trigger on LadybugDB.
        # Filter anonymous/iife inline closures so they don't consume
        # LIMIT slots ahead of real entry-point handlers.
        cypher = (
            "MATCH (m:Module)-[:DEFINES]->(f:Function|Method) "
            "WHERE NOT f.name STARTS WITH 'anonymous_' "
            "  AND NOT f.name STARTS WITH 'iife_arrow_' "
            "  AND ("
            "     toLower(m.name) CONTAINS 'router' "
            "  OR toLower(m.name) CONTAINS 'route' "
            "  OR toLower(m.name) CONTAINS 'cli' "
            "  OR toLower(m.name) CONTAINS 'main' "
            "  OR toLower(m.name) CONTAINS 'handler' "
            "  OR toLower(f.name) STARTS WITH 'handle_' "
            "  OR toLower(f.name) STARTS WITH 'create_' "
            "  OR toLower(f.name) = 'main' "
            "  ) "
            "RETURN DISTINCT f.qualified_name AS qn "
            f"LIMIT {int(limit)}"
        )
        rows = _result_to_rows(conn.execute(cypher))  # type: ignore[attr-defined]
        return [r["qn"] for r in rows if r.get("qn")]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Summary-chunk snippet hydration (Priority 2 — design-context gaps)
# ---------------------------------------------------------------------------
#
# Hierarchical summary chunks (``{qname}::Class::summary``,
# ``{module}::Module::summary``) live only in the DuckDB vector store — they
# are not LadybugDB graph nodes, so ``fetch_sources_for_symbols`` returns an
# empty snippet for them and the bundle carried only their label. The vector
# store row records ``file_path``/``start_line``/``end_line``, which is enough
# to hydrate a real snippet: the head of the span (class signature + docstring
# + early members, or a package ``__init__``'s import/export surface) is the
# high-signal part.

_SUMMARY_QNAME_MARKER = "::summary"

from ..services.symbol_cards import (
    SYMBOL_CARD_MARKER as _SYMBOL_CARD_MARKER,
    fold_card_qname as _card_parent,
)

# Max lines of the span head included per summary snippet. Class spans can run
# to hundreds of lines; the leading lines carry the signature, docstring, and
# (for __init__.py modules) the import/__all__ surface, which is what a
# design-task consumer needs from a *summary* chunk.
_SUMMARY_SNIPPET_MAX_LINES = 40


def _hydrate_summary_snippets(
    repo_slug: str, qnames: list[str]
) -> dict[str, str]:
    """Build source snippets for summary-chunk qualified names.

    Args:
        repo_slug: Repo slug used to locate the per-repo ``.duck`` file.
        qnames: Qualified names containing ``::summary`` to hydrate.

    Returns:
        ``{qname → snippet}`` for every qname whose vector-store row and
        backing file could both be read. Missing rows/files are silently
        skipped (best-effort — the bundle still carries the label).
    """
    if not qnames:
        return {}
    try:
        import duckdb  # noqa: PLC0415

        duck_path = settings.vec_db_path_for_repo(repo_slug)
        if not Path(duck_path).exists():
            return {}
        con = duckdb.connect(duck_path, read_only=True)
        try:
            rows = con.execute(
                "SELECT qualified_name, file_path, start_line, end_line "
                "FROM embeddings WHERE qualified_name IN "
                f"({','.join('?' * len(qnames))})",
                qnames,
            ).fetchall()
        finally:
            con.close()
    except Exception:
        return {}

    out: dict[str, str] = {}
    for qn, file_path, start, end in rows:
        if not file_path:
            continue
        try:
            text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = text.splitlines()
        start_i = max(int(start or 1) - 1, 0)
        end_i = min(int(end or len(lines)), len(lines))
        span = lines[start_i:end_i]
        truncated = len(span) > _SUMMARY_SNIPPET_MAX_LINES
        span = span[:_SUMMARY_SNIPPET_MAX_LINES]
        if "::Module::" in qn:
            kind = "Module"
        elif "::File::" in qn:
            kind = "File"
        elif ".md::" in qn:
            kind = "Doc"
        else:
            kind = "Class"
        header = f"# {kind} summary — {file_path}:{start}-{end}"
        body = "\n".join(span)
        if truncated:
            body += "\n# … (span truncated)"
        out[qn] = f"{header}\n{body}"
    return out


def _lexical_seed_hits(repo_slug: str, query: str, limit: int) -> list[dict]:
    """Top Tantivy BM25 hits for the raw task description.

    Some facets are reachable by NO other leg: signal that lives only in
    comments (e.g. an auth provider documented as "AAD"/"MSAL" in a file
    header) embeds too weakly to seed semantically, and by-reference
    wiring (``router.use(requireIdentity)``) produces no CALLS edge for
    the graph walk. BM25 over the lexical index is the only surface that
    sees raw token matches in bodies and comments.

    Best-effort: any failure (missing index, tantivy unavailable) returns
    ``[]`` so the bundle degrades to the semantic + boost legs.

    Args:
        repo_slug: Canonical repo slug (locates ``<slug>.tantivy/``).
        query: Free-text task description, passed verbatim to BM25.
        limit: Target seed count; up to ``2 * limit`` candidates return
            so the caller can filter noise and still fill its slots.

    Returns:
        Dicts with ``qn`` / ``file_path`` / ``start_line`` / ``end_line``
        / ``kind`` in BM25 rank order — the span metadata lets the caller
        hydrate snippets for hits that exist only in the lexical index
        (markdown chunks have no graph node and may have no duck row).
    """
    if limit <= 0 or not query.strip():
        return []
    try:
        from .search import lexical_search  # noqa: PLC0415

        # Over-fetch heavily: the caller filters noise/test paths after us
        # and slices to ``limit``, and the diversity caps below skip hits.
        resp = lexical_search(q=query, repo=repo_slug, limit=min(limit * 8, 60))

        # Diversity caps. Markdown chunks (term-rich prose) BM25-dominate
        # code symbols, and section chunks of one doc arrive as a block —
        # without caps a single ADR fills every slot. Markdown stays
        # valuable (it IS design context) but must not crowd out code.
        per_file_cap = 2
        md_quota = max(1, limit // 2)

        out: list[dict] = []
        seen: set[str] = set()
        per_file: dict[str, int] = {}
        n_md = 0
        for h in resp.results:
            qn = h.symbol_qname
            if not qn or qn in seen:
                continue
            if per_file.get(h.file_path, 0) >= per_file_cap:
                continue
            is_md = h.symbol_kind == "MarkdownDoc"
            if is_md and n_md >= md_quota:
                continue
            seen.add(qn)
            per_file[h.file_path] = per_file.get(h.file_path, 0) + 1
            if is_md:
                n_md += 1
            out.append(
                {
                    "qn": qn,
                    "file_path": h.file_path,
                    "start_line": h.start_line,
                    "end_line": h.end_line,
                    "kind": h.symbol_kind,
                }
            )
            if len(out) >= limit * 2:
                break
        return out
    except Exception:
        return []


# Per-intent retrieval parameters — a single knob per intent that the
# route handler reads to shape the bundle.  Changing these values tunes
# the breadth/depth tradeoff without touching control flow.
_INTENT_PARAMS: dict[str, dict[str, int | bool]] = {
    # Current default — deep call-graph walk off tight seeds.
    "symbol":     {"k": 12, "depth": 3, "module_limit": 0,  "entrypoint_limit": 0},
    # Wider seeds + full module inclusion; shallower expansion keeps
    # token count manageable when whole routers are dropped in.
    "conceptual": {"k": 20, "depth": 2, "module_limit": 25, "entrypoint_limit": 0},
    # Like conceptual but also pulls in entry-point handlers so the
    # flow's starting points are always grounded.
    "howto":      {"k": 15, "depth": 3, "module_limit": 20, "entrypoint_limit": 10},
    # Imperative feature work ("Add…", "Extend…"). Keeps symbol-intent
    # seed/depth but adds a moderate module-keyword boost so the files the
    # task names by topic (e.g. "notifications" → notification-service.ts)
    # are present even when no single function embeds close to the query.
    # caller_cap: design tasks need *wiring* context (who mounts/calls the
    # seeded handlers — middleware, auth guards, registration) that a
    # callee-only walk can never reach.
    # lexical_limit: BM25 seed leg — catches comment-only and identifier
    # signal (AAD/MSAL in headers, by-reference middleware) invisible to
    # both the dense embedding and the CALLS graph.
    # snippet_cap_chars: design bundles are budget-saturated — one 11k-char
    # snippet costs ~25% of the 12k-token budget and evicts the depth-2/3
    # layer where cross-facet context lives (vector_store was reachable but
    # always truncated out of dsg-cgr-001 bundles). Capping per-snippet size
    # trades tail-of-function detail for layer survival.
    # exclude_test_callees: the callee walk has no test filter (only inbound
    # callers are filtered), so design bundles flooded with tests.* symbols
    # that carry no design signal.
    "design":     {"k": 12, "depth": 2, "module_limit": 15, "entrypoint_limit": 0, "caller_cap": 3, "lexical_limit": 6, "snippet_cap_chars": 2000, "exclude_test_callees": True},
}


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/context-bundle", response_model=ContextBundleResponse)
def build_context_bundle(req: ContextBundleRequest) -> ContextBundleResponse:
    """Build a grounded code context bundle for a dev-agent task prompt.

    Steps:
        1. Semantic search: find top-k functions/methods most relevant to
           the task.
        2. Expand via CALLS graph up to ``depth`` hops.
        3. Fetch source snippets for every symbol in the expanded set.
        4. Return ``{symbols, source_snippets, call_graph, total_tokens}``.

    Args:
        req: Validated request body with repo path, task description, k,
            and depth parameters.

    Returns:
        ContextBundleResponse: The assembled bundle; empty fields when the
        seed search returns no matches.

    Raises:
        HTTPException: 503 when semantic search is unavailable.
    """
    # Merged seed scores survive out of the seeding ``try`` block so the
    # final response can be ordered by relevance and surface per-symbol
    # scores (LE-182). Keyed by the qualified names actually chosen as seeds.
    seed_scores: dict[str, float] = {}
    # Span metadata for lexical-leg seeds — survives the ``try`` so the
    # snippet-hydration step below can render markdown hits that exist
    # only in the Tantivy index (no graph node, no duck row).
    lexical_seed_meta: dict[str, dict] = {}

    # 0. Classify retrieval intent and apply per-intent k/depth overrides.
    # The caller can pin an intent explicitly (e.g. an agent that already
    # knows it's doing a conceptual summary); otherwise we classify from
    # the task description using regex heuristics — zero-latency, no ML.
    intent = req.intent if req.intent in _INTENT_PARAMS else None
    if intent is None:
        intent = _classify_intent(req.task_description)
    intent_params = _INTENT_PARAMS.get(intent, _INTENT_PARAMS["symbol"])
    # The intent's k/depth only upgrade the caller's request — never
    # downgrade.  This preserves explicit caller intent (e.g. k=30 for
    # a heavy-context agent turn) while still broadening when the user
    # left the defaults in place.  Explicit zeros are honoured as
    # opt-outs (e.g. depth=0 → no CALLS expansion at all).
    effective_k = req.k if req.k == 0 else max(req.k, intent_params["k"])
    effective_depth = (
        req.depth if req.depth == 0 else max(req.depth, intent_params["depth"])
    )

    # 1. Semantic seed — find the most task-relevant functions/methods.
    # Point code-graph-rag at the per-repo DB *before* the semantic search
    # so the vector_store shim resolves to the correct ``<slug>.duck`` file.
    # (Historical note: this used to read from ``<slug>.embeddings.npy``
    # before the DuckDB swap retired the numpy backend.)
    # Canonical {org}__{repo} slug from the git remote — the dir basename
    # alone misses indexes keyed by /index's slug derivation (BUC-1580) and
    # 503s on repos indexed from a local checkout.
    from ..services.slug import derive_slug

    _resolved = Path(req.repo_path).resolve()
    repo_slug = derive_slug(_resolved, _resolved.name)
    # If the canonical slug has no index but the basename does (e.g. the
    # checkout's remote changed after indexing), use the index that exists.
    if (
        not Path(settings.db_path_for_repo(repo_slug)).exists()
        and Path(settings.db_path_for_repo(_resolved.name)).exists()
    ):
        repo_slug = _resolved.name
    _repo_db = settings.db_path_for_repo(repo_slug)
    try:
        from codebase_rag.config import settings as _cgr_settings  # type: ignore[import-untyped]
        _cgr_settings.LADYBUG_DB_PATH = _repo_db
    except Exception:
        pass  # non-fatal; falls back to default

    try:
        # LE-180: seed from the SAME ranking the HTTP /search/semantic surface
        # uses, not from ``codebase_rag.tools.semantic_search`` directly.
        #
        # Root cause of the seed-quality bug: ``semantic_code_search`` embeds
        # the query with ``codebase_rag.embedder.embed_query`` (the legacy
        # in-package embedder), while the ``.duck`` vector store was written by
        # ``app.embedders`` (the configured backend, e.g. local E5). The two
        # live in different embedding spaces, so the cosine scores from
        # ``semantic_code_search`` are near-degenerate (~0.12) and rank
        # script/test files that merely mention the query terms above the real
        # implementation. The retrieval service embeds via
        # ``app.embedders.sync_bridge`` (matching the index) AND applies the
        # descriptive-query rewriter + PageRank + RRF/BM25 fusion — which is
        # exactly why /search/semantic ranks the implementation at ~0.83.
        # Reusing it makes the bundle's seeds match that quality.
        from app.services.retrieval import semantic_search as semantic_search_service  # noqa: PLC0415

        # Request a deep result pool so the score-based merge + test/script
        # down-weight below has room to rank past the script cluster.
        # The retrieval service is called directly (not via the HTTP surface),
        # so the route's ``k <= 100`` Query bound does not apply; it
        # over-fetches ``k * 50`` candidates internally before de-noising and
        # slicing to ``k``.
        _seed_pool = max(effective_k * 5, 100)
        _sem_resp = semantic_search_service(
            q=req.task_description,
            k=_seed_pool,
            repo=repo_slug,
            rerank=bool(req.rerank),
        )
        # Adapt the SemanticResult list to the {qualified_name, score} shape
        # the existing de-noise + ranking code consumes. ``_sem_resp.results``
        # is already in final (fused/reranked) rank order — preserve it
        # verbatim; re-sorting by the raw cosine ``.score`` would revert that
        # ordering and change which symbols seed. Symbol-card hits are folded
        # to their parent upstream in ``_semantic_search_impl``, so no card
        # qname reaches here.
        seed_results = [
            {"qualified_name": r.symbol, "score": r.score}
            for r in _sem_resp.results
        ]
        # Drop noise: test fixtures AND anonymous inline arrows/callbacks
        # (named `anonymous_LINE_COL` by the parser). Both have degenerate
        # embeddings that crowd real code out of the top-k window.
        _FIXTURE_SEGMENTS = {"fixtures", "large-file", "__fixtures__"}
        import re as _re
        _ANON_RE = _re.compile(r"^anonymous_\d+_\d+$")

        def _is_noise(sym: str) -> bool:
            parts = sym.split(".")
            if any(seg in _FIXTURE_SEGMENTS for seg in parts):
                return True
            if any(_ANON_RE.match(seg) for seg in parts):
                return True
            # Duplicated trailing segments from inner-scope closures.
            if len(parts) >= 2 and parts[-1] == parts[-2]:
                return True
            return False

        # Preserve the bi-encoder cosine score alongside each FQN so seed
        # ranking below mirrors /search/semantic (which ranks the actual
        # implementation at the top) rather than discarding scores and
        # letting lexical boosts dominate by mere arrival order. We keep the
        # full de-noised, score-ordered list (already sorted by the vector
        # store) and slice to effective_k *after* applying the test/script
        # down-weight — so an implementation symbol can't be pushed out of the
        # window by a higher-arriving script.
        semantic_ranked: list[tuple[str, float]] = []
        for rank, r in enumerate(seed_results):
            qn = r["qualified_name"]
            if _is_noise(qn):
                continue
            # ``score`` is the cosine similarity from the vector store
            # (higher = more relevant). When absent, synthesise a
            # monotonically-decreasing score from rank so ordering is stable.
            raw_score = r.get("score")
            score = float(raw_score) if isinstance(raw_score, (int, float)) else max(0.0, 1.0 - rank * 0.001)
            semantic_ranked.append((qn, score))

        seed_symbols = [qn for qn, _ in semantic_ranked][: effective_k]

        # Open a connection for the boost queries below.  Shared across
        # the three boosters (exact-name, module-level, entry-points) so
        # we don't pay LadybugDB connect cost three times.
        boost_conn: object | None = None
        try:
            boost_conn = _get_conn(repo_slug)
        except Exception:
            boost_conn = None  # non-fatal — boosts will no-op below

        # Exact-name boost: when the task description mentions a symbol name
        # verbatim (e.g. "what does createGateStateMachine do"), force-include
        # every indexed symbol whose trailing segment matches that word —
        # semantic embeddings alone can miss long functions whose bodies are
        # lexically distant from short natural-language prompts.
        exact_hits: list[str] = []
        try:
            import re as _re_name
            _words = set(
                _re_name.findall(
                    r"[A-Za-z_][A-Za-z0-9_]{2,}",
                    req.task_description,
                )
            )
            if _words and boost_conn is not None:
                exact_rows = _result_to_rows(
                    boost_conn.execute(  # type: ignore[attr-defined]
                        "MATCH (n) WHERE n.name IN $names "
                        "RETURN DISTINCT n.qualified_name AS qn",
                        {"names": list(_words)},
                    )
                )
                exact_hits = [
                    r["qn"] for r in exact_rows
                    if r.get("qn") and not _is_noise(r["qn"])
                ]
        except Exception:
            # Exact-match boost is best-effort — never fail the bundle on it.
            pass

        # Intent-driven boosts — add module-wide function coverage for
        # conceptual/howto queries, and entry-point handlers for howto.
        # These are what lets "how does indexing work?" return the actual
        # router handler bodies rather than just utilities.
        module_hits: list[str] = []
        entrypoint_hits: list[str] = []
        if boost_conn is not None and intent_params["module_limit"] > 0:
            keywords = _extract_module_keywords(req.task_description)
            module_hits = [
                qn for qn in _module_level_symbols(
                    boost_conn, keywords, limit=intent_params["module_limit"],
                )
                if not _is_noise(qn)
            ]
        if boost_conn is not None and intent_params["entrypoint_limit"] > 0:
            entrypoint_hits = [
                qn for qn in _entrypoint_symbols(
                    boost_conn, limit=intent_params["entrypoint_limit"],
                )
                if not _is_noise(qn)
            ]

        # Lexical (BM25) seed leg — raw token matches over bodies AND
        # comments. The only leg that reaches comment-only facets and
        # by-reference wiring (no CALLS edge, weak embedding).
        lexical_hits: list[str] = []
        _lex_limit = int(intent_params.get("lexical_limit", 0))
        if _lex_limit > 0:
            for _lh in _lexical_seed_hits(
                repo_slug, req.task_description, _lex_limit,
            ):
                if len(lexical_hits) >= _lex_limit:
                    break
                if _is_noise(_lh["qn"]) or _is_test_or_script_path(_lh["qn"]):
                    continue
                lexical_hits.append(_lh["qn"])
                lexical_seed_meta[_lh["qn"]] = _lh

        # Score-based seed merge (LE-180). Previously this prepended the
        # lexical boosts (exact-name / module-keyword / entry-point) AHEAD of
        # the semantic seeds and capped by arrival order. For a query like
        # "where is the zero-retrieval refusal gate implemented and what
        # threshold does it use?" the exact-name boost matched *scripts* that
        # merely mention "refusal" / "gate" / "threshold" (calibrate-refusal,
        # latency-bench, …) and flooded the cap, evicting the real
        # implementation that /search/semantic ranks at the top (~0.83).
        #
        # New approach — assign every candidate a comparable score, then sort:
        #   * semantic seeds keep their bi-encoder cosine score (the same
        #     signal that makes /search/semantic correct);
        #   * lexical boosts are *augmentation* — they get a base score placed
        #     just below the strongest semantic hit so they enrich the bundle
        #     (long handlers with low embedding signal, exact-name asks) WITHOUT
        #     outranking a clear semantic winner;
        #   * every candidate whose FQN looks like a test/script file is
        #     multiplied by CONTEXT_BUNDLE_TEST_PATH_PENALTY (~0.4) so it sorts
        #     below real implementation — mirroring TheForge's orchestrator.
        # The BFS then expands from genuinely high-signal seeds.
        scored_seeds: dict[str, float] = {}

        # Top semantic score anchors the boost band. When semantic returned
        # nothing (degenerate repo), fall back to 1.0 so boosts still rank.
        top_semantic = semantic_ranked[0][1] if semantic_ranked else 1.0

        for qn, score in semantic_ranked[: effective_k * 2]:
            scored_seeds[qn] = max(scored_seeds.get(qn, 0.0), score)

        # Boost base scores sit just under the top semantic hit so a strong
        # implementation match always leads, but boosts still beat the long
        # tail of weak semantic seeds. Exact-name asks rank highest among
        # boosts (the user typed the symbol), then module handlers, then
        # generic entry points.
        #
        # E5-family cosine scores are tightly compressed (the whole candidate
        # pool typically spans ~0.78–0.87), so multiplicative bands off the
        # top (``top * 0.90``) land BELOW nearly every semantic candidate and
        # the module/entry-point boosts get sliced out of the seed window
        # entirely. Anchor those bands to the MIDDLE of the semantic seed
        # window instead: boosts displace the weak semantic tail, never the
        # head. Clamped under the exact band so ordering between boost kinds
        # is preserved.
        mid_idx = (
            min(max(effective_k // 2 - 1, 0), len(semantic_ranked) - 1)
            if semantic_ranked else 0
        )
        mid_semantic = semantic_ranked[mid_idx][1] if semantic_ranked else 1.0
        _BOOST_BASE = {
            "exact": top_semantic * 0.98,
            # BM25 over the verbatim query is more query-specific than the
            # keyword module round-robin — band it between exact and module.
            "lexical": min(top_semantic * 0.975, mid_semantic * 1.0025),
            "module": min(top_semantic * 0.97, mid_semantic * 1.002),
            "entrypoint": min(top_semantic * 0.96, mid_semantic * 1.001),
        }
        # Within each boost class, decay by list position rather than scoring
        # every hit identically. ``_module_level_symbols`` round-robins across
        # source modules, so its head is one function per matched module —
        # flat scores would hand the tie-break to the FQN sort and let a
        # single keyword's modules (``gate-*``) alphabetically crowd out the
        # others (``notification-service``) when the seed cap bites.
        _POSITION_DECAY = 1e-4
        for i, qn in enumerate(exact_hits):
            s = _BOOST_BASE["exact"] * (1.0 - i * _POSITION_DECAY)
            scored_seeds[qn] = max(scored_seeds.get(qn, 0.0), s)
        for i, qn in enumerate(lexical_hits):
            s = _BOOST_BASE["lexical"] * (1.0 - i * _POSITION_DECAY)
            scored_seeds[qn] = max(scored_seeds.get(qn, 0.0), s)
        for i, qn in enumerate(module_hits):
            s = _BOOST_BASE["module"] * (1.0 - i * _POSITION_DECAY)
            scored_seeds[qn] = max(scored_seeds.get(qn, 0.0), s)
        for i, qn in enumerate(entrypoint_hits):
            s = _BOOST_BASE["entrypoint"] * (1.0 - i * _POSITION_DECAY)
            scored_seeds[qn] = max(scored_seeds.get(qn, 0.0), s)

        # Apply the test/script down-weight. Clamp the penalty to a sane
        # (0, 1] band; out-of-range config falls back to 0.4.
        _penalty = settings.CONTEXT_BUNDLE_TEST_PATH_PENALTY
        if not (0.0 < _penalty <= 1.0):
            _penalty = 0.4
        for qn in list(scored_seeds):
            if _is_test_or_script_path(qn):
                scored_seeds[qn] *= _penalty

        # Sort by adjusted score (desc); break ties on FQN for determinism.
        ranked = sorted(scored_seeds.items(), key=lambda kv: (-kv[1], kv[0]))

        # Cap at 2× effective_k so a wide module boost on a small repo doesn't
        # blow up the BFS frontier.
        seed_cap = min(
            max(
                effective_k,
                len(exact_hits) + len(lexical_hits)
                + len(module_hits) + len(entrypoint_hits),
            ),
            effective_k * 2,
        )
        # Summary chunks augment, never displace. A topic with many matching
        # file headers (parser-heavy queries match a dozen ``::File::summary``
        # rows) floods the seed window with orientation text, evicting the
        # implementation symbols the bundle exists to carry. Cap them DURING
        # selection so the freed slots backfill with the next-ranked
        # implementation candidates rather than shrinking the seed set.
        _MAX_SUMMARY_SEEDS = 4

        def _select_seeds(cands: list[str], cap_total: int) -> list[str]:
            out: list[str] = []
            n_summary = 0
            for qn in cands:
                if len(out) >= cap_total:
                    break
                if _SUMMARY_QNAME_MARKER in qn:
                    if n_summary >= _MAX_SUMMARY_SEEDS:
                        continue
                    n_summary += 1
                out.append(qn)
            return out

        seed_symbols = _select_seeds([qn for qn, _ in ranked], seed_cap)

        # Guaranteed boost quota. Score bands alone cannot ensure module /
        # entry-point hits survive the cap: E5 score distributions are often
        # flat enough that 15+ semantic candidates sit above any band we can
        # safely place, slicing every boost hit out. Reserve a fixed slice of
        # the cap for the head of the boost lists (which ``_module_level_-
        # symbols`` round-robins across source modules, so the head is one
        # function per matched module — maximum breadth per slot).
        # The quota is ADDITIVE up to the 2× hard cap: evicting ranked seeds
        # to make room trades one facet for another (observed: the websocket
        # seed covering a query's transport aspect dropped to admit module
        # hits). Only when the hard cap leaves no headroom do guaranteed
        # boosts displace the ranked tail.
        # Lexical hits get their own (small) guaranteed slice ahead of the
        # module/entry-point quota: the entire point of the BM25 leg is that
        # these symbols carry NO competitive semantic score, so a band alone
        # cannot keep them inside the cap. The slice is additive — it never
        # shrinks the existing module/entry-point quota.
        boost_ordered = module_hits + entrypoint_hits
        if boost_ordered or lexical_hits:
            quota = min(len(boost_ordered), max(seed_cap // 3, 6))
            _lex_set = set(lexical_hits)
            guaranteed = lexical_hits + [
                qn for qn in boost_ordered[:quota] if qn not in _lex_set
            ]
            gset = set(guaranteed)
            head = [qn for qn, _ in ranked if qn not in gset]
            hard_cap = effective_k * 2
            head_keep = min(seed_cap, max(hard_cap - len(guaranteed), 1))
            seed_symbols = _select_seeds(head, head_keep) + guaranteed

        # Optional listwise rerank of the merged seed set (disabled by default).
        # When RERANK_ENABLED=true, this runs BEFORE BFS expansion so the
        # call-graph walk spends its hop budget on the most-relevant seeds;
        # reordering after expansion would already have wasted hops on weak
        # seeds. Best-effort — any failure leaves seed_symbols in the merged
        # boost+semantic order. LM Studio was retired (TheForge PR #168);
        # future rerank implementations will wire via a different backend.
        if settings.RERANK_ENABLED and req.rerank and seed_symbols:
            try:
                from ..services import reranker  # noqa: WPS433
                cand = [{"qualified_name": qn} for qn in seed_symbols]
                reordered = reranker.rerank(req.task_description, cand)
                if reordered and len(reordered) == len(cand):
                    seed_symbols = [c["qualified_name"] for c in reordered]
            except Exception:
                # Non-fatal — preserve merged order on any rerank failure.
                pass

        # Capture the merged relevance score for each chosen seed so the
        # response can be ordered + scored (LE-182). When a listwise rerank
        # reordered ``seed_symbols``, mirror that ordering in the scores by
        # assigning a monotonically-decreasing score (the rerank intent is
        # that earlier = more relevant); otherwise keep the merged scores so
        # the absolute values stay meaningful to downstream consumers.
        if settings.RERANK_ENABLED and req.rerank and seed_symbols:
            n = len(seed_symbols)
            for i, qn in enumerate(seed_symbols):
                # Spread reranked seeds across (0, top_semantic] preserving order.
                seed_scores[qn] = top_semantic * (1.0 - i / (n + 1))
        else:
            for qn in seed_symbols:
                seed_scores[qn] = scored_seeds.get(qn, 0.01)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Semantic search unavailable: {exc}",
        ) from exc

    # No seed matches → return an empty bundle rather than an error; the
    # caller can decide whether to widen the task description.
    if not seed_symbols:
        return ContextBundleResponse(
            symbols=[],
            source_snippets={},
            call_graph={},
            total_tokens=0,
        )

    # 2. Expand call graph to pick up callees the LLM will need to reason
    #    about.  Depth is intent-driven: conceptual queries stay shallow
    #    (we already have whole modules via the boost) while symbol and
    #    howto queries dig deeper into callee chains.
    conn = _get_conn(repo_slug)
    all_symbols, call_graph, symbol_depth = _expand_call_graph(
        conn, seed_symbols, effective_depth,
        caller_cap=intent_params.get("caller_cap", 0),
        exclude_test_callees=bool(
            intent_params.get("exclude_test_callees", False)
        ),
    )

    # 3. Fetch source snippets — sorted for deterministic output.
    source_snippets = _fetch_source_for_symbols(conn, sorted(all_symbols))

    # 3b. Summary chunks (``::Class::summary`` / ``::Module::summary``) and
    #     markdown doc chunks (``{slug}::{path}.md::{section}`` — lexical
    #     seed leg) are vector-store-only — the graph fetch above leaves
    #     them empty. Hydrate their snippets from the duck row's file span
    #     so the bundle carries the text, not just a label.
    summary_qnames = [
        s for s in all_symbols
        if (_SUMMARY_QNAME_MARKER in s or ".md::" in s)
        and not source_snippets.get(s)
    ]
    if summary_qnames:
        source_snippets.update(
            _hydrate_summary_snippets(repo_slug, summary_qnames)
        )

    # 3c. Lexical-leg seeds that STILL have no snippet (markdown chunks
    #     live only in the Tantivy index — no graph node, and repos
    #     indexed before the markdown embed pass have no duck row either)
    #     hydrate straight from the span metadata Tantivy stored.
    for _qn, _meta in lexical_seed_meta.items():
        if _qn not in all_symbols or source_snippets.get(_qn):
            continue
        _fp = _meta.get("file_path") or ""
        if not _fp:
            continue
        _p = Path(_fp)
        if not _p.is_absolute():
            _p = Path(req.repo_path) / _p
        try:
            _lines = _p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        _sl = max(int(_meta.get("start_line") or 1), 1)
        _el = min(int(_meta.get("end_line") or len(_lines)), len(_lines))
        _span = _lines[_sl - 1 : _el]
        _truncated = len(_span) > _SUMMARY_SNIPPET_MAX_LINES
        _body = "\n".join(_span[:_SUMMARY_SNIPPET_MAX_LINES])
        if _truncated:
            _body += "\n# … (span truncated)"
        source_snippets[_qn] = f"# Doc — {_fp}:{_sl}-{_el}\n{_body}"

    # 3d. Per-snippet cap (design intent). Budget-saturated bundles are a
    #     zero-sum game: one whole-function snippet at 11k chars costs a
    #     quarter of the token budget and evicts entire BFS layers. Clip
    #     each snippet at a line boundary so breadth (layer survival)
    #     wins over tail-of-function detail.
    _snip_cap = int(intent_params.get("snippet_cap_chars", 0))
    if _snip_cap > 0:
        for _qn, _snip in source_snippets.items():
            if len(_snip) <= _snip_cap:
                continue
            _cut = _snip.rfind("\n", 0, _snip_cap)
            if _cut <= 0:
                _cut = _snip_cap
            source_snippets[_qn] = (
                _snip[:_cut] + "\n# … (snippet capped)"
            )

    # 4. Token-budget truncation — drop deepest-hop symbols first when
    #    the bundle would exceed ``_TOKEN_BUDGET``.  Seeds (depth 0) are
    #    always preserved; we shrink from the deepest hop inward,
    #    re-counting after each layer is dropped.  Without this, a
    #    repo with high call-graph fan-out can return 60k+ tokens for
    #    a single bundle, blowing past the orchestrator's 10k tier-2
    #    cap and forcing every caller to truncate downstream anyway.
    total_chars = sum(len(s) for s in source_snippets.values())
    total_tokens = total_chars // _CHARS_PER_TOKEN
    if total_tokens > _TOKEN_BUDGET and symbol_depth:
        # Score the full pre-truncation set so the refill pass inside
        # ``_truncate_to_budget`` re-admits the most relevant dropped
        # symbols first (final response scores are recomputed on the
        # survivor set below).
        pre_scores = compute_symbol_scores(
            all_symbols=all_symbols,
            seed_scores=seed_scores,
            call_graph=call_graph,
            symbol_depth=symbol_depth,
        )
        all_symbols, source_snippets, call_graph, total_tokens = _truncate_to_budget(
            all_symbols=all_symbols,
            source_snippets=source_snippets,
            call_graph=call_graph,
            symbol_depth=symbol_depth,
            budget=_TOKEN_BUDGET,
            scores=pre_scores,
        )

    # 5. Relevance ordering (LE-182). Score every surviving symbol — seeds
    #    keep their merged semantic+boost score, neighbours inherit a decayed
    #    fraction clamped below the lowest seed — then emit ``symbols`` in
    #    score-descending order so a consumer that truncates by array order
    #    keeps the highest-signal symbols. ``scores`` is surfaced additively
    #    for downstream re-ranking. Truncation may have dropped symbols, so we
    #    score the final survivor set.
    scores = compute_symbol_scores(
        all_symbols=all_symbols,
        seed_scores=seed_scores,
        call_graph=call_graph,
        symbol_depth=symbol_depth,
    )
    ordered_symbols = order_symbols_by_score(all_symbols, scores)

    return ContextBundleResponse(
        symbols=ordered_symbols,
        source_snippets=source_snippets,
        call_graph=call_graph,
        total_tokens=total_tokens,
        scores=scores,
    )
