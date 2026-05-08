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
        symbols: Every qualified name in the bundle (seeds + expansion).
        source_snippets: Map of qualified name → source code. Empty string
            when the symbol's file could not be read.
        call_graph: Adjacency list ``caller → [callees]`` limited to edges
            discovered during BFS expansion.
        total_tokens: Rough estimate of token cost if every snippet were
            concatenated into a prompt.
    """

    symbols: list[str]
    source_snippets: dict[str, str]
    call_graph: dict[str, list[str]]
    total_tokens: int


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


def _expand_call_graph(
    conn: object, seed_symbols: list[str], depth: int
) -> tuple[set[str], dict[str, list[str]], dict[str, int]]:
    """BFS over the CALLS graph up to ``depth`` hops from the seed symbols.

    Args:
        conn: An open LadybugDB connection.
        seed_symbols: Qualified names to start BFS from (seed set).
        depth: Maximum number of hops to traverse. 0 returns only seeds.

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

    # Drop deepest layers until under budget or only seeds remain.
    max_depth = max(symbol_depth.get(s, 0) for s in kept)
    current = sum(len(snippets.get(s, "")) for s in kept) // _CHARS_PER_TOKEN

    while current > budget and max_depth > 0:
        layer = {s for s in kept if symbol_depth.get(s, 0) == max_depth}
        kept -= layer
        for s in layer:
            snippets.pop(s, None)
        current = sum(len(snippets.get(s, "")) for s in kept) // _CHARS_PER_TOKEN
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
            current -= len(removed) // _CHARS_PER_TOKEN

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
            f"LIMIT {int(limit)}"
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
            f"LIMIT {int(limit)}"
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


# Per-intent retrieval parameters — a single knob per intent that the
# route handler reads to shape the bundle.  Changing these values tunes
# the breadth/depth tradeoff without touching control flow.
_INTENT_PARAMS: dict[str, dict[str, int]] = {
    # Current default — deep call-graph walk off tight seeds.
    "symbol":     {"k": 12, "depth": 3, "module_limit": 0,  "entrypoint_limit": 0},
    # Wider seeds + full module inclusion; shallower expansion keeps
    # token count manageable when whole routers are dropped in.
    "conceptual": {"k": 20, "depth": 2, "module_limit": 25, "entrypoint_limit": 0},
    # Like conceptual but also pulls in entry-point handlers so the
    # flow's starting points are always grounded.
    "howto":      {"k": 15, "depth": 3, "module_limit": 20, "entrypoint_limit": 10},
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
    repo_slug = Path(req.repo_path).resolve().name
    _repo_db = settings.db_path_for_repo(repo_slug)
    try:
        from codebase_rag.config import settings as _cgr_settings  # type: ignore[import-untyped]
        _cgr_settings.LADYBUG_DB_PATH = _repo_db
    except Exception:
        pass  # non-fatal; falls back to default

    try:
        from codebase_rag.tools.semantic_search import semantic_code_search
        # Over-fetch aggressively: fixture functions tend to have degenerate
        # embeddings that score high for any query.  Fetch 500+ to guarantee
        # we reach real application code below the fixture cluster.
        seed_results = semantic_code_search(req.task_description, top_k=max(effective_k * 50, 500))
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

        seed_symbols = [
            r["qualified_name"]
            for r in seed_results
            if not _is_noise(r["qualified_name"])
        ][: effective_k]

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

        # Merge order matters — highest-signal first so the BFS expansion
        # prioritises them: exact name matches → module handlers → entry
        # points → semantic seeds.  De-dupe while preserving order.
        seen: set[str] = set()
        merged: list[str] = []
        for qn in exact_hits + module_hits + entrypoint_hits + seed_symbols:
            if qn not in seen:
                seen.add(qn)
                merged.append(qn)
        # Cap at 2× effective_k so a wide module boost on a small repo
        # doesn't blow up the BFS frontier.  The exact cap adapts to
        # how many boosts fired.
        seed_cap = max(effective_k, len(exact_hits) + len(module_hits) + len(entrypoint_hits))
        seed_symbols = merged[: min(seed_cap, effective_k * 2)]

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
    )

    # 3. Fetch source snippets — sorted for deterministic output.
    source_snippets = _fetch_source_for_symbols(conn, sorted(all_symbols))

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
        all_symbols, source_snippets, call_graph, total_tokens = _truncate_to_budget(
            all_symbols=all_symbols,
            source_snippets=source_snippets,
            call_graph=call_graph,
            symbol_depth=symbol_depth,
            budget=_TOKEN_BUDGET,
        )

    return ContextBundleResponse(
        symbols=sorted(all_symbols),
        source_snippets=source_snippets,
        call_graph=call_graph,
        total_tokens=total_tokens,
    )
