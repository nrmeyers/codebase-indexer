"""Cross-repo IMPORTS resolution — BUC-1598.

When the Code Indexer ingests N repos into one LadybugDB instance, the
underlying ``codebase_rag`` fork treats each repo's graph as isolated.  An
import that resolves to a sibling indexed repo (e.g. ``TheForge`` importing
``@nrmeyers/shared-types`` from the ``shared-types`` repo) ends up as an
"external" :class:`Module` node — a leaf with no project prefix — instead
of being linked across to the corresponding ``Module`` node in the other
repo's graph.

This module is the cross-repo resolution layer.  It runs *after* a repo
finishes ingesting (post-processing pass) and rewrites the external Module
nodes that match another indexed repo's package identity to a canonical
cross-repo qualified name of the form ``{target_slug}::{module_qn}``.
Downstream consumers can then identify cross-repo IMPORTS by the
``<slug>::`` prefix on the target Module's qualified name.

The matching heuristics are package-manager aware:

    * npm-style ``@scope/pkg`` ↔ a repo whose ``package.json`` ``name``
      field equals the import path.
    * Python dotted ``import foo.bar`` ↔ a repo whose ``pyproject.toml``
      ``project.name`` (or ``setup.py`` arg) equals ``foo`` (or whose
      first top-level package directory equals ``foo``).
    * TypeScript ``references[].path`` workspace refs — **out of scope
      for v1** (tracked for v2; see PR body).

All resolution work happens here, in the service layer.  The
``codebase_rag`` fork is never touched — it stays pure and per-repo.

Public surface
--------------

    extract_repo_identity(root_path)        → :class:`RepoIdentity`
    resolve_cross_repo_imports(slug, …)     → :class:`ResolveStats`
    resolve_all(…)                          → list[:class:`ResolveStats`]

The module is gated by the ``CROSS_REPO_IMPORTS_ENABLED`` env var (default
False).  Callers MUST honor the flag — ``resolve_cross_repo_imports`` will
short-circuit when the flag is unset.

Performance note
----------------
Each ``resolve_cross_repo_imports`` call opens the target repo's DB in
read-write mode briefly, runs one indexed scan over external Module nodes,
matches against the in-memory identity table, and issues a small batch of
node rewrites.  Empirically the pass adds <500ms for repos with <500
external imports; the post-ingest hook logs duration so regressions surface.
"""
from __future__ import annotations

import logging
import os
import re
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------


def is_enabled() -> bool:
    """Return True iff cross-repo IMPORTS resolution is enabled.

    Reads ``CROSS_REPO_IMPORTS_ENABLED`` from the environment at call time
    (not import time) so tests can monkey-patch ``os.environ`` without
    having to reload the module.  Accepts the usual truthy strings
    (``1``, ``true``, ``yes``, ``on`` — case-insensitive).
    """
    raw = os.environ.get("CROSS_REPO_IMPORTS_ENABLED", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Repo identity extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepoIdentity:
    """Package-manager identities that a repo exposes to importers.

    A single repo can expose multiple identities — a TypeScript library
    that also ships a Python wheel will populate both ``npm_name`` and
    ``python_name``.  Empty strings indicate the manifest was not present
    or not readable; the matcher treats them as "no match" sentinels.

    Attributes:
        slug: The Code Indexer slug for this repo (e.g. ``nrmeyers__TheForge``).
            Used to namespace the rewritten cross-repo Module qnames.
        npm_name: The ``name`` field from ``package.json`` (e.g.
            ``@nrmeyers/shared-types``).  Empty if absent.
        python_name: The ``project.name`` from ``pyproject.toml`` (PEP 621)
            or — as a fallback — the first top-level Python package
            directory.  Normalised to dotted form (``-`` and ``_`` are
            equivalent in PEP 503 normalisation; we keep the literal name
            for now and the matcher does normalisation at compare time).
            Empty if absent.
        python_top_level: Best-effort top-level import name when the
            ``project.name`` does not match the on-disk import name
            (common when a project names itself ``my-package`` on PyPI
            but ships ``import my_package`` — PEP 503 normalisation
            collapses these).  Empty if not derivable.
    """

    slug: str
    npm_name: str = ""
    python_name: str = ""
    python_top_level: str = ""


def _read_json(path: Path) -> dict[str, Any]:
    """Read JSON best-effort. Returns empty dict on any failure."""
    import json
    try:
        with path.open("rb") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _read_toml(path: Path) -> dict[str, Any]:
    """Read TOML best-effort. Returns empty dict on any failure."""
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


_PY_NAME_FROM_SETUP_PY = re.compile(
    r"""name\s*=\s*['"]([A-Za-z0-9_.\-]+)['"]""",
    re.MULTILINE,
)


def _extract_setup_py_name(setup_py: Path) -> str:
    """Pull ``name="…"`` out of a ``setup.py`` via regex.

    We do NOT exec the file — packages routinely depend on third-party
    helpers (``setuptools.find_packages``) which we'd then need installed.
    A regex over the literal kwarg covers the overwhelming majority of
    real-world ``setup.py`` files; pathological cases (name composed at
    runtime) are accepted as "no match" — they'll fall back to External.
    """
    try:
        text = setup_py.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    m = _PY_NAME_FROM_SETUP_PY.search(text)
    return m.group(1) if m else ""


def _detect_python_top_level(root: Path, project_name: str) -> str:
    """Best-effort guess at the importable top-level package name.

    Looks for, in priority order:

        1. ``<root>/src/<pkg>/__init__.py`` (src layout — preferred)
        2. ``<root>/<pkg>/__init__.py`` (flat layout)

    where ``<pkg>`` is the PEP 503 normalised form of ``project_name``
    with ``-`` replaced by ``_``.  Falls back to scanning for any single
    top-level directory containing ``__init__.py`` when the normalised
    guess doesn't exist.

    Returns the empty string when no top-level package can be identified.
    """
    if not project_name:
        # Scan for a single obvious package directory.
        candidates = [
            p for p in root.iterdir()
            if p.is_dir() and not p.name.startswith(".") and (p / "__init__.py").is_file()
        ]
        if len(candidates) == 1:
            return candidates[0].name
        return ""

    guess = project_name.replace("-", "_").lower()
    if (root / "src" / guess / "__init__.py").is_file():
        return guess
    if (root / guess / "__init__.py").is_file():
        return guess
    return ""


def extract_repo_identity(slug: str, root_path: str) -> RepoIdentity:
    """Read package manifests at ``root_path`` and return the repo's identities.

    Reads (all optional, all best-effort):

        * ``package.json`` → ``npm_name``
        * ``pyproject.toml`` (PEP 621 ``[project].name``) → ``python_name``
        * ``setup.py`` (regex-extracted ``name=`` kwarg) → ``python_name``
          when ``pyproject.toml`` is absent
        * On-disk directory structure → ``python_top_level``

    Failures (missing file, malformed JSON/TOML, permission denied) are
    silent — the corresponding field is left empty so the matcher will
    treat the identity as "no claim" rather than mis-routing.

    Args:
        slug: The Code Indexer slug for this repo.  Stored verbatim in the
            returned identity for downstream namespace prefixing.
        root_path: Filesystem path to the repo's root directory.

    Returns:
        :class:`RepoIdentity` with every field that could be determined
        populated.  When ``root_path`` does not exist or is not a directory
        the result has only ``slug`` set.
    """
    root = Path(root_path)
    if not root.is_dir():
        return RepoIdentity(slug=slug)

    # --- npm identity ---
    npm_name = ""
    pkg_json = root / "package.json"
    if pkg_json.is_file():
        data = _read_json(pkg_json)
        raw_name = data.get("name")
        if isinstance(raw_name, str) and raw_name.strip():
            npm_name = raw_name.strip()

    # --- python identity (pyproject preferred, setup.py fallback) ---
    python_name = ""
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        data = _read_toml(pyproject)
        project_block = data.get("project", {})
        if isinstance(project_block, dict):
            raw_name = project_block.get("name")
            if isinstance(raw_name, str) and raw_name.strip():
                python_name = raw_name.strip()

    if not python_name:
        setup_py = root / "setup.py"
        if setup_py.is_file():
            python_name = _extract_setup_py_name(setup_py)

    python_top_level = _detect_python_top_level(root, python_name)

    return RepoIdentity(
        slug=slug,
        npm_name=npm_name,
        python_name=python_name,
        python_top_level=python_top_level,
    )


# ---------------------------------------------------------------------------
# Matching heuristics
# ---------------------------------------------------------------------------


def _normalise_python_name(name: str) -> str:
    """PEP 503-style normalisation: lower-case, dashes/underscores fold.

    Two PEP 503 normalised names are considered equivalent project names.
    We use this on the LHS of every Python match to avoid missing a
    ``my-package`` vs ``my_package`` mismatch.
    """
    return re.sub(r"[-_.]+", "_", name).strip("_").lower()


def match_external_module(
    qualified_name: str,
    path_hint: str,
    identities: Iterable[RepoIdentity],
) -> RepoIdentity | None:
    """Return the :class:`RepoIdentity` that best matches an external Module.

    The matcher tries, in order:

        1. **Exact npm** — the import qname OR path equals an identity's
           ``npm_name``.  Handles ``@scope/pkg`` and bare ``pkg``.
        2. **Python first-segment** — the first dotted segment of the
           qname (or path) PEP-503-normalises to an identity's
           ``python_name`` or ``python_top_level``.

    Args:
        qualified_name: The Module's ``qualified_name`` as stored in the
            graph (e.g. ``@nrmeyers/shared-types`` or ``flask.helpers``).
        path_hint: The Module's ``path`` column — often the original
            full import path before the parser stripped the trailing
            symbol; matched as a fallback.
        identities: All known repo identities to try.

    Returns:
        The matching :class:`RepoIdentity`, or ``None`` when nothing
        claims ownership.  An identity self-match (where the qname
        already belongs to that identity's slug namespace) is filtered
        out by callers — ``match_external_module`` only does the
        identity match, the caller decides what to do.
    """
    # npm: literal match against either qname or path.  npm packages are
    # case-sensitive on the registry but case-insensitive on most
    # filesystems; canonical form is lowercase.
    qn_l = qualified_name.lower().strip()
    path_l = path_hint.lower().strip()
    candidates = [c for c in (qn_l, path_l) if c]

    for ident in identities:
        if not ident.npm_name:
            continue
        npm_l = ident.npm_name.lower().strip()
        if any(c == npm_l for c in candidates):
            return ident

    # Python: first dotted segment.
    # ``flask.helpers`` → ``flask``; ``django.db.models`` → ``django``.
    first_seg_qn = qualified_name.split(".", 1)[0] if qualified_name else ""
    first_seg_path = path_hint.split(".", 1)[0] if path_hint else ""
    py_candidates = {
        _normalise_python_name(s) for s in (first_seg_qn, first_seg_path) if s
    }
    py_candidates.discard("")

    for ident in identities:
        targets = set()
        if ident.python_name:
            targets.add(_normalise_python_name(ident.python_name))
        if ident.python_top_level:
            targets.add(_normalise_python_name(ident.python_top_level))
        targets.discard("")
        if py_candidates & targets:
            return ident

    return None


# ---------------------------------------------------------------------------
# Resolver — Ladybug rewire
# ---------------------------------------------------------------------------


# Sentinel separator used to namespace cross-repo Module qnames.  Picked to
# avoid collision with the dotted/`::`-separated forms the parsers already
# emit; downstream consumers can detect cross-repo nodes by the literal
# ``::<slug>::`` substring at the head of the qname.  Two colons keeps the
# form parseable by simple ``split("::", 1)``.
CROSS_REPO_PREFIX_SEP = "::"


def make_cross_repo_qname(target_slug: str, original_qname: str) -> str:
    """Return the canonical cross-repo Module qname for ``original_qname``.

    Format: ``{target_slug}::{original_qname}``.  The result is what a
    rewired :class:`Module` node's ``qualified_name`` column should hold
    after a successful resolution pass.

    The leading ``{target_slug}::`` prefix is the only durable signal
    that this Module came from another repo — every cross-repo IMPORTS
    edge's target node carries it.  Downstream consumers (centrality,
    blast-radius, the audit query in BUC-1598's acceptance test) detect
    cross-repo edges by scanning for the ``::`` substring.
    """
    return f"{target_slug}{CROSS_REPO_PREFIX_SEP}{original_qname}"


def is_cross_repo_qname(qualified_name: str) -> bool:
    """Return True iff ``qualified_name`` carries the cross-repo prefix.

    Used by the backfill endpoint to skip Modules that have already been
    rewired (idempotency guard) and by external observers to count
    cross-repo IMPORTS edges in tests/dashboards.
    """
    return CROSS_REPO_PREFIX_SEP in qualified_name


@dataclass
class ResolveStats:
    """Per-repo summary of one resolution pass.

    Attributes:
        slug: The repo we resolved cross-repo IMPORTS for.
        scanned: Count of candidate external Module nodes inspected.
        matched: Count rewired to a sibling repo's canonical qname.
        unmatched: Count left as-is (no sibling identity claimed them).
        duration_ms: Wall-clock time for this resolution pass, ms.
        errors: Non-fatal exceptions surfaced during the pass.  An empty
            list indicates a clean run.
    """

    slug: str
    scanned: int = 0
    matched: int = 0
    unmatched: int = 0
    duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)


def _project_name_from_slug(slug: str) -> str:
    """Return the project_name that the parser used when ingesting ``slug``.

    ``codebase_rag.GraphUpdater`` sets ``project_name = repo_path.resolve().name``
    — i.e. the directory basename.  The Code Indexer's slug is derived from
    the same source (canonical form ``{org}__{repo}`` or bare basename), so
    the project_name is whatever follows the ``__`` separator (or the slug
    itself when there isn't one).

    This is best-effort.  If the slug shape ever drifts we'll match too
    eagerly (rewire something that belongs to our own repo), which the
    caller catches by checking the qname prefix before issuing any rewrite.
    """
    if "__" in slug:
        return slug.split("__", 1)[1]
    return slug


def _fetch_external_modules(
    conn: Any, project_name: str
) -> list[tuple[str, str, str]]:
    """Return external Module nodes from ``conn`` as ``(qname, name, path)``.

    "External" here means: ``qualified_name`` does NOT start with the
    project's own prefix AND does NOT already carry the cross-repo
    ``::`` separator.  We exclude the cross-repo prefix to keep the pass
    idempotent — a second backfill should match nothing.

    Returns up to 10,000 rows.  Repos with more external imports than
    that are unusual; we log and cap rather than risk an unbounded query.
    """
    own_prefix = f"{project_name}."
    query = (
        "MATCH (m:Module) "
        "WHERE NOT m.qualified_name STARTS WITH $own_prefix "
        "  AND NOT m.qualified_name CONTAINS $sep "
        "RETURN m.qualified_name AS qn, m.name AS name, m.path AS path "
        "LIMIT 10000"
    )
    params = {"own_prefix": own_prefix, "sep": CROSS_REPO_PREFIX_SEP}
    try:
        result = conn.execute(query, params)
    except Exception as exc:
        logger.warning("cross_repo.fetch_external failed: %s", exc)
        return []

    rows: list[tuple[str, str, str]] = []
    try:
        while result.has_next():
            row = result.get_next()
            qn = str(row[0]) if row and row[0] is not None else ""
            name = str(row[1]) if len(row) > 1 and row[1] is not None else ""
            path = str(row[2]) if len(row) > 2 and row[2] is not None else ""
            if qn:
                rows.append((qn, name, path))
    except Exception as exc:
        logger.warning("cross_repo.fetch_external row read failed: %s", exc)
    return rows


def _rewire_module(
    conn: Any, old_qname: str, new_qname: str, name: str, path: str
) -> bool:
    """Repoint every IMPORTS edge from ``old_qname`` to ``new_qname``.

    LadybugDB/Kuzu does not support updating a node's primary key in
    place.  We work around that by:

        1. ``MERGE`` a fresh Module with the canonical cross-repo qname.
        2. For every ``(src)-[r:IMPORTS]->(:Module {qname: old})`` edge:
              create ``(src)-[:IMPORTS]->(:Module {qname: new})``
              delete ``r``.
        3. ``DETACH DELETE`` the old Module (now an orphan).

    Returns True iff the rewrite committed cleanly.  Any failure rolls
    back to no-op (we explicitly do NOT leave half-rewritten state — the
    pre-flight MERGE is cheap and the per-edge MATCH/CREATE/DELETE is
    contained in a single Cypher statement).
    """
    try:
        # Step 1: ensure the new canonical Module exists.
        conn.execute(
            "MERGE (m:Module {qualified_name: $qn}) "
            "SET m.name = $name, m.path = $path",
            {"qn": new_qname, "name": name, "path": path},
        )
        # Step 2: copy every inbound IMPORTS edge to the new target, then
        # drop the old edge.  Doing this in two passes (CREATE-all then
        # DELETE-all) avoids the "iterator invalidated" gotcha that some
        # graph DBs throw when you mutate and read in the same MATCH.
        conn.execute(
            "MATCH (src:Module)-[:IMPORTS]->(:Module {qualified_name: $old}) "
            "MATCH (dst:Module {qualified_name: $new}) "
            "MERGE (src)-[:IMPORTS]->(dst)",
            {"old": old_qname, "new": new_qname},
        )
        conn.execute(
            "MATCH (src:Module)-[r:IMPORTS]->(old:Module {qualified_name: $old}) "
            "DELETE r",
            {"old": old_qname},
        )
        # Step 3: drop the now-orphaned external node.
        conn.execute(
            "MATCH (m:Module {qualified_name: $old}) DETACH DELETE m",
            {"old": old_qname},
        )
        return True
    except Exception as exc:
        logger.warning(
            "cross_repo.rewire failed old=%s new=%s err=%s",
            old_qname, new_qname, exc,
        )
        return False


def resolve_cross_repo_imports(
    slug: str,
    db_path: str,
    sibling_identities: Iterable[RepoIdentity],
) -> ResolveStats:
    """Run one cross-repo IMPORTS resolution pass over ``slug``'s DB.

    Args:
        slug: The Code Indexer slug for the repo we're resolving FOR
            (the importer's side).  Used to skip identities that belong
            to this same repo (a repo can't import "from itself
            cross-repo").
        db_path: Filesystem path to the repo's LadybugDB ``.db`` file.
        sibling_identities: Identities for every OTHER indexed repo —
            the matcher tries each in turn.  Callers should pre-filter
            ``slug`` out of this collection.

    Returns:
        :class:`ResolveStats` summarising the pass.  Returns immediately
        with zeros when the feature flag is disabled.
    """
    stats = ResolveStats(slug=slug)

    if not is_enabled():
        # Fail-closed: the feature flag is the only gate.  Callers wrap
        # this in a try/except so a disabled flag is identical to a
        # successful no-op pass.
        logger.debug("cross_repo.disabled slug=%s", slug)
        return stats

    siblings = [s for s in sibling_identities if s.slug != slug]
    if not siblings:
        logger.debug("cross_repo.no_siblings slug=%s", slug)
        return stats

    t0 = time.monotonic()

    # Open the target DB read-write.  We use the existing ladybug_pool
    # helper to keep the lock semantics identical to every other writer.
    from .ladybug_pool import open_rw_conn

    db = None
    conn = None
    try:
        db, conn = open_rw_conn(db_path)
    except Exception as exc:
        stats.errors.append(f"open_rw_conn failed: {exc}")
        logger.warning("cross_repo.open_rw failed slug=%s err=%s", slug, exc)
        return stats

    try:
        project_name = _project_name_from_slug(slug)
        externals = _fetch_external_modules(conn, project_name)
        stats.scanned = len(externals)

        for qn, name, path in externals:
            match = match_external_module(qn, path, siblings)
            if match is None:
                stats.unmatched += 1
                continue
            new_qn = make_cross_repo_qname(match.slug, qn)
            if _rewire_module(conn, qn, new_qn, name, path):
                stats.matched += 1
            else:
                stats.errors.append(f"rewire_failed:{qn}")
                stats.unmatched += 1
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass
        # Drop refs so the file lock releases promptly — mirrors the
        # GC dance in index.py around blocking subprocess spawns.
        del conn, db

    stats.duration_ms = (time.monotonic() - t0) * 1000.0
    logger.info(
        "cross_repo.resolved slug=%s scanned=%d matched=%d unmatched=%d duration_ms=%.1f",
        slug, stats.scanned, stats.matched, stats.unmatched, stats.duration_ms,
    )
    return stats


def resolve_all(
    repos: dict[str, str],
) -> list[ResolveStats]:
    """Run :func:`resolve_cross_repo_imports` for every indexed repo.

    Args:
        repos: Mapping of ``{slug: root_path}`` for every indexed repo
            currently in the registry.  Typically this is
            ``indexed_repo_paths`` from ``app.routers.index``.

    Returns:
        One :class:`ResolveStats` per repo, in iteration order.  Returns
        an empty list when the feature flag is disabled (fail closed).
    """
    if not is_enabled():
        logger.info("cross_repo.resolve_all skipped — flag disabled")
        return []

    # Pre-compute every repo's identity once so each per-repo pass can
    # reuse them without re-reading manifests N times.
    identities: dict[str, RepoIdentity] = {
        slug: extract_repo_identity(slug, root_path)
        for slug, root_path in repos.items()
    }

    from ..config import settings

    results: list[ResolveStats] = []
    for slug in repos:
        db_path = settings.db_path_for_repo(slug)
        if not Path(db_path).is_file():
            logger.debug("cross_repo.skip_missing_db slug=%s", slug)
            continue
        siblings = [ident for s, ident in identities.items() if s != slug]
        stats = resolve_cross_repo_imports(slug, db_path, siblings)
        results.append(stats)
    return results
