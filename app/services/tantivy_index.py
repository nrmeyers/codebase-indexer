"""Per-repo Tantivy BM25 lexical index — Phase 1.1 of the optimization roadmap.

Wraps the Apache 2.0 `tantivy` Python binding behind a small façade so
routers don't import tantivy directly.  One index lives per repo at
``.cgr/repos/<slug>.tantivy/`` (sibling to the existing ``.db`` and
``.duck`` files), giving us:

* persisted BM25 ranking — no per-search rebuild like the in-memory
  ``bm25s`` we used previously
* exact + prefix lookups on ``file_path`` (replaces the substring-only
  Cypher scan in ``/search/files``)
* cross-arm ranking by score (returned to the orchestrator's mergeAndRank
  as Arm 6)

Schema (matches OPTIMIZATION_ROADMAP.md §1.1):

* ``symbol_qname`` — text, indexed + stored
* ``file_path``    — text, indexed + stored
* ``symbol_kind``  — text, indexed + stored ("Function" / "Method" / "File")
* ``content``      — text, indexed only (function/method body, joined identifiers)
* ``start_line``   — u64, stored
* ``end_line``     — u64, stored
* ``repo``         — text, indexed + stored + fast (per-repo filter is
                     a hard requirement so cross-repo bleed is impossible)

Concurrency model — single-writer, multi-reader (mirrors the existing
LadybugDB pattern in ``ladybug_pool.py``).  A ``threading.Lock`` guards
the writer; readers are issued lock-free via ``Index.searcher()`` after
``reload()``.

Best-effort by design: any tantivy import / IO / schema failure degrades
to no-op behaviour rather than raising — failure here must NEVER take
down ingestion or search.  See ``index.py`` and ``search.py`` for the
non-fatal call sites.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Tantivy is imported lazily so a missing wheel can't break the rest of
# the service surface (``/health`` etc. must still work even if the
# native bindings haven't been installed in this venv).
_tantivy: Any | None = None
_tantivy_unavailable: bool = False


def _load_tantivy() -> Any | None:
    """Return the imported ``tantivy`` module or ``None`` once a load fails.

    The first failure latches ``_tantivy_unavailable`` so subsequent calls
    short-circuit without paying the import-error cost on every request.
    """
    global _tantivy, _tantivy_unavailable  # noqa: PLW0603
    if _tantivy_unavailable:
        return None
    if _tantivy is not None:
        return _tantivy
    try:
        import tantivy  # type: ignore[import-not-found]
    except ImportError as exc:
        logger.warning("tantivy unavailable (%s) — lexical arm disabled", exc)
        _tantivy_unavailable = True
        return None
    _tantivy = tantivy
    return tantivy


def _build_schema(tantivy: Any) -> Any:
    sb = tantivy.SchemaBuilder()
    # ``stored=True`` so we can return the field from search hits;
    # ``indexed`` defaults to True for text fields in tantivy-py.
    sb.add_text_field("symbol_qname", stored=True)
    sb.add_text_field("file_path", stored=True)
    sb.add_text_field("symbol_kind", stored=True)
    # content is the BM25 ranking corpus — body + joined identifiers.
    # No need to retain it in the stored doc (huge index bloat for no gain).
    sb.add_text_field("content", stored=False)
    sb.add_unsigned_field("start_line", stored=True)
    sb.add_unsigned_field("end_line", stored=True)
    # repo is a fast-field so per-repo filtering is O(1); also stored
    # so search results can confirm provenance without a side query.
    sb.add_text_field("repo", stored=True, fast=True)
    return sb.build()


def _index_dir_for_repo(repo_root: Path | str, repo_slug: str) -> Path:
    """Return the on-disk dir for a repo's tantivy index.

    Args:
        repo_root: The ``.cgr/repos`` parent dir (a Path or string).
        repo_slug: Slugged repo name (matches the ``.duck`` / ``.db`` siblings).
    """
    return Path(repo_root) / f"{repo_slug}.tantivy"


class TantivyIndex:
    """Single-writer / multi-reader Tantivy index for one repo.

    Lifecycle:
        1. ``with TantivyIndex(root, slug) as idx:``
        2. ``idx.add(symbol)`` once per symbol (or call site)
        3. ``idx.commit()`` to flush to disk and make readers see the new docs
        4. exit the ``with`` block — the writer closes; subsequent searches
           open new short-lived ``Searcher``s

    A single instance can be used for both writing (during ingestion) and
    reading (during search), but writes are serialised internally via
    ``self._lock``.  In practice, ingest and search are disjoint phases —
    we open a writer for ingest, close it, and then router code opens a
    fresh ``TantivyIndex`` for read-only search.
    """

    def __init__(self, repo_root: Path | str, repo_slug: str) -> None:
        """Create or open a Tantivy index for ``repo_slug`` under ``repo_root``.

        Args:
            repo_root: Parent dir under which ``<slug>.tantivy/`` lives.
                Typically ``.cgr/repos`` (matches ``LADYBUG_DB_DIR``).
            repo_slug: Slugged repo name (mirrors the ``.duck`` neighbour).
        """
        self.repo_slug = repo_slug
        self._dir = _index_dir_for_repo(repo_root, repo_slug)
        self._lock = threading.Lock()
        self._writer: Any | None = None
        self._index: Any | None = None
        self._unavailable = False

        tantivy = _load_tantivy()
        if tantivy is None:
            self._unavailable = True
            return

        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._schema = _build_schema(tantivy)
            # ``Index(schema, path=...)`` opens the existing on-disk index
            # if the meta.json is already there, otherwise creates one.
            self._index = tantivy.Index(self._schema, path=str(self._dir))
        except Exception as exc:
            logger.warning("tantivy.open failed for %s (%s) — disabled", self._dir, exc)
            self._unavailable = True

    # ------------------------------------------------------------------
    # Context-manager glue (so callers can `with TantivyIndex(...) as idx`)
    # ------------------------------------------------------------------

    def __enter__(self) -> "TantivyIndex":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def _ensure_writer(self) -> Any | None:
        """Lazily allocate the index writer (single-writer invariant).

        Returns ``None`` when the index is unavailable (caller should no-op).
        """
        if self._unavailable or self._index is None:
            return None
        if self._writer is None:
            try:
                # heap_size = 50 MiB — comfortable for repos up to ~100k docs;
                # tantivy auto-flushes when this fills up.
                self._writer = self._index.writer(50_000_000)
            except Exception as exc:
                logger.warning("tantivy.writer failed (%s) — disabled", exc)
                self._unavailable = True
                return None
        return self._writer

    def add(
        self,
        *,
        symbol_qname: str,
        file_path: str,
        symbol_kind: str,
        content: str,
        start_line: int,
        end_line: int,
        repo: str,
    ) -> bool:
        """Add a single symbol document to the index.

        All fields are required; supply empty strings / 0 for unknown values
        rather than ``None``.  Returns ``True`` on success, ``False`` when
        the index is unavailable (best-effort contract — never raises).

        The writer must be ``commit()``-ed for these docs to become visible
        to searchers.
        """
        if self._unavailable:
            return False
        tantivy = _load_tantivy()
        if tantivy is None:
            return False
        with self._lock:
            writer = self._ensure_writer()
            if writer is None:
                return False
            try:
                doc = tantivy.Document()
                doc.add_text("symbol_qname", symbol_qname or "")
                doc.add_text("file_path", file_path or "")
                doc.add_text("symbol_kind", symbol_kind or "")
                doc.add_text("content", content or "")
                doc.add_unsigned("start_line", int(start_line) if start_line else 0)
                doc.add_unsigned("end_line", int(end_line) if end_line else 0)
                doc.add_text("repo", repo or "")
                writer.add_document(doc)
                return True
            except Exception as exc:
                logger.debug("tantivy.add failed for %s: %s", symbol_qname, exc)
                return False

    def commit(self) -> bool:
        """Flush pending writes and close the writer (so readers see them)."""
        if self._unavailable or self._writer is None:
            return False
        with self._lock:
            try:
                self._writer.commit()
                # Drop the writer so a future add() opens a fresh one;
                # tantivy enforces single-writer-at-a-time.
                self._writer = None
                if self._index is not None:
                    try:
                        self._index.reload()
                    except Exception:
                        # reload() failure is non-fatal — readers can still
                        # open a searcher, they just may miss the very latest
                        # docs until a subsequent reload.
                        pass
                return True
            except Exception as exc:
                logger.warning("tantivy.commit failed: %s", exc)
                return False

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 20,
        *,
        repo: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run a BM25 query and return up to ``k`` ranked documents.

        Args:
            query: Free-text query.  Tantivy parses it across ``content``,
                ``symbol_qname``, and ``file_path`` so a query like
                ``getInstallationOctokit`` matches identifier hits and
                substring-style queries like ``github-app-client`` match
                file paths.
            k: Maximum hits to return.
            repo: Optional repo slug filter — when set, only docs whose
                ``repo`` field matches are eligible.  Hard-required for
                multi-tenant correctness; without it a query against repo
                A could leak repo B's docs.

        Returns:
            list[dict]: each dict has keys ``symbol_qname``, ``file_path``,
            ``symbol_kind``, ``score``, ``start_line``, ``end_line``.
            Empty list on any failure / empty index / unavailable index.
        """
        if self._unavailable or self._index is None or not query.strip():
            return []
        tantivy = _load_tantivy()
        if tantivy is None:
            return []
        try:
            # Reload to pick up any commits that landed since the last search.
            self._index.reload()
            searcher = self._index.searcher()

            # Parse against the searchable text fields; tantivy will OR
            # the per-field matches and rank by BM25.
            try:
                base_query = self._index.parse_query(
                    query,
                    ["content", "symbol_qname", "file_path"],
                )
            except Exception:
                # Some queries (e.g. unbalanced quotes) trip the parser —
                # fall back to a literal phrase search on identifiers.
                safe = query.replace('"', " ").strip()
                if not safe:
                    return []
                base_query = self._index.parse_query(
                    safe,
                    ["content", "symbol_qname", "file_path"],
                )

            if repo:
                # Repo filter: AND the user query with a term query on the
                # ``repo`` text field.  We use parse_query with a quoted
                # term to avoid analyzer-induced surprises (slugs contain
                # dashes that the default tokenizer otherwise splits).
                try:
                    repo_query = self._index.parse_query(f'"{repo}"', ["repo"])
                    combined = tantivy.Query.boolean_query(
                        [
                            (tantivy.Occur.Must, base_query),
                            (tantivy.Occur.Must, repo_query),
                        ]
                    )
                except Exception:
                    # If boolean composition fails (older binding), fall
                    # back to post-filtering on the python side below.
                    combined = base_query
            else:
                combined = base_query

            top = searcher.search(combined, max(int(k), 1)).hits
            out: list[dict[str, Any]] = []
            for score, addr in top:
                doc = searcher.doc(addr)
                d = doc.to_dict() if hasattr(doc, "to_dict") else {}
                # tantivy-py returns each field as a list (multi-valued).
                hit_repo = (d.get("repo") or [""])[0]
                if repo and hit_repo and hit_repo != repo:
                    # Defensive post-filter for the fallback branch above.
                    continue
                out.append(
                    {
                        "symbol_qname": (d.get("symbol_qname") or [""])[0],
                        "file_path": (d.get("file_path") or [""])[0],
                        "symbol_kind": (d.get("symbol_kind") or [""])[0],
                        "score": float(score),
                        "start_line": int((d.get("start_line") or [0])[0] or 0),
                        "end_line": int((d.get("end_line") or [0])[0] or 0),
                    }
                )
                if len(out) >= k:
                    break
            return out
        except Exception as exc:
            logger.debug("tantivy.search failed for %r: %s", query, exc)
            return []

    def close(self) -> None:
        """Release the writer if one is open.  Safe to call repeatedly."""
        if self._writer is not None:
            try:
                # Final flush — anything not yet committed is dropped on
                # close, so try one more commit before giving up.
                self._writer.commit()
            except Exception:
                pass
            self._writer = None

    # ------------------------------------------------------------------
    # Convenience factory used by routers (matches the spec signature in
    # OPTIMIZATION_ROADMAP.md §1.1: __init__(repo_root: Path)).
    # ------------------------------------------------------------------

    @classmethod
    def open_or_create(cls, repo_root: Path | str, repo_slug: str) -> "TantivyIndex":
        """Open / create the index for ``repo_slug`` rooted at ``repo_root``."""
        return cls(repo_root, repo_slug)


# ---------------------------------------------------------------------------
# Full rebuild from the symbol graph + source text
# ---------------------------------------------------------------------------

# Caps keep the index bounded on repos with giant generated files.
_SYMBOL_SOURCE_CAP_CHARS = 6000
_FILE_HEADER_MAX_LINES = 120
_FILE_HEADER_CAP_CHARS = 6000


def rebuild_lexical_index(
    repo_db_path: str | Path,
    repo_root: str | Path,
    slug: str,
    db_dir: str | Path,
) -> int:
    """Rebuild ``<db_dir>/<slug>.tantivy/`` from the LadybugDB graph + source.

    Each Function/Method doc's ``content`` is qualified name + docstring +
    the symbol's **source span** (comments included). BM25 must see raw
    text, not just identifiers, to catch comment-only signal — e.g. an
    auth provider documented as "AAD" in a comment is invisible to both
    the dense embedding and the CALLS graph, and was previously invisible
    to the lexical arm too (content used to be qname + docstring only).

    One extra doc per Module carries the file-header region under
    ``{module_qname}::File::summary`` so top-of-file prose is searchable;
    the qname matches the embed driver's file-summary chunks, so the
    context-bundle's existing summary hydration renders these hits.

    Wipes any existing index dir first — this is a rebuild, not an append.
    Callers that mirror additional corpora into the same index (markdown
    pass) must run AFTER this. Best-effort: returns 0 and logs on any
    environment failure rather than raising.

    Args:
        repo_db_path: Path to the per-repo LadybugDB ``.db`` file.
        repo_root: Repo checkout root — resolves relative Module paths.
        slug: Canonical repo slug (index dir name + ``repo`` field).
        db_dir: Parent data dir holding ``<slug>.tantivy/``.

    Returns:
        Number of documents added.
    """
    import shutil

    if _load_tantivy() is None:
        return 0
    try:
        import real_ladybug as _lb  # type: ignore[import-untyped]
        from .ladybug_buffer_pool import resolve_buffer_pool_size
    except Exception as exc:  # pragma: no cover - env-specific
        logger.warning("rebuild_lexical_index: ladybug unavailable: %s", exc)
        return 0

    root = Path(repo_root)
    index_dir = _index_dir_for_repo(db_dir, slug)
    try:
        if index_dir.exists():
            shutil.rmtree(index_dir)
    except OSError as exc:
        logger.warning("rebuild_lexical_index: wipe failed: %s", exc)

    _line_cache: dict[str, list[str] | None] = {}

    def _file_lines(rel_or_abs: str) -> list[str] | None:
        if rel_or_abs in _line_cache:
            return _line_cache[rel_or_abs]
        p = Path(rel_or_abs)
        if not p.is_absolute():
            p = root / p
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            lines = None
        _line_cache[rel_or_abs] = lines
        return lines

    added = 0
    db = _lb.Database(
        str(repo_db_path), read_only=True,
        buffer_pool_size=resolve_buffer_pool_size(),
    )
    conn = _lb.Connection(db)
    idx = TantivyIndex(db_dir, slug)
    try:
        # --- Function / Method docs: qname + docstring + source span ---
        res = conn.execute(
            "MATCH (m:Module)-[:DEFINES]->(n:Function) "
            "RETURN n.qualified_name AS qn, n.start_line AS sl, "
            "n.end_line AS el, m.path AS p, n.docstring AS doc, "
            "'Function' AS kind "
            "UNION ALL "
            "MATCH (m:Module)-[:DEFINES]->(:Class)-[:DEFINES_METHOD]->(n:Method) "
            "RETURN n.qualified_name AS qn, n.start_line AS sl, "
            "n.end_line AS el, m.path AS p, n.docstring AS doc, "
            "'Method' AS kind"
        )
        cols = res.get_column_names()
        while res.has_next():
            row = dict(zip(cols, res.get_next()))
            qn = row.get("qn") or ""
            if not qn:
                continue
            parts = [qn]
            doc = row.get("doc")
            if isinstance(doc, str) and doc:
                parts.append(doc)
            path = str(row.get("p") or "")
            sl = int(row.get("sl") or 0)
            el = int(row.get("el") or 0)
            if path and sl > 0 and el >= sl:
                lines = _file_lines(path)
                if lines:
                    src = "\n".join(lines[sl - 1 : el])
                    parts.append(src[:_SYMBOL_SOURCE_CAP_CHARS])
            if idx.add(
                symbol_qname=str(qn),
                file_path=path,
                symbol_kind=str(row.get("kind") or "Function"),
                content="\n".join(parts),
                start_line=sl,
                end_line=el,
                repo=slug,
            ):
                added += 1

        # --- File-header docs: top-of-file prose under ::File::summary ---
        res = conn.execute(
            "MATCH (m:Module) RETURN m.qualified_name AS qn, m.path AS p"
        )
        cols = res.get_column_names()
        while res.has_next():
            row = dict(zip(cols, res.get_next()))
            qn = row.get("qn") or ""
            path = str(row.get("p") or "")
            if not qn or not path:
                continue
            lines = _file_lines(path)
            if not lines:
                continue
            header = "\n".join(lines[:_FILE_HEADER_MAX_LINES])
            if idx.add(
                symbol_qname=f"{qn}::File::summary",
                file_path=path,
                symbol_kind="File",
                content=f"{path}\n{header[:_FILE_HEADER_CAP_CHARS]}",
                start_line=1,
                end_line=min(len(lines), _FILE_HEADER_MAX_LINES),
                repo=slug,
            ):
                added += 1

        idx.commit()
    finally:
        idx.close()
        try:
            conn.close()
        except Exception:
            pass
        try:
            db.close()
        except Exception:
            pass
    return added
