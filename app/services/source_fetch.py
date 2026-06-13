"""Source-snippet resolution for symbol qualified names.

Centralises the "given a list of qualified names, return the on-disk
source code for each" logic that is needed by:

* ``/context-bundle`` — to build grounded prompt context
* ``/search/semantic?rerank=true`` — to give the listwise reranker code
  body context (CodeRankLLM / Qwen ranks substantially better with
  snippets than with bare identifier names)

Both callers used to maintain their own near-identical helpers; this
module is the single source of truth so a fix in one place propagates
to both retrieval paths.

The query joins the cosine-search results back to LadybugDB ``Module``
nodes (via ``CYPHER_GET_FUNCTION_SOURCE_LOCATION``) to find the file
path + line range, then reads the file from disk.  Best-effort
throughout — any single symbol whose location can't be resolved gets
an empty string in the output dict rather than aborting the whole
request.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


# Leading comment block: parsers anchor a symbol's span at its declaration
# line, cutting off the docblock ABOVE it (JSDoc, line-comment runs,
# decorators). Those blocks carry the design intent ("callers should mount
# requireIdentity first…") that retrieval consumers need, so snippets walk
# upward over a contiguous comment/decorator run — bounded, never across a
# blank-line gap to unrelated code.
_LEADING_COMMENT_PREFIXES = ("//", "/*", "*", "#", '"""', "'''", "@")
_LEADING_COMMENT_MAX_LINES = 30


def _extend_to_leading_comment(lines: list[str], start: int) -> int:
    """Return the 0-indexed start extended over a contiguous leading
    comment/decorator block (bounded by ``_LEADING_COMMENT_MAX_LINES``)."""
    i = start - 1
    taken = 0
    while i >= 0 and taken < _LEADING_COMMENT_MAX_LINES:
        stripped = lines[i].strip()
        if not stripped:
            break
        if not (
            stripped.startswith(_LEADING_COMMENT_PREFIXES)
            or stripped.endswith("*/")
        ):
            break
        i -= 1
        taken += 1
    return i + 1


def fetch_source(
    file_path: str, line_start: int | None, line_end: int | None
) -> str:
    """Read a source slice from disk between 1-indexed start/end lines.

    The start extends upward over a contiguous leading comment/decorator
    block (see ``_extend_to_leading_comment``) so docblocks above the
    declaration line are part of the snippet.

    Args:
        file_path: Absolute path to the file.  Must exist on disk; this
            function does not handle missing-file fallbacks beyond
            returning the empty string.
        line_start: 1-indexed start line; defaults to 1 when ``None``.
        line_end: 1-indexed inclusive end line; defaults to one line
            past ``line_start`` when ``None``.

    Returns:
        str: The joined source lines, or empty string on any read
        failure (missing file, encoding error, permission denied, etc.).
    """
    if not file_path or not Path(file_path).exists():
        return ""
    try:
        lines = Path(file_path).read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        # 1-indexed → 0-indexed slice start; default line 1 when unset.
        start = max(0, (line_start or 1) - 1)
        end = line_end or (start + 1)
        start = _extend_to_leading_comment(lines, start)
        return "\n".join(lines[start:end])
    except Exception:
        # Swallow — caller is expected to use whatever it can get.
        return ""


def _result_to_rows(result: object) -> list[dict[str, Any]]:
    """Consume a LadybugDB result iterator into column-keyed dicts.

    Duplicated from each router so this module has no router-side
    dependencies.  Removing the duplication would require lifting
    ``_result_to_rows`` into a shared util too — left as a follow-up
    so this change stays scoped.
    """
    rows: list[dict[str, Any]] = []
    col_names = result.get_column_names()  # type: ignore[attr-defined]
    while result.has_next():  # type: ignore[attr-defined]
        raw = result.get_next()  # type: ignore[attr-defined]
        rows.append(dict(zip(col_names, raw)))
    return rows


def fetch_sources_for_symbols(
    conn: object, qualified_names: list[str]
) -> dict[str, str]:
    """Return ``{qualified_name → source_snippet}`` for a list of symbols.

    Args:
        conn: An open LadybugDB connection.  The caller owns the
            connection lifecycle; this function does not close it.
        qualified_names: Symbols whose source should be read.  Order is
            preserved in the iteration but irrelevant to the dict
            output.

    Returns:
        dict[str, str]: Per-symbol source snippets.  Missing or
        unreadable symbols map to an empty string rather than being
        omitted, so the caller can distinguish "no symbol" from "no
        snippet" at the dict-key level.
    """
    from codebase_rag.cypher_queries import (  # type: ignore[import-untyped]
        CYPHER_GET_FUNCTION_SOURCE_LOCATION,
    )

    snippets: dict[str, str] = {}
    for qn in qualified_names:
        try:
            rows = _result_to_rows(
                conn.execute(  # type: ignore[attr-defined]
                    CYPHER_GET_FUNCTION_SOURCE_LOCATION, {"node_id": qn}
                )
            )
            if not rows:
                snippets[qn] = ""
                continue
            r = rows[0]
            file_path: str = r.get("path") or ""
            root_path: str = r.get("root_path") or ""
            # CYPHER_GET_FUNCTION_SOURCE_LOCATION stores module paths
            # relative to the repo root.  Resolve to absolute using
            # root_path (stored on the Project node) before passing
            # to fetch_source, which checks os.path.exists().  Without
            # this every snippet is empty.
            if file_path and root_path and not Path(file_path).is_absolute():
                file_path = str(Path(root_path) / file_path)
            snippets[qn] = fetch_source(
                file_path, r.get("start_line"), r.get("end_line")
            )
        except Exception:
            # Empty string lets callers see which symbols failed to
            # resolve rather than silently dropping them from the
            # output dict.
            snippets[qn] = ""
    return snippets
