"""Markdown corpus discovery + chunking (LE-136 / REG-D).

The code-graph-rag ``GraphUpdater`` is intentionally Tree-sitter centric — it
only walks source files for languages with a registered parser (Python,
TypeScript, Go, etc.).  Project documentation (``.planning/*.md``,
``docs/*.md``, ``README.md``, ``CLAUDE.md``, ``CONTEXT.md``, …) therefore
never reaches the embeddings / tantivy indexes, and TheForge's ``/ai`` chat
turn cannot answer questions about its own LE numbering, ADRs, or planning
notes.  See ``TheForge/.planning/le-dogfood-2026-05-26T16-postwave3-baseline.md``
§ REG-D.

This module provides an **additive, post-pass** corpus discovery surface:

1. :func:`discover_markdown_files` walks a repo and returns the list of
   ``.md`` files that are eligible for indexing (``.planning/``, ``docs/``,
   and a tight set of root-level docs).  Anything outside those buckets is
   ignored on purpose — we do not want to embed arbitrary vendored
   markdown.

2. :func:`chunk_markdown_file` splits one markdown file into header-keyed
   sections.  Each chunk has a stable ``qualified_name``
   (``<repo>::<rel_path>::<heading_slug>``), the embed text body, the line
   range, and the heading title used for surfacing the chunk in search
   metadata.

Both functions are pure and deterministic so they can be unit-tested
without any LadybugDB / DuckDB / SageMaker dependency.  The wiring into
the indexer pipeline (``app/routers/index.py``) and embedder bridge lives
separately and is exercised only by the live indexer.

Tree-sitter is NOT used for parsing — markdown has no AST node type in
our graph schema and we treat the content as plain text.  Embedding goes
through the existing SageMaker code-embedding endpoint (e5-base-v2 /
Jina); the slight domain mismatch is acceptable because recall on
"LE-123" / "REG-D" style noun-phrase queries is dominated by the lexical
tantivy arm anyway.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Corpus selection — strict whitelist
# ---------------------------------------------------------------------------
#
# We deliberately do NOT index every ``.md`` file in the repo.  Vendored
# READMEs (``node_modules/foo/README.md``), test fixtures, and so on add
# noise without adding signal.  The buckets below mirror what TheForge
# actually keeps under version control as "real" docs:
#
#  * ``.planning/`` — phase plans, briefs, dogfood reports, LE tickets.
#  * ``docs/``       — ADRs, runbooks, architecture, deployment.
#  * Root-level documentation files that operators read first.
#
# Subdirectories under ``.planning/`` and ``docs/`` are included
# recursively; everything else is filtered out.

#: Directories whose ``.md`` files are eligible for indexing (recursive).
INCLUDED_DIRS: tuple[str, ...] = (".planning", "docs")

#: Root-level markdown filenames that are eligible (case-sensitive — these
#: match the canonical capitalisation used across our repos).
ROOT_DOC_FILES: tuple[str, ...] = (
    "README.md",
    "CLAUDE.md",
    "CONTEXT.md",
    "AGENTS.md",
    "ROADMAP.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "ARCHITECTURE.md",
)

#: Path fragments that disqualify a markdown file even when it lives under
#: ``.planning/`` or ``docs/``.  Mirrors the embed_driver SKIP_PATTERNS
#: intent (no vendored / generated / build output).
EXCLUDE_FRAGMENTS: tuple[str, ...] = (
    "node_modules/",
    ".venv/",
    "vendor/",
    "dist/",
    "build/",
    ".git/",
)


def _is_eligible(rel_path: str) -> bool:
    """True when ``rel_path`` (forward-slash separated) should be indexed.

    Args:
        rel_path: Repo-relative POSIX path (e.g. ``docs/adr/0001-foo.md``).

    Returns:
        ``True`` when the path matches the whitelist and does not contain
        any of :data:`EXCLUDE_FRAGMENTS`.
    """
    if not rel_path.endswith(".md"):
        return False
    for frag in EXCLUDE_FRAGMENTS:
        if frag in rel_path:
            return False
    # Root-level whitelist (no slash in the path).
    if "/" not in rel_path:
        return rel_path in ROOT_DOC_FILES
    # Directory whitelist — match on the first path segment.
    first = rel_path.split("/", 1)[0]
    return first in INCLUDED_DIRS


def discover_markdown_files(repo_root: Path) -> list[Path]:
    """Walk ``repo_root`` and return eligible ``.md`` file paths.

    Results are sorted for deterministic ordering (helps tests + diffs in
    progress logs).  Absolute paths are returned; callers compute the
    repo-relative path via ``path.relative_to(repo_root)``.

    Args:
        repo_root: Absolute path to the repo checkout root.

    Returns:
        Sorted list of absolute ``Path`` objects.  Empty when no eligible
        markdown is present.
    """
    out: list[Path] = []
    repo_root = repo_root.resolve()
    # rglob is fine here — the EXCLUDE_FRAGMENTS filter discards the heavy
    # noise dirs (node_modules / .venv) before we ever read a byte.
    for path in repo_root.rglob("*.md"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(repo_root).as_posix()
        except ValueError:
            continue
        if _is_eligible(rel):
            out.append(path)
    out.sort()
    return out


# ---------------------------------------------------------------------------
# Section-aware chunking
# ---------------------------------------------------------------------------
#
# Markdown is split on ATX headings (``#``, ``##``, ``###`` only — we
# ignore ``####+`` because deep subsections rarely contain enough text to
# stand on their own).  Each chunk carries:
#
#   * ``heading`` — the heading text, used for the ``qualified_name`` slug
#     and for surfacing the chunk in /search/semantic responses.
#   * ``body``    — the section body up to (but not including) the next
#     heading of equal or higher rank.
#   * ``start_line`` / ``end_line`` — 1-indexed inclusive line range,
#     matching the convention used elsewhere in the indexer.
#
# Files without any headings produce a single chunk covering the whole
# document, keyed by the filename stem.

# Limits — keep individual chunks within the SageMaker token budget.  The
# e5/Jina endpoints accept ~512 tokens of content; we cap at ~3500 chars
# (conservative ~4 chars/token) and emit a small overlap so a long
# section does not get truncated to a useless mid-sentence cutoff.
_MAX_CHARS = 3500
_OVERLAP_CHARS = 200

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")


def _slugify_heading(text: str) -> str:
    """Lowercase, ASCII-safe heading slug for ``qualified_name``.

    Args:
        text: Raw heading text.

    Returns:
        URL-safe slug (e.g. ``"REG-D — Markdown gap"`` -> ``"reg-d-markdown-gap"``).
        Empty input yields ``"section"``.
    """
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return cleaned or "section"


@dataclass(frozen=True)
class MarkdownChunk:
    """One section-keyed slice of a markdown file.

    Attributes mirror what :class:`codebase_rag.storage.vector_store.EmbeddingRow`
    needs at insert time, plus the heading title for downstream UX.
    """

    qualified_name: str
    file_path: str  # repo-relative POSIX path
    heading: str
    body: str
    start_line: int
    end_line: int


def _split_oversized(body: str, base_start_line: int) -> Iterable[tuple[str, int, int]]:
    """Yield ``(slice, start_line, end_line)`` for a too-large body.

    Args:
        body: The section body that exceeded :data:`_MAX_CHARS`.
        base_start_line: The 1-indexed line where ``body`` started.

    Yields:
        Tuples whose slice is at most :data:`_MAX_CHARS` chars (the last
        slice may be shorter).  Adjacent slices overlap by
        :data:`_OVERLAP_CHARS` chars on the character axis; the line
        ranges are approximate (computed by counting newlines in each
        slice).
    """
    pos = 0
    n = len(body)
    while pos < n:
        end = min(pos + _MAX_CHARS, n)
        slice_text = body[pos:end]
        # Line bounds within the slice.
        lines_before = body[:pos].count("\n")
        lines_in = slice_text.count("\n")
        start_line = base_start_line + lines_before
        end_line = start_line + lines_in
        yield slice_text, start_line, end_line
        if end >= n:
            break
        pos = end - _OVERLAP_CHARS
        if pos <= 0:
            pos = end


def chunk_markdown_file(
    *,
    repo_name: str,
    rel_path: str,
    content: str,
) -> list[MarkdownChunk]:
    """Split a markdown document into section-keyed chunks.

    Args:
        repo_name: Canonical repo slug (used to namespace the
            ``qualified_name``).
        rel_path: Repo-relative POSIX path of the markdown file.
        content: Raw file contents (UTF-8 decoded text).

    Returns:
        Ordered list of chunks; never empty for a non-empty file.  A
        document with no headings collapses to a single
        ``"<repo>::<rel_path>::<stem>"`` chunk.
    """
    lines = content.splitlines()
    if not lines:
        return []

    # First pass — find every H1/H2/H3 anchor with its line number.
    anchors: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m:
            anchors.append((idx, m.group(2).strip()))

    chunks: list[MarkdownChunk] = []

    if not anchors:
        # Whole-document fallback chunk.
        body = content.strip()
        if not body:
            return []
        stem = rel_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        qn_base = f"{repo_name}::{rel_path}::{_slugify_heading(stem)}"
        if len(body) <= _MAX_CHARS:
            chunks.append(
                MarkdownChunk(
                    qualified_name=qn_base,
                    file_path=rel_path,
                    heading=stem,
                    body=body,
                    start_line=1,
                    end_line=len(lines),
                )
            )
        else:
            for i, (slice_text, sl, el) in enumerate(
                _split_oversized(body, base_start_line=1)
            ):
                chunks.append(
                    MarkdownChunk(
                        qualified_name=f"{qn_base}#part{i + 1}",
                        file_path=rel_path,
                        heading=stem,
                        body=slice_text,
                        start_line=sl,
                        end_line=el,
                    )
                )
        return chunks

    # Add a sentinel so the last section gets a defined end-line.
    anchors_with_end = [(idx, head, anchors[i + 1][0] if i + 1 < len(anchors) else len(lines))
                       for i, (idx, head) in enumerate(anchors)]

    # Any content BEFORE the first heading becomes a preamble chunk.
    if anchors[0][0] > 0:
        preamble = "\n".join(lines[: anchors[0][0]]).strip()
        if preamble:
            stem = rel_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            qn = f"{repo_name}::{rel_path}::{_slugify_heading(stem)}-preamble"
            chunks.append(
                MarkdownChunk(
                    qualified_name=qn,
                    file_path=rel_path,
                    heading=f"{stem} (preamble)",
                    body=preamble,
                    start_line=1,
                    end_line=anchors[0][0],
                )
            )

    seen_slugs: dict[str, int] = {}
    for start_idx, heading, end_idx in anchors_with_end:
        section_lines = lines[start_idx:end_idx]
        body = "\n".join(section_lines).strip()
        if not body:
            continue
        slug = _slugify_heading(heading)
        # De-duplicate slugs within the same file (two ``## Notes`` sections
        # would otherwise collide on qualified_name PRIMARY KEY).
        seen_slugs[slug] = seen_slugs.get(slug, 0) + 1
        if seen_slugs[slug] > 1:
            slug = f"{slug}-{seen_slugs[slug]}"
        qn_base = f"{repo_name}::{rel_path}::{slug}"
        start_line = start_idx + 1  # 1-indexed
        end_line = end_idx  # exclusive in the slice -> inclusive line-count
        if len(body) <= _MAX_CHARS:
            chunks.append(
                MarkdownChunk(
                    qualified_name=qn_base,
                    file_path=rel_path,
                    heading=heading,
                    body=body,
                    start_line=start_line,
                    end_line=end_line,
                )
            )
        else:
            for i, (slice_text, sl, el) in enumerate(
                _split_oversized(body, base_start_line=start_line)
            ):
                chunks.append(
                    MarkdownChunk(
                        qualified_name=f"{qn_base}#part{i + 1}",
                        file_path=rel_path,
                        heading=heading,
                        body=slice_text,
                        start_line=sl,
                        end_line=min(el, end_line),
                    )
                )

    return chunks


def compose_markdown_embed_text(chunk: MarkdownChunk) -> str:
    """Build the embed input for one :class:`MarkdownChunk`.

    Mirrors the header layout used by
    :func:`app.scripts.embed_driver.compose_function_method_embed_text` so
    operators reading raw embed_input logs see a consistent shape:

        # MarkdownDoc: <qualified_name>
        # File: <rel_path>
        # Heading: <heading>
        # ---
        <body>

    Args:
        chunk: The chunk to format.

    Returns:
        Multi-line string used both as the SageMaker embed input AND the
        SHA-1 content-hash input (so the embed_driver-style incremental
        skip works for markdown without further plumbing).
    """
    return (
        f"# MarkdownDoc: {chunk.qualified_name}\n"
        f"# File: {chunk.file_path}\n"
        f"# Heading: {chunk.heading}\n"
        "# ---\n"
        f"{chunk.body}"
    )
