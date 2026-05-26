"""Unit tests for ``app.services.markdown_indexer`` (LE-136 / REG-D).

The module is deliberately pure (no LadybugDB, no SageMaker, no
filesystem-mutating side effects beyond reading via ``rglob``) so we can
exercise the corpus-selection and chunking behaviour without bringing up
the live indexer stack.

Coverage:

* :func:`discover_markdown_files` — whitelist (``.planning/``, ``docs/``,
  root README-style filenames) and exclude fragments
  (``node_modules/``, ``.venv/``).
* :func:`chunk_markdown_file` — heading-keyed split, preamble extraction,
  qualified_name slugging, dedup of repeated headings, oversize splitting.
* :func:`compose_markdown_embed_text` — embed-input header layout (pinned
  so any drift forces code review).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.services.markdown_indexer import (
    chunk_markdown_file,
    compose_markdown_embed_text,
    discover_markdown_files,
)


# ---------------------------------------------------------------------------
# discover_markdown_files
# ---------------------------------------------------------------------------


def _write(root: Path, rel: str, body: str = "# stub\n") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_should_pick_up_planning_and_docs_and_root_readmes_when_present(
    tmp_path: Path,
) -> None:
    _write(tmp_path, ".planning/LE-136-brief.md")
    _write(tmp_path, ".planning/phase-plans/PHASE_9.md")
    _write(tmp_path, "docs/adr/0001-foo.md")
    _write(tmp_path, "docs/README.md")
    _write(tmp_path, "README.md")
    _write(tmp_path, "CLAUDE.md")
    _write(tmp_path, "CONTEXT.md")
    # Noise that should be excluded.
    _write(tmp_path, "src/foo.md")  # not in whitelist
    _write(tmp_path, "NOTES.md")    # not in ROOT_DOC_FILES
    _write(tmp_path, "node_modules/pkg/README.md")
    _write(tmp_path, "docs/.venv/junk.md")
    _write(tmp_path, "docs/vendor/x.md")

    found = discover_markdown_files(tmp_path)
    rels = sorted(p.relative_to(tmp_path).as_posix() for p in found)

    assert rels == [
        ".planning/LE-136-brief.md",
        ".planning/phase-plans/PHASE_9.md",
        "CLAUDE.md",
        "CONTEXT.md",
        "README.md",
        "docs/README.md",
        "docs/adr/0001-foo.md",
    ]


def test_should_return_empty_when_no_eligible_markdown_present(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "src/foo.md")
    _write(tmp_path, "internal/notes.md")

    assert discover_markdown_files(tmp_path) == []


def test_should_be_deterministic_when_invoked_repeatedly(tmp_path: Path) -> None:
    _write(tmp_path, ".planning/a.md")
    _write(tmp_path, ".planning/b.md")
    _write(tmp_path, "docs/c.md")

    a = discover_markdown_files(tmp_path)
    b = discover_markdown_files(tmp_path)
    assert a == b


# ---------------------------------------------------------------------------
# chunk_markdown_file — heading-keyed split
# ---------------------------------------------------------------------------


def test_should_split_on_h1_h2_h3_when_headings_present() -> None:
    content = (
        "# Title\n"
        "Intro line.\n"
        "\n"
        "## Section A\n"
        "Body of A.\n"
        "\n"
        "### Subsection A1\n"
        "Detail.\n"
        "\n"
        "## Section B\n"
        "Body of B.\n"
    )
    chunks = chunk_markdown_file(
        repo_name="forge", rel_path=".planning/x.md", content=content
    )
    headings = [c.heading for c in chunks]
    assert headings == ["Title", "Section A", "Subsection A1", "Section B"]
    qnames = [c.qualified_name for c in chunks]
    assert qnames[0] == "forge::.planning/x.md::title"
    assert qnames[1] == "forge::.planning/x.md::section-a"
    # Section A chunk body should contain "Body of A" and the H3 subsection
    # body (because chunks are bounded by EQUAL-OR-HIGHER rank headings).
    assert "Body of A." in chunks[1].body


def test_should_extract_preamble_when_text_precedes_first_heading() -> None:
    content = (
        "Some prose with no heading.\n"
        "Another line.\n"
        "\n"
        "# First heading\n"
        "Body.\n"
    )
    chunks = chunk_markdown_file(
        repo_name="forge", rel_path="docs/intro.md", content=content
    )
    # First chunk is the preamble; second is the heading section.
    assert chunks[0].heading == "intro (preamble)"
    assert "Some prose" in chunks[0].body
    assert chunks[1].heading == "First heading"


def test_should_collapse_to_single_chunk_when_no_headings_present() -> None:
    content = "Just a flat document.\nNo headings here.\n"
    chunks = chunk_markdown_file(
        repo_name="forge", rel_path="README.md", content=content
    )
    assert len(chunks) == 1
    assert chunks[0].heading == "README"
    assert chunks[0].qualified_name == "forge::README.md::readme"
    assert "Just a flat document." in chunks[0].body


def test_should_dedupe_qualified_name_when_headings_repeat() -> None:
    content = (
        "## Notes\n"
        "First batch.\n"
        "\n"
        "## Notes\n"
        "Second batch.\n"
    )
    chunks = chunk_markdown_file(
        repo_name="forge", rel_path=".planning/dup.md", content=content
    )
    qnames = [c.qualified_name for c in chunks]
    # Two identical headings must not collide on PRIMARY KEY.
    assert len(set(qnames)) == len(qnames)
    assert qnames[0] == "forge::.planning/dup.md::notes"
    assert qnames[1] == "forge::.planning/dup.md::notes-2"


def test_should_return_empty_list_when_content_is_empty() -> None:
    assert chunk_markdown_file(
        repo_name="forge", rel_path="docs/empty.md", content=""
    ) == []
    assert chunk_markdown_file(
        repo_name="forge", rel_path="docs/empty.md", content="   \n  \n"
    ) == []


def test_should_split_oversized_section_when_body_exceeds_cap() -> None:
    # ~10k chars of body under one heading; the splitter must emit
    # multiple part-chunks rather than one huge one.
    big_body = "para " * 3000  # ~15k chars
    content = f"## Long\n{big_body}\n"
    chunks = chunk_markdown_file(
        repo_name="forge", rel_path=".planning/long.md", content=content
    )
    assert len(chunks) >= 2
    assert all(c.heading == "Long" for c in chunks)
    assert chunks[0].qualified_name.endswith("#part1")
    assert chunks[1].qualified_name.endswith("#part2")


# ---------------------------------------------------------------------------
# compose_markdown_embed_text — header layout pinned
# ---------------------------------------------------------------------------


def test_should_emit_pinned_header_layout_when_composing_embed_text() -> None:
    chunks = chunk_markdown_file(
        repo_name="forge",
        rel_path=".planning/x.md",
        content="## Heading\nBody.\n",
    )
    assert len(chunks) == 1
    out = compose_markdown_embed_text(chunks[0])
    lines = out.splitlines()
    assert lines[0] == "# MarkdownDoc: forge::.planning/x.md::heading"
    assert lines[1] == "# File: .planning/x.md"
    assert lines[2] == "# Heading: Heading"
    assert lines[3] == "# ---"
    assert "Body." in out


# ---------------------------------------------------------------------------
# Integration sketch — chunk → embed-text → hash (no SageMaker)
# ---------------------------------------------------------------------------


def test_should_produce_stable_content_hash_when_input_unchanged() -> None:
    """Embed-text composition must be deterministic so BUC-1518 skip works."""
    import hashlib

    content = "## Same\nSame body.\n"
    out_a = compose_markdown_embed_text(
        chunk_markdown_file(repo_name="r", rel_path="docs/a.md", content=content)[0]
    )
    out_b = compose_markdown_embed_text(
        chunk_markdown_file(repo_name="r", rel_path="docs/a.md", content=content)[0]
    )
    assert out_a == out_b
    assert hashlib.sha1(out_a.encode()).hexdigest() == hashlib.sha1(out_b.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Sanity guard — qualified_name namespace
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "heading,expected_slug",
    [
        ("REG-D Markdown gap", "reg-d-markdown-gap"),
        ("LE-123 vs LE-127", "le-123-vs-le-127"),
        ("   ", "section"),
        ("Section/with slashes", "section-with-slashes"),
        ("🚀 Launch", "launch"),
    ],
)
def test_should_slugify_heading_to_safe_qualified_name_form(
    heading: str, expected_slug: str
) -> None:
    content = f"## {heading}\nBody.\n"
    chunks = chunk_markdown_file(
        repo_name="r", rel_path="docs/x.md", content=content
    )
    assert len(chunks) == 1
    assert chunks[0].qualified_name == f"r::docs/x.md::{expected_slug}"
