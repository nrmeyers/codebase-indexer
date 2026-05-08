"""Hierarchical chunking strategies for semantic indexing (Phase 1.2).

Beyond per-Function/Method embedding, semantic search benefits from coarser
"summary" chunks that surface high-level intent:

  * **File**       — one short LLM-generated summary per source file
  * **Class**      — deterministic embed text built from tree-sitter (signature
                     + docstring + member names); no LLM call
  * **Module**     — Python ``__init__.py`` deterministic chunk built from
                     module docstring + ``__all__`` (or top-level public names)

This module is **pure data + pure-python helpers**.  It does not perform any
I/O, hold any state, or call any external service.  The embed driver in
``app/routers/index.py`` constructs these objects, builds the embed text via
the helpers below, and pipes the result through the existing SageMaker
embedding batcher and DuckDB ``embeddings`` table.  ``symbol_type`` already
exists on the schema (``EmbeddingRow.symbol_type``) — the new kinds simply
ride alongside ``Function`` / ``Method``.

Qualified-name convention for summary chunks (so they don't collide with
real symbols):

    {repo}.{module_path}::{kind}::summary

Phase 1.2 ships Class chunks (deterministic) end-to-end.  File chunks (LLM
+ cost cap) and Module chunks (deterministic, requires Python AST inside
the embed subprocess) are queued for Phase 1.2b — see PR description.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# File-summary prompt cap on the *content* portion of the prompt.  8 KB is
# enough for the vast majority of source files; truncating beyond this avoids
# blowing past Haiku's context budget when a single file is unusually large
# (vendored code, generated machine output, etc.).
FILE_SUMMARY_CONTENT_CAP_BYTES: Final[int] = 8_192

# Maximum cumulative spend on per-file summarization for a single repo
# index pass.  Hard ceiling: when crossed, the summarization phase aborts
# with a WARN log and ingestion of Function/Method continues normally.
FILE_SUMMARY_REPO_COST_CAP_USD: Final[float] = 1.50

# Verified Manifest pricing for ``claude-haiku-4-5`` (TheForge
# ``src/services/orchestration/pricing.ts:36``):
#   input  $0.80 per 1M tokens
#   output $4.00 per 1M tokens
# Per-file: ~600 in + 180 out  ≈  $0.0012 / file.
HAIKU_INPUT_USD_PER_TOKEN: Final[float] = 0.80 / 1_000_000
HAIKU_OUTPUT_USD_PER_TOKEN: Final[float] = 4.00 / 1_000_000


# ---------------------------------------------------------------------------
# Data classes — one per chunk kind
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FileSummaryChunk:
    """LLM-generated summary of a single source file.

    Generated post-hoc once the structural pass has completed.  The summary
    text is what gets embedded; ``file_path`` is stored as the chunk's
    ``file_path`` and the qname follows the ``::File::summary`` convention.
    """

    qname: str
    file_path: str
    summary_text: str
    symbol_kind: str = "File"


@dataclass(frozen=True)
class ClassSummaryChunk:
    """Deterministic class summary built from tree-sitter output.

    No LLM call: the embed input is a structured header listing the class
    signature, docstring, and the qualified names of every method/attribute
    the class defines.  This makes class-level semantic search hit the
    container, not just one of its methods.
    """

    qname: str
    file_path: str
    signature: str
    member_qnames: list[str] = field(default_factory=list)
    docstring: str = ""
    symbol_kind: str = "Class"


@dataclass(frozen=True)
class ModuleChunk:
    """Python ``__init__.py`` summary.

    Other languages don't have a directly comparable "module surface" file,
    so this chunk kind is Python-only for now.  Built from the module
    docstring and the names exposed via ``__all__`` (falling back to
    top-level non-underscore names when ``__all__`` is absent).
    """

    qname: str
    file_path: str
    docstring: str
    public_symbols: list[str] = field(default_factory=list)
    symbol_kind: str = "Module"


# ---------------------------------------------------------------------------
# Helpers — embed input builders
# ---------------------------------------------------------------------------


# File-summary prompt — sent verbatim to Manifest Haiku.  Tightly worded to
# discourage filler.  Phase 1.2b wiring point.
FILE_SUMMARY_PROMPT_TEMPLATE: Final[str] = (
    "Summarize this file in <=180 tokens. Focus on:\n"
    "- What it does (one sentence)\n"
    "- Top-level exports\n"
    "- What it imports / depends on (if relevant)\n"
    "- Any non-obvious gotchas\n"
    "Avoid: vague platitudes, \"this file is a TypeScript file\", etc.\n"
    "File: {path}\n"
    "Content: {content}"
)


def build_file_summary_input(path: str, content: str) -> str:
    """Build the LLM prompt for a file-summary chunk.

    The content is byte-capped at :data:`FILE_SUMMARY_CONTENT_CAP_BYTES`
    so a single oversized file can never blow past Haiku's context window
    or the per-file cost estimate.  The cap is applied on the UTF-8 byte
    length and back-truncated to a valid string boundary.
    """
    encoded = content.encode("utf-8", errors="replace")
    if len(encoded) > FILE_SUMMARY_CONTENT_CAP_BYTES:
        encoded = encoded[:FILE_SUMMARY_CONTENT_CAP_BYTES]
        # Truncate to a valid UTF-8 boundary (drop trailing partial bytes).
        content = encoded.decode("utf-8", errors="ignore")
    return FILE_SUMMARY_PROMPT_TEMPLATE.format(path=path, content=content)


def build_class_chunk_input(
    class_qname: str,
    class_signature: str,
    member_names: list[str],
    docstring: str = "",
    module_path: str = "",
) -> str:
    """Build the embed text for a Class summary chunk.

    Output format (matches the spec):

        # Class: <qname>
        # Module: <module_path>
        # Members: <member1>, <member2>, ...
        # ---
        <signature>
        <docstring>

    ``Members`` is omitted when there are zero members; ``Module`` is
    omitted when ``module_path`` is empty.  Trailing whitespace is stripped.
    """
    lines: list[str] = [f"# Class: {class_qname}"]
    if module_path:
        lines.append(f"# Module: {module_path}")
    if member_names:
        lines.append(f"# Members: {', '.join(member_names)}")
    lines.append("# ---")
    if class_signature:
        lines.append(class_signature)
    if docstring:
        lines.append(docstring)
    return "\n".join(lines).rstrip()


def build_module_chunk_input(
    module_qname: str,
    module_path: str,
    docstring: str,
    public_symbols: list[str],
) -> str:
    """Build the embed text for a Python ``__init__.py`` Module chunk.

    Output format:

        # Module: <qname>
        # Path: <module_path>
        # Public: <symbol1>, <symbol2>, ...
        # ---
        <docstring>
    """
    lines: list[str] = [f"# Module: {module_qname}"]
    if module_path:
        lines.append(f"# Path: {module_path}")
    if public_symbols:
        lines.append(f"# Public: {', '.join(public_symbols)}")
    lines.append("# ---")
    if docstring:
        lines.append(docstring)
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------


def estimate_haiku_call_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost of a single Haiku call at verified pricing.

    Per ``src/services/orchestration/pricing.ts:36`` in TheForge:
      claude-haiku-4-5 = $0.80/MTok input, $4.00/MTok output.

    Used by the embed driver to keep a running total against the
    repo-level cost cap (:data:`FILE_SUMMARY_REPO_COST_CAP_USD`).
    """
    return (
        input_tokens * HAIKU_INPUT_USD_PER_TOKEN
        + output_tokens * HAIKU_OUTPUT_USD_PER_TOKEN
    )


def make_summary_qname(repo: str, module_path: str, kind: str) -> str:
    """Build the conventional summary-chunk qualified name.

    Format: ``{repo}.{module_path}::{kind}::summary``

    The ``::`` separator avoids collision with real Function/Method qnames
    (which use ``.`` joins).  ``module_path`` may be empty for top-level
    files; the result is still distinct because of the ``::`` markers.
    """
    base = f"{repo}.{module_path}" if module_path else repo
    return f"{base}::{kind}::summary"
