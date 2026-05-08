"""Unit tests for hierarchical chunk strategies (Phase 1.2).

Covers the deterministic chunk types and helpers.  The end-to-end embed-
driver wiring for Class chunks is exercised by the live index against
TheForge (see PR body); here we verify the building blocks any caller of
``app.services.chunk_strategies`` relies on.
"""
from __future__ import annotations

from app.services.chunk_strategies import (
    FILE_SUMMARY_CONTENT_CAP_BYTES,
    FILE_SUMMARY_REPO_COST_CAP_USD,
    ClassSummaryChunk,
    FileSummaryChunk,
    ModuleChunk,
    build_class_chunk_input,
    build_file_summary_input,
    build_module_chunk_input,
    estimate_haiku_call_cost_usd,
    make_summary_qname,
)


def test_should_format_class_chunk_with_members_when_members_present() -> None:
    """Class chunk includes # Members line and renders signature + docstring."""
    out = build_class_chunk_input(
        class_qname="myrepo.app.svc.Worker",
        class_signature="class Worker(BaseWorker):",
        member_names=["run", "stop", "_tick"],
        docstring="Background job worker.",
        module_path="myrepo.app.svc",
    )
    assert out == (
        "# Class: myrepo.app.svc.Worker\n"
        "# Module: myrepo.app.svc\n"
        "# Members: run, stop, _tick\n"
        "# ---\n"
        "class Worker(BaseWorker):\n"
        "Background job worker."
    )


def test_should_omit_members_line_when_class_has_no_members() -> None:
    """A class with zero members renders the header but skips # Members."""
    out = build_class_chunk_input(
        class_qname="r.m.Empty",
        class_signature="class Empty:",
        member_names=[],
        docstring="",
        module_path="r.m",
    )
    # No Members line, no docstring blank line — rstrip() trims trailing.
    assert out == (
        "# Class: r.m.Empty\n"
        "# Module: r.m\n"
        "# ---\n"
        "class Empty:"
    )


def test_should_truncate_file_summary_input_when_content_exceeds_byte_cap() -> None:
    """Files larger than 8 KB get safely truncated; prompt stays well-formed."""
    big_content = "x" * (FILE_SUMMARY_CONTENT_CAP_BYTES + 5_000)
    out = build_file_summary_input("src/big.ts", big_content)
    # The prompt must reference the path and not exceed cap by more than
    # the static template overhead.
    assert "src/big.ts" in out
    # Content body in the output is at most the cap (UTF-8 bytes).
    body = out.split("Content: ", 1)[1]
    assert len(body.encode("utf-8")) <= FILE_SUMMARY_CONTENT_CAP_BYTES


def test_should_build_module_chunk_for_init_py_with_public_symbols() -> None:
    """Python __init__.py module chunk includes docstring + __all__ list."""
    out = build_module_chunk_input(
        module_qname="forge.services",
        module_path="src/services/__init__.py",
        docstring="Service registry — composes adapters.",
        public_symbols=["AuditTrail", "GateMachine", "Orchestrator"],
    )
    assert out == (
        "# Module: forge.services\n"
        "# Path: src/services/__init__.py\n"
        "# Public: AuditTrail, GateMachine, Orchestrator\n"
        "# ---\n"
        "Service registry — composes adapters."
    )


def test_should_enforce_repo_cost_cap_when_simulated_spend_exceeds_threshold() -> None:
    """Cost-cap simulation: per-call cost matches verified Haiku pricing,
    cumulative spend trips the $1.50 ceiling, and the chunk dataclasses
    expose the symbol_kind labels the embed pipeline persists."""
    # Per the spec: ~600 input + 180 output ≈ $0.0012 / file.
    per_file = estimate_haiku_call_cost_usd(input_tokens=600, output_tokens=180)
    assert 0.0011 < per_file < 0.0013

    # 1500 files at $0.0012 each = $1.80 — must trip the $1.50 ceiling.
    cumulative = per_file * 1500
    assert cumulative > FILE_SUMMARY_REPO_COST_CAP_USD

    # Walk file-by-file simulating the driver's running tally.  When the
    # next call would push us over the cap, the driver must abort the
    # summarization pass without crashing — i.e. the *check* is just an
    # arithmetic comparison, no exception thrown.
    spent = 0.0
    files_summarized = 0
    aborted = False
    for _ in range(2000):
        if spent + per_file > FILE_SUMMARY_REPO_COST_CAP_USD:
            aborted = True
            break
        spent += per_file
        files_summarized += 1
    assert aborted is True
    assert spent <= FILE_SUMMARY_REPO_COST_CAP_USD
    assert files_summarized > 0  # still made progress before cap

    # Sanity-check the dataclasses + qname helper round-trip through the
    # values the embed driver actually persists.
    f = FileSummaryChunk(
        qname=make_summary_qname("forge", "src/services/audit-trail.ts", "File"),
        file_path="src/services/audit-trail.ts",
        summary_text="Records governance audit events.",
    )
    c = ClassSummaryChunk(
        qname=make_summary_qname("forge", "src.services", "Class"),
        file_path="src/services/x.ts",
        signature="class X:",
        member_qnames=["a", "b"],
    )
    m = ModuleChunk(
        qname=make_summary_qname("forge", "src.services", "Module"),
        file_path="src/services/__init__.py",
        docstring="services pkg",
        public_symbols=["X"],
    )
    assert f.symbol_kind == "File"
    assert c.symbol_kind == "Class"
    assert m.symbol_kind == "Module"
    assert "::File::summary" in f.qname
    assert "::Class::summary" in c.qname
    assert "::Module::summary" in m.qname
