"""Unit tests for the embed-driver helper functions (BUC-1601).

These cover the parts of ``app/scripts/embed_driver.py`` that used to be
trapped inside a ``python -c`` f-string and were therefore impossible to
unit-test:

* :func:`should_skip_embed` — the BUC-1519 skip-filter regex set.  Drift
  here is dangerous because it directly drives how much we spend on
  SageMaker; a regression toward "skip everything" would silently
  break semantic search recall.
* :func:`compute_content_hash` — the BUC-1518 SHA-1 fingerprint used to
  short-circuit re-embedding when a symbol is unchanged.  The hash MUST
  be stable across processes — if it ever drifts, every previously
  embedded symbol re-embeds on the next index, costing real money.
* :func:`compose_function_method_embed_text` — the header layout used as
  the actual embed input.  Pinned here so any change to the header
  shape is forced through code review (rather than buried inside a
  shelled-out string).

The driver's live LadybugDB / DuckDB / SageMaker code paths are NOT
exercised here — those still run as a subprocess in production and are
covered by the integration suite.
"""
from __future__ import annotations

import pytest

from app.scripts.embed_driver import (
    SKIP_PATTERNS,
    compose_function_method_embed_text,
    compute_content_hash,
    should_skip_embed,
)


# ---------------------------------------------------------------------------
# should_skip_embed — positive (must-skip) cases.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        # /tests/ and /test/ directory hits
        "tests/foo.py",
        "pkg/tests/helpers.py",
        "src/test/util.go",
        # language-specific suffix patterns
        "internal/sqlx_test.go",
        "src/foo.test.ts",
        "src/foo.spec.tsx",
        "src/foo.test.jsx",
        "components/Button.spec.js",
        # JS/TS __tests__ folder convention
        "src/__tests__/Button.test.ts",
        # Python pytest discovery patterns
        "pkg/test_models.py",
        "tests/conftest.py",
        "pkg/conftest.py",
        # protobuf / grpc generated stubs
        "api/svc.pb.go",
        "api/svc.pb.py",
        "api/svc_pb2.py",
        "api/svc_pb2_grpc.py",
        # explicit generated dirs and suffixes
        "internal/generated/types.go",
        "api/types_generated.go",
        "src/types_generated.ts",
        # vendored / build outputs
        "vendor/github.com/foo/bar/pkg.go",
        "frontend/node_modules/react/index.js",
        "scripts/.venv/lib/python3.12/site-packages/foo.py",
        "frontend/dist/bundle.js",
        "src/build/static/main.css",
    ],
)
def test_should_skip_test_generated_and_vendored_files_when_matched_by_skip_patterns(
    path: str,
) -> None:
    """should_skip_embed returns True for paths matching SKIP_PATTERNS."""
    assert should_skip_embed(path) is True, f"expected True for {path!r}"


# ---------------------------------------------------------------------------
# should_skip_embed — negative (must-not-skip) cases.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        # ordinary source files
        "src/router.py",
        "internal/handler.go",
        "src/components/Button.tsx",
        "pkg/foo.py",
        "app/services/orchestrator.py",
        # "testing" is a real word in some package names
        "pkg/testing_utils/inner.go",  # not under /tests/ nor matches _test\.go
        # files whose names happen to contain "test" but not as a suffix marker
        "src/contest.py",
        "src/latest.py",
        # tooling configuration at repo root
        "pyproject.toml",
        "package.json",
        # python init files at non-test paths
        "app/__init__.py",
    ],
)
def test_should_embed_normal_source_files_when_not_matched_by_skip_patterns(
    path: str,
) -> None:
    """should_skip_embed returns False for paths NOT matching SKIP_PATTERNS."""
    assert should_skip_embed(path) is False, f"expected False for {path!r}"


def test_skip_patterns_compile_and_count_matches_inline_list() -> None:
    """SKIP_PATTERNS is the source of truth; this pins the count.

    A drop in count is almost always a deletion mistake; a jump should
    come with a deliberate code review.  17 patterns at the time of
    BUC-1601 (Phase 1.4 baseline).
    """
    assert len(SKIP_PATTERNS) == 17


# ---------------------------------------------------------------------------
# compute_content_hash — stability + sensitivity to input changes.
# ---------------------------------------------------------------------------


def test_should_return_stable_sha1_when_input_unchanged() -> None:
    """The same input must always produce the same hex digest.

    BUC-1518 relies on this contract: if the hash drifts run-to-run we
    re-embed every symbol on every index even when nothing changed.
    """
    text = "# Function: pkg.fn\n# ---\ndef fn():\n    return 1\n"
    digest_a = compute_content_hash(text)
    digest_b = compute_content_hash(text)
    assert digest_a == digest_b
    # SHA-1 hex is 40 lowercase chars
    assert len(digest_a) == 40
    assert all(c in "0123456789abcdef" for c in digest_a)


def test_should_change_hash_when_any_byte_of_input_changes() -> None:
    """A single-character change flips the digest.

    Pin that the underlying hash is a real cryptographic hash, not a
    cheap collision-prone fingerprint.
    """
    base = "def fn():\n    return 1\n"
    perturbed = "def fn():\n    return 2\n"
    assert compute_content_hash(base) != compute_content_hash(perturbed)


# ---------------------------------------------------------------------------
# compose_function_method_embed_text — header layout contract.
# ---------------------------------------------------------------------------


def test_should_emit_module_and_callers_header_when_both_present() -> None:
    """Full header: type, qname, module, callers, separator, doc, source."""
    out = compose_function_method_embed_text(
        stype="Function",
        qname="pkg.mod.fn",
        callers=3,
        docstring="The fn() docstring.",
        src="def fn():\n    return 1",
        format_docstring=lambda d: d,  # passthrough
    )
    assert out.splitlines() == [
        "# Function: pkg.mod.fn",
        "# Module: pkg.mod",
        "# Callers: 3",
        "# ---",
        "The fn() docstring.",
        "def fn():",
        "    return 1",
    ]


def test_should_omit_callers_header_when_zero_callers() -> None:
    """Zero callers means the "# Callers: 0" line is suppressed.

    Saves a byte per symbol and keeps the embed input minimal for the
    common case of brand-new code.
    """
    out = compose_function_method_embed_text(
        stype="Method",
        qname="pkg.mod.Cls.fn",
        callers=0,
        docstring="",
        src="def fn(self): pass",
        format_docstring=lambda d: d,
    )
    # No "# Callers:" line.
    assert "Callers" not in out


def test_should_omit_module_header_when_qname_has_no_module_prefix() -> None:
    """Top-level qnames like "fn" omit the # Module: line."""
    out = compose_function_method_embed_text(
        stype="Function",
        qname="fn",
        callers=0,
        docstring="",
        src="def fn(): pass",
        format_docstring=lambda d: d,
    )
    assert "# Module:" not in out


def test_should_invoke_format_docstring_callback_when_docstring_nonempty() -> None:
    """format_docstring is the injected layout normaliser.

    Passing a custom format_docstring proves the helper is the only one
    deciding docstring layout, not the driver — which is the point of
    the parameter (test seam).
    """
    captured: list[str] = []

    def fake_format(d: str) -> str:
        captured.append(d)
        return f"<<{d}>>"

    out = compose_function_method_embed_text(
        stype="Function",
        qname="pkg.fn",
        callers=0,
        docstring="raw doc",
        src="def fn(): pass",
        format_docstring=fake_format,
    )
    assert captured == ["raw doc"]
    assert "<<raw doc>>" in out
