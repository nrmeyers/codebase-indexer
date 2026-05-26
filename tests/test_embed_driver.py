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
    partition_batch_result,
    resolve_batch_embedder,
    resolve_ingest_concurrency,
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


# ---------------------------------------------------------------------------
# resolve_batch_embedder — LE-151.
#
# The ingest pass MUST embed with the SAME model the query path uses
# (app/routers/search.py::_embed_query). These tests pin the resolution
# priority and prove that when a backend is configured, ingest routes
# through app.embedders (the query PRIMARY) and does NOT call the legacy
# codebase_rag.embedder.embed_code_batch CodeRankEmbed path.
# ---------------------------------------------------------------------------


class _FakeBackend:
    """Minimal EmbedderBackend stub with the async batch ``embed`` contract."""

    name = "sagemaker"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        # 768-dim is irrelevant for these assertions; 3-dim keeps it cheap.
        return [[float(len(t)), 0.0, 0.0] for t in texts]


@pytest.fixture
def _embedder_env(monkeypatch: pytest.MonkeyPatch):
    """Install fake ``app.embedders.sync_bridge``, ``app.services.lm_studio``
    and ``codebase_rag.embedder`` modules so ``resolve_batch_embedder`` can be
    exercised without a real backend / network. Returns the spies."""
    import sys
    import types

    spies: dict[str, object] = {}

    # codebase_rag.embedder.embed_code_batch — the legacy path we must NOT use
    # when a configured backend is available.
    code_calls: list[list[str]] = []

    def _embed_code_batch(texts: list[str]) -> list[list[float]]:
        code_calls.append(list(texts))
        return [[1.0, 1.0, 1.0] for _ in texts]

    codebase_rag = types.ModuleType("codebase_rag")
    codebase_rag_embedder = types.ModuleType("codebase_rag.embedder")
    codebase_rag_embedder.embed_code_batch = _embed_code_batch  # type: ignore[attr-defined]
    codebase_rag.embedder = codebase_rag_embedder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "codebase_rag", codebase_rag)
    monkeypatch.setitem(sys.modules, "codebase_rag.embedder", codebase_rag_embedder)
    spies["code_calls"] = code_calls

    return spies


def test_should_route_ingest_through_configured_backend_when_available(
    monkeypatch: pytest.MonkeyPatch, _embedder_env: dict[str, object]
) -> None:
    """With a configured backend, ingest embeds via app.embedders — the same
    PRIMARY the query path uses — and never calls embed_code_batch."""
    backend = _FakeBackend()

    def _get_embedder_or_none():
        return backend

    monkeypatch.setattr(
        "app.embedders.sync_bridge.get_embedder_or_none",
        _get_embedder_or_none,
    )

    fn = resolve_batch_embedder()
    out = fn(["hello", "world!"])

    # Routed through the configured backend with RAW text (no prefix) — the
    # exact symmetry contract with _embed_query.
    assert backend.calls == [["hello", "world!"]]
    assert out == [[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]]
    # The legacy CodeRankEmbed path was NOT touched.
    assert _embedder_env["code_calls"] == []


def test_should_resolve_same_embedder_for_ingest_and_query(
    monkeypatch: pytest.MonkeyPatch, _embedder_env: dict[str, object]
) -> None:
    """Ingest (resolve_batch_embedder) and query (get_embedder_or_none, used by
    _embed_query via embed_text_sync) resolve to the SAME backend object."""
    backend = _FakeBackend()
    monkeypatch.setattr(
        "app.embedders.sync_bridge.get_embedder_or_none",
        lambda: backend,
    )

    # Query side resolves the backend via the same sync_bridge entry point.
    from app.embedders.sync_bridge import get_embedder_or_none

    query_backend = get_embedder_or_none()

    # Ingest resolves through resolve_batch_embedder; prove it dispatches to
    # the identical object by capturing the call.
    fn = resolve_batch_embedder()
    fn(["x"])

    assert query_backend is backend
    assert backend.calls == [["x"]], "ingest dispatched to the query backend"


def test_should_fall_back_to_embed_code_batch_when_no_backend_or_lm_studio(
    monkeypatch: pytest.MonkeyPatch, _embedder_env: dict[str, object]
) -> None:
    """Last-resort symmetry: with NO configured backend and NO LM Studio,
    ingest falls back to the in-process embed_code_batch (matching the query
    side's final torch fallback)."""
    monkeypatch.setattr(
        "app.embedders.sync_bridge.get_embedder_or_none",
        lambda: None,
    )

    import sys
    import types

    lm = types.ModuleType("app.services.lm_studio")
    lm.can_embed = lambda: False  # type: ignore[attr-defined]
    services = sys.modules.get("app.services") or types.ModuleType("app.services")
    monkeypatch.setitem(sys.modules, "app.services", services)
    monkeypatch.setitem(sys.modules, "app.services.lm_studio", lm)

    fn = resolve_batch_embedder()
    out = fn(["a", "bb"])

    assert _embedder_env["code_calls"] == [["a", "bb"]]
    assert out == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]


# ---------------------------------------------------------------------------
# resolve_ingest_concurrency — LE-151b.
#
# Default MUST be 1: a bulk re-embed at the old default of 2 fanned ~8
# simultaneous invocations into a serverless endpoint and OOM'd the model
# worker ("Worker died." 500). Sequential batches prevent that.
# ---------------------------------------------------------------------------


def test_should_default_ingest_concurrency_to_one_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default concurrency is 1 (sequential) to survive a serverless endpoint."""
    monkeypatch.delenv("SAGEMAKER_EMBED_CONCURRENCY", raising=False)
    assert resolve_ingest_concurrency() == 1


def test_should_honour_concurrency_override_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operators on a provisioned endpoint can raise concurrency explicitly."""
    monkeypatch.setenv("SAGEMAKER_EMBED_CONCURRENCY", "4")
    assert resolve_ingest_concurrency() == 4


@pytest.mark.parametrize("bad", ["0", "-3", "garbage", ""])
def test_should_clamp_invalid_or_nonpositive_concurrency_to_one(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    """Invalid / non-positive overrides fall back to the safe default of 1."""
    monkeypatch.setenv("SAGEMAKER_EMBED_CONCURRENCY", bad)
    assert resolve_ingest_concurrency() == 1


# ---------------------------------------------------------------------------
# partition_batch_result — LE-151b fail-loud contract.
#
# When a batch embed fails after all SageMaker retries, the driver must
# count the failure and persist NOTHING for that batch — never fabricate
# empty/zero vectors and report success.
# ---------------------------------------------------------------------------

_META: list[tuple[str, str, int, int, str, str]] = [
    ("pkg.a", "/abs/a.py", 1, 2, "Function", "hasha"),
    ("pkg.b", "/abs/b.py", 3, 4, "Function", "hashb"),
]


def test_should_return_insertable_pairs_when_batch_succeeds() -> None:
    """Happy path: every (meta, vector) pair is returned, zero failures."""
    embs = [[0.1, 0.2], [0.3, 0.4]]
    pairs, failed = partition_batch_result(_META, embs, None)
    assert failed == 0
    assert [v for _m, v in pairs] == embs
    assert [m[0] for m, _v in pairs] == ["pkg.a", "pkg.b"]


def test_should_count_whole_batch_failed_and_persist_nothing_when_error() -> None:
    """A batch that raised after retries persists NOTHING and counts as failed.

    This is the core anti-silent-corruption guarantee: a failed embed is
    surfaced as a failure, never stored as empty vectors.
    """
    pairs, failed = partition_batch_result(
        _META, None, RuntimeError("Worker died. (after 5 attempts)")
    )
    assert pairs == []
    assert failed == len(_META) == 2


def test_should_count_batch_failed_when_embedding_count_mismatches_meta() -> None:
    """A truncated/corrupt result is a whole-batch failure, not a partial write.

    Zipping a short embedding list against the meta would silently drop the
    unmatched symbols; instead we fail the whole batch loudly.
    """
    pairs, failed = partition_batch_result(_META, [[0.1, 0.2]], None)
    assert pairs == []
    assert failed == 2
