"""Regression: silent definition write-drop is caught at the flush that drops.

Root cause of the "369-node truncation" (fix/definition-node-flush-at-scale):
on a memory-pressured, co-tenanted box (local 24 GB LLM + indexer), Kùzu's
mmap-backed buffer pool cannot back its dirty pages, so a bulk node write
*no-ops* while ``Connection.execute()`` raises a binder / IO error.  The
per-node ``try/except`` in ``flush_nodes`` previously swallowed that error to
DEBUG, cleared the buffer, and returned as if successful — so the parse logged
"1649 files" while ZERO Function/Method/Class nodes landed.  The job reported
``status=done`` (or, with the #96 guard, failed only 9.5 min later in the
post-job count) over a structural-only skeleton.

These tests prove ``flush_nodes`` now raises ``DefinitionFlushError`` at the
exact flush that drops every definition write — and, critically, that the
benign paths (idempotent re-write, structural-label failure, mixed
partial-success) do NOT trip the guard.  They exercise the real
``LadybugIngestor.flush_nodes`` code path with a connection stub that injects
the silent-drop failure mode, so the test fails if the at-source detection is
ever removed (unlike the in-isolation primitive tests, which passed while the
real path silently truncated).
"""
from __future__ import annotations

import threading
from typing import Any

import pytest

from app.services.ladybug_ingestor import (
    DefinitionFlushError,
    LadybugIngestor,
    _DEFINITION_NODE_LABELS,
)


class _FakeConn:
    """Connection stub whose ``execute`` selectively raises a runtime error.

    ``fail_substrings`` — when any appears in the Cypher query, ``execute``
    raises a non-idempotent RuntimeError (mimics the Kùzu mmap write-drop:
    the write no-ops but execute() reports a binder/IO failure).  Every other
    query succeeds (returns an empty result-like object).
    """

    def __init__(self, fail_substrings: tuple[str, ...]) -> None:
        self._fail = fail_substrings
        self.calls: list[str] = []

    def execute(self, query: str, params: dict[str, Any] | None = None) -> Any:
        self.calls.append(query)
        if any(s in query for s in self._fail):
            # NOT an "already exists" / "constraint" message → real failure
            # that flush_nodes must NOT treat as idempotent success.
            raise RuntimeError(
                "Buffer manager exception: Mmap for size 8796093022208 failed"
            )

        class _Res:
            def has_next(self) -> bool:
                return False

            def get_column_names(self) -> list[str]:
                return []

        return _Res()


def _ingestor_with_conn(conn: _FakeConn) -> LadybugIngestor:
    ing = LadybugIngestor.__new__(LadybugIngestor)  # bypass __enter__ DB open
    ing.conn = conn  # type: ignore[assignment]
    ing._conn_lock = threading.Lock()
    ing._use_merge = True
    ing.node_buffer = []
    ing._rel_count = 0
    ing._node_count_total = 0
    ing._rel_count_total = 0
    ing.batch_size = 1000
    return ing


def test_definition_labels_set_matches_schema() -> None:
    """The guard must cover every code-definition label the parser emits."""
    assert _DEFINITION_NODE_LABELS == frozenset(
        {"Function", "Method", "Class", "Interface", "Enum"}
    )


def test_raises_when_every_function_write_silently_drops() -> None:
    """A non-empty Function batch that lands ZERO rows must fail loud."""
    conn = _FakeConn(fail_substrings=("MERGE (n:Function",))
    ing = _ingestor_with_conn(conn)

    # Queue structural nodes (which succeed) + 50 Function nodes (which drop).
    ing.ensure_node_batch("Project", {"name": "Repo"})
    for i in range(50):
        ing.ensure_node_batch(
            "Function", {"qualified_name": f"Repo.mod.f{i}", "name": f"f{i}"}
        )

    with pytest.raises(DefinitionFlushError) as exc:
        ing.flush_nodes()

    # The error must name the dropped label and the zero/total write ratio.
    assert "Function" in str(exc.value)
    assert "0/50" in str(exc.value)
    # Buffer is cleared even on the failing flush so state stays consistent.
    assert ing.node_buffer == []


def test_method_drop_also_raises() -> None:
    """Method (the BUC-1621 symptom label) is covered by the guard too."""
    conn = _FakeConn(fail_substrings=("MERGE (n:Method",))
    ing = _ingestor_with_conn(conn)
    for i in range(10):
        ing.ensure_node_batch(
            "Method", {"qualified_name": f"Repo.C.m{i}", "name": f"m{i}"}
        )
    with pytest.raises(DefinitionFlushError):
        ing.flush_nodes()


def test_structural_label_drop_does_not_raise() -> None:
    """A structural-only label failing must NOT trip the definition guard.

    A legitimately code-free repo (or a transient structural hiccup) should
    not be force-failed by this guard — it targets only the lost-definitions
    truncation.  flush_nodes returns normally; the post-job guard / counts
    own the broader health check.
    """
    conn = _FakeConn(fail_substrings=("MERGE (n:Folder",))
    ing = _ingestor_with_conn(conn)
    for i in range(20):
        ing.ensure_node_batch("Folder", {"path": f"dir/{i}"})
    # No raise — Folder is not a definition label.
    ing.flush_nodes()
    assert ing.node_buffer == []


def test_idempotent_already_exists_does_not_raise() -> None:
    """A batch where every write is a benign 'already exists' is success."""

    class _AlreadyExistsConn(_FakeConn):
        def execute(self, query: str, params: dict[str, Any] | None = None) -> Any:
            self.calls.append(query)
            if "MERGE (n:Function" in query:
                raise RuntimeError("Runtime exception: node already exists")
            return super().execute(query, params)

    conn = _AlreadyExistsConn(fail_substrings=())
    ing = _ingestor_with_conn(conn)
    for i in range(30):
        ing.ensure_node_batch(
            "Function", {"qualified_name": f"Repo.f{i}", "name": f"f{i}"}
        )
    # "already exists" is counted as a flushed (idempotent) success → no raise.
    ing.flush_nodes()
    assert ing.node_count == 30


def test_partial_success_does_not_raise() -> None:
    """If even ONE definition write lands, the batch is not a silent drop."""

    class _OneSucceedsConn(_FakeConn):
        def __init__(self) -> None:
            super().__init__(fail_substrings=())
            self._seen = 0

        def execute(self, query: str, params: dict[str, Any] | None = None) -> Any:
            self.calls.append(query)
            if "MERGE (n:Function" in query:
                self._seen += 1
                if self._seen > 1:  # first write succeeds, rest drop
                    raise RuntimeError("Buffer manager exception: Mmap failed")
            return super().execute(query, params)

    conn = _OneSucceedsConn()
    ing = _ingestor_with_conn(conn)
    for i in range(40):
        ing.ensure_node_batch(
            "Function", {"qualified_name": f"Repo.f{i}", "name": f"f{i}"}
        )
    # flushed == 1 (> 0) → not a total drop → no raise (loud per-row ERRORs
    # already logged; the post-job count guard catches systemic loss).
    ing.flush_nodes()
    assert ing.node_count == 1
