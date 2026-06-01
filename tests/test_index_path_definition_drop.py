"""Integration regression: the REAL ``/index`` path fails loud on a silent drop.

Unlike ``test_definition_node_flush_at_scale`` (ingestor in isolation) and
``test_definition_silent_drop_guard`` (``flush_nodes`` in isolation), this test
drives the actual blocking-index entry point — ``_blocking_index`` — on a
real multi-file fixture, with the Kùzu write-drop injected into the live
``LadybugIngestor`` connection.  It is the test that would have caught the
production "369-node truncation" while the in-isolation primitive tests passed:
the parse genuinely walks the fixture files, but every ``Function`` write
no-ops at the engine while ``execute`` raises (the documented buffer-pool mmap
failure under memory pressure).

Guarantee: ``_blocking_index`` propagates ``DefinitionFlushError`` so the job
is marked failed at the flush that dropped — it does NOT swallow the drop,
proceed to the embedding pass, and report a structural-only skeleton as done.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _write_fixture_repo(root: Path) -> None:
    """A tiny but real multi-file Python repo the parser can fully ingest."""
    (root / "pkg").mkdir(parents=True, exist_ok=True)
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "alpha.py").write_text(
        "def alpha_one():\n"
        "    return 1\n\n"
        "def alpha_two(x):\n"
        "    return alpha_one() + x\n"
    )
    (root / "pkg" / "beta.py").write_text(
        "class Beta:\n"
        "    def method_a(self):\n"
        "        return 2\n\n"
        "def beta_fn():\n"
        "    return Beta().method_a()\n"
    )


def test_blocking_index_fails_loud_on_silent_definition_drop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "fixture_repo"
    repo.mkdir()
    _write_fixture_repo(repo)

    # Route the per-repo DB into tmp so the live indexer's graphs are untouched.
    db_dir = tmp_path / "dbs"
    db_dir.mkdir()
    monkeypatch.setenv("LADYBUG_DB_DIR", str(db_dir))
    monkeypatch.setenv("KUZU_BUFFER_POOL_SIZE", str(512 * 1024 * 1024))

    from app.routers import index as index_mod
    from app.services import ladybug_ingestor as li_mod

    real_execute = li_mod.LadybugIngestor._execute_query

    def _dropping_execute(self, query: str, params=None):  # type: ignore[no-untyped-def]
        # Mimic the engine write-drop: Function node writes no-op while the
        # engine reports a buffer-pool mmap failure (NOT an idempotency hit).
        if "MERGE (n:Function" in query or "CREATE (n:Function" in query:
            raise RuntimeError(
                "Buffer manager exception: Mmap for size 8796093022208 failed"
            )
        return real_execute(self, query, params)

    monkeypatch.setattr(
        li_mod.LadybugIngestor, "_execute_query", _dropping_execute
    )

    job = index_mod._Job(job_id="t-drop", repo_path=str(repo))

    # The real index path must surface the drop as a DefinitionFlushError
    # raised from within the parse/flush, NOT proceed to embedding and report
    # a structural-only graph as done.
    with pytest.raises(li_mod.DefinitionFlushError):
        index_mod._blocking_index(job, force_reindex=True)


def test_blocking_index_succeeds_when_writes_land(tmp_path: Path, monkeypatch) -> None:
    """Control: the same fixture with no injected drop ingests definitions.

    Proves the guard added for the drop case does not false-positive on a
    healthy parse.  Embeddings are skipped via the GraphUpdater
    ``skip_embeddings`` wiring already used by the index path; the external
    embedding subprocess is stubbed so the test stays hermetic and fast.
    """
    repo = tmp_path / "ok_repo"
    repo.mkdir()
    _write_fixture_repo(repo)

    db_dir = tmp_path / "dbs_ok"
    db_dir.mkdir()
    monkeypatch.setenv("LADYBUG_DB_DIR", str(db_dir))
    monkeypatch.setenv("KUZU_BUFFER_POOL_SIZE", str(512 * 1024 * 1024))

    from app.routers import index as index_mod

    # Stub the embedding subprocess — it requires the embedder model + network
    # and is out of scope for the graph-persistence guarantee under test.
    monkeypatch.setattr(index_mod, "_blocking_embed", lambda _job: None)

    job = index_mod._Job(job_id="t-ok", repo_path=str(repo))
    index_mod._blocking_index(job, force_reindex=True)

    # No DefinitionFlushError raised and the post-job truncation guard did not
    # fire (it raises before counts are stamped). ``job.node_count`` /
    # ``job.rel_count`` are populated by the authoritative post-flush count
    # query in ``_blocking_index`` itself, so asserting on them exercises the
    # real persistence path without a fragile second DB open. The fixture has
    # 3 functions + 1 method + a Beta class across 2 modules, so the count
    # comfortably exceeds the structural-only skeleton (which is what the
    # 369-node bug produced).
    assert job.status != "failed", f"index unexpectedly failed: {job.error}"
    assert job.node_count > 6, (
        f"only {job.node_count} nodes persisted — definitions were dropped "
        f"(structural-only skeleton, the 369-node bug)"
    )
    assert job.rel_count > 0, "no relationships persisted"
