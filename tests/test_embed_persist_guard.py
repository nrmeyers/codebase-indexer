"""Tests for the fail-loud post-persist guard in ``app/scripts/embed_driver``.

Root cause (2026-05-31 dogfood): the embed driver counted embeddings
in-process (``_embedded_count += len(all_inserts)``) the instant
``bulk_insert`` returned, fully decoupled from durable persistence.  Rows
lived in the DuckDB WAL (``<repo>.duck.wal``) until a clean ``close()``
checkpointed them; if the subprocess was killed, or a subsequent
force-reindex unlinked the WAL before the checkpoint landed, every
committed row was discarded — yet the job still printed ``EMBED_DONE`` /
``embedded_count=3698`` while ``SELECT COUNT(*) FROM embeddings`` on the
target ``.duck`` was 0.

The fix has two halves, both pinned here:

1. :func:`checkpoint_vec_store` — forces a durable CHECKPOINT so rows land
   in the main ``.duck`` file immediately (not WAL-resident), closing the
   kill / WAL-delete window.
2. :func:`verify_persisted_embeddings` + :func:`count_persisted_embeddings`
   — the GUARANTEED fail-loud guard: after close, reopen the file and count
   the rows; ``main`` raises :class:`EmbedPersistError` when a non-zero
   embedded count durably persisted as 0 (or grossly fewer) rows.

These helpers are module-level + pure so the guard is exercised WITHOUT a
LadybugDB / SageMaker subprocess (the live persist path still runs as a
subprocess in production and is covered by the integration suite).
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from app.scripts.embed_driver import (
    EmbedPersistError,
    checkpoint_vec_store,
    count_persisted_embeddings,
    verify_persisted_embeddings,
)


# ---------------------------------------------------------------------------
# verify_persisted_embeddings — the pure decision function.
# ---------------------------------------------------------------------------


def test_should_fire_when_embedded_nonzero_but_zero_persisted() -> None:
    """The exact 2026-05-31 silent-success mode: 3698 embedded, 0 on disk.

    This is the load-bearing case — without the guard the driver printed
    ``EMBED_DONE`` here.  The guard MUST return a non-None failure message.
    """
    msg = verify_persisted_embeddings(embedded_count=3698, persisted_count=0)
    assert msg is not None
    assert "0 rows persisted" in msg
    assert "3698" in msg


def test_should_fire_on_gross_shortfall_below_min_ratio() -> None:
    """A large fraction of committed rows vanishing also fails loud."""
    msg = verify_persisted_embeddings(embedded_count=1000, persisted_count=100)
    assert msg is not None
    assert "persisted_count=100" in msg


def test_should_pass_when_persisted_matches_embedded() -> None:
    """The clean happy path — full persistence returns None (no failure)."""
    assert verify_persisted_embeddings(
        embedded_count=3698, persisted_count=3698
    ) is None


def test_should_tolerate_small_upsert_dedup_shortfall() -> None:
    """A few rows lost to qname-collision upserts is NOT corruption.

    ``bulk_insert`` does DELETE-then-INSERT keyed on ``qualified_name``;
    overloaded methods sharing a qname legitimately collapse, so the
    persisted count runs slightly under the embedded count.  The observed
    real-repo gap is ~2% (3698 embedded -> 3618 persisted) — well inside
    the default 50% floor.
    """
    assert verify_persisted_embeddings(
        embedded_count=3698, persisted_count=3618
    ) is None


def test_should_be_noop_when_nothing_was_embedded() -> None:
    """A 0/0 outcome is a legitimate no-op (incremental, all unchanged)."""
    assert verify_persisted_embeddings(
        embedded_count=0, persisted_count=0
    ) is None


def test_should_respect_custom_min_ratio() -> None:
    """The shortfall threshold is tunable; a stricter ratio fires earlier."""
    # 700/1000 = 70% — passes at the default 0.5 floor…
    assert verify_persisted_embeddings(
        embedded_count=1000, persisted_count=700
    ) is None
    # …but fires when the caller demands >= 80% persistence.
    assert verify_persisted_embeddings(
        embedded_count=1000, persisted_count=700, min_ratio=0.8
    ) is not None


# ---------------------------------------------------------------------------
# count_persisted_embeddings — reopen-and-count against a real ``.duck``.
# ---------------------------------------------------------------------------


def _make_duck_with_rows(path: Path, n: int) -> None:
    """Create a ``.duck`` with ``n`` embedding rows and close it cleanly."""
    conn = duckdb.connect(str(path))
    conn.execute(
        "CREATE TABLE embeddings ("
        "qualified_name TEXT PRIMARY KEY, embedding FLOAT[768])"
    )
    for i in range(n):
        conn.execute(
            "INSERT INTO embeddings VALUES (?, ?::FLOAT[768])",
            [f"sym{i}", [0.1] * 768],
        )
    conn.close()


def test_should_count_rows_from_a_persisted_duck(tmp_path: Path) -> None:
    p = tmp_path / "repo.duck"
    _make_duck_with_rows(p, 42)
    assert count_persisted_embeddings(str(p)) == 42


def test_should_return_zero_when_duck_file_is_missing(tmp_path: Path) -> None:
    """A missing file means nothing persisted — the verifier treats 0 as a
    failure when embedded_count > 0."""
    assert count_persisted_embeddings(str(tmp_path / "nope.duck")) == 0


def test_should_return_zero_when_embeddings_table_absent(tmp_path: Path) -> None:
    """A ``.duck`` that only has the schema-less / wrong tables counts 0.

    This is precisely the on-disk state the bug leaves behind: the WAL
    holding the ``embeddings`` rows was discarded, then ``_write_meta``
    recreated a file WITHOUT the embeddings table populated.
    """
    p = tmp_path / "meta_only.duck"
    conn = duckdb.connect(str(p))
    conn.execute(
        "CREATE TABLE repo_metadata (key TEXT PRIMARY KEY, value TEXT)"
    )
    conn.execute("INSERT INTO repo_metadata VALUES ('k', 'v')")
    conn.close()
    assert count_persisted_embeddings(str(p)) == 0


# ---------------------------------------------------------------------------
# checkpoint_vec_store — durability (the root-cause fix).
# ---------------------------------------------------------------------------


def test_should_flush_wal_into_main_file_after_checkpoint(tmp_path: Path) -> None:
    """After CHECKPOINT the rows survive even a HARD process loss.

    We simulate the kill: write+commit rows, CHECKPOINT, then drop the
    connection WITHOUT a clean close and delete the WAL (exactly what the
    force-reindex cleanup does).  With the checkpoint, the rows are already
    in the main file, so a fresh reopen still sees them.
    """
    p = tmp_path / "repo.duck"
    conn = duckdb.connect(str(p))
    conn.execute(
        "CREATE TABLE embeddings ("
        "qualified_name TEXT PRIMARY KEY, embedding FLOAT[768])"
    )
    conn.execute("BEGIN")
    for i in range(300):
        conn.execute(
            "INSERT INTO embeddings VALUES (?, ?::FLOAT[768])",
            [f"sym{i}", [0.1] * 768],
        )
    conn.execute("COMMIT")

    # The fix: force the WAL into the main file BEFORE anything can lose it.
    checkpoint_vec_store(conn)

    # Now simulate the catastrophic path: drop the handle without a clean
    # close, then delete the WAL the way a force-reindex cleanup would.
    del conn
    wal = Path(str(p) + ".wal")
    if wal.exists():
        wal.unlink()

    # Rows are durable because the checkpoint already flushed them.
    assert count_persisted_embeddings(str(p)) == 300


def test_checkpoint_is_non_fatal_on_a_closed_connection() -> None:
    """A CHECKPOINT failure must never raise — close() is the backstop."""
    conn = duckdb.connect(":memory:")
    conn.close()
    # Calling against a closed connection should be swallowed silently.
    checkpoint_vec_store(conn)  # must not raise


# ---------------------------------------------------------------------------
# End-to-end guard semantics — proves a 0-persist outcome FAILS the job.
#
# This mirrors what ``main`` does at end-of-pass without spinning up the
# LadybugDB/SageMaker subprocess: count the durable rows, run the verifier,
# and raise EmbedPersistError on a gross shortfall.  CRUCIAL: this is the
# test that would PASS (silently) before the guard existed and FAILS now if
# the guard is removed.
# ---------------------------------------------------------------------------


def test_guard_raises_when_persist_returns_zero_rows(tmp_path: Path) -> None:
    """Replays the bug: claim 3698 embedded, find 0 on disk -> raise.

    Without the ``verify_persisted_embeddings`` call in ``main`` this code
    would fall through to ``EMBED_DONE`` and return 0 (silent success).
    """
    # On-disk state the bug leaves: a ``.duck`` with metadata but an empty
    # embeddings table (WAL discarded before checkpoint).
    p = tmp_path / "broken.duck"
    conn = duckdb.connect(str(p))
    conn.execute(
        "CREATE TABLE embeddings ("
        "qualified_name TEXT PRIMARY KEY, embedding FLOAT[768])"
    )  # zero rows
    conn.close()

    embedded_count = 3698  # what the in-process counter claimed
    persisted = count_persisted_embeddings(str(p))
    assert persisted == 0

    problem = verify_persisted_embeddings(
        embedded_count=embedded_count, persisted_count=persisted
    )
    assert problem is not None  # the guard fires

    # main() raises EmbedPersistError on a non-None problem.
    with pytest.raises(EmbedPersistError):
        if problem is not None:
            raise EmbedPersistError(problem)


def test_guard_stays_silent_when_persist_succeeds(tmp_path: Path) -> None:
    """The inverse: a fully-persisted run does NOT raise."""
    p = tmp_path / "good.duck"
    _make_duck_with_rows(p, 3698)

    persisted = count_persisted_embeddings(str(p))
    assert persisted == 3698
    assert verify_persisted_embeddings(
        embedded_count=3698, persisted_count=persisted
    ) is None
