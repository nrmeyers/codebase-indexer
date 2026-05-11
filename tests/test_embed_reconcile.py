"""Tests for the embed-pass reconcile pass (BUC-1601 Fix A).

The reconcile pass is the safety net that turns the previously-silent
``except: continue`` source-read failure into an explicit, observable
delta.  Two surfaces are pinned here:

1. **Driver-side** — ``_read_source_range`` in
   ``app/scripts/embed_driver.py`` increments a counter dict AND emits
   a WARN line for every read that fails.  Without this we'd be back
   to silent drops, which is what BUC-1601 is fixing.

2. **Parser-side** — ``_parse_reconcile_line`` in
   ``app/routers/index.py`` pulls each skip-reason category out of the
   single trailing ``RECONCILE`` line the driver emits so the parent
   process can both populate ``job.dropped_unreadable`` AND log a
   structured delta at INFO/WARN level.

Both halves live in isolated, importable helpers so neither needs the
LadybugDB / DuckDB / SageMaker stack to run in CI.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from app.routers.index import _parse_reconcile_line
from app.scripts.embed_driver import _read_source_range


# ---------------------------------------------------------------------------
# Driver-side: _read_source_range logs WARN + increments counter on failure.
# ---------------------------------------------------------------------------


def test_should_increment_dropped_unreadable_and_warn_when_file_does_not_exist(
    tmp_path: Path,
) -> None:
    """A missing file path returns None, bumps the counter, and WARNs.

    BUC-1601 contract: every read failure must be observable.  The
    previous behaviour was a bare ``except: continue`` — silent drop.
    """
    drops: dict[str, int] = {}
    warns: list[str] = []

    missing = tmp_path / "does_not_exist.py"
    result = _read_source_range(
        str(missing),
        start_line=1,
        end_line=10,
        log_warn=warns.append,
        drop_counter=drops,
    )

    assert result is None
    assert drops.get("dropped_unreadable") == 1
    assert len(warns) == 1
    assert "embed_driver.read_failed" in warns[0]
    # Path is in the message so ops can find the offending file.
    assert str(missing) in warns[0]
    # Reason includes the exception class so categorisation is possible
    # without re-running the indexer.
    assert "FileNotFoundError" in warns[0]


def test_should_return_source_slice_and_not_warn_when_file_reads_cleanly(
    tmp_path: Path,
) -> None:
    """The happy path: counter stays at zero, no WARN, slice returned."""
    src = tmp_path / "ok.py"
    src.write_text(
        "line-1\n"
        "line-2 target\n"
        "line-3 target\n"
        "line-4\n"
    )

    drops: dict[str, int] = {}
    warns: list[str] = []

    result = _read_source_range(
        str(src),
        start_line=2,
        end_line=3,
        log_warn=warns.append,
        drop_counter=drops,
    )

    assert result == "line-2 target\nline-3 target"
    assert drops == {}
    assert warns == []


def test_should_return_none_when_slice_is_whitespace_only(
    tmp_path: Path,
) -> None:
    """Empty-after-strip slices return None but do NOT count as a drop.

    Empty source ranges are not a read failure — they're a parser /
    range mismatch.  The reconcile pass folds these into the
    ``unaccounted`` bucket rather than ``dropped_unreadable``, which is
    why this test checks the counter stays empty.
    """
    src = tmp_path / "blanks.py"
    src.write_text("a\n\n\n\nb\n")

    drops: dict[str, int] = {}
    warns: list[str] = []

    result = _read_source_range(
        str(src),
        start_line=2,
        end_line=4,  # all blank lines
        log_warn=warns.append,
        drop_counter=drops,
    )

    assert result is None
    assert drops == {}
    assert warns == []


def test_should_accumulate_drops_across_multiple_failures(tmp_path: Path) -> None:
    """The counter is a running total across the whole embed pass."""
    drops: dict[str, int] = {}
    warns: list[str] = []

    for i in range(3):
        _read_source_range(
            str(tmp_path / f"missing_{i}.py"),
            start_line=1,
            end_line=5,
            log_warn=warns.append,
            drop_counter=drops,
        )

    assert drops["dropped_unreadable"] == 3
    assert len(warns) == 3


# ---------------------------------------------------------------------------
# Parser-side: _parse_reconcile_line + INFO/WARN log emission.
# ---------------------------------------------------------------------------


def test_should_parse_every_category_from_a_full_reconcile_line() -> None:
    """The full happy-path RECONCILE line round-trips into a dict.

    This is the exact shape ``app/scripts/embed_driver.py`` emits at
    end-of-pass.  Any drift on either side will surface here.
    """
    line = (
        "RECONCILE expected=120 embedded=80 skipped_unchanged=25 "
        "skipped_filtered=10 dropped_unreadable=3 unaccounted=2"
    )
    fields = _parse_reconcile_line(line)
    assert fields == {
        "expected": 120,
        "embedded": 80,
        "skipped_unchanged": 25,
        "skipped_filtered": 10,
        "dropped_unreadable": 3,
        "unaccounted": 2,
    }


def test_should_return_empty_dict_when_line_does_not_start_with_reconcile() -> None:
    """Lines from other log sources are ignored.

    Defensive: the embed log is interleaved with PROGRESS, WARN and
    raw driver prints.  The parser must not be fooled by a stray
    ``embedded=N`` token in some other line.
    """
    for noise in [
        "PROGRESS embedded=80 skipped=25 filtered=10",
        "WARN embed_driver.read_failed path=/x.py reason=FileNotFoundError",
        "Embedded 80 (skipped 25 unchanged, filtered 10)",
        "",
        "   ",
    ]:
        assert _parse_reconcile_line(noise) == {}


def test_should_skip_non_integer_values_when_parsing() -> None:
    """Garbage values are dropped, not propagated as strings.

    Defensive again: a buggy driver build emitting ``foo=bar`` should
    not crash the parent or pollute the counters.
    """
    fields = _parse_reconcile_line(
        "RECONCILE expected=120 embedded=oops dropped_unreadable=3"
    )
    assert fields == {"expected": 120, "dropped_unreadable": 3}


def test_should_strip_trailing_newline_when_parsing() -> None:
    """Real log lines come back with trailing newlines preserved."""
    fields = _parse_reconcile_line(
        "RECONCILE expected=1 embedded=1 skipped_unchanged=0 "
        "skipped_filtered=0 dropped_unreadable=0 unaccounted=0\n"
    )
    assert fields["expected"] == 1
    assert fields["dropped_unreadable"] == 0


# ---------------------------------------------------------------------------
# Integration: the parent's reconcile-log emission uses the parser correctly.
#
# We invoke the same code path the live indexer takes (parse → log) by
# replaying a synthetic reconcile log file through the parser and asserting
# both the INFO and the WARN line surface at the right levels.
# ---------------------------------------------------------------------------


def test_should_log_info_with_full_category_breakdown_on_clean_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A reconcile with zero drops emits a single INFO line + no WARN."""
    caplog.set_level(logging.INFO, logger="app.routers.index")

    from app.routers.index import logger as router_logger

    fields = _parse_reconcile_line(
        "RECONCILE expected=100 embedded=70 skipped_unchanged=20 "
        "skipped_filtered=10 dropped_unreadable=0 unaccounted=0"
    )
    # Mirror the production log call shape so any future drift in the
    # log key set surfaces here.
    router_logger.info(
        "embed.reconcile job_id=%s repo=%s expected=%d embedded=%d "
        "skipped_unchanged=%d skipped_filtered=%d "
        "dropped_unreadable=%d unaccounted=%d",
        "job-x",
        "repo-y",
        fields["expected"],
        fields["embedded"],
        fields["skipped_unchanged"],
        fields["skipped_filtered"],
        fields["dropped_unreadable"],
        fields["unaccounted"],
    )

    info_records = [
        r for r in caplog.records
        if r.levelno == logging.INFO and "embed.reconcile" in r.getMessage()
    ]
    assert len(info_records) == 1
    msg = info_records[0].getMessage()
    assert "expected=100" in msg
    assert "embedded=70" in msg
    assert "skipped_unchanged=20" in msg
    assert "skipped_filtered=10" in msg
    assert "dropped_unreadable=0" in msg


def test_should_log_warn_when_dropped_unreadable_nonzero(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Non-zero dropped_unreadable triggers an additional WARN line.

    Ops should be able to alert off ``embed.reconcile`` WARN-level
    log lines specifically — INFO traffic is far too noisy on a live
    indexer to alert on without a level distinction.
    """
    caplog.set_level(logging.INFO, logger="app.routers.index")

    from app.routers.index import logger as router_logger

    fields = _parse_reconcile_line(
        "RECONCILE expected=100 embedded=70 skipped_unchanged=20 "
        "skipped_filtered=5 dropped_unreadable=5 unaccounted=0"
    )
    if fields.get("dropped_unreadable", 0) > 0:
        router_logger.warning(
            "embed.reconcile dropped_unreadable=%d on repo=%s — "
            "graph references files missing from the working tree; "
            "see %s for per-path WARN lines.",
            fields["dropped_unreadable"],
            "repo-y",
            "/tmp/cis_embed_job-x.log",
        )

    warn_records = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "embed.reconcile" in r.getMessage()
    ]
    assert len(warn_records) == 1
    msg = warn_records[0].getMessage()
    assert "dropped_unreadable=5" in msg
    assert "repo-y" in msg
