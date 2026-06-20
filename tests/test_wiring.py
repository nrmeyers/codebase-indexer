"""Unit tests for harness wiring (``app.cli.wiring``).

These exercise the pure inject/replace/remove/detect helpers on a tmp repo —
no daemon, no network.
"""
from __future__ import annotations

import pytest

from app.cli import wiring


def test_wire_creates_claude_md_when_absent(tmp_path) -> None:
    res = wiring.wire_repo(tmp_path, slug="acme__widgets", base_url="http://localhost:8003")
    target = tmp_path / "CLAUDE.md"
    assert res["harness"] == "claude"
    assert res["action"] == "created"
    body = target.read_text(encoding="utf-8")
    assert wiring.SENTINEL_BEGIN in body and wiring.SENTINEL_END in body
    assert "acme__widgets" in body


def test_wire_appends_and_preserves_user_content(tmp_path) -> None:
    (tmp_path / "CLAUDE.md").write_text("# My repo\n\nHand-written notes.\n", encoding="utf-8")
    res = wiring.wire_repo(tmp_path, slug="a__b", base_url="http://x")
    assert res["action"] == "appended"
    body = (tmp_path / "CLAUDE.md").read_text(encoding="utf-8")
    assert "Hand-written notes." in body
    assert wiring.SENTINEL_BEGIN in body


def test_rewire_replaces_in_place_single_block(tmp_path) -> None:
    wiring.wire_repo(tmp_path, slug="old__one", base_url="http://x")
    res = wiring.wire_repo(tmp_path, slug="new__two", base_url="http://x")
    assert res["action"] == "updated"
    body = (tmp_path / "CLAUDE.md").read_text(encoding="utf-8")
    assert body.count(wiring.SENTINEL_BEGIN) == 1
    assert "new__two" in body
    assert "old__one" not in body


def test_unwire_removes_block_keeps_user_content(tmp_path) -> None:
    (tmp_path / "CLAUDE.md").write_text("# Repo\n\nKeep me.\n", encoding="utf-8")
    wiring.wire_repo(tmp_path, slug="s__t", base_url="u")
    res = wiring.unwire_repo(tmp_path)
    body = (tmp_path / "CLAUDE.md").read_text(encoding="utf-8")
    assert wiring.SENTINEL_BEGIN not in body
    assert "Keep me." in body
    assert str(tmp_path / "CLAUDE.md") in res["removed"]


def test_detect_prefers_cursor_over_claude(tmp_path) -> None:
    (tmp_path / "CLAUDE.md").write_text("x", encoding="utf-8")
    (tmp_path / ".cursor").mkdir()
    name, target = wiring.detect_harness(tmp_path)
    assert name == "cursor"
    assert target == tmp_path / ".cursor" / "rules" / "codebase-indexer.mdc"


def test_detect_defaults_to_claude(tmp_path) -> None:
    name, target = wiring.detect_harness(tmp_path)
    assert name == "claude"
    assert target == tmp_path / "CLAUDE.md"


def test_explicit_harness_override_writes_agents(tmp_path) -> None:
    res = wiring.wire_repo(tmp_path, slug="s__t", base_url="u", harness="agents")
    assert res["harness"] == "agents"
    assert (tmp_path / "AGENTS.md").exists()


def test_replace_marked_block_rejects_duplicate_markers() -> None:
    dup = (
        f"{wiring.SENTINEL_BEGIN}\nx\n{wiring.SENTINEL_END}\n"
        f"{wiring.SENTINEL_BEGIN}\ny\n{wiring.SENTINEL_END}"
    )
    with pytest.raises(ValueError):
        wiring.replace_marked_block(dup, "new")
