"""Tests for scripts/generate_symbol_cards.py.

Covers the three load-bearing units the script depends on:
* ``_read_source`` returns only [start, end] inclusive (no neighbour bleed).
* ``_describe`` parses the Ollama response shape correctly and strips the
  residual preamble small models still emit.
* ``generate_for_repo`` merges new entries with an existing sidecar instead
  of overwriting (resumability), uses the new {desc, src_hash} schema, and
  writes the file atomically via os.replace.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# Make ``scripts/`` importable so we can ``import generate_symbol_cards``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import generate_symbol_cards as gsc  # noqa: E402


def test_read_source_returns_only_span_lines(tmp_path: Path) -> None:
    src = tmp_path / "foo.py"
    src.write_text(
        "line-1\n"
        "line-2\n"
        "line-3\n"
        "line-4\n"
        "line-5\n"
    )
    # [2, 4] inclusive must return exactly lines 2-4 and never bleed into
    # line-5 (the neighbour-bleed bug that produced descriptions of the
    # wrong function).
    out = gsc._read_source(tmp_path, "foo.py", 2, 4)
    assert out == "line-2\nline-3\nline-4"
    assert "line-1" not in out
    assert "line-5" not in out


def test_describe_parses_response_and_strips_preamble() -> None:
    client = MagicMock()
    fake_resp = MagicMock()
    fake_resp.json.return_value = {
        "response": "This function checks the user's role and rejects unauthorised callers."
    }
    fake_resp.raise_for_status = MagicMock()
    client.post.return_value = fake_resp

    desc = gsc._describe(client, "qwen3.5:0.8b", "requireRole", "def f(): pass\n")
    assert desc is not None
    # "This function " preamble must be stripped; result begins with verb.
    assert not desc.startswith("This function")
    assert desc.startswith("Checks")


def test_describe_strips_quotes_and_collapses_whitespace() -> None:
    client = MagicMock()
    fake_resp = MagicMock()
    fake_resp.json.return_value = {
        "response": '"  Validates  the  payload  shape.  "'
    }
    fake_resp.raise_for_status = MagicMock()
    client.post.return_value = fake_resp

    desc = gsc._describe(client, "qwen3.5:0.8b", "validate", "def f(): pass\n")
    assert desc == "Validates the payload shape."


def test_generate_for_repo_merges_with_existing_sidecar(tmp_path: Path) -> None:
    # Pre-seed an existing cards.json so we can prove the new run keeps it
    # and only adds the missing entry (resumability).
    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()

    # Fixture source file each symbol points at — body is what
    # ``_src_hash`` will fingerprint. ``pkg.kept`` points at lines 1-2 so
    # we can pre-compute its src_hash and store it; the resumability
    # check then sees a hash match and never re-asks the LLM.
    src = tmp_path / "src.py"
    src.write_text("def kept_fn():\n    return 0\ndef new_fn():\n    return 1\n")
    kept_hash = gsc._src_hash(gsc._read_source(tmp_path, "src.py", 1, 2))

    out_path = cards_dir / "fake-slug.cards.json"
    out_path.write_text(json.dumps({
        "pkg.kept": {"desc": "Kept from prior run.", "src_hash": kept_hash},
    }))

    rows = [
        {"qn": "pkg.kept", "sl": 1, "el": 2, "rel_path": "src.py", "docstring": ""},
        {"qn": "pkg.new",  "sl": 3, "el": 4, "rel_path": "src.py", "docstring": ""},
    ]

    # Patch the small surface generate_for_repo touches:
    #   - settings.db_path_for_repo + settings.LADYBUG_DB_DIR
    #   - _rows_for_repo (don't open a real ladybug DB)
    #   - should_skip_embed (no rows are tests in this fixture)
    #   - httpx.Client (return a deterministic description)
    fake_db = tmp_path / "fake.db"
    fake_db.write_bytes(b"")  # exists() must return True

    fake_resp = MagicMock()
    fake_resp.json.return_value = {"response": "Returns the integer one."}
    fake_resp.raise_for_status = MagicMock()
    fake_client = MagicMock()
    fake_client.post.return_value = fake_resp
    fake_client_cm = MagicMock()
    fake_client_cm.__enter__ = MagicMock(return_value=fake_client)
    fake_client_cm.__exit__ = MagicMock(return_value=False)

    fake_settings = MagicMock()
    fake_settings.db_path_for_repo = MagicMock(return_value=str(fake_db))
    fake_settings.LADYBUG_DB_DIR = str(cards_dir)

    with patch.object(gsc, "settings", fake_settings), \
         patch.object(gsc, "_rows_for_repo", return_value=rows), \
         patch.object(gsc, "should_skip_embed", return_value=False), \
         patch.object(gsc.httpx, "Client", return_value=fake_client_cm):
        done = gsc.generate_for_repo(
            "fake-slug", tmp_path, "qwen3.5:0.8b", concurrency=1, limit=None,
        )

    assert done == 1
    final = json.loads(out_path.read_text())
    # Prior entry preserved verbatim.
    assert final["pkg.kept"]["desc"] == "Kept from prior run."
    assert final["pkg.kept"]["src_hash"] == kept_hash
    # New entry written under the new schema with a non-empty src_hash.
    assert "pkg.new" in final
    assert isinstance(final["pkg.new"], dict)
    assert final["pkg.new"]["desc"].startswith("Returns")
    assert final["pkg.new"]["src_hash"]


def test_atomic_write_cards_uses_os_replace(tmp_path: Path) -> None:
    """The temp file must be written, then os.replace'd into place — never
    a direct write_text on out_path (which can leave a half-empty file)."""
    out_path = tmp_path / "x.cards.json"
    gsc._atomic_write_cards(out_path, {"a": {"desc": "Adds one.", "src_hash": "h"}})
    assert out_path.exists()
    assert not out_path.with_suffix(".json.tmp").exists()
    assert json.loads(out_path.read_text())["a"]["desc"] == "Adds one."
