"""Harness wiring — inject a sentinel-bounded "this repo is indexed" block into
a repo's agent-config file so a coding agent uses codebase-indexer for it
automatically, with no separate skill-install step.

Mirrors the agentalloy ``wire`` approach: one idempotent, sentinel-bounded
block per repo, written into the harness config file detected from the repo
(CLAUDE.md / AGENTS.md / GEMINI.md / .clinerules / .cursor). ``unwire`` removes
only the bytes between the markers, leaving the user's own content intact.
"""
from __future__ import annotations

from pathlib import Path

SENTINEL_BEGIN = "<!-- BEGIN codebase-indexer -->"
SENTINEL_END = "<!-- END codebase-indexer -->"

# harness name -> path (relative to the repo root) of the file we write into.
# Detection (below) picks the first whose marker exists; --harness overrides.
_HARNESS_TARGETS: dict[str, str] = {
    "claude": "CLAUDE.md",
    "agents": "AGENTS.md",
    "gemini": "GEMINI.md",
    "cline": ".clinerules",
    "cursor": ".cursor/rules/codebase-indexer.mdc",
}


def detect_line_ending(content: str) -> str:
    """Return ``\\r\\n`` if the content uses CRLF, else ``\\n``."""
    return "\r\n" if "\r\n" in content else "\n"


def harness_target(repo: Path, harness: str) -> tuple[str, Path]:
    """Resolve an explicit ``--harness`` name to its (name, target path)."""
    key = harness.lower()
    if key not in _HARNESS_TARGETS:
        raise ValueError(
            f"unknown harness {harness!r}; choose one of {sorted(_HARNESS_TARGETS)}"
        )
    return key, repo / _HARNESS_TARGETS[key]


def detect_harness(repo: Path) -> tuple[str, Path]:
    """Detect the harness from repo markers; default to Claude (CLAUDE.md).

    Priority mirrors agentalloy: tool-specific markers outrank CLAUDE.md (which
    several agents share and is a weaker signal). CLAUDE.md is created when no
    marker is present so wiring always has a target.
    """
    if (repo / ".cursor").is_dir() or (repo / ".cursorrules").exists():
        return "cursor", repo / _HARNESS_TARGETS["cursor"]
    if (repo / "GEMINI.md").exists():
        return "gemini", repo / "GEMINI.md"
    if (repo / ".clinerules").exists():
        return "cline", repo / ".clinerules"
    if (repo / "CLAUDE.md").exists():
        return "claude", repo / "CLAUDE.md"
    if (repo / "AGENTS.md").exists():
        return "agents", repo / "AGENTS.md"
    return "claude", repo / "CLAUDE.md"


def _candidate_targets(repo: Path) -> list[tuple[str, Path]]:
    """Every possible target file (for unwire to sweep without a hint)."""
    return [(name, repo / rel) for name, rel in _HARNESS_TARGETS.items()]


def build_block(slug: str, base_url: str) -> str:
    """The inner markdown block (without sentinels) injected per repo.

    Kept deliberately small — it loads into every agent session for this repo,
    so it carries only the slug, the daemon check, and the scoped commands.
    """
    return (
        "## codebase-indexer — code intelligence for this repo\n"
        "\n"
        f"This repo is indexed by codebase-indexer (slug `{slug}`, service "
        f"`{base_url}`). Prefer it over grep/file-reading to find code by "
        "intent, trace call graphs, or assemble cross-file context.\n"
        "\n"
        "Check the daemon first: `code-indexer status` (start: `code-indexer "
        "start`). Then query — stdout is JSON, scoped to this repo with "
        f"`--repo {slug}`:\n"
        "\n"
        f"- `code-indexer --json search \"<intent>\" -k 10 --repo {slug}`\n"
        f"- `code-indexer --json symbol <fqn> --repo {slug}`\n"
        f"- `code-indexer --json callers <fqn> --repo {slug}` (or `callees`)\n"
        "- `code-indexer --json bundle \"<task>\" --repo .` — grounded "
        "multi-file context\n"
        "\n"
        "Re-run `code-indexer index .` after large changes. This block is "
        "managed by codebase-indexer (`code-indexer unwire .` to remove); edit "
        "outside the markers."
    )


def replace_marked_block(
    existing: str, block: str, begin: str = SENTINEL_BEGIN, end: str = SENTINEL_END
) -> str:
    """Insert or replace a sentinel-bounded block in ``existing`` content.

    Replaces between existing markers in place, else appends. Raises on
    duplicate or inverted markers so a malformed file is never corrupted.
    """
    nl = detect_line_ending(existing) if existing else "\n"
    full_block = f"{begin}{nl}{block}{nl}{end}"

    if existing.count(begin) > 1 or existing.count(end) > 1:
        raise ValueError(
            "target file contains duplicate codebase-indexer sentinels; "
            "refusing to write."
        )

    if begin in existing and end in existing:
        begin_idx = existing.index(begin)
        end_idx = existing.index(end) + len(end)
        if existing.index(end) < begin_idx:
            raise ValueError(
                "sentinel END appears before BEGIN in target file; refusing to write."
            )
        # consume a trailing newline after the old END marker
        if end_idx < len(existing):
            if existing[end_idx : end_idx + 2] == "\r\n":
                end_idx += 2
            elif existing[end_idx] in ("\n", "\r"):
                end_idx += 1
        return existing[:begin_idx] + full_block + nl + existing[end_idx:]

    if existing and not existing.endswith(nl):
        existing += nl
    if existing:
        existing += nl  # blank line separator before the appended block
    return existing + full_block + nl


def remove_sentinel_block(
    content: str, begin: str = SENTINEL_BEGIN, end: str = SENTINEL_END
) -> str:
    """Remove the sentinel-bounded block (inclusive) from ``content``.

    Returns the content unchanged if the markers are absent. Raises on
    inverted markers.
    """
    if begin not in content or end not in content:
        return content
    b = content.index(begin)
    e = content.index(end) + len(end)
    if content.index(end) < b:
        raise ValueError(
            "sentinel END appears before BEGIN in target file; refusing to remove."
        )
    if content[e : e + 2] == "\r\n":
        e += 2
    elif e < len(content) and content[e] == "\n":
        e += 1
    # also drop one blank separator line preceding the block
    if b >= 1 and content[b - 1] == "\n":
        b -= 1
        if b >= 1 and content[b - 1] == "\n":
            b -= 1
    result = content[:b] + content[e:]
    while "\n\n\n" in result:
        result = result.replace("\n\n\n", "\n\n")
    return result


def wire_repo(
    repo: Path, *, slug: str, base_url: str, harness: str | None = None
) -> dict[str, str]:
    """Write/refresh the sentinel block into the repo's harness file.

    Returns ``{harness, target, action, slug}`` where action is one of
    ``created`` / ``appended`` / ``updated``.
    """
    repo = Path(repo)
    name, target = harness_target(repo, harness) if harness else detect_harness(repo)
    existing = target.read_text(encoding="utf-8") if target.exists() else ""
    if SENTINEL_BEGIN in existing:
        action = "updated"
    elif existing:
        action = "appended"
    else:
        action = "created"
    updated = replace_marked_block(existing, build_block(slug, base_url))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(updated, encoding="utf-8")
    return {"harness": name, "target": str(target), "action": action, "slug": slug}


def unwire_repo(repo: Path, *, harness: str | None = None) -> dict[str, list[str]]:
    """Remove the sentinel block from the harness file(s).

    With an explicit ``harness`` only that file is touched; otherwise every
    known target containing our marker is swept. A dedicated file we own
    (``.mdc``) that becomes empty is deleted.
    """
    repo = Path(repo)
    targets = (
        [harness_target(repo, harness)] if harness else _candidate_targets(repo)
    )
    removed: list[str] = []
    for name, target in targets:
        if not target.exists():
            continue
        content = target.read_text(encoding="utf-8")
        if SENTINEL_BEGIN not in content:
            continue
        new = remove_sentinel_block(content)
        if new.strip():
            target.write_text(new, encoding="utf-8")
        elif name == "cursor":
            target.unlink()  # dedicated file we created
        else:
            target.write_text(new, encoding="utf-8")
        removed.append(str(target))
    return {"removed": removed}
