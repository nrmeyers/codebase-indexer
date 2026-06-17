"""Batch-generate per-symbol "card" descriptions for a repo.

Symbol cards (docs/retrieval-methodology-from-agentalloy.md §5) are the
document-expansion technique: a one-line, **task-vocabulary** description
of what a symbol does — using the words a developer would put in a ticket
("login", "rate limit", "permissions"), NOT the symbol's own identifiers.
These bridge "how users ask" to "how code is named", and were AgentAlloy's
single largest measured win (+0.067).

This script produces the descriptions OFFLINE (decoupled from indexing, so
generation cost is paid once and the result is auditable/regenerable). It
writes a sidecar ``{slug}.cards.json`` = ``{qualified_name: description}``
next to the repo's graph DB; ``embed_driver`` reads it on the next index
pass and emits a ``{qname}::Symbol::card`` chunk per entry.

Generation uses a small LOCAL model via Ollama (qwen3.5:0.8b by default —
the "local LM in one afternoon" path). Best-effort and resumable: existing
entries are kept, only missing symbols are generated, any per-symbol
failure is skipped. A symbol with no card simply gets no card chunk (its
identity header already lives in the function embed text), so the feature
degrades to today's behaviour.

Run with the service STOPPED or alongside it (graph is opened read-only).
CPU-pin the process; the LM runs on the Ollama host (3060).

Usage:
    uv run python scripts/generate_symbol_cards.py SLUG=REPO_ROOT [...] \
        [--limit N] [--model qwen3.5:0.8b] [--concurrency 6]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import errno
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings, slugify_repo  # noqa: E402
from app.scripts.embed_driver import should_skip_embed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("symbol_cards")

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "qwen3.5:0.8b"
_SRC_CAP_LINES = 45
_SRC_CAP_CHARS = 1700
_MAX_DESC_WORDS = 20

# Symbol-enumeration cypher — mirrors embed_driver's Function/Method read.
_CYPHER = """
MATCH (m:Module)-[:DEFINES]->(n:Function)
RETURN n.qualified_name AS qn, n.start_line AS sl, n.end_line AS el,
       m.path AS rel_path, n.docstring AS docstring
UNION ALL
MATCH (m:Module)-[:DEFINES]->(_c:Class)-[:DEFINES_METHOD]->(n:Method)
RETURN n.qualified_name AS qn, n.start_line AS sl, n.end_line AS el,
       m.path AS rel_path, n.docstring AS docstring
"""

# The style rule is load-bearing: the description must use the words a TASK
# contains, not the code's own identifiers (that's what makes it a bridge
# rather than a restatement of the name the embedder already has).
_PROMPT = """Write a one-line description of what this code does, for a \
search index.

Use plain problem-domain words — the kind that appear in a bug report or \
ticket ("login", "retry", "rate limit", "permissions", "cache", \
"pagination", "webhook") — and TRANSLATE the code's identifiers into those \
words rather than repeating them.

STRICT RULES:
- Begin with a present-tense verb (Checks, Stores, Routes, Retries, \
Validates, ...). Do NOT begin with "A developer", "This function", or \
"This code".
- Do NOT use the symbol's own name or any class/type identifiers from the \
code. If the only honest description repeats the name, describe the \
PURPOSE instead.
- One sentence, max {maxw} words. No quotes, no preamble, no trailing notes.

Symbol: {name}
Code:
{src}

Description:"""


def _rows_for_repo(repo_db_path: str) -> list[dict[str, Any]]:
    import ladybug as lb  # type: ignore[import-untyped]

    from app.services.ladybug_buffer_pool import resolve_buffer_pool_size

    db = lb.Database(
        repo_db_path, read_only=True, buffer_pool_size=resolve_buffer_pool_size()
    )
    conn = lb.Connection(db)
    res = conn.execute(_CYPHER)
    cols = res.get_column_names()
    rows: list[dict[str, Any]] = []
    while res.has_next():
        rows.append(dict(zip(cols, res.get_next())))
    return rows


def _read_source(repo_root: Path, rel_path: str, sl: int, el: int) -> str:
    try:
        p = Path(rel_path)
        if not p.is_absolute():
            p = repo_root / rel_path
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    start = max(0, int(sl or 1) - 1)
    # Read the symbol's OWN span [sl, el] only — never bleed into the next
    # function (that caused descriptions of the wrong, longer neighbour).
    # _SRC_CAP_LINES is a hard maximum for very long bodies.
    end = int(el) if el else start + 12
    end = min(end, start + _SRC_CAP_LINES, len(lines))
    end = max(end, start + 1)
    return "\n".join(lines[start:end])[:_SRC_CAP_CHARS]


def _describe(client: httpx.Client, model: str, name: str, src: str) -> str | None:
    if not src.strip():
        return None
    body = {
        "model": model,
        "prompt": _PROMPT.format(maxw=_MAX_DESC_WORDS, name=name, src=src),
        "stream": False,
        # §8 trap: pin thinking OFF or small models route the answer to a
        # hidden reasoning field and return empty content.
        "think": False,
        "options": {"temperature": 0.2, "num_predict": 96},
    }
    r = client.post(OLLAMA_URL, json=body, timeout=90)
    r.raise_for_status()
    desc = (r.json().get("response") or "").strip()
    # Sanitise: collapse whitespace, drop wrapping quotes, clamp length.
    desc = " ".join(desc.replace("\n", " ").split()).strip('"').strip()
    # Strip the residual preamble the model sometimes still emits, so the
    # card opens on the verb / problem words (cleaner retrieval signal).
    for _pre in ("This function ", "This code ", "This method ", "It "):
        if desc.startswith(_pre):
            desc = desc[len(_pre):]
            desc = desc[:1].upper() + desc[1:]
            break
    if not desc:
        return None
    words = desc.split()
    if len(words) > _MAX_DESC_WORDS + 6:
        desc = " ".join(words[: _MAX_DESC_WORDS + 6])
    return desc


def _src_hash(src: str) -> str:
    """SHA-1 fingerprint of the symbol's source body; resumability key."""
    return hashlib.sha1(src.encode("utf-8", errors="replace")).hexdigest()


def _atomic_write_cards(out_path: Path, cards: dict[str, dict[str, str]]) -> None:
    """Atomic write of the sidecar so a crash mid-flush cannot mask a clean
    empty file for the next run. ENOSPC is logged and re-raised — better to
    abort the slug than to leave a half-written cards.json on disk."""
    tmp = out_path.with_suffix(".json.tmp")
    payload = json.dumps(cards, indent=0, sort_keys=True)
    try:
        tmp.write_text(payload)
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            log.error("ENOSPC writing %s — aborting checkpoint", tmp)
        # Best-effort cleanup; ignore unlink failures.
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    os.replace(tmp, out_path)


def _load_cards(out_path: Path) -> dict[str, dict[str, str]]:
    """Load + migrate the sidecar to the {qn: {desc, src_hash}} shape.

    Legacy sidecars stored ``{qn: desc}`` (str values); migrate in-memory
    so existing entries with no known src_hash are treated as stale and
    refreshed on the next pass (src_hash="" never matches a real hash).
    """
    if not out_path.exists():
        return {}
    try:
        raw = json.loads(out_path.read_text())
    except Exception:
        return {}
    out: dict[str, dict[str, str]] = {}
    for qn, entry in raw.items():
        if isinstance(entry, str):
            out[qn] = {"desc": entry, "src_hash": ""}
        elif isinstance(entry, dict) and entry.get("desc"):
            out[qn] = {
                "desc": str(entry["desc"]),
                "src_hash": str(entry.get("src_hash") or ""),
            }
    return out


def generate_for_repo(
    slug: str, repo_root: Path, model: str, concurrency: int, limit: int | None
) -> int:
    repo_db_path = settings.db_path_for_repo(slug)
    if not Path(repo_db_path).exists():
        log.error("%s: no graph db at %s — skipping", slug, repo_db_path)
        return 0
    rows = _rows_for_repo(repo_db_path)
    # Skip test/generated/vendored symbols — they don't earn a card.
    todo = [
        r for r in rows
        if r.get("qn") and not should_skip_embed(str(r.get("rel_path") or ""))
    ]

    out_path = Path(settings.LADYBUG_DB_DIR) / f"{slug}.cards.json"
    cards = _load_cards(out_path)

    repo_root = repo_root.expanduser().resolve()

    # Pre-hash each row's source so the resumability check is body-aware:
    # a stored card whose src_hash no longer matches the body has gone stale
    # and must be regenerated.
    pending: list[tuple[dict[str, Any], str, str]] = []
    for r in todo:
        src = _read_source(repo_root, str(r.get("rel_path") or ""), r["sl"], r["el"])
        if not src.strip():
            continue
        h = _src_hash(src)
        existing = cards.get(r["qn"])
        if existing and existing.get("src_hash") == h:
            continue
        pending.append((r, src, h))
    if limit:
        pending = pending[:limit]
    log.info(
        "%s: %d symbols, %d already carded, %d to generate",
        slug, len(todo), len(cards), len(pending),
    )
    if not pending:
        return 0

    aborted = False

    with httpx.Client() as client:
        def _work(item: tuple[dict[str, Any], str, str]) -> tuple[str, str, str | None]:
            r, src, h = item
            name = str(r["qn"]).split("::")[0].split(".")[-1]
            return r["qn"], h, _describe(client, model, name, src)

        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [ex.submit(_work, item) for item in pending]
            for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
                try:
                    qn, h, desc = fut.result()
                except httpx.HTTPStatusError as exc:
                    if 500 <= exc.response.status_code < 600:
                        log.error(
                            "%s: Ollama 5xx (%s) — aborting slug",
                            slug, exc.response.status_code,
                        )
                        aborted = True
                        break
                    continue
                except Exception:
                    continue
                if desc:
                    cards[qn] = {"desc": desc, "src_hash": h}
                    done += 1
                if i % 25 == 0:
                    log.info("  %s: %d/%d generated", slug, i, len(pending))
                    _atomic_write_cards(out_path, cards)

    _atomic_write_cards(out_path, cards)
    log.info("%s: wrote %d cards -> %s%s",
             slug, len(cards), out_path, " (aborted)" if aborted else "")
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pairs", nargs="+", help="SLUG=REPO_ROOT ...")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="cap symbols (testing)")
    args = ap.parse_args()

    for pair in args.pairs:
        if "=" not in pair:
            log.error("bad arg (want SLUG=REPO_ROOT): %s", pair)
            return 2
        slug_raw, root = pair.split("=", 1)
        generate_for_repo(
            slugify_repo(slug_raw), Path(root), args.model, args.concurrency, args.limit
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
