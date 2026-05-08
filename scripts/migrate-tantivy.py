#!/usr/bin/env python3
"""Phase 1.1 — Migrate already-indexed repos into the Tantivy BM25 index.

Reads symbol metadata from each per-repo LadybugDB (``.cgr/repos/<slug>.db``)
and writes equivalent docs to the matching Tantivy index
(``.cgr/repos/<slug>.tantivy/``).  Idempotent: re-running over a populated
index simply re-adds documents — Tantivy has no unique-key concept so we
trade a small amount of duplication on repeat runs for a one-shot bulk
loader that requires zero coordination with a running indexer.

Usage:
    uv run python scripts/migrate-tantivy.py            # all indexed repos
    uv run python scripts/migrate-tantivy.py myrepo     # one specific repo

Exit codes:
    0  success (every repo migrated, or no repos to migrate)
    1  any per-repo failure (other repos still attempted; see stderr)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running both as ``python scripts/migrate-tantivy.py`` (cwd repo
# root) and ``uv run scripts/migrate-tantivy.py`` (uv adds the project
# root to ``sys.path`` automatically).
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import settings, slugify_repo  # noqa: E402
from app.services.tantivy_index import TantivyIndex  # noqa: E402


_SYMBOL_CYPHER = (
    "MATCH (m:Module)-[:DEFINES]->(n:Function) "
    "RETURN n.qualified_name AS qn, n.start_line AS sl, "
    "n.end_line AS el, m.path AS p, n.docstring AS doc, "
    "'Function' AS kind "
    "UNION ALL "
    "MATCH (m:Module)-[:DEFINES]->(:Class)-[:DEFINES_METHOD]->(n:Method) "
    "RETURN n.qualified_name AS qn, n.start_line AS sl, "
    "n.end_line AS el, m.path AS p, n.docstring AS doc, "
    "'Method' AS kind"
)


def migrate_repo(slug: str, db_path: Path) -> int:
    """Migrate one repo's symbols into its Tantivy index. Return docs added."""
    import real_ladybug as lb  # type: ignore[import-untyped]

    db = lb.Database(str(db_path), read_only=True)
    conn = lb.Connection(db)
    added = 0
    try:
        res = conn.execute(_SYMBOL_CYPHER)
        cols = res.get_column_names()
        with TantivyIndex(settings.LADYBUG_DB_DIR, slug) as idx:
            while res.has_next():
                row = dict(zip(cols, res.get_next()))
                qn = row.get("qn") or ""
                if not qn:
                    continue
                content_parts = [str(qn)]
                doc = row.get("doc")
                if isinstance(doc, str) and doc:
                    content_parts.append(doc)
                idx.add(
                    symbol_qname=str(qn),
                    file_path=str(row.get("p") or ""),
                    symbol_kind=str(row.get("kind") or "Function"),
                    content=" ".join(content_parts),
                    start_line=int(row.get("sl") or 0),
                    end_line=int(row.get("el") or 0),
                    repo=slug,
                )
                added += 1
            idx.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass
        try:
            db.close()
        except Exception:
            pass
    return added


def main(argv: list[str]) -> int:
    db_dir = Path(settings.LADYBUG_DB_DIR)
    if not db_dir.is_dir():
        print(f"[migrate-tantivy] no LadybugDB dir at {db_dir}", file=sys.stderr)
        return 0

    if len(argv) > 1:
        # Specific repo by name (gets slugified to match the .db filename).
        targets = [(slugify_repo(argv[1]), db_dir / f"{slugify_repo(argv[1])}.db")]
    else:
        targets = [
            (p.stem, p)
            for p in sorted(db_dir.glob("*.db"))
            if p.stem != "graph"  # skip legacy combined DB
        ]

    if not targets:
        print("[migrate-tantivy] no per-repo DBs to migrate")
        return 0

    failures = 0
    for slug, db_path in targets:
        if not db_path.exists():
            print(f"[migrate-tantivy] skip {slug} — no DB at {db_path}", file=sys.stderr)
            continue
        try:
            n = migrate_repo(slug, db_path)
            print(f"[migrate-tantivy] {slug}: {n} symbols indexed")
        except Exception as exc:
            print(f"[migrate-tantivy] {slug}: FAILED ({exc})", file=sys.stderr)
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
