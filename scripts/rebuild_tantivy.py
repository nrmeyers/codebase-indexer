"""Rebuild per-repo Tantivy lexical indexes without a full re-index.

Wraps :func:`app.services.tantivy_index.rebuild_lexical_index` (graph +
source text) and then re-mirrors the markdown corpus (the rebuild wipes
the index dir, which would otherwise drop the markdown docs that
``_index_markdown_corpus`` added during the last full index run).

Run with the service STOPPED — LadybugDB is effectively single-process.

Usage:
    uv run python scripts/rebuild_tantivy.py SLUG=REPO_ROOT [SLUG=REPO_ROOT ...]

Example:
    uv run python scripts/rebuild_tantivy.py \
        navistone__TheForge=$HOME/dev/claude/TheForge \
        code-indexer-service=$HOME/dev/claude/code-indexer-service
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings, slugify_repo  # noqa: E402
from app.services.tantivy_index import TantivyIndex, rebuild_lexical_index  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("rebuild_tantivy")


def _mirror_markdown(repo_root: Path, slug: str) -> int:
    """Re-add markdown chunk docs (mirrors ``_index_markdown_corpus``'s
    tantivy pass — chunking only, no embedding; duck rows already exist)."""
    from app.services.markdown_indexer import (
        chunk_markdown_file,
        discover_markdown_files,
    )

    md_files = discover_markdown_files(repo_root)
    if not md_files:
        return 0
    added = 0
    idx = TantivyIndex(settings.LADYBUG_DB_DIR, slug)
    try:
        for path in md_files:
            try:
                rel = path.relative_to(repo_root).as_posix()
            except ValueError:
                rel = str(path)
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for chunk in chunk_markdown_file(
                repo_name=slug, rel_path=rel, content=content
            ):
                if idx.add(
                    symbol_qname=chunk.qualified_name,
                    file_path=chunk.file_path,
                    symbol_kind="MarkdownDoc",
                    content=f"{chunk.heading}\n{chunk.body}",
                    start_line=int(chunk.start_line),
                    end_line=int(chunk.end_line),
                    repo=slug,
                ):
                    added += 1
        idx.commit()
    finally:
        idx.close()
    return added


def main() -> int:
    pairs = []
    for arg in sys.argv[1:]:
        if "=" not in arg:
            logger.error("bad arg (want SLUG=REPO_ROOT): %s", arg)
            return 2
        slug, root = arg.split("=", 1)
        pairs.append((slugify_repo(slug), Path(root).expanduser().resolve()))
    if not pairs:
        logger.error("no repos given; see module docstring")
        return 2

    for slug, root in pairs:
        db_path = settings.db_path_for_repo(slug)
        if not Path(db_path).exists():
            logger.error("%s: no graph db at %s — skipping", slug, db_path)
            continue
        n = rebuild_lexical_index(db_path, root, slug, settings.LADYBUG_DB_DIR)
        md = _mirror_markdown(root, slug)
        logger.info("%s: %d symbol/file docs + %d markdown docs", slug, n, md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
