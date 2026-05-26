"""Embed-pass driver — runs as a subprocess from ``app.routers.index``.

Historically this body lived as a ~270-line f-string in
``app/routers/index.py`` and was shelled out via ``python -c``.  Every
brace had to be doubled and every regex backslash hand-escaped, which
made it impossible to unit-test the skip filter or hash composition.

BUC-1601 promoted that body to this real module.  The CLI surface
(``--repo-db-path``, ``--vec-db-path``, ``--repo-path``) replaces the
previous ``{repr(...)}`` interpolation; everything else is a verbatim
port of the f-string body so the persistence + skip + cost-cap
contracts remain byte-identical with what was running before.

Invocation (from the parent worker)::

    python -m app.scripts.embed_driver \\
        --repo-db-path /path/to/repo.lb \\
        --vec-db-path /path/to/repo.duck \\
        --repo-path   /path/to/checkout

The parent worker pipes both stdout and stderr to ``/tmp/cis_embed_<id>.log``
and parses the trailing ``Embedded ...`` summary line plus the new
``RECONCILE ...`` line (BUC-1601 Fix A) to populate the embed job
record.

Importable helpers (kept module-level so tests do not need a
subprocess):

* ``should_skip_embed(path)`` — BUC-1519 skip-filter predicate.
* ``compute_content_hash(text)`` — BUC-1518 SHA-1 fingerprint used to
  short-circuit re-embedding when nothing changed.
* ``compose_function_method_embed_text(...)`` — assembles the embed
  input string for a Function/Method symbol.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import os
import re
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Skip filter — BUC-1519
# ---------------------------------------------------------------------------
#
# Symbols whose source files match these patterns add nothing to
# semantic search but cost SageMaker time and money.  Test files dominate
# this list (often 25-35% of repos); generated code is usually 5-10%.
#
# Patterns are kept module-level so the unit tests in
# ``tests/test_embed_driver.py`` can import and exercise them directly.
SKIP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p) for p in [
        r"(^|/)tests?/",                           # /tests/ or /test/ dir
        r"_test\.(py|go|rs|js|ts|tsx)$",           # foo_test.go etc.
        r"\.test\.(js|ts|tsx|jsx)$",               # foo.test.ts
        r"\.spec\.(js|ts|tsx|jsx)$",               # foo.spec.ts
        r"(^|/)__tests__/",                        # JS/TS __tests__/
        r"(^|/)test_[^/]+\.py$",                   # test_foo.py
        r"(^|/)conftest\.py$",                     # pytest fixtures
        r"\.pb\.(go|py|cc|h)$",                    # protobuf-generated
        r"_pb2\.py$",                              # protobuf-generated python
        r"_pb2_grpc\.py$",                         # grpc-generated
        r"(^|/)generated/",                        # */generated/* dirs
        r"_generated\.(go|py|ts|tsx)$",
        r"(^|/)vendor/",                           # vendored deps
        r"(^|/)node_modules/",                     # JS deps
        r"(^|/)\.venv/",                           # python venv
        r"(^|/)dist/",                             # build outputs
        r"(^|/)build/",                            # build outputs
    ]
]


def should_skip_embed(file_path: str) -> bool:
    """True for paths whose symbols are not worth embedding.

    Test / generated / vendored sources rarely contribute to semantic
    search recall.  The cost (SageMaker tokens) of embedding them is
    real, so we filter on the relative path before queueing.

    Args:
        file_path: Repo-relative path (forward-slash separated).

    Returns:
        True when at least one pattern in :data:`SKIP_PATTERNS` matches
        ``file_path``; False otherwise.
    """
    return any(p.search(file_path) for p in SKIP_PATTERNS)


def compute_content_hash(text: str) -> str:
    """Stable SHA-1 fingerprint of an embed input string.

    Used by BUC-1518 incremental-embedding: if the stored hash on the
    ``.duck`` row matches the freshly computed one, the symbol is
    unchanged since the last index and we skip the SageMaker call
    entirely.

    SHA-1 is fine here — we are not using it cryptographically, only as
    a content fingerprint.

    Args:
        text: The fully-assembled embed input (header + source).

    Returns:
        Lowercase hex SHA-1 digest of the UTF-8 encoded ``text``.
    """
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def compose_function_method_embed_text(
    *,
    stype: str,
    qname: str,
    callers: int,
    docstring: str,
    src: str,
    format_docstring: Any,
) -> str:
    """Build the per-symbol embed input for a Function or Method.

    Mirrors the loop in :func:`_run_function_method_pass` so it can be
    unit-tested without spinning up SageMaker.

    Args:
        stype: Either ``"Function"`` or ``"Method"`` — appears verbatim
            in the header comment.
        qname: Fully qualified symbol name (``pkg.mod.Cls.fn``).
        callers: Number of inbound CALLS edges in the graph; appended
            to the header so caller-counts can bias rerank.
        docstring: Raw docstring extracted from the graph; passed through
            ``format_docstring`` for layout normalisation.
        src: The actual source range (``start_line``..``end_line``)
            already concatenated with newlines.
        format_docstring: Injected callable (normally
            ``codebase_rag.storage.docstring_format.format_docstring``).
            Taken as a parameter so the unit tests can stub it out.

    Returns:
        Multi-line string used both as the embed input and the SHA-1
        content-hash input.
    """
    header_parts = [f"# {stype}: {qname}"]
    mod_path = ".".join(qname.split(".")[:-1])
    if mod_path:
        header_parts.append(f"# Module: {mod_path}")
    if callers > 0:
        header_parts.append(f"# Callers: {callers}")
    header_parts.append("# ---")
    formatted_doc = format_docstring(docstring)
    if formatted_doc:
        header_parts.append(formatted_doc)
    header_parts.append(src)
    return "\n".join(header_parts)


# ---------------------------------------------------------------------------
# Embedder resolution — LE-151.
#
# CRITICAL recall fix: the ingest embedding pass MUST produce vectors from
# the SAME model (with the same post-processing) as the query path in
# ``app/routers/search.py::_embed_query``.  Historically this driver called
# ``codebase_rag.embedder.embed_code_batch`` (CodeRankEmbed) while the query
# side resolved the configured ``EMBEDDER_BACKEND`` (prod = SageMaker E5).
# Stored passage vectors and query vectors therefore lived in incompatible
# spaces → cosine ≈ 0.09 for correct matches → recall@5 ≈ 19%.
#
# ``resolve_batch_embedder`` returns a callable with the SAME signature as
# ``embed_code_batch`` (``list[str] -> list[list[float]]``) but routes
# through the configured backend, mirroring the query side's fallback
# chain exactly: configured backend (raw text, no prefix) → LM Studio dev
# (asymmetric document prefix) → in-process torch (``embed_code_batch``).
# Symmetry is what matters: the configured backend (the PRIMARY) is used by
# both ingest and query, so with EMBEDDER_BACKEND=sagemaker both become E5.
# ---------------------------------------------------------------------------

# LM Studio dev-fallback prefix for the *passage* (document) side. The query
# side passes ``"search_query: "``; this is the asymmetric counterpart for
# indexed code so the dev-only LM Studio path stays symmetric. Only used when
# the configured backend is unavailable AND LM Studio is reachable.
_LM_STUDIO_DOC_PREFIX = "search_document: "


def partition_batch_result(
    meta: list[tuple[str, str, int, int, str, str]],
    embeddings: list[list[float]] | None,
    error: BaseException | None,
) -> tuple[list[tuple[tuple[str, str, int, int, str, str], list[float]]], int]:
    """Classify one batch's embed outcome into (insertable_rows, failed_count).

    LE-151b fail-loud contract — the single source of truth for deciding
    whether a batch's results are safe to persist:

    * ``error is not None`` → the embed call raised even after the
      embedder's internal retry/backoff.  The ENTIRE batch is counted as
      failed and NOTHING is persisted for it (no fabricated / empty
      vectors).
    * ``len(embeddings) != len(meta)`` → a corrupt/truncated result.
      Treated as a whole-batch failure rather than zipping a partial set
      into the store (which would silently drop symbols).
    * Otherwise → every (meta, vector) pair is returned for insertion and
      the failed count is 0.

    Args:
        meta: Per-symbol metadata tuples for the batch.
        embeddings: The embedder's output (one vector per ``meta`` entry),
            or None when ``error`` is set.
        error: The exception raised by the batch embed call, or None on
            success.

    Returns:
        ``(pairs, failed)`` where ``pairs`` is the list of
        ``(meta_tuple, vector)`` safe to insert and ``failed`` is the
        number of symbols that could NOT be embedded.
    """
    if error is not None:
        return ([], len(meta))
    if embeddings is None or len(embeddings) != len(meta):
        return ([], len(meta))
    return ([(_m, _e) for _m, _e in zip(meta, embeddings)], 0)


def resolve_ingest_concurrency() -> int:
    """Resolve the number of concurrent SageMaker batch invocations.

    LE-151b: the default is **1** (sequential batches).  A bulk re-embed at
    the previous default of 2 fanned ~8 simultaneous invocations into a
    small Serverless Inference endpoint and OOM'd the model worker
    (``InternalServerException: Worker died.`` → whole job crashed).
    Serverless endpoints scale workers slowly with a tiny per-worker memory
    ceiling, so the reliable default for a hosted bulk embed is to
    serialise.  Throughput is recovered by the per-batch backoff+retry in
    :meth:`app.embedders.sagemaker.SageMakerEmbedder._invoke_with_retry`.

    Operators with a provisioned (non-serverless) endpoint that can absorb
    parallelism can raise this via ``SAGEMAKER_EMBED_CONCURRENCY``.  Invalid,
    missing, or non-positive values fall back to 1.

    Returns:
        Concurrency >= 1.
    """
    try:
        value = int(os.environ.get("SAGEMAKER_EMBED_CONCURRENCY") or "1")
    except (TypeError, ValueError):
        return 1
    return value if value >= 1 else 1


def resolve_batch_embedder() -> Any:
    """Return a ``list[str] -> list[list[float]]`` batch embedder callable.

    Mirrors the query-side provider priority in
    ``app/routers/search.py::_embed_query`` so ingest and query resolve to
    the SAME model:

    1. **Configured backend (PRIMARY).** ``EMBEDDER_BACKEND`` via
       ``app.embedders`` (prod = SageMaker E5). The backend exposes an async
       batch ``embed(texts)`` which we run on a fresh event loop. No
       asymmetric prefix — matching ``_embed_query`` which sends raw text.
    2. **LM Studio (dev fallback).** Per-text ``lm_studio.embed`` with the
       passage prefix (counterpart to the query side's ``search_query: ``).
    3. **In-process torch (last resort).** ``codebase_rag.embedder
       .embed_code_batch`` — the legacy CodeRankEmbed path, retained ONLY as
       the final fallback so a fully-offline install still embeds.

    Returns:
        A callable accepting ``list[str]`` and returning
        ``list[list[float]]`` (one vector per input, order-preserving).

    Raises:
        RuntimeError: when no provider is available (no configured backend,
            no LM Studio, and ``embed_code_batch`` import fails).
    """
    # 1. Configured backend (matches query PRIMARY).
    try:
        from app.embedders.sync_bridge import get_embedder_or_none
        backend = get_embedder_or_none()
    except Exception:  # noqa: BLE001 — import/config failure is non-fatal here
        backend = None

    if backend is not None:
        import asyncio

        def _embed_via_backend(texts: list[str]) -> list[list[float]]:
            # Async batch ``embed`` run on a fresh loop — this driver is a
            # subprocess with no live asyncio context, so ``asyncio.run`` is
            # safe (same pattern as ``embed_text_sync``). Raw text, no
            # prefix — symmetric with ``_embed_query``.
            return asyncio.run(backend.embed(list(texts)))

        print(f"embedder: configured backend '{backend.name}'", flush=True)
        return _embed_via_backend

    # 2. LM Studio dev fallback (matches query secondary).
    try:
        from app.services import lm_studio
        if lm_studio.can_embed():
            def _embed_via_lm_studio(texts: list[str]) -> list[list[float]]:
                out: list[list[float]] = []
                for _t in texts:
                    _v = lm_studio.embed(_t, prefix=_LM_STUDIO_DOC_PREFIX)
                    if _v is None:
                        raise RuntimeError(
                            "lm_studio.embed returned None during ingest"
                        )
                    out.append(_v)
                return out

            print("embedder: LM Studio dev fallback", flush=True)
            return _embed_via_lm_studio
    except Exception:  # noqa: BLE001 — LM Studio probing is best-effort
        pass

    # 3. In-process torch last resort (legacy CodeRankEmbed).
    from codebase_rag.embedder import embed_code_batch
    print("embedder: in-process torch (embed_code_batch) last resort", flush=True)
    return embed_code_batch


# ---------------------------------------------------------------------------
# Driver entry-point.  Everything below this point talks to LadybugDB,
# DuckDB and SageMaker and is exercised only by the live indexer (not by
# unit tests).
# ---------------------------------------------------------------------------


def _alarm_handler(signum: int, frame: Any) -> None:
    """SIGALRM handler used as a hard watchdog on each batch embed call.

    150s is generous: batch=16 with cold start is ~30s, sustained is
    ~16s; 150s catches genuinely stuck calls fast.
    """
    raise TimeoutError("embed_code_batch exceeded 150s — single call wedged")


def _read_source_range(
    abs_path: str,
    start_line: int,
    end_line: int,
    *,
    log_warn: Any,
    drop_counter: dict[str, int],
) -> str | None:
    """Read a slice of a source file, accounting WARN on failure.

    BUC-1601 Fix A: previously the read happened inside a bare ``except``
    that silently dropped any failure.  Now we log a WARN line with the
    path and reason, and increment the ``dropped_unreadable`` counter so
    the parent process can reconcile the delta.

    Args:
        abs_path: Absolute filesystem path to read.
        start_line: 1-indexed inclusive.
        end_line: 1-indexed inclusive.
        log_warn: Callable accepting a single message string.
        drop_counter: Mutable counter dict; ``dropped_unreadable`` is
            bumped on failure.

    Returns:
        The newline-joined slice, or ``None`` when the read failed (or
        the slice was empty after stripping).
    """
    try:
        lines = Path(abs_path).read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except Exception as exc:  # noqa: BLE001
        log_warn(
            f"embed_driver.read_failed path={abs_path} "
            f"reason={type(exc).__name__}:{exc}"
        )
        drop_counter["dropped_unreadable"] = drop_counter.get(
            "dropped_unreadable", 0
        ) + 1
        return None
    src = "\n".join(lines[max(0, int(start_line) - 1):int(end_line)])
    if not src.strip():
        return None
    return src


def _extract_module_metadata(_path: str, _content: str) -> tuple[str, list[str]] | None:
    """Stdlib-AST based ``__init__.py`` metadata extraction.

    Returns ``(docstring, public_names)`` or None when the file is not
    Python or the parse fails.
    """
    if not _path.endswith(".py"):
        return None
    try:
        _tree = ast.parse(_content)
    except (SyntaxError, ValueError):
        return None
    _doc = ast.get_docstring(_tree) or ""
    _all: list[str] | None = None
    for _node in _tree.body:
        if isinstance(_node, ast.Assign):
            for _t in _node.targets:
                if isinstance(_t, ast.Name) and _t.id == "__all__":
                    if isinstance(_node.value, (ast.List, ast.Tuple, ast.Set)):
                        _names: list[str] = []
                        for _elt in _node.value.elts:
                            if isinstance(_elt, ast.Constant) and isinstance(
                                _elt.value, str
                            ):
                                _names.append(_elt.value)
                        _all = _names
                    break
            if _all is not None:
                break
    if _all is None:
        _all = []
        for _node in _tree.body:
            if isinstance(_node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if not _node.name.startswith("_"):
                    _all.append(_node.name)
    return (_doc, _all)


def main(argv: list[str] | None = None) -> int:
    """Driver entry-point.

    Args:
        argv: Optional argv override (used by tests / direct calls).
            When None, ``sys.argv[1:]`` is parsed.

    Returns:
        Process exit code: 0 on success.  Non-zero exit codes propagate
        from unhandled exceptions raised below — same as the previous
        ``python -c`` driver.
    """
    parser = argparse.ArgumentParser(
        prog="embed_driver",
        description=(
            "Run the embedding pass against an already-indexed repo. "
            "Reads Function/Method/Class/Module/File metadata from "
            "LadybugDB and writes 768-dim vectors to the per-repo "
            ".duck file."
        ),
    )
    parser.add_argument(
        "--repo-db-path",
        required=True,
        help="Path to the LadybugDB graph file (.lb) for the repo.",
    )
    parser.add_argument(
        "--vec-db-path",
        required=True,
        help="Path to the DuckDB vector store file (.duck) for the repo.",
    )
    parser.add_argument(
        "--repo-path",
        default="",
        help=(
            "Absolute path to the on-disk checkout.  Used to resolve "
            "relative Module.path values to absolute paths for "
            "file IO. Optional — when omitted, the relative paths must "
            "already be absolute on the rows."
        ),
    )
    args = parser.parse_args(argv)

    # Imports of code-graph-rag + real-ladybug happen here (not at module
    # top) so that ``import app.scripts.embed_driver`` from a unit test
    # does not require the full embedding stack to be installed.
    import real_ladybug as lb
    from codebase_rag.storage.vector_store import (
        EmbeddingRow,
        bulk_insert,
        open_or_create,
        read_content_hashes,
    )
    from codebase_rag.storage.docstring_format import format_docstring

    signal.signal(signal.SIGALRM, _alarm_handler)

    # LE-151 — resolve the batch embedder ONCE up front so every flush uses
    # the same model the query side uses. Same callable contract as the old
    # ``embed_code_batch`` (list[str] -> list[list[float]]); the SIGALRM
    # watchdog + batch sizing + bookkeeping below are unchanged.
    _embed_batch = resolve_batch_embedder()

    # BUC-1517 / LE-151b: number of concurrent SageMaker invocations.
    # Default is now 1 (sequential batches) — see
    # ``resolve_ingest_concurrency`` for the full rationale.  A bulk
    # re-embed at the previous default of 2 fanned ~8 simultaneous
    # invocations into a small Serverless Inference endpoint and OOM'd the
    # model worker ("InternalServerException: Worker died." → whole job
    # crashed).  Throughput is recovered by the per-batch backoff+retry in
    # ``SageMakerEmbedder._invoke_with_retry``.
    _CONCURRENCY = resolve_ingest_concurrency()

    repo_db_path = args.repo_db_path
    vec_db_path = args.vec_db_path
    _root_path = args.repo_path

    def _warn(msg: str) -> None:
        # WARN lines are tail-parsed by the parent on failure.  Keep them
        # cheap to grep: prefix + space-separated key=value pairs.
        print(f"WARN {msg}", flush=True)

    # ------------------------------------------------------------------
    # Reconcile counters (BUC-1601 Fix A).
    #
    # ``dropped_unreadable`` increments every time we hit an OSError or
    # UnicodeError trying to slurp a source file off disk; we used to
    # silently swallow those.  Counters are kept in a dict so
    # ``_read_source_range`` can mutate them by name.
    # ------------------------------------------------------------------
    _drops: dict[str, int] = {"dropped_unreadable": 0}

    # ------------------------------------------------------------------
    # 0. Open LadybugDB read-only and pull every Function / Method row.
    # ------------------------------------------------------------------
    #
    # ``read_only=True`` is critical here: when /index/embed is invoked
    # while uvicorn is also live, the parent process already holds the DB
    # file open via the count-query block above (and FastAPI tooling can
    # also keep handles around).  LadybugDB takes a write lock by default
    # (``IO exception: Could not set lock on file: …``) and the embed
    # subprocess fails with exit 1 before the user ever sees progress.
    # Read-only opens skip the write lock and multiple readers can coexist
    # with the live indexer — exactly what we want here, since the embed
    # pass only QUERIES the graph and writes vectors to a separate .duck
    # file.
    _db = lb.Database(repo_db_path, read_only=True)
    _conn_lb = lb.Connection(_db)

    _cypher = """
MATCH (m:Module)-[:DEFINES]->(n:Function)
OPTIONAL MATCH (_caller)-[:CALLS]->(n)
WITH m, n, count(_caller) AS caller_count
RETURN n.qualified_name AS qualified_name,
       n.start_line     AS start_line,
       n.end_line       AS end_line,
       m.path           AS rel_path,
       n.docstring      AS docstring,
       'Function'       AS symbol_type,
       caller_count     AS caller_count
UNION ALL
MATCH (m:Module)-[:DEFINES]->(_c:Class)-[:DEFINES_METHOD]->(n:Method)
OPTIONAL MATCH (_caller)-[:CALLS]->(n)
WITH m, n, count(_caller) AS caller_count
RETURN n.qualified_name AS qualified_name,
       n.start_line     AS start_line,
       n.end_line       AS end_line,
       m.path           AS rel_path,
       n.docstring      AS docstring,
       'Method'         AS symbol_type,
       caller_count     AS caller_count
"""
    _result = _conn_lb.execute(_cypher)
    _col_names = _result.get_column_names()
    _rows: list[dict[str, Any]] = []
    while _result.has_next():
        _raw = _result.get_next()
        _rows.append(dict(zip(_col_names, _raw)))

    _conn_lb.close()
    del _conn_lb, _db

    # ------------------------------------------------------------------
    # 1. Open the DuckDB vector store and pre-load known content_hashes.
    # ------------------------------------------------------------------
    _vec_conn = open_or_create(vec_db_path)

    # BUC-1518 C2 — incremental embedding. Pre-load every existing
    # content_hash from the .duck file. For each candidate symbol, hash its
    # source range and skip the SageMaker call entirely if the hash matches
    # the stored one (== content unchanged since last index).  For typical
    # commits touching a few files, this skips 95-99% of the work.
    _existing_hashes = read_content_hashes(_vec_conn)
    print(f"existing content_hashes: {len(_existing_hashes)}", flush=True)

    # The shapes the loop accumulates into:
    #   _batch_texts[i] aligns with _batch_meta[i]; we flush in lock-step.
    _BATCH = 50
    _embedded_count = 0
    _skipped_unchanged = 0
    _skipped_filtered = 0
    # LE-151b: symbols whose batch embedding failed after all SageMaker
    # retries were exhausted (or which raised a hard error). These are NOT
    # written to the vector store — failing loud is the whole point, so we
    # count them and surface the total in RECONCILE + a non-zero exit so a
    # partial embed can never masquerade as success.
    _failed_count = 0
    _batch_texts: list[str] = []
    _batch_meta: list[tuple[str, str, int, int, str, str]] = []
    _pending_batches: list[
        tuple[list[str], list[tuple[str, str, int, int, str, str]]]
    ] = []

    def _flush_pending(pool: ThreadPoolExecutor) -> None:
        """Dispatch every queued batch to SageMaker in parallel + insert.

        Submits each pending outer batch to a thread, gathers the
        results in submission order, then bulk-inserts the resulting
        EmbeddingRow list.  Bumps the live ``_embedded_count`` and
        prints a PROGRESS line the parent log-tailer parses.

        LE-151b: if a batch's embed call raises (transient SageMaker
        failure that survived all retries, or a hard error), we do NOT
        abort the entire job and we do NOT write empty/zero vectors for
        that batch.  Instead the batch's symbol count is added to
        ``_failed_count`` and a loud ``WARN embed_batch.failed`` line is
        emitted.  ``main`` returns a non-zero exit code when
        ``_failed_count > 0`` so the parent never records the job as a
        clean success.  Successfully-embedded batches in the same flush
        are still persisted.
        """
        nonlocal _embedded_count, _failed_count
        if not _pending_batches:
            return
        # +30s margin per concurrent batch
        signal.alarm(150 + 30 * len(_pending_batches))
        try:
            futures = [
                pool.submit(_embed_batch, texts)
                for texts, _meta in _pending_batches
            ]
            all_inserts = []
            for fut, (_texts, meta) in zip(futures, _pending_batches):
                _err: BaseException | None = None
                _embs: list[list[float]] | None = None
                try:
                    _embs = fut.result()
                except Exception as exc:  # noqa: BLE001 — surfaced, not swallowed
                    _err = exc
                _pairs, _failed = partition_batch_result(meta, _embs, _err)
                if _failed:
                    _failed_count += _failed
                    if _err is not None:
                        # Embedding this batch failed even after the
                        # embedder's internal retry/backoff.  Record the
                        # loss; do NOT fabricate vectors.
                        _warn(
                            f"embed_batch.failed symbols={_failed} "
                            f"reason={type(_err).__name__}:{_err}"
                        )
                    else:
                        _warn(
                            f"embed_batch.length_mismatch symbols={_failed} "
                            f"got={len(_embs) if _embs is not None else 'none'}"
                        )
                    continue
                for _m, _e in _pairs:
                    all_inserts.append(EmbeddingRow(
                        qualified_name=_m[0], embedding=_e,
                        file_path=_m[1], start_line=_m[2], end_line=_m[3],
                        symbol_type=_m[4], content_hash=_m[5],
                    ))
            if all_inserts:
                bulk_insert(_vec_conn, all_inserts)
                _embedded_count += len(all_inserts)
        finally:
            signal.alarm(0)
        _pending_batches.clear()
        print(
            f"PROGRESS embedded={_embedded_count} "
            f"skipped={_skipped_unchanged} filtered={_skipped_filtered} "
            f"failed={_failed_count}",
            flush=True,
        )

    _pool = ThreadPoolExecutor(max_workers=_CONCURRENCY)

    # ------------------------------------------------------------------
    # 2. Function / Method loop.
    # ------------------------------------------------------------------
    for _row in _rows:
        _qname = _row.get("qualified_name")
        _start = _row.get("start_line")
        _end = _row.get("end_line")
        _rel = _row.get("rel_path") or ""
        _doc = _row.get("docstring") or ""
        _stype = _row.get("symbol_type") or "Function"
        _callers = int(_row.get("caller_count") or 0)

        if not _qname or _start is None or _end is None or not _rel:
            continue

        # BUC-1519 — skip embedding for tests / generated / vendored files.
        # Test files are rarely the target of semantic search and dominate
        # the symbol count in many repos.  Filter on the relative path so
        # patterns like /tests/ or .test.ts work portably.
        if should_skip_embed(_rel):
            _skipped_filtered += 1
            continue

        _abs = _rel if Path(_rel).is_absolute() else (
            str(Path(_root_path) / _rel) if _root_path else _rel
        )

        _src = _read_source_range(
            _abs, int(_start), int(_end),
            log_warn=_warn, drop_counter=_drops,
        )
        if _src is None:
            continue

        _embed_text = compose_function_method_embed_text(
            stype=_stype,
            qname=_qname,
            callers=_callers,
            docstring=_doc,
            src=_src,
            format_docstring=format_docstring,
        )
        _content_hash = compute_content_hash(_embed_text)
        if _existing_hashes.get(_qname) == _content_hash:
            _skipped_unchanged += 1
            continue
        _batch_texts.append(_embed_text)
        _batch_meta.append(
            (_qname, _abs, int(_start), int(_end), _stype, _content_hash)
        )

        if len(_batch_texts) >= _BATCH:
            _pending_batches.append((_batch_texts, _batch_meta))
            _batch_texts = []
            _batch_meta = []
            if len(_pending_batches) >= _CONCURRENCY:
                _flush_pending(_pool)

    if _batch_texts:
        _pending_batches.append((_batch_texts, _batch_meta))
    if _pending_batches:
        _flush_pending(_pool)

    # ------------------------------------------------------------------
    # 3. Class summaries (deterministic — Phase 1.2).
    # ------------------------------------------------------------------
    _class_db = lb.Database(repo_db_path, read_only=True)
    _class_conn = lb.Connection(_class_db)
    _class_cypher = """
MATCH (m:Module)-[:DEFINES]->(c:Class)
OPTIONAL MATCH (c)-[:DEFINES_METHOD]->(meth:Method)
WITH m, c, collect(meth.name) AS method_names
RETURN c.qualified_name AS qualified_name,
       c.name           AS class_name,
       c.start_line     AS start_line,
       c.end_line       AS end_line,
       c.docstring      AS docstring,
       m.path           AS rel_path,
       m.qualified_name AS module_qname,
       method_names     AS method_names
"""
    _class_result = _class_conn.execute(_class_cypher)
    _class_cols = _class_result.get_column_names()
    _class_rows: list[dict[str, Any]] = []
    while _class_result.has_next():
        _raw = _class_result.get_next()
        _class_rows.append(dict(zip(_class_cols, _raw)))
    _class_conn.close()
    del _class_conn, _class_db
    print(f"class summary candidates: {len(_class_rows)}", flush=True)

    _class_skipped_filtered = 0
    _class_skipped_unchanged = 0
    _class_emitted = 0

    for _row in _class_rows:
        _qname = _row.get("qualified_name") or ""
        _cname = _row.get("class_name") or ""
        _start = _row.get("start_line")
        _end = _row.get("end_line")
        _rel = _row.get("rel_path") or ""
        _doc = _row.get("docstring") or ""
        _mod_qn = _row.get("module_qname") or ""
        _members = _row.get("method_names") or []

        if not _qname or not _rel:
            continue
        if should_skip_embed(_rel):
            _class_skipped_filtered += 1
            continue

        # Read the class signature line from disk.  The first line of the
        # class (the actual ``class Foo(Bar):`` line) is captured as the
        # signature; this is the start_line the parser stored.
        _abs = _rel if Path(_rel).is_absolute() else (
            str(Path(_root_path) / _rel) if _root_path else _rel
        )
        _signature = ""
        try:
            if _start is not None:
                _all_lines = Path(_abs).read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                _signature = _all_lines[max(0, int(_start) - 1)].rstrip()
        except Exception as exc:  # noqa: BLE001
            _warn(
                f"embed_driver.read_failed path={_abs} "
                f"reason={type(exc).__name__}:{exc}"
            )
            _drops["dropped_unreadable"] += 1
            _signature = f"class {_cname}:"

        # Filter junk member names (None, empty) — Cypher's collect() can
        # leave Nones when OPTIONAL MATCH yielded zero rows.
        _clean_members = [m for m in _members if m]

        _header = [f"# Class: {_qname}"]
        if _mod_qn:
            _header.append(f"# Module: {_mod_qn}")
        if _clean_members:
            _header.append(f"# Members: {', '.join(_clean_members)}")
        _header.append("# ---")
        if _signature:
            _header.append(_signature)
        if _doc:
            _header.append(_doc)
        _embed_text = "\n".join(_header).rstrip()

        # Summary-chunk qname convention: never collides with real qnames.
        _summary_qname = f"{_qname}::Class::summary"

        _content_hash = compute_content_hash(_embed_text)
        if _existing_hashes.get(_summary_qname) == _content_hash:
            _class_skipped_unchanged += 1
            continue

        _batch_texts.append(_embed_text)
        _batch_meta.append((
            _summary_qname, _abs,
            int(_start) if _start is not None else 0,
            int(_end) if _end is not None else 0,
            "Class", _content_hash,
        ))
        _class_emitted += 1

        if len(_batch_texts) >= _BATCH:
            _pending_batches.append((_batch_texts, _batch_meta))
            _batch_texts = []
            _batch_meta = []
            if len(_pending_batches) >= _CONCURRENCY:
                _flush_pending(_pool)

    if _batch_texts:
        _pending_batches.append((_batch_texts, _batch_meta))
    if _pending_batches:
        _flush_pending(_pool)

    print(
        f"Class summaries: emitted={_class_emitted} "
        f"skipped_unchanged={_class_skipped_unchanged} "
        f"filtered={_class_skipped_filtered}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 4. Module summaries (deterministic — Phase 1.2b).
    # ------------------------------------------------------------------
    _module_db = lb.Database(repo_db_path, read_only=True)
    _module_conn = lb.Connection(_module_db)
    _module_cypher = """
MATCH (m:Module)
RETURN m.qualified_name AS qualified_name, m.path AS rel_path
"""
    _module_result = _module_conn.execute(_module_cypher)
    _module_cols = _module_result.get_column_names()
    _module_rows: list[dict[str, Any]] = []
    while _module_result.has_next():
        _module_rows.append(
            dict(zip(_module_cols, _module_result.get_next()))
        )
    _module_conn.close()
    del _module_conn, _module_db

    _module_emitted = 0
    _module_skipped_unchanged = 0
    _module_skipped_filtered = 0

    for _row in _module_rows:
        _rel = _row.get("rel_path") or ""
        _qname = _row.get("qualified_name") or ""
        if not _rel or not _qname:
            continue
        if not _rel.endswith("__init__.py"):
            continue
        if should_skip_embed(_rel):
            _module_skipped_filtered += 1
            continue
        _abs = _rel if Path(_rel).is_absolute() else (
            str(Path(_root_path) / _rel) if _root_path else _rel
        )
        try:
            _content = Path(_abs).read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            _warn(
                f"embed_driver.read_failed path={_abs} "
                f"reason={type(exc).__name__}:{exc}"
            )
            _drops["dropped_unreadable"] += 1
            continue
        _meta = _extract_module_metadata(_rel, _content)
        if _meta is None:
            continue
        _doc, _public = _meta

        _lines = [f"# Module: {_qname}"]
        if _rel:
            _lines.append(f"# Path: {_rel}")
        if _public:
            _lines.append(f"# Public: {', '.join(_public)}")
        _lines.append("# ---")
        if _doc:
            _lines.append(_doc)
        _embed_text = "\n".join(_lines).rstrip()

        _summary_qname = f"{_qname}::Module::summary"
        _content_hash = compute_content_hash(_embed_text)
        if _existing_hashes.get(_summary_qname) == _content_hash:
            _module_skipped_unchanged += 1
            continue
        _batch_texts.append(_embed_text)
        _batch_meta.append(
            (_summary_qname, _abs, 0, 0, "Module", _content_hash)
        )
        _module_emitted += 1
        if len(_batch_texts) >= _BATCH:
            _pending_batches.append((_batch_texts, _batch_meta))
            _batch_texts = []
            _batch_meta = []
            if len(_pending_batches) >= _CONCURRENCY:
                _flush_pending(_pool)

    if _batch_texts:
        _pending_batches.append((_batch_texts, _batch_meta))
    if _pending_batches:
        _flush_pending(_pool)

    print(
        f"Module summaries: emitted={_module_emitted} "
        f"skipped_unchanged={_module_skipped_unchanged} "
        f"filtered={_module_skipped_filtered}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 5. File summaries (Manifest Haiku — Phase 1.2b, cost-capped).
    # ------------------------------------------------------------------
    _file_db = lb.Database(repo_db_path, read_only=True)
    _file_conn = lb.Connection(_file_db)
    _file_cypher = """
MATCH (m:Module)
RETURN m.qualified_name AS qualified_name, m.path AS rel_path
"""
    _file_result = _file_conn.execute(_file_cypher)
    _file_cols = _file_result.get_column_names()
    _file_rows: list[dict[str, Any]] = []
    while _file_result.has_next():
        _file_rows.append(
            dict(zip(_file_cols, _file_result.get_next()))
        )
    _file_conn.close()
    del _file_conn, _file_db

    # Inline the File summary helpers so the subprocess doesn't need to
    # import app.services (no sys.path setup in this driver).
    _FILE_SUMMARY_CONTENT_CAP = 8192
    _FILE_SUMMARY_COST_CAP = 1.50
    _HAIKU_IN_USD = 0.80 / 1_000_000
    _HAIKU_OUT_USD = 4.00 / 1_000_000
    _FILE_PROMPT_TEMPLATE = (
        "Summarize this file in <=180 tokens. Focus on:\n"
        "- What it does (one sentence)\n"
        "- Top-level exports\n"
        "- What it imports / depends on (if relevant)\n"
        "- Any non-obvious gotchas\n"
        "Avoid vague platitudes and filler.\n"
        "File: {path}\n"
        "Content: {content}"
    )

    def _build_file_prompt(_p: str, _c: str) -> str:
        _enc = _c.encode("utf-8", errors="replace")
        if len(_enc) > _FILE_SUMMARY_CONTENT_CAP:
            _enc = _enc[:_FILE_SUMMARY_CONTENT_CAP]
            _c = _enc.decode("utf-8", errors="ignore")
        return _FILE_PROMPT_TEMPLATE.format(path=_p, content=_c)

    def _summarize_file_via_manifest(
        _p: str, _c: str
    ) -> tuple[str, int, int] | None:
        import httpx as _hx
        _url = os.environ.get("MANIFEST_URL")
        _key = os.environ.get("MANIFEST_AGENT_KEY")
        if not _url or not _key:
            return None
        _prompt = _build_file_prompt(_p, _c)
        _body = {
            "model": os.environ.get("MANIFEST_FILE_SUMMARY_MODEL")
            or "claude-haiku-4-5",
            "messages": [{"role": "user", "content": _prompt}],
            "max_tokens": 220,
            "temperature": 0.2,
        }
        try:
            with _hx.Client(timeout=15.0) as _client:
                _resp = _client.post(
                    _url.rstrip("/") + "/v1/chat/completions",
                    json=_body,
                    headers={
                        "Authorization": f"Bearer {_key}",
                        "Content-Type": "application/json",
                    },
                )
            if _resp.status_code >= 400:
                print(
                    f"WARN manifest.summarize_http path={_p} "
                    f"status={_resp.status_code}",
                    flush=True,
                )
                return None
            _data = _resp.json()
        except Exception as _exc:  # noqa: BLE001
            print(
                f"WARN manifest.summarize_failed path={_p} err={_exc}",
                flush=True,
            )
            return None
        try:
            _summary = (_data["choices"][0]["message"]["content"] or "").strip()
        except Exception:  # noqa: BLE001
            return None
        if not _summary:
            return None
        _u = _data.get("usage") or {}
        return (
            _summary,
            int(_u.get("prompt_tokens") or 0),
            int(_u.get("completion_tokens") or 0),
        )

    _file_emitted = 0
    _file_skipped_filtered = 0
    _file_skipped_unchanged = 0
    _file_skipped_nosum = 0
    _cumulative_cost_usd = 0.0
    _cost_aborted = False

    for _row in _file_rows:
        _rel = _row.get("rel_path") or ""
        _qname = _row.get("qualified_name") or ""
        if not _rel or not _qname:
            continue
        if _rel.endswith("__init__.py"):
            # Already covered by the Module summary pass above.
            continue
        if should_skip_embed(_rel):
            _file_skipped_filtered += 1
            continue
        _abs = _rel if Path(_rel).is_absolute() else (
            str(Path(_root_path) / _rel) if _root_path else _rel
        )
        try:
            _content = Path(_abs).read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            _warn(
                f"embed_driver.read_failed path={_abs} "
                f"reason={type(exc).__name__}:{exc}"
            )
            _drops["dropped_unreadable"] += 1
            continue
        if not _content.strip():
            continue

        # Estimate cost upper bound BEFORE the call (verified pricing): a
        # single Haiku summary at ~600 in + 180 out is ≈ $0.0012.  Cap the
        # estimate at the worst plausible case to stay under-budget.
        _est_cost = 600 * _HAIKU_IN_USD + 220 * _HAIKU_OUT_USD
        if _cumulative_cost_usd + _est_cost > _FILE_SUMMARY_COST_CAP:
            if not _cost_aborted:
                print(
                    f"WARN file_summary.cost_cap_exceeded "
                    f"spent={_cumulative_cost_usd:.4f} "
                    f"cap={_FILE_SUMMARY_COST_CAP} — aborting File-summary pass",
                    flush=True,
                )
                _cost_aborted = True
            break

        _result = _summarize_file_via_manifest(_rel, _content)
        if _result is None:
            _file_skipped_nosum += 1
            continue
        _summary, _in_tok, _out_tok = _result
        _cumulative_cost_usd += (
            _in_tok * _HAIKU_IN_USD + _out_tok * _HAIKU_OUT_USD
        )

        _embed_text = (
            f"# File: {_qname}\n"
            f"# Path: {_rel}\n"
            f"# ---\n"
            f"{_summary}"
        )
        _summary_qname = f"{_qname}::File::summary"
        _content_hash = compute_content_hash(_embed_text)
        if _existing_hashes.get(_summary_qname) == _content_hash:
            _file_skipped_unchanged += 1
            continue
        _batch_texts.append(_embed_text)
        _batch_meta.append(
            (_summary_qname, _abs, 0, 0, "File", _content_hash)
        )
        _file_emitted += 1
        if len(_batch_texts) >= _BATCH:
            _pending_batches.append((_batch_texts, _batch_meta))
            _batch_texts = []
            _batch_meta = []
            if len(_pending_batches) >= _CONCURRENCY:
                _flush_pending(_pool)

    if _batch_texts:
        _pending_batches.append((_batch_texts, _batch_meta))
    if _pending_batches:
        _flush_pending(_pool)

    print(
        f"File summaries: emitted={_file_emitted} "
        f"skipped_unchanged={_file_skipped_unchanged} "
        f"filtered={_file_skipped_filtered} "
        f"no_summary={_file_skipped_nosum} "
        f"cost_usd={_cumulative_cost_usd:.4f} aborted={_cost_aborted}",
        flush=True,
    )

    _pool.shutdown(wait=True)
    _vec_conn.close()

    # ------------------------------------------------------------------
    # 6. Reconcile pass (BUC-1601 Fix A).
    #
    # Compare expected (rows we pulled from the graph) against actually
    # embedded + skipped, broken down by reason category.  Drift here is
    # almost always a bug — either the skip filter regressed or a file
    # read started silently failing.  We surface the delta in a single
    # ``RECONCILE`` line so /index/.../diff_metrics + the PR-body audit
    # script can pick it up without re-reading the whole subprocess log.
    # ------------------------------------------------------------------
    _expected_function_method = len(_rows)
    _function_method_accounted = (
        _embedded_count + _skipped_unchanged + _skipped_filtered
        + _drops["dropped_unreadable"]
    )
    # Note: empty-source-after-strip drops (the rare ``not _src.strip()``
    # case in ``_read_source_range``) are intentionally folded into the
    # ``unaccounted`` bucket — they are not a failure mode worth its own
    # counter, but they should still surface in the delta when material.
    _unaccounted = _expected_function_method - _function_method_accounted
    print(
        f"RECONCILE expected={_expected_function_method} "
        f"embedded={_embedded_count} "
        f"skipped_unchanged={_skipped_unchanged} "
        f"skipped_filtered={_skipped_filtered} "
        f"dropped_unreadable={_drops['dropped_unreadable']} "
        f"failed={_failed_count} "
        f"unaccounted={_unaccounted}",
        flush=True,
    )

    print(
        f"Embedded {_embedded_count} "
        f"(skipped {_skipped_unchanged} unchanged, "
        f"filtered {_skipped_filtered}, "
        f"failed {_failed_count})"
    )

    # LE-151b: fail loud. If ANY batch failed to embed (transient SageMaker
    # failure that survived retries, or a hard error), exit non-zero so the
    # parent worker records the embed job as failed and the operator can
    # re-run.  We do NOT print ``EMBED_DONE`` (the success sentinel) in that
    # case — a partial embed must never look like a clean success.
    if _failed_count > 0:
        print(
            f"EMBED_FAILED failed={_failed_count} embedded={_embedded_count}",
            flush=True,
        )
        return 1

    print("EMBED_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
