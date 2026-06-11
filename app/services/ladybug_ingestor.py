"""CI-4: LadybugDB ingestor — replaces MemgraphIngestor (DEV-1172).

Drop-in replacement for MemgraphIngestor. Same public interface:
    - __init__(db_path, batch_size, use_merge)
    - __enter__ / __exit__ (context manager)
    - ensure_node_batch / ensure_relationship_batch
    - flush_nodes / flush_relationships / flush_all
    - fetch_all / execute_write
    - clean_database / list_projects / delete_project
    - ensure_constraints (no-op: LadybugDB uses typed schema)
    - export_graph_to_dict

Key differences from MemgraphIngestor:
    - Embedded DB (file path instead of host:port)
    - Single-writer (no parallel ThreadPoolExecutor)
    - Schema migration runs on connect()
    - QueryResult uses has_next/get_next instead of cursor.fetchall()
    - MERGE semantics work natively via openCypher
"""
from __future__ import annotations

import threading
import types
from collections import defaultdict
from datetime import UTC, datetime

import real_ladybug as lb
from loguru import logger

from .ladybug_buffer_pool import resolve_buffer_pool_size

from codebase_rag import exceptions as ex
from codebase_rag import logs as ls
from codebase_rag.constants import (
    ERR_SUBSTR_ALREADY_EXISTS,
    ERR_SUBSTR_CONSTRAINT,
    KEY_FROM_VAL,
    KEY_NAME,
    KEY_PROJECT_NAME,
    KEY_PROPS,
    KEY_TO_VAL,
    NODE_UNIQUE_CONSTRAINTS,
    REL_TYPE_CALLS,
)
from codebase_rag.cypher_queries import (
    CYPHER_DELETE_ALL,
    CYPHER_DELETE_PROJECT,
    CYPHER_EXPORT_NODES,
    CYPHER_EXPORT_RELATIONSHIPS,
    CYPHER_LIST_PROJECTS,
    wrap_with_unwind,
)
from codebase_rag.types_defs import (
    BatchParams,
    BatchWrapper,
    GraphData,
    GraphMetadata,
    PropertyValue,
    RelBatchRow,
    ResultRow,
)
from app.services.ladybug_schema import migrate


#: Node labels that carry code *definitions* (Function / Method / Class /
#: Interface / Enum).  A healthy parse of a code repo MUST land thousands of
#: these.  When a non-empty batch of one of these labels flushes with ZERO
#: successful writes AND the failures were genuine runtime errors (not benign
#: "already exists" idempotency), the write was silently dropped — the
#: deterministic "369-node truncation" footgun: Kùzu's mmap-backed buffer pool
#: cannot back its dirty pages under host memory pressure (the documented
#: ``ladybug_buffer_pool`` failure mode, but at write time rather than open
#: time), so the COPY/SET no-ops while ``execute()`` raises a binder/IO error
#: that the per-node try/except previously swallowed to DEBUG.  Catching it at
#: the flush that dropped — instead of 9.5 min later in the post-job guard —
#: lets the caller fail loud + retry against the exact failing batch.
_DEFINITION_NODE_LABELS: frozenset[str] = frozenset(
    {"Function", "Method", "Class", "Interface", "Enum"}
)


class DefinitionFlushError(RuntimeError):
    """Raised when a definition-node batch flushed with zero successful writes.

    Signals a silent write-drop (every Function/Method/Class/Interface/Enum in
    a non-empty batch failed with a real runtime error rather than a benign
    idempotency hit).  Raising here surfaces the drop at the exact flush that
    lost the data so the indexer can fail loud and the operator can retry,
    rather than reporting ``status=done`` over a structural-only skeleton.
    """


class RelationshipFlushError(RuntimeError):
    """Raised when a behavioral relationship type lands ZERO edges at flush.

    The CALLS-drop counterpart to :class:`DefinitionFlushError`.  A graph whose
    definition nodes persist but whose behavioral edges (CALLS especially)
    silently vanish renders the knowledge-graph viewer disconnected ("nothing
    is connected").  Historically the per-row fallback in
    ``flush_relationships`` swallowed every inner failure without counting or
    raising, so a batch that hit Kùzu's ``unordered_map::at: key not found``
    on the UNWIND-MERGE path and then ALSO failed per-row produced 0 edges
    while the job reported success.  This exception fires when a behavioral
    rel type was *attempted* (rows were buffered) yet landed 0 successful
    writes, surfacing the loss at the exact flush instead of over a
    call-less graph 9 minutes later.
    """


# Behavioral relationship types whose total loss (attempted > 0, successful
# == 0) is a fail-loud condition.  CALLS is the load-bearing one for the
# knowledge-graph viewer; INHERITS / IMPLEMENTS / OVERRIDES are included
# because a 0-edge landing for any of them is the same silent-drop family.
# Structural rels (DEFINES / CONTAINS_* / BELONGS_TO) are intentionally
# EXCLUDED: their loss is already covered by the definition-node guard and
# a partial structural drop should not abort an otherwise-complete index.
_BEHAVIORAL_REL_TYPES: frozenset[str] = frozenset(
    {"CALLS", "INHERITS", "IMPLEMENTS", "OVERRIDES"}
)

# Substring of Kùzu's Binder exception when an endpoint's resolved label is
# not a legal FROM/TO pair for the rel table (e.g. the call resolver tagged a
# constructor call's callee as ``Class`` but CALLS only declares
# Function/Method/Module endpoints).  Such a row CANNOT legally become an
# edge — dropping it is correct, not a silent write-drop — so the fail-loud
# guard treats a 0-landing caused purely by these as benign.  Distinct from a
# genuine RUNTIME drop (Kùzu planner fault / endpoint not visible), which IS
# fatal for a behavioral type.
ERR_SUBSTR_SCHEMA_VIOLATION = "violates schema"


def _result_to_rows(result: lb.QueryResult) -> list[ResultRow]:
    """Convert a LadybugDB QueryResult to the same list[ResultRow] shape
    that MemgraphIngestor returned from its cursor-based API."""
    rows: list[ResultRow] = []
    col_names = result.get_column_names()
    while result.has_next():
        raw = result.get_next()
        rows.append(dict(zip(col_names, raw)))
    return rows


class LadybugIngestor:
    """LadybugDB-backed graph ingestor.

    Replaces MemgraphIngestor with an embedded, Docker-free, MIT-licensed
    graph store. Single-writer; batching is serialized (no thread pool).
    """

    __slots__ = (
        "_db_path",
        "_use_merge",
        "_rel_count",
        "_rel_groups",
        "_conn_lock",
        "batch_size",
        "_db",
        "conn",
        "node_buffer",
        "_node_count_total",
        "_rel_count_total",
    )

    @property
    def node_count(self) -> int:
        """Total nodes successfully flushed since __enter__."""
        return self._node_count_total

    @property
    def rel_count(self) -> int:
        """Total relationships successfully flushed since __enter__."""
        return self._rel_count_total

    def __init__(
        self,
        db_path: str,
        batch_size: int = 1000,
        use_merge: bool = True,
    ):
        if batch_size < 1:
            raise ValueError(ex.BATCH_SIZE)
        self._db_path = db_path
        self.batch_size = batch_size
        self._use_merge = use_merge
        self._conn_lock = threading.Lock()
        self._db: lb.Database | None = None
        self.conn: lb.Connection | None = None
        self.node_buffer: list[tuple[str, dict[str, PropertyValue]]] = []
        self._rel_count = 0
        self._rel_groups: defaultdict[
            tuple[str, str, str, str, str], list[RelBatchRow]
        ] = defaultdict(list)
        self._node_count_total = 0
        self._rel_count_total = 0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> LadybugIngestor:
        logger.info(f"LadybugDB connecting to: {self._db_path}")
        migrate(self._db_path)
        self._db = lb.Database(
            self._db_path, buffer_pool_size=resolve_buffer_pool_size()
        )
        self.conn = lb.Connection(self._db)
        logger.info("LadybugDB connected ✓")
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        try:
            if exc_type:
                logger.exception(ls.MG_EXCEPTION.format(error=exc_val))
                try:
                    self.flush_all()
                except Exception as flush_err:
                    logger.error(ls.MG_FLUSH_ERROR.format(error=flush_err))
            else:
                self.flush_all()
        finally:
            # Explicitly close handles so real_ladybug releases the OS file
            # lock immediately. Assigning None alone relies on CPython's GC
            # timing which is too slow when another process (the embedding
            # subprocess) expects to acquire the same lock within ms.
            try:
                if self.conn is not None and hasattr(self.conn, "close"):
                    self.conn.close()
            except Exception:
                pass
            try:
                if self._db is not None and hasattr(self._db, "close"):
                    self._db.close()
            except Exception:
                pass
            self.conn = None
            self._db = None
            import gc as _gc
            _gc.collect()
            logger.info("LadybugDB disconnected")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_query(
        self,
        query: str,
        params: dict[str, PropertyValue] | None = None,
    ) -> list[ResultRow]:
        if not self.conn:
            raise ConnectionError(ex.CONN)
        params = params or {}
        try:
            result = self.conn.execute(query, params)
            return _result_to_rows(result)
        except Exception as e:
            err_str = str(e).lower()
            if (
                ERR_SUBSTR_ALREADY_EXISTS not in err_str
                and ERR_SUBSTR_CONSTRAINT not in err_str
            ):
                # Log at DEBUG level — the caller decides whether the failure
                # is operationally significant. Many callers (per-row fallback
                # in ``flush_relationships``, idempotent CREATE in
                # ``flush_nodes``) treat a raised exception as a soft-success
                # and would otherwise produce misleading ERROR-level noise on
                # an otherwise clean indexing pass.  Real failures bubble up
                # to a caller-level ``logger.warning`` / ``logger.error``.
                logger.debug(ls.MG_CYPHER_ERROR.format(error=e))
                logger.debug(ls.MG_CYPHER_QUERY.format(query=query))
                logger.debug(ls.MG_CYPHER_PARAMS.format(params=params))
            raise

    def _execute_batch(
        self,
        query: str,
        params_list: list[BatchParams],
    ) -> list[ResultRow]:
        """Execute an UNWIND batch query against LadybugDB (single-writer)."""
        if not params_list:
            return []
        if not self.conn:
            raise ConnectionError(ex.CONN)
        try:
            result = self.conn.execute(
                wrap_with_unwind(query), BatchWrapper(batch=params_list)
            )
            return _result_to_rows(result)
        except Exception as e:
            err_str = str(e).lower()
            if ERR_SUBSTR_ALREADY_EXISTS not in err_str:
                # Log at DEBUG — ``flush_relationships`` catches this and
                # decides escalation (warning + per-row fallback).  Logging
                # at ERROR here floods operational logs for every benign
                # batch failure that the caller already handles.
                logger.debug(ls.MG_BATCH_ERROR.format(error=e))
                logger.debug(ls.MG_CYPHER_QUERY.format(query=query))
            raise

    # ------------------------------------------------------------------
    # Schema / constraints (LadybugDB uses typed DDL — no-op here)
    # ------------------------------------------------------------------

    def ensure_constraints(self) -> None:
        """No-op for LadybugDB: constraints are declared in the schema DDL
        via PRIMARY KEY. Migration already ran in __enter__."""
        logger.debug("LadybugIngestor.ensure_constraints: schema already migrated")

    # ------------------------------------------------------------------
    # Node batching
    # ------------------------------------------------------------------

    def ensure_node_batch(
        self, label: str, properties: dict[str, PropertyValue]
    ) -> None:
        self.node_buffer.append((label, properties))
        if len(self.node_buffer) >= self.batch_size:
            logger.debug(ls.MG_NODE_BUFFER_FLUSH, size=self.batch_size)
            self.flush_nodes()

    def flush_nodes(self) -> None:
        if not self.node_buffer:
            return

        buffer_size = len(self.node_buffer)
        nodes_by_label: defaultdict[str, list[dict[str, PropertyValue]]] = defaultdict(
            list
        )
        for label, props in self.node_buffer:
            nodes_by_label[label].append(props)

        flushed_total = 0
        skipped_total = 0
        # Definition labels whose batch had inputs but landed ZERO rows because
        # of genuine runtime errors (not benign idempotency).  Populated below;
        # a non-empty set means a silent write-drop that we raise on after the
        # loop so the buffer state + counters are still consistent.
        dropped_definition_labels: dict[str, tuple[int, str | None]] = {}

        for label, props_list in nodes_by_label.items():
            id_key = NODE_UNIQUE_CONSTRAINTS.get(label)
            if not id_key:
                logger.warning(ls.MG_NO_CONSTRAINT.format(label=label))
                skipped_total += len(props_list)
                continue

            skipped = 0
            flushed = 0
            # Track real (non-idempotent) failures separately from
            # missing-PK skips so we only trip the definition-drop guard on a
            # genuine write failure, not on a malformed-input skip.
            error_skips = 0
            last_error: str | None = None
            for props in props_list:
                if id_key not in props:
                    logger.warning(
                        ls.MG_MISSING_PROP.format(
                            label=label, key=id_key, prop_keys=list(props.keys())
                        )
                    )
                    skipped += 1
                    continue
                id_val = props[id_key]
                other_props = {k: v for k, v in props.items() if k != id_key}

                # LadybugDB forbids SET on vector-indexed columns once a vector
                # index exists — even ON CREATE SET triggers the same runtime
                # error ("Cannot set property vec … Try delete and then insert").
                # Strategy:
                #   • Nodes WITH vector properties → always CREATE (all props).
                #     Primary-key constraint violations are caught and treated as
                #     success (idempotent "already exists" semantics).
                #   • Nodes WITHOUT vector properties → MERGE + explicit SET
                #     (safe to update scalar columns in place).
                def _is_vector(v: object) -> bool:
                    return (
                        isinstance(v, list)
                        and len(v) > 10
                        and bool(v)
                        and isinstance(v[0], float)
                    )

                has_vector = any(_is_vector(v) for v in other_props.values())

                if has_vector:
                    # CREATE path: include all properties inline.
                    # If the node already exists the PRIMARY KEY constraint fires;
                    # we catch that and count it as a successful idempotent write.
                    all_kv = ", ".join(f"{k}: ${k}" for k in props)
                    query = f"CREATE (n:{label} {{{all_kv}}})"
                    params: dict[str, PropertyValue] = dict(props)
                elif self._use_merge:
                    # Pure-scalar MERGE: safe to SET all non-PK properties.
                    params = {id_key: id_val, **other_props}
                    if other_props:
                        set_clause = ", ".join(f"n.{k} = ${k}" for k in other_props)
                        query = f"MERGE (n:{label} {{{id_key}: ${id_key}}}) SET {set_clause}"
                    else:
                        query = f"MERGE (n:{label} {{{id_key}: ${id_key}}})"
                else:
                    params = dict(props)
                    if other_props:
                        all_kv = ", ".join(f"{k}: ${k}" for k in props)
                        query = f"CREATE (n:{label} {{{all_kv}}})"
                    else:
                        query = f"CREATE (n:{label} {{{id_key}: ${id_key}}})"

                try:
                    with self._conn_lock:
                        self._execute_query(query, params)
                    flushed += 1
                except Exception as e:
                    err_str = str(e).lower()
                    if ERR_SUBSTR_ALREADY_EXISTS in err_str or ERR_SUBSTR_CONSTRAINT in err_str:
                        flushed += 1  # idempotent — already exists is ok
                    else:
                        logger.error(ls.MG_LABEL_FLUSH_ERROR.format(label=label, error=e))
                        skipped += 1
                        error_skips += 1
                        last_error = str(e)

            skipped_total += skipped
            flushed_total += flushed

            # Silent-write-drop detection: a non-empty definition-label batch
            # that landed ZERO rows because every write raised a real runtime
            # error (not idempotency, not a missing-PK skip) is the "369-node
            # truncation" footgun.  Record it; we raise after the loop so the
            # buffer is cleared + counters are consistent first.
            if (
                label in _DEFINITION_NODE_LABELS
                and props_list
                and flushed == 0
                and error_skips > 0
            ):
                dropped_definition_labels[label] = (error_skips, last_error)

        self._node_count_total += flushed_total
        logger.info(
            ls.MG_NODES_FLUSHED.format(flushed=flushed_total, total=buffer_size)
        )
        if skipped_total:
            logger.info(ls.MG_NODES_SKIPPED.format(count=skipped_total))
        self.node_buffer.clear()

        # Fail loud on a silent definition write-drop. The buffer is already
        # cleared and counters logged above, so the ingestor state is
        # consistent before we raise. The caller (GraphUpdater.run via
        # _blocking_index) lets this propagate so the job is marked failed at
        # the EXACT flush that lost data — instead of the post-job guard
        # discovering 0 definitions 9.5 minutes later over a skeleton graph.
        if dropped_definition_labels:
            detail = ", ".join(
                f"{lbl} (0/{len(nodes_by_label[lbl])} written; "
                f"{cnt} runtime errors; last: {err})"
                for lbl, (cnt, err) in dropped_definition_labels.items()
            )
            logger.error(
                "Definition write-drop: %s. Every write in these batches "
                "failed with a runtime error (not idempotency) — the graph "
                "would be truncated to structural-only nodes. Failing the "
                "flush so the index job surfaces the loss immediately and the "
                "batch can be retried. Likely cause: Kùzu buffer-pool mmap "
                "cannot back its dirty pages under host memory pressure "
                "(lower co-tenant RAM use or KUZU_BUFFER_POOL_SIZE).",
                detail,
            )
            raise DefinitionFlushError(
                f"definition nodes silently dropped during flush: {detail}"
            )

    # ------------------------------------------------------------------
    # Relationship batching
    # ------------------------------------------------------------------

    def ensure_relationship_batch(
        self,
        from_spec: tuple[str, str, PropertyValue],
        rel_type: str,
        to_spec: tuple[str, str, PropertyValue],
        properties: dict[str, PropertyValue] | None = None,
    ) -> None:
        from_label, from_key, from_val = from_spec
        to_label, to_key, to_val = to_spec
        pattern = (from_label, from_key, rel_type, to_label, to_key)
        self._rel_groups[pattern].append(
            RelBatchRow(from_val=from_val, to_val=to_val, props=properties or {})
        )
        self._rel_count += 1
        if self._rel_count >= self.batch_size:
            # Flush pending nodes early so they accumulate in the DB, but do
            # NOT flush relationships here.  Relationships reference nodes from
            # files that haven't been parsed yet — flushing them mid-ingestion
            # produces "unordered_map::at: key not found" because the target
            # node doesn't exist in LadybugDB yet.  Relationships are flushed
            # in bulk by flush_all() after every node has been committed.
            logger.debug(ls.MG_NODE_BUFFER_FLUSH, size=self.batch_size)
            self.flush_nodes()

    def flush_relationships(self) -> None:
        """Flush buffered relationships via grouped UNWIND batches.

        Previously this method ran one Cypher query per relationship, which
        turned each batch_size=1000 flush into 1000 round-trips through the
        Python <-> LadybugDB boundary.  On a moderately relationship-heavy
        repo that's tens of thousands of queries and pegs a single CPU for
        tens of minutes before the ingest even reaches the 90% milestone.

        The optimised path groups rows by
        ``(pattern, sorted(property-keys))`` so every UNWIND batch is a
        single query against a single rel shape.  One LadybugDB call plans
        the MATCH-MATCH-MERGE across all rows in the batch instead of re-
        parsing the same statement 1000 times.  Empirical speedup on
        moderate C# repos is ~20-40x for the relationship flush phase.

        On batch failure we fall back to per-row execution so a single
        malformed row (e.g. wrong primary-key shape) doesn't poison the
        whole batch — we preserve the same "best effort, count successes"
        semantics the previous implementation had.
        """
        if not self._rel_count:
            return

        total_attempted = 0
        total_successful = 0
        # Per-behavioral-rel-type (attempted, successful) accumulated across
        # every pattern group in this flush.  A type that ends with
        # attempted > 0 and successful == 0 trips the fail-loud guard below.
        behavioral_totals: dict[str, tuple[int, int, int]] = {}

        for pattern, params_list in self._rel_groups.items():
            from_label, from_key, rel_type, to_label, to_key = pattern
            attempted = 0
            successful = 0
            # Per-group failure classification.  ``rel_runtime_dropped`` is the
            # silent-drop family (Kùzu planner fault / invisible endpoint) the
            # fail-loud guard treats as fatal for a behavioral type;
            # ``rel_schema_rejected`` is benign (resolver produced a callee
            # whose label is not a legal endpoint for the rel table).
            rel_runtime_dropped = 0
            rel_schema_rejected = 0

            # Group rows by the set of property keys so each UNWIND batch
            # has a consistent column layout (LadybugDB's UNWIND expects
            # every row in $batch to have the same shape).
            by_shape: dict[tuple[str, ...], list[dict[str, PropertyValue]]] = defaultdict(list)
            for row in params_list:
                rel_props = row.get(KEY_PROPS, {}) or {}
                shape = tuple(sorted(rel_props.keys()))
                entry: dict[str, PropertyValue] = {
                    "from_val": row[KEY_FROM_VAL],
                    "to_val": row[KEY_TO_VAL],
                }
                entry.update(rel_props)
                by_shape[shape].append(entry)

            for prop_keys, batch_rows in by_shape.items():
                # Bind the endpoints with ``MATCH (n:Label) WHERE n.key = …``
                # rather than an inline property pattern
                # ``MATCH (n:Label {key: row.val})``.  The inline-property form
                # inside an UNWIND-MERGE batch triggers a Kùzu rel-MERGE planner
                # path that throws ``unordered_map::at: key not found`` — and it
                # throws even when both endpoint nodes are fully committed and
                # visible (confirmed: a CHECKPOINT between flush_nodes and
                # flush_relationships does NOT fix it; switching the node bind to
                # ``MATCH … WHERE`` does, while preserving MERGE idempotency on
                # re-index).  This is the prevention that keeps CALLS off the
                # slow, silently-failing per-row fallback path.  Two sequential
                # MATCHes (not comma-separated) still avoid the secondary
                # hash-join footgun.
                match_clause = (
                    f"MATCH (a:{from_label})\n"
                    f"WHERE a.{from_key} = row.from_val\n"
                    f"MATCH (b:{to_label})\n"
                    f"WHERE b.{to_key} = row.to_val"
                )
                if self._use_merge:
                    if prop_keys:
                        prop_set = ", ".join(f"r.{k} = row.{k}" for k in prop_keys)
                        rel_clause = f"MERGE (a)-[r:{rel_type}]->(b) SET {prop_set}"
                    else:
                        rel_clause = f"MERGE (a)-[:{rel_type}]->(b)"
                elif prop_keys:
                    prop_inline = ", ".join(f"{k}: row.{k}" for k in prop_keys)
                    rel_clause = f"CREATE (a)-[:{rel_type} {{{prop_inline}}}]->(b)"
                else:
                    rel_clause = f"CREATE (a)-[:{rel_type}]->(b)"
                batch_query = f"{match_clause}\n{rel_clause}"

                batch_size_n = len(batch_rows)
                attempted += batch_size_n

                try:
                    with self._conn_lock:
                        self._execute_batch(batch_query, batch_rows)
                    successful += batch_size_n
                except Exception as e:
                    err_str = str(e).lower()
                    if (
                        ERR_SUBSTR_ALREADY_EXISTS in err_str
                        or ERR_SUBSTR_CONSTRAINT in err_str
                    ):
                        # Whole batch "already exists" is a soft-success —
                        # idempotent MERGE semantics say the graph is correct.
                        successful += batch_size_n
                    elif ERR_SUBSTR_SCHEMA_VIOLATION in err_str:
                        # The whole group's endpoint labels are not a legal
                        # FROM/TO pair for this rel table (resolver tagged the
                        # callee with a label the schema doesn't allow).  Every
                        # row here is schema-illegal — none can legally become
                        # an edge.  Record the benign reject (NOT a runtime
                        # drop) and skip the per-row retry, which would just
                        # re-raise the same Binder exception batch_size_n times.
                        rel_schema_rejected += batch_size_n
                        logger.warning(
                            ls.MG_REL_FLUSH_ERROR.format(pattern=pattern, error=e)
                        )
                    else:
                        # Fall back to per-row so one bad row doesn't poison
                        # the batch.  Rare path (schema bugs, missing PKs).
                        logger.warning(
                            ls.MG_REL_FLUSH_ERROR.format(pattern=pattern, error=e)
                        )
                        # Build a per-row variant of the query by swapping
                        # UNWIND-style `row.X` references for `$X` params.
                        per_row_query = (
                            batch_query
                            .replace("row.from_val", "$from_val")
                            .replace("row.to_val", "$to_val")
                        )
                        for k in prop_keys:
                            per_row_query = per_row_query.replace(
                                f"row.{k}", f"$rp_{k}"
                            )
                        for row_entry in batch_rows:
                            row_params: dict[str, PropertyValue] = {
                                "from_val": row_entry["from_val"],
                                "to_val": row_entry["to_val"],
                            }
                            if prop_keys:
                                row_params.update(
                                    {f"rp_{k}": row_entry[k] for k in prop_keys}
                                )
                            try:
                                with self._conn_lock:
                                    self._execute_query(per_row_query, row_params)
                                successful += 1
                            except Exception as inner_e:
                                inner_err = str(inner_e).lower()
                                if (
                                    ERR_SUBSTR_ALREADY_EXISTS in inner_err
                                    or ERR_SUBSTR_CONSTRAINT in inner_err
                                ):
                                    successful += 1
                                elif ERR_SUBSTR_SCHEMA_VIOLATION in inner_err:
                                    # Single schema-illegal row — benign reject,
                                    # not a runtime drop.  Cannot legally exist.
                                    rel_schema_rejected += 1
                                else:
                                    # A genuine RUNTIME per-row failure (Kùzu
                                    # planner fault, endpoint node not visible).
                                    # Previously swallowed without a trace; now
                                    # logged at DEBUG and counted so the
                                    # fail-loud guard below can distinguish it
                                    # from a benign schema reject.  Stays counted
                                    # as a failure (``successful`` not bumped).
                                    rel_runtime_dropped += 1
                                    logger.debug(
                                        "rel per-row RUNTIME drop pattern=%s "
                                        "from=%r to=%r reason=%s",
                                        pattern,
                                        row_entry.get("from_val"),
                                        row_entry.get("to_val"),
                                        str(inner_e)[:160],
                                    )

            failed = attempted - successful
            if rel_type == REL_TYPE_CALLS and failed > 0:
                logger.warning(ls.MG_CALLS_FAILED.format(count=failed))

            # Fail-loud accounting for behavioral edges (CALLS the load-bearing
            # one).  A behavioral rel type that was attempted (rows buffered)
            # yet landed ZERO successful writes WITH at least one genuine
            # runtime drop is the silent CALLS-drop the knowledge-graph viewer
            # surfaces as "nothing is connected".  Accumulate per-type
            # (attempted, successful, runtime_dropped) so the guard after the
            # loop can raise on the exact flush instead of reporting success
            # over a call-less graph — while NOT firing on a type whose only
            # failures were benign schema rejects.
            if rel_type in _BEHAVIORAL_REL_TYPES and attempted > 0:
                p_att, p_succ, p_rt = behavioral_totals.get(rel_type, (0, 0, 0))
                behavioral_totals[rel_type] = (
                    p_att + attempted,
                    p_succ + successful,
                    p_rt + rel_runtime_dropped,
                )

            total_attempted += attempted
            total_successful += successful

        self._rel_count_total += total_successful
        logger.info(
            ls.MG_RELS_FLUSHED.format(
                total=self._rel_count,
                success=total_successful,
                failed=total_attempted - total_successful,
            )
        )
        self._rel_count = 0
        self._rel_groups.clear()

        # Fail-loud on a total behavioral-edge drop CAUSED BY A RUNTIME FAULT.
        # The buffers are already cleared and counters logged above, so the
        # ingestor state is consistent before we raise.  A behavioral rel type
        # (CALLS especially) that was attempted but landed ZERO edges *with at
        # least one genuine runtime drop* means the batch hit Kùzu's
        # UNWIND-MERGE planner bug AND the per-row fallback also failed — the
        # exact silent path that produced a definition-rich but call-less
        # graph.  Raising here surfaces the loss at the flush instead of the
        # job reporting success over a disconnected knowledge graph.
        #
        # Two deliberate exclusions keep this from over-firing:
        #   • PARTIAL drops (some edges landed) are logged via MG_CALLS_FAILED
        #     but do NOT abort — only a total wipe is fatal, mirroring the
        #     definition-node guard's "0 written" trip condition.
        #   • SCHEMA-only rejects (0 landed but every failure was a Binder
        #     "violates schema" — e.g. ``Interface INHERITS Class`` which the
        #     schema forbids, or a constructor call resolved to a ``Class``
        #     callee CALLS cannot target) are benign: those edges cannot
        #     legally exist, so dropping them is correct.  They are excluded by
        #     requiring ``runtime_dropped > 0``.
        zero_landed = {
            rel_type: (att, rt)
            for rel_type, (att, succ, rt) in behavioral_totals.items()
            if att > 0 and succ == 0 and rt > 0
        }
        if zero_landed:
            detail = ", ".join(
                f"{rt_name} (0/{att} edges written; {rt} runtime drops)"
                for rt_name, (att, rt) in zero_landed.items()
            )
            logger.error(
                "Behavioral relationship write-drop: %s. Every edge in these "
                "types failed both the batched UNWIND insert and the per-row "
                "fallback with a runtime fault — the knowledge graph would "
                "render disconnected (\"nothing is connected\"). Failing the "
                "flush so the index job surfaces the loss immediately. Likely "
                "cause: a Kùzu rel-MERGE planner fault on the UNWIND batch "
                "path, or endpoint nodes not visible to the relationship "
                "MATCH.",
                detail,
            )
            raise RelationshipFlushError(
                f"behavioral relationships silently dropped during flush: {detail}"
            )

    def _backfill_defines_file_paths(self) -> None:
        """NAVI-92-A: populate DEFINES.file_path from the connected Module node.

        After ``flush_relationships`` writes DEFINES edges (which carry
        ``file_path = ''`` because the parsers don't populate the column yet),
        this method runs a single batch Cypher to copy the Module's ``path``
        property onto every DEFINES edge whose ``file_path`` is still empty.

        Single-pass, constant-time relative to edge count (one planner call).
        Best-effort: any LadybugDB failure is logged at WARNING and does NOT
        abort the flush — the graph is structurally correct regardless; only
        the file-path enrichment is missing, which degrades to the pre-NAVI-92
        behaviour (centrality overlay shows empty file paths rather than crashing).
        """
        if self.conn is None:
            return
        try:
            with self._conn_lock:
                self.conn.execute(
                    "MATCH (m:Module)-[r:DEFINES]->()\n"
                    "WHERE r.file_path = '' AND m.path IS NOT NULL AND m.path <> ''\n"
                    "SET r.file_path = m.path"
                )
            logger.debug("DEFINES.file_path backfill complete")
        except Exception as exc:
            logger.warning("DEFINES.file_path backfill failed (non-fatal): %s", exc)

    def flush_all(self) -> None:
        logger.info(ls.MG_FLUSH_START)
        self.flush_nodes()
        self.flush_relationships()
        self._backfill_defines_file_paths()
        logger.info(ls.MG_FLUSH_COMPLETE)

    # ------------------------------------------------------------------
    # Query / write API
    # ------------------------------------------------------------------

    def fetch_all(
        self, query: str, params: dict[str, PropertyValue] | None = None
    ) -> list[ResultRow]:
        logger.debug(ls.MG_FETCH_QUERY, query=query, params=params)
        return self._execute_query(query, params)

    def execute_write(
        self, query: str, params: dict[str, PropertyValue] | None = None
    ) -> None:
        logger.debug(ls.MG_WRITE_QUERY, query=query, params=params)
        self._execute_query(query, params)

    # ------------------------------------------------------------------
    # Project management
    # ------------------------------------------------------------------

    def clean_database(self) -> None:
        logger.info(ls.MG_CLEANING_DB)
        self._execute_query(CYPHER_DELETE_ALL)
        logger.info(ls.MG_DB_CLEANED)

    def list_projects(self) -> list[str]:
        result = self.fetch_all(CYPHER_LIST_PROJECTS)
        return [str(r[KEY_NAME]) for r in result]

    def delete_project(self, project_name: str) -> None:
        logger.info(ls.MG_DELETING_PROJECT.format(project_name=project_name))
        self._execute_query(CYPHER_DELETE_PROJECT, {KEY_PROJECT_NAME: project_name})
        logger.info(ls.MG_PROJECT_DELETED.format(project_name=project_name))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_graph_to_dict(self) -> GraphData:
        logger.info(ls.MG_EXPORTING)
        nodes_data = self.fetch_all(CYPHER_EXPORT_NODES)
        relationships_data = self.fetch_all(CYPHER_EXPORT_RELATIONSHIPS)
        metadata = GraphMetadata(
            total_nodes=len(nodes_data),
            total_relationships=len(relationships_data),
            exported_at=datetime.now(UTC).isoformat(),
        )
        logger.info(
            ls.MG_EXPORTED.format(
                nodes=len(nodes_data), rels=len(relationships_data)
            )
        )
        return GraphData(
            nodes=nodes_data,
            relationships=relationships_data,
            metadata=metadata,
        )
