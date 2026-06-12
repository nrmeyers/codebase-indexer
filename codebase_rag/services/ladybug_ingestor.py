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

from .. import exceptions as ex
from .. import logs as ls
from ..constants import (
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
from ..cypher_queries import (
    CYPHER_DELETE_ALL,
    CYPHER_DELETE_PROJECT,
    CYPHER_EXPORT_NODES,
    CYPHER_EXPORT_RELATIONSHIPS,
    CYPHER_LIST_PROJECTS,
    wrap_with_unwind,
)
from ..types_defs import (
    BatchParams,
    BatchWrapper,
    GraphData,
    GraphMetadata,
    PropertyValue,
    RelBatchRow,
    ResultRow,
)
from .ladybug_schema import migrate


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
    )

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

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> LadybugIngestor:
        logger.info(f"LadybugDB connecting to: {self._db_path}")
        migrate(self._db_path)
        self._db = lb.Database(self._db_path)
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

        for label, props_list in nodes_by_label.items():
            id_key = NODE_UNIQUE_CONSTRAINTS.get(label)
            if not id_key:
                logger.warning(ls.MG_NO_CONSTRAINT.format(label=label))
                skipped_total += len(props_list)
                continue

            skipped = 0
            flushed = 0
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

            skipped_total += skipped
            flushed_total += flushed

        logger.info(
            ls.MG_NODES_FLUSHED.format(flushed=flushed_total, total=buffer_size)
        )
        if skipped_total:
            logger.info(ls.MG_NODES_SKIPPED.format(count=skipped_total))
        self.node_buffer.clear()

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

        for pattern, params_list in self._rel_groups.items():
            from_label, from_key, rel_type, to_label, to_key = pattern
            attempted = 0
            successful = 0

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
                # Use two sequential MATCH clauses instead of comma-separated
                # MATCH (a), (b). The comma pattern triggers a Kuzu hash-join
                # that uses an internal unordered_map which may not include
                # nodes inserted in prior execute() calls on the same connection,
                # producing "unordered_map::at: key not found". Sequential MATCHes
                # perform two independent index lookups and avoid the hash join.
                match_clause = (
                    f"MATCH (a:{from_label} {{{from_key}: row.from_val}})\n"
                    f"MATCH (b:{to_label} {{{to_key}: row.to_val}})"
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

            if rel_type == REL_TYPE_CALLS and (attempted - successful) > 0:
                logger.warning(
                    ls.MG_CALLS_FAILED.format(count=attempted - successful)
                )

            total_attempted += attempted
            total_successful += successful

        logger.info(
            ls.MG_RELS_FLUSHED.format(
                total=self._rel_count,
                success=total_successful,
                failed=total_attempted - total_successful,
            )
        )
        self._rel_count = 0
        self._rel_groups.clear()

    def flush_all(self) -> None:
        logger.info(ls.MG_FLUSH_START)
        self.flush_nodes()
        self.flush_relationships()
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
