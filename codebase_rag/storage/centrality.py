"""PageRank computation over the LadybugDB CALLS graph.

Reads Function/Method node identifiers and CALLS edges from a LadybugDB
file, builds an in-memory NetworkX DiGraph, computes PageRank with the
default damping factor (0.85), and returns scores normalised to [0, 1]
by dividing by the maximum observed score so they fuse cleanly with
cosine similarity.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def compute_pagerank(repo_db_path: str) -> dict[str, float]:
    """Compute and return normalised PageRank scores for every callable.

    Args:
        repo_db_path: Filesystem path to the LadybugDB ``.db`` file.

    Returns:
        dict mapping qualified_name → score in [0.0, 1.0].  Empty dict
        when the graph has no CALLS edges (single-file repos, etc.).
    """
    import networkx as nx
    import real_ladybug as lb

    db = lb.Database(repo_db_path)
    conn = lb.Connection(db)
    try:
        # Fetch all callable qualified names.
        #
        # LadybugDB's Cypher parser rejects the predicate form
        # ``MATCH (n) WHERE (n:Function OR n:Method)`` with a parse exception
        # (it supports the label-union shorthand ``(n:Function|Method)`` but
        # not a boolean OR over label tests — see
        # ``codebase_rag/cypher_queries.py``). We therefore issue one MATCH
        # per concrete label and merge the results in Python. This keeps the
        # query set portable and is the contract exercised by the LE-32
        # regression guard in code-indexer-service.
        nodes: list[str] = []
        for label in ("Function", "Method"):
            node_res = conn.execute(
                f"MATCH (n:{label}) RETURN n.qualified_name AS qn"
            )
            while node_res.has_next():
                qn = node_res.get_next()[0]
                if isinstance(qn, str):
                    nodes.append(qn)

        # Fetch all CALLS edges, one MATCH per (source-label, target-label)
        # pair, for the same parser-compatibility reason as the node queries.
        edges: list[tuple[str, str]] = []
        for src_label in ("Function", "Method"):
            for dst_label in ("Function", "Method"):
                edge_res = conn.execute(
                    f"""MATCH (s:{src_label})-[:CALLS]->(t:{dst_label})
                       RETURN s.qualified_name AS src, t.qualified_name AS dst"""
                )
                while edge_res.has_next():
                    row = edge_res.get_next()
                    src, dst = row[0], row[1]
                    if isinstance(src, str) and isinstance(dst, str):
                        edges.append((src, dst))
    finally:
        conn.close()
        del conn, db

    if not edges:
        return {}

    g: nx.DiGraph = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    raw = nx.pagerank(g, alpha=0.85)
    if not raw:
        return {}
    max_score = max(raw.values()) or 1.0
    return {qn: score / max_score for qn, score in raw.items()}
