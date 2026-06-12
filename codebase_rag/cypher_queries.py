from .constants import CYPHER_DEFAULT_LIMIT

CYPHER_DELETE_ALL = "MATCH (n) DETACH DELETE n;"

# ---------------------------------------------------------------------------
# Per-file / incremental deletion helpers
# ---------------------------------------------------------------------------

# Step 1 of 3: delete Method nodes hanging off Classes that this Module defines.
# Must run before step 2 so DEFINES_METHOD relationships are gone first.
CYPHER_DELETE_MODULE_METHODS = """
MATCH (m:Module {qualified_name: $qn})-[:DEFINES]->(c:Class)-[:DEFINES_METHOD]->(meth:Method)
DETACH DELETE meth
"""

# Step 2 of 3: delete Functions, Classes, Interfaces, and Enums directly defined
# by the module (DETACH DELETE removes their outgoing relationships automatically).
CYPHER_DELETE_MODULE_DEFINES = """
MATCH (m:Module {qualified_name: $qn})-[:DEFINES]->(node)
DETACH DELETE node
"""

# Step 3 of 3: delete the Module node itself (DETACH DELETE removes remaining
# CONTAINS_MODULE, IMPORTS, CALLS, BELONGS_TO edges on the module).
CYPHER_DELETE_MODULE_NODE = """
MATCH (m:Module {qualified_name: $qn})
DETACH DELETE m
"""

# Remove Package nodes that no longer contain any Module or sub-Package children.
# Runs after module deletion so stale parent packages are cleaned up.
CYPHER_DELETE_ORPHAN_PACKAGES = """
MATCH (pkg:Package)
WHERE NOT (pkg)-[:CONTAINS_MODULE]->(:Module)
AND NOT (pkg)-[:CONTAINS_PACKAGE]->(:Package)
DETACH DELETE pkg
"""

CYPHER_LIST_PROJECTS = "MATCH (p:Project) RETURN p.name AS name ORDER BY p.name"

CYPHER_DELETE_PROJECT = """
MATCH (p:Project {name: $project_name})
OPTIONAL MATCH (p)-[:CONTAINS_PACKAGE|CONTAINS_FOLDER|CONTAINS_FILE|CONTAINS_MODULE*]->(container)
OPTIONAL MATCH (container)-[:DEFINES|DEFINES_METHOD*]->(defined)
DETACH DELETE p, container, defined
"""

CYPHER_EXAMPLE_DECORATED_FUNCTIONS = f"""MATCH (n:Function|Method)
WHERE ANY(d IN n.decorators WHERE toLower(d) IN ['flow', 'task'])
RETURN n.name AS name, n.qualified_name AS qualified_name, labels(n) AS type
LIMIT {CYPHER_DEFAULT_LIMIT}"""

CYPHER_EXAMPLE_CONTENT_BY_PATH = f"""MATCH (n)
WHERE n.path IS NOT NULL AND n.path STARTS WITH 'workflows'
RETURN n.name AS name, n.path AS path, labels(n) AS type
LIMIT {CYPHER_DEFAULT_LIMIT}"""

CYPHER_EXAMPLE_KEYWORD_SEARCH = f"""MATCH (n)
WHERE toLower(n.name) CONTAINS 'database' OR (n.qualified_name IS NOT NULL AND toLower(n.qualified_name) CONTAINS 'database')
RETURN n.name AS name, n.qualified_name AS qualified_name, labels(n) AS type
LIMIT {CYPHER_DEFAULT_LIMIT}"""

CYPHER_EXAMPLE_FIND_FILE = """MATCH (f:File) WHERE toLower(f.name) = 'readme.md' AND f.path = 'README.md'
RETURN f.path as path, f.name as name, labels(f) as type"""

CYPHER_EXAMPLE_README = f"""MATCH (f:File)
WHERE toLower(f.name) CONTAINS 'readme'
RETURN f.path AS path, f.name AS name, labels(f) AS type
LIMIT {CYPHER_DEFAULT_LIMIT}"""

CYPHER_EXAMPLE_PYTHON_FILES = f"""MATCH (f:File)
WHERE f.extension = '.py'
RETURN f.path AS path, f.name AS name, labels(f) AS type
LIMIT {CYPHER_DEFAULT_LIMIT}"""

CYPHER_EXAMPLE_TASKS = f"""MATCH (n:Function|Method)
WHERE 'task' IN n.decorators
RETURN n.qualified_name AS qualified_name, n.name AS name, labels(n) AS type
LIMIT {CYPHER_DEFAULT_LIMIT}"""

CYPHER_EXAMPLE_FILES_IN_FOLDER = f"""MATCH (f:File)
WHERE f.path STARTS WITH 'services'
RETURN f.path AS path, f.name AS name, labels(f) AS type
LIMIT {CYPHER_DEFAULT_LIMIT}"""

CYPHER_EXAMPLE_LIMIT_ONE = """MATCH (f:File) RETURN f.path as path, f.name as name, labels(f) as type LIMIT 1"""

CYPHER_EXAMPLE_CLASS_METHODS = f"""MATCH (c:Class)-[:DEFINES_METHOD]->(m:Method)
WHERE c.qualified_name ENDS WITH '.UserService'
RETURN m.name AS name, m.qualified_name AS qualified_name, labels(m) AS type
LIMIT {CYPHER_DEFAULT_LIMIT}"""

# LadybugDB: properties(n)/properties(r) and type(r) are not supported.
# Use `RETURN n` (whole-node dict with _ID/_LABEL + all props) and label(r).
CYPHER_EXPORT_NODES = """
MATCH (n)
RETURN label(n) AS label, n AS node_data
"""

CYPHER_EXPORT_RELATIONSHIPS = """
MATCH (a)-[r]->(b)
RETURN label(r) AS type, a AS from_node, b AS to_node
"""

CYPHER_RETURN_COUNT = "RETURN count(r) as created"
CYPHER_SET_PROPS_RETURN_COUNT = "SET r += row.props\nRETURN count(r) as created"

# LadybugDB: look up by qualified_name (no integer id(n) exists).
# Two MATCH patterns cover top-level symbols (Module -[:DEFINES]-> Function/Class)
# and nested methods (Module -[:DEFINES]-> Class -[:DEFINES_METHOD]-> Method).
# The Project is found by matching its name as the leading component of the
# symbol's qualified_name (e.g. "myproject.pkg.mod.fn" -> Project {name:"myproject"}).
# This avoids an OPTIONAL MATCH cartesian product when multiple projects share a DB.
CYPHER_GET_FUNCTION_SOURCE_LOCATION = """
MATCH (m:Module)-[:DEFINES]->(n)
WHERE n.qualified_name = $node_id
OPTIONAL MATCH (proj:Project)
WHERE m.qualified_name STARTS WITH proj.name
RETURN n.qualified_name AS qualified_name, n.start_line AS start_line,
       n.end_line AS end_line, m.path AS path, proj.root_path AS root_path,
       n.docstring AS docstring
UNION
MATCH (m:Module)-[:DEFINES]->(c:Class)-[:DEFINES_METHOD]->(n:Method)
WHERE n.qualified_name = $node_id
OPTIONAL MATCH (proj:Project)
WHERE m.qualified_name STARTS WITH proj.name
RETURN n.qualified_name AS qualified_name, n.start_line AS start_line,
       n.end_line AS end_line, m.path AS path, proj.root_path AS root_path,
       n.docstring AS docstring
"""

CYPHER_FIND_BY_QUALIFIED_NAME = """
MATCH (n) WHERE n.qualified_name = $qn
OPTIONAL MATCH (m:Module)-[*]-(n)
RETURN n.name AS name, n.start_line AS start, n.end_line AS end, m.path AS path, n.docstring AS docstring
LIMIT 1
"""


def wrap_with_unwind(query: str) -> str:
    return f"UNWIND $batch AS row\n{query}"


def build_nodes_by_ids_query(node_ids: list[str]) -> str:  # type: ignore[override]
    """Build a Cypher query to look up Function/Method nodes by qualified_name.

    LadybugDB has no integer id(n); ``node_ids`` are qualified_name strings.
    Note: LadybugDB supports label union syntax (Function|Method) but not
    WHERE (n:Function OR n:Method).
    """
    placeholders = ", ".join(f"${i}" for i in range(len(node_ids)))
    return f"""
MATCH (n:Function|Method)
WHERE n.qualified_name IN [{placeholders}]
RETURN n.qualified_name AS node_id, n.qualified_name AS qualified_name,
       label(n) AS type, n.name AS name
ORDER BY n.qualified_name
"""


def build_constraint_query(label: str, prop: str) -> str:
    return f"CREATE CONSTRAINT ON (n:{label}) ASSERT n.{prop} IS UNIQUE;"


def build_index_query(label: str, prop: str) -> str:
    return f"CREATE INDEX ON :{label}({prop});"


def build_merge_node_query(label: str, id_key: str) -> str:
    return f"MERGE (n:{label} {{{id_key}: row.id}})\nSET n += row.props"


def build_merge_relationship_query(
    from_label: str,
    from_key: str,
    rel_type: str,
    to_label: str,
    to_key: str,
    has_props: bool = False,
) -> str:
    query = (
        f"MATCH (a:{from_label} {{{from_key}: row.from_val}}), "
        f"(b:{to_label} {{{to_key}: row.to_val}})\n"
        f"MERGE (a)-[r:{rel_type}]->(b)\n"
    )
    query += CYPHER_SET_PROPS_RETURN_COUNT if has_props else CYPHER_RETURN_COUNT
    return query


def build_create_node_query(label: str, id_key: str) -> str:
    return f"CREATE (n:{label} {{{id_key}: row.id}})\nSET n += row.props"


def build_create_relationship_query(
    from_label: str,
    from_key: str,
    rel_type: str,
    to_label: str,
    to_key: str,
    has_props: bool = False,
) -> str:
    query = (
        f"MATCH (a:{from_label} {{{from_key}: row.from_val}}), "
        f"(b:{to_label} {{{to_key}: row.to_val}})\n"
        f"CREATE (a)-[r:{rel_type}]->(b)\n"
    )
    query += CYPHER_SET_PROPS_RETURN_COUNT if has_props else CYPHER_RETURN_COUNT
    return query
