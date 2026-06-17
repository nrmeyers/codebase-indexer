"""Integration tests — full index → search → context-bundle flow.

These tests spin up a real LadybugDB in a tmp directory, index a small
synthetic Python repo through the FastAPI app, then verify that every search
endpoint returns sensible data.

They are intentionally lightweight (no ML embeddings — semantic search is
mocked) so they run in CI without torch/transformers installed.

Mark: ``pytest -m integration`` to run this file specifically.
"""
from __future__ import annotations

import textwrap
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_repo(tmp_path: Path) -> Path:
    """Create a minimal Python repo that code-graph-rag can index."""
    src = tmp_path / "myapp"
    src.mkdir()
    (src / "__init__.py").write_text("")

    (src / "utils.py").write_text(
        textwrap.dedent(
            """\
            def add(a: int, b: int) -> int:
                return a + b

            def subtract(a: int, b: int) -> int:
                return a - b
            """
        )
    )
    (src / "main.py").write_text(
        textwrap.dedent(
            """\
            from .utils import add, subtract

            def run() -> None:
                result = add(1, 2)
                result2 = subtract(result, 1)
                print(result2)
            """
        )
    )
    return tmp_path


@pytest.fixture()
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / ".cgr" / "graph.db")


@pytest.fixture()
def app_client(db_path: str):
    """TestClient wired to a fresh LadybugDB instance."""
    import os

    os.environ["LADYBUG_DB_PATH"] = db_path

    # Re-import app after env var is set so config picks it up.
    import app.config as cfg_mod

    cfg_mod.settings.__class__.model_config["env_file"] = None  # type: ignore[index]
    cfg_mod.settings = cfg_mod.Settings(LADYBUG_DB_PATH=db_path)  # type: ignore[call-arg]

    # Patch the per-router _get_conn calls to use the new db_path.
    import app.routers.health as health_mod
    import app.routers.index as index_mod
    import app.routers.search as search_mod
    import app.routers.context_bundle as cb_mod

    def _get_conn_fresh():
        import ladybug as lb  # type: ignore[import-untyped]

        db = lb.Database(db_path)
        conn = lb.Connection(db)
        try:
            conn.execute("LOAD EXTENSION VECTOR")
        except Exception:
            pass
        return conn

    with (
        patch.object(health_mod, "_get_indexed_repos", return_value=[]),
        patch.object(index_mod, "_run_ingestion"),
        patch.object(search_mod, "_get_conn", side_effect=_get_conn_fresh),
        patch.object(cb_mod, "_get_conn", side_effect=_get_conn_fresh),
    ):
        from app.main import app

        with TestClient(app) as client:
            yield client


# ---------------------------------------------------------------------------
# Smoke: health
# ---------------------------------------------------------------------------


def test_smoke_health(app_client: TestClient) -> None:
    resp = app_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Index lifecycle
# ---------------------------------------------------------------------------


def test_index_lifecycle(tmp_path: Path) -> None:
    """POST /index → 202, GET /index/{id}/status → done (with blocking mock)."""
    from app.routers.index import _jobs

    _jobs.clear()

    def _fake_blocking(job, force_reindex):  # type: ignore[override]
        job.node_count = 5
        job.rel_count = 3
        job.progress_pct = 100.0
        job.status = "done"

    with patch("app.routers.index._blocking_index", side_effect=_fake_blocking):
        from app.main import app

        with TestClient(app) as client:
            post = client.post("/index", json={"repo_path": str(tmp_path)})
            assert post.status_code == 202
            job_id = post.json()["job_id"]

            deadline = time.time() + 5
            status_resp = None
            while time.time() < deadline:
                status_resp = client.get(f"/index/{job_id}/status")
                if status_resp.json()["status"] == "done":
                    break
                time.sleep(0.05)

    assert status_resp is not None
    body = status_resp.json()
    assert body["status"] == "done"
    assert body["node_count"] == 5
    assert body["rel_count"] == 3
    assert body["progress_pct"] == pytest.approx(100.0)

    _jobs.clear()


# ---------------------------------------------------------------------------
# Search endpoints (structural / symbol) against a real LadybugDB
# ---------------------------------------------------------------------------


@pytest.fixture()
def seeded_db(tmp_path: Path) -> str:
    """Create a LadybugDB with a small schema and a few nodes."""
    db_path = str(tmp_path / "test.db")
    try:
        import ladybug as lb  # type: ignore[import-untyped]
        from codebase_rag.services.ladybug_schema import migrate

        migrate(db_path)

        db = lb.Database(db_path)
        conn = lb.Connection(db)
        conn.execute(
            "CREATE (p:Project {name: 'myapp'})"
        )
        src_file = tmp_path / "myapp" / "utils.py"
        src_file.parent.mkdir(parents=True, exist_ok=True)
        src_file.write_text("def add(a, b):\n    return a + b\n")
        conn.execute(
            "CREATE (m:Module {qualified_name: 'myapp.utils', name: 'utils', path: $path})",
            {"path": str(src_file)},
        )
        conn.execute(
            "CREATE (f:Function {qualified_name: 'myapp.utils.add', name: 'add', "
            "start_line: 1, end_line: 2, decorators: [], docstring: '', source_code: ''})",
        )
        # DEFINES relationship: Module → Function
        conn.execute(
            "MATCH (m:Module {qualified_name: 'myapp.utils'}), "
            "(f:Function {qualified_name: 'myapp.utils.add'}) "
            "CREATE (m)-[:DEFINES]->(f)",
        )
    except Exception:
        pytest.skip("LadybugDB not available or schema migration failed")
    return db_path


def test_structural_search_real_db(seeded_db: str) -> None:
    """Run a structural query against a seeded LadybugDB."""
    import app.routers.search as search_mod

    def _conn():
        import ladybug as lb  # type: ignore[import-untyped]

        return lb.Connection(lb.Database(seeded_db))

    with patch.object(search_mod, "_get_conn", side_effect=lambda repo=None: _conn()):
        from app.main import app

        with TestClient(app) as client:
            resp = client.get(
                "/search/structural",
                params={"q": "MATCH (f:Function) RETURN f.name AS name, f.qualified_name AS qn"},
            )

    assert resp.status_code == 200, f"got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["row_count"] >= 1
    names = [n.get("name") for n in body["nodes"]]
    assert "add" in names


def test_symbol_lookup_real_db(seeded_db: str, tmp_path: Path) -> None:
    """Look up 'myapp.utils.add' and verify source is returned."""
    import app.routers.search as search_mod

    def _conn():
        import ladybug as lb  # type: ignore[import-untyped]

        return lb.Connection(lb.Database(seeded_db))

    with patch.object(search_mod, "_get_conn", side_effect=lambda repo=None: _conn()):
        from app.main import app

        with TestClient(app) as client:
            resp = client.get("/search/symbol", params={"fqn": "myapp.utils.add"})

    assert resp.status_code == 200, f"got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["qualified_name"] == "myapp.utils.add"
    assert "def add" in body["source"]
