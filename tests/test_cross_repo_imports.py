"""Tests for ``app.services.cross_repo_imports`` (BUC-1598).

Three layers of coverage:

    * **Unit** — identity extraction (package.json / pyproject / setup.py)
      and matcher heuristics (npm exact, python first-segment normalised).
    * **Integration** — full resolver flow against a real LadybugDB,
      seeded with one importer and one external Module that matches a
      sibling repo's identity.  Verifies the cross-repo qname is
      written, the old External Module is gone, and IMPORTS now points
      at the canonical Module.
    * **API** — ``POST /admin/resolve-cross-repo-imports`` honors the
      feature flag and reports per-repo stats.

Tests that exercise real LadybugDB write to ``tmp_path`` and never share
state between tests.  Tests that exercise the feature flag use
``monkeypatch.setenv`` to keep the global ``settings`` instance
unmolested.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import cross_repo_imports as cri
from app.services.cross_repo_imports import (
    CROSS_REPO_PREFIX_SEP,
    RepoIdentity,
    extract_repo_identity,
    is_cross_repo_qname,
    is_enabled,
    make_cross_repo_qname,
    match_external_module,
    resolve_cross_repo_imports,
)


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------


class TestFeatureFlag:
    """``CROSS_REPO_IMPORTS_ENABLED`` env-var parsing."""

    def test_defaults_to_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No env var set should return False so behaviour is unchanged."""
        monkeypatch.delenv("CROSS_REPO_IMPORTS_ENABLED", raising=False)
        assert is_enabled() is False

    @pytest.mark.parametrize("val", ["true", "TRUE", "1", "yes", "on", "True"])
    def test_truthy_values_enable(self, monkeypatch: pytest.MonkeyPatch, val: str) -> None:
        """Any of the canonical truthy strings should flip the flag on."""
        monkeypatch.setenv("CROSS_REPO_IMPORTS_ENABLED", val)
        assert is_enabled() is True

    @pytest.mark.parametrize("val", ["false", "0", "no", "off", "", "maybe"])
    def test_falsy_values_disable(self, monkeypatch: pytest.MonkeyPatch, val: str) -> None:
        """Everything that isn't an opt-in string keeps the flag off."""
        monkeypatch.setenv("CROSS_REPO_IMPORTS_ENABLED", val)
        assert is_enabled() is False


# ---------------------------------------------------------------------------
# Identity extraction
# ---------------------------------------------------------------------------


class TestExtractRepoIdentity:
    """``extract_repo_identity`` reads package.json / pyproject / setup.py."""

    def test_npm_name_from_package_json(self, tmp_path: Path) -> None:
        """A scoped npm name in package.json populates ``npm_name``."""
        (tmp_path / "package.json").write_text(
            json.dumps({"name": "@navistone/shared-types", "version": "1.0.0"})
        )
        ident = extract_repo_identity("navistone__shared-types", str(tmp_path))
        assert ident.npm_name == "@navistone/shared-types"
        assert ident.python_name == ""

    def test_python_name_from_pyproject(self, tmp_path: Path) -> None:
        """A PEP 621 ``project.name`` populates ``python_name``."""
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "my-package"\nversion = "0.1.0"\n'
        )
        # Mirror the on-disk import layout so top-level detection works.
        (tmp_path / "my_package").mkdir()
        (tmp_path / "my_package" / "__init__.py").write_text("")
        ident = extract_repo_identity("acme__my-package", str(tmp_path))
        assert ident.python_name == "my-package"
        assert ident.python_top_level == "my_package"

    def test_python_name_from_setup_py_fallback(self, tmp_path: Path) -> None:
        """``setup.py`` regex fallback when pyproject is absent."""
        (tmp_path / "setup.py").write_text(
            "from setuptools import setup\n"
            'setup(name="legacy-pkg", version="0.1.0", packages=["legacy_pkg"])\n'
        )
        ident = extract_repo_identity("acme__legacy-pkg", str(tmp_path))
        assert ident.python_name == "legacy-pkg"

    def test_missing_manifests_yields_empty_identity(self, tmp_path: Path) -> None:
        """No manifest = no identity claims; matcher will skip the repo."""
        ident = extract_repo_identity("acme__empty", str(tmp_path))
        assert ident.slug == "acme__empty"
        assert ident.npm_name == ""
        assert ident.python_name == ""

    def test_malformed_json_is_silent(self, tmp_path: Path) -> None:
        """A broken package.json should NOT raise — we treat it as 'no claim'."""
        (tmp_path / "package.json").write_text("{ this is not json")
        ident = extract_repo_identity("acme__broken", str(tmp_path))
        assert ident.npm_name == ""

    def test_nonexistent_root_returns_slug_only(self) -> None:
        """A bad root_path yields a slug-only identity, never raises."""
        ident = extract_repo_identity("acme__ghost", "/nonexistent/path/ghost")
        assert ident == RepoIdentity(slug="acme__ghost")

    def test_src_layout_top_level_detection(self, tmp_path: Path) -> None:
        """``src/<pkg>/__init__.py`` layout populates ``python_top_level``."""
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "code-indexer-service"\n'
        )
        (tmp_path / "src" / "code_indexer_service").mkdir(parents=True)
        (tmp_path / "src" / "code_indexer_service" / "__init__.py").write_text("")
        ident = extract_repo_identity(
            "navistone__code-indexer-service", str(tmp_path)
        )
        assert ident.python_top_level == "code_indexer_service"


# ---------------------------------------------------------------------------
# Matcher heuristics
# ---------------------------------------------------------------------------


class TestMatchExternalModule:
    """``match_external_module`` picks the right sibling — or None."""

    def test_npm_scoped_exact_match(self) -> None:
        """``@navistone/shared-types`` matches a repo whose npm_name equals it."""
        siblings = [
            RepoIdentity(slug="navistone__shared-types", npm_name="@navistone/shared-types"),
            RepoIdentity(slug="navistone__other", npm_name="@navistone/other"),
        ]
        ident = match_external_module("@navistone/shared-types", "", siblings)
        assert ident is not None
        assert ident.slug == "navistone__shared-types"

    def test_npm_bare_name_match(self) -> None:
        """Unscoped npm name (``lodash``) matches via ``npm_name``."""
        siblings = [RepoIdentity(slug="vendor__lodash", npm_name="lodash")]
        ident = match_external_module("lodash", "", siblings)
        assert ident is not None
        assert ident.slug == "vendor__lodash"

    def test_python_first_segment_match(self) -> None:
        """``django.db.models`` matches a repo whose python_name is ``django``."""
        siblings = [
            RepoIdentity(slug="djangoproject__django", python_name="django"),
        ]
        ident = match_external_module("django.db.models", "django.db.models", siblings)
        assert ident is not None
        assert ident.slug == "djangoproject__django"

    def test_python_name_normalisation(self) -> None:
        """PEP 503: ``my-package`` ↔ ``my_package`` collapse for matching."""
        siblings = [
            RepoIdentity(slug="acme__my-pkg", python_name="my-package", python_top_level="my_package"),
        ]
        # Import ``my_package`` from on-disk should match ``my-package`` PyPI name.
        ident = match_external_module("my_package.helpers", "my_package.helpers", siblings)
        assert ident is not None
        assert ident.slug == "acme__my-pkg"

    def test_no_match_returns_none(self) -> None:
        """Nothing claims ``unknown_lib`` → None, caller falls back to External."""
        siblings = [
            RepoIdentity(slug="acme__a", npm_name="@acme/a"),
            RepoIdentity(slug="acme__b", python_name="b_pkg"),
        ]
        ident = match_external_module("unknown_lib", "unknown_lib", siblings)
        assert ident is None

    def test_empty_identity_does_not_match(self) -> None:
        """An identity with no claims must never match anything."""
        siblings = [RepoIdentity(slug="acme__empty")]
        assert match_external_module("anything", "anything", siblings) is None

    def test_path_hint_is_used_when_qname_fails(self) -> None:
        """When the qname is normalised but the original path still has the npm form."""
        siblings = [RepoIdentity(slug="acme__pkg", npm_name="@acme/pkg")]
        # qname = stripped form, path = literal import path.
        ident = match_external_module("@acme.pkg", "@acme/pkg", siblings)
        assert ident is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestQnameHelpers:
    """Cross-repo qname construction + detection."""

    def test_make_cross_repo_qname(self) -> None:
        """``{slug}::{qname}`` is the canonical form."""
        assert (
            make_cross_repo_qname("navistone__shared-types", "@navistone/shared-types")
            == "navistone__shared-types::@navistone/shared-types"
        )

    def test_is_cross_repo_qname(self) -> None:
        """The ``::`` separator is the only marker."""
        assert is_cross_repo_qname("acme__pkg::foo.bar") is True
        assert is_cross_repo_qname("foo.bar") is False
        assert CROSS_REPO_PREFIX_SEP == "::"


# ---------------------------------------------------------------------------
# Integration — real LadybugDB
# ---------------------------------------------------------------------------


@pytest.fixture()
def seeded_importer_db(tmp_path: Path) -> str:
    """Build a tiny LadybugDB with one Project, two Modules, one IMPORTS edge.

    Layout:

        Module(qualified_name='theforge.web.app',  name='app',  path='...') -- our own module
        Module(qualified_name='@navistone/shared-types', name='shared-types', path='@navistone/shared-types') -- external
        (theforge.web.app)-[:IMPORTS]->(@navistone/shared-types)

    The resolver should rewrite the external Module's qname to
    ``navistone__shared-types::@navistone/shared-types`` and copy the
    IMPORTS edge over to point at the new node.
    """
    import real_ladybug as lb  # type: ignore[import-untyped]
    from codebase_rag.services.ladybug_schema import migrate

    db_path = str(tmp_path / "theforge.db")
    migrate(db_path)

    db = lb.Database(db_path)
    conn = lb.Connection(db)
    try:
        conn.execute("CREATE (p:Project {name: 'TheForge'})")
        conn.execute(
            "CREATE (m:Module {qualified_name: 'TheForge.web.app', "
            "name: 'app', path: 'web/src/app.ts'})"
        )
        conn.execute(
            "CREATE (m:Module {qualified_name: '@navistone/shared-types', "
            "name: 'shared-types', path: '@navistone/shared-types'})"
        )
        conn.execute(
            "MATCH (src:Module {qualified_name: 'TheForge.web.app'}) "
            "MATCH (dst:Module {qualified_name: '@navistone/shared-types'}) "
            "CREATE (src)-[:IMPORTS]->(dst)"
        )
    finally:
        conn.close()
        del conn, db
        gc.collect()
    return db_path


@pytest.fixture()
def sibling_repo_root(tmp_path: Path) -> Path:
    """A throwaway sibling repo with package.json claiming the npm name."""
    root = tmp_path / "sibling"
    root.mkdir()
    (root / "package.json").write_text(
        json.dumps({"name": "@navistone/shared-types", "version": "1.0.0"})
    )
    return root


class TestResolveCrossRepoImports:
    """End-to-end resolution against a real LadybugDB instance."""

    def test_flag_disabled_is_noop(
        self,
        seeded_importer_db: str,
        sibling_repo_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With the flag off the resolver returns zeros and touches nothing."""
        monkeypatch.delenv("CROSS_REPO_IMPORTS_ENABLED", raising=False)

        sibling = extract_repo_identity("navistone__shared-types", str(sibling_repo_root))
        stats = resolve_cross_repo_imports(
            "navistone__TheForge", seeded_importer_db, [sibling]
        )
        assert stats.scanned == 0
        assert stats.matched == 0

    def test_rewires_external_to_cross_repo(
        self,
        seeded_importer_db: str,
        sibling_repo_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The acceptance test: an external Module becomes a cross-repo Module
        carrying the ``{sibling_slug}::`` prefix, and its inbound IMPORTS
        edge now points at the new node.
        """
        monkeypatch.setenv("CROSS_REPO_IMPORTS_ENABLED", "true")

        sibling = extract_repo_identity(
            "navistone__shared-types", str(sibling_repo_root)
        )
        assert sibling.npm_name == "@navistone/shared-types"

        stats = resolve_cross_repo_imports(
            "navistone__TheForge", seeded_importer_db, [sibling]
        )
        assert stats.matched == 1, f"expected 1 rewire, got {stats}"
        assert stats.scanned == 1
        assert stats.unmatched == 0
        assert stats.errors == []

        # Verify the rewrite committed: the old external Module is gone,
        # the new cross-repo Module exists, and the IMPORTS edge now
        # targets the new node.
        import real_ladybug as lb  # type: ignore[import-untyped]
        db = lb.Database(seeded_importer_db, read_only=True)
        conn = lb.Connection(db)
        try:
            res = conn.execute(
                "MATCH (m:Module {qualified_name: '@navistone/shared-types'}) "
                "RETURN count(m) AS cnt"
            )
            assert res.has_next()
            assert int(res.get_next()[0]) == 0, "old external Module should be deleted"

            new_qn = make_cross_repo_qname(
                "navistone__shared-types", "@navistone/shared-types"
            )
            res = conn.execute(
                "MATCH (m:Module {qualified_name: $qn}) RETURN m.name AS name",
                {"qn": new_qn},
            )
            assert res.has_next()
            row = res.get_next()
            assert row[0] == "shared-types"

            res = conn.execute(
                "MATCH (src:Module)-[:IMPORTS]->(dst:Module) "
                "WHERE dst.qualified_name CONTAINS '::' "
                "RETURN count(*) AS cnt"
            )
            assert res.has_next()
            count_cross_repo = int(res.get_next()[0])
            # BUC-1598 acceptance: at least one cross-repo IMPORTS edge.
            assert count_cross_repo >= 1
        finally:
            conn.close()
            del conn, db
            gc.collect()

    def test_unmatched_external_stays_as_external(
        self,
        seeded_importer_db: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When no sibling identity claims the external, leave it alone."""
        monkeypatch.setenv("CROSS_REPO_IMPORTS_ENABLED", "true")

        # Sibling claims a DIFFERENT npm name — won't match.
        other_root = tmp_path / "unrelated"
        other_root.mkdir()
        (other_root / "package.json").write_text(
            json.dumps({"name": "@unrelated/thing"})
        )
        sibling = extract_repo_identity("unrelated__thing", str(other_root))

        stats = resolve_cross_repo_imports(
            "navistone__TheForge", seeded_importer_db, [sibling]
        )
        assert stats.scanned == 1
        assert stats.matched == 0
        assert stats.unmatched == 1

        # Verify the original external Module is still there untouched.
        import real_ladybug as lb  # type: ignore[import-untyped]
        db = lb.Database(seeded_importer_db, read_only=True)
        conn = lb.Connection(db)
        try:
            res = conn.execute(
                "MATCH (m:Module {qualified_name: '@navistone/shared-types'}) "
                "RETURN m.name AS name"
            )
            assert res.has_next()
        finally:
            conn.close()
            del conn, db
            gc.collect()

    def test_idempotent_second_run_is_zero(
        self,
        seeded_importer_db: str,
        sibling_repo_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A second resolution pass must match nothing (acceptance criterion)."""
        monkeypatch.setenv("CROSS_REPO_IMPORTS_ENABLED", "true")
        sibling = extract_repo_identity(
            "navistone__shared-types", str(sibling_repo_root)
        )

        first = resolve_cross_repo_imports(
            "navistone__TheForge", seeded_importer_db, [sibling]
        )
        assert first.matched == 1

        second = resolve_cross_repo_imports(
            "navistone__TheForge", seeded_importer_db, [sibling]
        )
        assert second.matched == 0
        assert second.scanned == 0  # filtered out by the `::` exclusion


# ---------------------------------------------------------------------------
# API — POST /admin/resolve-cross-repo-imports
# ---------------------------------------------------------------------------


client = TestClient(app)


class TestAdminResolveCrossRepoImports:
    """``POST /admin/resolve-cross-repo-imports`` happy + flag paths."""

    def test_flag_disabled_returns_empty_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With flag off, endpoint responds 200 with enabled=False."""
        monkeypatch.delenv("CROSS_REPO_IMPORTS_ENABLED", raising=False)
        resp = client.post("/admin/resolve-cross-repo-imports")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["enabled"] is False
        assert body["results"] == []
        assert body["total_matched"] == 0

    def test_flag_enabled_with_empty_registry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Flag on but no indexed repos → error='no_indexed_repos'."""
        monkeypatch.setenv("CROSS_REPO_IMPORTS_ENABLED", "true")
        with patch("app.routers.index.indexed_repo_paths", {}):
            resp = client.post("/admin/resolve-cross-repo-imports")
        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] is True
        assert body["results"] == []
        assert body["error"] == "no_indexed_repos"

    def test_flag_enabled_runs_resolver(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With flag on + registry populated, resolver is invoked."""
        monkeypatch.setenv("CROSS_REPO_IMPORTS_ENABLED", "true")
        fake_results = [
            cri.ResolveStats(
                slug="navistone__TheForge",
                scanned=10,
                matched=3,
                unmatched=7,
                duration_ms=42.0,
            )
        ]
        with (
            patch(
                "app.routers.index.indexed_repo_paths",
                {"navistone__TheForge": "/tmp/forge"},
            ),
            patch(
                "app.services.cross_repo_imports.resolve_all",
                return_value=fake_results,
            ),
        ):
            resp = client.post("/admin/resolve-cross-repo-imports")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["enabled"] is True
        assert body["total_matched"] == 3
        assert len(body["results"]) == 1
        assert body["results"][0]["slug"] == "navistone__TheForge"
