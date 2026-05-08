"""Tests for canonical slug normalisation (BUC-1580).

Covers the pure-python helpers in ``app.services.slug`` plus the
``POST /admin/migrate-slugs`` endpoint's collision-refusal contract.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.slug import (
    canonical_slug_for_path,
    derive_slug,
    parse_github_remote,
)

client = TestClient(app)


class TestParseGithubRemote:
    """Unit tests for ``parse_github_remote`` URL parsing."""

    def test_parses_ssh_url(self) -> None:
        """SSH form ``git@github.com:org/repo.git`` parses to (org, repo)."""
        assert parse_github_remote("git@github.com:navistone/TheForge.git") == (
            "navistone",
            "TheForge",
        )

    def test_parses_https_url_without_dot_git(self) -> None:
        """HTTPS form without ``.git`` suffix parses cleanly."""
        assert parse_github_remote("https://github.com/navistone/TheForge") == (
            "navistone",
            "TheForge",
        )

    def test_returns_none_for_non_github_host(self) -> None:
        """Non-GitHub hosts (gitlab) return None so caller falls back."""
        assert parse_github_remote("https://gitlab.com/foo/bar.git") is None

    def test_returns_none_for_garbage_input(self) -> None:
        """Empty / non-URL strings return None instead of raising."""
        assert parse_github_remote("") is None
        assert parse_github_remote("not a url at all") is None


class TestCanonicalSlugForPath:
    """Tests for the git-subprocess wrapper."""

    def test_returns_canonical_slug_when_remote_parses(self, tmp_path: Path) -> None:
        """Happy path: ``origin`` is a single GitHub remote → canonical slug."""
        local = tmp_path / "TheForge"
        local.mkdir()

        # Two subprocess calls fire in sequence: ``git remote`` then
        # ``git remote get-url origin``.  side_effect models both.
        from subprocess import CompletedProcess

        results = [
            CompletedProcess(args=[], returncode=0, stdout="origin\n", stderr=""),
            CompletedProcess(
                args=[],
                returncode=0,
                stdout="git@github.com:navistone/TheForge.git\n",
                stderr="",
            ),
        ]
        with patch("app.services.slug.subprocess.run", side_effect=results):
            assert canonical_slug_for_path(local) == "navistone__TheForge"


class TestDeriveSlug:
    """End-to-end behaviour of the public ``derive_slug`` entry point."""

    def test_falls_back_to_basename_when_canonical_unavailable(
        self, tmp_path: Path
    ) -> None:
        """No GitHub remote → use the basename (passed through ``slugify_repo``)."""
        local = tmp_path / "my-local-repo"
        local.mkdir()
        # Force the canonical lookup to return None as if git is absent.
        with patch(
            "app.services.slug.canonical_slug_for_path", return_value=None
        ):
            assert derive_slug(local, "my-local-repo") == "my-local-repo"

    def test_returns_canonical_when_remote_parses(self, tmp_path: Path) -> None:
        """When the canonical helper finds a slug, that wins over the basename."""
        local = tmp_path / "TheForge"
        local.mkdir()
        with patch(
            "app.services.slug.canonical_slug_for_path",
            return_value="navistone__TheForge",
        ):
            assert derive_slug(local, "TheForge") == "navistone__TheForge"


class TestMigrateSlugsCollision:
    """Migration endpoint refuses to clobber an existing canonical slug."""

    def test_refuses_when_canonical_slug_already_has_content(
        self, tmp_path: Path
    ) -> None:
        """The collision case from the BUC-1580 ticket: ``TheForge`` AND
        ``navistone__TheForge`` both indexed — refuse with explicit reason
        so the operator can decide which to keep."""
        # Stage a fake LADYBUG_DB_DIR with two .db files that simulate the
        # collision: ``TheForge.db`` (bare) and ``navistone__TheForge.db``
        # (canonical, already populated).
        db_dir = tmp_path / "repos"
        db_dir.mkdir()
        (db_dir / "TheForge.db").write_bytes(b"x" * 8192)
        (db_dir / "navistone__TheForge.db").write_bytes(b"y" * 8192)

        # Minimal in-memory state: TheForge is registered with a root_path
        # whose ``derive_slug`` will canonicalise to ``navistone__TheForge``.
        from app.routers import index as index_module

        index_module.indexed_repos.add("TheForge")
        index_module.indexed_repo_paths["TheForge"] = str(tmp_path / "TheForge_clone")

        try:
            with (
                patch("app.routers.admin.settings.LADYBUG_DB_DIR", str(db_dir)),
                patch(
                    "app.routers.admin.derive_slug",
                    return_value="navistone__TheForge",
                ),
                patch(
                    "app.routers.index._read_meta",
                    return_value={"root_path": str(tmp_path / "TheForge_clone")},
                ),
            ):
                resp = client.post("/admin/migrate-slugs")
        finally:
            index_module.indexed_repos.discard("TheForge")
            index_module.indexed_repo_paths.pop("TheForge", None)

        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        # The bare-basename slug must NOT be migrated — collision refused.
        assert body["migrated"] == [] or all(
            entry["old"] != "TheForge" for entry in body["migrated"]
        )
        # And the skip reason must mention the canonical slug clearly.
        skipped_for_theforge = [
            s for s in body["skipped"] if s["slug"] == "TheForge"
        ]
        assert len(skipped_for_theforge) == 1
        assert "navistone__TheForge" in skipped_for_theforge[0]["reason"]
        # Verify both files still exist on disk — no half-migration.
        assert (db_dir / "TheForge.db").exists()
        assert (db_dir / "navistone__TheForge.db").exists()
