"""S3 snapshot restore (BUC-1555b).

Pulls a previously-written snapshot bundle from
``s3://{bucket}/{prefix}/snapshots/<timestamp>/`` back into a target
directory, verifying SHA-256 of every file against the bundle's
``index.json`` manifest.

Two entry points:

- :func:`restore_latest` — restore the newest available bundle.
- :func:`restore_specific` — restore a named ``snapshot_key``
  (the full ``{prefix}/snapshots/<timestamp>`` path).

Both return a :class:`RestoreResult` dataclass.  Restores are
idempotent: any file already present locally with a matching SHA-256
is skipped.

The whole module is a no-op when boto3 is unavailable or
``S3_INDEX_BUCKET`` is unset; ``restore_*`` returns
``ok=False, error="s3 not configured"``.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .s3_store import (
    _MANIFEST_NAME,
    _bucket,
    _make_client,
    _snapshots_prefix,
    list_snapshots,
)

logger = logging.getLogger(__name__)


@dataclass
class RestoreResult:
    """Outcome of a restore attempt — surfaced via /admin/s3/restore."""

    ok: bool
    files_restored: int = 0
    bytes: int = 0
    snapshot_key: str = ""
    error: str | None = None
    skipped: list[str] = field(default_factory=list)
    verified: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(client, bucket: str, snapshot_key: str) -> dict | None:
    """Fetch the bundle's ``index.json`` manifest.  Returns None when absent."""
    key = f"{snapshot_key}/{_MANIFEST_NAME}"
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
        body = resp["Body"].read()
        return json.loads(body.decode("utf-8"))
    except Exception as exc:
        logger.info("s3_restore: manifest %s not found or unreadable: %s", key, exc)
        return None


def _list_bundle_objects(client, bucket: str, snapshot_key: str) -> list[dict]:
    """Fall-back when no manifest: list every object under the snapshot prefix."""
    objects: list[dict] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=snapshot_key + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            name = key[len(snapshot_key) + 1:]
            if name == _MANIFEST_NAME or "/" in name:
                continue
            objects.append({"name": name, "size_bytes": int(obj.get("Size", 0))})
    return objects


def restore_specific(
    bucket: str | None,
    prefix: str | None,
    snapshot_key: str,
    target_dir: str | Path,
) -> RestoreResult:
    """Restore a named snapshot bundle into *target_dir*.

    Args:
        bucket: S3 bucket override.  Pass ``None`` to use the configured
            ``S3_INDEX_BUCKET``.
        prefix: Currently informational; ``snapshot_key`` is the absolute
            path under the bucket so this is unused but accepted for symmetry
            with :func:`restore_latest`.  Pass ``None`` to ignore.
        snapshot_key: Full snapshot prefix, e.g.
            ``code-indexer/indexes/snapshots/20260507T120000Z``.
        target_dir: Local directory to populate.  Created if absent.

    Returns:
        RestoreResult with ``ok=True`` when every file in the manifest
        downloaded and verified.  ``ok=False`` on any SHA mismatch or
        S3 error — partial files left in place for forensic inspection
        but are NOT considered restored.
    """
    _ = prefix  # accepted for API symmetry; unused
    client = _make_client()
    if client is None:
        return RestoreResult(ok=False, error="s3 not configured", snapshot_key=snapshot_key)

    bkt = bucket or _bucket()
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(client, bkt, snapshot_key)
    if manifest is None:
        # Bundle pre-dates manifest support — fall back to listing.
        files = _list_bundle_objects(client, bkt, snapshot_key)
        if not files:
            return RestoreResult(
                ok=False,
                error=f"snapshot {snapshot_key} is empty or unreachable",
                snapshot_key=snapshot_key,
            )
        manifest = {"version": 0, "snapshot_key": snapshot_key, "files": files}
        manifest_has_sha = False
    else:
        manifest_has_sha = all("sha256" in f for f in manifest.get("files", []))

    if manifest.get("partial"):
        return RestoreResult(
            ok=False,
            error=f"refusing to restore partial snapshot: {manifest.get('error', 'unknown')}",
            snapshot_key=snapshot_key,
        )

    restored = 0
    skipped: list[str] = []
    verified: list[str] = []
    total_bytes = 0
    error: str | None = None

    for fmeta in manifest.get("files", []):
        name = fmeta["name"]
        expected_sha = fmeta.get("sha256")
        size = int(fmeta.get("size_bytes", 0))
        local = target / name

        # Idempotency: skip when local already matches expected SHA.
        if local.exists() and expected_sha:
            try:
                if _sha256_file(local) == expected_sha:
                    skipped.append(name)
                    verified.append(name)
                    total_bytes += size
                    continue
            except OSError:
                pass  # re-download

        key = f"{snapshot_key}/{name}"
        try:
            client.download_file(bkt, key, str(local))
        except Exception as exc:
            error = f"download of {name} failed: {exc}"
            logger.warning("s3_restore: %s", error)
            break

        if expected_sha:
            actual = _sha256_file(local)
            if actual != expected_sha:
                error = (
                    f"sha256 mismatch on {name}: "
                    f"expected {expected_sha[:12]}…, got {actual[:12]}…"
                )
                logger.error("s3_restore: %s", error)
                break
            verified.append(name)
        elif manifest_has_sha is False:
            logger.info("s3_restore: %s restored without sha verification (legacy bundle)", name)

        restored += 1
        total_bytes += size

    ok = error is None
    if ok:
        logger.info(
            "s3_restore: restored %d file(s) from %s (%d skipped, %d verified)",
            restored, snapshot_key, len(skipped), len(verified),
        )
    return RestoreResult(
        ok=ok,
        files_restored=restored,
        bytes=total_bytes,
        snapshot_key=snapshot_key,
        error=error,
        skipped=skipped,
        verified=verified,
    )


def restore_latest(
    bucket: str | None,
    prefix: str | None,
    target_dir: str | Path,
) -> RestoreResult:
    """Restore the newest available snapshot bundle into *target_dir*.

    Convenience wrapper that calls :func:`list_snapshots` and dispatches
    to :func:`restore_specific`.  Returns ``ok=False`` when no bundles
    exist in the bucket.
    """
    _ = (bucket, prefix)  # configured globally via env
    if _make_client() is None:
        return RestoreResult(ok=False, error="s3 not configured")

    bundles = list_snapshots()
    if not bundles:
        return RestoreResult(
            ok=False,
            error=f"no snapshots found under {_snapshots_prefix()}",
        )
    newest = bundles[0]
    return restore_specific(bucket, prefix, newest["key"], target_dir)
