#!/usr/bin/env python3
"""768-dim embedder POC driver (docs/embedder-poc-768.md).

Per roster model: spin up an ISOLATED service instance (fresh per-model index
dir, S3 disabled) on port 8001, index the 3 corpus repos on GPU 0, then run
the recall / arms / probes harnesses against it and capture the numbers.

Each model gets its own ``.cgr-poc/<tag>`` dir so indexing is always from
scratch — sidestepping the content-hash "skip re-embed" trap when only the
model changes. Quality metrics (recall/arms/probes) are device-independent;
CPU query latency is measured separately on the contenders.

Runs nothing against the qwen :8000 service. Use --only <tags> to run a
subset; already-completed models (non-empty recall.json) are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS = ["TheForge", "code-indexer-service", "code-graph-rag"]
PORT = 8001
BASE = f"http://127.0.0.1:{PORT}"

# tag -> HF model id. All 768-dim, OSI-licensed, <1B params (see POC doc).
ROSTER: list[tuple[str, str]] = [
    ("e5", "intfloat/e5-base-v2"),                              # 0 baseline
    ("coderank", "nomic-ai/CodeRankEmbed"),                     # 1
    ("jina", "jinaai/jina-embeddings-v2-base-code"),            # 2
    ("gte-modernbert", "Alibaba-NLP/gte-modernbert-base"),      # 3
    ("granite-r2", "ibm-granite/granite-embedding-english-r2"), # 4
]


def repo_path(name: str) -> str:
    home = Path.home()
    for c in (home / name, home / "dev" / "claude" / name, home / "dev" / name):
        if c.is_dir():
            return str(c)
    return str(home / name)


def wait_health(timeout: float = 1800.0) -> dict:
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout:
        try:
            r = httpx.get(f"{BASE}/health", timeout=5)
            if r.status_code == 200:
                return r.json()
        except Exception as e:  # noqa: BLE001
            last = str(e)
        time.sleep(2)
    raise RuntimeError(f"health timeout after {timeout}s (last: {last})")


def index_repo(client: httpx.Client, name: str) -> tuple[str, float]:
    rp = repo_path(name)
    r = client.post(f"{BASE}/index", json={"repo_path": rp, "force_reindex": True}, timeout=30)
    if r.status_code == 409:
        jobs = client.get(f"{BASE}/index/jobs", timeout=10).json()
        active = [j for j in jobs if j.get("status") in ("running", "queued")]
        if not active:
            raise RuntimeError(f"409 but no active job for {name}")
        job_id = active[0]["job_id"]
    else:
        r.raise_for_status()
        job_id = r.json()["job_id"]
    t0 = time.time()
    while True:
        s = client.get(f"{BASE}/index/{job_id}/status", timeout=10).json()
        st = s.get("status")
        if st in ("done", "failed", "cancelled", "interrupted"):
            return st, time.time() - t0
        time.sleep(1)


def run_capture(args_list: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(args_list, cwd=REPO_ROOT, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def run_model(tag: str, hf: str, outdir: Path) -> None:
    mdir = outdir / tag
    mdir.mkdir(parents=True, exist_ok=True)
    if (mdir / "recall.json").exists() and (mdir / "recall.json").stat().st_size > 0:
        print(f"[{tag}] already done — skipping", flush=True)
        return

    idx_dir = REPO_ROOT / ".cgr-poc" / tag
    if idx_dir.exists():
        shutil.rmtree(idx_dir)
    (idx_dir / "repos").mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.update({
        "EMBEDDER_BACKEND": "local",
        "LOCAL_EMBED_MODEL": hf,
        "LOCAL_TRUST_REMOTE_CODE": "1",
        # GPU embedding: EMBED_DEVICE != "cpu" stops the index router from
        # forcing the embed subprocess onto CPU (index.py:1981), so it inherits
        # these CUDA vars and runs on GPU. PCI_BUS_ID order makes device 0 the
        # RTX 3060 (~10 GB free); default CUDA order makes device 0 the RTX 3090,
        # which llama-server has nearly full → CUDA OOM.
        "EMBED_DEVICE": "cuda",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # Small encode batch: long-context code models spike activation memory
        # on long files (code-indexer-service) and OOM the 3060 at batch 32.
        "LOCAL_ENCODE_BATCH_SIZE": "8",
        "LADYBUG_DB_DIR": str(idx_dir / "repos"),
        "JOBS_DB_PATH": str(idx_dir / "jobs.sqlite"),
        "S3_INDEX_BUCKET": "",
        "WATCH_ENABLED": "false",
        "HOST": "127.0.0.1",
        "PORT": str(PORT),
    })
    log = open(mdir / "service.log", "w")
    print(f"[{tag}] starting service ({hf})", flush=True)
    proc = subprocess.Popen(
        ["uv", "run", "python", "-m", "uvicorn", "app.main:app",
         "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    try:
        wait_health()
        print(f"[{tag}] healthy — indexing corpus", flush=True)
        idx: dict[str, dict] = {}
        with httpx.Client() as client:
            for name in CORPUS:
                st, dur = index_repo(client, name)
                idx[name] = {"status": st, "seconds": round(dur, 1)}
                print(f"[{tag}]   {name}: {st} in {dur:.0f}s", flush=True)
                if st != "done":
                    raise RuntimeError(f"index {name} -> {st}")

        rc, out, err = run_capture(["uv", "run", "python", "scripts/run_recall.py", "--service-url", BASE])
        (mdir / "recall.json").write_text(out)
        (mdir / "recall.stderr").write_text(err)

        rc2, out2, err2 = run_capture(["uv", "run", "python", "scripts/run_arms.py", "--service-url", BASE, "--allow-dirty"])
        (mdir / "arms.out").write_text(out2 + "\n===STDERR===\n" + err2)

        rc3, out3, err3 = run_capture(["uv", "run", "python", "scripts/run_probes.py", "--service-url", BASE])
        (mdir / "probes.out").write_text(out3 + "\n===STDERR===\n" + err3)

        size = sum(f.stat().st_size for f in idx_dir.rglob("*") if f.is_file())
        (mdir / "meta.json").write_text(json.dumps(
            {"tag": tag, "model": hf, "index": idx, "index_bytes": size,
             "recall_rc": rc, "arms_rc": rc2, "probes_rc": rc3}, indent=1))
        print(f"[{tag}] DONE recall_rc={rc} arms_rc={rc2} probes_rc={rc3} "
              f"size={size // 1024 // 1024}MB", flush=True)
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=40)
        except Exception:  # noqa: BLE001
            proc.kill()
        log.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="comma-separated tags to run")
    ap.add_argument("--outdir", default=str(REPO_ROOT / ".planning" / "runs" / "768-poc"))
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for tag, hf in ROSTER:
        if only and tag not in only:
            continue
        try:
            run_model(tag, hf, outdir)
        except Exception as e:  # noqa: BLE001 — one model failing must not sink the roster
            print(f"[{tag}] ERROR: {e}", flush=True)
    print("POC COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
