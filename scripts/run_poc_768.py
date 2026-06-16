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

# llama-server (GGUF) backend wiring for the nomic-v1.5 rerun. The in-process
# sentence-transformers path was both slow and OOM-prone on the long-context
# code corpus; swapping in llama.cpp's --embeddings server sidesteps both. We
# launch llama-server in a rootless podman container that mirrors the existing
# llama-reranker/llama-judge pods: same ``localhost/llama-server-cuda:latest``
# image, same ``/mnt/ai-data/llama/models`` read-only mount, GPU-0 device
# passthrough (nvidia0 = RTX 3060). The on-disk ``llama-server-cuda.sh``
# wrapper is broken on this host (expects ``/usr/lib/ollama/``); the podman
# image bakes the working binary.
LLAMA_PORT = 8090
LLAMA_BASE = f"http://127.0.0.1:{LLAMA_PORT}"
LLAMA_IMAGE = "localhost/llama-server-cuda:latest"
LLAMA_CONTAINER = "cgr-poc-llama"
LLAMA_MODELS_DIR = "/mnt/ai-data/llama/models"
# CDI device handle — pulls in libcuda + the right /dev/nvidia* and DRI nodes
# via /etc/cdi/nvidia.yaml. The raw-device enumeration that llama-reranker
# shows in inspect output is what podman *expands* CDI to; passing the raw
# nodes directly skips the createContainer hook that injects libcuda.so.1,
# which makes the container die with "cannot open libcuda.so.1".
LLAMA_GPU = "nvidia.com/gpu=0"  # GPU 0 = RTX 3060 (~5 GB free)
# Per-tag GGUF + serving recipe. Only models with a llama-server entry are
# routed through the llama_server backend; the rest use the in-process local
# backend with sentence-transformers.
LLAMA_SERVE: dict[str, dict[str, str]] = {
    "nomic-v1.5": {
        # Path INSIDE the container — ``/models`` is the bind-mount target.
        "gguf_in_container": "/models/nomic-embed-text-v1.5.Q8_0.gguf",
        # Path on host for the existence check before we launch the container.
        "gguf_host": f"{LLAMA_MODELS_DIR}/nomic-embed-text-v1.5.Q8_0.gguf",
        "model_name": "nomic-embed-text-v1.5.Q8_0.gguf",
        "tokenizer": "nomic-ai/nomic-embed-text-v1.5",
        # --ubatch-size 2048 mandatory: the server's default 512 makes any
        # chunk >512 tokens return HTTP 500 (does NOT silently truncate).
        # --ctx-size 2048 matches n_ctx_train; nomic's 8K needs rope-scaling
        # that doesn't auto-apply. --pooling mean per the model card.
        "extra_flags": "--embeddings --pooling mean --ctx-size 2048 --ubatch-size 2048 -ngl 99",
    },
}

# tag -> HF model id. Verdict reached (MEMORY.md decision_nomic_v1_5): the
# winning model is nomic-v1.5. The bake-off roster lived here for the
# multi-arm POC; to reproduce, pass --legacy-roster.
ROSTER: list[tuple[str, str]] = [
    ("nomic-v1.5", "nomic-ai/nomic-embed-text-v1.5"),
]

LEGACY_ROSTER: list[tuple[str, str]] = [
    ("e5", "intfloat/e5-base-v2"),                              # baseline
    ("coderank", "nomic-ai/CodeRankEmbed"),
    ("jina", "jinaai/jina-embeddings-v2-base-code"),
    ("gte-modernbert", "Alibaba-NLP/gte-modernbert-base"),
    ("granite-r2", "ibm-granite/granite-embedding-english-r2"),
    ("nomic-v1.5", "nomic-ai/nomic-embed-text-v1.5"),
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
        # A transient slow status response must NOT abort the whole model —
        # retry the poll on timeout/transport errors rather than propagating.
        try:
            s = client.get(f"{BASE}/index/{job_id}/status", timeout=30).json()
        except httpx.HTTPError:
            time.sleep(2)
            continue
        st = s.get("status")
        if st in ("done", "failed", "cancelled", "interrupted"):
            return st, time.time() - t0
        time.sleep(1)


def run_capture(args_list: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(args_list, cwd=REPO_ROOT, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def wait_llama_health(timeout: float = 180.0) -> None:
    """Poll llama-server's /health until READY (model loaded)."""
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout:
        try:
            r = httpx.get(f"{LLAMA_BASE}/health", timeout=5)
            if r.status_code == 200:
                # llama-server reports {"status":"ok"} when ready, or
                # {"status":"loading model"} during boot.
                body = r.json()
                if isinstance(body, dict) and body.get("status") == "ok":
                    return
                last = str(body)[:200]
            else:
                last = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            last = str(e)
        time.sleep(2)
    raise RuntimeError(f"llama-server health timeout after {timeout}s (last: {last})")


def probe_llama_dim(model_name: str) -> int:
    """One-shot /v1/embeddings probe; return the vector length."""
    payload = {"model": model_name, "input": "search_document: ping"}
    r = httpx.post(f"{LLAMA_BASE}/v1/embeddings", json=payload, timeout=30)
    r.raise_for_status()
    body = r.json()
    vec = body["data"][0]["embedding"]
    return len(vec)


def _podman_rm(name: str) -> None:
    """Best-effort kill+remove of a podman container by name. Idempotent."""
    subprocess.run(["podman", "rm", "-f", name],
                   capture_output=True, text=True, timeout=30)


def launch_llama_server(tag: str, mdir: Path) -> str:
    """Start the podman llama-server container for ``tag``; return container id.

    The container is launched detached (``-d``); its logs stream into
    ``mdir/llama.log`` via ``podman logs -f`` in a side process. We return the
    container name (used as a handle for teardown) — the side process is
    started here too and torn down in :func:`run_model`'s finally block.
    """
    recipe = LLAMA_SERVE[tag]
    gguf_host = recipe["gguf_host"]
    if not Path(gguf_host).is_file():
        raise RuntimeError(f"GGUF not found on host: {gguf_host}")
    # Wipe any stale container from a prior aborted run before reusing the name.
    _podman_rm(LLAMA_CONTAINER)

    cmd = [
        "podman", "run", "-d", "--rm",
        "--name", LLAMA_CONTAINER,
        "--device", LLAMA_GPU,
        "-v", f"{LLAMA_MODELS_DIR}:/models:ro",
        "-p", f"127.0.0.1:{LLAMA_PORT}:{LLAMA_PORT}",
        LLAMA_IMAGE,
        "--model", recipe["gguf_in_container"],
        *recipe["extra_flags"].split(),
        # Bind 0.0.0.0 inside the container; podman maps it to host loopback.
        "--host", "0.0.0.0", "--port", str(LLAMA_PORT),
    ]

    print(f"[{tag}] starting llama-server container on port {LLAMA_PORT}",
          flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(
            f"podman run failed (rc={r.returncode}): {r.stderr.strip()[:500]}"
        )
    # Tail container logs into the per-model log file for forensics.
    log = open(mdir / "llama.log", "w")
    subprocess.Popen(["podman", "logs", "-f", LLAMA_CONTAINER],
                     stdout=log, stderr=subprocess.STDOUT)

    try:
        wait_llama_health()
    except Exception:
        _podman_rm(LLAMA_CONTAINER)
        log.close()
        raise
    dim = probe_llama_dim(recipe["model_name"])
    if dim != 768:
        _podman_rm(LLAMA_CONTAINER)
        log.close()
        raise RuntimeError(f"llama-server dim probe = {dim}, expected 768")
    print(f"[{tag}] llama-server healthy, dim=768", flush=True)
    return LLAMA_CONTAINER


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

    llama_handle: str | None = None
    use_llama = tag in LLAMA_SERVE

    env = dict(os.environ)
    if use_llama:
        llama_handle = launch_llama_server(tag, mdir)
        recipe = LLAMA_SERVE[tag]
        env.update({
            "EMBEDDER_BACKEND": "llama_server",
            "LLAMA_SERVER_URL": LLAMA_BASE,
            "LLAMA_SERVER_MODEL": recipe["model_name"],
            "LLAMA_SERVER_TOKENIZER": recipe["tokenizer"],
            "LLAMA_SERVER_MAX_TOKENS": "2048",
            "LLAMA_SERVER_BATCH_SIZE": "16",
            # No EMBED_DEVICE / CUDA_* — the indexer subprocess just makes
            # HTTP calls; the GPU work lives in llama-server.
            "LADYBUG_DB_DIR": str(idx_dir / "repos"),
            "JOBS_DB_PATH": str(idx_dir / "jobs.sqlite"),
            "S3_INDEX_BUCKET": "",
            "WATCH_ENABLED": "false",
            "HOST": "127.0.0.1",
            "PORT": str(PORT),
        })
    else:
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
        if llama_handle is not None:
            _podman_rm(llama_handle)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="comma-separated tags to run")
    ap.add_argument("--outdir", default=str(REPO_ROOT / ".planning" / "runs" / "768-poc"))
    ap.add_argument(
        "--legacy-roster",
        action="store_true",
        help="run the pre-verdict bake-off roster (e5, coderank, jina, gte-modernbert, granite-r2, nomic-v1.5)",
    )
    ap.add_argument(
        "--clean",
        action="store_true",
        help="rmtree .cgr-poc/ and --outdir before running (POC artifacts are not committed)",
    )
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    outdir = Path(a.outdir)
    if a.clean:
        import shutil
        shutil.rmtree(REPO_ROOT / ".cgr-poc", ignore_errors=True)
        shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True, exist_ok=True)
    roster = LEGACY_ROSTER if a.legacy_roster else ROSTER
    for tag, hf in roster:
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
