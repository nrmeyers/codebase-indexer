# syntax=docker/dockerfile:1.7
#
# Code Indexer Service — production image.
#
# Build context: THIS repo's root. The graph engine is vendored at
# `codebase_rag/` (see codebase_rag/VENDORED.md) and its runtime deps are
# merged into pyproject.toml, so there is no longer a `code-graph-rag`
# sibling to COPY and no `[tool.uv.sources]` path dep.
#
#   podman build -t codebase-indexer .        # from the repo root
#   docker build -t codebase-indexer .        # equivalent
#
# Runtime container behaviour:
#   - Listens on 0.0.0.0:8000 (matches `code-indexer:8000` in the
#     deploy/forge/docker-compose.yml service-internal DNS).
#   - HEALTHCHECK hits /health every 15 s.
#   - Non-root user `forge:10001`.
#   - tini reaps subprocess children (the embed subprocess in particular).

# ---------- Stage 1: builder ----------
FROM ghcr.io/astral-sh/uv:0.10@sha256:edd1fd89f3e5b005814cc8f777610445d7b7e3ed05361f9ddfae67bebfe8456a AS uv

FROM python:3.12-slim AS builder
COPY --from=uv /uv /uvx /bin/

# Build deps for any wheel that needs to compile (e.g. C extensions).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential cmake git libssl-dev zlib1g-dev libzstd-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Manifest + lockfile only — `uv sync --no-install-project` resolves deps
# without copying app source, so later app-only changes hit a warm dep
# cache. README.md is REQUIRED: pyproject.toml declares `readme = "README.md"`
# and hatchling reads it during the project install.
COPY pyproject.toml uv.lock README.md ./

# Install lockfile-resolved deps (no app source yet, no dev/extras).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# pyarrow is the 380× bulk_insert speedup. The vendored codebase_rag
# vector_store auto-detects it at import time and routes to the
# bulk_insert_arrow path when present, falling back to executemany
# otherwise. It is not a default dependency, so pull it explicitly with
# `--no-deps` to keep the locked resolution above intact.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps "pyarrow>=15.0"

# Now copy app source + the vendored engine, then install the project
# itself (`app` + `codebase_rag` are the two wheel packages).
COPY app ./app
COPY codebase_rag ./codebase_rag
COPY main.py ./main.py
COPY scripts ./scripts

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---------- Stage 2: runtime ----------
FROM python:3.12-slim AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl tini libssl3 zlib1g libzstd1 ripgrep && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --uid 10001 forge

WORKDIR /app

# Copy the whole project dir from the builder in one shot: the venv plus
# the `app` and vendored `codebase_rag` source trees (the project is
# installed editable, so both source trees must live at the path uv
# recorded — /app — for imports to resolve at runtime), along with the
# manifests/scripts. Done as a single COPY because podman/buildah can
# fail to stat individually cherry-picked subdirs from a prior stage
# that ended on a cache-mounted RUN.
COPY --from=builder --chown=forge:forge /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    LM_STUDIO_URL=http://host.docker.internal:1234 \
    CGR_DATA_DIR=/var/lib/forge/cgr \
    LADYBUG_DB_DIR=/var/lib/forge/cgr/repos \
    JOBS_DB_PATH=/var/lib/forge/jobs/jobs.sqlite

# Persist indexes + job state across container restarts by mounting a
# volume at /var/lib/forge (the dirs above all live under it).
RUN mkdir -p /var/lib/forge/cgr/repos /var/lib/forge/jobs && \
    chown -R forge:forge /var/lib/forge
VOLUME ["/var/lib/forge"]

USER forge
EXPOSE 8000

# Long start_period because the first request loads the embedder weights
# (downloaded to the HF cache on cold start) and pyarrow lazy-imports —
# both can take tens of seconds on a fresh container.
#
# NOTE: podman/buildah IGNORE HEALTHCHECK when building in the default OCI
# image format (you'll see a "HEALTHCHECK is not supported for OCI image
# format" warning). To bake it in under podman, build with
# `podman build --format docker .` (or export BUILDAH_FORMAT=docker); docker
# honours it natively. Independently, orchestrators (compose/k8s) should
# health-check `GET /health` directly rather than rely on this directive.
HEALTHCHECK --interval=15s --timeout=3s --start-period=45s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
