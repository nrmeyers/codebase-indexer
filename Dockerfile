# syntax=docker/dockerfile:1.7
#
# Code Indexer Service — production image.
# Per .planning/phase-plans/PHASE_3_DOCKER.md §5.
#
# Build context expectation: the PARENT directory of this repo. The CI
# workflow at TheForge/.github/workflows/build-and-push.yml checks out
# both `code-indexer-service` and `code-graph-rag` as siblings and runs
# `docker build -f code-indexer-service/Dockerfile .` from that parent
# so this Dockerfile can `COPY code-graph-rag` to satisfy the
# [tool.uv.sources] path dep declared in pyproject.toml.
#
# Local equivalent (from your home dir):
#   cd ~ && docker build -t forge-code-indexer -f code-indexer-service/Dockerfile .
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

WORKDIR /workspace

# code-graph-rag is the path-dep sibling (per pyproject.toml's
# [tool.uv.sources]: `code-graph-rag = { path = "../code-graph-rag" }`).
# Copy it FIRST so changes to the indexer's pyproject.toml don't bust
# the cgr install layer.
#
# Build-context expectation: the build context must contain a
# `code-graph-rag/` directory at its root. The CI workflow at
# TheForge/.github/workflows/build-and-push.yml is responsible for
# ensuring the cgr fork is checked out to that path — see the companion
# TheForge PR that fixes the workflow's `path:` setting from
# `vitali87-code-graph-rag` to `code-graph-rag`.
COPY code-graph-rag ./code-graph-rag

# Indexer manifest + lockfile only — `uv sync --no-install-project`
# resolves deps without copying app source so subsequent app-only
# changes hit a warm dep cache.
COPY code-indexer-service/pyproject.toml code-indexer-service/uv.lock ./code-indexer-service/

WORKDIR /workspace/code-indexer-service

# Install deps WITH the [arrow] extra. Bench results
# (code-graph-rag/scripts/BENCH_RESULTS_2026-04-27.md) showed pyarrow
# bulk_insert is 380x faster than executemany on FLOAT[768] payloads;
# the ~30 MB image cost is trivially worth the indexing throughput.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra arrow

# Now copy app source and re-sync to install code-indexer-service itself.
COPY code-indexer-service/app ./app
COPY code-indexer-service/main.py ./main.py
COPY code-indexer-service/scripts ./scripts

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra arrow

# ---------- Stage 2: runtime ----------
FROM python:3.12-slim AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl tini libssl3 zlib1g libzstd1 ripgrep && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --uid 10001 forge

WORKDIR /app

# Copy the venv first (heavy, changes infrequently) then app source
# (light, changes often) — keeps the runtime image rebuild fast on
# code-only changes.
COPY --from=builder --chown=forge:forge /workspace/code-indexer-service/.venv /app/.venv
COPY --from=builder --chown=forge:forge /workspace/code-indexer-service/app /app/app
COPY --from=builder --chown=forge:forge /workspace/code-indexer-service/scripts /app/scripts
COPY --from=builder --chown=forge:forge /workspace/code-indexer-service/main.py /app/main.py
# code-graph-rag is editable-installed into the venv; keeping its source
# tree at the path uv recorded means import resolves at runtime.
COPY --from=builder --chown=forge:forge /workspace/code-graph-rag /code-graph-rag

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    LM_STUDIO_URL=http://host.docker.internal:1234 \
    CGR_DATA_DIR=/var/lib/forge/cgr \
    JOBS_DB_PATH=/var/lib/forge/jobs/jobs.sqlite

USER forge
EXPOSE 8000

# Long start_period because the first request loads the embedder weights
# and pyarrow lazy imports — both can take ~20-30 s on a cold container.
HEALTHCHECK --interval=15s --timeout=3s --start-period=45s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
