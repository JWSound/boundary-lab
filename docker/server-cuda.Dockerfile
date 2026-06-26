FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG JULIA_CHANNEL=release
ARG BLAB_BUILD_SYSIMAGE=1

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    JULIAUP_HOME=/opt/juliaup \
    JULIA_DEPOT_PATH=/opt/julia-depot \
    PATH=/opt/juliaup/bin:/app/.venv/bin:$PATH \
    BLAB_SERVER_HOST=0.0.0.0 \
    BLAB_SERVER_PORT=8765 \
    BLAB_SERVER_SOLVER=beat_cuda \
    BLAB_JULIA_EXECUTABLE=/opt/juliaup/bin/julia \
    BLAB_JULIA_THREADS=auto \
    BLAB_JULIA_SYSIMAGE=/app/blab-beat-cuda.so \
    BLAB_JULIA_CPU_TARGET=generic,+aes \
    BLAB_WARM_SOLVER=off \
    BLAB_MAX_RUNNING_JOBS=1 \
    BLAB_LOG_LEVEL=INFO \
    BLAB_ARTIFACT_DIR=/data/server_jobs

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        g++ \
        gcc \
        git \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://install.julialang.org | sh -s -- -y --path /opt/juliaup --default-channel "${JULIA_CHANNEL}" \
    && chmod -R a+rX /opt/juliaup

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY docker ./docker

RUN python3 -m venv /app/.venv \
    && pip install --upgrade pip setuptools wheel \
    && pip install -e .

RUN mkdir -p /opt/julia-depot /data/server_jobs \
    && julia --project=/app/src/blab/solvers/julia_cuda --startup-file=no -e 'using Pkg; Pkg.instantiate(); using CUDA; CUDA.set_runtime_version!(v"12.6")' \
    && julia --project=/app/src/blab/solvers/julia_cuda --startup-file=no -e 'using Pkg; Pkg.precompile(); using CUDA; CUDA.precompile_runtime()' \
    && if [ "${BLAB_BUILD_SYSIMAGE}" = "1" ]; then julia --startup-file=no /app/docker/build-beat-cuda-sysimage.jl; else echo "Skipping Julia sysimage build."; fi \
    && chmod -R a+rX /opt/julia-depot \
    && if [ -f /app/blab-beat-cuda.so ]; then chmod a+rX /app/blab-beat-cuda.so; fi

RUN cp /app/docker/server-cuda-entrypoint.sh /usr/local/bin/blab-server-cuda \
    && chmod +x /usr/local/bin/blab-server-cuda

VOLUME ["/data"]
EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=5m --retries=3 \
    CMD python3 -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"BLAB_SERVER_PORT\", \"8765\")}/health', timeout=3).read()" || exit 1

ENTRYPOINT ["blab-server-cuda"]
