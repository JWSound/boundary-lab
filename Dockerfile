# syntax=docker/dockerfile:1
ARG JULIA_IMAGE=julia:1.12.6-bookworm@sha256:0d9424d38430320596424b22a7d03add630ea24313e139eaff5b521093f74f17
FROM ${JULIA_IMAGE} AS base
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
    JULIA_DEPOT_PATH=/opt/julia-depot JULIA_CPU_TARGET=generic \
    JULIA_NUM_PRECOMPILE_TASKS=2 JULIA_NUM_THREADS=2 \
    BLAB_SERVER_MODE=hosted BLAB_SERVER_ROOT=/data/jobs BLAB_SERVER_PORT=8765
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv ca-certificates libglu1-mesa tini \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && useradd --create-home --uid 10001 blab \
    && mkdir -p /data/jobs /opt/julia-depot \
    && chown -R blab:blab /data /opt/julia-depot
WORKDIR /opt/boundary-lab
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . && pip freeze > /opt/python-packages.txt
RUN python -m beat_engine instantiate --backend cpu \
    && python -m beat_engine doctor --backend cpu > /opt/beat-cpu.json \
    && chown -R blab:blab /opt/julia-depot
COPY docker /opt/blab-container
ARG VCS_REF=local
LABEL org.opencontainers.image.title="Boundary Lab Server" \
      org.opencontainers.image.source="https://github.com/JWSound/boundary-lab" \
      org.opencontainers.image.revision=${VCS_REF}
EXPOSE 8765
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=15s --timeout=5s --start-period=150s --retries=3 \
    CMD ["python", "/opt/blab-container/healthcheck.py"]
ENTRYPOINT ["/usr/bin/tini", "--", "python", "/opt/blab-container/entrypoint.py"]

FROM base AS cpu
USER blab

FROM base AS cuda
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN python /opt/blab-container/prepare_cuda.py \
    && chown -R blab:blab /opt/julia-depot
USER blab
