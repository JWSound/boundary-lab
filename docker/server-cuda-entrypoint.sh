#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "" ]]; then
    exec "$@"
fi

mkdir -p "${BLAB_ARTIFACT_DIR}"

exec blab server \
    --host "${BLAB_SERVER_HOST}" \
    --port "${BLAB_SERVER_PORT}" \
    --solver "${BLAB_SERVER_SOLVER}" \
    --julia-executable "${BLAB_JULIA_EXECUTABLE}" \
    --julia-threads "${BLAB_JULIA_THREADS}" \
    --max-running-jobs "${BLAB_MAX_RUNNING_JOBS}" \
    --log-level "${BLAB_LOG_LEVEL}" \
    --artifact-dir "${BLAB_ARTIFACT_DIR}"
