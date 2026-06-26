#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "" ]]; then
    exec "$@"
fi

mkdir -p "${BLAB_ARTIFACT_DIR}"

args=(
    blab server
    --host "${BLAB_SERVER_HOST}"
    --port "${BLAB_SERVER_PORT}"
    --solver "${BLAB_SERVER_SOLVER}"
    --julia-executable "${BLAB_JULIA_EXECUTABLE}"
    --julia-threads "${BLAB_JULIA_THREADS}"
    --warm-solver "${BLAB_WARM_SOLVER}"
    --max-running-jobs "${BLAB_MAX_RUNNING_JOBS}"
    --log-level "${BLAB_LOG_LEVEL}"
    --artifact-dir "${BLAB_ARTIFACT_DIR}"
)

if [[ -n "${BLAB_JULIA_SYSIMAGE:-}" && -f "${BLAB_JULIA_SYSIMAGE}" ]]; then
    args+=(--julia-sysimage "${BLAB_JULIA_SYSIMAGE}")
elif [[ -n "${BLAB_JULIA_SYSIMAGE:-}" ]]; then
    echo "Julia sysimage not found at ${BLAB_JULIA_SYSIMAGE}; starting without --julia-sysimage." >&2
fi

exec "${args[@]}"
