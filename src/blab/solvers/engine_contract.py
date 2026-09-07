"""Application-facing import boundary for the selected engine contract."""

from blab.solvers.engine_distribution import ENGINE_DISTRIBUTION

if ENGINE_DISTRIBUTION == "external":
    from beat_engine.beat_contract import (
        COMPILED_SYSTEM_VERSION,
        SUPPORTED_SYSTEM_RESULT_VERSIONS,
        SYSTEM_RESULT_VERSION,
        SYSTEM_SOLVE_REQUEST_VERSION,
        validate_compiled_system,
        validate_solve_request,
    )
    from beat_engine.beat_contract.worker import (
        WorkerCompatibilityError,
        negotiate_submission,
        validate_worker_ready,
    )
else:
    from blab.solvers.beat_contract import (
        COMPILED_SYSTEM_VERSION,
        SUPPORTED_SYSTEM_RESULT_VERSIONS,
        SYSTEM_RESULT_VERSION,
        SYSTEM_SOLVE_REQUEST_VERSION,
        validate_compiled_system,
        validate_solve_request,
    )
    from blab.solvers.beat_contract.worker import (
        WorkerCompatibilityError,
        negotiate_submission,
        validate_worker_ready,
    )

__all__ = [
    "WorkerCompatibilityError",
    "negotiate_submission",
    "validate_worker_ready",
    "COMPILED_SYSTEM_VERSION",
    "SUPPORTED_SYSTEM_RESULT_VERSIONS",
    "SYSTEM_RESULT_VERSION",
    "SYSTEM_SOLVE_REQUEST_VERSION",
    "validate_compiled_system",
    "validate_solve_request",
]
