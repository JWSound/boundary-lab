"""Compatibility imports for the renamed Afterburner backend."""

from blab.solvers.afterburner_backend import (
    DEFAULT_AFTERBURNER_PROJECT,
    DEFAULT_AFTERBURNER_SOLVER_SCRIPT,
    DEFAULT_JULIA_PROJECT,
    DEFAULT_JULIA_SOLVER_SCRIPT,
    AfterburnerBackend,
    AfterburnerSession,
    AfterburnerWorkerProcess,
    JuliaLocalBackend,
    JuliaLocalSession,
    JuliaWorkerProcess,
    _resolve_julia_threads,
    shutdown_afterburner_workers,
    shutdown_julia_workers,
)
