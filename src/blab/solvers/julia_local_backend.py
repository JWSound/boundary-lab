"""Compatibility imports for the BEAT Engine backend."""

from blab.solvers.beat_engine_backend import (
    DEFAULT_BEAT_ENGINE_PROJECT,
    DEFAULT_BEAT_ENGINE_SOLVER_SCRIPT,
    DEFAULT_JULIA_PROJECT,
    DEFAULT_JULIA_SOLVER_SCRIPT,
    BeatEngineBackend,
    BeatEngineCpuBackend,
    BeatEngineCudaBackend,
    BeatEngineSession,
    BeatEngineWorkerProcess,
    JuliaLocalBackend,
    JuliaLocalSession,
    JuliaWorkerProcess,
    _resolve_julia_threads,
    shutdown_beat_engine_workers,
    shutdown_julia_workers,
)
