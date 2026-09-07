"""Boundary Lab's bundled BEAT paths and hardware-environment configuration.

Application policy stays here. beat_worker contains the independently reusable
subprocess client. All application adapters share the same worker pool below.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from blab.rocm import discover_rocm
from blab.solvers.beat_worker import (
    WorkerPool,
    WorkerProcess,
    format_julia_error,
)
from blab.solvers.beat_worker import (
    julia_command as julia_command,
)
from blab.solvers.beat_worker import (
    julia_worker_command as julia_worker_command,
)
from blab.solvers.beat_worker import (
    resolve_julia_threads as resolve_julia_threads,
)

DEFAULT_BEAT_ENGINE_SOLVER_SCRIPT = Path(__file__).with_name("julia_local") / "solver.jl"
DEFAULT_BEAT_ENGINE_CPU_PROJECT = DEFAULT_BEAT_ENGINE_SOLVER_SCRIPT.parent
DEFAULT_BEAT_ENGINE_CUDA_PROJECT = Path(__file__).with_name("julia_cuda")
DEFAULT_BEAT_ENGINE_ROCM_PROJECT = Path(__file__).with_name("julia_rocm")
DEFAULT_BEAT_ENGINE_PROJECT = DEFAULT_BEAT_ENGINE_CPU_PROJECT
DEFAULT_BEAT_ENGINE_SYSTEM_SOLVER_SCRIPT = DEFAULT_BEAT_ENGINE_CPU_PROJECT / "coupled_solver.jl"
BEAT_ENGINE_CUDA_BACKEND = "cuda"
BEAT_ENGINE_CPU_BACKEND = "cpu"
BEAT_ENGINE_ROCM_BACKEND = "rocm"
BEAT_ENGINE_BACKENDS = {
    BEAT_ENGINE_CUDA_BACKEND,
    BEAT_ENGINE_CPU_BACKEND,
    BEAT_ENGINE_ROCM_BACKEND,
}


def normalize_beat_engine_backend(value: object) -> str:
    text = str(value or BEAT_ENGINE_CUDA_BACKEND).strip().lower()
    aliases = {
        "beat_cuda": BEAT_ENGINE_CUDA_BACKEND,
        "cuda": BEAT_ENGINE_CUDA_BACKEND,
        "gpu": BEAT_ENGINE_CUDA_BACKEND,
        "julia_local": BEAT_ENGINE_CUDA_BACKEND,
        "local_julia": BEAT_ENGINE_CUDA_BACKEND,
        "beat_cpu": BEAT_ENGINE_CPU_BACKEND,
        "cpu": BEAT_ENGINE_CPU_BACKEND,
        "beat_rocm": BEAT_ENGINE_ROCM_BACKEND,
        "rocm": BEAT_ENGINE_ROCM_BACKEND,
        "amd": BEAT_ENGINE_ROCM_BACKEND,
        "amdgpu": BEAT_ENGINE_ROCM_BACKEND,
    }
    backend = aliases.get(text, text)
    if backend not in BEAT_ENGINE_BACKENDS:
        raise ValueError(f"Unknown BEAT Engine backend: {value}")
    return backend


def default_beat_engine_project(beat_engine_backend: str) -> Path:
    if beat_engine_backend == BEAT_ENGINE_CPU_BACKEND:
        return DEFAULT_BEAT_ENGINE_CPU_PROJECT
    if beat_engine_backend == BEAT_ENGINE_ROCM_BACKEND:
        return DEFAULT_BEAT_ENGINE_ROCM_PROJECT
    return DEFAULT_BEAT_ENGINE_CUDA_PROJECT


def _julia_project_backend_label(project_path: Path, beat_engine_backend: str | None) -> str:
    if beat_engine_backend == BEAT_ENGINE_CUDA_BACKEND or project_path == DEFAULT_BEAT_ENGINE_CUDA_PROJECT:
        return "BEAT Engine (Nvidia CUDA)"
    if beat_engine_backend == BEAT_ENGINE_CPU_BACKEND or project_path == DEFAULT_BEAT_ENGINE_CPU_PROJECT:
        return "BEAT Engine (CPU)"
    if beat_engine_backend == BEAT_ENGINE_ROCM_BACKEND or project_path == DEFAULT_BEAT_ENGINE_ROCM_PROJECT:
        return "BEAT Engine (AMD ROCm)"
    return "the selected BEAT Engine backend"


def julia_process_env(
    julia_threads: str | int = "auto",
    julia_project: str | Path | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env["JULIA_NUM_THREADS"] = resolve_julia_threads(julia_threads)
    if julia_project is not None:
        try:
            is_rocm_project = Path(julia_project).resolve() == DEFAULT_BEAT_ENGINE_ROCM_PROJECT.resolve()
        except OSError:
            is_rocm_project = False
        if is_rocm_project:
            env["BLAB_BEAT_ENGINE_GPU_BACKEND"] = BEAT_ENGINE_ROCM_BACKEND
            installation = discover_rocm(environ=env)
            if installation is not None:
                rocm_root = str(installation.root)
                env["BLAB_ROCM_PATH"] = rocm_root
                env["ROCM_PATH"] = rocm_root
                env["ROCM_HOME"] = rocm_root
                env["HIP_PATH"] = rocm_root
                rocm_bin = str(installation.root / "bin")
                path_entries = env.get("PATH", "").split(os.pathsep)
                if os.path.normcase(rocm_bin) not in {os.path.normcase(entry) for entry in path_entries}:
                    env["PATH"] = rocm_bin + os.pathsep + env.get("PATH", "")
    if julia_project is not None:
        try:
            is_cuda_project = Path(julia_project).resolve() == DEFAULT_BEAT_ENGINE_CUDA_PROJECT.resolve()
        except OSError:
            is_cuda_project = False
        if is_cuda_project:
            env["BLAB_BEAT_ENGINE_GPU_BACKEND"] = BEAT_ENGINE_CUDA_BACKEND
    return env


def friendly_julia_error(
    message: str,
    *,
    julia_project: str | Path | None,
    beat_engine_backend: str | None = None,
    detection_text: str | None = None,
) -> str:
    label = (
        _julia_project_backend_label(Path(julia_project), beat_engine_backend)
        if julia_project is not None
        else "the selected BEAT Engine backend"
    )
    return format_julia_error(message, julia_project=julia_project, backend_label=label, detection_text=detection_text)


class BeatEngineWorkerProcess(WorkerProcess):
    """Worker configured for Boundary Lab's bundled Julia environments."""

    def __init__(
        self,
        *,
        julia_executable: str,
        solver_script: Path,
        julia_threads: str | int,
        julia_project: Path | None,
        julia_sysimage: Path | None = None,
        environment: Mapping[str, str] | None = None,
        backend_label: str | None = None,
    ):
        super().__init__(
            julia_executable=julia_executable,
            solver_script=solver_script,
            julia_threads=julia_threads,
            julia_project=julia_project,
            julia_sysimage=julia_sysimage,
            environment=julia_process_env(julia_threads, julia_project) if environment is None else environment,
            backend_label=backend_label
            or (
                _julia_project_backend_label(julia_project, None)
                if julia_project is not None
                else "the selected BEAT Engine backend"
            ),
        )


_WORKER_POOL = WorkerPool(BeatEngineWorkerProcess)


def get_beat_engine_worker(
    *,
    julia_executable: str,
    solver_script: Path,
    julia_threads: str | int,
    julia_project: Path | None,
    julia_sysimage: Path | None = None,
) -> WorkerProcess:
    return _WORKER_POOL.get_worker(
        julia_executable=julia_executable,
        solver_script=solver_script,
        julia_threads=julia_threads,
        julia_project=julia_project,
        julia_sysimage=julia_sysimage,
        environment=julia_process_env(julia_threads, julia_project),
        backend_label=_julia_project_backend_label(julia_project, None)
        if julia_project is not None
        else "the selected BEAT Engine backend",
    )


def shutdown_beat_engine_workers() -> None:
    _WORKER_POOL.shutdown()
