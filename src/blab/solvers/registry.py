"""Solver backend registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from blab.solvers.base import SolverBackend, SolverCapabilities


@dataclass(frozen=True)
class SolverBackendInfo:
    backend_id: str
    label: str
    capabilities: SolverCapabilities
    factory: Callable[..., SolverBackend] | None = None
    available: bool = True
    description: str = ""


_BACKENDS: dict[str, SolverBackendInfo] = {
    "server": SolverBackendInfo(
        backend_id="server",
        label="Server",
        capabilities=SolverCapabilities(
            supports_remote_assets=True,
            supports_parallel_workers=True,
            is_remote=True,
        ),
        factory=lambda **kwargs: _create_bempp_server_backend(**kwargs),
        description="Use a Boundary Lab HTTP solve server.",
    ),
    "beat_cuda": SolverBackendInfo(
        backend_id="beat_cuda",
        label="BEAT Engine (CUDA)",
        capabilities=SolverCapabilities(
            supports_remote_assets=False,
            supports_parallel_workers=False,
            supports_symmetry=True,
            supports_channel_resynthesis=True,
            is_remote=False,
        ),
        factory=lambda **kwargs: _create_beat_engine_backend(beat_engine_backend="cuda", **kwargs),
        description="Run the local Boundary Element Acoustic Toolkit Engine CUDA solver through the Boundary Lab subprocess adapter.",
    ),
    "beat_cpu": SolverBackendInfo(
        backend_id="beat_cpu",
        label="BEAT Engine (CPU)",
        capabilities=SolverCapabilities(
            supports_remote_assets=False,
            supports_parallel_workers=False,
            supports_symmetry=True,
            supports_channel_resynthesis=True,
            is_remote=False,
        ),
        factory=lambda **kwargs: _create_beat_engine_backend(beat_engine_backend="cpu", **kwargs),
        description="Run the local Boundary Element Acoustic Toolkit Engine CPU solver through the Boundary Lab subprocess adapter.",
    ),
    "beat_rocm": SolverBackendInfo(
        backend_id="beat_rocm",
        label="BEAT Engine (ROCm)",
        capabilities=SolverCapabilities(
            supports_remote_assets=False,
            supports_parallel_workers=False,
            supports_symmetry=True,
            supports_channel_resynthesis=True,
            is_remote=False,
        ),
        factory=lambda **kwargs: _create_beat_engine_backend(beat_engine_backend="rocm", **kwargs),
        description="Run the local Boundary Element Acoustic Toolkit Engine ROCm solver through the Boundary Lab subprocess adapter.",
    ),
    "local": SolverBackendInfo(
        backend_id="local",
        label="Bempp (OpenCL CPU)",
        capabilities=SolverCapabilities(
            supports_remote_assets=False,
            supports_parallel_workers=True,
            is_remote=False,
        ),
        factory=lambda **_kwargs: _create_bempp_local_backend(),
        description="Run the bundled bempp-cl OpenCL CPU solver in the GUI process.",
    ),
}


def available_backend_infos() -> tuple[SolverBackendInfo, ...]:
    return tuple(info for info in _BACKENDS.values() if info.available)


def backend_info(backend_id: str) -> SolverBackendInfo:
    normalized_id = normalize_backend_id(backend_id)
    try:
        return _BACKENDS[normalized_id]
    except KeyError as exc:
        raise ValueError(f"Unknown solver backend: {backend_id}") from exc


def create_backend(backend_id: str, **kwargs: Any) -> SolverBackend:
    info = backend_info(backend_id)
    if info.factory is None:
        raise ValueError(f"Solver backend '{info.label}' is not available through the local backend factory.")
    return info.factory(**kwargs)


def normalize_backend_id(backend_id: str) -> str:
    text = str(backend_id or "").strip()
    aliases = {
        "bempp": "local",
        "bempp_cpu": "local",
        "bempp_local": "local",
        "bempp_server": "server",
        "http_server": "server",
        "local_bempp": "local",
        "local_bempp_cl": "local",
        "julia_local": "beat_cuda",
        "local_julia": "beat_cuda",
        "beat": "beat_cuda",
        "beat_engine": "beat_cuda",
        "beat_cuda": "beat_cuda",
        "beat_gpu": "beat_cuda",
        "cuda": "beat_cuda",
        "beat_cpu": "beat_cpu",
        "cpu_beat": "beat_cpu",
        "beat_rocm": "beat_rocm",
        "rocm": "beat_rocm",
        "amd": "beat_rocm",
        "amdgpu": "beat_rocm",
    }
    return aliases.get(text, text or "local")


def backend_label_to_id() -> dict[str, str]:
    return {info.label: info.backend_id for info in available_backend_infos()}


def _create_bempp_local_backend() -> SolverBackend:
    from blab.solvers.bempp_local import BemppLocalBackend

    return BemppLocalBackend()


def _create_bempp_server_backend(*, server_url: str = "http://127.0.0.1:8765", **_kwargs: Any) -> SolverBackend:
    return _create_http_server_backend(server_url=server_url)


def _create_http_server_backend(*, server_url: str = "http://127.0.0.1:8765", **_kwargs: Any) -> SolverBackend:
    from blab.solvers.http_server import HttpServerBackend

    return HttpServerBackend(server_url)


def _create_beat_engine_backend(
    *,
    julia_executable: str = "julia",
    solver_script: str | None = None,
    julia_threads: str | int = "auto",
    julia_project: str | None = "__default__",
    julia_sysimage: str | None = None,
    persistent_worker: bool = True,
    beat_engine_backend: str = "cuda",
    **_kwargs: Any,
) -> SolverBackend:
    from blab.solvers.beat_engine_backend import (
        DEFAULT_BEAT_ENGINE_CPU_PROJECT,
        DEFAULT_BEAT_ENGINE_CUDA_PROJECT,
        DEFAULT_BEAT_ENGINE_ROCM_PROJECT,
        BeatEngineBackend,
    )

    normalized_backend = {
        "cpu": "cpu",
        "rocm": "rocm",
        "beat_rocm": "rocm",
    }.get(str(beat_engine_backend).strip().lower(), "cuda")
    backend_id = f"beat_{normalized_backend}"
    label = {
        "cpu": "BEAT Engine (CPU)",
        "cuda": "BEAT Engine (CUDA)",
        "rocm": "BEAT Engine (ROCm)",
    }[normalized_backend]
    default_project = {
        "cpu": DEFAULT_BEAT_ENGINE_CPU_PROJECT,
        "cuda": DEFAULT_BEAT_ENGINE_CUDA_PROJECT,
        "rocm": DEFAULT_BEAT_ENGINE_ROCM_PROJECT,
    }[normalized_backend]
    kwargs: dict[str, Any] = {
        "julia_executable": julia_executable,
        "julia_threads": julia_threads,
        "julia_project": default_project,
        "julia_sysimage": julia_sysimage,
        "persistent_worker": persistent_worker,
        "backend_id": backend_id,
        "label": label,
        "beat_engine_backend": normalized_backend,
    }
    if solver_script:
        kwargs["solver_script"] = solver_script
    if julia_project != "__default__":
        kwargs["julia_project"] = julia_project
    return BeatEngineBackend(**kwargs)


_create_julia_local_backend = _create_beat_engine_backend
