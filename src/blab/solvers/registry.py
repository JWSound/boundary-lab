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
        description="Use the Boundary Lab bempp-cl solve server.",
    ),
    "julia_local": SolverBackendInfo(
        backend_id="julia_local",
        label="Julia CUDA GPU",
        capabilities=SolverCapabilities(
            supports_remote_assets=False,
            supports_parallel_workers=False,
            is_remote=False,
        ),
        factory=lambda **kwargs: _create_julia_local_backend(**kwargs),
        description="Run the local Julia solver through the Boundary Lab subprocess adapter.",
    ),
    "local": SolverBackendInfo(
        backend_id="local",
        label="Bempp OpenCL CPU",
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
        "bempp_local": "local",
        "bempp_server": "server",
        "local_bempp": "local",
        "local_bempp_cl": "local",
        "local_julia": "julia_local",
    }
    return aliases.get(text, text or "local")


def backend_label_to_id() -> dict[str, str]:
    return {info.label: info.backend_id for info in available_backend_infos()}


def _create_bempp_local_backend() -> SolverBackend:
    from blab.solvers.bempp_local import BemppLocalBackend

    return BemppLocalBackend()


def _create_bempp_server_backend(*, server_url: str = "http://127.0.0.1:8765", **_kwargs: Any) -> SolverBackend:
    from blab.solvers.bempp_server import BemppServerBackend

    return BemppServerBackend(server_url)


def _create_julia_local_backend(
    *,
    julia_executable: str = "julia",
    solver_script: str | None = None,
    julia_threads: str | int = "auto",
    julia_project: str | None = "__default__",
    persistent_worker: bool = True,
    **_kwargs: Any,
) -> SolverBackend:
    from blab.solvers.julia_local_backend import JuliaLocalBackend

    kwargs: dict[str, Any] = {
        "julia_executable": julia_executable,
        "julia_threads": julia_threads,
        "persistent_worker": persistent_worker,
    }
    if solver_script:
        kwargs["solver_script"] = solver_script
    if julia_project != "__default__":
        kwargs["julia_project"] = julia_project
    return JuliaLocalBackend(**kwargs)
