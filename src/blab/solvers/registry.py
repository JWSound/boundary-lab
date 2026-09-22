"""Solver backend registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from blab.solvers.base import SolverBackend, SolverCapabilities
from blab.solvers.engine_distribution import backend_catalog, engine_backend_info, engine_paths


@dataclass(frozen=True)
class SolverBackendInfo:
    backend_id: str
    label: str
    capabilities: SolverCapabilities
    factory: Callable[..., SolverBackend] | None = None
    available: bool = True
    description: str = ""


_BACKENDS: dict[str, SolverBackendInfo] = {
    "beat_remote": SolverBackendInfo(
        backend_id="beat_remote",
        label="Boundary Lab Server",
        capabilities=SolverCapabilities(
            supports_remote_assets=True, supports_symmetry=True, supports_channel_resynthesis=True, is_remote=True
        ),
        description="Run the physical system on a Boundary Lab server.",
    ),
    **{
        f"beat_{info.backend_id}": SolverBackendInfo(
            backend_id=f"beat_{info.backend_id}",
            label=info.label,
            capabilities=SolverCapabilities(
                supports_remote_assets=False,
                supports_parallel_workers=False,
                supports_symmetry=info.supports_symmetry,
                supports_channel_resynthesis=info.supports_channel_resynthesis,
                is_remote=False,
            ),
            factory=lambda backend=info.backend_id, **kwargs: _create_beat_engine_backend(
                beat_engine_backend=backend, **kwargs
            ),
            description=f"Run the local {info.label} solver.",
        )
        for info in backend_catalog()
    },
}


#: Backends that can run compiled physical-system (exterior and coupled FEM-BEM) solves.
PHYSICAL_SYSTEM_BACKEND_IDS = frozenset(
    {f"beat_{info.backend_id}" for info in backend_catalog() if "coupled_fem_bem_lem" in info.solve_kinds}
    | {"beat_remote"}
)
#: Backends that condense the FEM interior onto the retained interface for coupled solves.
CONDENSING_BACKEND_IDS = frozenset(
    {f"beat_{info.backend_id}" for info in backend_catalog() if info.condenses_fem_interior} | {"beat_remote"}
)


def supports_physical_system_solves(backend_id: str) -> bool:
    """Return whether a backend can run compiled physical-system solves."""

    return normalize_backend_id(backend_id) in PHYSICAL_SYSTEM_BACKEND_IDS


def backend_condenses_fem_interior(backend_id: str) -> bool:
    """Return whether coupled solves use FEM interface condensation."""

    return normalize_backend_id(backend_id) in CONDENSING_BACKEND_IDS


def available_backend_infos() -> tuple[SolverBackendInfo, ...]:
    return tuple(info for info in _BACKENDS.values() if info.available)


def backend_info(backend_id: str) -> SolverBackendInfo:
    normalized_id = normalize_backend_id(backend_id)
    if normalized_id in {"local", "server"}:
        raise ValueError("The legacy Bempp and HTTP solve backends are retired. Select a BEAT Engine backend.")
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
        # Compatibility alias for projects, settings, and scripts written before the CPU
        # monolithic and condensed selectors were consolidated.
        "beat_cpu_condensed": "beat_cpu",
        "beat_rocm": "beat_rocm",
        "rocm": "beat_rocm",
        "amd": "beat_rocm",
        "amdgpu": "beat_rocm",
    }
    aliases.update({info.backend_id: f"beat_{info.backend_id}" for info in backend_catalog()})
    return aliases.get(text, text or "beat_cpu")


def backend_label_to_id() -> dict[str, str]:
    return {info.label: info.backend_id for info in available_backend_infos()}


def _create_beat_engine_backend(
    *,
    julia_executable: str = "julia",
    solver_script: str | None = None,
    julia_threads: str | int = "auto",
    julia_project: str | None = "__default__",
    julia_sysimage: str | None = None,
    persistent_worker: bool = True,
    beat_engine_backend: str = "cuda",
    backend_id_override: str | None = None,
    label_override: str | None = None,
    **_kwargs: Any,
) -> SolverBackend:
    from blab.solvers.beat_engine_backend import BeatEngineBackend
    from blab.solvers.beat_engine_runtime import normalize_beat_engine_backend

    normalized_backend = normalize_beat_engine_backend(beat_engine_backend)
    info = engine_backend_info(normalized_backend)
    backend_id = backend_id_override or f"beat_{normalized_backend}"
    label = label_override or info.label
    default_project = engine_paths(normalized_backend).project
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
