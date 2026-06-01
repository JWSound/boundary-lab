"""Shared solver backend contract and result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator, Protocol

import numpy as np

from blab.config import SimulationConfig


@dataclass(frozen=True)
class FrequencySolveTimings:
    assembly_s: float = 0.0
    solve_s: float = 0.0
    field_s: float = 0.0


@dataclass(frozen=True)
class SolverDiagnostics:
    convergence_info: int | None = None
    message: str | None = None


@dataclass(frozen=True)
class FrequencyResult:
    freq_hz: float
    horizontal_spl_norm_db: np.ndarray
    vertical_spl_norm_db: np.ndarray
    impedance: np.ndarray
    horizontal_spl_db: np.ndarray | None = None
    vertical_spl_db: np.ndarray | None = None
    sphere_spl_norm_db: np.ndarray | None = None
    timings: FrequencySolveTimings = field(default_factory=FrequencySolveTimings)
    diagnostics: SolverDiagnostics | None = None


@dataclass(frozen=True)
class MeshAsset:
    original_path: str
    filename: str
    content_base64: str
    kind: str = "mesh"


@dataclass(frozen=True)
class SolveRequest:
    config: SimulationConfig
    frequencies_hz: np.ndarray
    assets: tuple[MeshAsset, ...] = ()
    status_callback: Callable[[str], None] | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class SolveMetadata:
    polar_angle_deg: np.ndarray
    radiator_names: np.ndarray
    sphere_metadata: dict[str, np.ndarray] | None = None


@dataclass(frozen=True)
class SolverCapabilities:
    supports_spherical_sampling: bool = True
    supports_impedance: bool = True
    supports_burton_miller: bool = True
    supports_flat_target_normalization: bool = True
    supports_cancellation: bool = True
    supports_streaming: bool = True
    supports_remote_assets: bool = False
    supports_parallel_workers: bool = False
    supports_symmetry: bool = False
    is_remote: bool = False


class SolverSession(Protocol):
    @property
    def metadata(self) -> SolveMetadata:
        ...

    def solve_stream(
        self,
        *,
        stop_requested: Callable[[], bool] | None = None,
    ) -> Iterator[FrequencyResult]:
        ...

    def stop(self) -> None:
        ...


class SolverBackend(Protocol):
    backend_id: str
    label: str
    capabilities: SolverCapabilities

    def create_session(self, request: SolveRequest) -> SolverSession:
        ...
