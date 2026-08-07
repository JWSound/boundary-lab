"""Application adapter from coupled-system results to the existing live plots."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from blab.config import normalize_symmetry
from blab.live import build_log_frequencies, order_frequencies_for_live_plotting
from blab.physical_compiler import PhysicalSystemCompiler
from blab.physical_model import BoundaryKind, PhysicalSystem
from blab.solvers.base import FrequencyResult, FrequencySolveTimings, SolverDiagnostics
from blab.solvers.coupled_backend import CoupledProductionBackend
from blab.solvers.registry import normalize_backend_id
from blab.system_contract import OutputRequest, SystemFrequencyResult, SystemSolveRequest

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoupledUiSolveRequest:
    request: SystemSolveRequest
    backend_id: str
    polar_angle_deg: np.ndarray
    excitation_channel_names: np.ndarray
    excitation_component_names: np.ndarray
    horizontal_count: int
    vertical_count: int
    sphere_metadata: dict[str, np.ndarray] | None = None


def prepare_coupled_ui_solve(
    system: PhysicalSystem,
    *,
    freq_min_hz: float,
    freq_max_hz: float,
    freq_count: int,
    observation_distance_m: float,
    polar_angle_step_deg: float,
    spherical_sampling_enabled: bool = False,
    spherical_sampling_points: int = 0,
    component_channel_by_id: dict[str, str] | None = None,
    backend_id: str = "beat_cpu",
    symmetry_mode: str = "off",
) -> CoupledUiSolveRequest:
    """Compile an editable system and request the field points used by current plots."""

    symmetry = normalize_symmetry(symmetry_mode)
    if any(boundary.kind == BoundaryKind.UNUSED for boundary in system.boundaries):
        raise ValueError("The coupled solver does not yet support unused surface groups.")
    compiled = PhysicalSystemCompiler().compile(system, symmetry_mode=symmetry)
    frequencies = build_log_frequencies(
        float(min(freq_min_hz, freq_max_hz)),
        float(max(freq_min_hz, freq_max_hz)),
        int(freq_count),
    )
    ordered = order_frequencies_for_live_plotting(frequencies)
    angles, horizontal, vertical = _polar_observation_points(
        distance_m=float(observation_distance_m),
        step_deg=float(polar_angle_step_deg),
    )
    point_blocks = [horizontal, vertical]
    sphere_metadata = None
    if spherical_sampling_enabled:
        sphere, sphere_metadata = _fibonacci_sphere_points(
            max(int(spherical_sampling_points), 1),
            float(observation_distance_m),
        )
        point_blocks.append(sphere)
    points = np.vstack(point_blocks)

    components = {component.id: component for component in compiled.components}
    port_ids = tuple(port.id for port in compiled.excitation_ports)
    if not port_ids:
        raise ValueError("Add at least one prescribed-velocity component before solving.")
    channel_names = []
    component_names = []
    channel_by_component = component_channel_by_id or {}
    for port in compiled.excitation_ports:
        component = components[port.component_id]
        channel_names.append(str(channel_by_component.get(component.id, "main")))
        component_names.append(component.name)

    normalized_backend_id = normalize_backend_id(backend_id)
    if normalized_backend_id not in {"beat_cpu", "beat_cuda"}:
        raise ValueError("Coupled systems require BEAT Engine (CPU) or BEAT Engine (CUDA) in Preferences.")
    request = SystemSolveRequest(
        compiled_system=compiled,
        frequencies_hz=tuple(float(value) for value in ordered),
        excitation_port_ids=port_ids,
        outputs=(
            OutputRequest(
                id="ui:exterior-pressure",
                quantity="exterior_pressure",
                options={"points_m": points.tolist()},
            ),
        ),
        solver_options={
            "quadrature_order": 2,
            "singular_order": 2,
            "validation_diagnostics": False,
            "cache_frequency_invariant": True,
            "static_condensation": normalized_backend_id == "beat_cuda",
            "symmetry": symmetry,
        },
    )
    return CoupledUiSolveRequest(
        request=request,
        backend_id=normalized_backend_id,
        polar_angle_deg=angles,
        excitation_channel_names=np.asarray(channel_names),
        excitation_component_names=np.asarray(component_names),
        horizontal_count=len(horizontal),
        vertical_count=len(vertical),
        sphere_metadata=sphere_metadata,
    )


class CoupledSolveWorker(QObject):
    """Run the selected FP32 production backend and emit legacy-shaped live results."""

    initialized = Signal(object, object, object)
    result_ready = Signal(object)
    status = Signal(str)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, prepared: CoupledUiSolveRequest):
        super().__init__()
        self.prepared = prepared
        self._stop = False
        self._session = None

    @Slot()
    def run(self) -> None:
        try:
            request = replace(self.prepared.request, status_callback=self._log_backend_status)
            bem_backend = "cuda" if self.prepared.backend_id == "beat_cuda" else "cpu"
            backend = CoupledProductionBackend(
                bem_backend=bem_backend,
                julia_executable=os.environ.get("BLAB_JULIA_EXE", "julia"),
            )
            session = backend.create_system_session(request)
            self._session = session
            self.initialized.emit(
                self.prepared.polar_angle_deg,
                self.prepared.excitation_component_names,
                self.prepared.sphere_metadata,
            )
            for result in session.solve_stream(stop_requested=lambda: self._stop):
                self.result_ready.emit(self._to_live_result(result))
        except Exception as exc:
            if not self._stop:
                self.failed.emit(str(exc))
        finally:
            self.finished.emit()

    @Slot()
    def stop(self) -> None:
        self._stop = True
        if self._session is not None:
            self._session.stop()

    @staticmethod
    def _log_backend_status(message: str) -> None:
        LOGGER.info("Coupled solver backend status: %s", message)

    def _to_live_result(self, result: SystemFrequencyResult) -> FrequencyResult:
        quantity = next(
            (item for item in result.quantities if item.id == "ui:exterior-pressure"),
            None,
        )
        if quantity is None:
            raise ValueError("Coupled solver result did not contain exterior pressure.")
        pressure = np.asarray(quantity.values, dtype=np.complex64)
        if pressure.ndim != 2:
            raise ValueError("Coupled exterior pressure must have shape (excitation, observation).")
        horizontal_end = self.prepared.horizontal_count
        vertical_end = horizontal_end + self.prepared.vertical_count
        horizontal = pressure[:, :horizontal_end]
        vertical = pressure[:, horizontal_end:vertical_end]
        sphere = pressure[:, vertical_end:] if pressure.shape[1] > vertical_end else None
        channel_names, horizontal, vertical, sphere = self._combine_channel_rows(
            horizontal,
            vertical,
            sphere,
        )
        angle_count = self.prepared.polar_angle_deg.size
        placeholder = np.zeros(angle_count, dtype=np.float32)
        impedance = np.full(
            (self.prepared.excitation_component_names.size, 2),
            np.nan,
            dtype=np.float32,
        )
        residual = result.diagnostics.get("relative_residual")
        continuity = result.diagnostics.get("pressure_continuity_error")
        diagnostic_message = (
            f"Coupled residual {float(residual):.3g}"
            if residual is not None
            else None
            if continuity is None
            else f"Interface continuity {float(continuity):.3g}"
        )
        raw_timings = result.diagnostics.get("timings", {})
        raw_timings = raw_timings if isinstance(raw_timings, dict) else {}
        assembly_s = (
            float(raw_timings.get("assembly_s", 0.0))
            + float(raw_timings.get("mesh_setup_s", 0.0))
            + float(raw_timings.get("cache_setup_s", 0.0))
        )
        return FrequencyResult(
            freq_hz=float(result.freq_hz),
            horizontal_spl_norm_db=placeholder.copy(),
            vertical_spl_norm_db=placeholder.copy(),
            impedance=impedance,
            horizontal_spl_db=placeholder.copy(),
            vertical_spl_db=placeholder.copy(),
            channel_names=channel_names,
            horizontal_pressure=horizontal.astype(np.complex64),
            vertical_pressure=vertical.astype(np.complex64),
            sphere_pressure=None if sphere is None else sphere.astype(np.complex64),
            timings=FrequencySolveTimings(
                assembly_s=assembly_s,
                solve_s=float(raw_timings.get("solve_s", 0.0)),
                field_s=float(raw_timings.get("field_s", 0.0)),
            ),
            diagnostics=SolverDiagnostics(message=diagnostic_message),
        )

    def _combine_channel_rows(
        self,
        horizontal: np.ndarray,
        vertical: np.ndarray,
        sphere: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        names = [str(value) for value in self.prepared.excitation_channel_names.tolist()]
        ordered_names = list(dict.fromkeys(names))
        grouped_horizontal = []
        grouped_vertical = []
        grouped_sphere = []
        for name in ordered_names:
            indices = [index for index, candidate in enumerate(names) if candidate == name]
            grouped_horizontal.append(np.sum(horizontal[indices], axis=0))
            grouped_vertical.append(np.sum(vertical[indices], axis=0))
            if sphere is not None:
                grouped_sphere.append(np.sum(sphere[indices], axis=0))
        return (
            np.asarray(ordered_names),
            np.vstack(grouped_horizontal),
            np.vstack(grouped_vertical),
            None if sphere is None else np.vstack(grouped_sphere),
        )


def _polar_observation_points(
    *,
    distance_m: float,
    step_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if distance_m <= 0.0:
        raise ValueError("Polar observation distance must be greater than zero.")
    if step_deg <= 0.0:
        raise ValueError("Polar angle step must be greater than zero.")
    angles = np.arange(-180.0, 180.0 + 0.5 * step_deg, step_deg, dtype=np.float32)
    angles = np.clip(angles, -180.0, 180.0)
    radians = np.deg2rad(angles.astype(float))
    horizontal = distance_m * np.column_stack([np.sin(radians), np.zeros_like(radians), np.cos(radians)])
    vertical = distance_m * np.column_stack([np.zeros_like(radians), np.sin(radians), np.cos(radians)])
    return angles, horizontal, vertical


def _fibonacci_sphere_points(
    count: int,
    distance_m: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    indices = np.arange(count, dtype=float)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (indices + 0.5) / count
    radius_xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = golden_angle * indices
    x = radius_xy * np.cos(phi)
    y = radius_xy * np.sin(phi)
    points = distance_m * np.column_stack([x, y, z])
    return points, {
        "r_distance_m": np.full(count, distance_m, dtype=np.float32),
        "theta_polar_rad": np.arccos(np.clip(z, -1.0, 1.0)).astype(np.float32),
        "phi_azimuth_rad": np.arctan2(y, x).astype(np.float32),
    }


__all__ = [
    "CoupledSolveWorker",
    "CoupledUiSolveRequest",
    "prepare_coupled_ui_solve",
]
