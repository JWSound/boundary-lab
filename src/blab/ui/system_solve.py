"""Qt worker adapter for physical-system solves and existing live plots."""

from __future__ import annotations

import logging
import os
from dataclasses import replace

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from blab.solve_results import (
    HORIZONTAL_POLAR_PRESSURE_ID,
    RADIATION_IMPEDANCE_ID,
    SPHERE_PRESSURE_ID,
    VERTICAL_POLAR_PRESSURE_ID,
)
from blab.solvers.base import FrequencyResult, FrequencySolveTimings, SolverDiagnostics
from blab.solvers.coupled_backend import PhysicalSystemProductionBackend
from blab.system_contract import SystemFrequencyResult
from blab.system_solve import (
    SystemUiSolveRequest,
    canonicalize_observation_result,
    prepare_coupled_ui_solve,
    prepare_system_ui_solve,
    supports_exterior_system_protocol,
)

LOGGER = logging.getLogger(__name__)


class SystemSolveWorker(QObject):
    """Run the selected physical-system backend and emit legacy-shaped live results."""

    initialized = Signal(object, object, object)
    result_ready = Signal(object)
    system_result_ready = Signal(object)
    status = Signal(str)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, prepared: SystemUiSolveRequest):
        super().__init__()
        self.prepared = prepared
        self._stop = False
        self._session = None

    @Slot()
    def run(self) -> None:
        try:
            request = replace(self.prepared.request, status_callback=self._log_backend_status)
            bem_backend = "cuda" if self.prepared.backend_id == "beat_cuda" else "cpu"
            backend = PhysicalSystemProductionBackend(
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
                canonical_result = self._canonical_result(result)
                self.system_result_ready.emit(canonical_result)
                self.result_ready.emit(self._to_live_result(canonical_result))
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
        LOGGER.info("Physical-system solver backend status: %s", message)

    def _to_live_result(self, result: SystemFrequencyResult) -> FrequencyResult:
        result = self._canonical_result(result)
        horizontal = self._pressure_values(result, HORIZONTAL_POLAR_PRESSURE_ID)
        vertical = self._pressure_values(result, VERTICAL_POLAR_PRESSURE_ID)
        sphere_quantity = next((item for item in result.quantities if item.id == SPHERE_PRESSURE_ID), None)
        sphere = None if sphere_quantity is None else np.asarray(sphere_quantity.values, dtype=np.complex64)
        channel_names, horizontal, vertical, sphere = self._combine_channel_rows(
            horizontal,
            vertical,
            sphere,
        )
        angle_count = self.prepared.polar_angle_deg.size
        placeholder = np.zeros(angle_count, dtype=np.float32)
        impedance = self._impedance_values(result)
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

    def _canonical_result(self, result: SystemFrequencyResult) -> SystemFrequencyResult:
        return canonicalize_observation_result(self.prepared, result)

    @staticmethod
    def _pressure_values(result: SystemFrequencyResult, quantity_id: str) -> np.ndarray:
        quantity = next((item for item in result.quantities if item.id == quantity_id), None)
        if quantity is None:
            raise ValueError(f"Coupled solver result did not contain {quantity_id!r}.")
        pressure = np.asarray(quantity.values, dtype=np.complex64)
        if pressure.ndim != 2:
            raise ValueError(f"System pressure quantity {quantity_id!r} must have two dimensions.")
        return pressure

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

    def _impedance_values(self, result: SystemFrequencyResult) -> np.ndarray:
        quantity = next((item for item in result.quantities if item.id == RADIATION_IMPEDANCE_ID), None)
        if quantity is None:
            return np.full(
                (self.prepared.excitation_component_names.size, 2),
                np.nan,
                dtype=np.float32,
            )
        values = np.asarray(quantity.values, dtype=np.complex64)
        if values.shape != (self.prepared.excitation_component_names.size,):
            raise ValueError("Radiation impedance does not align with the physical components.")
        return np.column_stack((values.real, values.imag)).astype(np.float32, copy=False)


__all__ = [
    "SystemSolveWorker",
    "SystemUiSolveRequest",
    "prepare_system_ui_solve",
    "supports_exterior_system_protocol",
    "CoupledSolveWorker",
    "CoupledUiSolveRequest",
    "prepare_coupled_ui_solve",
]


CoupledUiSolveRequest = SystemUiSolveRequest
CoupledSolveWorker = SystemSolveWorker
