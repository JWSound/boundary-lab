"""Qt-free projections from canonical solve results into live plotting caches."""

from __future__ import annotations

import numpy as np

from blab.phasor import solver_to_standard_phasor
from blab.physical_model import PhysicalSolveKind
from blab.solve_results.model import (
    HORIZONTAL_POLAR_PRESSURE_ID,
    RADIATION_IMPEDANCE_ID,
    SPHERE_PRESSURE_ID,
    VERTICAL_POLAR_PRESSURE_ID,
)
from blab.solvers.base import FrequencyResult, FrequencySolveTimings, SolverDiagnostics
from blab.system_contract import SystemFrequencyResult
from blab.system_solve import PreparedSystemSolve, canonicalize_observation_result


def frequency_result_timings(result: SystemFrequencyResult) -> FrequencySolveTimings:
    raw = result.diagnostics.get("timings", {})
    raw = raw if isinstance(raw, dict) else {}
    return FrequencySolveTimings(
        assembly_s=sum(float(raw.get(key, 0.0)) for key in ("assembly_s", "mesh_setup_s", "cache_setup_s")),
        solve_s=float(raw.get("solve_s", 0.0)),
        field_s=float(raw.get("field_s", 0.0)),
    )


class LiveResultProjector:
    """Derive channel-grouped plot data while retaining the canonical excitation basis."""

    def __init__(self, prepared: PreparedSystemSolve):
        self.prepared = prepared

    def project(self, result: SystemFrequencyResult) -> FrequencyResult:
        result = canonicalize_observation_result(self.prepared, result)
        if self.prepared.solve_kind == PhysicalSolveKind.INTERIOR_FEM:
            excitation_count = len(result.excitation_port_ids)
            empty_pressure = np.zeros((excitation_count, 0), dtype=np.complex64)
            channel_names, horizontal, vertical, _sphere = self._combine_channel_rows(
                empty_pressure,
                empty_pressure,
                None,
            )
            residual = result.diagnostics.get("relative_residual")
            return FrequencyResult(
                freq_hz=float(result.freq_hz),
                horizontal_spl_norm_db=np.empty(0, dtype=np.float32),
                vertical_spl_norm_db=np.empty(0, dtype=np.float32),
                impedance=self._impedance_values(result),
                horizontal_spl_db=np.empty(0, dtype=np.float32),
                vertical_spl_db=np.empty(0, dtype=np.float32),
                channel_names=channel_names,
                horizontal_pressure=horizontal,
                vertical_pressure=vertical,
                timings=frequency_result_timings(result),
                diagnostics=SolverDiagnostics(
                    message=None if residual is None else f"Interior FEM residual {float(residual):.3g}"
                ),
            )
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
            timings=frequency_result_timings(result),
            diagnostics=SolverDiagnostics(message=diagnostic_message),
        )

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
        display_values = solver_to_standard_phasor(values)
        return np.column_stack((display_values.real, display_values.imag)).astype(
            np.float32,
            copy=False,
        )
