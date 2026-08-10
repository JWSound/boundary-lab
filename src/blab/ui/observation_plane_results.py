"""Projection of canonical coupled results into observation-plane fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blab.channel_synthesis import channel_drive, flat_target_corrections
from blab.config import ChannelConfig
from blab.observation_planes import ObservationPlaneDisplay
from blab.solve_results import (
    FEM_NODAL_PRESSURE_ID,
    FEM_VOLUME_DOMAIN_ID,
    HORIZONTAL_POLAR_PRESSURE_ID,
    SolvedSystem,
    phase_deg,
    pressure_spl_db,
)


@dataclass(frozen=True)
class FieldScalarProjection:
    values: np.ndarray
    title: str
    cmap: str
    clim: tuple[float, float]


@dataclass(frozen=True)
class InteriorFieldResults:
    """FEM topology and nodal pressure bases for one canonical solve run."""

    run_id: str
    frequencies_hz: np.ndarray
    frequency_indices: np.ndarray
    points_m: np.ndarray
    tetrahedra: np.ndarray
    pressure_by_frequency: np.ndarray
    excitation_ids: tuple[str, ...]
    channel_names_by_excitation: tuple[str, ...]
    channel_configs: tuple[ChannelConfig, ...]
    horizontal_pressure_by_frequency: np.ndarray | None = None
    horizontal_angles_deg: np.ndarray | None = None
    flat_target_enabled: bool = False
    flat_target_reference_angle_deg: float = 0.0
    symmetry: str = "off"

    @property
    def response_options(self) -> tuple[tuple[str, str], ...]:
        channel_names = tuple(dict.fromkeys(self.channel_names_by_excitation))
        return (("system", "System Response"),) + tuple(
            (f"channel:{name}", f"Channel: {name}") for name in channel_names
        )

    def frequency_index(self, frequency_hz: float | None) -> int:
        if not self.frequencies_hz.size:
            raise ValueError("Interior field results contain no solved frequencies.")
        if frequency_hz is None or not np.isfinite(frequency_hz):
            return 0
        return int(np.argmin(np.abs(self.frequencies_hz - float(frequency_hz))))

    def pressure(self, frequency_hz: float | None, response_id: str) -> np.ndarray:
        available_index = self.frequency_index(frequency_hz)
        solve_index = int(self.frequency_indices[available_index])
        frequency = float(self.frequencies_hz[available_index])
        basis = np.asarray(self.pressure_by_frequency[solve_index])
        weights = self._excitation_weights(solve_index, frequency, response_id)
        return np.sum(basis * weights[:, np.newaxis], axis=0).astype(np.complex64, copy=False)

    def _excitation_weights(self, solve_index: int, frequency_hz: float, response_id: str) -> np.ndarray:
        configs = {channel.name: channel for channel in self.channel_configs}
        channel_names = tuple(dict.fromkeys(self.channel_names_by_excitation))
        correction_by_channel = {name: 1.0 for name in channel_names}
        if (
            self.flat_target_enabled
            and self.horizontal_pressure_by_frequency is not None
            and self.horizontal_angles_deg is not None
        ):
            horizontal_basis = np.asarray(self.horizontal_pressure_by_frequency[solve_index])
            grouped = np.vstack(
                [
                    np.sum(
                        horizontal_basis[
                            [
                                index
                                for index, candidate in enumerate(self.channel_names_by_excitation)
                                if candidate == name
                            ]
                        ],
                        axis=0,
                    )
                    for name in channel_names
                ]
            )
            corrections = flat_target_corrections(
                grouped,
                self.horizontal_angles_deg,
                self.flat_target_reference_angle_deg,
                enabled=True,
            )
            correction_by_channel.update(
                {name: float(correction) for name, correction in zip(channel_names, corrections, strict=True)}
            )

        selected_channel = response_id.removeprefix("channel:") if response_id.startswith("channel:") else None
        if selected_channel not in channel_names:
            selected_channel = None
        weights = np.zeros(len(self.excitation_ids), dtype=np.complex64)
        for index, channel_name in enumerate(self.channel_names_by_excitation):
            if selected_channel is not None and channel_name != selected_channel:
                continue
            config = configs.get(channel_name, ChannelConfig(name=channel_name))
            weights[index] = correction_by_channel[channel_name] * channel_drive(config, frequency_hz)
        return weights


def interior_field_results_from_solved_system(
    solved: SolvedSystem | None,
    *,
    component_channel_by_id: dict[str, str] | None = None,
    channel_configs: tuple[ChannelConfig, ...] = (),
    flat_target_enabled: bool = False,
    flat_target_reference_angle_deg: float = 0.0,
) -> InteriorFieldResults | None:
    if solved is None or solved.provenance.solve_kind != "coupled_bem_fem":
        return None
    domain = solved.domains.get(FEM_VOLUME_DOMAIN_ID)
    quantity = solved.quantities.get(FEM_NODAL_PRESSURE_ID)
    if domain is None or quantity is None:
        return None
    points = np.asarray(domain.coordinates.get("points_m"), dtype=float)
    tetrahedra = np.asarray(domain.topology.get("tetrahedra"), dtype=np.int64)
    pressure = np.asarray(quantity.values)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("The FEM result domain points must have shape (node, 3).")
    if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4:
        raise ValueError("The FEM result domain topology must have shape (tetrahedron, 4).")
    if quantity.dimensions != ("frequency", "excitation", "fem_node"):
        raise ValueError("FEM nodal pressure has unexpected dimensions.")
    if pressure.shape != (solved.frequencies_hz.size, len(solved.excitation_ids), points.shape[0]):
        raise ValueError("FEM nodal pressure does not align with the solved frequencies, excitations, and nodes.")

    availability = np.asarray(solved.completion_mask, dtype=bool) & np.asarray(
        quantity.available_frequency_mask, dtype=bool
    )
    frequency_indices = np.flatnonzero(availability)
    if not frequency_indices.size:
        return None
    frequency_indices = frequency_indices[
        np.argsort(np.asarray(solved.frequencies_hz)[frequency_indices], kind="stable")
    ]

    component_channels = component_channel_by_id or {}
    component_by_port: dict[str, str] = {}
    if solved.compiled_system is not None:
        component_by_port = {port.id: port.component_id for port in solved.compiled_system.excitation_ports}
    channel_names = tuple(
        str(component_channels.get(component_by_port.get(port_id, ""), "main")) for port_id in solved.excitation_ids
    )

    horizontal = solved.quantities.get(HORIZONTAL_POLAR_PRESSURE_ID)
    horizontal_values = None
    horizontal_angles = None
    if horizontal is not None:
        candidate = np.asarray(horizontal.values)
        horizontal_domain = solved.domains.get(horizontal.domain_id or "")
        if candidate.ndim == 3 and candidate.shape[:2] == pressure.shape[:2] and horizontal_domain is not None:
            angles = np.asarray(horizontal_domain.coordinates.get("angle_deg"), dtype=float)
            if angles.shape == (candidate.shape[2],):
                horizontal_values = candidate
                horizontal_angles = angles

    symmetry = str(solved.provenance.solver_options.get("symmetry", "off")).strip().lower()
    if symmetry not in {"off", "x", "xy"}:
        symmetry = "off"
    return InteriorFieldResults(
        run_id=solved.run_id,
        frequencies_hz=np.asarray(solved.frequencies_hz)[frequency_indices],
        frequency_indices=frequency_indices,
        points_m=points,
        tetrahedra=tetrahedra,
        pressure_by_frequency=pressure,
        excitation_ids=solved.excitation_ids,
        channel_names_by_excitation=channel_names,
        channel_configs=tuple(channel_configs),
        horizontal_pressure_by_frequency=horizontal_values,
        horizontal_angles_deg=horizontal_angles,
        flat_target_enabled=bool(flat_target_enabled),
        flat_target_reference_angle_deg=float(flat_target_reference_angle_deg),
        symmetry=symmetry,
    )


def project_field_scalars(
    pressure: np.ndarray,
    display: ObservationPlaneDisplay,
    *,
    animation_phase_deg: float | None = None,
) -> FieldScalarProjection:
    pressure = np.asarray(pressure)
    if animation_phase_deg is not None:
        values = np.real(pressure * np.exp(-1j * np.deg2rad(float(animation_phase_deg))))
        # Keep the animation scale stable for the entire cycle.  The complex
        # magnitude is the maximum instantaneous amplitude each sample can
        # reach, so its global maximum is a phase-independent symmetric limit.
        limit = _finite_abs_max(np.abs(pressure))
        return FieldScalarProjection(values, "Instantaneous Pressure (Pa)", "coolwarm", (-limit, limit))
    display = ObservationPlaneDisplay(display)
    if display == ObservationPlaneDisplay.SPL:
        values = pressure_spl_db(pressure)
        maximum = _finite_max(values, fallback=0.0)
        return FieldScalarProjection(values, "SPL (dB re 20 µPa)", "turbo", (maximum - 60.0, maximum))
    if display == ObservationPlaneDisplay.NORMALIZED_SPL:
        values = pressure_spl_db(pressure)
        values = values - _finite_max(values, fallback=0.0)
        return FieldScalarProjection(values, "Normalized SPL (dB)", "turbo", (-40.0, 0.0))
    if display == ObservationPlaneDisplay.PHASE:
        return FieldScalarProjection(phase_deg(pressure), "Phase (degrees)", "twilight", (-180.0, 180.0))
    values = np.real(pressure) if display == ObservationPlaneDisplay.REAL_PRESSURE else np.imag(pressure)
    limit = _finite_abs_max(values)
    title = "Real Pressure (Pa)" if display == ObservationPlaneDisplay.REAL_PRESSURE else "Imaginary Pressure (Pa)"
    return FieldScalarProjection(values, title, "coolwarm", (-limit, limit))


def _finite_max(values: np.ndarray, *, fallback: float) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    return fallback if not finite.size else float(np.max(finite))


def _finite_abs_max(values: np.ndarray) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    maximum = 1.0 if not finite.size else float(np.max(np.abs(finite)))
    return max(maximum, np.finfo(float).eps)


__all__ = [
    "FieldScalarProjection",
    "InteriorFieldResults",
    "interior_field_results_from_solved_system",
    "project_field_scalars",
]
