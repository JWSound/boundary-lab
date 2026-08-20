"""Projection of canonical physical-system results into observation-plane fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blab.channel_synthesis import channel_drive, flat_target_corrections
from blab.config import ChannelConfig
from blab.observation_planes import ObservationPlaneDisplay, ObservationPlaneType
from blab.physical_model import AcousticRegionKind
from blab.solve_results import (
    FEM_NODAL_PRESSURE_ID,
    FEM_VOLUME_DOMAIN_ID,
    HORIZONTAL_POLAR_PRESSURE_ID,
    BemBoundaryTraces,
    SolvedSystem,
    bem_boundary_traces_from_solved_system,
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
    tetrahedron_density_kg_per_m3: np.ndarray
    pressure_shape_gradients: np.ndarray
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
        return _response_options(self.channel_names_by_excitation)

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
        weights = _excitation_weights(
            solve_index=solve_index,
            frequency_hz=frequency,
            response_id=response_id,
            excitation_ids=self.excitation_ids,
            channel_names_by_excitation=self.channel_names_by_excitation,
            channel_configs=self.channel_configs,
            horizontal_pressure_by_frequency=self.horizontal_pressure_by_frequency,
            horizontal_angles_deg=self.horizontal_angles_deg,
            flat_target_enabled=self.flat_target_enabled,
            flat_target_reference_angle_deg=self.flat_target_reference_angle_deg,
        )
        return np.sum(basis * weights[:, np.newaxis], axis=0).astype(np.complex64, copy=False)

    def particle_velocity(self, frequency_hz: float | None, response_id: str) -> np.ndarray:
        """Return the complex P1 particle-velocity vector in each tetrahedron."""

        available_index = self.frequency_index(frequency_hz)
        frequency = float(self.frequencies_hz[available_index])
        pressure = self.pressure(frequency, response_id)
        pressure_gradient = np.einsum(
            "ti,tij->tj",
            pressure[np.asarray(self.tetrahedra, dtype=np.int64)],
            self.pressure_shape_gradients,
            optimize=True,
        )
        omega = 2.0 * np.pi * frequency
        velocity = pressure_gradient / (1j * omega * self.tetrahedron_density_kg_per_m3[:, np.newaxis])
        return velocity.astype(np.complex64, copy=False)


@dataclass(frozen=True)
class ExteriorFieldResults:
    """Retained BEM traces and response synthesis for exterior evaluation."""

    traces: BemBoundaryTraces
    backend_id: str
    sound_speed_m_per_s: float
    channel_names_by_excitation: tuple[str, ...]
    channel_configs: tuple[ChannelConfig, ...]
    horizontal_pressure_by_frequency: np.ndarray | None = None
    horizontal_angles_deg: np.ndarray | None = None
    flat_target_enabled: bool = False
    flat_target_reference_angle_deg: float = 0.0

    @property
    def run_id(self) -> str:
        return self.traces.run_id

    @property
    def frequencies_hz(self) -> np.ndarray:
        return self.traces.frequencies_hz

    @property
    def response_options(self) -> tuple[tuple[str, str], ...]:
        return _response_options(self.channel_names_by_excitation)

    def resolved_frequency(self, frequency_hz: float | None) -> float:
        index = _nearest_frequency_index(self.frequencies_hz, frequency_hz, label="Exterior field")
        return float(self.frequencies_hz[index])

    def boundary_response(
        self,
        frequency_hz: float | None,
        response_id: str,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        available_index = _nearest_frequency_index(self.frequencies_hz, frequency_hz, label="Exterior field")
        solve_index = int(self.traces.frequency_indices[available_index])
        frequency = float(self.frequencies_hz[available_index])
        pressure_basis, normal_basis = self.traces.excitation_basis(frequency)
        weights = _excitation_weights(
            solve_index=solve_index,
            frequency_hz=frequency,
            response_id=response_id,
            excitation_ids=self.traces.excitation_ids,
            channel_names_by_excitation=self.channel_names_by_excitation,
            channel_configs=self.channel_configs,
            horizontal_pressure_by_frequency=self.horizontal_pressure_by_frequency,
            horizontal_angles_deg=self.horizontal_angles_deg,
            flat_target_enabled=self.flat_target_enabled,
            flat_target_reference_angle_deg=self.flat_target_reference_angle_deg,
        )
        return (
            frequency,
            np.sum(pressure_basis * weights[:, np.newaxis], axis=0).astype(np.complex64, copy=False),
            np.sum(normal_basis * weights[:, np.newaxis], axis=0).astype(np.complex64, copy=False),
        )


@dataclass(frozen=True)
class ObservationFieldResults:
    interior: InteriorFieldResults | None = None
    exterior: ExteriorFieldResults | None = None

    @property
    def frequencies_hz(self) -> np.ndarray:
        source = self.exterior or self.interior
        return np.asarray(()) if source is None else np.asarray(source.frequencies_hz)

    def frequencies_for(self, plane_type: ObservationPlaneType) -> np.ndarray:
        if plane_type == ObservationPlaneType.INTERIOR:
            source = self.interior
            return np.asarray(()) if source is None else np.asarray(source.frequencies_hz)
        if plane_type == ObservationPlaneType.EXTERIOR:
            source = self.exterior
            return np.asarray(()) if source is None else np.asarray(source.frequencies_hz)
        if self.interior is not None and self.exterior is not None:
            # Both result families originate from the same solve, but an
            # interrupted or partially retained solve can leave different
            # availability masks. Combined fields must never silently pair
            # values from different frequencies.
            return np.intersect1d(
                np.asarray(self.interior.frequencies_hz, dtype=float),
                np.asarray(self.exterior.frequencies_hz, dtype=float),
                assume_unique=False,
            )
        return np.asarray(())

    @property
    def response_options(self) -> tuple[tuple[str, str], ...]:
        source = self.exterior or self.interior
        return () if source is None else source.response_options


def interior_field_results_from_solved_system(
    solved: SolvedSystem | None,
    *,
    component_channel_by_id: dict[str, str] | None = None,
    channel_configs: tuple[ChannelConfig, ...] = (),
    flat_target_enabled: bool = False,
    flat_target_reference_angle_deg: float = 0.0,
) -> InteriorFieldResults | None:
    if solved is None or solved.provenance.solve_kind not in {"coupled_bem_fem", "interior_fem"}:
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
    shape_gradients = tetrahedral_shape_gradients(points, tetrahedra)
    tetrahedron_density = _tetrahedron_density_kg_per_m3(solved, domain, tetrahedra.shape[0])

    availability = np.asarray(solved.completion_mask, dtype=bool) & np.asarray(
        quantity.available_frequency_mask, dtype=bool
    )
    frequency_indices = np.flatnonzero(availability)
    if not frequency_indices.size:
        return None
    frequency_indices = frequency_indices[
        np.argsort(np.asarray(solved.frequencies_hz)[frequency_indices], kind="stable")
    ]

    channel_names = _channel_names_by_excitation(solved, component_channel_by_id)
    horizontal_values, horizontal_angles = _horizontal_pressure_basis(solved, pressure.shape[:2])

    symmetry = str(solved.provenance.solver_options.get("symmetry", "off")).strip().lower()
    if symmetry not in {"off", "x", "xy"}:
        symmetry = "off"
    return InteriorFieldResults(
        run_id=solved.run_id,
        frequencies_hz=np.asarray(solved.frequencies_hz)[frequency_indices],
        frequency_indices=frequency_indices,
        points_m=points,
        tetrahedra=tetrahedra,
        tetrahedron_density_kg_per_m3=tetrahedron_density,
        pressure_shape_gradients=shape_gradients,
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


def tetrahedral_shape_gradients(points_m: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    """Return physical P1 basis gradients with shape ``(tetrahedron, 4, xyz)``."""

    points = np.asarray(points_m, dtype=float)
    cells = np.asarray(tetrahedra, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("FEM points must have shape (node, 3).")
    if cells.ndim != 2 or cells.shape[1] != 4:
        raise ValueError("FEM tetrahedra must have shape (tetrahedron, 4).")
    if np.any(cells < 0) or np.any(cells >= points.shape[0]):
        raise ValueError("FEM tetrahedra reference nodes outside the point array.")
    vertices = points[cells]
    jacobians = np.stack(
        (
            vertices[:, 1] - vertices[:, 0],
            vertices[:, 2] - vertices[:, 0],
            vertices[:, 3] - vertices[:, 0],
        ),
        axis=2,
    )
    determinants = np.linalg.det(jacobians)
    edge_scale = np.max(np.abs(jacobians), axis=(1, 2))
    tolerance = np.finfo(float).eps * edge_scale**3
    if np.any(np.abs(determinants) <= tolerance):
        raise ValueError("FEM volume mesh contains a numerically degenerate tetrahedron.")
    reference_gradients = np.asarray(
        [[-1.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 1.0]],
        dtype=float,
    )
    right_hand_side = np.broadcast_to(reference_gradients, (cells.shape[0], 3, 4))
    gradients = np.linalg.solve(np.swapaxes(jacobians, 1, 2), right_hand_side)
    return np.swapaxes(gradients, 1, 2)


def _tetrahedron_density_kg_per_m3(
    solved: SolvedSystem,
    domain,
    tetrahedron_count: int,
) -> np.ndarray:
    raw_densities = domain.metadata.get("density_kg_per_m3")
    if raw_densities is None and solved.compiled_system is not None:
        raw_densities = [
            region.density_kg_per_m3
            for region in getattr(solved.compiled_system, "regions", ())
            if region.kind == AcousticRegionKind.BOUNDED_AIR
        ]
    densities = np.asarray(() if raw_densities is None else raw_densities, dtype=float)
    if densities.ndim != 1 or not densities.size:
        raise ValueError("FEM particle velocity requires bounded-region density metadata.")
    if not np.all(np.isfinite(densities)) or np.any(densities <= 0.0):
        raise ValueError("FEM bounded-region densities must be finite and greater than zero.")
    raw_region_indices = domain.topology.get("region_index")
    if raw_region_indices is None:
        if densities.size != 1:
            raise ValueError("FEM tetrahedra require region indices when multiple densities are present.")
        region_indices = np.zeros(tetrahedron_count, dtype=np.int64)
    else:
        region_indices = np.asarray(raw_region_indices, dtype=np.int64)
    if region_indices.shape != (tetrahedron_count,):
        raise ValueError("FEM tetrahedron region indices do not align with the volume topology.")
    if np.any(region_indices < 0) or np.any(region_indices >= densities.size):
        raise ValueError("FEM tetrahedron region indices reference unavailable density values.")
    return densities[region_indices]


def exterior_field_results_from_solved_system(
    solved: SolvedSystem | None,
    *,
    component_channel_by_id: dict[str, str] | None = None,
    channel_configs: tuple[ChannelConfig, ...] = (),
    flat_target_enabled: bool = False,
    flat_target_reference_angle_deg: float = 0.0,
) -> ExteriorFieldResults | None:
    if solved is None or solved.provenance.solve_kind not in {"coupled_bem_fem", "exterior_bem"}:
        return None
    traces = bem_boundary_traces_from_solved_system(solved)
    if traces is None:
        return None
    unbounded_regions = (
        ()
        if solved.compiled_system is None
        else tuple(
            region for region in solved.compiled_system.regions if region.kind == AcousticRegionKind.UNBOUNDED_AIR
        )
    )
    if len(unbounded_regions) != 1:
        raise ValueError("Exterior field results require exactly one unbounded acoustic region.")
    channel_names = _channel_names_by_excitation(solved, component_channel_by_id)
    horizontal_values, horizontal_angles = _horizontal_pressure_basis(
        solved,
        (solved.frequencies_hz.size, len(solved.excitation_ids)),
    )
    return ExteriorFieldResults(
        traces=traces,
        backend_id=solved.provenance.backend_id,
        sound_speed_m_per_s=float(unbounded_regions[0].sound_speed_m_per_s),
        channel_names_by_excitation=channel_names,
        channel_configs=tuple(channel_configs),
        horizontal_pressure_by_frequency=horizontal_values,
        horizontal_angles_deg=horizontal_angles,
        flat_target_enabled=bool(flat_target_enabled),
        flat_target_reference_angle_deg=float(flat_target_reference_angle_deg),
    )


def observation_field_results_from_solved_system(
    solved: SolvedSystem | None,
    *,
    component_channel_by_id: dict[str, str] | None = None,
    channel_configs: tuple[ChannelConfig, ...] = (),
    flat_target_enabled: bool = False,
    flat_target_reference_angle_deg: float = 0.0,
) -> ObservationFieldResults | None:
    options = {
        "component_channel_by_id": component_channel_by_id,
        "channel_configs": channel_configs,
        "flat_target_enabled": flat_target_enabled,
        "flat_target_reference_angle_deg": flat_target_reference_angle_deg,
    }
    interior = interior_field_results_from_solved_system(solved, **options)
    exterior = exterior_field_results_from_solved_system(solved, **options)
    if interior is None and exterior is None:
        return None
    return ObservationFieldResults(interior=interior, exterior=exterior)


def _channel_names_by_excitation(
    solved: SolvedSystem,
    component_channel_by_id: dict[str, str] | None,
) -> tuple[str, ...]:
    component_channels = component_channel_by_id or {}
    component_by_port: dict[str, str] = {}
    if solved.compiled_system is not None:
        component_by_port = {port.id: port.component_id for port in solved.compiled_system.excitation_ports}
    return tuple(
        str(component_channels.get(component_by_port.get(port_id, ""), "main")) for port_id in solved.excitation_ids
    )


def _horizontal_pressure_basis(
    solved: SolvedSystem,
    leading_shape: tuple[int, int],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    horizontal = solved.quantities.get(HORIZONTAL_POLAR_PRESSURE_ID)
    if horizontal is None:
        return None, None
    candidate = np.asarray(horizontal.values)
    horizontal_domain = solved.domains.get(horizontal.domain_id or "")
    if candidate.ndim != 3 or candidate.shape[:2] != leading_shape or horizontal_domain is None:
        return None, None
    angles = np.asarray(horizontal_domain.coordinates.get("angle_deg"), dtype=float)
    if angles.shape != (candidate.shape[2],):
        return None, None
    return candidate, angles


def _response_options(channel_names_by_excitation: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    channel_names = tuple(dict.fromkeys(channel_names_by_excitation))
    return (("system", "System Response"),) + tuple((f"channel:{name}", f"Channel: {name}") for name in channel_names)


def _nearest_frequency_index(frequencies_hz: np.ndarray, frequency_hz: float | None, *, label: str) -> int:
    if not frequencies_hz.size:
        raise ValueError(f"{label} results contain no solved frequencies.")
    if frequency_hz is None or not np.isfinite(frequency_hz):
        return 0
    return int(np.argmin(np.abs(frequencies_hz - float(frequency_hz))))


def _excitation_weights(
    *,
    solve_index: int,
    frequency_hz: float,
    response_id: str,
    excitation_ids: tuple[str, ...],
    channel_names_by_excitation: tuple[str, ...],
    channel_configs: tuple[ChannelConfig, ...],
    horizontal_pressure_by_frequency: np.ndarray | None,
    horizontal_angles_deg: np.ndarray | None,
    flat_target_enabled: bool,
    flat_target_reference_angle_deg: float,
) -> np.ndarray:
    configs = {channel.name: channel for channel in channel_configs}
    channel_names = tuple(dict.fromkeys(channel_names_by_excitation))
    correction_by_channel = {name: 1.0 for name in channel_names}
    if flat_target_enabled and horizontal_pressure_by_frequency is not None and horizontal_angles_deg is not None:
        horizontal_basis = np.asarray(horizontal_pressure_by_frequency[solve_index])
        grouped = np.vstack(
            [
                np.sum(
                    horizontal_basis[
                        [index for index, candidate in enumerate(channel_names_by_excitation) if candidate == name]
                    ],
                    axis=0,
                )
                for name in channel_names
            ]
        )
        corrections = flat_target_corrections(
            grouped,
            horizontal_angles_deg,
            flat_target_reference_angle_deg,
            enabled=True,
        )
        correction_by_channel.update(
            {name: float(correction) for name, correction in zip(channel_names, corrections, strict=True)}
        )

    selected_channel = response_id.removeprefix("channel:") if response_id.startswith("channel:") else None
    if selected_channel not in channel_names:
        selected_channel = None
    weights = np.zeros(len(excitation_ids), dtype=np.complex64)
    for index, channel_name in enumerate(channel_names_by_excitation):
        if selected_channel is not None and channel_name != selected_channel:
            continue
        config = configs.get(channel_name, ChannelConfig(name=channel_name))
        weights[index] = correction_by_channel[channel_name] * channel_drive(config, frequency_hz)
    return weights


def project_field_scalars(
    pressure: np.ndarray,
    display: ObservationPlaneDisplay,
    *,
    animation_phase_deg: float | None = None,
    normalized_reference_db: float | None = None,
    pressure_color_limit_pa: float | None = None,
) -> FieldScalarProjection:
    pressure = np.asarray(pressure)
    display = ObservationPlaneDisplay(display)
    if display == ObservationPlaneDisplay.PARTICLE_VELOCITY:
        if pressure.ndim != 2 or pressure.shape[1] != 3:
            raise ValueError("Particle velocity values must have shape (sample, 3).")
        complex_magnitude = np.sqrt(np.sum(np.abs(pressure) ** 2, axis=1))
        if animation_phase_deg is None:
            values = complex_magnitude
            title = "Particle Velocity Magnitude (m/s)"
        else:
            instantaneous = np.real(pressure * np.exp(-1j * np.deg2rad(float(animation_phase_deg))))
            values = np.linalg.norm(instantaneous, axis=1)
            title = "Instantaneous Particle Speed (m/s)"
        maximum = _finite_abs_max(complex_magnitude)
        return FieldScalarProjection(values.astype(np.float32, copy=False), title, "turbo", (0.0, maximum))
    if animation_phase_deg is not None:
        values = np.real(pressure * np.exp(-1j * np.deg2rad(float(animation_phase_deg))))
        # Keep the animation scale stable for the entire cycle.  The complex
        # magnitude is the maximum instantaneous amplitude each sample can
        # reach, so its global maximum is a phase-independent symmetric limit.
        limit = _pressure_color_limit(np.abs(pressure), pressure_color_limit_pa)
        return FieldScalarProjection(values, "Instantaneous Pressure (Pa)", "coolwarm", (-limit, limit))
    if display == ObservationPlaneDisplay.SPL:
        values = pressure_spl_db(pressure)
        maximum = _finite_max(values, fallback=0.0)
        return FieldScalarProjection(values, "SPL (dB re 20 µPa)", "turbo", (maximum - 60.0, maximum))
    if display == ObservationPlaneDisplay.NORMALIZED_SPL:
        values = pressure_spl_db(pressure)
        reference_db = (
            _finite_max(values, fallback=0.0)
            if normalized_reference_db is None or not np.isfinite(normalized_reference_db)
            else float(normalized_reference_db)
        )
        values = values - reference_db
        return FieldScalarProjection(values, "Normalized SPL (dB)", "turbo", (-40.0, 0.0))
    if display == ObservationPlaneDisplay.PHASE:
        return FieldScalarProjection(phase_deg(pressure), "Phase (degrees)", "twilight", (-180.0, 180.0))
    values = np.real(pressure) if display == ObservationPlaneDisplay.REAL_PRESSURE else np.imag(pressure)
    limit = _pressure_color_limit(values, pressure_color_limit_pa)
    title = "Real Pressure (Pa)" if display == ObservationPlaneDisplay.REAL_PRESSURE else "Imaginary Pressure (Pa)"
    return FieldScalarProjection(values, title, "coolwarm", (-limit, limit))


def _finite_max(values: np.ndarray, *, fallback: float) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    return fallback if not finite.size else float(np.max(finite))


def _finite_abs_max(values: np.ndarray) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    maximum = 1.0 if not finite.size else float(np.max(np.abs(finite)))
    return max(maximum, np.finfo(float).eps)


def _pressure_color_limit(values: np.ndarray, manual_limit_pa: float | None) -> float:
    if manual_limit_pa is None:
        return _finite_abs_max(values)
    limit = float(manual_limit_pa)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("pressure_color_limit_pa must be finite and greater than zero.")
    return limit


def normalized_spl_reference_db(pressure: np.ndarray) -> float:
    """Return the finite peak SPL used as the zero-dB normalization reference."""

    return _finite_max(pressure_spl_db(np.asarray(pressure)), fallback=0.0)


__all__ = [
    "ExteriorFieldResults",
    "FieldScalarProjection",
    "InteriorFieldResults",
    "ObservationFieldResults",
    "exterior_field_results_from_solved_system",
    "interior_field_results_from_solved_system",
    "observation_field_results_from_solved_system",
    "normalized_spl_reference_db",
    "project_field_scalars",
    "tetrahedral_shape_gradients",
]
