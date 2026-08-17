"""Qt-free construction and normalization of physical-system solve requests."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from blab.config import normalize_symmetry
from blab.live import build_log_frequencies, order_frequencies_for_live_plotting
from blab.observation_planes import ObservationPlane, ObservationPlaneType
from blab.physical_compiler import PhysicalSystemCompiler
from blab.physical_model import (
    BoundaryKind,
    ComponentKind,
    PhysicalSolveKind,
    PhysicalSystem,
    infer_physical_solve_kind,
)
from blab.solve_results import (
    BEM_BOUNDARY_DOMAIN_ID,
    BEM_BOUNDARY_NEUMANN_ID,
    BEM_BOUNDARY_PRESSURE_ID,
    DIAPHRAGM_VELOCITY_ID,
    FEM_NODAL_PRESSURE_ID,
    FEM_VOLUME_DOMAIN_ID,
    HORIZONTAL_POLAR_DOMAIN_ID,
    HORIZONTAL_POLAR_PRESSURE_ID,
    RADIATION_IMPEDANCE_ID,
    RADIATOR_DOMAIN_ID,
    SPHERE_DOMAIN_ID,
    SPHERE_PRESSURE_ID,
    TRANSDUCER_DOMAIN_ID,
    VERTICAL_POLAR_DOMAIN_ID,
    VERTICAL_POLAR_PRESSURE_ID,
    VOICE_COIL_CURRENT_ID,
    ResultDomain,
    bem_boundary_result_domain,
    fem_volume_result_domain,
)
from blab.solvers.registry import (
    backend_condenses_fem_interior,
    normalize_backend_id,
    supports_physical_system_solves,
)
from blab.system_contract import OutputRequest, QuantityResult, SystemFrequencyResult, SystemSolveRequest


@dataclass(frozen=True)
class SystemUiSolveRequest:
    """Prepared physical-system solve plus the metadata used by result consumers."""

    request: SystemSolveRequest
    backend_id: str
    solve_kind: PhysicalSolveKind
    polar_angle_deg: np.ndarray
    excitation_channel_names: np.ndarray
    excitation_component_names: np.ndarray
    horizontal_count: int
    vertical_count: int
    sphere_metadata: dict[str, np.ndarray] | None = None
    result_domains: tuple[ResultDomain, ...] = ()


def prepare_system_ui_solve(
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
    observation_planes: tuple[ObservationPlane, ...] = (),
) -> SystemUiSolveRequest:
    """Compile an editable physical system and request the fields used by the UI."""

    symmetry = normalize_symmetry(symmetry_mode)
    if any(boundary.kind == BoundaryKind.UNUSED for boundary in system.boundaries):
        raise ValueError("The coupled solver does not yet support unused surface groups.")
    compiled = PhysicalSystemCompiler().compile(system, symmetry_mode=symmetry)
    solve_kind = infer_physical_solve_kind(system)
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
    observation_domains = [
        {
            "id": HORIZONTAL_POLAR_DOMAIN_ID,
            "quantity_id": HORIZONTAL_POLAR_PRESSURE_ID,
            "offset": 0,
            "count": len(horizontal),
        },
        {
            "id": VERTICAL_POLAR_DOMAIN_ID,
            "quantity_id": VERTICAL_POLAR_PRESSURE_ID,
            "offset": len(horizontal),
            "count": len(vertical),
        },
    ]
    result_domains = [
        ResultDomain(
            id=HORIZONTAL_POLAR_DOMAIN_ID,
            kind="polar_observation",
            dimensions=("observation",),
            coordinates={"points_m": horizontal, "angle_deg": angles},
            metadata={"plane": "horizontal"},
        ),
        ResultDomain(
            id=VERTICAL_POLAR_DOMAIN_ID,
            kind="polar_observation",
            dimensions=("observation",),
            coordinates={"points_m": vertical, "angle_deg": angles},
            metadata={"plane": "vertical"},
        ),
    ]
    if solve_kind == PhysicalSolveKind.EXTERIOR_BEM:
        result_domains.append(
            ResultDomain(
                id=RADIATOR_DOMAIN_ID,
                kind="component_collection",
                dimensions=("radiator",),
                coordinates={
                    "component_id": np.asarray([component.id for component in compiled.components]),
                    "name": np.asarray([component.name for component in compiled.components]),
                },
            )
        )
    sphere_metadata = None
    if spherical_sampling_enabled:
        sphere, sphere_metadata = _fibonacci_sphere_points(
            max(int(spherical_sampling_points), 1),
            float(observation_distance_m),
        )
        sphere_offset = sum(len(block) for block in point_blocks)
        point_blocks.append(sphere)
        observation_domains.append(
            {
                "id": SPHERE_DOMAIN_ID,
                "quantity_id": SPHERE_PRESSURE_ID,
                "offset": sphere_offset,
                "count": len(sphere),
            }
        )
        result_domains.append(
            ResultDomain(
                id=SPHERE_DOMAIN_ID,
                kind="spherical_observation",
                dimensions=("observation",),
                coordinates={"points_m": sphere, **sphere_metadata},
            )
        )
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
    if not supports_physical_system_solves(normalized_backend_id):
        raise ValueError(
            "Physical-system solves require BEAT Engine CPU, Nvidia CUDA, or AMD ROCm."
        )
    outputs = [
        OutputRequest(
            id="ui:exterior-pressure",
            quantity="exterior_pressure",
            options={
                "points_m": points.tolist(),
                "observation_domains": observation_domains,
            },
        )
    ]
    if solve_kind == PhysicalSolveKind.EXTERIOR_BEM:
        outputs.append(
            OutputRequest(
                id=RADIATION_IMPEDANCE_ID,
                quantity="radiation_impedance",
                target_ids=(RADIATOR_DOMAIN_ID,),
            )
        )
    retain_interior_field = any(
        plane.plane_type in {ObservationPlaneType.INTERIOR, ObservationPlaneType.COMBINED}
        for plane in observation_planes
    )
    retain_exterior_field = any(
        plane.plane_type in {ObservationPlaneType.EXTERIOR, ObservationPlaneType.COMBINED}
        for plane in observation_planes
    )
    if solve_kind == PhysicalSolveKind.EXTERIOR_BEM and retain_interior_field:
        raise ValueError("Exterior-only systems support Exterior observation planes only.")
    if retain_exterior_field:
        outputs.extend(
            (
                OutputRequest(
                    id=BEM_BOUNDARY_PRESSURE_ID,
                    quantity="bem_boundary_pressure",
                    target_ids=(BEM_BOUNDARY_DOMAIN_ID,),
                ),
                OutputRequest(
                    id=BEM_BOUNDARY_NEUMANN_ID,
                    quantity="bem_boundary_neumann",
                    target_ids=(BEM_BOUNDARY_DOMAIN_ID,),
                ),
            )
        )
        result_domains.append(bem_boundary_result_domain(compiled, symmetry=symmetry))
    if retain_interior_field:
        outputs.append(
            OutputRequest(
                id=FEM_NODAL_PRESSURE_ID,
                quantity="fem_nodal_pressure",
                target_ids=(FEM_VOLUME_DOMAIN_ID,),
            )
        )
        result_domains.append(fem_volume_result_domain(compiled))
    transducers = [
        component for component in compiled.components if component.kind == ComponentKind.ELECTRODYNAMIC_TRANSDUCER
    ]
    if transducers:
        result_domains.append(
            ResultDomain(
                id=TRANSDUCER_DOMAIN_ID,
                kind="component_collection",
                dimensions=("transducer",),
                coordinates={
                    "component_id": np.asarray([component.id for component in transducers]),
                    "name": np.asarray([component.name for component in transducers]),
                },
            )
        )
        outputs.extend(
            [
                OutputRequest(
                    id=DIAPHRAGM_VELOCITY_ID,
                    quantity="diaphragm_velocity",
                    target_ids=(TRANSDUCER_DOMAIN_ID,),
                ),
                OutputRequest(
                    id=VOICE_COIL_CURRENT_ID,
                    quantity="voice_coil_current",
                    target_ids=(TRANSDUCER_DOMAIN_ID,),
                ),
            ]
        )

    is_coupled = solve_kind == PhysicalSolveKind.COUPLED_BEM_FEM
    request = SystemSolveRequest(
        compiled_system=compiled,
        frequencies_hz=tuple(float(value) for value in ordered),
        excitation_port_ids=port_ids,
        outputs=tuple(outputs),
        solver_options={
            "quadrature_order": 2 if is_coupled else 4,
            "singular_order": 2 if is_coupled else 4,
            "validation_diagnostics": False,
            "cache_frequency_invariant": True,
            "static_condensation": is_coupled
            and backend_condenses_fem_interior(normalized_backend_id),
            "symmetry": symmetry,
        },
    )
    return SystemUiSolveRequest(
        request=request,
        backend_id=normalized_backend_id,
        solve_kind=solve_kind,
        polar_angle_deg=angles,
        excitation_channel_names=np.asarray(channel_names),
        excitation_component_names=np.asarray(component_names),
        horizontal_count=len(horizontal),
        vertical_count=len(vertical),
        sphere_metadata=sphere_metadata,
        result_domains=tuple(result_domains),
    )


def prepare_coupled_ui_solve(*args, **kwargs) -> SystemUiSolveRequest:
    """Compatibility name for callers that predate exterior system solves."""

    return prepare_system_ui_solve(*args, **kwargs)


def supports_exterior_system_protocol(
    system: PhysicalSystem,
    *,
    backend_id: str,
    stitch_exterior_meshes: bool,
) -> bool:
    """Return whether an exterior project can use the local system worker."""

    if stitch_exterior_meshes or not supports_physical_system_solves(backend_id):
        return False
    try:
        if infer_physical_solve_kind(system) != PhysicalSolveKind.EXTERIOR_BEM:
            return False
    except ValueError:
        return False
    if any(boundary.kind not in {BoundaryKind.RIGID, BoundaryKind.MOVING} for boundary in system.boundaries):
        return False
    if any(boundary.parameters for boundary in system.boundaries):
        return False
    if any(region.loss_model for region in system.regions):
        return False
    for component in system.components:
        if component.kind != ComponentKind.IDEAL_VELOCITY_SOURCE:
            return False
        if component.parameters.get("motion_profile", "uniform") != "uniform":
            return False
        if set(component.parameters) - {"motion_profile", "boundary_motion_weights"}:
            return False
    return True


def canonicalize_observation_result(
    prepared: SystemUiSolveRequest,
    result: SystemFrequencyResult,
) -> SystemFrequencyResult:
    """Split a compact exterior-pressure block into named observation quantities."""

    combined = next((item for item in result.quantities if item.id == "ui:exterior-pressure"), None)
    if combined is None:
        return result
    pressure = np.asarray(combined.values)
    if pressure.ndim != 2:
        raise ValueError("Exterior pressure must have shape (excitation, observation).")
    output = next(item for item in prepared.request.outputs if item.id == "ui:exterior-pressure")
    domains = output.options.get("observation_domains", ())
    quantities = [item for item in result.quantities if item.id != combined.id]
    for domain in domains:
        offset = int(domain["offset"])
        count = int(domain["count"])
        quantities.append(
            QuantityResult(
                id=str(domain["quantity_id"]),
                quantity=combined.quantity,
                unit=combined.unit,
                values=pressure[:, offset : offset + count],
                target_id=str(domain["id"]),
                axes=combined.axes,
                metadata={"source_output_id": combined.id},
            )
        )
    return replace(result, quantities=tuple(quantities))


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
    "SystemUiSolveRequest",
    "canonicalize_observation_result",
    "prepare_coupled_ui_solve",
    "prepare_system_ui_solve",
    "supports_exterior_system_protocol",
]
