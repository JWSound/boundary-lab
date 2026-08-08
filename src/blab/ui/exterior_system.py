"""Adapters from the physical authoring model to the established exterior BEM solver."""

from __future__ import annotations

import math
from dataclasses import dataclass

from blab.config import MeshConfig, RadiatorConfig
from blab.physical_compiler import PhysicalSystemCompiler
from blab.physical_model import (
    AcousticRegionKind,
    BoundaryKind,
    ComponentKind,
    PhysicalSolveKind,
    PhysicalSystem,
    infer_physical_solve_kind,
)


@dataclass(frozen=True)
class ExteriorBemInputs:
    mesh_configs: tuple[MeshConfig, ...]
    radiators: tuple[RadiatorConfig, ...]


def exterior_bem_inputs(
    system: PhysicalSystem,
    *,
    component_channel_by_id: dict[str, str] | None = None,
    symmetry_mode: str = "off",
) -> ExteriorBemInputs:
    """Compile an exterior-only system and translate its prescribed sources."""

    if infer_physical_solve_kind(system) != PhysicalSolveKind.EXTERIOR_BEM:
        raise ValueError("Exterior BEM inputs require an exterior-only physical system.")
    unsupported = [
        component.name for component in system.components if component.kind != ComponentKind.IDEAL_VELOCITY_SOURCE
    ]
    if unsupported:
        raise ValueError(
            "Exterior-only systems currently support prescribed-velocity components only: " + ", ".join(unsupported)
        )

    compiled = PhysicalSystemCompiler().compile(system, symmetry_mode=symmetry_mode)
    exterior = next(region for region in compiled.regions if region.kind == AcousticRegionKind.UNBOUNDED_AIR)
    meshes_by_id = {mesh.id: mesh for mesh in compiled.meshes}
    boundaries_by_id = {boundary.id: boundary for boundary in compiled.boundaries}
    channels = component_channel_by_id or {}
    mesh_configs = tuple(
        MeshConfig(
            name=meshes_by_id[mesh_id].name,
            file=meshes_by_id[mesh_id].file,
            scale_factor=meshes_by_id[mesh_id].scale_to_m,
            translation_m=meshes_by_id[mesh_id].translation_m,
        )
        for mesh_id in exterior.mesh_ids
    )

    radiators = []
    for component in compiled.components:
        raw_weights = component.parameters.get("boundary_motion_weights", {})
        weights = raw_weights if isinstance(raw_weights, dict) else {}
        for boundary_id in component.boundary_ids:
            boundary = boundaries_by_id[boundary_id]
            if boundary.kind != BoundaryKind.MOVING:
                continue
            if boundary.region_id != exterior.id:
                raise ValueError(
                    f"Exterior prescribed-velocity component '{component.name}' references "
                    "a boundary outside the exterior region."
                )
            weight = float(weights.get(boundary_id, 1.0))
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError(f"Component '{component.name}' has an invalid motion weight for '{boundary.name}'.")
            mesh = meshes_by_id[boundary.group.mesh_id]
            group_name = boundary.group.name or boundary.name
            radiators.append(
                RadiatorConfig(
                    name=f"{mesh.name}:{group_name}",
                    mesh=mesh.name,
                    tag=int(boundary.group.tag),
                    channel=str(channels.get(component.id, "main")),
                    velocity_offset_db=float(20.0 * math.log10(weight)),
                )
            )
    if not radiators:
        raise ValueError("Add at least one prescribed-velocity component before solving.")
    return ExteriorBemInputs(mesh_configs=mesh_configs, radiators=tuple(radiators))


__all__ = ["ExteriorBemInputs", "exterior_bem_inputs"]
