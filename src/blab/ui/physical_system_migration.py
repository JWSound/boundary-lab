"""Mesh-aware migration from legacy radiator assignments to a physical system."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import meshio
import numpy as np

from blab.config import MeshConfig, RadiatorConfig
from blab.physical_model import (
    AcousticRegion,
    AcousticRegionKind,
    Boundary,
    BoundaryKind,
    ComponentKind,
    ExcitationPort,
    ExcitationPortKind,
    MeshPurpose,
    MeshResource,
    PhysicalComponent,
    PhysicalGroupRef,
    PhysicalSystem,
)
from blab.ui.system_config import AvailableSystemMesh

AUTO_SEEDED_EXTERIOR_KEY = "auto_seeded_exterior"


class PhysicalSystemMigrationError(ValueError):
    """A legacy exterior project could not be represented as a physical system."""


@dataclass
class _DriveGroup:
    mesh_name: str
    component_name: str
    channel: str
    boundary_ids: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)


def seed_exterior_system(
    meshes: tuple[AvailableSystemMesh, ...],
    radiators: tuple[RadiatorConfig, ...],
) -> tuple[PhysicalSystem, dict[str, str]]:
    """Build a deterministic exterior system from currently materialized surface meshes."""

    exterior_meshes = tuple(mesh for mesh in meshes if not mesh.has_tetrahedra)
    if not exterior_meshes:
        raise ValueError("Exterior source migration requires at least one BEM surface mesh.")

    used_ids: set[str] = set()
    resources = []
    resource_by_name = {}
    for mesh in exterior_meshes:
        resource = MeshResource(
            id=_unique_id(f"mesh:{_slug(mesh.name)}", used_ids),
            name=mesh.name,
            file=mesh.file,
            purpose=MeshPurpose.BEM_SURFACE,
            scale_to_m=mesh.scale_to_m,
            translation_m=mesh.translation_m,
        )
        resources.append(resource)
        resource_by_name[mesh.name] = resource

    region_id = _unique_id("region:exterior-air", used_ids)
    region = AcousticRegion(
        id=region_id,
        name="Exterior Air",
        kind=AcousticRegionKind.UNBOUNDED_AIR,
        mesh_ids=tuple(resource.id for resource in resources),
    )
    radiator_by_key = _radiators_by_mesh_tag(radiators, tuple(resource_by_name))
    boundaries = []
    driven_boundaries = []
    for mesh in exterior_meshes:
        resource = resource_by_name[mesh.name]
        tags_by_name = _surface_tags_by_name(Path(mesh.file))
        for group_name in mesh.surface_groups:
            tag = tags_by_name.get(group_name)
            if tag is None:
                continue
            boundary_id = _unique_id(
                f"boundary:{_slug(region_id)}:{_slug(mesh.name)}:{_slug(group_name)}",
                used_ids,
            )
            radiator = radiator_by_key.get((mesh.name, tag))
            boundaries.append(
                Boundary(
                    id=boundary_id,
                    name=group_name,
                    region_id=region_id,
                    group=PhysicalGroupRef(mesh_id=resource.id, dimension=2, name=group_name),
                    kind=BoundaryKind.MOVING if radiator is not None else BoundaryKind.RIGID,
                )
            )
            if radiator is None:
                continue
            driven_boundaries.append((mesh.name, group_name, boundary_id, radiator))

    grouped_drives: dict[tuple[str, str, str], _DriveGroup] = {}
    for mesh_name, group_name, boundary_id, radiator in driven_boundaries:
        channel = radiator.channel or "main"
        if radiator.drive_group:
            group_key = (mesh_name, radiator.drive_group, channel)
            component_name = radiator.drive_group_name or radiator.drive_group
        else:
            group_key = (mesh_name, f"boundary:{boundary_id}", channel)
            component_name = group_name
        group = grouped_drives.setdefault(
            group_key,
            _DriveGroup(mesh_name=mesh_name, component_name=component_name, channel=channel),
        )
        group.boundary_ids.append(boundary_id)
        group.weights[boundary_id] = float(10.0 ** (radiator.velocity_offset_db / 20.0))

    components = []
    ports = []
    component_channels = {}
    for group in grouped_drives.values():
        component_id = _unique_id(
            f"component:{_slug(group.mesh_name)}:{_slug(group.component_name)}",
            used_ids,
        )
        components.append(
            PhysicalComponent(
                id=component_id,
                name=group.component_name,
                kind=ComponentKind.IDEAL_VELOCITY_SOURCE,
                boundary_ids=tuple(group.boundary_ids),
                parameters={
                    "motion_profile": "uniform",
                    "boundary_motion_weights": dict(group.weights),
                },
            )
        )
        port_id = _unique_id(f"excitation:{_slug(component_id)}", used_ids)
        ports.append(
            ExcitationPort(
                id=port_id,
                name=f"{group.component_name} unit normal velocity",
                component_id=component_id,
                kind=ExcitationPortKind.NORMAL_VELOCITY,
            )
        )
        component_channels[component_id] = group.channel

    return (
        PhysicalSystem(
            id="system:loudspeaker",
            name="Loudspeaker System",
            meshes=tuple(resources),
            regions=(region,),
            boundaries=tuple(boundaries),
            components=tuple(components),
            excitation_ports=tuple(ports),
            metadata={AUTO_SEEDED_EXTERIOR_KEY: True},
        ),
        component_channels,
    )


def seed_exterior_system_from_solver_inputs(
    meshes: tuple[MeshConfig, ...],
    radiators: tuple[RadiatorConfig, ...],
) -> tuple[PhysicalSystem, dict[str, str]]:
    """Represent a materialized legacy mesh assembly as a physical system."""

    available = []
    for mesh_config in meshes:
        path = Path(mesh_config.file)
        mesh = meshio.read(path)
        surface_groups = tuple(
            sorted(
                str(name)
                for name, raw in mesh.field_data.items()
                if int(np.asarray(raw).tolist()[1]) == 2
            )
        )
        has_tetrahedra = any(block.type in {"tetra", "tetra4"} and len(block.data) for block in mesh.cells)
        if has_tetrahedra:
            raise ValueError(f"Exterior compatibility mesh '{mesh_config.name}' contains tetrahedra.")
        available.append(
            AvailableSystemMesh(
                name=mesh_config.name,
                source_file=str(path),
                file=str(path),
                scale_to_m=float(mesh_config.scale_factor if mesh_config.scale_factor is not None else 0.001),
                translation_m=tuple(float(value) for value in mesh_config.translation_m),
                surface_groups=surface_groups,
                volume_groups=(),
                has_tetrahedra=False,
            )
        )
    return seed_exterior_system(tuple(available), radiators)


def _radiators_by_mesh_tag(
    radiators: tuple[RadiatorConfig, ...],
    mesh_names: tuple[str, ...],
) -> dict[tuple[str, int], RadiatorConfig]:
    default_mesh = mesh_names[0] if len(mesh_names) == 1 else None
    return {
        (str(radiator.mesh or default_mesh), int(radiator.tag)): radiator
        for radiator in radiators
        if radiator.mesh is not None or default_mesh is not None
    }


def _surface_tags_by_name(path: Path) -> dict[str, int]:
    mesh = meshio.read(path)
    return {str(name): int(np.asarray(raw)[0]) for name, raw in mesh.field_data.items() if int(np.asarray(raw)[1]) == 2}


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")
    return slug or "item"


def _unique_id(base: str, used: set[str]) -> str:
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}-{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


__all__ = [
    "AUTO_SEEDED_EXTERIOR_KEY",
    "PhysicalSystemMigrationError",
    "seed_exterior_system",
    "seed_exterior_system_from_solver_inputs",
]
