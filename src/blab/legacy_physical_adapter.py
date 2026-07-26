"""Compatibility adapter from the existing BEM configuration to a physical model."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import meshio
import numpy as np

from blab.config import MeshConfig, RadiatorConfig, SimulationConfig
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


def physical_system_from_legacy_config(config: SimulationConfig) -> PhysicalSystem:
    """Represent a legacy exterior BEM solve using the physical-system schema."""

    mesh_configs = config.meshes or (MeshConfig(name="mesh", file=config.mesh_file, scale_factor=config.scale_factor),)
    mesh_resources = tuple(
        MeshResource(
            id=_mesh_id(mesh.name),
            name=mesh.name,
            file=mesh.file,
            purpose=MeshPurpose.BEM_SURFACE,
            scale_to_m=float(config.scale_factor if mesh.scale_factor is None else mesh.scale_factor),
            translation_m=tuple(float(value) for value in mesh.translation_m),
        )
        for mesh in mesh_configs
    )
    mesh_names = {mesh.name for mesh in mesh_configs}
    radiators = config.radiators or (RadiatorConfig(name="Radiator", tag=config.tag_throat, mesh=mesh_configs[0].name),)
    resolved_radiators = tuple(_resolve_radiator_mesh(radiator, mesh_configs) for radiator in radiators)
    radiator_by_group: dict[tuple[str, int], RadiatorConfig] = {}
    for radiator in resolved_radiators:
        key = (str(radiator.mesh), int(radiator.tag))
        if key in radiator_by_group:
            raise ValueError(
                f"Legacy radiators '{radiator_by_group[key].name}' and '{radiator.name}' "
                f"reference the same mesh physical group {key}."
            )
        radiator_by_group[key] = radiator

    boundaries = []
    boundary_id_by_group: dict[tuple[str, int], str] = {}
    for mesh_config in mesh_configs:
        for tag, group_name in _surface_groups(mesh_config.file):
            key = (mesh_config.name, tag)
            boundary_id = f"boundary:{mesh_config.name}:{tag}"
            boundary_id_by_group[key] = boundary_id
            radiator = radiator_by_group.get(key)
            boundaries.append(
                Boundary(
                    id=boundary_id,
                    name=f"{mesh_config.name}:{group_name}",
                    region_id="region:exterior",
                    group=PhysicalGroupRef(
                        mesh_id=_mesh_id(mesh_config.name),
                        dimension=2,
                        name=group_name,
                        tag=tag,
                    ),
                    kind=BoundaryKind.MOVING if radiator is not None else BoundaryKind.RIGID,
                )
            )

    missing_radiators = sorted(set(radiator_by_group) - set(boundary_id_by_group))
    if missing_radiators:
        raise ValueError(
            "Legacy radiator groups were not found in the mesh assets: "
            + ", ".join(f"{mesh}:{tag}" for mesh, tag in missing_radiators)
        )

    components = []
    excitation_ports = []
    for radiator in resolved_radiators:
        component_id = f"component:{radiator.name}"
        components.append(
            PhysicalComponent(
                id=component_id,
                name=radiator.name,
                kind=ComponentKind.IDEAL_VELOCITY_SOURCE,
                boundary_ids=(boundary_id_by_group[(str(radiator.mesh), int(radiator.tag))],),
                parameters={
                    "motion_profile": "uniform",
                    "velocity_offset_db": float(radiator.velocity_offset_db),
                },
            )
        )
        excitation_ports.append(
            ExcitationPort(
                id=f"excitation:{radiator.name}",
                name=f"{radiator.name} unit velocity",
                component_id=component_id,
                kind=ExcitationPortKind.NORMAL_VELOCITY,
            )
        )

    unknown_meshes = sorted(
        str(radiator.mesh) for radiator in resolved_radiators if str(radiator.mesh) not in mesh_names
    )
    if unknown_meshes:
        raise ValueError(f"Legacy radiators reference unknown meshes: {', '.join(unknown_meshes)}")

    return PhysicalSystem(
        id="legacy-bem-system",
        name="Legacy Boundary Lab BEM system",
        meshes=mesh_resources,
        regions=(
            AcousticRegion(
                id="region:exterior",
                name="Exterior air",
                kind=AcousticRegionKind.UNBOUNDED_AIR,
                mesh_ids=tuple(mesh.id for mesh in mesh_resources),
                sound_speed_m_per_s=float(config.sound_speed),
                density_kg_per_m3=float(config.rho),
            ),
        ),
        boundaries=tuple(boundaries),
        components=tuple(components),
        excitation_ports=tuple(excitation_ports),
        metadata={
            "source": "legacy_simulation_config",
            "application_synthesis_excluded": True,
        },
    )


def _resolve_radiator_mesh(
    radiator: RadiatorConfig,
    meshes: tuple[MeshConfig, ...],
) -> RadiatorConfig:
    if radiator.mesh is not None:
        return radiator
    if len(meshes) != 1:
        raise ValueError(f"Legacy radiator '{radiator.name}' must specify mesh when multiple meshes are configured.")
    return replace(radiator, mesh=meshes[0].name)


def _surface_groups(mesh_path: str) -> tuple[tuple[int, str], ...]:
    mesh = meshio.read(Path(mesh_path))
    physical_blocks = mesh.cell_data.get("gmsh:physical")
    if physical_blocks is None:
        raise ValueError(f"Legacy mesh does not contain gmsh:physical tags: {mesh_path}")
    triangle_tag_blocks = [
        np.asarray(physical_blocks[index], dtype=np.int64)
        for index, block in enumerate(mesh.cells)
        if block.type in {"triangle", "triangle3"}
    ]
    if not triangle_tag_blocks:
        raise ValueError(f"Legacy BEM mesh does not contain triangular surface elements: {mesh_path}")
    tags = np.concatenate(triangle_tag_blocks)
    names_by_tag = {
        int(np.asarray(value)[0]): name for name, value in mesh.field_data.items() if int(np.asarray(value)[1]) == 2
    }
    return tuple((tag, names_by_tag.get(tag, f"surface_{tag}")) for tag in sorted(map(int, np.unique(tags))))


def _mesh_id(name: str) -> str:
    return f"mesh:{name}"
