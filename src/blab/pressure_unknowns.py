"""Per-frequency pressure unknowns on the active prepared meshes."""

from __future__ import annotations

import meshio
import numpy as np

from blab.fem_topology import select_tetrahedra
from blab.physical_compiler import PhysicalSystemCompiler
from blab.physical_model import AcousticRegionKind, MeshPurpose, PhysicalSystem


def pressure_unknown_counts(
    system: PhysicalSystem | None,
    meshes: dict[str, meshio.Mesh],
    active_names: set[str],
    *,
    exterior_ready: bool = True,
) -> tuple[int | None, int | None]:
    """Count pressure spaces before condensation; None means unavailable."""
    if system is None:
        return None, None
    resources = {resource.id: resource for resource in system.meshes}
    stitched = {entry["region_id"]: entry for entry in system.metadata.get("exterior_preparation", ())}
    counts = {AcousticRegionKind.UNBOUNDED_AIR: 0, AcousticRegionKind.BOUNDED_AIR: 0}
    for region in system.regions:
        exterior = region.kind == AcousticRegionKind.UNBOUNDED_AIR
        if counts[region.kind] is None:
            continue
        try:
            for mesh_id in region.mesh_ids:
                resource = resources[mesh_id]
                entry = stitched.get(region.id) if exterior else None
                if entry is None and resource.name not in active_names:
                    continue
                if exterior:
                    count = _exterior_count(resource, meshes, active_names, entry, exterior_ready)
                else:
                    count = _interior_count(resource, meshes[resource.name], region.volume_groups)
                counts[region.kind] += count
        except (KeyError, ValueError):
            counts[region.kind] = None
    return counts[AcousticRegionKind.UNBOUNDED_AIR], counts[AcousticRegionKind.BOUNDED_AIR]


def _exterior_count(resource, meshes, active_names, preparation, ready):
    if preparation:
        sources = preparation["sources"]
        active_ids = {source["id"] for source in sources if source["name"] in active_names}
        if not active_ids:
            return 0
    if not ready or resource.purpose != MeshPurpose.BEM_SURFACE:
        raise ValueError("Exterior mesh unavailable")
    mesh = meshes[resource.name]
    if preparation and len(active_ids) != len(sources):
        # Limit a mixed stitched surface to the faces contributed by active sources.
        tags = [
            mesh.field_data[item["assembled_name"]][0]
            for item in preparation["surface_map"]
            if item["mesh_id"] in active_ids
        ]
        selected = np.isin(mesh.cell_data_dict["gmsh:physical"]["triangle"], tags)
        if not selected.any():
            raise ValueError("Active exterior surface unavailable")
        return np.unique(mesh.cells_dict["triangle"][selected]).size

    # BEAT constructs the exterior P1 space with one DOF per prepared vertex.
    if not any(block.type in {"triangle", "triangle3"} and len(block.data) for block in mesh.cells):
        raise ValueError("No linear BEM triangles")
    return len(mesh.points)


def _interior_count(resource, mesh, groups):
    if resource.purpose != MeshPurpose.FEM_VOLUME:
        raise ValueError("Interior volume unavailable")
    tags = {
        PhysicalSystemCompiler.resolve_mesh_group(mesh, group).tag
        for group in groups
        if group.mesh_id == resource.id and group.dimension == 3
    }
    cells = select_tetrahedra(mesh, tags)
    if not cells.size:
        raise ValueError("No active tetrahedra")
    return np.unique(cells).size
