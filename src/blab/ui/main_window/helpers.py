"""Pure helpers used by the main window mixins."""

from __future__ import annotations

from dataclasses import replace

from blab.physical_model import (
    AcousticRegionKind,
    BoundaryKind,
    MeshPurpose,
    PhysicalSystem,
)
from blab.ui.dialogs import (
    MeshDialogEntry,
)


def _mesh_entries_with_file_overrides(
    meshes: tuple[MeshDialogEntry, ...],
    overrides_by_name: dict[str, str],
) -> tuple[MeshDialogEntry, ...]:
    return tuple(
        replace(mesh, cleaned_file=overrides_by_name.get(mesh.name, mesh.cleaned_file))
        for mesh in meshes
    )


def _physical_system_preview_metadata(
    system: PhysicalSystem | None,
    surface_tags_by_mesh: dict[str, dict[str, int]],
) -> tuple[set[tuple[str, int]], set[tuple[str, int]], dict[str, str], bool]:
    if system is None:
        return set(), set(), {}, False

    meshes_by_id = {mesh.id: mesh for mesh in system.meshes}
    boundaries_by_id = {boundary.id: boundary for boundary in system.boundaries}
    mesh_regions = {
        mesh.name: "interior" if mesh.purpose == MeshPurpose.FEM_VOLUME else "exterior"
        for mesh in system.meshes
    }
    has_interior_region = any(region.kind == AcousticRegionKind.BOUNDED_AIR for region in system.regions)

    def resolved_surface(boundary_id: str) -> tuple[str, int] | None:
        boundary = boundaries_by_id.get(boundary_id)
        if boundary is None:
            return None
        mesh = meshes_by_id.get(boundary.group.mesh_id)
        if mesh is None:
            return None
        tag = boundary.group.tag
        if tag is None and boundary.group.name is not None:
            tag = surface_tags_by_mesh.get(mesh.name, {}).get(boundary.group.name)
        if tag is None:
            return None
        return mesh.name, int(tag)

    interface_surfaces: set[tuple[str, int]] = set()
    for boundary in system.boundaries:
        if boundary.kind != BoundaryKind.INTERFACE:
            continue
        surface = resolved_surface(boundary.id)
        if surface is not None:
            interface_surfaces.add(surface)

    component_surfaces: set[tuple[str, int]] = set()
    for component in system.components:
        for boundary_id in component.boundary_ids:
            boundary = boundaries_by_id.get(boundary_id)
            if boundary is None or boundary.kind != BoundaryKind.MOVING:
                continue
            surface = resolved_surface(boundary_id)
            if surface is not None:
                component_surfaces.add(surface)
    return interface_surfaces, component_surfaces, mesh_regions, has_interior_region
