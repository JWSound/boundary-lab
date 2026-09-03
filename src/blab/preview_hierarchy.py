"""Qt-free hierarchy for controlling mesh-preview actor visibility."""

from __future__ import annotations

from dataclasses import dataclass

from blab.config import MeshConfig
from blab.physical_model import PhysicalSystem

PreviewSurfaceKey = tuple[str, int | None]


@dataclass(frozen=True)
class PreviewBoundaryNode:
    id: str
    name: str
    surface_keys: tuple[PreviewSurfaceKey, ...]


@dataclass(frozen=True)
class PreviewMeshNode:
    id: str
    name: str
    boundaries: tuple[PreviewBoundaryNode, ...]


@dataclass(frozen=True)
class PreviewRegionNode:
    id: str
    name: str
    meshes: tuple[PreviewMeshNode, ...]


@dataclass(frozen=True)
class PreviewHierarchy:
    regions: tuple[PreviewRegionNode, ...]

    @property
    def surface_keys(self) -> tuple[PreviewSurfaceKey, ...]:
        return tuple(
            dict.fromkeys(
                surface_key
                for region in self.regions
                for mesh in region.meshes
                for boundary in mesh.boundaries
                for surface_key in boundary.surface_keys
            )
        )


def build_preview_hierarchy(
    system: PhysicalSystem | None,
    *,
    source_mesh_configs: tuple[MeshConfig, ...],
    source_surface_tags_by_mesh: dict[str, dict[str, int]],
    solver_surface_by_source: dict[PreviewSurfaceKey, PreviewSurfaceKey],
) -> PreviewHierarchy:
    """Describe regions, source meshes, and rendered boundary actors."""

    if system is None:
        meshes = tuple(
            _fallback_mesh_node(
                mesh.name,
                source_surface_tags_by_mesh.get(mesh.name, {}),
                solver_surface_by_source,
                node_id=f"mesh:exterior:{mesh.name}",
            )
            for mesh in source_mesh_configs
        )
        return PreviewHierarchy((PreviewRegionNode("region:exterior", "Exterior", meshes),))

    resources_by_id = {mesh.id: mesh for mesh in system.meshes}
    boundaries_by_region_mesh: dict[tuple[str, str], list] = {}
    for boundary in system.boundaries:
        boundaries_by_region_mesh.setdefault((boundary.region_id, boundary.group.mesh_id), []).append(boundary)

    regions = []
    represented_mesh_names = set()
    for region in system.regions:
        region_meshes = []
        for mesh_id in region.mesh_ids:
            resource = resources_by_id.get(mesh_id)
            if resource is None:
                continue
            represented_mesh_names.add(resource.name)
            source_tags = source_surface_tags_by_mesh.get(resource.name, {})
            names_by_tag = {int(tag): name for name, tag in source_tags.items()}
            boundary_nodes = []
            assigned_source_keys = set()
            for boundary in boundaries_by_region_mesh.get((region.id, mesh_id), []):
                source_tag = boundary.group.tag
                if source_tag is None and boundary.group.name is not None:
                    source_tag = source_tags.get(boundary.group.name)
                source_key = (resource.name, None if source_tag is None else int(source_tag))
                solver_key = solver_surface_by_source.get(source_key)
                if solver_key is None:
                    continue
                assigned_source_keys.add(source_key)
                boundary_nodes.append(
                    PreviewBoundaryNode(
                        id=f"boundary:{boundary.id}",
                        name=boundary.name,
                        surface_keys=(solver_key,),
                    )
                )
            for source_key, solver_key in _source_surfaces_for_mesh(
                resource.name,
                solver_surface_by_source,
            ):
                if source_key in assigned_source_keys:
                    continue
                tag = source_key[1]
                boundary_nodes.append(
                    PreviewBoundaryNode(
                        id=f"surface:{region.id}:{resource.id}:{tag}",
                        name=_surface_name(tag, names_by_tag),
                        surface_keys=(solver_key,),
                    )
                )
            region_meshes.append(
                PreviewMeshNode(
                    id=f"mesh:{region.id}:{resource.id}",
                    name=resource.name,
                    boundaries=tuple(boundary_nodes),
                )
            )
        regions.append(PreviewRegionNode(f"region:{region.id}", region.name, tuple(region_meshes)))

    unassigned_meshes = tuple(
        _fallback_mesh_node(
            mesh.name,
            source_surface_tags_by_mesh.get(mesh.name, {}),
            solver_surface_by_source,
            node_id=f"mesh:unassigned:{mesh.name}",
        )
        for mesh in source_mesh_configs
        if mesh.name not in represented_mesh_names
    )
    if unassigned_meshes:
        regions.append(PreviewRegionNode("region:unassigned", "Unassigned", unassigned_meshes))
    return PreviewHierarchy(tuple(regions))


def _fallback_mesh_node(
    mesh_name: str,
    surface_tags: dict[str, int],
    solver_surface_by_source: dict[PreviewSurfaceKey, PreviewSurfaceKey],
    *,
    node_id: str,
) -> PreviewMeshNode:
    names_by_tag = {int(tag): name for name, tag in surface_tags.items()}
    boundaries = tuple(
        PreviewBoundaryNode(
            id=f"surface:{node_id}:{source_key[1]}",
            name=_surface_name(source_key[1], names_by_tag),
            surface_keys=(solver_key,),
        )
        for source_key, solver_key in _source_surfaces_for_mesh(mesh_name, solver_surface_by_source)
    )
    return PreviewMeshNode(node_id, mesh_name, boundaries)


def _source_surfaces_for_mesh(
    mesh_name: str,
    solver_surface_by_source: dict[PreviewSurfaceKey, PreviewSurfaceKey],
) -> tuple[tuple[PreviewSurfaceKey, PreviewSurfaceKey], ...]:
    return tuple(
        sorted(
            (
                (source_key, solver_key)
                for source_key, solver_key in solver_surface_by_source.items()
                if source_key[0] == mesh_name
            ),
            key=lambda item: -1 if item[0][1] is None else int(item[0][1]),
        )
    )


def _surface_name(tag: int | None, names_by_tag: dict[int, str]) -> str:
    if tag is None:
        return "Untagged surface"
    return names_by_tag.get(int(tag), f"Tag {int(tag)}")


__all__ = [
    "PreviewBoundaryNode",
    "PreviewHierarchy",
    "PreviewMeshNode",
    "PreviewRegionNode",
    "PreviewSurfaceKey",
    "build_preview_hierarchy",
]
