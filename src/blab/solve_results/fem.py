"""Frequency-invariant FEM domains used by spatial solve results."""

from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np

from blab.physical_model import AcousticRegionKind, CompiledPhysicalSystem
from blab.solve_results.model import FEM_VOLUME_DOMAIN_ID, ResultDomain
from blab.symmetry import snap_points_to_symmetry_planes


def fem_volume_result_domain(system: CompiledPhysicalSystem, *, symmetry: str = "off") -> ResultDomain:
    """Reconstruct the node and tetrahedron order used by the coupled backend.

    Julia restricts each bounded region to its selected physical-volume tags,
    compacts the active nodes in source-index order, and then concatenates the
    regions. Mirroring that operation here keeps nodal result columns aligned
    without putting mesh topology into every streamed frequency result.
    """

    meshes_by_id = {mesh.id: mesh for mesh in system.meshes}
    points_by_region: list[np.ndarray] = []
    tetrahedra_by_region: list[np.ndarray] = []
    node_region_indices: list[np.ndarray] = []
    tetra_region_indices: list[np.ndarray] = []
    node_offsets: list[int] = []
    node_counts: list[int] = []
    tetra_offsets: list[int] = []
    tetra_counts: list[int] = []
    mesh_ids: list[str] = []
    region_ids: list[str] = []
    node_offset = 0
    tetra_offset = 0

    bounded_regions = tuple(region for region in system.regions if region.kind == AcousticRegionKind.BOUNDED_AIR)
    if not bounded_regions:
        raise ValueError("An FEM result domain requires at least one bounded acoustic region.")

    for region_index, region in enumerate(bounded_regions):
        if len(region.mesh_ids) != 1:
            raise ValueError(f"Bounded region {region.id!r} must reference exactly one FEM mesh.")
        mesh_id = region.mesh_ids[0]
        try:
            resource = meshes_by_id[mesh_id]
        except KeyError as exc:
            raise ValueError(f"Bounded region {region.id!r} references unknown mesh {mesh_id!r}.") from exc
        selected_tags = {int(group.tag) for group in region.volume_groups if group.mesh_id == mesh_id}
        if not selected_tags:
            raise ValueError(f"Bounded region {region.id!r} does not select a physical volume group.")

        mesh = meshio.read(Path(resource.file))
        selected_tetrahedra = _selected_tetrahedra(mesh, selected_tags)
        if not selected_tetrahedra.size:
            raise ValueError(f"Bounded region {region.id!r} contains no selected first-order tetrahedra.")
        active_vertices = np.unique(selected_tetrahedra.reshape(-1))
        source_to_local = np.full(len(mesh.points), -1, dtype=np.int64)
        source_to_local[active_vertices] = np.arange(active_vertices.size, dtype=np.int64)
        local_tetrahedra = source_to_local[selected_tetrahedra]
        if np.any(local_tetrahedra < 0):
            raise ValueError(f"Could not compact the FEM nodes for bounded region {region.id!r}.")

        points = np.asarray(mesh.points, dtype=np.float64)[active_vertices]
        points = points * float(resource.scale_to_m) + np.asarray(resource.translation_m, dtype=float)
        points = snap_points_to_symmetry_planes(points, symmetry)
        points_by_region.append(points)
        tetrahedra_by_region.append(local_tetrahedra + node_offset)
        node_region_indices.append(np.full(points.shape[0], region_index, dtype=np.int32))
        tetra_region_indices.append(np.full(local_tetrahedra.shape[0], region_index, dtype=np.int32))
        node_offsets.append(node_offset)
        node_counts.append(points.shape[0])
        tetra_offsets.append(tetra_offset)
        tetra_counts.append(local_tetrahedra.shape[0])
        mesh_ids.append(mesh_id)
        region_ids.append(region.id)
        node_offset += points.shape[0]
        tetra_offset += local_tetrahedra.shape[0]

    return ResultDomain(
        id=FEM_VOLUME_DOMAIN_ID,
        kind="fem_volume",
        dimensions=("fem_node",),
        coordinates={
            "points_m": np.vstack(points_by_region),
            "region_index": np.concatenate(node_region_indices),
        },
        topology={
            "tetrahedra": np.vstack(tetrahedra_by_region),
            "region_index": np.concatenate(tetra_region_indices),
        },
        metadata={
            "mesh_ids": mesh_ids,
            "region_ids": region_ids,
            "node_offsets": node_offsets,
            "node_counts": node_counts,
            "tetra_offsets": tetra_offsets,
            "tetra_counts": tetra_counts,
        },
    )


def _selected_tetrahedra(mesh: meshio.Mesh, selected_tags: set[int]) -> np.ndarray:
    physical_blocks = mesh.cell_data.get("gmsh:physical")
    if physical_blocks is None or len(physical_blocks) != len(mesh.cells):
        raise ValueError("FEM mesh tetrahedra do not contain gmsh:physical volume tags.")
    selected: list[np.ndarray] = []
    for block, raw_tags in zip(mesh.cells, physical_blocks, strict=True):
        if block.type not in {"tetra", "tetra4"}:
            continue
        cells = np.asarray(block.data, dtype=np.int64)
        tags = np.asarray(raw_tags, dtype=np.int64)
        if tags.shape != (cells.shape[0],):
            raise ValueError("FEM mesh physical-volume tags do not align with its tetrahedra.")
        mask = np.isin(tags, tuple(selected_tags))
        if np.any(mask):
            selected.append(cells[mask])
    return np.vstack(selected) if selected else np.empty((0, 4), dtype=np.int64)


__all__ = ["fem_volume_result_domain"]
