"""Build a conforming BEM interface from authoritative FEM boundary facets.

The implementation supports two bounded Boundary Lab cases:

* the FEM interface is a tagged set of triangular tetrahedron boundary facets;
* when the FEM and BEM interface seams have identical discrete topology and are
  geometrically near-coincident, the BEM interface is replaced directly and
  may be fully curved or span multiple geometrical surfaces;
* otherwise the FEM and BEM interface perimeters describe the same planar
  opening, surrounded by a planar BEM physical surface, possibly split across
  multiple geometrical entities.

The FEM interface triangles are retained exactly. A matching discrete seam
snaps only its BEM perimeter vertices to the authoritative FEM coordinates and
preserves the rest of the surrounding BEM surface. The planar fallback remeshes
that surface with a hole whose inner loop is the FEM interface perimeter. Both
paths avoid modifying or retetrahedralizing the FEM volume.
"""

from __future__ import annotations

import argparse
import uuid
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import meshio
import numpy as np
from scipy.spatial import cKDTree


class InterfaceConformError(ValueError):
    """Raised when an interface cannot be conformed without unsafe guessing."""


@dataclass(frozen=True)
class InterfaceIdentityReport:
    interface_triangles: int
    interface_vertices: int
    max_coordinate_error: float
    fem_facets_on_tetra_boundary: int
    bem_boundary_edges: int


@dataclass(frozen=True)
class InterfaceTopologyMap:
    """Explicit topology correspondence for a conforming FEM-BEM interface.

    Tuple entries are ordered by the FEM interface vertices/faces. Face indices
    refer to the concatenated triangle blocks returned by ``meshio``.
    ``normal_sign`` is +1 when corresponding FEM and BEM triangle normals agree
    and -1 when they oppose one another.
    """

    fem_vertex_indices: tuple[int, ...]
    fem_to_bem_vertex_indices: tuple[int, ...]
    fem_face_indices: tuple[int, ...]
    bem_face_indices: tuple[int, ...]
    normal_sign: tuple[int, ...]
    max_coordinate_error: float
    fem_facets_on_tetra_boundary: int
    bem_boundary_edges: int


@dataclass(frozen=True)
class InterfaceConformResult:
    fem_interface_triangles: int
    original_bem_interface_triangles: int
    original_adjacent_triangles: int
    remeshed_adjacent_triangles: int
    output_vertices: int
    output_triangles: int
    max_original_boundary_deviation: float
    interface_orientation_flipped: bool
    identity: InterfaceIdentityReport
    seam_simplification_used: bool = False
    simplified_bem_seam_edges: int = 0


@dataclass(frozen=True)
class _TriangleData:
    triangles: np.ndarray
    physical_tags: np.ndarray
    geometrical_tags: np.ndarray


def conform_bem_interface_to_fem(
    fem_mesh: meshio.Mesh,
    bem_mesh: meshio.Mesh,
    *,
    fem_interface_name: str = "Interface",
    bem_interface_name: str = "Interface",
    geometry_tolerance: float | None = None,
    seam_tolerance: float | None = None,
    merge_tolerance: float = 1e-8,
    symmetry_mode: str = "off",
    protected_bem_interface_names: tuple[str, ...] = (),
) -> tuple[meshio.Mesh, InterfaceConformResult]:
    """Return a BEM mesh whose interface facets exactly match the FEM facets.

    A curved or multi-surface interface is replaced directly when its FEM and
    BEM seams have identical discrete topology and differ only within the seam
    tolerance. Otherwise, the BEM interface must have a planar perimeter
    surrounded by a planar physical surface. The FEM interface is not remeshed.
    """

    symmetry = _normalized_symmetry_mode(symmetry_mode)
    symmetry_axes = _symmetry_axes(symmetry)
    if merge_tolerance <= 0.0:
        raise InterfaceConformError("merge_tolerance must be greater than zero.")

    fem_interface_tag = _physical_surface_tag(fem_mesh, fem_interface_name)
    bem_interface_tag = _physical_surface_tag(bem_mesh, bem_interface_name)
    protected_bem_tags = {
        _physical_surface_tag(bem_mesh, name) for name in protected_bem_interface_names if name != bem_interface_name
    }
    fem_data = _triangle_data(fem_mesh, require_geometrical=False)
    bem_data = _triangle_data(bem_mesh, require_geometrical=True)

    fem_interface = fem_data.triangles[fem_data.physical_tags == fem_interface_tag]
    bem_interface_mask = bem_data.physical_tags == bem_interface_tag
    bem_interface = bem_data.triangles[bem_interface_mask]
    if len(fem_interface) == 0:
        raise InterfaceConformError(f"FEM physical surface '{fem_interface_name}' contains no triangles.")
    if len(bem_interface) == 0:
        raise InterfaceConformError(f"BEM physical surface '{bem_interface_name}' contains no triangles.")

    fem_interface_loops = _boundary_loops(fem_interface)
    bem_interface_loops = _boundary_loops(bem_interface)
    if len(fem_interface_loops) != 1 or len(bem_interface_loops) != 1:
        raise InterfaceConformError("The interface conformer requires one closed perimeter loop per interface.")
    fem_inner_loop = fem_interface_loops[0]
    old_bem_inner_loop = bem_interface_loops[0]

    interface_diameter = _point_cloud_diameter(fem_mesh.points[np.unique(fem_interface)])
    resolved_geometry_tolerance = (
        max(interface_diameter * 5e-3, 1e-9) if geometry_tolerance is None else float(geometry_tolerance)
    )
    if resolved_geometry_tolerance <= 0.0:
        raise InterfaceConformError("geometry_tolerance must be greater than zero.")
    resolved_seam_tolerance = (
        max(interface_diameter * 5e-5, merge_tolerance) if seam_tolerance is None else float(seam_tolerance)
    )
    if resolved_seam_tolerance <= 0.0:
        raise InterfaceConformError("seam_tolerance must be greater than zero.")
    symmetry_plane_tolerance = max(interface_diameter * 1e-6, merge_tolerance * 10.0, 1e-9)

    if symmetry_axes:
        _validate_positive_symmetry_domain(fem_mesh.points, symmetry_axes, symmetry_plane_tolerance, "FEM")
        _validate_positive_symmetry_domain(bem_mesh.points, symmetry_axes, symmetry_plane_tolerance, "BEM")
        fem_interface_symmetry_axes = _perimeter_symmetry_axes(
            fem_mesh.points,
            fem_inner_loop,
            symmetry_axes,
            symmetry_plane_tolerance,
        )
        bem_interface_symmetry_axes = _perimeter_symmetry_axes(
            bem_mesh.points,
            old_bem_inner_loop,
            symmetry_axes,
            symmetry_plane_tolerance,
        )
        if fem_interface_symmetry_axes != bem_interface_symmetry_axes:
            raise InterfaceConformError(
                "The FEM and BEM interface perimeters are cut by different active symmetry planes."
            )
        interface_symmetry_axes = fem_interface_symmetry_axes
    else:
        interface_symmetry_axes = ()

    if interface_symmetry_axes:
        fem_opening_path = _physical_opening_path(
            fem_mesh.points,
            fem_interface,
            interface_symmetry_axes,
            symmetry_plane_tolerance,
        )
        old_bem_opening_path = _physical_opening_path(
            bem_mesh.points,
            bem_interface,
            interface_symmetry_axes,
            symmetry_plane_tolerance,
        )
    else:
        fem_opening_path = fem_inner_loop
        old_bem_opening_path = old_bem_inner_loop

    direct_seam_mapping, direct_seam_deviation = _discrete_path_correspondence(
        fem_mesh.points,
        fem_opening_path,
        bem_mesh.points,
        old_bem_opening_path,
        tolerance=resolved_seam_tolerance,
        closed=not interface_symmetry_axes,
    )
    if direct_seam_mapping is not None:
        output_mesh, interface_orientation_flipped = _replace_bem_interface_directly(
            fem_mesh,
            bem_mesh,
            fem_interface,
            bem_data,
            bem_interface_mask,
            bem_interface_tag,
            fem_seam_path=fem_opening_path,
            bem_seam_path=old_bem_opening_path,
            fem_to_bem_path_indices=direct_seam_mapping,
        )
        identity = validate_conforming_interfaces(
            fem_mesh,
            output_mesh,
            fem_interface_name=fem_interface_name,
            bem_interface_name=bem_interface_name,
            coordinate_tolerance=max(merge_tolerance * 10.0, 1e-9),
            require_closed_bem=True,
            symmetry_mode=symmetry,
        )
        result = InterfaceConformResult(
            fem_interface_triangles=len(fem_interface),
            original_bem_interface_triangles=len(bem_interface),
            original_adjacent_triangles=0,
            remeshed_adjacent_triangles=0,
            output_vertices=len(output_mesh.points),
            output_triangles=len(_triangle_data(output_mesh, require_geometrical=True).triangles),
            max_original_boundary_deviation=direct_seam_deviation,
            interface_orientation_flipped=interface_orientation_flipped,
            identity=identity,
        )
        return output_mesh, result

    seam_plane_origin, seam_plane_normal = _best_fit_plane(bem_mesh.points[old_bem_opening_path])
    seam_plane_deviation = _maximum_plane_deviation(
        seam_plane_origin,
        seam_plane_normal,
        bem_mesh.points[old_bem_opening_path],
        fem_mesh.points[fem_opening_path],
    )
    seam_simplification = (
        _ordered_seam_simplification_correspondence(
            fem_mesh.points,
            fem_opening_path,
            bem_mesh.points,
            old_bem_opening_path,
            tolerance=resolved_geometry_tolerance,
            closed=not interface_symmetry_axes,
        )
        if seam_plane_deviation > resolved_geometry_tolerance
        else None
    )
    if seam_simplification is not None:
        aligned_bem_path, bem_to_fem_path_indices, seam_deviation = seam_simplification
        output_mesh, interface_orientation_flipped, simplified_edges = _replace_bem_interface_with_simplified_seam(
            fem_mesh,
            bem_mesh,
            fem_interface,
            bem_data,
            bem_interface_mask,
            bem_interface_tag,
            fem_seam_path=fem_opening_path,
            bem_seam_path=aligned_bem_path,
            bem_to_fem_path_indices=bem_to_fem_path_indices,
            merge_tolerance=merge_tolerance,
            closed=not interface_symmetry_axes,
            protected_bem_tags=protected_bem_tags,
        )
        identity = validate_conforming_interfaces(
            fem_mesh,
            output_mesh,
            fem_interface_name=fem_interface_name,
            bem_interface_name=bem_interface_name,
            coordinate_tolerance=max(merge_tolerance * 10.0, 1e-9),
            require_closed_bem=True,
            symmetry_mode=symmetry,
        )
        result = InterfaceConformResult(
            fem_interface_triangles=len(fem_interface),
            original_bem_interface_triangles=len(bem_interface),
            original_adjacent_triangles=0,
            remeshed_adjacent_triangles=0,
            output_vertices=len(output_mesh.points),
            output_triangles=len(_triangle_data(output_mesh, require_geometrical=True).triangles),
            max_original_boundary_deviation=seam_deviation,
            interface_orientation_flipped=interface_orientation_flipped,
            identity=identity,
            seam_simplification_used=True,
            simplified_bem_seam_edges=simplified_edges,
        )
        return output_mesh, result

    plane_origin, perimeter_normal = _best_fit_plane(bem_mesh.points[old_bem_opening_path])
    triangle_plane_deviation = np.max(
        np.abs((bem_mesh.points[bem_data.triangles] - plane_origin) @ perimeter_normal),
        axis=1,
    )
    protected_mask = np.isin(
        bem_data.physical_tags,
        np.asarray(sorted(protected_bem_tags), dtype=bem_data.physical_tags.dtype),
    )
    adjacent_mask = (
        (~bem_interface_mask) & (~protected_mask) & (triangle_plane_deviation <= resolved_geometry_tolerance)
    )
    if not np.any(adjacent_mask):
        raise InterfaceConformError(
            "Could not find a planar non-interface BEM surface surrounding the interface perimeter. "
            "A curved interface seam can be replaced only when its FEM and BEM vertices and edges "
            "already match within merge_tolerance."
        )

    adjacent_triangles = bem_data.triangles[adjacent_mask]
    adjacent_physical_tags = np.unique(bem_data.physical_tags[adjacent_mask])
    if len(adjacent_physical_tags) != 1:
        raise InterfaceConformError("The surrounding BEM surface must have exactly one physical tag.")
    adjacent_physical_tag = int(adjacent_physical_tags[0])

    adjacent_loops = _boundary_loops(adjacent_triangles)
    if interface_symmetry_axes:
        if len(adjacent_loops) != 1:
            raise InterfaceConformError(
                "The planar reduced BEM surface surrounding the interface must have exactly one perimeter loop."
            )
        adjacent_loop = adjacent_loops[0]
        adjacent_edges = np.column_stack((adjacent_loop, np.roll(adjacent_loop, -1)))
        adjacent_midpoints = np.mean(bem_mesh.points[adjacent_edges], axis=1)
        opening_distances = _points_to_open_polyline_distances(
            adjacent_midpoints,
            bem_mesh.points[old_bem_opening_path],
        )
        inner_edge_mask = opening_distances <= resolved_geometry_tolerance
        if not np.any(inner_edge_mask) or np.all(inner_edge_mask):
            raise InterfaceConformError("Could not separate the reduced BEM opening from its exterior perimeter.")
        adjacent_inner_path = _ordered_open_path(adjacent_edges[inner_edge_mask])
        adjacent_outer_path = _ordered_open_path(adjacent_edges[~inner_edge_mask])
    else:
        if len(adjacent_loops) != 2:
            raise InterfaceConformError(
                "The planar BEM surface surrounding the interface must be an annulus with exactly two perimeter loops."
            )
        loop_deviations = [
            _symmetric_polyline_deviation(
                bem_mesh.points[loop],
                bem_mesh.points[old_bem_inner_loop],
            )
            for loop in adjacent_loops
        ]
        inner_loop_index = int(np.argmin(loop_deviations))
        adjacent_inner_path = adjacent_loops[inner_loop_index]
        adjacent_outer_path = adjacent_loops[1 - inner_loop_index]

    adjacent_normal = _surface_normal(bem_mesh.points, adjacent_triangles)
    old_interface_normal = _surface_normal(bem_mesh.points, bem_interface)
    fem_interface_normal = _surface_normal(fem_mesh.points, fem_interface)
    plane_deviation = _maximum_plane_deviation(
        plane_origin,
        adjacent_normal,
        bem_mesh.points[adjacent_outer_path],
        bem_mesh.points[old_bem_opening_path],
        fem_mesh.points[fem_opening_path],
    )

    if plane_deviation > resolved_geometry_tolerance:
        raise InterfaceConformError(
            "The FEM and BEM interface perimeters do not lie on the surrounding BEM surface plane: "
            f"maximum perimeter plane deviation {plane_deviation:g} exceeds tolerance "
            f"{resolved_geometry_tolerance:g}."
        )

    deviation_function = (
        _symmetric_open_polyline_deviation if interface_symmetry_axes else _symmetric_polyline_deviation
    )
    interface_boundary_deviation = deviation_function(
        fem_mesh.points[fem_opening_path],
        bem_mesh.points[old_bem_opening_path],
    )
    surrounding_boundary_deviation = deviation_function(
        fem_mesh.points[fem_opening_path],
        bem_mesh.points[adjacent_inner_path],
    )
    boundary_deviation = max(interface_boundary_deviation, surrounding_boundary_deviation)
    if boundary_deviation > resolved_geometry_tolerance:
        raise InterfaceConformError(
            "The FEM interface, BEM interface, and surrounding opening do not describe the same perimeter: "
            f"maximum deviation {boundary_deviation:g} exceeds tolerance "
            f"{resolved_geometry_tolerance:g}."
        )

    if interface_symmetry_axes:
        outer_coordinates, fem_inner_coordinates = _aligned_open_paths(
            bem_mesh.points[adjacent_outer_path],
            fem_mesh.points[fem_opening_path],
        )
        polygon_coordinates = _closed_polygon_from_paths(
            outer_coordinates,
            fem_inner_coordinates,
            merge_tolerance,
        )
        polygon_coordinates = _oriented_planar_loop(polygon_coordinates, adjacent_normal)
        annulus_points, annulus_triangles = _remesh_planar_surface((polygon_coordinates,))
    else:
        outer_coordinates, fem_inner_coordinates = _oriented_annulus_loops(
            bem_mesh.points[adjacent_outer_path],
            fem_mesh.points[fem_opening_path],
            adjacent_normal,
        )
        annulus_points, annulus_triangles = _remesh_planar_annulus(
            outer_coordinates,
            fem_inner_coordinates,
        )
    annulus_triangles = _orient_triangles(
        annulus_points,
        annulus_triangles,
        adjacent_normal,
    )

    interface_orientation_flipped = bool(np.dot(fem_interface_normal, old_interface_normal) < 0.0)
    copied_interface = np.asarray(fem_interface, dtype=np.int64).copy()
    if interface_orientation_flipped:
        copied_interface = copied_interface[:, [0, 2, 1]]

    keep_mask = ~(bem_interface_mask | adjacent_mask)
    kept_triangles = bem_data.triangles[keep_mask]
    kept_physical = bem_data.physical_tags[keep_mask]
    kept_geometrical = bem_data.geometrical_tags[keep_mask]

    annulus_offset = len(bem_mesh.points)
    fem_offset = annulus_offset + len(annulus_points)
    combined_points = np.vstack(
        (
            np.asarray(bem_mesh.points, dtype=float),
            annulus_points,
            np.asarray(fem_mesh.points, dtype=float),
        )
    )
    combined_triangles = np.vstack(
        (
            kept_triangles,
            annulus_triangles + annulus_offset,
            copied_interface + fem_offset,
        )
    )
    combined_physical = np.concatenate(
        (
            kept_physical,
            np.full(len(annulus_triangles), adjacent_physical_tag, dtype=np.int32),
            np.full(len(copied_interface), bem_interface_tag, dtype=np.int32),
        )
    )
    adjacent_geometrical_tag = int(np.min(bem_data.geometrical_tags[adjacent_mask]))
    interface_geometrical_tag = int(np.min(bem_data.geometrical_tags[bem_interface_mask]))
    combined_geometrical = np.concatenate(
        (
            kept_geometrical,
            np.full(len(annulus_triangles), adjacent_geometrical_tag, dtype=np.int32),
            np.full(len(copied_interface), interface_geometrical_tag, dtype=np.int32),
        )
    )

    merged_points, merged_triangles = _merge_and_compact_points(
        combined_points,
        combined_triangles,
        merge_tolerance,
    )
    output_mesh = meshio.Mesh(
        points=merged_points,
        cells=[("triangle", merged_triangles)],
        cell_data={
            "gmsh:physical": [combined_physical],
            "gmsh:geometrical": [combined_geometrical],
        },
        field_data={name: np.asarray(value).copy() for name, value in bem_mesh.field_data.items()},
    )
    identity = validate_conforming_interfaces(
        fem_mesh,
        output_mesh,
        fem_interface_name=fem_interface_name,
        bem_interface_name=bem_interface_name,
        coordinate_tolerance=max(merge_tolerance * 10.0, 1e-9),
        require_closed_bem=True,
        symmetry_mode=symmetry,
    )
    result = InterfaceConformResult(
        fem_interface_triangles=len(fem_interface),
        original_bem_interface_triangles=len(bem_interface),
        original_adjacent_triangles=len(adjacent_triangles),
        remeshed_adjacent_triangles=len(annulus_triangles),
        output_vertices=len(merged_points),
        output_triangles=len(merged_triangles),
        max_original_boundary_deviation=boundary_deviation,
        interface_orientation_flipped=interface_orientation_flipped,
        identity=identity,
    )
    return output_mesh, result


def conform_interface_mesh_files(
    fem_mesh_path: str | Path,
    bem_mesh_path: str | Path,
    output_mesh_path: str | Path,
    *,
    fem_interface_name: str = "Interface",
    bem_interface_name: str = "Interface",
    geometry_tolerance: float | None = None,
    seam_tolerance: float | None = None,
    merge_tolerance: float = 1e-8,
    symmetry_mode: str = "off",
    binary: bool = False,
) -> InterfaceConformResult:
    """Conform two mesh files and write a Gmsh 2.2 BEM surface mesh."""

    fem_path = Path(fem_mesh_path)
    bem_path = Path(bem_mesh_path)
    output_path = Path(output_mesh_path)
    fem_mesh = meshio.read(fem_path)
    bem_mesh = meshio.read(bem_path)
    output_mesh, result = conform_bem_interface_to_fem(
        fem_mesh,
        bem_mesh,
        fem_interface_name=fem_interface_name,
        bem_interface_name=bem_interface_name,
        geometry_tolerance=geometry_tolerance,
        seam_tolerance=seam_tolerance,
        merge_tolerance=merge_tolerance,
        symmetry_mode=symmetry_mode,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(output_path, output_mesh, file_format="gmsh22", binary=binary)
    return result


def validate_conforming_interfaces(
    fem_mesh: meshio.Mesh,
    bem_mesh: meshio.Mesh,
    *,
    fem_interface_name: str = "Interface",
    bem_interface_name: str = "Interface",
    coordinate_tolerance: float = 1e-8,
    require_closed_bem: bool = True,
    symmetry_mode: str = "off",
) -> InterfaceIdentityReport:
    """Validate coordinate/connectivity identity and surrounding topology."""

    topology = build_conforming_interface_map(
        fem_mesh,
        bem_mesh,
        fem_interface_name=fem_interface_name,
        bem_interface_name=bem_interface_name,
        coordinate_tolerance=coordinate_tolerance,
        require_closed_bem=require_closed_bem,
        symmetry_mode=symmetry_mode,
    )
    return InterfaceIdentityReport(
        interface_triangles=len(topology.fem_face_indices),
        interface_vertices=len(topology.fem_vertex_indices),
        max_coordinate_error=topology.max_coordinate_error,
        fem_facets_on_tetra_boundary=topology.fem_facets_on_tetra_boundary,
        bem_boundary_edges=topology.bem_boundary_edges,
    )


def build_conforming_interface_map(
    fem_mesh: meshio.Mesh,
    bem_mesh: meshio.Mesh,
    *,
    fem_interface_name: str = "Interface",
    bem_interface_name: str = "Interface",
    coordinate_tolerance: float = 1e-8,
    require_closed_bem: bool = True,
    symmetry_mode: str = "off",
) -> InterfaceTopologyMap:
    """Validate and return the explicit FEM-to-BEM interface correspondence."""

    symmetry = _normalized_symmetry_mode(symmetry_mode)
    symmetry_axes = _symmetry_axes(symmetry)
    if coordinate_tolerance <= 0.0:
        raise InterfaceConformError("coordinate_tolerance must be greater than zero.")
    symmetry_plane_tolerance = max(
        coordinate_tolerance,
        _point_cloud_diameter(bem_mesh.points) * 1e-6,
        1e-12,
    )
    if symmetry_axes:
        _validate_positive_symmetry_domain(
            fem_mesh.points,
            symmetry_axes,
            symmetry_plane_tolerance,
            "FEM",
        )
        _validate_positive_symmetry_domain(
            bem_mesh.points,
            symmetry_axes,
            symmetry_plane_tolerance,
            "BEM",
        )
    fem_tag = _physical_surface_tag(fem_mesh, fem_interface_name)
    bem_tag = _physical_surface_tag(bem_mesh, bem_interface_name)
    fem_data = _triangle_data(fem_mesh, require_geometrical=False)
    bem_data = _triangle_data(bem_mesh, require_geometrical=False)
    fem_triangles = fem_data.triangles[fem_data.physical_tags == fem_tag]
    bem_triangles = bem_data.triangles[bem_data.physical_tags == bem_tag]
    if len(fem_triangles) != len(bem_triangles):
        raise InterfaceConformError(
            f"Interface triangle counts differ: FEM has {len(fem_triangles)}, BEM has {len(bem_triangles)}."
        )

    fem_vertices = np.unique(fem_triangles)
    bem_vertices = np.unique(bem_triangles)
    if len(fem_vertices) != len(bem_vertices):
        raise InterfaceConformError(
            f"Interface vertex counts differ: FEM has {len(fem_vertices)}, BEM has {len(bem_vertices)}."
        )
    distances, local_indices = cKDTree(bem_mesh.points[bem_vertices]).query(fem_mesh.points[fem_vertices])
    max_coordinate_error = float(np.max(distances, initial=0.0))
    if max_coordinate_error > coordinate_tolerance:
        raise InterfaceConformError(
            f"Interface coordinate error {max_coordinate_error:g} exceeds tolerance {coordinate_tolerance:g}."
        )
    mapped_bem_vertices = bem_vertices[np.asarray(local_indices, dtype=np.int64)]
    if len(np.unique(mapped_bem_vertices)) != len(mapped_bem_vertices):
        raise InterfaceConformError("FEM interface vertices do not map one-to-one onto BEM interface vertices.")

    fem_to_bem = dict(zip(map(int, fem_vertices), map(int, mapped_bem_vertices), strict=True))
    bem_interface_indices = np.flatnonzero(bem_data.physical_tags == bem_tag)
    bem_face_by_key: dict[tuple[int, int, int], int] = {}
    for face_index, triangle in zip(bem_interface_indices, bem_triangles, strict=True):
        key = tuple(sorted(map(int, triangle)))
        if key in bem_face_by_key:
            raise InterfaceConformError("BEM interface contains duplicate triangle connectivity.")
        bem_face_by_key[key] = int(face_index)

    fem_interface_indices = np.flatnonzero(fem_data.physical_tags == fem_tag)
    corresponding_bem_faces: list[int] = []
    normal_signs: list[int] = []
    for triangle in fem_triangles:
        mapped_key = tuple(sorted(fem_to_bem[int(vertex)] for vertex in triangle))
        bem_face_index = bem_face_by_key.get(mapped_key)
        if bem_face_index is None:
            raise InterfaceConformError("FEM and BEM interface triangle connectivity differs.")
        corresponding_bem_faces.append(bem_face_index)

        fem_normal = _triangle_normal(fem_mesh.points, triangle)
        bem_normal = _triangle_normal(bem_mesh.points, bem_data.triangles[bem_face_index])
        normal_signs.append(1 if float(np.dot(fem_normal, bem_normal)) >= 0.0 else -1)

    if len(corresponding_bem_faces) != len(bem_face_by_key):
        raise InterfaceConformError("FEM and BEM interface triangle connectivity differs.")

    tetra_boundary = _tetra_boundary_faces(fem_mesh)
    fem_face_keys = {tuple(sorted(map(int, triangle))) for triangle in fem_triangles}
    matched_tetra_facets = len(fem_face_keys & tetra_boundary)
    if matched_tetra_facets != len(fem_face_keys):
        raise InterfaceConformError(
            f"Only {matched_tetra_facets}/{len(fem_face_keys)} FEM interface triangles are tetrahedron boundary facets."
        )

    boundary_edges = _surface_boundary_edges(bem_data.triangles)
    bem_boundary_edges = len(boundary_edges)
    if require_closed_bem and bem_boundary_edges != 0:
        invalid_boundary_edges = _edges_off_symmetry_planes(
            bem_mesh.points,
            boundary_edges,
            symmetry_axes,
            symmetry_plane_tolerance,
        )
        if len(invalid_boundary_edges):
            raise InterfaceConformError(
                f"Conformed BEM surface has {len(invalid_boundary_edges)} open boundary edges "
                "away from the active symmetry planes."
            )
    return InterfaceTopologyMap(
        fem_vertex_indices=tuple(map(int, fem_vertices)),
        fem_to_bem_vertex_indices=tuple(fem_to_bem[int(vertex)] for vertex in fem_vertices),
        fem_face_indices=tuple(map(int, fem_interface_indices)),
        bem_face_indices=tuple(corresponding_bem_faces),
        normal_sign=tuple(normal_signs),
        max_coordinate_error=max_coordinate_error,
        fem_facets_on_tetra_boundary=matched_tetra_facets,
        bem_boundary_edges=bem_boundary_edges,
    )


def _triangle_normal(points: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    vertices = np.asarray(points, dtype=float)[np.asarray(triangle, dtype=np.int64)]
    normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    magnitude = float(np.linalg.norm(normal))
    if magnitude <= 0.0:
        raise InterfaceConformError("Interface contains a degenerate triangle.")
    return normal / magnitude


def _physical_surface_tag(mesh: meshio.Mesh, name: str) -> int:
    value = mesh.field_data.get(name)
    if value is None:
        available = ", ".join(sorted(mesh.field_data)) or "(none)"
        raise InterfaceConformError(f"Physical surface '{name}' not found. Available groups: {available}.")
    tag, dimension = map(int, np.asarray(value).tolist())
    if dimension != 2:
        raise InterfaceConformError(f"Physical group '{name}' has dimension {dimension}, expected surface dimension 2.")
    return tag


def _triangle_data(mesh: meshio.Mesh, *, require_geometrical: bool) -> _TriangleData:
    triangles = []
    physical_tags = []
    geometrical_tags = []
    physical_blocks = mesh.cell_data.get("gmsh:physical")
    geometrical_blocks = mesh.cell_data.get("gmsh:geometrical")
    if physical_blocks is None:
        raise InterfaceConformError("Mesh triangles do not contain gmsh:physical tags.")
    if require_geometrical and geometrical_blocks is None:
        raise InterfaceConformError("BEM triangles do not contain gmsh:geometrical tags.")

    for index, cell_block in enumerate(mesh.cells):
        if cell_block.type not in {"triangle", "triangle3"}:
            continue
        block = np.asarray(cell_block.data, dtype=np.int64)
        physical = np.asarray(physical_blocks[index], dtype=np.int32)
        if len(physical) != len(block):
            raise InterfaceConformError("Triangle physical-tag data length does not match the triangle block.")
        triangles.append(block)
        physical_tags.append(physical)
        if geometrical_blocks is None:
            geometrical_tags.append(np.zeros(len(block), dtype=np.int32))
        else:
            geometrical = np.asarray(geometrical_blocks[index], dtype=np.int32)
            if len(geometrical) != len(block):
                raise InterfaceConformError("Triangle geometrical-tag data length does not match the triangle block.")
            geometrical_tags.append(geometrical)
    if not triangles:
        raise InterfaceConformError("Mesh contains no triangular surface elements.")
    return _TriangleData(
        triangles=np.vstack(triangles),
        physical_tags=np.concatenate(physical_tags),
        geometrical_tags=np.concatenate(geometrical_tags),
    )


def _normalized_symmetry_mode(mode: object) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in {"off", "x", "xy"}:
        raise InterfaceConformError(f"Unsupported symmetry mode {mode!r}; expected 'off', 'x', or 'xy'.")
    return normalized


def _symmetry_axes(mode: str) -> tuple[int, ...]:
    return () if mode == "off" else (0,) if mode == "x" else (0, 1)


def _perimeter_symmetry_axes(
    points: np.ndarray,
    perimeter: np.ndarray,
    active_axes: tuple[int, ...],
    tolerance: float,
) -> tuple[int, ...]:
    loop = np.asarray(perimeter, dtype=np.int64)
    edges = np.column_stack((loop, np.roll(loop, -1)))
    coordinates = np.asarray(points, dtype=float)
    return tuple(axis for axis in active_axes if np.any(np.max(np.abs(coordinates[edges, axis]), axis=1) <= tolerance))


def _validate_positive_symmetry_domain(
    points: np.ndarray,
    axes: tuple[int, ...],
    tolerance: float,
    label: str,
) -> None:
    coordinates = np.asarray(points, dtype=float)
    axis_names = ("X", "Y", "Z")
    for axis in axes:
        minimum = float(np.min(coordinates[:, axis], initial=0.0))
        if minimum < -tolerance:
            raise InterfaceConformError(
                f"{label} mesh extends to {minimum:g} on the negative {axis_names[axis]} axis "
                "outside the requested symmetry fundamental domain."
            )


def _edges_off_symmetry_planes(
    points: np.ndarray,
    edges: np.ndarray,
    axes: tuple[int, ...],
    tolerance: float,
) -> np.ndarray:
    candidates = np.asarray(edges, dtype=np.int64).reshape((-1, 2))
    if not axes or len(candidates) == 0:
        return candidates
    coordinates = np.asarray(points, dtype=float)
    on_symmetry_plane = np.zeros(len(candidates), dtype=bool)
    for axis in axes:
        on_symmetry_plane |= np.max(np.abs(coordinates[candidates, axis]), axis=1) <= tolerance
    return candidates[~on_symmetry_plane]


def _physical_opening_path(
    points: np.ndarray,
    triangles: np.ndarray,
    symmetry_axes: tuple[int, ...],
    tolerance: float,
) -> np.ndarray:
    physical_edges = _edges_off_symmetry_planes(
        points,
        _surface_boundary_edges(triangles),
        symmetry_axes,
        tolerance,
    )
    if len(physical_edges) == 0:
        raise InterfaceConformError("Reduced interface has no perimeter away from its symmetry planes.")
    return _ordered_open_path(physical_edges)


def _ordered_open_path(edges: np.ndarray) -> np.ndarray:
    candidates = np.asarray(edges, dtype=np.int64).reshape((-1, 2))
    if len(candidates) == 0:
        raise InterfaceConformError("Cannot order an empty open boundary path.")
    adjacency: dict[int, list[int]] = defaultdict(list)
    for start, end in candidates:
        adjacency[int(start)].append(int(end))
        adjacency[int(end)].append(int(start))
    endpoints = sorted(vertex for vertex, neighbors in adjacency.items() if len(neighbors) == 1)
    invalid = [vertex for vertex, neighbors in adjacency.items() if len(neighbors) not in {1, 2}]
    if len(endpoints) != 2 or invalid:
        raise InterfaceConformError("Reduced interface opening is not one simple open perimeter path.")

    ordered = [endpoints[0]]
    previous = None
    current = endpoints[0]
    while current != endpoints[1]:
        next_vertices = [neighbor for neighbor in adjacency[current] if neighbor != previous]
        if len(next_vertices) != 1:
            raise InterfaceConformError("Could not order the reduced interface opening path.")
        next_vertex = next_vertices[0]
        ordered.append(next_vertex)
        previous, current = current, next_vertex
    if len(ordered) != len(candidates) + 1:
        raise InterfaceConformError("Reduced interface opening contains disconnected perimeter edges.")
    return np.asarray(ordered, dtype=np.int64)


def _boundary_loops(triangles: np.ndarray) -> list[np.ndarray]:
    boundary_edges = _surface_boundary_edges(triangles)
    adjacency: dict[int, list[int]] = defaultdict(list)
    for start, end in boundary_edges:
        adjacency[int(start)].append(int(end))
        adjacency[int(end)].append(int(start))
    invalid = [vertex for vertex, neighbors in adjacency.items() if len(neighbors) != 2]
    if invalid:
        raise InterfaceConformError(
            "Surface boundary is not a collection of simple closed loops; "
            f"{len(invalid)} boundary vertices do not have degree two."
        )

    remaining = {tuple(map(int, edge)) for edge in boundary_edges}
    loops = []
    while remaining:
        start, first_neighbor = next(iter(remaining))
        loop = [start]
        previous = None
        current = start
        while True:
            next_vertex = next(
                (
                    neighbor
                    for neighbor in adjacency[current]
                    if neighbor != previous and tuple(sorted((current, neighbor))) in remaining
                ),
                None,
            )
            if next_vertex is None:
                raise InterfaceConformError("Could not order a surface boundary loop.")
            remaining.remove(tuple(sorted((current, next_vertex))))
            if next_vertex == loop[0]:
                break
            loop.append(next_vertex)
            previous, current = current, next_vertex
        loops.append(np.asarray(loop, dtype=np.int64))
    return loops


def _surface_boundary_edges(triangles: np.ndarray) -> np.ndarray:
    edges = np.sort(
        np.vstack(
            (
                triangles[:, [0, 1]],
                triangles[:, [1, 2]],
                triangles[:, [2, 0]],
            )
        ),
        axis=1,
    )
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    if np.any(counts > 2):
        raise InterfaceConformError("Surface contains non-manifold edges with more than two incident triangles.")
    return unique_edges[counts == 1]


def _surface_normal(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    cross_products = np.cross(
        points[triangles[:, 1]] - points[triangles[:, 0]],
        points[triangles[:, 2]] - points[triangles[:, 0]],
    )
    normal = np.sum(cross_products, axis=0)
    magnitude = float(np.linalg.norm(normal))
    if magnitude <= 0.0:
        raise InterfaceConformError("Surface has zero aggregate normal.")
    return normal / magnitude


def _best_fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(points, dtype=float)
    if len(coordinates) < 3:
        raise InterfaceConformError("Interface perimeter requires at least three points.")
    origin = np.mean(coordinates, axis=0)
    _u, singular_values, vectors = np.linalg.svd(coordinates - origin, full_matrices=False)
    if len(singular_values) < 2 or singular_values[1] <= np.finfo(float).eps:
        raise InterfaceConformError("Interface perimeter points do not define a plane.")
    return origin, vectors[-1]


def _maximum_plane_deviation(origin: np.ndarray, normal: np.ndarray, *point_sets: np.ndarray) -> float:
    return max(
        (float(np.max(np.abs((points - origin) @ normal), initial=0.0)) for points in point_sets),
        default=0.0,
    )


def _point_cloud_diameter(points: np.ndarray) -> float:
    bounds = np.ptp(np.asarray(points, dtype=float), axis=0)
    return float(np.linalg.norm(bounds))


def _symmetric_polyline_deviation(first: np.ndarray, second: np.ndarray) -> float:
    return max(
        _points_to_closed_polyline_distance(first, second),
        _points_to_closed_polyline_distance(second, first),
    )


def _symmetric_open_polyline_deviation(first: np.ndarray, second: np.ndarray) -> float:
    return max(
        float(np.max(_points_to_open_polyline_distances(first, second), initial=0.0)),
        float(np.max(_points_to_open_polyline_distances(second, first), initial=0.0)),
    )


def _points_to_open_polyline_distances(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(polyline, dtype=float)
    if len(coordinates) < 2:
        raise InterfaceConformError("Interface opening requires at least two perimeter points.")
    starts = coordinates[:-1]
    ends = coordinates[1:]
    segments = ends - starts
    segment_length_sq = np.sum(segments * segments, axis=1)
    if np.any(segment_length_sq <= 0.0):
        raise InterfaceConformError("Interface opening contains a zero-length segment.")
    relative = np.asarray(points, dtype=float)[:, np.newaxis, :] - starts[np.newaxis, :, :]
    parameters = np.sum(relative * segments[np.newaxis, :, :], axis=2) / segment_length_sq[np.newaxis, :]
    parameters = np.clip(parameters, 0.0, 1.0)
    return np.min(np.linalg.norm(relative - parameters[:, :, np.newaxis] * segments, axis=2), axis=1)


def _points_to_closed_polyline_distance(points: np.ndarray, polyline: np.ndarray) -> float:
    starts = polyline
    ends = np.roll(polyline, -1, axis=0)
    segments = ends - starts
    segment_length_sq = np.sum(segments * segments, axis=1)
    if np.any(segment_length_sq <= 0.0):
        raise InterfaceConformError("Interface perimeter contains a zero-length segment.")
    relative = points[:, np.newaxis, :] - starts[np.newaxis, :, :]
    parameters = np.sum(relative * segments[np.newaxis, :, :], axis=2) / segment_length_sq[np.newaxis, :]
    parameters = np.clip(parameters, 0.0, 1.0)
    projections = starts[np.newaxis, :, :] + parameters[:, :, np.newaxis] * segments[np.newaxis, :, :]
    distances = np.linalg.norm(points[:, np.newaxis, :] - projections, axis=2)
    return float(np.max(np.min(distances, axis=1), initial=0.0))


def _matching_discrete_paths(
    first_points: np.ndarray,
    first_path: np.ndarray,
    second_points: np.ndarray,
    second_path: np.ndarray,
    *,
    tolerance: float,
    closed: bool,
) -> tuple[bool, float]:
    """Return whether two paths have bijective vertices and identical edges."""

    correspondence, maximum_deviation = _discrete_path_correspondence(
        first_points,
        first_path,
        second_points,
        second_path,
        tolerance=tolerance,
        closed=closed,
    )
    return correspondence is not None, maximum_deviation


def _discrete_path_correspondence(
    first_points: np.ndarray,
    first_path: np.ndarray,
    second_points: np.ndarray,
    second_path: np.ndarray,
    *,
    tolerance: float,
    closed: bool,
) -> tuple[np.ndarray | None, float]:
    """Return first-to-second path indices for a bijective edge-preserving match."""

    first_coordinates = np.asarray(first_points, dtype=float)[np.asarray(first_path, dtype=np.int64)]
    second_coordinates = np.asarray(second_points, dtype=float)[np.asarray(second_path, dtype=np.int64)]
    if len(first_coordinates) != len(second_coordinates):
        return None, float("inf")

    first_distances, first_to_second = cKDTree(second_coordinates).query(first_coordinates, k=1)
    second_distances, second_to_first = cKDTree(first_coordinates).query(second_coordinates, k=1)
    maximum_deviation = max(
        float(np.max(first_distances, initial=0.0)),
        float(np.max(second_distances, initial=0.0)),
    )
    if maximum_deviation > tolerance:
        return None, maximum_deviation
    if len(np.unique(first_to_second)) != len(first_coordinates):
        return None, maximum_deviation
    if len(np.unique(second_to_first)) != len(second_coordinates):
        return None, maximum_deviation
    if not np.array_equal(
        second_to_first[first_to_second],
        np.arange(len(first_coordinates)),
    ):
        return None, maximum_deviation

    def path_edges(vertex_count: int) -> set[tuple[int, int]]:
        stop = vertex_count if closed else vertex_count - 1
        return {tuple(sorted((index, (index + 1) % vertex_count))) for index in range(stop)}

    mapped_first_edges = {
        tuple(sorted((int(first_to_second[start]), int(first_to_second[end]))))
        for start, end in path_edges(len(first_coordinates))
    }
    if mapped_first_edges != path_edges(len(second_coordinates)):
        return None, maximum_deviation
    return np.asarray(first_to_second, dtype=np.int64), maximum_deviation


def _ordered_seam_simplification_correspondence(
    fem_points: np.ndarray,
    fem_path: np.ndarray,
    bem_points: np.ndarray,
    bem_path: np.ndarray,
    *,
    tolerance: float,
    closed: bool,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Map a more finely divided BEM seam onto an ordered FEM seam.

    This intentionally supports simplification only. Adding vertices to the
    surrounding BEM surface needs a separate edge-splitting operation and is
    not treated as an equivalent safe fallback.
    """

    fem_vertices = np.asarray(fem_path, dtype=np.int64)
    bem_vertices = np.asarray(bem_path, dtype=np.int64)
    if len(bem_vertices) <= len(fem_vertices):
        return None

    fem_coordinates = np.asarray(fem_points, dtype=float)[fem_vertices]
    bem_coordinates = np.asarray(bem_points, dtype=float)[bem_vertices]
    deviation_function = _symmetric_polyline_deviation if closed else _symmetric_open_polyline_deviation
    seam_deviation = deviation_function(fem_coordinates, bem_coordinates)
    if seam_deviation > tolerance:
        return None

    candidates = (bem_vertices, bem_vertices[::-1])
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for candidate_path in candidates:
        candidate_coordinates = np.asarray(bem_points, dtype=float)[candidate_path]
        vertex_distances, mapping = cKDTree(fem_coordinates).query(candidate_coordinates, k=1)
        mapping = np.asarray(mapping, dtype=np.int64)
        if set(map(int, mapping)) != set(range(len(fem_vertices))):
            continue
        if closed:
            increments = np.mod(np.roll(mapping, -1) - mapping, len(fem_vertices))
            if not np.all((increments == 0) | (increments == 1)):
                continue
        else:
            if mapping[0] != 0 or mapping[-1] != len(fem_vertices) - 1:
                continue
            increments = np.diff(mapping)
            if not np.all((increments == 0) | (increments == 1)):
                continue
        maximum_vertex_move = float(np.max(vertex_distances, initial=0.0))
        if best is None or maximum_vertex_move < best[2]:
            best = (candidate_path.copy(), mapping, maximum_vertex_move)

    if best is None:
        return None
    return best[0], best[1], seam_deviation


def _replace_bem_interface_with_simplified_seam(
    fem_mesh: meshio.Mesh,
    bem_mesh: meshio.Mesh,
    fem_interface: np.ndarray,
    bem_data: _TriangleData,
    bem_interface_mask: np.ndarray,
    bem_interface_tag: int,
    *,
    fem_seam_path: np.ndarray,
    bem_seam_path: np.ndarray,
    bem_to_fem_path_indices: np.ndarray,
    merge_tolerance: float,
    closed: bool,
    protected_bem_tags: set[int],
) -> tuple[meshio.Mesh, bool, int]:
    """Collapse redundant ordered BEM seam edges and copy exact FEM facets."""

    fem_path = np.asarray(fem_seam_path, dtype=np.int64)
    bem_path = np.asarray(bem_seam_path, dtype=np.int64)
    path_mapping = np.asarray(bem_to_fem_path_indices, dtype=np.int64)
    edge_stop = len(bem_path) if closed else len(bem_path) - 1
    collapsed_edge_positions = [
        index for index in range(edge_stop) if path_mapping[index] == path_mapping[(index + 1) % len(bem_path)]
    ]
    expected_collapses = len(bem_path) - len(fem_path)
    if len(collapsed_edge_positions) != expected_collapses or expected_collapses <= 0:
        raise InterfaceConformError(
            "Ordered seam simplification rejected: the redundant BEM vertices do not form consecutive edges."
        )

    keep_mask = ~bem_interface_mask
    kept_original_indices = np.flatnonzero(keep_mask)
    kept_original_triangles = bem_data.triangles[keep_mask]
    kept_boundary_edges = _surface_boundary_edges(kept_original_triangles)
    eligible_vertices = np.unique(np.concatenate((bem_path, kept_boundary_edges.ravel())))
    eligible_tree = cKDTree(np.asarray(bem_mesh.points, dtype=float)[eligible_vertices])

    point_map = np.arange(len(bem_mesh.points), dtype=np.int64)
    representative_by_fem_index: dict[int, int] = {}
    for fem_index in range(len(fem_path)):
        positions = np.flatnonzero(path_mapping == fem_index)
        if len(positions) == 0:
            raise InterfaceConformError("Ordered seam simplification rejected: the seam mapping is not onto.")
        representative_by_fem_index[fem_index] = int(bem_path[int(positions[0])])

    for position, bem_vertex in enumerate(bem_path):
        representative = representative_by_fem_index[int(path_mapping[position])]
        nearby = eligible_vertices[
            eligible_tree.query_ball_point(
                np.asarray(bem_mesh.points, dtype=float)[int(bem_vertex)],
                merge_tolerance,
            )
        ]
        for vertex in map(int, nearby):
            existing = int(point_map[vertex])
            if existing != vertex and existing != representative:
                raise InterfaceConformError(
                    "Ordered seam simplification rejected: a BEM seam vertex has conflicting matches."
                )
            point_map[vertex] = representative

    simplified_points = np.asarray(bem_mesh.points, dtype=float).copy()
    for fem_index, representative in representative_by_fem_index.items():
        simplified_points[representative] = np.asarray(fem_mesh.points, dtype=float)[fem_path[fem_index]]

    mapped_kept_triangles = point_map[kept_original_triangles]
    degenerate_mask = (
        (mapped_kept_triangles[:, 0] == mapped_kept_triangles[:, 1])
        | (mapped_kept_triangles[:, 1] == mapped_kept_triangles[:, 2])
        | (mapped_kept_triangles[:, 0] == mapped_kept_triangles[:, 2])
    )
    if int(np.count_nonzero(degenerate_mask)) != expected_collapses:
        raise InterfaceConformError(
            "Ordered seam simplification rejected: collapsing the redundant seam did not remove exactly "
            "one surrounding triangle per redundant edge."
        )
    if np.any(np.isin(bem_data.physical_tags[kept_original_indices[degenerate_mask]], tuple(protected_bem_tags))):
        raise InterfaceConformError(
            "Ordered seam simplification rejected: it would modify another protected interface surface."
        )

    retained_original_triangles = kept_original_triangles[~degenerate_mask]
    retained_triangles = mapped_kept_triangles[~degenerate_mask]
    old_cross = np.cross(
        np.asarray(bem_mesh.points, dtype=float)[retained_original_triangles[:, 1]]
        - np.asarray(bem_mesh.points, dtype=float)[retained_original_triangles[:, 0]],
        np.asarray(bem_mesh.points, dtype=float)[retained_original_triangles[:, 2]]
        - np.asarray(bem_mesh.points, dtype=float)[retained_original_triangles[:, 0]],
    )
    new_cross = np.cross(
        simplified_points[retained_triangles[:, 1]] - simplified_points[retained_triangles[:, 0]],
        simplified_points[retained_triangles[:, 2]] - simplified_points[retained_triangles[:, 0]],
    )
    old_magnitudes = np.linalg.norm(old_cross, axis=1)
    new_magnitudes = np.linalg.norm(new_cross, axis=1)
    if np.any(old_magnitudes <= 0.0) or np.any(new_magnitudes <= 0.0):
        raise InterfaceConformError(
            "Ordered seam simplification rejected: it would create a zero-area surrounding triangle."
        )
    normal_alignment = np.sum(old_cross * new_cross, axis=1) / (old_magnitudes * new_magnitudes)
    area_ratio = new_magnitudes / old_magnitudes
    if np.any(normal_alignment <= 0.0):
        raise InterfaceConformError("Ordered seam simplification rejected: it would invert a surrounding triangle.")
    if np.any((area_ratio < 0.05) | (area_ratio > 20.0)):
        raise InterfaceConformError(
            "Ordered seam simplification rejected: surrounding triangle area would change by more than 20x."
        )

    old_bem_interface = bem_data.triangles[bem_interface_mask]
    fem_interface_normal = _surface_normal(fem_mesh.points, fem_interface)
    old_interface_normal = _surface_normal(bem_mesh.points, old_bem_interface)
    interface_orientation_flipped = bool(np.dot(fem_interface_normal, old_interface_normal) < 0.0)
    copied_interface = np.asarray(fem_interface, dtype=np.int64).copy()
    if interface_orientation_flipped:
        copied_interface = copied_interface[:, [0, 2, 1]]

    fem_vertex_map = np.full(len(fem_mesh.points), -1, dtype=np.int64)
    for fem_index, representative in representative_by_fem_index.items():
        fem_vertex_map[fem_path[fem_index]] = representative
    fem_vertices = np.unique(copied_interface)
    appended_fem_vertices = fem_vertices[fem_vertex_map[fem_vertices] < 0]
    appended_start = len(simplified_points)
    fem_vertex_map[appended_fem_vertices] = np.arange(
        appended_start,
        appended_start + len(appended_fem_vertices),
        dtype=np.int64,
    )
    combined_points = np.vstack((simplified_points, np.asarray(fem_mesh.points, dtype=float)[appended_fem_vertices]))
    copied_interface = fem_vertex_map[copied_interface]
    combined_triangles = np.vstack((retained_triangles, copied_interface))
    connectivity_keys = np.sort(combined_triangles, axis=1)
    if len(np.unique(connectivity_keys, axis=0)) != len(connectivity_keys):
        raise InterfaceConformError(
            "Ordered seam simplification rejected: it would create duplicate triangle connectivity."
        )

    retained_indices = kept_original_indices[~degenerate_mask]
    combined_physical = np.concatenate(
        (
            bem_data.physical_tags[retained_indices],
            np.full(len(copied_interface), bem_interface_tag, dtype=np.int32),
        )
    )
    interface_geometrical_tag = int(np.min(bem_data.geometrical_tags[bem_interface_mask]))
    combined_geometrical = np.concatenate(
        (
            bem_data.geometrical_tags[retained_indices],
            np.full(len(copied_interface), interface_geometrical_tag, dtype=np.int32),
        )
    )
    compact_points, compact_triangles = _compact_points(combined_points, combined_triangles)
    output_mesh = meshio.Mesh(
        points=compact_points,
        cells=[("triangle", compact_triangles)],
        cell_data={
            "gmsh:physical": [combined_physical],
            "gmsh:geometrical": [combined_geometrical],
        },
        field_data={name: np.asarray(value).copy() for name, value in bem_mesh.field_data.items()},
    )
    return output_mesh, interface_orientation_flipped, expected_collapses


def _replace_bem_interface_directly(
    fem_mesh: meshio.Mesh,
    bem_mesh: meshio.Mesh,
    fem_interface: np.ndarray,
    bem_data: _TriangleData,
    bem_interface_mask: np.ndarray,
    bem_interface_tag: int,
    *,
    fem_seam_path: np.ndarray,
    bem_seam_path: np.ndarray,
    fem_to_bem_path_indices: np.ndarray,
) -> tuple[meshio.Mesh, bool]:
    """Copy FEM facets while explicitly reusing and snapping matched BEM seam vertices."""

    old_bem_interface = bem_data.triangles[bem_interface_mask]
    fem_interface_normal = _surface_normal(fem_mesh.points, fem_interface)
    old_interface_normal = _surface_normal(bem_mesh.points, old_bem_interface)
    interface_orientation_flipped = bool(np.dot(fem_interface_normal, old_interface_normal) < 0.0)
    copied_interface = np.asarray(fem_interface, dtype=np.int64).copy()
    if interface_orientation_flipped:
        copied_interface = copied_interface[:, [0, 2, 1]]

    keep_mask = ~bem_interface_mask
    combined_points = np.asarray(bem_mesh.points, dtype=float).copy()
    fem_vertex_map = np.full(len(fem_mesh.points), -1, dtype=np.int64)
    fem_path = np.asarray(fem_seam_path, dtype=np.int64)
    bem_path = np.asarray(bem_seam_path, dtype=np.int64)
    path_mapping = np.asarray(fem_to_bem_path_indices, dtype=np.int64)
    mapped_bem_vertices = bem_path[path_mapping]
    fem_vertex_map[fem_path] = mapped_bem_vertices
    combined_points[mapped_bem_vertices] = np.asarray(fem_mesh.points, dtype=float)[fem_path]

    fem_vertices = np.unique(copied_interface)
    appended_fem_vertices = fem_vertices[fem_vertex_map[fem_vertices] < 0]
    appended_start = len(combined_points)
    fem_vertex_map[appended_fem_vertices] = np.arange(
        appended_start,
        appended_start + len(appended_fem_vertices),
        dtype=np.int64,
    )
    combined_points = np.vstack((combined_points, np.asarray(fem_mesh.points, dtype=float)[appended_fem_vertices]))
    copied_interface = fem_vertex_map[copied_interface]
    combined_triangles = np.vstack(
        (
            bem_data.triangles[keep_mask],
            copied_interface,
        )
    )
    combined_physical = np.concatenate(
        (
            bem_data.physical_tags[keep_mask],
            np.full(len(copied_interface), bem_interface_tag, dtype=np.int32),
        )
    )
    interface_geometrical_tag = int(np.min(bem_data.geometrical_tags[bem_interface_mask]))
    combined_geometrical = np.concatenate(
        (
            bem_data.geometrical_tags[keep_mask],
            np.full(
                len(copied_interface),
                interface_geometrical_tag,
                dtype=np.int32,
            ),
        )
    )

    compact_points, compact_triangles = _compact_points(
        combined_points,
        combined_triangles,
    )
    output_mesh = meshio.Mesh(
        points=compact_points,
        cells=[("triangle", compact_triangles)],
        cell_data={
            "gmsh:physical": [combined_physical],
            "gmsh:geometrical": [combined_geometrical],
        },
        field_data={name: np.asarray(value).copy() for name, value in bem_mesh.field_data.items()},
    )
    return output_mesh, interface_orientation_flipped


def _aligned_open_paths(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(first, dtype=float)
    candidate = np.asarray(second, dtype=float)
    same_direction = np.linalg.norm(reference[0] - candidate[0]) + np.linalg.norm(reference[-1] - candidate[-1])
    reverse_direction = np.linalg.norm(reference[0] - candidate[-1]) + np.linalg.norm(reference[-1] - candidate[0])
    return reference, candidate if same_direction <= reverse_direction else candidate[::-1]


def _closed_polygon_from_paths(
    outer_path: np.ndarray,
    inner_path: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    coordinates = np.vstack((outer_path, inner_path[::-1]))
    keep = np.ones(len(coordinates), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(coordinates, axis=0), axis=1) > tolerance
    coordinates = coordinates[keep]
    if len(coordinates) > 1 and np.linalg.norm(coordinates[-1] - coordinates[0]) <= tolerance:
        coordinates = coordinates[:-1]
    if len(coordinates) < 3:
        raise InterfaceConformError("Reduced surrounding surface does not define a planar polygon.")
    return coordinates


def _oriented_planar_loop(coordinates: np.ndarray, normal: np.ndarray) -> np.ndarray:
    loop = np.asarray(coordinates, dtype=float)
    polygon_normal = np.sum(np.cross(loop, np.roll(loop, -1, axis=0)), axis=0)
    if float(np.dot(polygon_normal, normal)) < 0.0:
        loop = loop[::-1]
    return loop


def _oriented_annulus_loops(
    outer_coordinates: np.ndarray,
    inner_coordinates: np.ndarray,
    normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    origin = np.mean(outer_coordinates, axis=0)
    first_edge = outer_coordinates[1] - outer_coordinates[0]
    first_edge -= normal * float(np.dot(first_edge, normal))
    first_edge_magnitude = float(np.linalg.norm(first_edge))
    if first_edge_magnitude <= 0.0:
        raise InterfaceConformError("Could not construct a basis for the planar BEM surface.")
    axis_u = first_edge / first_edge_magnitude
    axis_v = np.cross(normal, axis_u)

    def signed_area(coordinates: np.ndarray) -> float:
        relative = coordinates - origin
        x = relative @ axis_u
        y = relative @ axis_v
        return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))

    outer = np.asarray(outer_coordinates, dtype=float)
    inner = np.asarray(inner_coordinates, dtype=float)
    if signed_area(outer) < 0.0:
        outer = outer[::-1]
    if signed_area(inner) > 0.0:
        inner = inner[::-1]
    return outer, inner


def _remesh_planar_annulus(
    outer_coordinates: np.ndarray,
    inner_coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return _remesh_planar_surface((outer_coordinates, inner_coordinates))


def _remesh_planar_surface(
    boundary_loops: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import gmsh
    except ImportError as exc:
        raise InterfaceConformError("The gmsh Python package is required to remesh the BEM interface surface.") from exc

    was_initialized = bool(gmsh.isInitialized())
    previous_model = gmsh.model.getCurrent() if was_initialized else ""
    if not was_initialized:
        gmsh.initialize()
    previous_terminal = gmsh.option.getNumber("General.Terminal")
    previous_algorithm = gmsh.option.getNumber("Mesh.Algorithm")
    model_name = f"boundary_lab_interface_{uuid.uuid4().hex}"
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.add(model_name)

        def add_loop(coordinates: np.ndarray) -> int:
            segment_lengths = np.linalg.norm(np.roll(coordinates, -1, axis=0) - coordinates, axis=1)
            mesh_size = float(np.median(segment_lengths))
            point_tags = [
                gmsh.model.geo.addPoint(
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                    mesh_size,
                )
                for point in coordinates
            ]
            curve_tags = []
            for index, point_tag in enumerate(point_tags):
                curve_tag = gmsh.model.geo.addLine(point_tag, point_tags[(index + 1) % len(point_tags)])
                gmsh.model.geo.mesh.setTransfiniteCurve(curve_tag, 2)
                curve_tags.append(curve_tag)
            return gmsh.model.geo.addCurveLoop(curve_tags)

        curve_loops = [add_loop(coordinates) for coordinates in boundary_loops]
        surface = gmsh.model.geo.addPlaneSurface(curve_loops)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)

        node_tags, coordinates, _parameters = gmsh.model.mesh.getNodes()
        points = np.asarray(coordinates, dtype=float).reshape((-1, 3))
        element_types, _element_tags, element_nodes = gmsh.model.mesh.getElements(2, surface)
        triangle_indices = [index for index, element_type in enumerate(element_types) if int(element_type) == 2]
        if len(triangle_indices) != 1:
            raise InterfaceConformError("Gmsh did not generate one first-order triangle block for the BEM surface.")
        node_lookup = {int(tag): index for index, tag in enumerate(node_tags)}
        triangle_node_tags = np.asarray(element_nodes[triangle_indices[0]], dtype=np.int64).reshape((-1, 3))
        triangles = np.asarray(
            [[node_lookup[int(tag)] for tag in triangle] for triangle in triangle_node_tags],
            dtype=np.int64,
        )
        return points, triangles
    finally:
        if gmsh.model.getCurrent() == model_name:
            gmsh.model.remove()
        gmsh.option.setNumber("General.Terminal", previous_terminal)
        gmsh.option.setNumber("Mesh.Algorithm", previous_algorithm)
        if was_initialized:
            if previous_model:
                gmsh.model.setCurrent(previous_model)
        else:
            gmsh.finalize()


def _orient_triangles(points: np.ndarray, triangles: np.ndarray, target_normal: np.ndarray) -> np.ndarray:
    oriented = np.asarray(triangles, dtype=np.int64).copy()
    normal = _surface_normal(points, oriented)
    if np.dot(normal, target_normal) < 0.0:
        oriented = oriented[:, [0, 2, 1]]
    return oriented


def _merge_and_compact_points(
    points: np.ndarray,
    triangles: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    parent = np.arange(len(points), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(first: int, second: int) -> None:
        root_first = find(first)
        root_second = find(second)
        if root_first == root_second:
            return
        if root_first < root_second:
            parent[root_second] = root_first
        else:
            parent[root_first] = root_second

    for first, second in cKDTree(points).query_pairs(tolerance):
        union(int(first), int(second))
    roots = np.asarray([find(index) for index in range(len(points))], dtype=np.int64)
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    merged_points = np.asarray(points[unique_roots], dtype=float)
    merged_triangles = inverse[np.asarray(triangles, dtype=np.int64)]
    if np.any(
        (merged_triangles[:, 0] == merged_triangles[:, 1])
        | (merged_triangles[:, 1] == merged_triangles[:, 2])
        | (merged_triangles[:, 0] == merged_triangles[:, 2])
    ):
        raise InterfaceConformError("Point merging collapsed one or more output triangles.")

    return _compact_points(merged_points, merged_triangles)


def _compact_points(
    points: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove unused points without changing coordinates or merging topology."""

    coordinates = np.asarray(points, dtype=float)
    connectivity = np.asarray(triangles, dtype=np.int64)
    used = np.unique(connectivity)
    compact_index = np.full(len(coordinates), -1, dtype=np.int64)
    compact_index[used] = np.arange(len(used), dtype=np.int64)
    return coordinates[used], compact_index[connectivity]


def _tetra_boundary_faces(mesh: meshio.Mesh) -> set[tuple[int, int, int]]:
    tetrahedra = [
        np.asarray(cell_block.data, dtype=np.int64)
        for cell_block in mesh.cells
        if cell_block.type in {"tetra", "tetra4"}
    ]
    if not tetrahedra:
        raise InterfaceConformError("FEM mesh contains no first-order tetrahedra.")
    tetrahedra_array = np.vstack(tetrahedra)
    faces = np.sort(
        np.vstack(
            (
                tetrahedra_array[:, [0, 1, 2]],
                tetrahedra_array[:, [0, 1, 3]],
                tetrahedra_array[:, [0, 2, 3]],
                tetrahedra_array[:, [1, 2, 3]],
            )
        ),
        axis=1,
    )
    counts = Counter(map(tuple, faces.tolist()))
    return {face for face, count in counts.items() if count == 1}


def _build_arg_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Replace a BEM interface with matching authoritative FEM boundary facets.",
    )
    parser.add_argument("fem_mesh", help="Tetrahedral FEM .msh file")
    parser.add_argument("bem_mesh", help="Exterior BEM surface .msh file")
    parser.add_argument("output_mesh", help="Output conforming BEM .msh file")
    parser.add_argument("--fem-interface", default="Interface", help="FEM interface physical-group name")
    parser.add_argument("--bem-interface", default="Interface", help="BEM interface physical-group name")
    parser.add_argument(
        "--geometry-tol",
        type=float,
        default=None,
        help="Maximum original perimeter/plane mismatch in mesh units (default: 0.5%% of interface diameter)",
    )
    parser.add_argument(
        "--merge-tol",
        type=float,
        default=1e-8,
        help="Coincident-node merge tolerance in mesh units (default: 1e-8)",
    )
    parser.add_argument(
        "--seam-tol",
        type=float,
        default=None,
        help=("Maximum targeted curved-seam snap in mesh units (default: 0.005%% of interface diameter)"),
    )
    parser.add_argument(
        "--symmetry",
        choices=("off", "x", "xy"),
        default="off",
        help="Reduced-domain symmetry mode (default: off)",
    )
    parser.add_argument("--binary", action="store_true", help="Write binary Gmsh 2.2 output")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    return parser


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> None:
    parser = _build_arg_parser(prog)
    args = parser.parse_args(argv)
    output_path = Path(args.output_mesh)
    if output_path.exists() and not args.force:
        parser.error(f"Output already exists: {output_path}. Use --force to overwrite it.")
    try:
        result = conform_interface_mesh_files(
            args.fem_mesh,
            args.bem_mesh,
            output_path,
            fem_interface_name=args.fem_interface,
            bem_interface_name=args.bem_interface,
            geometry_tolerance=args.geometry_tol,
            seam_tolerance=args.seam_tol,
            merge_tolerance=args.merge_tol,
            symmetry_mode=args.symmetry,
            binary=args.binary,
        )
    except (InterfaceConformError, OSError) as exc:
        parser.error(str(exc))

    print(f"Wrote conforming BEM mesh: {output_path}")
    print(
        "Interface: "
        f"{result.original_bem_interface_triangles} BEM triangles replaced by "
        f"{result.fem_interface_triangles} FEM facets"
    )
    if result.seam_simplification_used:
        print(f"Surrounding BEM seam simplified: {result.simplified_bem_seam_edges} redundant boundary edges collapsed")
        print("Warning: inspect the conformed interface and surrounding triangles before solving.")
    elif result.original_adjacent_triangles == result.remeshed_adjacent_triangles == 0:
        print("Surrounding BEM surface preserved: interface seams already matched discretely")
    else:
        print(
            "Surrounding surface: "
            f"{result.original_adjacent_triangles} triangles replaced by "
            f"{result.remeshed_adjacent_triangles} triangles"
        )
    print(
        f"Output: {result.output_vertices} vertices, {result.output_triangles} triangles, "
        f"{result.identity.bem_boundary_edges} open boundary edges"
    )
    print(f"Maximum original perimeter deviation: {result.max_original_boundary_deviation:g}")


if __name__ == "__main__":
    main()
