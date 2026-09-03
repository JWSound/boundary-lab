"""Infer electrodynamic component multiplicity from reduced-mesh topology."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import meshio
import numpy as np

from blab.config import normalize_symmetry
from blab.physical_model import Boundary, MeshResource

_ACTIVE_AXIS_NAMES = {
    "off": (),
    "x": ("x",),
    "xy": ("x", "y"),
}
_AXIS_INDEX = {"x": 0, "y": 1}
SYMMETRY_PARAMETER_KEYS = frozenset(
    {
        "symmetry_role",
        "surface_completion_factor",
        "physical_driver_orbit_count",
        "fractional_symmetry_axes",
    }
)


class ComponentSymmetryInferenceError(ValueError):
    """Raised when selected moving surfaces do not have unambiguous symmetry topology."""


@dataclass(frozen=True)
class ProjectedDiaphragmAreaInference:
    """Completed projected area of one physical diaphragm and its acoustic sides."""

    projected_area_m2: float
    positive_side_area_m2: float
    negative_side_area_m2: float

    @property
    def has_opposing_sides(self) -> bool:
        return self.positive_side_area_m2 > 0.0 and self.negative_side_area_m2 > 0.0

    @property
    def relative_side_mismatch(self) -> float | None:
        if not self.has_opposing_sides:
            return None
        maximum = max(self.positive_side_area_m2, self.negative_side_area_m2)
        return abs(self.positive_side_area_m2 - self.negative_side_area_m2) / maximum


@dataclass
class ProjectedAreaGeometryCache:
    """Reusable mesh geometry needed by projected diaphragm-area inference."""

    opposite_vertex_by_resource: dict[tuple, dict[tuple[int, int, int], int]] = field(default_factory=dict)
    surface_geometry_by_resource_tag: dict[tuple, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)


@dataclass(frozen=True)
class ComponentSymmetryInference:
    symmetry_mode: str
    fractional_symmetry_axes: tuple[str, ...]
    surface_completion_factor: int
    physical_driver_orbit_count: int
    surface_patch_count: int
    perimeter_edge_count: int
    plane_edge_counts: tuple[tuple[str, int], ...]

    @property
    def symmetry_role(self) -> str:
        return "fractional_driver" if self.fractional_symmetry_axes else "complete_representative"

    def parameters(self) -> dict[str, object]:
        return {
            "symmetry_role": self.symmetry_role,
            "surface_completion_factor": self.surface_completion_factor,
            "physical_driver_orbit_count": self.physical_driver_orbit_count,
            "fractional_symmetry_axes": list(self.fractional_symmetry_axes),
        }

    def summary(self) -> str:
        component_label = "component" if self.physical_driver_orbit_count == 1 else "components"
        if self.symmetry_mode == "off":
            return (
                "Moving surface(s) not sliced along a symmetry axis. Detected 1 distinct component in the full system."
            )
        if self.fractional_symmetry_axes:
            axes = " and ".join(self.fractional_symmetry_axes)
            axis_label = "axis" if len(self.fractional_symmetry_axes) == 1 else "axes"
            slicing = f"sliced along the {axes} {axis_label}"
        else:
            active_axes = _ACTIVE_AXIS_NAMES[self.symmetry_mode]
            axes = " and ".join(active_axes)
            axis_label = "axis" if len(active_axes) == 1 else "axes"
            slicing = f"not sliced along the {axes} {axis_label}"
        return (
            f"Moving surface(s) {slicing}. Detected {self.physical_driver_orbit_count} "
            f"distinct {component_label} in the fully mirrored system."
        )


def infer_component_symmetry(
    boundaries: tuple[Boundary, ...],
    resources_by_id: dict[str, MeshResource],
    symmetry_mode: str,
    *,
    mesh_cache: dict[str, meshio.Mesh] | None = None,
    tolerance_m: float | None = None,
) -> ComponentSymmetryInference:
    """Infer component completion and orbit counts from moving-surface perimeter edges."""

    mode = normalize_symmetry(symmetry_mode)
    active_axes = _ACTIVE_AXIS_NAMES[mode]
    symmetry_factor = 2 ** len(active_axes)
    if not active_axes:
        return ComponentSymmetryInference(
            symmetry_mode=mode,
            fractional_symmetry_axes=(),
            surface_completion_factor=1,
            physical_driver_orbit_count=1,
            surface_patch_count=0,
            perimeter_edge_count=0,
            plane_edge_counts=(),
        )
    if not boundaries:
        raise ComponentSymmetryInferenceError(
            "Select at least one moving boundary before inferring component symmetry."
        )

    cache = {} if mesh_cache is None else mesh_cache
    boundaries_by_resource: dict[str, list[Boundary]] = defaultdict(list)
    for boundary in boundaries:
        boundaries_by_resource[boundary.group.mesh_id].append(boundary)

    patch_axes: list[tuple[str, ...]] = []
    total_perimeter_edges = 0
    plane_edge_counts = {axis: 0 for axis in active_axes}
    for resource_id, resource_boundaries in boundaries_by_resource.items():
        resource = resources_by_id.get(resource_id)
        if resource is None:
            raise ComponentSymmetryInferenceError(f"Moving surfaces reference unavailable mesh '{resource_id}'.")
        mesh = cache.get(resource_id)
        if mesh is None:
            mesh = _transformed_mesh(resource)
            cache[resource_id] = mesh
        tags = {_boundary_surface_tag(mesh, boundary) for boundary in resource_boundaries}
        triangles = _triangles_for_tags(mesh, tags, resource.name)
        for patch in _triangle_patches(triangles, resource.name):
            perimeter = _perimeter_edges(patch)
            total_perimeter_edges += len(perimeter)
            points = np.asarray(mesh.points, dtype=float)
            patch_vertices = np.unique(patch.reshape(-1))
            patch_points = points[patch_vertices]
            diameter = float(np.linalg.norm(np.ptp(patch_points, axis=0)))
            tolerance = max(1e-9, diameter * 1e-7) if tolerance_m is None else float(tolerance_m)
            axes = []
            for axis in active_axes:
                axis_index = _AXIS_INDEX[axis]
                minimum = float(np.min(patch_points[:, axis_index]))
                if minimum < -tolerance:
                    raise ComponentSymmetryInferenceError(
                        f"Component surface on mesh '{resource.name}' extends into negative "
                        f"{axis.upper()}; use the reduced {mode.upper()} solve mesh for symmetry inference."
                    )
                if float(np.max(np.abs(patch_points[:, axis_index]))) <= tolerance:
                    raise ComponentSymmetryInferenceError(
                        f"Component surface on mesh '{resource.name}' lies in the {axis.upper()} symmetry "
                        "plane, so its physical multiplicity is ambiguous."
                    )
                if len(perimeter):
                    on_plane = np.max(np.abs(points[perimeter, axis_index]), axis=1) <= tolerance
                    count = int(np.count_nonzero(on_plane))
                else:
                    count = 0
                plane_edge_counts[axis] += count
                if count:
                    axes.append(axis)
            patch_axes.append(tuple(axes))

    if not patch_axes:
        raise ComponentSymmetryInferenceError(
            "The selected moving boundaries contain no non-degenerate surface triangles."
        )
    distinct_axes = set(patch_axes)
    if len(distinct_axes) != 1:
        details = ", ".join(
            "none" if not axes else "+".join(axis.upper() for axis in axes) for axes in sorted(distinct_axes)
        )
        raise ComponentSymmetryInferenceError(
            "Disconnected moving-surface patches infer inconsistent symmetry cuts "
            f"({details}). Assign only surfaces belonging to the same physical driver."
        )

    fractional_axes = patch_axes[0]
    completion_factor = 2 ** len(fractional_axes)
    return ComponentSymmetryInference(
        symmetry_mode=mode,
        fractional_symmetry_axes=fractional_axes,
        surface_completion_factor=completion_factor,
        physical_driver_orbit_count=symmetry_factor // completion_factor,
        surface_patch_count=len(patch_axes),
        perimeter_edge_count=total_perimeter_edges,
        plane_edge_counts=tuple((axis, plane_edge_counts[axis]) for axis in active_axes),
    )


def infer_projected_diaphragm_area(
    boundaries: tuple[Boundary, ...],
    resources_by_id: dict[str, MeshResource],
    motion_axis: tuple[float, float, float] | np.ndarray,
    surface_completion_factor: int,
    *,
    boundary_motion_weights: dict[str, float] | None = None,
    boundary_side_keys: dict[str, str] | None = None,
    mesh_cache: dict[str, meshio.Mesh] | None = None,
    projected_geometry_cache: ProjectedAreaGeometryCache | None = None,
) -> ProjectedDiaphragmAreaInference:
    """Integrate the solver's projected rigid-translation area by acoustic side."""

    if not boundaries:
        raise ComponentSymmetryInferenceError("Select at least one moving boundary before calculating projected area.")
    axis = np.asarray(motion_axis, dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if not np.isfinite(axis).all() or axis_norm <= 0.0:
        raise ComponentSymmetryInferenceError("The motion axis must contain a finite nonzero direction.")
    axis /= axis_norm
    completion = int(surface_completion_factor)
    if completion not in {1, 2, 4}:
        raise ComponentSymmetryInferenceError("Surface completion factor must be 1, 2, or 4.")
    weights = {} if boundary_motion_weights is None else boundary_motion_weights
    cache = {} if mesh_cache is None else mesh_cache
    geometry_cache = ProjectedAreaGeometryCache() if projected_geometry_cache is None else projected_geometry_cache
    positive_area = 0.0
    negative_area = 0.0
    area_by_side: dict[str, float] = {}
    for boundary in boundaries:
        resource = resources_by_id.get(boundary.group.mesh_id)
        if resource is None:
            raise ComponentSymmetryInferenceError(
                f"Moving surface references unavailable mesh '{boundary.group.mesh_id}'."
            )
        mesh = cache.get(resource.id)
        if mesh is None:
            mesh = _transformed_mesh(resource)
            cache[resource.id] = mesh
        tag = _boundary_surface_tag(mesh, boundary)
        resource_key = _resource_geometry_key(resource)
        surface_key = (*resource_key, tag)
        surface_geometry = geometry_cache.surface_geometry_by_resource_tag.get(surface_key)
        if surface_geometry is None:
            triangles = _triangles_for_tags(mesh, {tag}, resource.name)
            points = np.asarray(mesh.points, dtype=float)
            vertices = points[triangles]
            cross = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
            double_area = np.linalg.norm(cross, axis=1)
            nondegenerate = double_area > np.finfo(float).eps
            if not np.any(nondegenerate):
                continue
            triangles = triangles[nondegenerate]
            vertices = vertices[nondegenerate]
            cross = cross[nondegenerate]
            double_area = double_area[nondegenerate]
            normals = cross / double_area[:, np.newaxis]
            opposite_vertices = geometry_cache.opposite_vertex_by_resource.get(resource_key)
            if opposite_vertices is None:
                opposite_vertices = _tetrahedron_opposite_vertex_by_face(mesh)
                geometry_cache.opposite_vertex_by_resource[resource_key] = opposite_vertices
            normals = _orient_surface_normals_outward(
                mesh,
                triangles,
                vertices,
                normals,
                opposite_vertices,
            )
            surface_geometry = (normals, double_area)
            geometry_cache.surface_geometry_by_resource_tag[surface_key] = surface_geometry
        normals, double_area = surface_geometry
        try:
            coefficient = float(weights.get(boundary.id, 1.0))
        except (TypeError, ValueError) as exc:
            raise ComponentSymmetryInferenceError(
                f"Moving boundary '{boundary.name}' has an invalid motion coefficient."
            ) from exc
        if not np.isfinite(coefficient) or coefficient <= 0.0:
            raise ComponentSymmetryInferenceError(
                f"Moving boundary '{boundary.name}' has a non-finite or non-positive motion weight."
            )
        projected = completion * coefficient * (normals @ axis) * (0.5 * double_area)
        positive_area += float(np.sum(projected[projected > 0.0]))
        negative_area += float(np.sum(-projected[projected < 0.0]))

        if boundary_side_keys is not None:
            side_key = boundary_side_keys.get(boundary.id)
            if side_key is None:
                raise ComponentSymmetryInferenceError(
                    f"Moving boundary '{boundary.name}' has no acoustic-side assignment."
                )
            area_by_side[side_key] = area_by_side.get(side_key, 0.0) + float(np.sum(np.abs(projected)))

    region_side_areas = [area for area in area_by_side.values() if area > np.finfo(float).eps]
    if len(region_side_areas) > 2:
        raise ComponentSymmetryInferenceError(
            "An electrodynamic transducer may drive surfaces in no more than two acoustic regions."
        )
    if len(region_side_areas) == 2:
        positive_area, negative_area = region_side_areas

    scale = max(positive_area, negative_area)
    tolerance = max(np.finfo(float).eps, scale * 1.0e-9)
    positive_area = positive_area if positive_area > tolerance else 0.0
    negative_area = negative_area if negative_area > tolerance else 0.0
    side_areas = [value for value in (positive_area, negative_area) if value > 0.0]
    if not side_areas:
        raise ComponentSymmetryInferenceError(
            "Selected moving surfaces have zero projected area along the motion axis."
        )
    return ProjectedDiaphragmAreaInference(
        projected_area_m2=float(np.mean(side_areas)),
        positive_side_area_m2=positive_area,
        negative_side_area_m2=negative_area,
    )


def infer_weighted_surface_area(
    boundaries: tuple[Boundary, ...],
    resources_by_id: dict[str, MeshResource],
    surface_completion_factor: int,
    *,
    boundary_motion_weights: dict[str, float] | None = None,
    mesh_cache: dict[str, meshio.Mesh] | None = None,
) -> float:
    """Integrate symmetry-completed physical area for normal-velocity sources."""

    if not boundaries:
        raise ComponentSymmetryInferenceError(
            "Select at least one moving boundary before calculating driven surface area."
        )
    completion = int(surface_completion_factor)
    if completion not in {1, 2, 4}:
        raise ComponentSymmetryInferenceError("Surface completion factor must be 1, 2, or 4.")
    weights = {} if boundary_motion_weights is None else boundary_motion_weights
    cache = {} if mesh_cache is None else mesh_cache
    total_area = 0.0
    for boundary in boundaries:
        resource = resources_by_id.get(boundary.group.mesh_id)
        if resource is None:
            raise ComponentSymmetryInferenceError(
                f"Moving surface references unavailable mesh '{boundary.group.mesh_id}'."
            )
        mesh = cache.get(resource.id)
        if mesh is None:
            mesh = _transformed_mesh(resource)
            cache[resource.id] = mesh
        tag = _boundary_surface_tag(mesh, boundary)
        triangles = _triangles_for_tags(mesh, {tag}, resource.name)
        vertices = np.asarray(mesh.points, dtype=float)[triangles]
        double_area = np.linalg.norm(
            np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]),
            axis=1,
        )
        try:
            coefficient = float(weights.get(boundary.id, 1.0))
        except (TypeError, ValueError) as exc:
            raise ComponentSymmetryInferenceError(
                f"Moving boundary '{boundary.name}' has an invalid motion coefficient."
            ) from exc
        if not np.isfinite(coefficient) or coefficient <= 0.0:
            raise ComponentSymmetryInferenceError(
                f"Moving boundary '{boundary.name}' has a non-finite or non-positive motion weight."
            )
        total_area += completion * coefficient * float(np.sum(0.5 * double_area))
    if not np.isfinite(total_area) or total_area <= np.finfo(float).eps:
        raise ComponentSymmetryInferenceError("Selected moving surfaces have zero physical area.")
    return total_area


def _resource_geometry_key(resource: MeshResource) -> tuple:
    return (
        resource.id,
        resource.file,
        float(resource.scale_to_m),
        tuple(float(value) for value in resource.translation_m),
    )


def _tetrahedron_opposite_vertex_by_face(
    mesh: meshio.Mesh,
) -> dict[tuple[int, int, int], int]:
    opposite_vertex_by_face: dict[tuple[int, int, int], int] = {}
    for block in mesh.cells:
        if block.type not in {"tetra", "tetra4", "tetra10"}:
            continue
        for raw_tetrahedron in np.asarray(block.data, dtype=np.int64):
            a, b, c, d = map(int, raw_tetrahedron[:4])
            for face, opposite in (
                ((a, b, c), d),
                ((a, b, d), c),
                ((a, c, d), b),
                ((b, c, d), a),
            ):
                opposite_vertex_by_face.setdefault(tuple(sorted(face)), opposite)
    return opposite_vertex_by_face


def _orient_surface_normals_outward(
    mesh: meshio.Mesh,
    triangles: np.ndarray,
    vertices: np.ndarray,
    normals: np.ndarray,
    opposite_vertex_by_face: dict[tuple[int, int, int], int],
) -> np.ndarray:
    if not opposite_vertex_by_face:
        return normals
    oriented = normals.copy()
    points = np.asarray(mesh.points, dtype=float)
    for index, triangle in enumerate(triangles):
        opposite = opposite_vertex_by_face.get(tuple(sorted(map(int, triangle))))
        if opposite is None:
            continue
        face_center = np.mean(vertices[index], axis=0)
        if float(np.dot(oriented[index], points[opposite] - face_center)) > 0.0:
            oriented[index] *= -1.0
    return oriented


def _transformed_mesh(resource: MeshResource) -> meshio.Mesh:
    try:
        mesh = meshio.read(Path(resource.file))
    except Exception as exc:
        raise ComponentSymmetryInferenceError(
            f"Could not read component mesh '{resource.name}' from {resource.file}: {exc}"
        ) from exc
    points = np.asarray(mesh.points, dtype=float) * float(resource.scale_to_m)
    points += np.asarray(resource.translation_m, dtype=float)
    return meshio.Mesh(
        points=points,
        cells=mesh.cells,
        point_data=mesh.point_data,
        cell_data=mesh.cell_data,
        field_data=mesh.field_data,
        cell_sets=mesh.cell_sets,
    )


def _boundary_surface_tag(mesh: meshio.Mesh, boundary: Boundary) -> int:
    if boundary.group.name is not None:
        field = mesh.field_data.get(boundary.group.name)
        if field is None:
            raise ComponentSymmetryInferenceError(
                f"Mesh '{boundary.group.mesh_id}' does not contain surface group '{boundary.group.name}'."
            )
        tag, dimension = map(int, np.asarray(field).tolist())
        if dimension != 2:
            raise ComponentSymmetryInferenceError(f"Physical group '{boundary.group.name}' is not a surface group.")
        return tag
    if boundary.group.tag is not None:
        return int(boundary.group.tag)
    raise ComponentSymmetryInferenceError(
        f"Moving boundary '{boundary.name}' must identify a surface group by name or tag."
    )


def _triangles_for_tags(mesh: meshio.Mesh, tags: set[int], mesh_name: str) -> np.ndarray:
    physical_blocks = mesh.cell_data.get("gmsh:physical")
    if physical_blocks is None:
        raise ComponentSymmetryInferenceError(f"Mesh '{mesh_name}' has no physical surface tags.")
    selected = []
    for index, block in enumerate(mesh.cells):
        if block.type not in {"triangle", "triangle3"}:
            continue
        triangles = np.asarray(block.data, dtype=np.int64)[:, :3]
        physical = np.asarray(physical_blocks[index], dtype=np.int64)
        if len(physical) != len(triangles):
            raise ComponentSymmetryInferenceError(f"Mesh '{mesh_name}' has inconsistent triangle physical tags.")
        selected.append(triangles[np.isin(physical, tuple(tags))])
    triangles = [block for block in selected if len(block)]
    if not triangles:
        raise ComponentSymmetryInferenceError(f"Selected moving surfaces on mesh '{mesh_name}' contain no triangles.")
    return np.vstack(triangles)


def _triangle_patches(triangles: np.ndarray, mesh_name: str) -> tuple[np.ndarray, ...]:
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index, triangle in enumerate(triangles):
        for start, end in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge_faces[tuple(sorted((int(start), int(end))))].append(face_index)
    if any(len(faces) > 2 for faces in edge_faces.values()):
        raise ComponentSymmetryInferenceError(
            f"Selected moving surfaces on mesh '{mesh_name}' contain non-manifold edges."
        )
    adjacency = [set() for _ in range(len(triangles))]
    for faces in edge_faces.values():
        if len(faces) == 2:
            first, second = faces
            adjacency[first].add(second)
            adjacency[second].add(first)
    unseen = set(range(len(triangles)))
    patches = []
    while unseen:
        start = unseen.pop()
        stack = [start]
        faces = [start]
        while stack:
            face = stack.pop()
            for neighbor in adjacency[face]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    faces.append(neighbor)
                    stack.append(neighbor)
        patches.append(np.asarray(triangles[np.asarray(faces, dtype=np.int64)], dtype=np.int64))
    return tuple(patches)


def _perimeter_edges(triangles: np.ndarray) -> np.ndarray:
    edge_counts: dict[tuple[int, int], int] = defaultdict(int)
    for triangle in triangles:
        for start, end in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge_counts[tuple(sorted((int(start), int(end))))] += 1
    return np.asarray(
        [edge for edge, count in edge_counts.items() if count == 1],
        dtype=np.int64,
    ).reshape((-1, 2))


__all__ = [
    "ComponentSymmetryInference",
    "ComponentSymmetryInferenceError",
    "ProjectedAreaGeometryCache",
    "ProjectedDiaphragmAreaInference",
    "SYMMETRY_PARAMETER_KEYS",
    "infer_component_symmetry",
    "infer_projected_diaphragm_area",
]
