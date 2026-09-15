"""Qt-free physical-system mesh editing and motion-axis inference."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import meshio
import numpy as np

from blab.config import normalize_symmetry
from blab.exterior_preparation import prepare_exterior_system
from blab.interface_conform import (
    APPLICATION_INTERFACE_GEOMETRY_TOLERANCE_M,
    InterfaceConformError,
    build_conforming_interface_map,
    conform_bem_interface_to_fem,
)
from blab.mesh_inventory import AvailableSystemMesh
from blab.physical_model import (
    AcousticInterface,
    Boundary,
    MeshPurpose,
    MeshResource,
    PhysicalSystem,
)


@dataclass(frozen=True)
class InterfaceRebuildResult:
    system: PhysicalSystem
    mesh_file_overrides_by_name: dict[str, str] = field(default_factory=dict)
    rebuilt_interface_ids: tuple[str, ...] = ()
    quality_warning_interface_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class MotionAxisInference:
    """Dominant unoriented rigid-translation axis inferred from surface normals."""

    axis: tuple[float, float, float]
    confidence: float
    mean_squared_alignment: float
    boundary_alignment: float
    area_m2: float
    triangle_count: int


def infer_component_motion_axis(
    boundaries: tuple[Boundary, ...],
    resources_by_id: dict[str, MeshResource],
    *,
    fractional_symmetry_axes: tuple[str, ...] = (),
    mesh_cache: dict[str, meshio.Mesh] | None = None,
) -> MotionAxisInference:
    """Infer a common translation axis from the completed physical-driver surface."""

    if not boundaries:
        raise ValueError("Select at least one moving boundary before inferring its motion axis.")
    symmetry_axes = tuple(str(axis).strip().lower() for axis in fractional_symmetry_axes)
    if len(symmetry_axes) != len(set(symmetry_axes)) or any(axis not in {"x", "y"} for axis in symmetry_axes):
        raise ValueError("Fractional symmetry axes must be unique axis names chosen from X and Y.")
    cache = {} if mesh_cache is None else mesh_cache
    tensors = []
    combined = np.zeros((3, 3), dtype=float)
    total_area = 0.0
    triangle_count = 0
    for boundary in boundaries:
        resource = resources_by_id.get(boundary.group.mesh_id)
        if resource is None:
            raise ValueError(
                f"Moving boundary '{boundary.name}' references unavailable mesh '{boundary.group.mesh_id}'."
            )
        cache_key = resource.id
        mesh = cache.get(cache_key)
        if mesh is None:
            mesh = _transformed_mesh(resource)
            cache[cache_key] = mesh
        tensor, area, count = _boundary_normal_tensor(mesh, boundary)
        tensors.append(tensor)
        combined += tensor
        total_area += area
        triangle_count += count
    if total_area <= 0.0 or triangle_count == 0:
        raise ValueError("The selected moving boundaries contain no non-degenerate triangles.")

    combined = _symmetry_completed_normal_tensor(combined, symmetry_axes)
    tensors = [_symmetry_completed_normal_tensor(tensor, symmetry_axes) for tensor in tensors]
    axis_tensor = _normal_tensor_in_symmetry_planes(combined, symmetry_axes)
    eigenvalues, eigenvectors = np.linalg.eigh(axis_tensor)
    axis = np.asarray(eigenvectors[:, -1], dtype=float)
    largest_component = int(np.argmax(np.abs(axis)))
    if axis[largest_component] < 0.0:
        axis *= -1.0
    trace = float(np.trace(combined))
    confidence = float(max(0.0, min(1.0, (eigenvalues[-1] - eigenvalues[-2]) / trace)))
    mean_squared_alignment = float(max(0.0, min(1.0, eigenvalues[-1] / trace)))

    boundary_axes = []
    for tensor in tensors:
        _values, vectors = np.linalg.eigh(_normal_tensor_in_symmetry_planes(tensor, symmetry_axes))
        boundary_axes.append(np.asarray(vectors[:, -1], dtype=float))
    boundary_alignment = min(
        (abs(float(np.dot(axis, boundary_axis))) for boundary_axis in boundary_axes),
        default=1.0,
    )
    confidence = min(confidence, boundary_alignment)
    return MotionAxisInference(
        axis=tuple(float(value) for value in axis),
        confidence=confidence,
        mean_squared_alignment=mean_squared_alignment,
        boundary_alignment=boundary_alignment,
        area_m2=total_area,
        triangle_count=triangle_count,
    )


def _symmetry_completed_normal_tensor(
    tensor: np.ndarray,
    symmetry_axes: tuple[str, ...],
) -> np.ndarray:
    completed = np.asarray(tensor, dtype=float).copy()
    axis_indices = {"x": 0, "y": 1}
    for axis in symmetry_axes:
        reflection = np.eye(3, dtype=float)
        reflection[axis_indices[axis], axis_indices[axis]] = -1.0
        completed += reflection @ completed @ reflection
    return completed


def _normal_tensor_in_symmetry_planes(
    tensor: np.ndarray,
    symmetry_axes: tuple[str, ...],
) -> np.ndarray:
    projected = np.asarray(tensor, dtype=float).copy()
    axis_indices = {"x": 0, "y": 1}
    for axis in symmetry_axes:
        axis_index = axis_indices[axis]
        projected[axis_index, :] = 0.0
        projected[:, axis_index] = 0.0
    return projected


def _boundary_normal_tensor(mesh: meshio.Mesh, boundary: Boundary) -> tuple[np.ndarray, float, int]:
    if boundary.group.name is not None:
        field = mesh.field_data.get(boundary.group.name)
        if field is None:
            raise ValueError(f"Mesh '{boundary.group.mesh_id}' does not contain surface group '{boundary.group.name}'.")
        tag, dimension = map(int, np.asarray(field).tolist())
        if dimension != 2:
            raise ValueError(f"Physical group '{boundary.group.name}' is not a surface group.")
    elif boundary.group.tag is not None:
        tag = int(boundary.group.tag)
    else:
        raise ValueError(f"Moving boundary '{boundary.name}' must identify a surface group by name or tag.")

    physical_blocks = mesh.cell_data.get("gmsh:physical")
    if physical_blocks is None:
        raise ValueError(f"Mesh '{boundary.group.mesh_id}' has no physical surface tags.")
    points = np.asarray(mesh.points, dtype=float)
    tensor = np.zeros((3, 3), dtype=float)
    total_area = 0.0
    count = 0
    for index, block in enumerate(mesh.cells):
        if block.type not in {"triangle", "triangle3"}:
            continue
        triangles = np.asarray(block.data, dtype=np.int64)
        physical = np.asarray(physical_blocks[index], dtype=np.int64)
        if len(physical) != len(triangles):
            raise ValueError(f"Mesh '{boundary.group.mesh_id}' has inconsistent triangle physical tags.")
        for triangle in triangles[physical == tag]:
            vertices = points[np.asarray(triangle[:3], dtype=np.int64)]
            area_vector = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
            magnitude = float(np.linalg.norm(area_vector))
            if magnitude <= 0.0:
                continue
            normal = area_vector / magnitude
            area = 0.5 * magnitude
            tensor += area * np.outer(normal, normal)
            total_area += area
            count += 1
    if count == 0:
        raise ValueError(f"Moving boundary '{boundary.name}' contains no non-degenerate triangles.")
    return tensor, total_area, count


def sync_physical_system_meshes(
    system: PhysicalSystem,
    meshes: tuple[AvailableSystemMesh, ...],
) -> PhysicalSystem:
    """Apply current application file/scale/translation settings by mesh name."""

    available_by_name = {mesh.name: mesh for mesh in meshes}
    resources = []
    for resource in system.meshes:
        available = available_by_name.get(resource.name)
        if available is None:
            resources.append(resource)
            continue
        resources.append(
            replace(
                resource,
                file=available.file,
                scale_to_m=available.scale_to_m,
                translation_m=available.translation_m,
            )
        )
    return replace(system, meshes=tuple(resources))


def interface_bem_mesh_names_for_changes(
    system: PhysicalSystem | None,
    changed_mesh_names: set[str],
) -> tuple[str, ...]:
    """Return BEM resources whose configured interfaces depend on changed meshes."""

    if system is None or not system.interfaces or not changed_mesh_names:
        return ()
    resources = {resource.id: resource for resource in system.meshes}
    boundaries = {boundary.id: boundary for boundary in system.boundaries}
    affected = set()
    for interface in system.interfaces:
        bounded = boundaries.get(interface.bounded_boundary_id)
        unbounded = boundaries.get(interface.unbounded_boundary_id)
        if bounded is None or unbounded is None:
            continue
        fem_resource = resources.get(bounded.group.mesh_id)
        bem_resource = resources.get(unbounded.group.mesh_id)
        if fem_resource is None or bem_resource is None:
            continue
        if {fem_resource.name, bem_resource.name} & changed_mesh_names:
            if fem_resource.purpose != MeshPurpose.FEM_VOLUME or bem_resource.purpose != MeshPurpose.BEM_SURFACE:
                raise InterfaceConformError(
                    f"Configured interface '{interface.name}' does not connect a FEM volume to a BEM surface."
                )
            affected.add(bem_resource.name)
    return tuple(sorted(affected))


def rebuild_configured_interfaces(
    system: PhysicalSystem,
    meshes: tuple[AvailableSystemMesh, ...],
    *,
    changed_mesh_names: set[str],
    interface_output_root: str | Path,
    symmetry_mode: str = "off",
    stitch_exterior_meshes: bool = False,
    stitch_tolerance_mm: float = 2.0,
    symmetry_analysis_meshes: tuple[AvailableSystemMesh, ...] | None = None,
) -> InterfaceRebuildResult:
    """Validate and, when needed, rebuild known FEM-BEM interface pairs."""

    affected_bem_names = set(interface_bem_mesh_names_for_changes(system, changed_mesh_names))
    synced_system = sync_physical_system_meshes(system, meshes)
    if stitch_exterior_meshes:
        # Retain canonical authoring resources, but validate with the same
        # generated variants that preview and solve use for active symmetry.
        assembly_system = sync_physical_system_meshes(
            synced_system,
            meshes if symmetry_analysis_meshes is None else symmetry_analysis_meshes,
        )
        prepared = prepare_exterior_system(
            assembly_system,
            stitch_tolerance_mm=stitch_tolerance_mm,
            symmetry_mode=symmetry_mode,
            output_root=interface_output_root,
        )
        return InterfaceRebuildResult(
            system=synced_system,
            quality_warning_interface_ids=tuple(
                pair_id
                for entry in prepared.metadata["exterior_preparation"]
                for pair_id in entry["quality_warning_interface_ids"]
            ),
        )
    if not affected_bem_names:
        return InterfaceRebuildResult(system=synced_system)

    available_by_name = {mesh.name: mesh for mesh in meshes}
    resources_by_id = {resource.id: resource for resource in synced_system.meshes}
    boundaries_by_id = {boundary.id: boundary for boundary in synced_system.boundaries}
    interfaces_by_bem_name: dict[str, list[tuple[AcousticInterface, Boundary, Boundary]]] = {}
    for interface in synced_system.interfaces:
        fem_boundary = boundaries_by_id.get(interface.bounded_boundary_id)
        bem_boundary = boundaries_by_id.get(interface.unbounded_boundary_id)
        if fem_boundary is None or bem_boundary is None:
            raise InterfaceConformError(f"Configured interface '{interface.name}' references a missing boundary.")
        bem_resource = resources_by_id.get(bem_boundary.group.mesh_id)
        if bem_resource is None:
            raise InterfaceConformError(f"Configured interface '{interface.name}' references a missing BEM mesh.")
        interfaces_by_bem_name.setdefault(bem_resource.name, []).append((interface, fem_boundary, bem_boundary))

    mesh_cache: dict[tuple[str, float, tuple[float, float, float]], meshio.Mesh] = {}

    def transformed(resource: MeshResource) -> meshio.Mesh:
        key = (
            str(Path(resource.file).resolve()),
            float(resource.scale_to_m),
            tuple(float(value) for value in resource.translation_m),
        )
        if key not in mesh_cache:
            mesh_cache[key] = _transformed_mesh(resource)
        return mesh_cache[key]

    overrides: dict[str, str] = {}
    rebuilt_interface_ids: list[str] = []
    quality_warning_interface_ids: list[str] = []
    normalized_symmetry = normalize_symmetry(symmetry_mode)
    output_root = Path(interface_output_root)
    for bem_name in sorted(affected_bem_names):
        pairs = interfaces_by_bem_name.get(bem_name, [])
        if not pairs:
            continue
        _first_interface, _first_fem_boundary, first_bem_boundary = pairs[0]
        bem_resource = resources_by_id[first_bem_boundary.group.mesh_id]
        available = available_by_name.get(bem_resource.name)
        if available is None:
            raise InterfaceConformError(f"BEM mesh '{bem_resource.name}' is not available for interface rebuilding.")
        if available.locked:
            raise InterfaceConformError(
                f"BEM mesh '{bem_resource.name}' is generated/locked. Interface rebuilding currently "
                "requires an imported BEM mesh."
            )
        if available.has_tetrahedra:
            raise InterfaceConformError(f"Interface rebuild target '{bem_resource.name}' contains FEM volume elements.")

        bem_mesh = transformed(bem_resource)
        rebuilt = False
        final_fem_resource = None
        final_fem_name = ""
        final_bem_name = ""
        protected_names = tuple(
            str(pair_bem.group.name)
            for _pair_interface, _pair_fem, pair_bem in pairs
            if pair_bem.group.name is not None
        )
        for interface, fem_boundary, bem_boundary in pairs:
            fem_resource = resources_by_id.get(fem_boundary.group.mesh_id)
            if fem_resource is None:
                raise InterfaceConformError(f"Configured interface '{interface.name}' references a missing FEM mesh.")
            fem_available = available_by_name.get(fem_resource.name)
            if fem_available is None or not fem_available.has_tetrahedra:
                raise InterfaceConformError(
                    f"Interface FEM mesh '{fem_resource.name}' is missing or contains no volume elements."
                )
            fem_name = str(fem_boundary.group.name)
            interface_bem_name = str(bem_boundary.group.name)
            fem_mesh = transformed(fem_resource)
            try:
                build_conforming_interface_map(
                    fem_mesh,
                    bem_mesh,
                    fem_interface_name=fem_name,
                    bem_interface_name=interface_bem_name,
                    coordinate_tolerance=float(interface.coordinate_tolerance_m),
                    require_closed_bem=True,
                    symmetry_mode=normalized_symmetry,
                )
            except InterfaceConformError:
                bem_mesh, _result = conform_bem_interface_to_fem(
                    fem_mesh,
                    bem_mesh,
                    fem_interface_name=fem_name,
                    geometry_tolerance=APPLICATION_INTERFACE_GEOMETRY_TOLERANCE_M,
                    bem_interface_name=interface_bem_name,
                    merge_tolerance=1e-8,
                    symmetry_mode=normalized_symmetry,
                    protected_bem_interface_names=tuple(name for name in protected_names if name != interface_bem_name),
                )
                rebuilt = True
                rebuilt_interface_ids.append(interface.id)
                if _result.seam_simplification_used:
                    quality_warning_interface_ids.append(interface.id)
            final_fem_resource = fem_resource
            final_fem_name = fem_name
            final_bem_name = interface_bem_name

        if not rebuilt or final_fem_resource is None:
            continue
        for interface, fem_boundary, bem_boundary in pairs:
            fem_resource = resources_by_id[fem_boundary.group.mesh_id]
            build_conforming_interface_map(
                transformed(fem_resource),
                bem_mesh,
                fem_interface_name=str(fem_boundary.group.name),
                bem_interface_name=str(bem_boundary.group.name),
                coordinate_tolerance=float(interface.coordinate_tolerance_m),
                require_closed_bem=True,
                symmetry_mode=normalized_symmetry,
            )
        output_path = _conformed_mesh_path(
            available,
            fem_resource=final_fem_resource,
            fem_interface_name=final_fem_name,
            bem_interface_name=final_bem_name,
            interface_output_root=output_root,
            symmetry_mode=normalized_symmetry,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        meshio.write(
            output_path,
            _mesh_in_resource_coordinates(bem_mesh, bem_resource),
            file_format="gmsh22",
            binary=False,
        )
        resources_by_id[bem_resource.id] = replace(bem_resource, file=str(output_path))
        overrides[bem_resource.name] = str(output_path)

    rebuilt_system = replace(
        synced_system,
        meshes=tuple(resources_by_id[resource.id] for resource in synced_system.meshes),
    )
    return InterfaceRebuildResult(
        system=rebuilt_system,
        mesh_file_overrides_by_name=overrides,
        rebuilt_interface_ids=tuple(rebuilt_interface_ids),
        quality_warning_interface_ids=tuple(quality_warning_interface_ids),
    )


def _transformed_mesh(resource: MeshResource) -> meshio.Mesh:
    mesh = meshio.read(Path(resource.file))
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


def _mesh_in_resource_coordinates(mesh: meshio.Mesh, resource: MeshResource) -> meshio.Mesh:
    scale = float(resource.scale_to_m)
    if scale <= 0.0:
        raise ValueError(f"Mesh '{resource.name}' scale must be greater than zero.")
    points = np.asarray(mesh.points, dtype=float) - np.asarray(resource.translation_m, dtype=float)
    points /= scale
    return meshio.Mesh(
        points=points,
        cells=mesh.cells,
        point_data=mesh.point_data,
        cell_data=mesh.cell_data,
        field_data=mesh.field_data,
        cell_sets=mesh.cell_sets,
    )


def _conformed_mesh_path(
    mesh: AvailableSystemMesh,
    *,
    fem_resource: MeshResource,
    fem_interface_name: str,
    bem_interface_name: str,
    interface_output_root: Path,
    symmetry_mode: str,
) -> Path:
    identity = "|".join(
        (
            str(Path(mesh.source_file).resolve()),
            repr(mesh.scale_to_m),
            repr(mesh.translation_m),
            str(Path(fem_resource.file).resolve()),
            repr(fem_resource.scale_to_m),
            repr(fem_resource.translation_m),
            fem_interface_name,
            bem_interface_name,
            normalize_symmetry(symmetry_mode),
        )
    )
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:10]
    return interface_output_root / f"{_slug(mesh.name)}_{digest}_interface_conformed.msh"


def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value)).strip("-").lower()
    return text or "item"
