"""Fast storage and working-set estimates for coupled speaker packages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import meshio
import numpy as np

from blab.component_symmetry import ComponentSymmetryInferenceError, infer_component_symmetry
from blab.config import normalize_symmetry
from blab.physical_model import BoundaryKind, ComponentKind, MeshPurpose, PhysicalSystem
from blab.symmetry import snap_points_to_symmetry_planes, symmetry_plane_tolerance_m

_ACTIVE_AXES = {"off": (), "x": (0,), "xy": (0, 1)}
_IMAGE_SIGNS = {
    "off": ((1.0, 1.0, 1.0),),
    "x": ((1.0, 1.0, 1.0), (-1.0, 1.0, 1.0)),
    "xy": (
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
        (1.0, -1.0, 1.0),
        (-1.0, -1.0, 1.0),
    ),
}
_TRIANGLE_TYPES = {"triangle", "triangle3"}


@dataclass(frozen=True)
class SpeakerPackagePreflight:
    frequency_count: int
    source_symmetry: str
    symmetry_image_count: int
    fem_node_count: int
    retained_fem_node_count: int
    interface_flux_count: int
    transducer_count: int
    excitation_count: int
    state_count: int
    bem_node_count: int
    bem_face_count: int
    source_mesh_bytes: int
    estimated_full_domain_mesh_bytes: int
    level_one_numeric_bytes: int
    level_two_numeric_bytes: int
    rom_rank: int
    parity_rom_numeric_bytes: int
    parity_rom_package_bytes_estimate: int
    rom_training_schur_bytes: int

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["sizes"] = {
            _size_key(key): _size_record(value)
            for key, value in values.items()
            if key.endswith("_bytes") or key.endswith("_bytes_estimate")
        }
        return values


def estimate_level_three_package(
    system: PhysicalSystem,
    *,
    symmetry: object,
    frequency_count: int,
    complex_bytes: int = 8,
    rom_rank: int = 32,
    sphere_point_count: int = 6600,
) -> SpeakerPackagePreflight:
    """Estimate Level-3 storage without materializing full-domain meshes or matrices.

    Counts follow the same reflection and cut-plane welding rules as the Level-3
    symmetry expander. Storage estimates deliberately describe uncompressed
    numeric payloads; archive compression is content dependent.
    """

    mode = normalize_symmetry(symmetry)
    if frequency_count <= 0:
        raise ValueError("Speaker package preflight requires at least one frequency.")
    if complex_bytes not in {8, 16}:
        raise ValueError("Complex storage width must be 8 or 16 bytes.")
    if rom_rank <= 0:
        raise ValueError("ROM rank must be positive.")
    if sphere_point_count <= 0:
        raise ValueError("Speaker package preflight requires at least one sphere point.")

    boundaries_by_mesh: dict[str, list[Any]] = {}
    for boundary in system.boundaries:
        boundaries_by_mesh.setdefault(boundary.group.mesh_id, []).append(boundary)

    fem_nodes = retained_nodes = interface_faces = bem_nodes = bem_faces = 0
    source_mesh_bytes = 0
    for resource in system.meshes:
        path = Path(resource.file)
        source_mesh_bytes += path.stat().st_size
        mesh = meshio.read(path)
        points = np.asarray(mesh.points, dtype=np.float64) * float(resource.scale_to_m)
        points += np.asarray(resource.translation_m, dtype=np.float64)
        all_triangles, triangle_tags = _triangles_and_tags(mesh)
        if resource.purpose == MeshPurpose.BEM_SURFACE:
            bem_nodes += _expanded_node_count(points, range(len(points)), mode)
            bem_faces += _expanded_face_count(points, all_triangles, mode, remove_cut_faces=False)
            continue
        if resource.purpose != MeshPurpose.FEM_VOLUME:
            continue
        fem_nodes += _expanded_node_count(points, range(len(points)), mode)
        active_tags = {
            _physical_group_tag(mesh, boundary.group.name, boundary.group.tag, boundary.group.dimension)
            for boundary in boundaries_by_mesh.get(resource.id, ())
            if boundary.kind in {BoundaryKind.INTERFACE, BoundaryKind.MOVING}
        }
        interface_tags = {
            _physical_group_tag(mesh, boundary.group.name, boundary.group.tag, boundary.group.dimension)
            for boundary in boundaries_by_mesh.get(resource.id, ())
            if boundary.kind == BoundaryKind.INTERFACE
        }
        active_faces = all_triangles[np.isin(triangle_tags, tuple(active_tags))]
        flux_faces = all_triangles[np.isin(triangle_tags, tuple(interface_tags))]
        active_source_nodes = np.unique(active_faces.reshape(-1)) if active_faces.size else np.empty(0, dtype=int)
        retained_nodes += _expanded_node_count(points, active_source_nodes, mode)
        interface_faces += _expanded_face_count(points, flux_faces, mode, remove_cut_faces=True)

    transducer_count_by_component = _expanded_component_counts(system, mode)
    transducer_count = sum(
        transducer_count_by_component.get(component.id, 1)
        for component in system.components
        if component.kind == ComponentKind.ELECTRODYNAMIC_TRANSDUCER
    )
    excitation_count = sum(transducer_count_by_component.get(port.component_id, 1) for port in system.excitation_ports)
    state_count = retained_nodes + interface_faces + 2 * transducer_count

    image_count = len(_IMAGE_SIGNS[mode])
    full_mesh_bytes = source_mesh_bytes * image_count

    # Levels 1 and 2 remain in every Level-3 package. Include their dominant
    # complex arrays so the package estimates are comparable rather than only
    # reporting the alternative interior payloads.
    level_one_numeric = frequency_count * excitation_count * sphere_point_count * complex_bytes
    level_two_numeric = frequency_count * excitation_count * (bem_nodes + bem_faces) * complex_bytes
    common_numeric = level_one_numeric + level_two_numeric

    rank = min(rom_rank, max(state_count, 1))
    node_orbits = max((bem_nodes + image_count - 1) // image_count, 1)
    face_orbits = max((bem_faces + image_count - 1) // image_count, 1)
    entries_per_sector = (
        rank * rank
        + rank * node_orbits
        + face_orbits * rank
        + rank * excitation_count
        + face_orbits * excitation_count
        + 2 * transducer_count * rank
        + 2 * transducer_count * excitation_count
    )
    parity_rom_numeric = frequency_count * image_count * entries_per_sector * complex_bytes
    training_schur = retained_nodes * retained_nodes * complex_bytes
    return SpeakerPackagePreflight(
        frequency_count=frequency_count,
        source_symmetry=mode,
        symmetry_image_count=image_count,
        fem_node_count=fem_nodes,
        retained_fem_node_count=retained_nodes,
        interface_flux_count=interface_faces,
        transducer_count=transducer_count,
        excitation_count=excitation_count,
        state_count=state_count,
        bem_node_count=bem_nodes,
        bem_face_count=bem_faces,
        source_mesh_bytes=source_mesh_bytes,
        estimated_full_domain_mesh_bytes=full_mesh_bytes,
        level_one_numeric_bytes=level_one_numeric,
        level_two_numeric_bytes=level_two_numeric,
        rom_rank=rank,
        parity_rom_numeric_bytes=parity_rom_numeric,
        parity_rom_package_bytes_estimate=parity_rom_numeric + common_numeric,
        rom_training_schur_bytes=training_schur,
    )


def _triangles_and_tags(mesh: meshio.Mesh) -> tuple[np.ndarray, np.ndarray]:
    triangles: list[np.ndarray] = []
    tags: list[np.ndarray] = []
    physical = mesh.cell_data.get("gmsh:physical", ())
    for index, block in enumerate(mesh.cells):
        if block.type not in _TRIANGLE_TYPES:
            continue
        values = np.asarray(block.data, dtype=np.int64)[:, :3]
        triangles.append(values)
        if index < len(physical):
            tags.append(np.asarray(physical[index], dtype=np.int64))
        else:
            tags.append(np.zeros(len(values), dtype=np.int64))
    if not triangles:
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(triangles), np.concatenate(tags)


def _expanded_node_count(points: np.ndarray, source_indices: Iterable[int], mode: str) -> int:
    snapped = snap_points_to_symmetry_planes(points, mode)
    tolerance = symmetry_plane_tolerance_m(snapped)
    scale = max(tolerance, np.finfo(float).eps)
    active_axes = _ACTIVE_AXES[mode]
    keys: set[tuple[object, ...]] = set()
    for image_index, signs in enumerate(_IMAGE_SIGNS[mode]):
        transformed = snapped * np.asarray(signs)
        for raw_index in source_indices:
            source_index = int(raw_index)
            point = transformed[source_index]
            quantized = tuple(int(round(float(value) / scale)) for value in point)
            on_plane = any(abs(float(point[axis])) <= tolerance for axis in active_axes)
            key = ("plane", *quantized) if on_plane else ("image", image_index, source_index)
            keys.add(key)
    return len(keys)


def _expanded_face_count(
    points: np.ndarray,
    faces: np.ndarray,
    mode: str,
    *,
    remove_cut_faces: bool,
) -> int:
    if not faces.size:
        return 0
    snapped = snap_points_to_symmetry_planes(points, mode)
    tolerance = symmetry_plane_tolerance_m(snapped)
    active_axes = _ACTIVE_AXES[mode]
    count = 0
    image_count = len(_IMAGE_SIGNS[mode])
    for face in faces:
        on_cut = any(np.max(np.abs(snapped[face, axis])) <= tolerance for axis in active_axes)
        if remove_cut_faces and on_cut:
            continue
        count += 1 if on_cut else image_count
    return count


def _physical_group_tag(mesh: meshio.Mesh, name: str | None, tag: int | None, dimension: int) -> int:
    if name is not None and name in mesh.field_data:
        candidate_tag, candidate_dimension = map(int, np.asarray(mesh.field_data[name]).tolist())
        if candidate_dimension == dimension:
            return candidate_tag
    if tag is not None:
        return int(tag)
    raise ValueError(f"Physical group {name!r} has no resolvable tag.")


def _expanded_component_counts(system: PhysicalSystem, mode: str) -> dict[str, int]:
    resources = {resource.id: resource for resource in system.meshes}
    boundaries = {boundary.id: boundary for boundary in system.boundaries}
    active_axis_names = tuple("xy"[axis] for axis in _ACTIVE_AXES[mode])
    result: dict[str, int] = {}
    for component in system.components:
        component_boundaries = tuple(boundaries[value] for value in component.boundary_ids if value in boundaries)
        try:
            fractional_axes = infer_component_symmetry(
                component_boundaries,
                resources,
                mode,
            ).fractional_symmetry_axes
        except ComponentSymmetryInferenceError:
            raw_axes = component.parameters.get("fractional_symmetry_axes", ())
            fractional_axes = tuple(str(value).lower() for value in raw_axes) if isinstance(raw_axes, list) else ()
        orbit_axis_count = sum(axis not in fractional_axes for axis in active_axis_names)
        result[component.id] = 2**orbit_axis_count
    return result


def _size_record(value: int) -> dict[str, float | int]:
    return {
        "bytes": int(value),
        "mb": round(value / 1_000_000.0, 3),
        "mib": round(value / 2**20, 3),
        "gb": round(value / 1_000_000_000.0, 3),
        "gib": round(value / 2**30, 3),
    }


def _size_key(value: str) -> str:
    if value.endswith("_bytes_estimate"):
        return value.removesuffix("_bytes_estimate") + "_estimate"
    return value.removesuffix("_bytes")


__all__ = ["SpeakerPackagePreflight", "estimate_level_three_package"]
