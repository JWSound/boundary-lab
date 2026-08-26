"""Temporary full-domain physical systems for dynamic speaker export.

Level-three speaker packages must respond to arbitrary incident fields.  A
symmetry-reduced solve only contains the even symmetry sector, so package
generation expands the authored solve meshes into an independent temporary
system and recompiles it with symmetry disabled.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import meshio
import numpy as np

from blab.component_symmetry import ComponentSymmetryInferenceError, infer_component_symmetry
from blab.config import normalize_symmetry
from blab.fem_topology import selected_volume_surface_tags
from blab.physical_model import (
    AcousticRegionKind,
    Boundary,
    MeshPurpose,
    PhysicalComponent,
    PhysicalGroupRef,
    PhysicalSystem,
)
from blab.symmetry import snap_points_to_symmetry_planes, symmetry_plane_tolerance_m

_AXIS_INDEX = {"x": 0, "y": 1}
_ACTIVE_AXES = {"off": (), "x": ("x",), "xy": ("x", "y")}
_TRIANGLE_TYPES = {"triangle", "triangle3"}
_TETRA_TYPES = {"tetra", "tetra4"}


@dataclass(frozen=True)
class FullDomainSpeakerSystem:
    """Export-only full-domain authoring system and its provenance maps."""

    system: PhysicalSystem
    component_source_ids: dict[str, str]
    excitation_port_source_ids: dict[str, str]
    preferred_full_mesh_names: tuple[str, ...]


@dataclass
class _ExpandedMesh:
    mesh: meshio.Mesh
    image_index_by_block: list[np.ndarray]
    image_bits: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _ComponentOrbit:
    component: PhysicalComponent
    fractional_axes: tuple[str, ...]
    group_by_image: tuple[int, ...]
    representative_image_by_group: tuple[int, ...]


def expand_speaker_system_for_export(
    system: PhysicalSystem,
    *,
    symmetry: object,
    output_dir: str | Path,
    preferred_full_mesh_by_name: Mapping[str, str | Path] | None = None,
) -> FullDomainSpeakerSystem:
    """Return a temporary full-domain system without modifying ``system``.

    Generated full-domain meshes may be supplied by resource name.  Other
    resources are reflected from their current reduced solve mesh.  Every
    written temporary resource is expressed directly in metres with zero
    translation, which makes all reflection and welding occur in the project
    coordinate frame.
    """

    mode = normalize_symmetry(symmetry)
    if mode == "off":
        return FullDomainSpeakerSystem(
            system=system,
            component_source_ids={component.id: component.id for component in system.components},
            excitation_port_source_ids={port.id: port.id for port in system.excitation_ports},
            preferred_full_mesh_names=(),
        )

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    preferred = {str(name): Path(path) for name, path in (preferred_full_mesh_by_name or {}).items()}
    resources_by_id = {resource.id: resource for resource in system.meshes}
    boundaries_by_id = {boundary.id: boundary for boundary in system.boundaries}
    image_bits = _symmetry_image_bits(mode)

    orbits = {
        component.id: _component_orbit(component, system, resources_by_id, boundaries_by_id, mode, image_bits)
        for component in system.components
    }

    expanded_by_resource: dict[str, _ExpandedMesh] = {}
    preferred_names: list[str] = []
    for resource in system.meshes:
        source_path = Path(resource.file)
        preferred_path = preferred.get(resource.name)
        use_preferred = (
            preferred_path is not None
            and preferred_path.is_file()
            and _mesh_is_full_domain(preferred_path, resource, mode)
        )
        selected_path = preferred_path if use_preferred else source_path
        source_mesh = _mesh_in_project_coordinates(meshio.read(selected_path), resource)
        if use_preferred:
            expanded = _classify_full_domain_mesh(source_mesh, mode)
            preferred_names.append(resource.name)
        else:
            expanded = _expand_reduced_mesh(source_mesh, mode, resource.purpose)
        expanded_by_resource[resource.id] = expanded

    boundary_clones, boundary_clone_by_component_group, retag_requests = _clone_component_boundaries(
        system,
        orbits,
        expanded_by_resource,
    )
    _retag_component_surfaces(expanded_by_resource, retag_requests)

    temporary_resources = []
    for resource in system.meshes:
        expanded = expanded_by_resource[resource.id]
        suffix = ".msh"
        mesh_path = destination / f"{_safe_name(resource.name)}__full{suffix}"
        _write_expanded_mesh(mesh_path, expanded.mesh, resource.purpose)
        temporary_resources.append(
            replace(
                resource,
                file=str(mesh_path),
                scale_to_m=1.0,
                translation_m=(0.0, 0.0, 0.0),
            )
        )

    active_boundaries = _active_boundaries_after_expansion(
        system,
        tuple(boundary_clones),
        expanded_by_resource,
    )
    active_boundary_ids = {boundary.id for boundary in active_boundaries}
    for interface in system.interfaces:
        if (
            interface.bounded_boundary_id not in active_boundary_ids
            or interface.unbounded_boundary_id not in active_boundary_ids
        ):
            raise ValueError(f"Symmetry expansion removed a boundary used by interface '{interface.name}'.")

    components, component_sources = _clone_components(
        system,
        orbits,
        boundary_clone_by_component_group,
    )
    ports, port_sources = _clone_excitation_ports(system, components, component_sources)
    metadata = dict(system.metadata)
    metadata["speaker_export_symmetry_expansion"] = {
        "source_symmetry": mode,
        "temporary_full_domain": True,
        "preferred_full_mesh_names": sorted(preferred_names),
        "component_source_ids": dict(component_sources),
        "excitation_port_source_ids": dict(port_sources),
    }
    temporary_system = replace(
        system,
        meshes=tuple(temporary_resources),
        boundaries=tuple(active_boundaries),
        components=tuple(components),
        excitation_ports=tuple(ports),
        metadata=metadata,
    )
    return FullDomainSpeakerSystem(
        system=temporary_system,
        component_source_ids=component_sources,
        excitation_port_source_ids=port_sources,
        preferred_full_mesh_names=tuple(sorted(preferred_names)),
    )


def preferred_full_meshes_from_project_payload(payload: Mapping[str, object]) -> dict[str, Path]:
    """Return retained generated full-domain meshes keyed by physical-system mesh name."""

    result: dict[str, Path] = {}
    documents = payload.get("generator_documents")
    if not isinstance(documents, list):
        return result
    for raw_document in documents:
        if not isinstance(raw_document, dict) or raw_document.get("mesh_enabled", True) is False:
            continue
        artifact = raw_document.get("artifact")
        if not isinstance(artifact, dict):
            continue
        raw_path = artifact.get("cleaned_mesh_path") or artifact.get("mesh_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        path = Path(raw_path)
        if path.is_file():
            result[_safe_name(str(raw_document.get("name", "mesh")))] = path
    return result


def _symmetry_image_bits(mode: str) -> tuple[tuple[int, int], ...]:
    if mode == "x":
        return ((0, 0), (1, 0))
    if mode == "xy":
        return ((0, 0), (1, 0), (0, 1), (1, 1))
    return ((0, 0),)


def _mesh_in_project_coordinates(mesh: meshio.Mesh, resource) -> meshio.Mesh:
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


def _mesh_is_full_domain(path: Path, resource, mode: str) -> bool:
    try:
        mesh = _mesh_in_project_coordinates(meshio.read(path), resource)
    except Exception:
        return False
    points = np.asarray(mesh.points, dtype=float)
    if not len(points):
        return False
    tolerance = symmetry_plane_tolerance_m(points)
    return all(float(np.min(points[:, _AXIS_INDEX[axis]])) < -tolerance for axis in _ACTIVE_AXES[mode])


def _classify_full_domain_mesh(mesh: meshio.Mesh, mode: str) -> _ExpandedMesh:
    points = np.asarray(mesh.points, dtype=float)
    tolerance = symmetry_plane_tolerance_m(points)
    bits = _symmetry_image_bits(mode)
    lookup = {value: index for index, value in enumerate(bits)}
    image_blocks = []
    for block in mesh.cells:
        connectivity = np.asarray(block.data, dtype=np.int64)
        centroids = np.mean(points[connectivity[:, : _corner_count(block.type)]], axis=1)
        classified = []
        for centroid in centroids:
            x_bit = int("x" in _ACTIVE_AXES[mode] and centroid[0] < -tolerance)
            y_bit = int("y" in _ACTIVE_AXES[mode] and centroid[1] < -tolerance)
            classified.append(lookup[(x_bit, y_bit)])
        image_blocks.append(np.asarray(classified, dtype=np.int8))
    return _ExpandedMesh(
        mesh=_remove_fem_cut_facets(mesh, image_blocks, mode), image_index_by_block=image_blocks, image_bits=bits
    )


def _expand_reduced_mesh(mesh: meshio.Mesh, mode: str, purpose: MeshPurpose) -> _ExpandedMesh:
    points = snap_points_to_symmetry_planes(np.asarray(mesh.points, dtype=float), mode)
    tolerance = symmetry_plane_tolerance_m(points)
    bits = _symmetry_image_bits(mode)
    expanded_points: list[np.ndarray] = []
    node_lookup: dict[tuple[object, ...], int] = {}
    node_maps: list[np.ndarray] = []
    scale = max(tolerance, np.finfo(float).eps)
    for image_index, (x_bit, y_bit) in enumerate(bits):
        signs = np.asarray((-1.0 if x_bit else 1.0, -1.0 if y_bit else 1.0, 1.0))
        transformed = points * signs
        node_map = np.empty(len(points), dtype=np.int64)
        for source_index, point in enumerate(transformed):
            quantized = tuple(int(round(float(value) / scale)) for value in point)
            on_symmetry_plane = any(
                abs(float(point[_AXIS_INDEX[axis]])) <= tolerance for axis in _ACTIVE_AXES[mode]
            )
            # Plane nodes are welded geometrically, including coincident source
            # nodes that came from separate Gmsh entities. Away from a cut
            # plane, image/source identity prevents accidental welding of
            # intentionally disconnected coincident surfaces.
            key = ("plane", *quantized) if on_symmetry_plane else ("image", image_index, source_index)
            target = node_lookup.get(key)
            if target is None:
                target = len(expanded_points)
                node_lookup[key] = target
                expanded_points.append(point.copy())
            node_map[source_index] = target
        node_maps.append(node_map)

    new_cells = []
    image_blocks: list[np.ndarray] = []
    kept_source_rows: list[np.ndarray] = []
    for block_index, block in enumerate(mesh.cells):
        source = np.asarray(block.data, dtype=np.int64)
        rows = []
        images = []
        sources = []
        seen: set[tuple[int, ...]] = set()
        for image_index, ((x_bit, y_bit), node_map) in enumerate(zip(bits, node_maps, strict=True)):
            reflected = (x_bit + y_bit) % 2 == 1
            for source_row, cell in enumerate(source):
                mapped = node_map[cell].copy()
                if reflected:
                    mapped = _reverse_cell_orientation(mapped, block.type)
                if purpose == MeshPurpose.FEM_VOLUME and block.type in _TRIANGLE_TYPES:
                    corners = np.asarray(expanded_points, dtype=float)[mapped[:3]]
                    if _lies_on_active_plane(corners, mode, tolerance):
                        continue
                key = (source_row, *sorted(int(value) for value in mapped[: _corner_count(block.type)]))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(mapped)
                images.append(image_index)
                sources.append(source_row)
        width = source.shape[1] if source.ndim == 2 else 0
        new_cells.append((block.type, np.asarray(rows, dtype=np.int64).reshape((-1, width))))
        image_blocks.append(np.asarray(images, dtype=np.int8))
        kept_source_rows.append(np.asarray(sources, dtype=np.int64))

    new_cell_data: dict[str, list[np.ndarray]] = {}
    for name, blocks in mesh.cell_data.items():
        new_cell_data[name] = [
            np.asarray(values)[source_rows] for values, source_rows in zip(blocks, kept_source_rows, strict=True)
        ]
    expanded_mesh = meshio.Mesh(
        points=np.asarray(expanded_points, dtype=float).reshape((-1, 3)),
        cells=new_cells,
        cell_data=new_cell_data,
        field_data={name: np.asarray(value).copy() for name, value in mesh.field_data.items()},
    )
    return _ExpandedMesh(mesh=expanded_mesh, image_index_by_block=image_blocks, image_bits=bits)


def _remove_fem_cut_facets(mesh: meshio.Mesh, image_blocks: list[np.ndarray], mode: str) -> meshio.Mesh:
    if not any(block.type in _TETRA_TYPES for block in mesh.cells):
        return mesh
    points = np.asarray(mesh.points, dtype=float)
    tolerance = symmetry_plane_tolerance_m(points)
    masks = []
    for block in mesh.cells:
        connectivity = np.asarray(block.data, dtype=np.int64)
        if block.type in _TRIANGLE_TYPES:
            masks.append(
                np.asarray(
                    [not _lies_on_active_plane(points[cell[:3]], mode, tolerance) for cell in connectivity],
                    dtype=bool,
                )
            )
        else:
            masks.append(np.ones(len(connectivity), dtype=bool))
    if all(np.all(mask) for mask in masks):
        return mesh
    cells = [(block.type, np.asarray(block.data)[mask]) for block, mask in zip(mesh.cells, masks, strict=True)]
    for index, mask in enumerate(masks):
        image_blocks[index] = image_blocks[index][mask]
    cell_data = {
        name: [np.asarray(values)[mask] for values, mask in zip(blocks, masks, strict=True)]
        for name, blocks in mesh.cell_data.items()
    }
    return meshio.Mesh(points=points, cells=cells, cell_data=cell_data, field_data=mesh.field_data)


def _lies_on_active_plane(points: np.ndarray, mode: str, tolerance: float) -> bool:
    return any(np.max(np.abs(points[:, _AXIS_INDEX[axis]])) <= tolerance for axis in _ACTIVE_AXES[mode])


def _corner_count(cell_type: str) -> int:
    if cell_type == "vertex":
        return 1
    if cell_type in _TRIANGLE_TYPES:
        return 3
    if cell_type in _TETRA_TYPES:
        return 4
    if cell_type in {"line", "line2"}:
        return 2
    raise ValueError(f"Level-3 symmetry expansion does not support mesh cell type {cell_type!r}.")


def _reverse_cell_orientation(cell: np.ndarray, cell_type: str) -> np.ndarray:
    if cell_type in _TRIANGLE_TYPES:
        cell[[1, 2]] = cell[[2, 1]]
    elif cell_type in _TETRA_TYPES:
        cell[[0, 1]] = cell[[1, 0]]
    elif cell_type not in {"vertex", "line", "line2"}:
        raise ValueError(f"Cannot correct reflected orientation for cell type {cell_type!r}.")
    return cell


def _component_orbit(
    component: PhysicalComponent,
    system: PhysicalSystem,
    resources_by_id,
    boundaries_by_id,
    mode: str,
    image_bits: tuple[tuple[int, int], ...],
) -> _ComponentOrbit:
    boundaries = tuple(boundaries_by_id[value] for value in component.boundary_ids)
    try:
        inference = infer_component_symmetry(boundaries, resources_by_id, mode)
        fractional_axes = inference.fractional_symmetry_axes
    except ComponentSymmetryInferenceError as exc:
        raw_axes = component.parameters.get("fractional_symmetry_axes")
        if not isinstance(raw_axes, list):
            raise ValueError(f"Could not expand component '{component.name}': {exc}") from exc
        fractional_axes = tuple(str(axis).lower() for axis in raw_axes)
    if any(axis not in _ACTIVE_AXES[mode] for axis in fractional_axes):
        raise ValueError(f"Component '{component.name}' has invalid fractional symmetry axes.")
    orbit_axes = tuple(axis for axis in _ACTIVE_AXES[mode] if axis not in fractional_axes)
    keys = []
    for x_bit, y_bit in image_bits:
        by_axis = {"x": x_bit, "y": y_bit}
        keys.append(tuple(by_axis[axis] for axis in orbit_axes))
    unique_keys = tuple(dict.fromkeys(keys))
    group_by_image = tuple(unique_keys.index(key) for key in keys)
    representatives = tuple(group_by_image.index(index) for index in range(len(unique_keys)))
    return _ComponentOrbit(component, fractional_axes, group_by_image, representatives)


def _clone_component_boundaries(system, orbits, expanded_by_resource):
    component_by_boundary: dict[str, str] = {}
    for component in system.components:
        for boundary_id in component.boundary_ids:
            previous = component_by_boundary.setdefault(boundary_id, component.id)
            if previous != component.id:
                raise ValueError(f"Moving boundary '{boundary_id}' belongs to more than one component.")

    clones: list[Boundary] = [boundary for boundary in system.boundaries if boundary.id not in component_by_boundary]
    clone_map: dict[tuple[str, int, str], Boundary] = {}
    retag_requests = []
    boundaries_by_id = {boundary.id: boundary for boundary in system.boundaries}
    next_tag_by_resource = {
        resource_id: _next_physical_tag(expanded.mesh, 2) for resource_id, expanded in expanded_by_resource.items()
    }
    for component in system.components:
        orbit = orbits[component.id]
        group_count = len(orbit.representative_image_by_group)
        for boundary_id in component.boundary_ids:
            boundary = boundaries_by_id[boundary_id]
            source_tag = _physical_group_tag(expanded_by_resource[boundary.group.mesh_id].mesh, boundary.group)
            for group_index in range(group_count):
                if group_index == 0:
                    tag = source_tag
                    name = boundary.group.name or _physical_name_for_tag(
                        expanded_by_resource[boundary.group.mesh_id].mesh, source_tag, 2
                    )
                    clone_id = boundary.id
                    clone_name = boundary.name
                else:
                    tag = next_tag_by_resource[boundary.group.mesh_id]
                    next_tag_by_resource[boundary.group.mesh_id] += 1
                    suffix = _image_label(
                        expanded_by_resource[boundary.group.mesh_id].image_bits[
                            orbit.representative_image_by_group[group_index]
                        ]
                    )
                    base_name = boundary.group.name or f"surface_{source_tag}"
                    name = f"{base_name}__{suffix}"
                    clone_id = f"{boundary.id}__{suffix}"
                    clone_name = f"{boundary.name} ({suffix})"
                clone = replace(
                    boundary,
                    id=clone_id,
                    name=clone_name,
                    group=PhysicalGroupRef(
                        mesh_id=boundary.group.mesh_id,
                        dimension=2,
                        name=name,
                        tag=tag,
                    ),
                )
                clones.append(clone)
                clone_map[(component.id, group_index, boundary_id)] = clone
                retag_requests.append(
                    (
                        boundary.group.mesh_id,
                        source_tag,
                        group_index,
                        orbit.group_by_image,
                        tag,
                        name,
                    )
                )
    return clones, clone_map, retag_requests


def _retag_component_surfaces(expanded_by_resource, requests) -> None:
    by_resource_tag: dict[tuple[str, int], list[tuple[int, tuple[int, ...], int, str]]] = defaultdict(list)
    for resource_id, source_tag, group, group_by_image, target_tag, name in requests:
        by_resource_tag[(resource_id, source_tag)].append((group, group_by_image, target_tag, name))
    for (resource_id, source_tag), rules in by_resource_tag.items():
        expanded = expanded_by_resource[resource_id]
        physical = expanded.mesh.cell_data.get("gmsh:physical")
        if physical is None:
            raise ValueError(f"Mesh resource '{resource_id}' has no physical cell tags.")
        for block_index, block in enumerate(expanded.mesh.cells):
            if block.type not in _TRIANGLE_TYPES:
                continue
            tags = np.asarray(physical[block_index])
            images = expanded.image_index_by_block[block_index]
            source_rows = np.flatnonzero(tags == source_tag)
            for row in source_rows:
                image_index = int(images[row])
                matching = [rule for rule in rules if rule[1][image_index] == rule[0]]
                if len(matching) != 1:
                    raise ValueError(f"Could not assign symmetry image {image_index} of physical tag {source_tag}.")
                tags[row] = matching[0][2]
        for _group, _group_by_image, target_tag, name in rules:
            expanded.mesh.field_data[name] = np.asarray([target_tag, 2], dtype=np.int32)


def _clone_components(system, orbits, clone_map):
    components = []
    source_by_clone = {}
    for component in system.components:
        orbit = orbits[component.id]
        for group_index, representative in enumerate(orbit.representative_image_by_group):
            bits = _symmetry_image_bits_from_index(orbit, representative)
            suffix = _image_label(bits)
            component_id = component.id if group_index == 0 else f"{component.id}__{suffix}"
            name = component.name if group_index == 0 else f"{component.name} ({suffix})"
            boundary_ids = tuple(
                clone_map[(component.id, group_index, boundary_id)].id for boundary_id in component.boundary_ids
            )
            parameters = {
                key: value
                for key, value in component.parameters.items()
                if key
                not in {
                    "symmetry_role",
                    "surface_completion_factor",
                    "physical_driver_orbit_count",
                    "fractional_symmetry_axes",
                }
            }
            for mapping_key in ("boundary_motion_signs", "boundary_motion_weights"):
                raw = parameters.get(mapping_key)
                if isinstance(raw, dict):
                    parameters[mapping_key] = {
                        clone_map[(component.id, group_index, boundary_id)].id: value
                        for boundary_id, value in raw.items()
                        if (component.id, group_index, boundary_id) in clone_map
                    }
            if "motion_axis" in parameters:
                axis = np.asarray(parameters["motion_axis"], dtype=float)
                signs = np.asarray((-1.0 if bits[0] else 1.0, -1.0 if bits[1] else 1.0, 1.0))
                parameters["motion_axis"] = (axis * signs).tolist()
            parameters.update(
                {
                    "symmetry_role": "complete_representative",
                    "surface_completion_factor": 1,
                    "physical_driver_orbit_count": 1,
                    "fractional_symmetry_axes": [],
                }
            )
            clone = replace(
                component,
                id=component_id,
                name=name,
                boundary_ids=boundary_ids,
                parameters=parameters,
            )
            components.append(clone)
            source_by_clone[component_id] = component.id
    return components, source_by_clone


def _symmetry_image_bits_from_index(orbit: _ComponentOrbit, image_index: int) -> tuple[int, int]:
    active_axes = set(orbit.fractional_axes)
    # Representative images are always chosen from the canonical identity/X/Y/XY ordering.
    all_bits = ((0, 0), (1, 0), (0, 1), (1, 1))
    bits = all_bits[image_index]
    return tuple(0 if axis in active_axes else bits[index] for index, axis in enumerate(("x", "y")))


def _clone_excitation_ports(system, components, component_sources):
    clones_by_source: dict[str, list[PhysicalComponent]] = defaultdict(list)
    for component in components:
        clones_by_source[component_sources[component.id]].append(component)
    ports = []
    source_by_port = {}
    for port in system.excitation_ports:
        component_clones = clones_by_source[port.component_id]
        for index, component in enumerate(component_clones):
            suffix = component.id.removeprefix(port.component_id).lstrip("_")
            port_id = port.id if index == 0 else f"{port.id}__{suffix}"
            name = port.name if index == 0 else f"{port.name} ({component.name})"
            clone = replace(port, id=port_id, name=name, component_id=component.id)
            ports.append(clone)
            source_by_port[port_id] = port.id
    return ports, source_by_port


def _active_boundaries_after_expansion(system, boundaries, expanded_by_resource):
    present_by_region_mesh: dict[tuple[str, str], set[int]] = {}
    for region in system.regions:
        for mesh_id in region.mesh_ids:
            mesh = expanded_by_resource[mesh_id].mesh
            if region.kind == AcousticRegionKind.BOUNDED_AIR:
                volume_tags = {int(group.tag) for group in region.volume_groups if group.mesh_id == mesh_id}
                present = set(selected_volume_surface_tags(mesh, volume_tags))
            else:
                physical = mesh.cell_data.get("gmsh:physical", [])
                present = {
                    int(tag)
                    for block, values in zip(mesh.cells, physical, strict=True)
                    if block.type in _TRIANGLE_TYPES
                    for tag in np.unique(values)
                }
            present_by_region_mesh[(region.id, mesh_id)] = present
    active = []
    for boundary in boundaries:
        tag = _physical_group_tag(expanded_by_resource[boundary.group.mesh_id].mesh, boundary.group)
        if tag in present_by_region_mesh.get((boundary.region_id, boundary.group.mesh_id), set()):
            active.append(boundary)
    return active


def _physical_group_tag(mesh: meshio.Mesh, group: PhysicalGroupRef) -> int:
    if group.name is not None and group.name in mesh.field_data:
        tag, dimension = map(int, np.asarray(mesh.field_data[group.name]).tolist())
        if dimension == group.dimension:
            return tag
    if group.tag is not None:
        return int(group.tag)
    raise ValueError(f"Physical group {group.name!r} has no resolvable tag.")


def _physical_name_for_tag(mesh: meshio.Mesh, tag: int, dimension: int) -> str:
    for name, raw in mesh.field_data.items():
        candidate_tag, candidate_dimension = map(int, np.asarray(raw).tolist())
        if candidate_tag == tag and candidate_dimension == dimension:
            return str(name)
    return f"physical_{dimension}_{tag}"


def _next_physical_tag(mesh: meshio.Mesh, dimension: int) -> int:
    tags = [int(np.asarray(raw)[0]) for raw in mesh.field_data.values() if int(np.asarray(raw)[1]) == dimension]
    return max(tags, default=0) + 1


def _write_expanded_mesh(path: Path, mesh: meshio.Mesh, purpose: MeshPurpose) -> None:
    if purpose == MeshPurpose.BEM_SURFACE:
        meshio.write(path, mesh, file_format="gmsh22", binary=False)
        return
    prepared = _gmsh41_mesh(mesh)
    meshio.write(path, prepared, file_format="gmsh", binary=False)


def _gmsh41_mesh(mesh: meshio.Mesh) -> meshio.Mesh:
    """Normalize blocks for meshio's Gmsh 4.1 entity-oriented writer."""

    physical = mesh.cell_data.get("gmsh:physical")
    geometrical = mesh.cell_data.get("gmsh:geometrical")
    if physical is None or geometrical is None:
        raise ValueError("Expanded FEM meshes require physical and geometrical cell tags.")
    # A Gmsh entity can belong to only one physical group of a given
    # dimension. Reflections intentionally reuse source geometrical tags, but
    # component re-expansion can assign their faces to different physical
    # groups. Give each (source entity, physical group) pair its own temporary
    # entity before asking meshio to write the entity-oriented 4.1 format.
    next_geometry_tag = {
        dimension: max(
            (
                int(tag)
                for block, values in zip(mesh.cells, geometrical, strict=True)
                if _cell_dimension(block.type) == dimension
                for tag in np.asarray(values)
            ),
            default=0,
        )
        + 1
        for dimension in range(4)
    }
    entity_tag_by_key: dict[tuple[int, int, int], int] = {}
    first_physical_by_entity: dict[tuple[int, int], int] = {}
    normalized_geometry: list[np.ndarray] = []
    for block, geometry_values, physical_values in zip(mesh.cells, geometrical, physical, strict=True):
        dimension = _cell_dimension(block.type)
        normalized = np.asarray(geometry_values, dtype=np.int64).copy()
        for row, (geometry_tag, physical_tag) in enumerate(
            zip(normalized, np.asarray(physical_values, dtype=np.int64), strict=True)
        ):
            entity_key = (dimension, int(geometry_tag))
            first_physical = first_physical_by_entity.setdefault(entity_key, int(physical_tag))
            key = (dimension, int(geometry_tag), int(physical_tag))
            if key not in entity_tag_by_key:
                if int(physical_tag) == first_physical:
                    entity_tag_by_key[key] = int(geometry_tag)
                else:
                    entity_tag_by_key[key] = next_geometry_tag[dimension]
                    next_geometry_tag[dimension] += 1
            normalized[row] = entity_tag_by_key[key]
        normalized_geometry.append(normalized)

    cells = []
    cell_data: dict[str, list[np.ndarray]] = {name: [] for name in mesh.cell_data}
    for block_index, block in enumerate(mesh.cells):
        geometry_tags = normalized_geometry[block_index]
        for geometry_tag in dict.fromkeys(int(value) for value in geometry_tags):
            mask = geometry_tags == geometry_tag
            if not np.any(mask):
                continue
            cells.append((block.type, np.asarray(block.data)[mask]))
            for name, blocks in mesh.cell_data.items():
                values = geometry_tags if name == "gmsh:geometrical" else np.asarray(blocks[block_index])
                cell_data[name].append(values[mask])
    entity_dim_tags = []
    for block, values in zip(cells, cell_data["gmsh:geometrical"], strict=True):
        if len(values):
            dimension = _cell_dimension(block[0])
            entity_dim_tags.append((dimension, int(values[0])))
    entity_dim_tags = list(dict.fromkeys(entity_dim_tags))
    if len(entity_dim_tags) > len(mesh.points):
        raise ValueError("Expanded FEM mesh has more geometric entities than nodes.")
    volume_entity = next((item for item in entity_dim_tags if item[0] == 3), (3, 1))
    node_dim_tags = np.tile(np.asarray([volume_entity], dtype=np.int32), (len(mesh.points), 1))
    for node_index, entity in enumerate(entity_dim_tags):
        node_dim_tags[node_index] = entity
    point_data = dict(mesh.point_data)
    point_data["gmsh:dim_tags"] = node_dim_tags
    return meshio.Mesh(
        points=np.asarray(mesh.points),
        cells=cells,
        point_data=point_data,
        cell_data=cell_data,
        field_data=mesh.field_data,
    )


def _cell_dimension(cell_type: str) -> int:
    if cell_type == "vertex":
        return 0
    if cell_type in {"line", "line2"}:
        return 1
    if cell_type in _TRIANGLE_TYPES:
        return 2
    if cell_type in _TETRA_TYPES:
        return 3
    raise ValueError(f"Level-3 symmetry expansion does not support mesh cell type {cell_type!r}.")


def _image_label(bits: tuple[int, int]) -> str:
    if bits == (0, 0):
        return "identity"
    axes = "".join(axis for axis, bit in zip(("x", "y"), bits, strict=True) if bit)
    return f"reflect_{axes}"


def _safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in value).strip("_")
    return cleaned or "mesh"


__all__ = [
    "FullDomainSpeakerSystem",
    "expand_speaker_system_for_export",
    "preferred_full_meshes_from_project_payload",
]
