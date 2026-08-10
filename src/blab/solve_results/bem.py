"""Frequency-invariant BEM domains used by retained boundary traces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from blab.physical_model import AcousticRegionKind, CompiledPhysicalSystem, MeshPurpose
from blab.solve_results.model import (
    BEM_BOUNDARY_DOMAIN_ID,
    BEM_BOUNDARY_NEUMANN_ID,
    BEM_BOUNDARY_PRESSURE_ID,
    ResultDomain,
    SolvedSystem,
)


@dataclass(frozen=True)
class BemBoundaryTraces:
    """Validated boundary geometry and excitation-basis traces for one run."""

    run_id: str
    frequencies_hz: np.ndarray
    frequency_indices: np.ndarray
    points_m: np.ndarray
    triangles: np.ndarray
    pressure_by_frequency: np.ndarray
    normal_derivative_by_frequency: np.ndarray
    excitation_ids: tuple[str, ...]
    symmetry: str = "off"

    @property
    def trace_storage_nbytes(self) -> int:
        return int(self.pressure_by_frequency.nbytes + self.normal_derivative_by_frequency.nbytes)

    def frequency_index(self, frequency_hz: float | None) -> int:
        if not self.frequencies_hz.size:
            raise ValueError("BEM boundary traces contain no solved frequencies.")
        if frequency_hz is None or not np.isfinite(frequency_hz):
            return int(self.frequency_indices[0])
        available_index = int(np.argmin(np.abs(self.frequencies_hz - float(frequency_hz))))
        return int(self.frequency_indices[available_index])

    def excitation_basis(self, frequency_hz: float | None) -> tuple[np.ndarray, np.ndarray]:
        solve_index = self.frequency_index(frequency_hz)
        return (
            np.asarray(self.pressure_by_frequency[solve_index]),
            np.asarray(self.normal_derivative_by_frequency[solve_index]),
        )


def bem_boundary_traces_from_solved_system(solved: SolvedSystem | None) -> BemBoundaryTraces | None:
    """Return a validated, zero-copy view of retained BEM boundary traces."""

    if solved is None:
        return None
    domain = solved.domains.get(BEM_BOUNDARY_DOMAIN_ID)
    pressure = solved.quantities.get(BEM_BOUNDARY_PRESSURE_ID)
    normal_derivative = solved.quantities.get(BEM_BOUNDARY_NEUMANN_ID)
    if domain is None and pressure is None and normal_derivative is None:
        return None
    if domain is None or pressure is None or normal_derivative is None:
        raise ValueError("The solved system contains an incomplete BEM boundary trace result.")

    points = np.asarray(domain.coordinates.get("points_m"), dtype=float)
    triangles = np.asarray(domain.topology.get("triangles"), dtype=np.int64)
    pressure_values = np.asarray(pressure.values)
    normal_values = np.asarray(normal_derivative.values)
    frequency_count = solved.frequencies_hz.size
    excitation_count = len(solved.excitation_ids)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("The BEM boundary domain points must have shape (bem_node, 3).")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("The BEM boundary topology must have shape (bem_face, 3).")
    if pressure.dimensions != ("frequency", "excitation", "bem_node"):
        raise ValueError("BEM boundary pressure has unexpected dimensions.")
    if normal_derivative.dimensions != ("frequency", "excitation", "bem_face"):
        raise ValueError("BEM boundary normal derivative has unexpected dimensions.")
    if pressure_values.shape != (frequency_count, excitation_count, points.shape[0]):
        raise ValueError("BEM boundary pressure does not align with its frequencies, excitations, and nodes.")
    if normal_values.shape != (frequency_count, excitation_count, triangles.shape[0]):
        raise ValueError("BEM boundary normal derivative does not align with its frequencies, excitations, and faces.")

    availability = (
        np.asarray(solved.completion_mask, dtype=bool)
        & np.asarray(pressure.available_frequency_mask, dtype=bool)
        & np.asarray(normal_derivative.available_frequency_mask, dtype=bool)
    )
    frequency_indices = np.flatnonzero(availability)
    if not frequency_indices.size:
        return None
    frequency_indices = frequency_indices[
        np.argsort(np.asarray(solved.frequencies_hz)[frequency_indices], kind="stable")
    ]
    symmetry = str(domain.metadata.get("symmetry", solved.provenance.solver_options.get("symmetry", "off")))
    if symmetry not in {"off", "x", "xy"}:
        symmetry = "off"
    return BemBoundaryTraces(
        run_id=solved.run_id,
        frequencies_hz=np.asarray(solved.frequencies_hz)[frequency_indices],
        frequency_indices=frequency_indices,
        points_m=points,
        triangles=triangles,
        pressure_by_frequency=pressure_values,
        normal_derivative_by_frequency=normal_values,
        excitation_ids=solved.excitation_ids,
        symmetry=symmetry,
    )


def bem_boundary_result_domain(
    system: CompiledPhysicalSystem,
    *,
    symmetry: str = "off",
) -> ResultDomain:
    """Reconstruct the combined node/face order used by the Julia BEM backend.

    The backend reads Gmsh 2.2 node and triangle records in file order, then
    concatenates the unbounded region's meshes without welding their vertices.
    Repeating that small operation here makes retained P1 pressure and DP0
    normal-derivative arrays self-describing without repeating geometry in
    every streamed frequency result.
    """

    unbounded_regions = tuple(region for region in system.regions if region.kind == AcousticRegionKind.UNBOUNDED_AIR)
    if len(unbounded_regions) != 1:
        raise ValueError("A BEM boundary result domain requires exactly one unbounded acoustic region.")
    region = unbounded_regions[0]
    meshes_by_id = {mesh.id: mesh for mesh in system.meshes}
    points_by_mesh: list[np.ndarray] = []
    faces_by_mesh: list[np.ndarray] = []
    tags_by_mesh: list[np.ndarray] = []
    node_mesh_indices: list[np.ndarray] = []
    face_mesh_indices: list[np.ndarray] = []
    node_offsets: list[int] = []
    node_counts: list[int] = []
    face_offsets: list[int] = []
    face_counts: list[int] = []
    node_offset = 0
    face_offset = 0

    if not region.mesh_ids:
        raise ValueError("The unbounded acoustic region does not reference a BEM mesh.")
    for mesh_index, mesh_id in enumerate(region.mesh_ids):
        try:
            resource = meshes_by_id[mesh_id]
        except KeyError as exc:
            raise ValueError(f"The unbounded acoustic region references unknown mesh {mesh_id!r}.") from exc
        if resource.purpose != MeshPurpose.BEM_SURFACE:
            raise ValueError(f"Unbounded region mesh {mesh_id!r} is not a BEM surface mesh.")

        points, faces, physical_tags = _read_gmsh22_boundary_geometry(Path(resource.file))
        points = points * float(resource.scale_to_m) + np.asarray(resource.translation_m, dtype=float)
        points_by_mesh.append(points)
        faces_by_mesh.append(faces + node_offset)
        tags_by_mesh.append(physical_tags)
        node_mesh_indices.append(np.full(points.shape[0], mesh_index, dtype=np.int32))
        face_mesh_indices.append(np.full(faces.shape[0], mesh_index, dtype=np.int32))
        node_offsets.append(node_offset)
        node_counts.append(points.shape[0])
        face_offsets.append(face_offset)
        face_counts.append(faces.shape[0])
        node_offset += points.shape[0]
        face_offset += faces.shape[0]

    return ResultDomain(
        id=BEM_BOUNDARY_DOMAIN_ID,
        kind="bem_boundary",
        dimensions=("bem_node", "bem_face"),
        coordinates={
            "points_m": np.vstack(points_by_mesh),
            "mesh_index": np.concatenate(node_mesh_indices),
        },
        topology={
            "triangles": np.vstack(faces_by_mesh),
            "mesh_index": np.concatenate(face_mesh_indices),
            "source_physical_tag": np.concatenate(tags_by_mesh),
        },
        metadata={
            "region_id": region.id,
            "mesh_ids": list(region.mesh_ids),
            "node_offsets": node_offsets,
            "node_counts": node_counts,
            "face_offsets": face_offsets,
            "face_counts": face_counts,
            "pressure_space": "P1",
            "normal_derivative_space": "DP0",
            "field_representation": "double_layer_pressure_minus_single_layer_normal_derivative",
            "symmetry": str(symmetry),
        },
    )


def _read_gmsh22_boundary_geometry(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read exactly the Gmsh records consumed by ``load_gmsh22_with_tags``."""

    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        node_start = lines.index("$Nodes")
        node_end = lines.index("$EndNodes")
        element_start = lines.index("$Elements")
        element_end = lines.index("$EndElements")
    except ValueError as exc:
        raise ValueError(f"BEM mesh {path} is not a supported Gmsh 2.2 ASCII mesh.") from exc

    node_id_to_index: dict[int, int] = {}
    points: list[tuple[float, float, float]] = []
    for line in lines[node_start + 2 : node_end]:
        fields = line.split()
        if len(fields) < 4:
            continue
        node_id = int(fields[0])
        node_id_to_index[node_id] = len(points)
        points.append((float(fields[1]), float(fields[2]), float(fields[3])))

    faces: list[tuple[int, int, int]] = []
    physical_tags: list[int] = []
    for line in lines[element_start + 2 : element_end]:
        fields = line.split()
        if len(fields) < 8 or int(fields[1]) != 2:
            continue
        try:
            face = tuple(node_id_to_index[int(node_id)] for node_id in fields[-3:])
        except KeyError:
            continue
        faces.append(face)
        physical_tags.append(int(fields[3]))

    if not points or not faces:
        raise ValueError(f"BEM mesh {path} does not contain Gmsh 2.2 nodes and triangles.")
    return (
        np.asarray(points, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        np.asarray(physical_tags, dtype=np.int32),
    )


__all__ = [
    "BemBoundaryTraces",
    "bem_boundary_result_domain",
    "bem_boundary_traces_from_solved_system",
]
