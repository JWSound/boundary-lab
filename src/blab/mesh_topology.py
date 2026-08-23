"""Solver-independent topology checks for exterior triangular BEM meshes."""

from __future__ import annotations

from dataclasses import dataclass

import meshio
import numpy as np

from blab.config import MeshConfig, normalize_symmetry
from blab.symmetry import symmetry_plane_tolerance_m

_SYMMETRY_AXES = {
    "off": (),
    "x": (0,),
    "xy": (0, 1),
}


@dataclass(frozen=True)
class MeshTopologyIssue:
    """Invalid edge incidence found in one solver mesh."""

    mesh_name: str
    open_edge_segments_m: np.ndarray
    nonmanifold_edge_segments_m: np.ndarray
    symmetry_edge_count: int = 0

    @property
    def open_edge_count(self) -> int:
        return int(len(self.open_edge_segments_m))

    @property
    def nonmanifold_edge_count(self) -> int:
        return int(len(self.nonmanifold_edge_segments_m))

    @property
    def problem_edge_segments_m(self) -> np.ndarray:
        parts = [
            segments
            for segments in (self.open_edge_segments_m, self.nonmanifold_edge_segments_m)
            if len(segments)
        ]
        if not parts:
            return np.empty((0, 2, 3), dtype=float)
        return np.concatenate(parts, axis=0)


@dataclass(frozen=True)
class ExteriorMeshTopologyReport:
    """Topology result for the exact mesh resources used by an exterior solve."""

    meshes: tuple[MeshTopologyIssue, ...]
    symmetry: str

    @property
    def open_edge_count(self) -> int:
        return sum(mesh.open_edge_count for mesh in self.meshes)

    @property
    def nonmanifold_edge_count(self) -> int:
        return sum(mesh.nonmanifold_edge_count for mesh in self.meshes)

    @property
    def symmetry_edge_count(self) -> int:
        return sum(mesh.symmetry_edge_count for mesh in self.meshes)

    @property
    def has_warnings(self) -> bool:
        return self.open_edge_count > 0 or self.nonmanifold_edge_count > 0


def analyze_exterior_mesh_topology(
    mesh_configs: tuple[MeshConfig, ...],
    *,
    symmetry: str = "off",
) -> ExteriorMeshTopologyReport:
    """Check edge incidence using the same per-resource connectivity as the solver.

    Separate solver meshes deliberately remain separate: both production BEM
    loaders offset and concatenate their vertex arrays without welding them.
    Open edges on active symmetry planes are valid boundaries of the reduced
    computational domain and are therefore reported but not treated as issues.
    """

    mode = normalize_symmetry(symmetry)
    analyses = tuple(_analyze_mesh_config(mesh_config, mode) for mesh_config in mesh_configs)
    return ExteriorMeshTopologyReport(meshes=analyses, symmetry=mode)


def exterior_mesh_topology_warning_text(report: ExteriorMeshTopologyReport) -> str:
    """Build the user-facing pre-solve warning for a failed topology check."""

    lines = [
        "Open edges were detected in the exterior BEM surface. The solve may fail or produce unreliable results. "
        "Problem edges are highlighted red in the mesh preview pane.",
        "",
        f"Open edges: {report.open_edge_count:,}",
    ]
    if report.nonmanifold_edge_count:
        lines.append(f"Non-manifold edges: {report.nonmanifold_edge_count:,}")
    return "\n".join(lines)


def _analyze_mesh_config(mesh_config: MeshConfig, symmetry: str) -> MeshTopologyIssue:
    mesh = meshio.read(mesh_config.file)
    triangles = _triangle_connectivity(mesh)
    scale_factor = 0.001 if mesh_config.scale_factor is None else float(mesh_config.scale_factor)
    points_m = np.asarray(mesh.points, dtype=float) * scale_factor
    points_m += np.asarray(mesh_config.translation_m, dtype=float)
    if not np.all(np.isfinite(points_m)):
        raise ValueError(f"Mesh '{mesh_config.name}' contains non-finite vertex coordinates.")

    unique_edges, incidence = _edge_incidence(triangles)
    open_edges = unique_edges[incidence == 1]
    nonmanifold_edges = unique_edges[incidence > 2]
    invalid_open_edges, symmetry_edge_count = _exclude_symmetry_plane_edges(
        points_m,
        open_edges,
        symmetry,
    )
    return MeshTopologyIssue(
        mesh_name=mesh_config.name,
        open_edge_segments_m=_edge_segments(points_m, invalid_open_edges),
        nonmanifold_edge_segments_m=_edge_segments(points_m, nonmanifold_edges),
        symmetry_edge_count=symmetry_edge_count,
    )


def _triangle_connectivity(mesh: meshio.Mesh) -> np.ndarray:
    blocks = [
        np.asarray(block.data, dtype=np.int64)
        for block in mesh.cells
        if block.type in {"triangle", "triangle3"}
    ]
    if not blocks:
        raise ValueError("No triangle surface cells found in mesh.")
    triangles = np.vstack(blocks)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("Triangle surface connectivity must have shape (element, 3).")
    return triangles


def _edge_incidence(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    return np.unique(edges, axis=0, return_counts=True)


def _exclude_symmetry_plane_edges(
    points_m: np.ndarray,
    edges: np.ndarray,
    symmetry: str,
) -> tuple[np.ndarray, int]:
    candidates = np.asarray(edges, dtype=np.int64).reshape((-1, 2))
    axes = _SYMMETRY_AXES[symmetry]
    if not axes or not len(candidates):
        return candidates, 0

    tolerance_m = symmetry_plane_tolerance_m(points_m)
    allowed = np.zeros(len(candidates), dtype=bool)
    for axis in axes:
        allowed |= np.max(np.abs(points_m[candidates, axis]), axis=1) <= tolerance_m
    return candidates[~allowed], int(np.sum(allowed))


def _edge_segments(points_m: np.ndarray, edges: np.ndarray) -> np.ndarray:
    indices = np.asarray(edges, dtype=np.int64).reshape((-1, 2))
    if not len(indices):
        return np.empty((0, 2, 3), dtype=float)
    return np.asarray(points_m[indices], dtype=float)


__all__ = [
    "ExteriorMeshTopologyReport",
    "MeshTopologyIssue",
    "analyze_exterior_mesh_topology",
    "exterior_mesh_topology_warning_text",
]
