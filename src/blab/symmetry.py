"""Application-side helpers for mirror-symmetry solve setup."""

from __future__ import annotations

from dataclasses import dataclass

import meshio
import numpy as np

from blab.config import MeshConfig, normalize_symmetry
from blab.solvers.registry import backend_info

_SYMMETRY_AXES = {
    "off": (),
    "x": (0,),
    "xy": (0, 1),
}
_AXIS_LABELS = ("X", "Y", "Z")
_ABSOLUTE_PLANE_TOLERANCE_M = 1e-9
_RELATIVE_PLANE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class SymmetryValidationIssue:
    mesh_name: str
    axis: str
    vertex_index: int
    coordinate_m: float


class SymmetryValidationError(ValueError):
    def __init__(self, issue: SymmetryValidationIssue, symmetry: str):
        self.issue = issue
        self.symmetry = symmetry
        super().__init__(
            f"Mesh '{issue.mesh_name}' is not in the positive {issue.axis} fundamental domain "
            f"for {symmetry.upper()} symmetry. Vertex {issue.vertex_index} has "
            f"{issue.axis.lower()}={issue.coordinate_m:.6g} m after scale and translation."
        )


def backend_supports_symmetry(backend_id: str) -> bool:
    return backend_info(backend_id).capabilities.supports_symmetry


def effective_symmetry_for_backend(symmetry: object, backend_id: str) -> str:
    mode = normalize_symmetry(symmetry)
    if mode == "off" or backend_supports_symmetry(backend_id):
        return mode
    return "off"


def symmetry_plane_tolerance_m(points_m: np.ndarray) -> float:
    """Return a conservative tolerance for numerical symmetry-plane noise.

    Gmsh coordinates that have passed through single-precision geometry tools
    can retain plane coordinates a few ULPs away from zero.  Scale the cutoff
    with the model while retaining a one-nanometre floor for small meshes.
    """

    points = np.asarray(points_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Mesh points must have shape (vertex, 3).")
    if not np.all(np.isfinite(points)):
        raise ValueError("Mesh points must contain only finite coordinates.")
    model_scale_m = float(np.linalg.norm(np.ptp(points, axis=0))) if len(points) else 0.0
    return max(_ABSOLUTE_PLANE_TOLERANCE_M, model_scale_m * _RELATIVE_PLANE_TOLERANCE)


def snap_points_to_symmetry_planes(
    points_m: np.ndarray,
    symmetry: object,
    *,
    tolerance_m: float | None = None,
) -> np.ndarray:
    """Copy points and set coordinates near active symmetry planes to zero."""

    mode = normalize_symmetry(symmetry)
    points = np.asarray(points_m, dtype=float)
    resolved_tolerance = symmetry_plane_tolerance_m(points) if tolerance_m is None else float(tolerance_m)
    if not np.isfinite(resolved_tolerance) or resolved_tolerance < 0.0:
        raise ValueError("Symmetry-plane tolerance must be finite and non-negative.")

    snapped = points.copy()
    for axis_index in _SYMMETRY_AXES[mode]:
        coordinates = snapped[:, axis_index]
        coordinates[np.abs(coordinates) <= resolved_tolerance] = 0.0
    return snapped


def validate_reduced_mesh_config(
    mesh_config: MeshConfig,
    symmetry: str,
    *,
    tolerance_m: float | None = None,
) -> None:
    mode = normalize_symmetry(symmetry)
    axes = _SYMMETRY_AXES[mode]
    if not axes:
        return

    mesh = meshio.read(mesh_config.file)
    scale_factor = 0.001 if mesh_config.scale_factor is None else float(mesh_config.scale_factor)
    points_m = np.asarray(mesh.points, dtype=float) * scale_factor + np.asarray(mesh_config.translation_m, dtype=float)
    resolved_tolerance = symmetry_plane_tolerance_m(points_m) if tolerance_m is None else float(tolerance_m)
    if not np.isfinite(resolved_tolerance) or resolved_tolerance < 0.0:
        raise ValueError("Symmetry-plane tolerance must be finite and non-negative.")

    for axis_index in axes:
        coordinates = points_m[:, axis_index]
        vertex_index = int(np.argmin(coordinates))
        coordinate = float(coordinates[vertex_index])
        if coordinate < -resolved_tolerance:
            raise SymmetryValidationError(
                SymmetryValidationIssue(
                    mesh_name=mesh_config.name,
                    axis=_AXIS_LABELS[axis_index],
                    vertex_index=vertex_index,
                    coordinate_m=coordinate,
                ),
                mode,
            )


def validate_reduced_mesh_configs(
    mesh_configs: tuple[MeshConfig, ...],
    symmetry: str,
    *,
    tolerance_m: float | None = None,
) -> None:
    for mesh_config in mesh_configs:
        validate_reduced_mesh_config(mesh_config, symmetry, tolerance_m=tolerance_m)
