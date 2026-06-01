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


def validate_reduced_mesh_config(
    mesh_config: MeshConfig,
    symmetry: str,
    *,
    tolerance_m: float = 1e-9,
) -> None:
    mode = normalize_symmetry(symmetry)
    axes = _SYMMETRY_AXES[mode]
    if not axes:
        return

    mesh = meshio.read(mesh_config.file)
    scale_factor = 0.001 if mesh_config.scale_factor is None else float(mesh_config.scale_factor)
    points_m = np.asarray(mesh.points, dtype=float) * scale_factor + np.asarray(mesh_config.translation_m, dtype=float)

    for axis_index in axes:
        coordinates = points_m[:, axis_index]
        vertex_index = int(np.argmin(coordinates))
        coordinate = float(coordinates[vertex_index])
        if coordinate < -float(tolerance_m):
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
    tolerance_m: float = 1e-9,
) -> None:
    for mesh_config in mesh_configs:
        validate_reduced_mesh_config(mesh_config, symmetry, tolerance_m=tolerance_m)
