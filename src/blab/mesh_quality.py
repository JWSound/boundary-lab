"""Precision-sensitive surface checks and repair through the mesh cleaner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from blab.mesh_clean import AREA_TOL, clean_mesh
from blab.mesh_data import read_resource_mesh
from blab.physical_model import MeshPurpose


@dataclass(frozen=True)
class NearCoincidentVertices:
    mesh_id: str
    mesh_name: str
    vertex_count: int
    minimum_distance_m: float
    merge_tolerance_m: float
    location_m: tuple[float, float, float]


class MeshQualityError(ValueError):
    def __init__(self, issues):
        self.issues = tuple(issues)
        super().__init__(near_coincident_warning_text(self.issues))


def inspect_near_coincident_vertices(system) -> tuple[NearCoincidentVertices, ...]:
    """Check distinct, used vertices within each BEM resource, never across parts.

    Eight float32 epsilons at the coordinate/extent scale is a screening
    tolerance, not a claim that every flagged mesh will fail. Nearest-neighbor
    queries bound memory even when many vertices occupy the same position.
    FEM resources are excluded: the surface cleaner cannot preserve their cells.
    """
    issues = []
    active_ids = {mesh_id for region in system.regions for mesh_id in region.mesh_ids}
    for resource in system.meshes:
        if resource.id not in active_ids or resource.purpose != MeshPurpose.BEM_SURFACE:
            continue
        issue = inspect_resource_vertices(resource)
        if issue is not None:
            issues.append(issue)
    return tuple(issues)


def inspect_resource_vertices(resource, mesh=None):
    mesh = read_resource_mesh(resource) if mesh is None else mesh
    triangles = [block.data for block in mesh.cells if block.type == "triangle"]
    if not triangles:
        return None
    used = np.unique(np.concatenate(triangles))
    if len(used) < 2:
        return None
    points = np.asarray(mesh.points[used], dtype=float) * resource.scale_to_m
    points += np.asarray(resource.translation_m)
    if not np.all(np.isfinite(points)):
        raise ValueError(f"Mesh '{resource.name}' contains non-finite vertex coordinates.")
    extent = float(np.max(np.ptp(points, axis=0)))
    scale = max(extent, float(np.max(np.abs(points))))
    tolerance = 8 * np.finfo(np.float32).eps * scale
    distances, neighbors = cKDTree(points).query(points, k=2)
    close = distances[:, 1] <= tolerance
    if not np.any(close):
        return None
    index = int(np.argmin(distances[:, 1]))
    return NearCoincidentVertices(
        mesh_id=resource.id,
        mesh_name=resource.name,
        vertex_count=int(np.count_nonzero(close)),
        minimum_distance_m=float(distances[index, 1]),
        merge_tolerance_m=float(tolerance),
        location_m=tuple((points[index] + points[neighbors[index, 1]]) / 2),
    )


def near_coincident_warning_text(issues) -> str:
    details = "\n".join(
        f"{issue.mesh_name}: {issue.vertex_count} vertices; closest separation {issue.minimum_distance_m * 1000:.6g} mm"
        for issue in issues
    )
    return (
        "Nearly coincident vertices were detected in the surface mesh. "
        "They can create extremely thin triangles and cause the solver to fail "
        "or return invalid results.\n\n"
        + details
        + "\n\nAuto-repair uses mesh cleanup to merge these vertices and remove collapsed triangles. "
        "Imported source files are preserved; their existing cleaned mesh is updated."
    )


def repair_near_coincident_vertices(resource, issue, *, symmetry):
    """Use the existing cleaner, rejecting loss of groups or worse topology."""
    from blab.config import MeshConfig
    from blab.mesh_data import MeshData
    from blab.mesh_topology import analyze_exterior_mesh_topology

    original = read_resource_mesh(resource)
    if any(block.type != "triangle" for block in original.cells):
        raise ValueError(
            f"'{resource.name}' cannot be auto-repaired by the triangle surface cleaner. Remesh this part."
        )
    cleaned, changes, _, _ = clean_mesh(
        original,
        merge_tol=issue.merge_tolerance_m / resource.scale_to_m,
        area_tol=AREA_TOL,
        mirror_x=False,
    )
    before_tags = set(np.concatenate(original.cell_data.get("gmsh:physical", ())).tolist())
    after_tags = set(np.concatenate(cleaned.cell_data.get("gmsh:physical", ())).tolist())
    if not changes["merged_vertices"] or before_tags != after_tags or not len(cleaned.cells[0].data):
        raise ValueError(f"Cleanup of '{resource.name}' could not preserve every physical surface. Remesh this part.")
    if inspect_resource_vertices(resource, cleaned) is not None:
        raise ValueError(f"Nearly coincident vertices remain in '{resource.name}' after cleanup. Remesh this part.")

    def topology(mesh):
        return analyze_exterior_mesh_topology(
            (
                MeshConfig(
                    name=resource.name,
                    file="",
                    scale_factor=resource.scale_to_m,
                    translation_m=resource.translation_m,
                    mesh_data=MeshData.from_meshio(mesh),
                ),
            ),
            symmetry=symmetry,
        )

    before, after = topology(original), topology(cleaned)
    if after.open_edge_count > before.open_edge_count or after.nonmanifold_edge_count > before.nonmanifold_edge_count:
        raise ValueError(f"Cleanup of '{resource.name}' would introduce invalid surface edges. Remesh this part.")
    return cleaned


def mesh_solve_failure_message(message: str) -> str:
    if "SingularException" in message or "singular matrix" in message.lower():
        return (
            "The solver could not solve the mesh equations (singular matrix). "
            "Nearly coincident vertices or degenerate triangles can cause this. "
            "If you continued past the mesh warning, try Auto-repair. If cleanup was already attempted, "
            "inspect or remesh the affected surfaces and check the physical-system assignments.\n\n"
            "Technical details:\n" + message
        )
    return message
