"""Read-only source-patch measurements, in metres, independent of Qt."""

from dataclasses import dataclass

import meshio
import numpy as np

from blab.interface_conform import (
    InterfaceConformError,
    _physical_surface_tag,
    _surface_boundary_edges,
    _triangle_data,
)


@dataclass(frozen=True)
class InterfacePatch:
    points: np.ndarray
    triangles: np.ndarray
    perimeter_segments: np.ndarray
    area_m2: float
    edge_lengths_m: np.ndarray
    plane_origin: np.ndarray
    plane_normal: np.ndarray
    planarity_m: float


@dataclass(frozen=True)
class InterfaceGeometry:
    fem: InterfacePatch
    bem: InterfacePatch
    plane_angle_deg: float
    centroid_normal_offset_m: float
    # Symmetric vertex-to-segment sampling, not the continuous Hausdorff distance.
    sampled_perimeter_deviation_m: float | None
    discrepancy_segment: np.ndarray | None

    def summary(self) -> str:
        lines = ["Source-surface measurements"]
        if self.sampled_perimeter_deviation_m is None:
            lines.append("Sampled perimeter deviation: unavailable (a surface has no perimeter).")
        else:
            lines.append(f"Maximum sampled perimeter deviation: {self.sampled_perimeter_deviation_m * 1000:.6g} mm")
        lines.extend(
            (
                f"Centroid offset along FEM fitted normal: {self.centroid_normal_offset_m * 1000:.6g} mm",
                f"Fitted-plane angle (ignoring normal direction): {self.plane_angle_deg:.6g}°",
                f"BEM/FEM area ratio: {self.bem.area_m2 / self.fem.area_m2:.6g}",
            )
        )
        for label, patch in (("FEM", self.fem), ("BEM", self.bem)):
            sizes = (
                np.asarray([patch.edge_lengths_m.min(), np.median(patch.edge_lengths_m), patch.edge_lengths_m.max()])
                * 1000
            )
            lines.extend(
                (
                    f"{label}: {len(patch.triangles)} triangles; area {patch.area_m2 * 1e6:.6g} mm²",
                    f"  Triangle edge lengths (min / median / max): "
                    f"{sizes[0]:.6g} / {sizes[1]:.6g} / {sizes[2]:.6g} mm",
                    f"  Maximum distance from fitted plane: {patch.planarity_m * 1000:.6g} mm",
                    f"  Perimeter edges: {len(patch.perimeter_segments)}",
                )
            )
        lines.append(
            "Perimeter deviation samples vertices against opposite perimeter segments in both directions. "
            "Plane measurements describe geometry; they are not construction pass/fail checks. "
            "Fitted planes on curved patches may not describe their interface alignment."
        )
        return "\n".join(lines)


def _patch(mesh: meshio.Mesh, name: str) -> InterfacePatch:
    data = _triangle_data(mesh, require_geometrical=False)
    faces = data.triangles[data.physical_tags == _physical_surface_tag(mesh, name)]
    if not len(faces):
        raise InterfaceConformError(f"Physical surface '{name}' contains no triangles.")
    indices, inverse = np.unique(faces, return_inverse=True)
    points = np.asarray(mesh.points[indices, :3], dtype=float)
    if not np.isfinite(points).all():
        raise InterfaceConformError(f"Physical surface '{name}' has nonfinite coordinates.")
    triangles = inverse.reshape((-1, 3))
    coords = points[triangles]
    area = np.linalg.norm(np.cross(coords[:, 1] - coords[:, 0], coords[:, 2] - coords[:, 0]), axis=1) / 2
    if np.any(area <= 0):
        raise InterfaceConformError(f"Physical surface '{name}' has degenerate triangles.")
    origin = np.average(coords.mean(axis=1), weights=area, axis=0)
    # Integrate the second moment of each P1 triangle exactly. This avoids a
    # tessellation-density-dependent plane fit for the same geometric patch.
    centered = coords - origin
    sums = centered.sum(axis=1)
    covariance = np.einsum("t,tki,tkj->ij", area / 12, centered, centered)
    covariance += np.einsum("t,ti,tj->ij", area / 12, sums, sums)
    _, vectors = np.linalg.eigh(covariance)
    normal = vectors[:, 0]
    lengths = np.linalg.norm(coords - np.roll(coords, -1, axis=1), axis=2).ravel()
    edges = _surface_boundary_edges(triangles)
    return InterfacePatch(
        points,
        triangles,
        points[edges],
        float(area.sum()),
        lengths,
        origin,
        normal,
        float(np.max(np.abs((points - origin) @ normal))),
    )


def _furthest_sample(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    """Furthest source endpoint from its closest point on the target segments."""
    vertices = np.unique(source.reshape((-1, 3)), axis=0)
    start = target[:, 0]
    direction = target[:, 1] - start
    squared_length = np.einsum("ij,ij->i", direction, direction)
    best_squared = -1.0
    witness = None
    # Keep memory bounded even for finely meshed perimeters.
    for point in vertices:
        t = np.clip(((point - start) * direction).sum(axis=1) / squared_length, 0, 1)
        closest = start + t[:, None] * direction
        squared = ((closest - point) ** 2).sum(axis=1)
        index = int(np.argmin(squared))
        if squared[index] > best_squared:
            best_squared = float(squared[index])
            witness = np.array([point, closest[index]])
    return float(np.sqrt(best_squared)), witness


def measure_interface_geometry(
    fem_mesh: meshio.Mesh,
    bem_mesh: meshio.Mesh,
    *,
    fem_interface_name: str,
    bem_interface_name: str,
) -> InterfaceGeometry:
    """Measure patches as supplied; never stitch, project, snap, or alter them."""
    fem = _patch(fem_mesh, fem_interface_name)
    bem = _patch(bem_mesh, bem_interface_name)
    distance = witness = None
    if len(fem.perimeter_segments) and len(bem.perimeter_segments):
        forward = _furthest_sample(fem.perimeter_segments, bem.perimeter_segments)
        backward = _furthest_sample(bem.perimeter_segments, fem.perimeter_segments)
        distance, witness = max((forward, backward), key=lambda result: result[0])
    return InterfaceGeometry(
        fem,
        bem,
        float(np.degrees(np.arccos(np.clip(abs(fem.plane_normal @ bem.plane_normal), 0, 1)))),
        float(abs((bem.plane_origin - fem.plane_origin) @ fem.plane_normal)),
        distance,
        witness,
    )
