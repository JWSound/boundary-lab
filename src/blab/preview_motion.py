"""Qt-free motion assignments and sparse arrow placement for mesh previews."""

from dataclasses import dataclass

import numpy as np

from blab.physical_model import BoundaryKind, ComponentKind, PhysicalSystem


@dataclass(frozen=True)
class PreviewMotion:
    component_name: str
    axis: tuple[float, float, float] | None


def preview_motion_assignments(
    system: PhysicalSystem | None,
    surface_tags_by_mesh: dict[str, dict[str, int]],
) -> dict[tuple[str, int], PreviewMotion]:
    """Resolve the assembled system's saved axes, without guessing or solving."""
    if system is None:
        return {}
    meshes = {mesh.id: mesh for mesh in system.meshes}
    boundaries = {boundary.id: boundary for boundary in system.boundaries}
    assignments = {}
    for component in system.components:
        parameters = component.parameters
        axial = component.kind == ComponentKind.ELECTRODYNAMIC_TRANSDUCER or (
            component.kind == ComponentKind.IDEAL_VELOCITY_SOURCE
            and parameters.get("motion_profile") == "rigid_translation"
        )
        axis = None
        if axial:
            try:
                raw = np.asarray(parameters.get("motion_axis"), dtype=float)
                if raw.shape != (3,) or not np.isfinite(raw).all() or not np.any(raw):
                    continue
                raw = raw / np.max(np.abs(raw))
                axis = raw / np.linalg.norm(raw)
            except (ValueError, TypeError):
                continue
        for boundary_id in component.boundary_ids:
            boundary = boundaries.get(boundary_id)
            if boundary is None or boundary.kind != BoundaryKind.MOVING:
                continue
            mesh = meshes.get(boundary.group.mesh_id)
            if mesh is None:
                continue
            tag = boundary.group.tag
            if tag is None:
                tag = surface_tags_by_mesh.get(mesh.name, {}).get(boundary.group.name)
            if tag is None:
                continue
            signs = parameters.get("boundary_motion_signs", {})
            if not isinstance(signs, dict):
                continue
            sign = signs.get(boundary_id, 1) if component.kind == ComponentKind.ELECTRODYNAMIC_TRANSDUCER else 1
            if sign not in (-1, 1):
                continue
            assignments[(mesh.name, int(tag))] = PreviewMotion(
                component.name,
                None if axis is None else tuple(float(value * sign) for value in axis),
            )
    return assignments


def motion_arrow_geometry(points, triangles, axis, *, limit: int = 32):
    """Return spaced arrow origins/vectors; inward arrows end just above the face.

    Area-stratified candidates followed by farthest-point sampling avoid both
    dense-mesh clutter and large clusters on finely tessellated patches.
    """
    vertices = np.asarray(points, dtype=float)[np.asarray(triangles, dtype=np.int64)]
    cross = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    areas = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(vertices).all(axis=(1, 2)) & (areas > 0)
    vertices, cross, areas = vertices[valid], cross[valid], areas[valid]
    if not len(vertices):
        return np.empty((0, 3)), np.empty((0, 3))
    centers = vertices.mean(axis=1)
    extent = float(np.linalg.norm(np.ptp(vertices.reshape(-1, 3), axis=0)))
    length = extent * 0.08
    cumulative = np.cumsum(areas)
    count = min(len(areas), 2048)
    samples = (np.arange(count) + 0.5) * (cumulative[-1] / count)
    candidates = np.unique(np.searchsorted(cumulative, samples))
    chosen = [int(candidates[np.argmax(areas[candidates])])]
    distances = np.full(len(candidates), np.inf)
    for _ in range(min(limit, len(candidates)) - 1):
        distances = np.minimum(distances, np.sum((centers[candidates] - centers[chosen[-1]]) ** 2, axis=1))
        if distances.max() < length**2:
            break
        chosen.append(int(candidates[np.argmax(distances)]))
    normals = cross[chosen] / areas[chosen, None]
    direction = np.asarray(axis, dtype=float)
    direction /= np.linalg.norm(direction)
    vectors = np.tile(direction * length, (len(chosen), 1))
    origins = centers[chosen] + normals * length * 0.06
    origins -= (normals @ direction < 0)[:, None] * vectors
    return origins, vectors
