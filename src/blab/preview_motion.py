"""Qt-free motion assignments and representative arrow placement for mesh previews."""

from dataclasses import dataclass

import numpy as np

from blab.physical_model import BoundaryKind, ComponentKind, PhysicalSystem


@dataclass(frozen=True)
class PreviewMotion:
    component_id: str
    component_name: str
    axis: tuple[float, float, float] | None


def preview_motion_assignments(
    system: PhysicalSystem | None,
    surface_tags_by_mesh: dict[str, dict[str, int]],
    *,
    driven_surfaces: set[tuple[str, int]] | None = None,
) -> dict[tuple[str, int], PreviewMotion]:
    """Resolve the assembled system's saved axes, without guessing or solving."""
    if system is None:
        return {
            (mesh, tag): PreviewMotion(f"legacy:{mesh}:{tag}", f"{mesh}: Tag {tag}", None)
            for mesh, tag in sorted(driven_surfaces or ())
        }
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
                component.id,
                component.name,
                None if axis is None else tuple(float(value * sign) for value in axis),
            )
    return assignments


def motion_arrow_anchor(points, triangles):
    """Choose the face centroid nearest the patch's area-weighted center."""
    vertices = np.asarray(points, dtype=float)[np.asarray(triangles, dtype=np.int64)]
    cross = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    areas = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(vertices).all(axis=(1, 2)) & (areas > 0)
    vertices, cross, areas = vertices[valid], cross[valid], areas[valid]
    if not len(vertices):
        return None
    centers = vertices.mean(axis=1)
    center = np.average(centers, axis=0, weights=areas)
    index = int(np.argmin(np.sum((centers - center) ** 2, axis=1)))
    length = float(np.linalg.norm(np.ptp(vertices.reshape(-1, 3), axis=0))) * 0.12
    return centers[index], cross[index] / areas[index], length


def motion_arrow_geometry(anchor, axis, camera_position=None):
    """Place one arrow above the camera-facing side, independent of winding.

    An absent axis represents a saved surface-normal source at the chosen face.
    Only the display offset changes with the camera; the motion vector does not.
    """
    center, normal, length = anchor
    direction = np.array(normal if axis is None else axis, dtype=float, copy=True)
    direction /= np.linalg.norm(direction)
    facing = normal.copy()
    if camera_position is not None and np.dot(facing, np.asarray(camera_position) - center) < 0:
        facing *= -1
    vector = direction * length
    offset = length * 0.06 + max(0.0, -float(np.dot(vector, facing)))
    origin = center + facing * offset
    return origin[None, :], vector[None, :]
