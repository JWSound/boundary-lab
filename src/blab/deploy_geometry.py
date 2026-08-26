"""Geometry checks shared by headless Boundary Lab Deploy solve preparation."""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurfaceDistanceResult:
    distance_m: float
    face_a: int
    face_b: int


@dataclass(frozen=True)
class SurfaceFacePair:
    """A directed-independent pair of surface faces within a distance limit."""

    distance_m: float
    face_a: int
    face_b: int


@dataclass(frozen=True)
class _BvhNode:
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    face_indices: np.ndarray
    left: "_BvhNode | None" = None
    right: "_BvhNode | None" = None

    @property
    def leaf(self) -> bool:
        return self.left is None and self.right is None


def transform_package_points(
    points_m: np.ndarray,
    *,
    position_x_m: float,
    position_height_m: float,
    position_z_m: float,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    yaw_deg: float,
) -> np.ndarray:
    """Apply the package-to-scene rotation, scene yaw, and translation."""

    points = np.asarray(points_m, dtype=np.float64)
    scene = np.column_stack((points[:, 0], -points[:, 2], points[:, 1]))
    roll = np.deg2rad(float(roll_deg))
    roll_cosine = np.cos(roll)
    roll_sine = np.sin(roll)
    rolled_x = roll_cosine * scene[:, 0] - roll_sine * scene[:, 1]
    rolled_y = roll_sine * scene[:, 0] + roll_cosine * scene[:, 1]
    pitch = np.deg2rad(float(pitch_deg))
    pitch_cosine = np.cos(pitch)
    pitch_sine = np.sin(pitch)
    pitched_y = pitch_cosine * rolled_y - pitch_sine * scene[:, 2]
    pitched_z = pitch_sine * rolled_y + pitch_cosine * scene[:, 2]
    yaw = np.deg2rad(float(yaw_deg))
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    rotated_x = cosine * rolled_x + sine * pitched_z
    rotated_z = -sine * rolled_x + cosine * pitched_z
    return np.column_stack(
        (
            rotated_x + float(position_x_m),
            pitched_y + float(position_height_m),
            rotated_z + float(position_z_m),
        )
    )


def _build_bvh(face_vertices: np.ndarray, face_indices: np.ndarray, leaf_size: int) -> _BvhNode:
    triangles = face_vertices[face_indices]
    bounds_min = np.min(triangles, axis=(0, 1))
    bounds_max = np.max(triangles, axis=(0, 1))
    if face_indices.size <= leaf_size:
        return _BvhNode(bounds_min, bounds_max, face_indices)
    centroids = np.mean(triangles, axis=1)
    axis = int(np.argmax(np.ptp(centroids, axis=0)))
    order = np.argsort(centroids[:, axis], kind="stable")
    midpoint = face_indices.size // 2
    if midpoint == 0 or midpoint == face_indices.size:
        return _BvhNode(bounds_min, bounds_max, face_indices)
    left = _build_bvh(face_vertices, face_indices[order[:midpoint]], leaf_size)
    right = _build_bvh(face_vertices, face_indices[order[midpoint:]], leaf_size)
    return _BvhNode(bounds_min, bounds_max, np.empty(0, dtype=np.int64), left, right)


def _aabb_distance_squared(first: _BvhNode, second: _BvhNode) -> float:
    separation = np.maximum(0.0, np.maximum(first.bounds_min - second.bounds_max, second.bounds_min - first.bounds_max))
    return float(np.dot(separation, separation))


def _point_segment_distance_squared(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator <= np.finfo(float).eps:
        delta = point - start
        return float(np.dot(delta, delta))
    parameter = float(np.clip(np.dot(point - start, segment) / denominator, 0.0, 1.0))
    delta = point - (start + parameter * segment)
    return float(np.dot(delta, delta))


def _point_triangle_distance_squared(point: np.ndarray, triangle: np.ndarray) -> float:
    first, second, third = triangle
    edge_a = second - first
    edge_b = third - first
    normal = np.cross(edge_a, edge_b)
    normal_squared = float(np.dot(normal, normal))
    if normal_squared > np.finfo(float).eps:
        signed_numerator = float(np.dot(point - first, normal))
        projection = point - (signed_numerator / normal_squared) * normal
        relative = projection - first
        dot_aa = float(np.dot(edge_a, edge_a))
        dot_ab = float(np.dot(edge_a, edge_b))
        dot_bb = float(np.dot(edge_b, edge_b))
        dot_ra = float(np.dot(relative, edge_a))
        dot_rb = float(np.dot(relative, edge_b))
        denominator = dot_aa * dot_bb - dot_ab * dot_ab
        if abs(denominator) > np.finfo(float).eps:
            coordinate_a = (dot_bb * dot_ra - dot_ab * dot_rb) / denominator
            coordinate_b = (dot_aa * dot_rb - dot_ab * dot_ra) / denominator
            if coordinate_a >= 0.0 and coordinate_b >= 0.0 and coordinate_a + coordinate_b <= 1.0:
                return signed_numerator * signed_numerator / normal_squared
    return min(
        _point_segment_distance_squared(point, first, second),
        _point_segment_distance_squared(point, second, third),
        _point_segment_distance_squared(point, third, first),
    )


def _segment_segment_distance_squared(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> float:
    first_direction = first_end - first_start
    second_direction = second_end - second_start
    relative = first_start - second_start
    aa = float(np.dot(first_direction, first_direction))
    bb = float(np.dot(first_direction, second_direction))
    cc = float(np.dot(second_direction, second_direction))
    dd = float(np.dot(first_direction, relative))
    ee = float(np.dot(second_direction, relative))
    epsilon = np.finfo(float).eps
    denominator = aa * cc - bb * bb
    first_parameter = 0.0 if denominator <= epsilon else float(np.clip((bb * ee - cc * dd) / denominator, 0.0, 1.0))
    second_parameter = (bb * first_parameter + ee) / cc if cc > epsilon else 0.0
    if second_parameter < 0.0:
        second_parameter = 0.0
        first_parameter = float(np.clip(-dd / aa, 0.0, 1.0)) if aa > epsilon else 0.0
    elif second_parameter > 1.0:
        second_parameter = 1.0
        first_parameter = float(np.clip((bb - dd) / aa, 0.0, 1.0)) if aa > epsilon else 0.0
    delta = (first_start + first_parameter * first_direction) - (second_start + second_parameter * second_direction)
    return float(np.dot(delta, delta))


def _segment_intersects_triangle(start: np.ndarray, end: np.ndarray, triangle: np.ndarray) -> bool:
    direction = end - start
    edge_a = triangle[1] - triangle[0]
    edge_b = triangle[2] - triangle[0]
    p_vector = np.cross(direction, edge_b)
    determinant = float(np.dot(edge_a, p_vector))
    epsilon = 1e-12
    if abs(determinant) <= epsilon:
        return False
    inverse = 1.0 / determinant
    t_vector = start - triangle[0]
    coordinate_a = float(np.dot(t_vector, p_vector)) * inverse
    if coordinate_a < -epsilon or coordinate_a > 1.0 + epsilon:
        return False
    q_vector = np.cross(t_vector, edge_a)
    coordinate_b = float(np.dot(direction, q_vector)) * inverse
    if coordinate_b < -epsilon or coordinate_a + coordinate_b > 1.0 + epsilon:
        return False
    parameter = float(np.dot(edge_b, q_vector)) * inverse
    return -epsilon <= parameter <= 1.0 + epsilon


def _triangle_distance_squared(first: np.ndarray, second: np.ndarray) -> float:
    edges = ((0, 1), (1, 2), (2, 0))
    if any(_segment_intersects_triangle(first[start], first[end], second) for start, end in edges):
        return 0.0
    if any(_segment_intersects_triangle(second[start], second[end], first) for start, end in edges):
        return 0.0
    best = min(
        *(_point_triangle_distance_squared(vertex, second) for vertex in first),
        *(_point_triangle_distance_squared(vertex, first) for vertex in second),
    )
    for first_start, first_end in edges:
        for second_start, second_end in edges:
            best = min(
                best,
                _segment_segment_distance_squared(
                    first[first_start],
                    first[first_end],
                    second[second_start],
                    second[second_end],
                ),
            )
    return best


def minimum_surface_distance(
    points_a: np.ndarray,
    triangles_a: np.ndarray,
    points_b: np.ndarray,
    triangles_b: np.ndarray,
    *,
    leaf_size: int = 8,
) -> SurfaceDistanceResult:
    """Return exact triangle-surface spacing using a paired AABB BVH traversal."""

    faces_a = np.asarray(points_a, dtype=np.float64)[np.asarray(triangles_a, dtype=np.int64)]
    faces_b = np.asarray(points_b, dtype=np.float64)[np.asarray(triangles_b, dtype=np.int64)]
    if faces_a.size == 0 or faces_b.size == 0:
        raise ValueError("Surface-distance checks require non-empty triangle meshes.")
    root_a = _build_bvh(faces_a, np.arange(faces_a.shape[0], dtype=np.int64), leaf_size)
    root_b = _build_bvh(faces_b, np.arange(faces_b.shape[0], dtype=np.int64), leaf_size)
    counter = itertools.count()
    queue: list[tuple[float, int, _BvhNode, _BvhNode]] = [
        (_aabb_distance_squared(root_a, root_b), next(counter), root_a, root_b)
    ]
    best_squared = float("inf")
    best_faces = (-1, -1)
    while queue:
        lower_bound, _, node_a, node_b = heapq.heappop(queue)
        if lower_bound >= best_squared:
            continue
        if node_a.leaf and node_b.leaf:
            for face_a in node_a.face_indices:
                for face_b in node_b.face_indices:
                    distance_squared = _triangle_distance_squared(faces_a[face_a], faces_b[face_b])
                    if distance_squared < best_squared:
                        best_squared = distance_squared
                        best_faces = (int(face_a), int(face_b))
            continue
        children_a = (node_a,) if node_a.leaf else (node_a.left, node_a.right)
        children_b = (node_b,) if node_b.leaf else (node_b.left, node_b.right)
        for child_a in children_a:
            for child_b in children_b:
                if child_a is None or child_b is None:
                    continue
                child_bound = _aabb_distance_squared(child_a, child_b)
                if child_bound < best_squared:
                    heapq.heappush(queue, (child_bound, next(counter), child_a, child_b))
    return SurfaceDistanceResult(float(np.sqrt(best_squared)), best_faces[0], best_faces[1])


def surface_face_pairs_within(
    points_a: np.ndarray,
    triangles_a: np.ndarray,
    points_b: np.ndarray,
    triangles_b: np.ndarray,
    max_distance_m: float,
    *,
    leaf_size: int = 8,
    exact: bool = True,
) -> list[SurfaceFacePair]:
    """Return every triangle pair whose exact surface distance is within a limit.

    The paired BVH traversal prunes node pairs farther apart than the inclusive
    threshold. When ``exact`` is false, face-AABB lower bounds are emitted as a
    conservative broad phase; this is appropriate for correction caches where
    a small number of false positives is cheaper than thousands of scalar
    triangle-distance evaluations. Results are deterministic and sorted by the
    two local face indices so the cache is reproducible.
    """

    maximum = float(max_distance_m)
    if not np.isfinite(maximum) or maximum < 0.0:
        raise ValueError("Surface-pair distance limit must be finite and non-negative.")
    faces_a = np.asarray(points_a, dtype=np.float64)[np.asarray(triangles_a, dtype=np.int64)]
    faces_b = np.asarray(points_b, dtype=np.float64)[np.asarray(triangles_b, dtype=np.int64)]
    if faces_a.size == 0 or faces_b.size == 0:
        raise ValueError("Surface-pair checks require non-empty triangle meshes.")
    root_a = _build_bvh(faces_a, np.arange(faces_a.shape[0], dtype=np.int64), leaf_size)
    root_b = _build_bvh(faces_b, np.arange(faces_b.shape[0], dtype=np.int64), leaf_size)
    maximum_squared = maximum * maximum
    stack: list[tuple[_BvhNode, _BvhNode]] = [(root_a, root_b)]
    pairs: list[SurfaceFacePair] = []
    while stack:
        node_a, node_b = stack.pop()
        if _aabb_distance_squared(node_a, node_b) > maximum_squared:
            continue
        if node_a.leaf and node_b.leaf:
            leaf_faces_a = faces_a[node_a.face_indices]
            leaf_faces_b = faces_b[node_b.face_indices]
            minimum_a = np.min(leaf_faces_a, axis=1)
            maximum_a = np.max(leaf_faces_a, axis=1)
            minimum_b = np.min(leaf_faces_b, axis=1)
            maximum_b = np.max(leaf_faces_b, axis=1)
            separation = np.maximum(
                0.0,
                np.maximum(
                    minimum_a[:, np.newaxis, :] - maximum_b[np.newaxis, :, :],
                    minimum_b[np.newaxis, :, :] - maximum_a[:, np.newaxis, :],
                ),
            )
            candidate_rows = np.argwhere(np.sum(separation * separation, axis=2) <= maximum_squared)
            for local_a, local_b in candidate_rows:
                face_a = int(node_a.face_indices[local_a])
                face_b = int(node_b.face_indices[local_b])
                aabb_distance_squared = float(np.dot(separation[local_a, local_b], separation[local_a, local_b]))
                if not exact:
                    pairs.append(SurfaceFacePair(float(np.sqrt(aabb_distance_squared)), face_a, face_b))
                    continue
                distance_squared = _triangle_distance_squared(faces_a[face_a], faces_b[face_b])
                if distance_squared <= maximum_squared:
                    pairs.append(
                        SurfaceFacePair(
                            float(np.sqrt(distance_squared)),
                            face_a,
                            face_b,
                        )
                    )
            continue
        children_a = (node_a,) if node_a.leaf else (node_a.left, node_a.right)
        children_b = (node_b,) if node_b.leaf else (node_b.left, node_b.right)
        for child_a in children_a:
            for child_b in children_b:
                if child_a is not None and child_b is not None:
                    stack.append((child_a, child_b))
    pairs.sort(key=lambda pair: (pair.face_a, pair.face_b))
    return pairs
