"""Prepare Fibonacci-sampled spherical directivity data for balloon plotting."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.spatial import ConvexHull, cKDTree


@dataclass(frozen=True)
class BalloonPrepConfig:
    min_db: float = -30.0
    max_db: float = 0.0


def prepare_balloon_data(
    raw_data: dict[str, np.ndarray],
    cfg: BalloonPrepConfig | None = None,
) -> dict[str, np.ndarray]:
    """Build one frequency-invariant triangular surface from solved directions."""

    prep_cfg = cfg or BalloonPrepConfig()
    freq_hz = np.asarray(raw_data["freq_hz"], dtype=np.float32)
    theta_polar_rad = np.asarray(raw_data["theta_polar_rad"], dtype=np.float32)
    phi_azimuth_rad = np.asarray(raw_data["phi_azimuth_rad"], dtype=np.float32)
    surface_spl = np.clip(
        np.asarray(raw_data["spl_norm"], dtype=np.float32),
        prep_cfg.min_db,
        prep_cfg.max_db,
    )

    if freq_hz.ndim != 1:
        raise ValueError("freq_hz must be a 1D array.")
    if theta_polar_rad.ndim != 1 or phi_azimuth_rad.ndim != 1:
        raise ValueError("theta_polar_rad and phi_azimuth_rad must be 1D arrays.")
    if theta_polar_rad.shape != phi_azimuth_rad.shape:
        raise ValueError("theta_polar_rad and phi_azimuth_rad must have matching shapes.")
    if theta_polar_rad.size < 4:
        raise ValueError("Balloon surface triangulation requires at least four spherical samples.")
    if surface_spl.shape != (freq_hz.size, theta_polar_rad.size):
        raise ValueError("spl_norm must have shape (len(freq_hz), len(theta_polar_rad)).")
    if not np.all(np.isfinite(theta_polar_rad)) or not np.all(np.isfinite(phi_azimuth_rad)):
        raise ValueError("Balloon sample angles must be finite.")
    if not np.all(np.isfinite(surface_spl)):
        raise ValueError("Balloon SPL samples must be finite.")
    if not np.isfinite(prep_cfg.min_db) or not np.isfinite(prep_cfg.max_db):
        raise ValueError("Balloon dB limits must be finite.")
    if prep_cfg.max_db <= prep_cfg.min_db:
        raise ValueError("Balloon maximum dB must be greater than minimum dB.")

    directions = _direction_vectors(theta_polar_rad, phi_azimuth_rad)
    triangle_indices = _cached_sphere_triangulation(directions.tobytes(), directions.shape[0])
    prepared = {
        "freq_hz": freq_hz,
        "theta_polar_rad": theta_polar_rad,
        "phi_azimuth_rad": phi_azimuth_rad,
        "directions_xyz": directions,
        "triangle_indices": triangle_indices,
        "balloon_surface_spl": surface_spl,
        "min_db": np.asarray(prep_cfg.min_db, dtype=np.float32),
        "max_db": np.asarray(prep_cfg.max_db, dtype=np.float32),
    }
    if "r_distance_m" in raw_data:
        r_distance_m = np.asarray(raw_data["r_distance_m"], dtype=np.float32)
        if r_distance_m.shape != theta_polar_rad.shape:
            raise ValueError("r_distance_m must match the spherical sample shape.")
        prepared["r_distance_m"] = r_distance_m
    return prepared


def balloon_surface_points(
    directions_xyz: np.ndarray,
    spl_db: np.ndarray,
    min_db: float,
) -> np.ndarray:
    directions = np.asarray(directions_xyz, dtype=np.float32)
    values = np.asarray(spl_db, dtype=np.float32)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("directions_xyz must have shape (point, 3).")
    if values.shape != (directions.shape[0],):
        raise ValueError("spl_db must contain one value per direction.")
    radius = np.maximum(values - float(min_db), 0.0)
    return (directions * radius[:, np.newaxis]).astype(np.float32, copy=False)


class BalloonSurfaceSampler:
    """Interpolate derived queries over the displayed triangular surface."""

    def __init__(self, directions_xyz: np.ndarray, triangle_indices: np.ndarray):
        self.directions = _normalized_directions(directions_xyz)
        self.triangles = np.asarray(triangle_indices, dtype=np.int32)
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError("triangle_indices must have shape (triangle, 3).")
        if self.triangles.size == 0:
            raise ValueError("Balloon surface must contain at least one triangle.")
        if np.min(self.triangles) < 0 or np.max(self.triangles) >= self.directions.shape[0]:
            raise ValueError("triangle_indices references a point outside directions_xyz.")
        centers = np.mean(self.directions[self.triangles], axis=1)
        self._triangle_centers = _normalized_directions(centers)
        self._triangle_tree = cKDTree(self._triangle_centers)
        self._point_tree = cKDTree(self.directions)

    def interpolate(self, point_values: np.ndarray, query_directions: np.ndarray) -> np.ndarray:
        values = np.asarray(point_values)
        if values.shape[-1] != self.directions.shape[0]:
            raise ValueError("point_values must use the balloon point axis as its final dimension.")
        vertex_indices, weights = self.query_weights(query_directions)
        sampled = np.sum(values[..., vertex_indices] * weights, axis=-1)
        return np.asarray(sampled)

    def query_weights(self, query_directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        queries = _normalized_directions(query_directions)
        candidate_count = min(12, self.triangles.shape[0])
        _, candidate_faces = self._triangle_tree.query(queries, k=candidate_count)
        if candidate_count == 1:
            candidate_faces = np.asarray(candidate_faces)[:, np.newaxis]

        vertex_indices = np.empty((queries.shape[0], 3), dtype=np.int32)
        weights = np.zeros((queries.shape[0], 3), dtype=np.float64)
        for query_index, (query, candidates) in enumerate(zip(queries, candidate_faces)):
            match = self._containing_face(query, np.atleast_1d(candidates))
            if match is None:
                _, point_index = self._point_tree.query(query, k=1)
                vertex_indices[query_index] = (int(point_index), int(point_index), int(point_index))
                weights[query_index, 0] = 1.0
                continue
            face_index, face_weights = match
            vertex_indices[query_index] = self.triangles[face_index]
            weights[query_index] = face_weights
        return vertex_indices, weights

    def _containing_face(
        self,
        query: np.ndarray,
        candidate_faces: np.ndarray,
    ) -> tuple[int, np.ndarray] | None:
        best: tuple[int, np.ndarray] | None = None
        best_min_weight = -np.inf
        for raw_face_index in candidate_faces:
            face_index = int(raw_face_index)
            triangle = self.directions[self.triangles[face_index]]
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            denominator = float(np.dot(normal, query))
            if np.isclose(denominator, 0.0):
                continue
            distance = float(np.dot(normal, triangle[0]) / denominator)
            if distance <= 0.0:
                continue
            face_weights = _triangle_barycentric_weights(distance * query, triangle)
            if face_weights is None:
                continue
            minimum = float(np.min(face_weights))
            if minimum > best_min_weight:
                best = face_index, face_weights
                best_min_weight = minimum
            if minimum >= -1e-7:
                return face_index, np.clip(face_weights, 0.0, 1.0)
        if best is not None and best_min_weight >= -1e-4:
            face_index, face_weights = best
            clipped = np.clip(face_weights, 0.0, 1.0)
            return face_index, clipped / np.sum(clipped)
        return None


def _direction_vectors(theta_rad: np.ndarray, phi_rad: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta_rad, dtype=np.float64)
    phi = np.asarray(phi_rad, dtype=np.float64)
    directions = np.column_stack(
        (
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        )
    )
    return _normalized_directions(directions).astype(np.float32)


def _normalized_directions(directions_xyz: np.ndarray) -> np.ndarray:
    directions = np.asarray(directions_xyz, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("Spherical directions must have shape (point, 3).")
    norms = np.linalg.norm(directions, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("Spherical directions must be finite and nonzero.")
    return directions / norms[:, np.newaxis]


def _triangulate_sphere(directions_xyz: np.ndarray) -> np.ndarray:
    directions = _normalized_directions(directions_xyz)
    if np.unique(np.round(directions, decimals=10), axis=0).shape[0] != directions.shape[0]:
        raise ValueError("Balloon spherical directions must be unique.")
    hull = ConvexHull(directions)
    faces = np.asarray(hull.simplices, dtype=np.int32).copy()
    triangles = directions[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    centroids = np.mean(triangles, axis=1)
    inward = np.einsum("ij,ij->i", normals, centroids) < 0.0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    return faces


@lru_cache(maxsize=4)
def _cached_sphere_triangulation(directions_bytes: bytes, point_count: int) -> np.ndarray:
    directions = np.frombuffer(directions_bytes, dtype=np.float32).reshape(int(point_count), 3)
    faces = _triangulate_sphere(directions)
    faces.setflags(write=False)
    return faces


def _triangle_barycentric_weights(point: np.ndarray, triangle: np.ndarray) -> np.ndarray | None:
    edge0 = triangle[1] - triangle[0]
    edge1 = triangle[2] - triangle[0]
    relative = point - triangle[0]
    dot00 = float(np.dot(edge0, edge0))
    dot01 = float(np.dot(edge0, edge1))
    dot11 = float(np.dot(edge1, edge1))
    dot20 = float(np.dot(relative, edge0))
    dot21 = float(np.dot(relative, edge1))
    denominator = dot00 * dot11 - dot01 * dot01
    if np.isclose(denominator, 0.0):
        return None
    weight1 = (dot11 * dot20 - dot01 * dot21) / denominator
    weight2 = (dot00 * dot21 - dot01 * dot20) / denominator
    return np.asarray((1.0 - weight1 - weight2, weight1, weight2), dtype=np.float64)


__all__ = [
    "BalloonPrepConfig",
    "BalloonSurfaceSampler",
    "balloon_surface_points",
    "prepare_balloon_data",
]
