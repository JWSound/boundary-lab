"""
Clean surface .msh files from Ath4 for BEMPP workflows.

1) Mirrors Ath quadrant-14 half meshes across X=0 and preserves surface tags.
2) Merges coincident (or near-coincident) vertices within a tolerance.
3) Rebuilds triangle connectivity.
4) Removes collapsed and duplicate triangles.
5) Removes unused vertices.
6) Reports topology stats before/after (boundary/open edges, non-manifold edges, etc.).

Notes:
- This script targets triangle surface meshes (common for BEM boundary meshes).
- It preserves triangle physical tags (gmsh:physical) when present.
- If true geometric holes exist, they will remain open; this script only stitches seams
  caused by duplicated/near-coincident vertices.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import meshio
import numpy as np

from blab.defaults import EXAMPLE_CLEAN_MESH_PATH, EXAMPLE_MESH_PATH

MERGE_TOL = 1e-9
AREA_TOL = 0.0
WRITE_BINARY = False
MIRROR_X = True


@dataclass
class MeshStats:
    vertices: int
    triangles: int
    boundary_edges: int
    nonmanifold_edges: int
    duplicate_faces: int
    degenerate_faces: int
    components: int


@dataclass(frozen=True)
class StitchResult:
    stitched_loop_pairs: int
    seam_vertices: int
    split_boundary_edges: int
    split_triangles: int
    before: MeshStats
    after: MeshStats


@dataclass(frozen=True)
class MeshQualityWarning:
    sliver_triangles: int
    float32_singular_triangles: int
    worst_triangle_index: int
    worst_altitude_edge_ratio: float
    worst_area: float
    worst_longest_edge: float

    @property
    def has_warnings(self) -> bool:
        return self.sliver_triangles > 0 or self.float32_singular_triangles > 0


@dataclass(frozen=True)
class _StitchLoopCandidate:
    score: float
    mesh_a: int
    loop_a: int
    mesh_b: int
    loop_b: int


@dataclass(frozen=True)
class _StitchPath:
    vertices: List[int]
    closed: bool


def _find_triangle_block(mesh: meshio.Mesh) -> Tuple[str, np.ndarray]:
    cells_dict = mesh.cells_dict
    if "triangle" in cells_dict:
        return "triangle", np.asarray(cells_dict["triangle"], dtype=np.int64)
    if "triangle3" in cells_dict:
        return "triangle3", np.asarray(cells_dict["triangle3"], dtype=np.int64)
    raise ValueError("No triangle/triangle3 cell block found in mesh.")


def _extract_triangle_cell_data(mesh: meshio.Mesh, tri_key: str) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for data_name, by_cell_type in mesh.cell_data_dict.items():
        if tri_key in by_cell_type:
            out[data_name] = np.asarray(by_cell_type[tri_key])
    return out


def _edge_counts(triangles: np.ndarray) -> Dict[Tuple[int, int], int]:
    counts: Dict[Tuple[int, int], int] = {}
    for a, b, c in triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            if u > v:
                u, v = v, u
            key = (int(u), int(v))
            counts[key] = counts.get(key, 0) + 1
    return counts


def _boundary_edges(triangles: np.ndarray) -> np.ndarray:
    edges = np.asarray([edge for edge, count in _edge_counts(triangles).items() if count == 1], dtype=np.int64)
    if edges.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    return edges.reshape((-1, 2))


def _boundary_loops(triangles: np.ndarray) -> List[List[int]]:
    edges = _boundary_edges(triangles)
    if len(edges) == 0:
        return []

    adjacency: Dict[int, List[int]] = {}
    for a, b in edges:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))

    loops: List[List[int]] = []
    seen_edges: set[Tuple[int, int]] = set()
    for start in sorted(adjacency):
        for first_next in sorted(adjacency[start]):
            edge_key = tuple(sorted((start, first_next)))
            if edge_key in seen_edges:
                continue

            loop = [start]
            prev = start
            current = first_next
            seen_edges.add(edge_key)
            while current != start:
                loop.append(current)
                candidates = [node for node in adjacency[current] if node != prev]
                if not candidates:
                    break
                nxt = candidates[0]
                edge_key = tuple(sorted((current, nxt)))
                if edge_key in seen_edges and nxt != start:
                    break
                seen_edges.add(edge_key)
                prev, current = current, nxt

            if current == start and len(loop) >= 3:
                loops.append(loop)

    return loops


def _edge_on_ignored_boundary_plane(
    points: np.ndarray, edge: np.ndarray, axes: tuple[str, ...], tolerance: float
) -> bool:
    axis_indices = {"x": 0, "y": 1, "z": 2}
    edge_points = points[np.asarray(edge, dtype=np.int64)]
    return any(np.all(np.abs(edge_points[:, axis_indices[axis]]) <= tolerance) for axis in axes)


def _order_boundary_component(adjacency: Dict[int, List[int]], component: List[int]) -> _StitchPath | None:
    degrees = {vertex: len(adjacency[vertex]) for vertex in component}
    endpoints = sorted(vertex for vertex, degree in degrees.items() if degree == 1)
    if len(endpoints) == 0 and all(degree == 2 for degree in degrees.values()):
        start = min(component)
        closed = True
    elif len(endpoints) == 2 and all(degree in (1, 2) for degree in degrees.values()):
        start = endpoints[0]
        closed = False
    else:
        return None

    ordered = [start]
    previous: int | None = None
    current = start
    seen_edges: set[Tuple[int, int]] = set()
    while True:
        candidates = [
            vertex
            for vertex in sorted(adjacency[current])
            if vertex != previous and tuple(sorted((current, vertex))) not in seen_edges
        ]
        if not candidates:
            break
        nxt = candidates[0]
        seen_edges.add(tuple(sorted((current, nxt))))
        if closed and nxt == start:
            break
        ordered.append(nxt)
        previous, current = current, nxt

    min_vertices = 3 if closed else 2
    return _StitchPath(ordered, closed) if len(ordered) >= min_vertices else None


def _boundary_stitch_paths(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    ignored_boundary_axes: tuple[str, ...] = (),
    boundary_plane_tol: float = 1e-9,
) -> List[_StitchPath]:
    if not ignored_boundary_axes:
        return [_StitchPath(loop, True) for loop in _boundary_loops(triangles)]

    ignored_axes = _normalize_mirror_axes(ignored_boundary_axes)
    edges = [
        tuple(int(value) for value in edge)
        for edge in _boundary_edges(triangles)
        if not _edge_on_ignored_boundary_plane(points, edge, ignored_axes, boundary_plane_tol)
    ]
    if not edges:
        return []

    adjacency: Dict[int, List[int]] = {}
    for a, b in edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    paths: List[_StitchPath] = []
    seen: set[int] = set()
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: List[int] = []
        while stack:
            vertex = stack.pop()
            component.append(vertex)
            for neighbor in adjacency[vertex]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        path = _order_boundary_component(adjacency, component)
        if path is not None:
            paths.append(path)
    return paths


def _connected_components(triangles: np.ndarray) -> int:
    if len(triangles) == 0:
        return 0

    edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
    for face_index, (a, b, c) in enumerate(triangles):
        for u, v in ((a, b), (b, c), (c, a)):
            if u > v:
                u, v = v, u
            edge_to_faces.setdefault((int(u), int(v)), []).append(face_index)

    adjacency: List[set] = [set() for _ in range(len(triangles))]
    for face_ids in edge_to_faces.values():
        if len(face_ids) < 2:
            continue
        for i in range(len(face_ids)):
            for j in range(i + 1, len(face_ids)):
                f0 = face_ids[i]
                f1 = face_ids[j]
                adjacency[f0].add(f1)
                adjacency[f1].add(f0)

    seen = np.zeros(len(triangles), dtype=bool)
    components = 0
    for start in range(len(triangles)):
        if seen[start]:
            continue
        components += 1
        stack = [start]
        seen[start] = True
        while stack:
            node = stack.pop()
            for nxt in adjacency[node]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)

    return components


def _degenerate_mask(points: np.ndarray, triangles: np.ndarray, area_tol: float) -> np.ndarray:
    v0 = points[triangles[:, 0]]
    v1 = points[triangles[:, 1]]
    v2 = points[triangles[:, 2]]

    repeated_vertex = (
        (triangles[:, 0] == triangles[:, 1])
        | (triangles[:, 1] == triangles[:, 2])
        | (triangles[:, 0] == triangles[:, 2])
    )
    area2 = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)  # 2 * area
    tiny_area = area2 <= (2.0 * area_tol)
    return repeated_vertex | tiny_area


def triangle_quality_warning(
    mesh: meshio.Mesh,
    *,
    altitude_edge_ratio_tol: float = 1e-3,
) -> MeshQualityWarning:
    """Return a lightweight warning summary for triangles that are numerically too thin."""
    _tri_key, triangles = _find_triangle_block(mesh)
    if len(triangles) == 0:
        return MeshQualityWarning(0, 0, 0, float("inf"), 0.0, 0.0)

    points64 = np.asarray(mesh.points, dtype=np.float64)
    v0 = points64[triangles[:, 0]]
    v1 = points64[triangles[:, 1]]
    v2 = points64[triangles[:, 2]]

    e01 = v1 - v0
    e02 = v2 - v0
    e12 = v2 - v1
    area2 = np.linalg.norm(np.cross(e01, e02), axis=1)
    longest_edge = np.maximum.reduce(
        (
            np.linalg.norm(e01, axis=1),
            np.linalg.norm(e02, axis=1),
            np.linalg.norm(e12, axis=1),
        )
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        altitude_edge_ratio = area2 / np.square(longest_edge)

    finite_ratio = np.where(np.isfinite(altitude_edge_ratio), altitude_edge_ratio, 0.0)
    positive_area = area2 > 0.0

    points32 = np.asarray(mesh.points, dtype=np.float32)
    f0 = points32[triangles[:, 0]]
    f1 = points32[triangles[:, 1]]
    f2 = points32[triangles[:, 2]]
    fe1 = f1 - f0
    fe2 = f2 - f0
    g11 = np.einsum("ij,ij->i", fe1, fe1)
    g12 = np.einsum("ij,ij->i", fe1, fe2)
    g22 = np.einsum("ij,ij->i", fe2, fe2)
    gram_det32 = g11 * g22 - g12 * g12
    gram_scale32 = np.maximum(g11 * g22, np.finfo(np.float32).tiny)
    near_singular32 = gram_det32 <= (np.float32(10.0) * np.finfo(np.float32).eps * gram_scale32)

    sliver_mask = positive_area & (finite_ratio < altitude_edge_ratio_tol)
    singular32_mask = positive_area & near_singular32
    warning_mask = sliver_mask | singular32_mask
    if not np.any(warning_mask):
        return MeshQualityWarning(0, 0, 0, float(np.min(finite_ratio)), 0.0, 0.0)

    warning_indices = np.flatnonzero(warning_mask)
    worst_index = int(warning_indices[np.argmin(finite_ratio[warning_indices])])
    return MeshQualityWarning(
        sliver_triangles=int(np.sum(sliver_mask)),
        float32_singular_triangles=int(np.sum(singular32_mask)),
        worst_triangle_index=worst_index + 1,
        worst_altitude_edge_ratio=float(finite_ratio[worst_index]),
        worst_area=float(0.5 * area2[worst_index]),
        worst_longest_edge=float(longest_edge[worst_index]),
    )


def _mesh_stats(points: np.ndarray, triangles: np.ndarray, area_tol: float) -> MeshStats:
    deg_mask = _degenerate_mask(points, triangles, area_tol)

    sorted_faces = np.sort(triangles, axis=1)
    unique_faces = {tuple(row) for row in sorted_faces}
    duplicate_faces = len(sorted_faces) - len(unique_faces)

    edge_count = _edge_counts(triangles)
    boundary_edges = sum(1 for c in edge_count.values() if c == 1)
    nonmanifold_edges = sum(1 for c in edge_count.values() if c > 2)

    components = _connected_components(triangles)

    return MeshStats(
        vertices=len(points),
        triangles=len(triangles),
        boundary_edges=boundary_edges,
        nonmanifold_edges=nonmanifold_edges,
        duplicate_faces=duplicate_faces,
        degenerate_faces=int(np.sum(deg_mask)),
        components=components,
    )


def _spatial_hash_merge(points: np.ndarray, tol: float) -> np.ndarray:
    """
    Returns representative index for each original point.
    Points within tol are merged (transitively) via local grid-neighborhood checks.
    """
    if tol <= 0:
        return np.arange(len(points), dtype=np.int64)

    cell_size = tol
    inv = 1.0 / cell_size
    cell_coords = np.floor(points * inv).astype(np.int64)

    # Build cell -> point list
    grid: Dict[Tuple[int, int, int], List[int]] = {}
    for idx, c in enumerate(cell_coords):
        key = (int(c[0]), int(c[1]), int(c[2]))
        grid.setdefault(key, []).append(idx)

    parent = np.arange(len(points), dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    # Neighbor offsets in 3D grid
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]

    for key, idxs in grid.items():
        # same cell comparisons
        for i in range(len(idxs)):
            ii = idxs[i]
            pi = points[ii]
            for j in range(i + 1, len(idxs)):
                jj = idxs[j]
                if np.linalg.norm(pi - points[jj]) <= tol:
                    union(ii, jj)

        # neighbor cells comparisons (only forward keys to avoid duplicate work)
        kx, ky, kz = key
        for dx, dy, dz in offsets:
            nk = (kx + dx, ky + dy, kz + dz)
            if nk <= key:
                continue
            if nk not in grid:
                continue
            neigh = grid[nk]
            for ii in idxs:
                pi = points[ii]
                for jj in neigh:
                    if np.linalg.norm(pi - points[jj]) <= tol:
                        union(ii, jj)

    rep = np.array([find(i) for i in range(len(points))], dtype=np.int64)
    return rep


def _remove_duplicate_faces(
    triangles: np.ndarray, cell_data: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, Dict[str, np.ndarray], int]:
    seen: Dict[Tuple[int, int, int], int] = {}
    keep_indices: List[int] = []
    removed = 0

    for idx, tri in enumerate(triangles):
        key = tuple(sorted((int(tri[0]), int(tri[1]), int(tri[2]))))
        if key in seen:
            removed += 1
            continue
        seen[key] = idx
        keep_indices.append(idx)

    keep = np.asarray(keep_indices, dtype=np.int64)
    triangles_out = triangles[keep]
    cell_data_out = {name: arr[keep] for name, arr in cell_data.items()}
    return triangles_out, cell_data_out, removed


def _point_on_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> Tuple[float, float, np.ndarray]:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 0.0:
        return float(np.linalg.norm(point - start)), 0.0, start
    t = float(np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0))
    closest = start + t * segment
    return float(np.linalg.norm(point - closest)), t, closest


def _loop_lengths(points: np.ndarray, loop: Sequence[int]) -> Tuple[np.ndarray, float]:
    starts = points[np.asarray(loop, dtype=np.int64)]
    ends = points[np.asarray([*loop[1:], loop[0]], dtype=np.int64)]
    edge_lengths = np.linalg.norm(ends - starts, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(edge_lengths[:-1])))
    return cumulative, float(np.sum(edge_lengths))


def _path_lengths(points: np.ndarray, path: Sequence[int], closed: bool) -> Tuple[np.ndarray, float]:
    if closed:
        return _loop_lengths(points, path)
    if len(path) < 2:
        return np.asarray([0.0]), 0.0
    starts = points[np.asarray(path[:-1], dtype=np.int64)]
    ends = points[np.asarray(path[1:], dtype=np.int64)]
    edge_lengths = np.linalg.norm(ends - starts, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(edge_lengths)))
    return cumulative, float(np.sum(edge_lengths))


def _path_edge_count(path: Sequence[int], closed: bool) -> int:
    return len(path) if closed else max(len(path) - 1, 0)


def _path_edge_vertices(path: Sequence[int], edge_index: int, closed: bool) -> Tuple[int, int]:
    start = int(path[edge_index])
    end_index = (edge_index + 1) % len(path) if closed else edge_index + 1
    return start, int(path[end_index])


def _closest_point_on_loop(
    point: np.ndarray,
    points: np.ndarray,
    loop: Sequence[int],
) -> Tuple[float, int, float, np.ndarray]:
    return _closest_point_on_path(point, points, loop, True)


def _closest_point_on_path(
    point: np.ndarray,
    points: np.ndarray,
    path: Sequence[int],
    closed: bool,
) -> Tuple[float, int, float, np.ndarray]:
    best_distance = float("inf")
    best_edge_index = 0
    best_t = 0.0
    best_point = points[path[0]]
    for edge_index in range(_path_edge_count(path, closed)):
        start_vertex, end_vertex = _path_edge_vertices(path, edge_index, closed)
        distance, t, closest = _point_on_segment(point, points[start_vertex], points[end_vertex])
        if distance < best_distance:
            best_distance = distance
            best_edge_index = edge_index
            best_t = t
            best_point = closest
    return best_distance, best_edge_index, best_t, best_point


def _loop_distance(points: np.ndarray, source_loop: Sequence[int], target_loop: Sequence[int]) -> float:
    if not source_loop or not target_loop:
        return float("inf")
    return _path_distance(points, source_loop, True, target_loop, True)


def _path_distance(
    points: np.ndarray,
    source_path: Sequence[int],
    source_closed: bool,
    target_path: Sequence[int],
    target_closed: bool,
) -> float:
    if not source_path or not target_path:
        return float("inf")
    distances = [
        _closest_point_on_path(points[vertex], points, target_path, target_closed)[0] for vertex in source_path
    ]
    return float(max(distances))


def _choose_stitch_loop_pair(
    points: np.ndarray,
    loops_a: Sequence[Sequence[int]],
    loops_b: Sequence[Sequence[int]],
    stitch_tol: float,
) -> Tuple[List[int], List[int]]:
    best_pair: Tuple[List[int], List[int]] | None = None
    best_score = float("inf")
    for loop_a in loops_a:
        for loop_b in loops_b:
            distance_ab = _loop_distance(points, loop_a, loop_b)
            distance_ba = _loop_distance(points, loop_b, loop_a)
            score = max(distance_ab, distance_ba)
            if score <= stitch_tol and score < best_score:
                best_pair = (list(loop_a), list(loop_b))
                best_score = score

    if best_pair is None:
        raise ValueError(f"No compatible boundary loop pair found within stitch tolerance {stitch_tol:g}.")
    return best_pair


def _choose_stitch_loop_pairs(
    points: np.ndarray,
    loops_by_mesh: Sequence[Sequence[_StitchPath]],
    stitch_tol: float,
) -> List[Tuple[_StitchPath, _StitchPath]]:
    candidates: List[_StitchLoopCandidate] = []
    for mesh_a, loops_a in enumerate(loops_by_mesh[:-1]):
        for mesh_b in range(mesh_a + 1, len(loops_by_mesh)):
            loops_b = loops_by_mesh[mesh_b]
            for loop_a_index, loop_a in enumerate(loops_a):
                for loop_b_index, loop_b in enumerate(loops_b):
                    distance_ab = _path_distance(points, loop_a.vertices, loop_a.closed, loop_b.vertices, loop_b.closed)
                    distance_ba = _path_distance(points, loop_b.vertices, loop_b.closed, loop_a.vertices, loop_a.closed)
                    score = max(distance_ab, distance_ba)
                    if score <= stitch_tol:
                        candidates.append(
                            _StitchLoopCandidate(
                                score=score,
                                mesh_a=mesh_a,
                                loop_a=loop_a_index,
                                mesh_b=mesh_b,
                                loop_b=loop_b_index,
                            )
                        )

    stitched_pairs: List[Tuple[_StitchPath, _StitchPath]] = []
    used_loops: set[Tuple[int, int]] = set()
    for candidate in sorted(candidates, key=lambda item: item.score):
        loop_key_a = (candidate.mesh_a, candidate.loop_a)
        loop_key_b = (candidate.mesh_b, candidate.loop_b)
        if loop_key_a in used_loops or loop_key_b in used_loops:
            continue
        used_loops.add(loop_key_a)
        used_loops.add(loop_key_b)
        stitched_pairs.append(
            (
                loops_by_mesh[candidate.mesh_a][candidate.loop_a],
                loops_by_mesh[candidate.mesh_b][candidate.loop_b],
            )
        )

    if not stitched_pairs:
        raise ValueError(f"No compatible boundary loop pair found within stitch tolerance {stitch_tol:g}.")
    return stitched_pairs


def _seam_points_from_loops(
    points: np.ndarray,
    reference_loop: Sequence[int],
    other_loop: Sequence[int],
    stitch_tol: float,
) -> np.ndarray:
    return _seam_points_from_paths(points, reference_loop, True, other_loop, True, stitch_tol)


def _seam_points_from_paths(
    points: np.ndarray,
    reference_path: Sequence[int],
    reference_closed: bool,
    other_path: Sequence[int],
    other_closed: bool,
    stitch_tol: float,
) -> np.ndarray:
    cumulative, perimeter = _path_lengths(points, reference_path, reference_closed)
    if perimeter <= 0.0:
        raise ValueError("Cannot stitch a zero-length boundary path.")

    proposals: list[tuple[float, np.ndarray]] = []
    for edge_index, vertex in enumerate(reference_path):
        ref_point = points[vertex]
        distance, _other_edge, _other_t, other_point = _closest_point_on_path(
            ref_point,
            points,
            other_path,
            other_closed,
        )
        if distance > stitch_tol:
            raise ValueError("Reference boundary path is not fully within the stitch tolerance.")
        proposals.append((float(cumulative[edge_index]), 0.5 * (ref_point + other_point)))

    for vertex in other_path:
        other_point = points[vertex]
        distance, ref_edge, ref_t, ref_point = _closest_point_on_path(
            other_point,
            points,
            reference_path,
            reference_closed,
        )
        if distance > stitch_tol:
            raise ValueError("Other boundary path is not fully within the stitch tolerance.")
        start_vertex, end_vertex = _path_edge_vertices(reference_path, ref_edge, reference_closed)
        start = points[start_vertex]
        end = points[end_vertex]
        edge_length = float(np.linalg.norm(end - start))
        param = float(cumulative[ref_edge] + ref_t * edge_length)
        if reference_closed and param >= perimeter:
            param = 0.0
        proposals.append((param, 0.5 * (other_point + ref_point)))

    proposals.sort(key=lambda item: item[0])
    merged: list[tuple[float, list[np.ndarray]]] = []
    param_tol = max(perimeter * 1e-10, 1e-8)
    for param, point in proposals:
        if merged and abs(param - merged[-1][0]) <= param_tol:
            merged[-1][1].append(point)
        elif reference_closed and merged and abs(param - perimeter) <= param_tol and abs(merged[0][0]) <= param_tol:
            merged[0][1].append(point)
        else:
            merged.append((param, [point]))

    seam_points = np.asarray([np.mean(group, axis=0) for _param, group in merged], dtype=float)
    min_seam_points = 3 if reference_closed else 2
    if len(seam_points) < min_seam_points:
        raise ValueError("Stitch seam needs at least three vertices.")
    return seam_points


def _oriented_edge_and_opposite(triangle: np.ndarray, edge_key: Tuple[int, int]) -> Tuple[int, int, int] | None:
    a, b, c = (int(triangle[0]), int(triangle[1]), int(triangle[2]))
    oriented_edges = ((a, b, c), (b, c, a), (c, a, b))
    for start, end, opposite in oriented_edges:
        if tuple(sorted((start, end))) == edge_key:
            return start, end, opposite
    return None


def _split_stitched_loop_edges(
    points: np.ndarray,
    triangles: np.ndarray,
    cell_data: Dict[str, np.ndarray],
    loop: Sequence[int],
    seam_points: np.ndarray,
    seam_vertex_ids: np.ndarray,
    stitch_tol: float,
    closed: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], int, int]:
    seam_point_by_vertex_id = {int(vertex_id): seam_points[index] for index, vertex_id in enumerate(seam_vertex_ids)}
    loop_vertex_to_seam: Dict[int, int] = {}
    for vertex in loop:
        distances = np.linalg.norm(seam_points - points[int(vertex)], axis=1)
        seam_index = int(np.argmin(distances))
        if float(distances[seam_index]) > stitch_tol:
            raise ValueError("A boundary vertex could not be mapped onto the stitched seam.")
        loop_vertex_to_seam[int(vertex)] = int(seam_vertex_ids[seam_index])
    edge_to_seam: Dict[Tuple[int, int], List[Tuple[float, int]]] = {
        tuple(sorted(_path_edge_vertices(loop, i, closed))): [] for i in range(_path_edge_count(loop, closed))
    }
    for i in range(_path_edge_count(loop, closed)):
        start_vertex, end_vertex = _path_edge_vertices(loop, i, closed)
        edge_key = tuple(sorted((start_vertex, end_vertex)))
        edge_to_seam[edge_key].extend(
            [
                (0.0, loop_vertex_to_seam[start_vertex]),
                (1.0, loop_vertex_to_seam[end_vertex]),
            ]
        )

    for seam_index, seam_point in enumerate(seam_points):
        best_distance = float("inf")
        best_edge: Tuple[int, int] | None = None
        best_t = 0.0
        edge_distances: list[tuple[float, Tuple[int, int], float]] = []
        for i in range(_path_edge_count(loop, closed)):
            start_vertex, end_vertex = _path_edge_vertices(loop, i, closed)
            distance, t, _closest = _point_on_segment(seam_point, points[start_vertex], points[end_vertex])
            edge_key = tuple(sorted((start_vertex, end_vertex)))
            edge_distances.append((distance, edge_key, t))
            if distance < best_distance:
                best_distance = distance
                best_edge = edge_key
                best_t = t
        if best_edge is None or best_distance > stitch_tol:
            raise ValueError("A seam vertex could not be projected onto one of the stitched boundary loops.")
        projection_tol = max(1e-8, stitch_tol * 1e-6)
        matching_edges = [(edge_key, t) for distance, edge_key, t in edge_distances if distance <= projection_tol] or [
            (best_edge, best_t)
        ]
        for edge_key, t in matching_edges:
            edge_to_seam[edge_key].append((t, int(seam_vertex_ids[seam_index])))

    triangle_by_edge: Dict[Tuple[int, int], int] = {}
    for triangle_index, triangle in enumerate(triangles):
        for edge in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edge_key = tuple(sorted((int(edge[0]), int(edge[1]))))
            if edge_key in edge_to_seam:
                if edge_key in triangle_by_edge:
                    raise ValueError("Stitch candidate edge is not a boundary edge.")
                triangle_by_edge[edge_key] = triangle_index

    replacement_by_triangle: Dict[int, List[np.ndarray]] = {}
    split_edges = 0
    split_triangles = 0
    for edge_key, seam_items in edge_to_seam.items():
        triangle_index = triangle_by_edge.get(edge_key)
        if triangle_index is None:
            raise ValueError("Stitch boundary edge was not found in the mesh triangles.")
        oriented = _oriented_edge_and_opposite(triangles[triangle_index], edge_key)
        if oriented is None:
            raise ValueError("Could not orient stitch boundary edge.")
        start, end, opposite = oriented
        start_point = points[start]
        ordered = sorted(seam_items, key=lambda item: item[0])
        if np.linalg.norm(seam_point_by_vertex_id[ordered[0][1]] - start_point) > np.linalg.norm(
            seam_point_by_vertex_id[ordered[-1][1]] - start_point
        ):
            ordered = list(reversed(ordered))

        sequence = []
        for _t, vertex_id in ordered:
            if not sequence or sequence[-1] != vertex_id:
                sequence.append(vertex_id)
        if len(sequence) < 2:
            raise ValueError("Stitch boundary edge does not contain enough seam vertices.")
        replacement = [
            np.asarray([sequence[i], sequence[i + 1], opposite], dtype=np.int64) for i in range(len(sequence) - 1)
        ]
        replacement_by_triangle.setdefault(triangle_index, []).extend(replacement)
        split_edges += 1
        split_triangles += len(replacement)

    new_triangles: List[np.ndarray] = []
    source_indices: List[int] = []
    for triangle_index, triangle in enumerate(triangles):
        replacements = replacement_by_triangle.get(triangle_index)
        if replacements is None:
            remapped = np.asarray(
                [loop_vertex_to_seam.get(int(vertex), int(vertex)) for vertex in triangle], dtype=np.int64
            )
            new_triangles.append(remapped)
            source_indices.append(triangle_index)
            continue
        if len(replacements) == 0:
            continue
        new_triangles.extend(replacements)
        source_indices.extend([triangle_index] * len(replacements))

    triangles_out = np.asarray(new_triangles, dtype=np.int64)
    source_index_array = np.asarray(source_indices, dtype=np.int64)
    cell_data_out: Dict[str, np.ndarray] = {}
    for name, arr in cell_data.items():
        cell_data_out[name] = np.asarray(arr[source_index_array], dtype=arr.dtype)

    return triangles_out, cell_data_out, split_edges, split_triangles


def _compact_vertices(points: np.ndarray, triangles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    used = np.unique(triangles.ravel())
    new_index = -np.ones(len(points), dtype=np.int64)
    new_index[used] = np.arange(len(used), dtype=np.int64)

    points_compact = points[used]
    triangles_compact = new_index[triangles]
    return points_compact, triangles_compact


def _normalize_mirror_axes(axes: tuple[str, ...] | list[str] | str) -> tuple[str, ...]:
    if isinstance(axes, str):
        candidates = tuple(axes.lower())
    else:
        candidates = tuple(str(axis).strip().lower() for axis in axes)

    normalized = []
    for axis in candidates:
        if axis not in {"x", "y", "z"}:
            raise ValueError(f"Unsupported mirror axis: {axis!r}")
        if axis not in normalized:
            normalized.append(axis)
    return tuple(normalized)


def _mirror_across_axis(
    points: np.ndarray,
    triangles: np.ndarray,
    cell_data: Dict[str, np.ndarray],
    axis: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    mirrored_points = points.copy()
    mirrored_points[:, axis_index] *= -1.0

    point_offset = len(points)
    # Reflection reverses handedness, so swap two vertices to keep normals oriented consistently.
    mirrored_triangles = triangles[:, [0, 2, 1]] + point_offset

    points_out = np.vstack((points, mirrored_points))
    triangles_out = np.vstack((triangles, mirrored_triangles))
    cell_data_out = {name: np.concatenate((arr, arr.copy()), axis=0) for name, arr in cell_data.items()}
    return points_out, triangles_out, cell_data_out


def clean_mesh(
    mesh: meshio.Mesh,
    merge_tol: float,
    area_tol: float,
    mirror_x: bool = MIRROR_X,
    mirror_axes: tuple[str, ...] | None = None,
) -> Tuple[meshio.Mesh, Dict[str, int], MeshStats, MeshStats]:
    tri_key, triangles = _find_triangle_block(mesh)
    points = np.asarray(mesh.points, dtype=float)
    cell_data = _extract_triangle_cell_data(mesh, tri_key)

    stats_before = _mesh_stats(points, triangles, area_tol)

    if mirror_axes is None:
        mirror_axes = ("x",) if mirror_x else ()
    mirror_axes = _normalize_mirror_axes(mirror_axes)

    mirrored_vertices = 0
    mirrored_faces = 0
    for axis in mirror_axes:
        original_vertices = len(points)
        original_faces = len(triangles)
        points, triangles, cell_data = _mirror_across_axis(points, triangles, cell_data, axis)
        mirrored_vertices += original_vertices
        mirrored_faces += original_faces

    # 1) Merge near-coincident points
    rep = _spatial_hash_merge(points, merge_tol)
    unique_reps, inverse = np.unique(rep, return_inverse=True)
    points_merged = points[unique_reps]
    triangles_merged = inverse[triangles]

    merged_vertices = len(points) - len(points_merged)

    # 2) Remove degenerate faces
    deg_mask = _degenerate_mask(points_merged, triangles_merged, area_tol)
    keep = ~deg_mask
    triangles_clean = triangles_merged[keep]
    cell_data_clean = {name: arr[keep] for name, arr in cell_data.items()}
    removed_degenerate = int(np.sum(deg_mask))

    # 3) Remove duplicate faces
    triangles_clean, cell_data_clean, removed_duplicate = _remove_duplicate_faces(triangles_clean, cell_data_clean)

    # 4) Compact vertex list to used vertices only
    points_clean, triangles_clean = _compact_vertices(points_merged, triangles_clean)

    # Build output mesh preserving field_data and point_data where possible
    out_mesh = meshio.Mesh(
        points=points_clean,
        cells=[("triangle", triangles_clean)],
        point_data={},
        cell_data={name: [arr] for name, arr in cell_data_clean.items()},
        field_data=mesh.field_data,
    )

    stats_after = _mesh_stats(points_clean, triangles_clean, area_tol)

    changes = {
        "mirrored_vertices": int(mirrored_vertices),
        "mirrored_faces": int(mirrored_faces),
        "merged_vertices": int(merged_vertices),
        "removed_degenerate_faces": int(removed_degenerate),
        "removed_duplicate_faces": int(removed_duplicate),
        "removed_unused_vertices": int(len(points_merged) - len(points_clean)),
    }

    return out_mesh, changes, stats_before, stats_after


def _surface_field_data_by_tag(mesh: meshio.Mesh) -> Dict[int, str]:
    names_by_tag: Dict[int, str] = {}
    for name, value in mesh.field_data.items():
        try:
            tag = int(value[0])
            dimension = int(value[1])
        except (TypeError, ValueError, IndexError):
            continue
        if dimension == 2:
            names_by_tag[tag] = name
    return names_by_tag


def _unique_field_name(name: str, field_data: dict, mesh_index: int) -> str:
    if name not in field_data:
        return name
    candidate = f"{name}_mesh{mesh_index + 1}"
    suffix = 2
    while candidate in field_data:
        candidate = f"{name}_mesh{mesh_index + 1}_{suffix}"
        suffix += 1
    return candidate


def stitch_meshes(
    meshes: Sequence[meshio.Mesh],
    *,
    stitch_tol: float,
    area_tol: float = AREA_TOL,
    ignored_boundary_axes: tuple[str, ...] = (),
    boundary_plane_tol: float = 1e-9,
) -> Tuple[meshio.Mesh, StitchResult]:
    if len(meshes) < 2:
        raise ValueError("Boundary stitching requires at least two input meshes.")
    if stitch_tol <= 0:
        raise ValueError("stitch_tol must be greater than zero.")
    ignored_boundary_axes = _normalize_mirror_axes(ignored_boundary_axes)

    point_parts = []
    triangle_parts = []
    cell_data_names: set[str] = set()
    per_mesh_cell_data: list[Dict[str, np.ndarray]] = []
    triangles_by_mesh: list[np.ndarray] = []
    vertex_offset = 0
    field_data = {}
    used_surface_tags: set[int] = set()
    next_surface_tag = 1

    for mesh_index, mesh in enumerate(meshes):
        tri_key, triangles = _find_triangle_block(mesh)
        points = np.asarray(mesh.points, dtype=float)
        cell_data = _extract_triangle_cell_data(mesh, tri_key)
        if "gmsh:physical" in cell_data:
            names_by_tag = _surface_field_data_by_tag(mesh)
            tag_map: Dict[int, int] = {}
            for old_tag in sorted(int(tag) for tag in np.unique(cell_data["gmsh:physical"])):
                surface_name = names_by_tag.get(old_tag, f"mesh{mesh_index + 1}_surface_{old_tag}")
                surface_name = _unique_field_name(surface_name, field_data, mesh_index)
                if old_tag not in used_surface_tags:
                    new_tag = old_tag
                else:
                    while next_surface_tag in used_surface_tags:
                        next_surface_tag += 1
                    new_tag = next_surface_tag
                used_surface_tags.add(new_tag)
                field_data[surface_name] = np.array([new_tag, 2], dtype=np.int32)
                tag_map[old_tag] = new_tag
            cell_data["gmsh:physical"] = np.asarray(
                [tag_map[int(tag)] for tag in cell_data["gmsh:physical"]],
                dtype=cell_data["gmsh:physical"].dtype,
            )
        cell_data_names.update(cell_data)

        offset_triangles = triangles + vertex_offset
        point_parts.append(points)
        triangle_parts.append(offset_triangles)
        per_mesh_cell_data.append(cell_data)
        triangles_by_mesh.append(offset_triangles)
        vertex_offset += len(points)

    points = np.vstack(point_parts)
    triangles = np.vstack(triangle_parts)
    before = _mesh_stats(points, triangles, area_tol)
    loops_by_mesh = [
        _boundary_stitch_paths(
            points,
            mesh_triangles,
            ignored_boundary_axes=ignored_boundary_axes,
            boundary_plane_tol=boundary_plane_tol,
        )
        for mesh_triangles in triangles_by_mesh
    ]

    cell_data: Dict[str, np.ndarray] = {}
    for name in sorted(cell_data_names):
        values = []
        for mesh, mesh_cell_data in zip(meshes, per_mesh_cell_data):
            tri_key, mesh_triangles = _find_triangle_block(mesh)
            if name in mesh_cell_data:
                values.append(mesh_cell_data[name])
            else:
                values.append(np.zeros(len(mesh_triangles), dtype=np.int32))
        cell_data[name] = np.concatenate(values, axis=0)

    stitch_loop_pairs = _choose_stitch_loop_pairs(points, loops_by_mesh, stitch_tol)
    points_with_seam = points
    triangles_split = triangles
    cell_data_split = cell_data
    seam_vertex_count = 0
    split_boundary_edges = 0
    split_triangles = 0

    for path_a, path_b in stitch_loop_pairs:
        seam_points = _seam_points_from_paths(
            points_with_seam,
            path_a.vertices,
            path_a.closed,
            path_b.vertices,
            path_b.closed,
            stitch_tol,
        )
        seam_vertex_ids = np.arange(
            len(points_with_seam),
            len(points_with_seam) + len(seam_points),
            dtype=np.int64,
        )
        points_with_seam = np.vstack((points_with_seam, seam_points))

        triangles_split, cell_data_split, split_edges_a, split_triangles_a = _split_stitched_loop_edges(
            points_with_seam,
            triangles_split,
            cell_data_split,
            path_a.vertices,
            seam_points,
            seam_vertex_ids,
            stitch_tol,
            path_a.closed,
        )
        triangles_split, cell_data_split, split_edges_b, split_triangles_b = _split_stitched_loop_edges(
            points_with_seam,
            triangles_split,
            cell_data_split,
            path_b.vertices,
            seam_points,
            seam_vertex_ids,
            stitch_tol,
            path_b.closed,
        )
        seam_vertex_count += int(len(seam_points))
        split_boundary_edges += int(split_edges_a + split_edges_b)
        split_triangles += int(split_triangles_a + split_triangles_b)

    deg_mask = _degenerate_mask(points_with_seam, triangles_split, area_tol)
    triangles_clean = triangles_split[~deg_mask]
    cell_data_clean = {name: arr[~deg_mask] for name, arr in cell_data_split.items()}
    triangles_clean, cell_data_clean, _removed_duplicate = _remove_duplicate_faces(triangles_clean, cell_data_clean)
    points_clean, triangles_clean = _compact_vertices(points_with_seam, triangles_clean)

    out_mesh = meshio.Mesh(
        points=points_clean,
        cells=[("triangle", triangles_clean)],
        point_data={},
        cell_data={name: [arr] for name, arr in cell_data_clean.items()},
        field_data=field_data,
    )
    after = _mesh_stats(points_clean, triangles_clean, area_tol)
    result = StitchResult(
        stitched_loop_pairs=len(stitch_loop_pairs),
        seam_vertices=seam_vertex_count,
        split_boundary_edges=split_boundary_edges,
        split_triangles=split_triangles,
        before=before,
        after=after,
    )
    return out_mesh, result


def stitch_mesh_files(
    input_msh_a: str,
    input_msh_b: str,
    output_msh: str,
    *,
    stitch_tol: float,
    area_tol: float = AREA_TOL,
    binary: bool = WRITE_BINARY,
) -> StitchResult:
    stitched, result = stitch_meshes(
        (meshio.read(input_msh_a), meshio.read(input_msh_b)),
        stitch_tol=stitch_tol,
        area_tol=area_tol,
    )
    meshio.write(output_msh, stitched, file_format="gmsh22", binary=binary)
    return result


def clean_mesh_file(
    input_msh: str,
    output_msh: str,
    *,
    merge_tol: float = MERGE_TOL,
    area_tol: float = AREA_TOL,
    mirror_x: bool = MIRROR_X,
    mirror_axes: tuple[str, ...] | None = None,
    binary: bool = WRITE_BINARY,
) -> Tuple[Dict[str, int], MeshStats, MeshStats]:
    mesh = meshio.read(input_msh)
    cleaned, changes, before, after = clean_mesh(
        mesh,
        merge_tol=merge_tol,
        area_tol=area_tol,
        mirror_x=mirror_x,
        mirror_axes=mirror_axes,
    )
    meshio.write(output_msh, cleaned, file_format="gmsh22", binary=binary)
    return changes, before, after


def _print_stats(label: str, s: MeshStats) -> None:
    print(f"{label}")
    print(f"  vertices          : {s.vertices}")
    print(f"  triangles         : {s.triangles}")
    print(f"  boundary edges    : {s.boundary_edges}")
    print(f"  nonmanifold edges : {s.nonmanifold_edges}")
    print(f"  duplicate faces   : {s.duplicate_faces}")
    print(f"  degenerate faces  : {s.degenerate_faces}")
    print(f"  components        : {s.components}")


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    parser = argparse.ArgumentParser(prog=prog, description="Clean/stitch a triangle .msh surface mesh for BEM.")
    parser.add_argument("input_msh", nargs="?", default=EXAMPLE_MESH_PATH, help="Input .msh file")
    parser.add_argument("output_msh", nargs="?", default=EXAMPLE_CLEAN_MESH_PATH, help="Output cleaned .msh file")
    parser.add_argument(
        "--merge-tol",
        type=float,
        default=MERGE_TOL,
        help="Vertex merge tolerance in mesh units (default: 1e-9)",
    )
    parser.add_argument(
        "--area-tol",
        type=float,
        default=AREA_TOL,
        help="Area tolerance for removing tiny triangles in mesh units^2 (default: 0.0)",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        default=WRITE_BINARY,
        help="Write binary .msh (default is ASCII gmsh22 for compatibility)",
    )
    parser.add_argument(
        "--no-mirror-x",
        action="store_false",
        dest="mirror_x",
        default=MIRROR_X,
        help="Skip mirroring the input half mesh across X=0",
    )
    parser.add_argument(
        "--mirror-axes",
        default=None,
        help="Axes to mirror across, such as x or xy. Overrides --no-mirror-x when supplied.",
    )
    parser.add_argument(
        "--stitch-with",
        help="Second .msh file to boundary-stitch with input_msh instead of running single-mesh cleaning",
    )
    parser.add_argument(
        "--stitch-tol",
        type=float,
        default=None,
        help="Boundary stitch tolerance in mesh units; required with --stitch-with",
    )

    args = parser.parse_args(argv)

    if args.stitch_with:
        if args.stitch_tol is None:
            parser.error("--stitch-tol is required with --stitch-with")
        result = stitch_mesh_files(
            args.input_msh,
            args.stitch_with,
            args.output_msh,
            stitch_tol=args.stitch_tol,
            area_tol=args.area_tol,
            binary=args.binary,
        )

        _print_stats("Before:", result.before)
        print("Stitch:")
        print(f"  stitched_loop_pairs     : {result.stitched_loop_pairs}")
        print(f"  seam_vertices           : {result.seam_vertices}")
        print(f"  split_boundary_edges    : {result.split_boundary_edges}")
        print(f"  split_triangles         : {result.split_triangles}")
        _print_stats("After:", result.after)
        print(f"\nWrote stitched mesh: {args.output_msh}")

        if result.after.boundary_edges > 0:
            print("Warning: stitched mesh still has open edges. Check for unrelated holes or unmatched seams.")
        return

    changes, before, after = clean_mesh_file(
        args.input_msh,
        args.output_msh,
        merge_tol=args.merge_tol,
        area_tol=args.area_tol,
        mirror_x=args.mirror_x,
        mirror_axes=None if args.mirror_axes is None else _normalize_mirror_axes(args.mirror_axes),
        binary=args.binary,
    )

    _print_stats("Before:", before)
    print("Changes:")
    for k, v in changes.items():
        print(f"  {k:24s}: {v}")
    _print_stats("After:", after)

    print(f"\nWrote cleaned mesh: {args.output_msh}")

    if after.boundary_edges > 0:
        print("Warning: mesh still has open edges. This usually means real holes (not just unstitched seams).")


if __name__ == "__main__":
    main()
