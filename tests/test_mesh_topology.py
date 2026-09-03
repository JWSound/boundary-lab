from pathlib import Path

import meshio
import numpy as np

from blab.config import MeshConfig
from blab.mesh_topology import analyze_exterior_mesh_topology, exterior_mesh_topology_warning_text


def _write_surface(path: Path, points: np.ndarray, triangles: np.ndarray) -> None:
    meshio.write(
        path,
        meshio.Mesh(
            points=np.asarray(points, dtype=float),
            cells=[("triangle", np.asarray(triangles, dtype=np.int64))],
        ),
        file_format="gmsh22",
        binary=False,
    )


def _tetrahedron_surface() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        np.array(
            [
                [0, 2, 1],
                [0, 1, 3],
                [1, 2, 3],
                [2, 0, 3],
            ]
        ),
    )


def test_closed_surface_passes_topology_check(tmp_path: Path) -> None:
    path = tmp_path / "closed.msh"
    _write_surface(path, *_tetrahedron_surface())

    report = analyze_exterior_mesh_topology((MeshConfig("closed", str(path), scale_factor=1.0),))

    assert report.has_warnings is False
    assert report.open_edge_count == 0
    assert report.nonmanifold_edge_count == 0


def test_open_surface_reports_edge_segments_in_transformed_metres(tmp_path: Path) -> None:
    path = tmp_path / "open.msh"
    _write_surface(
        path,
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        np.array([[0, 1, 2]]),
    )

    report = analyze_exterior_mesh_topology(
        (MeshConfig("open", str(path), scale_factor=0.5, translation_m=(1.0, 0.0, 0.0)),)
    )

    assert report.has_warnings is True
    assert report.open_edge_count == 3
    segments = report.meshes[0].open_edge_segments_m
    assert segments.shape == (3, 2, 3)
    assert float(np.min(segments[:, :, 0])) == 1.0
    assert float(np.max(segments[:, :, 0])) == 2.0


def test_nonmanifold_edge_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "nonmanifold.msh"
    _write_surface(
        path,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]]),
    )

    report = analyze_exterior_mesh_topology((MeshConfig("bad", str(path), scale_factor=1.0),))

    assert report.nonmanifold_edge_count == 1
    assert report.meshes[0].nonmanifold_edge_segments_m.shape == (1, 2, 3)


def test_reduced_surface_allows_open_edges_on_active_symmetry_plane(tmp_path: Path) -> None:
    path = tmp_path / "half.msh"
    points, triangles = _tetrahedron_surface()
    _write_surface(path, points, triangles[1:])

    report = analyze_exterior_mesh_topology(
        (MeshConfig("half", str(path), scale_factor=1.0),),
        symmetry="x",
    )

    assert report.has_warnings is False
    assert report.open_edge_count == 0
    assert report.symmetry_edge_count == 3


def test_separate_solver_meshes_are_not_welded_by_coordinate(tmp_path: Path) -> None:
    first = tmp_path / "first.msh"
    second = tmp_path / "second.msh"
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    _write_surface(first, points, np.array([[0, 1, 2]]))
    _write_surface(second, points, np.array([[0, 2, 1]]))

    report = analyze_exterior_mesh_topology(
        (
            MeshConfig("first", str(first), scale_factor=1.0),
            MeshConfig("second", str(second), scale_factor=1.0),
        )
    )

    assert report.open_edge_count == 6
    warning = exterior_mesh_topology_warning_text(report)
    assert "Open edges: 6" in warning
    assert "highlighted red in the mesh preview pane" in warning
