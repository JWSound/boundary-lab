from pathlib import Path

import meshio
import numpy as np
import pytest

from blab.config import MeshConfig
from blab.symmetry import (
    SymmetryValidationError,
    effective_symmetry_for_backend,
    snap_points_to_symmetry_planes,
    symmetry_plane_tolerance_m,
    validate_reduced_mesh_config,
)


def _write_triangle_mesh(path: Path, points: list[list[float]]) -> None:
    mesh = meshio.Mesh(
        points=np.asarray(points, dtype=float),
        cells=[("triangle", np.array([[0, 1, 2]], dtype=np.int64))],
    )
    meshio.write(path, mesh, file_format="gmsh22", binary=False)


def test_x_symmetry_accepts_positive_x_reduced_mesh(tmp_path: Path) -> None:
    mesh_path = tmp_path / "positive_x.msh"
    _write_triangle_mesh(
        mesh_path,
        [
            [0.0, -1.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
    )

    validate_reduced_mesh_config(MeshConfig(name="mesh", file=str(mesh_path), scale_factor=1.0), "x")


def test_x_symmetry_rejects_negative_x_vertices_after_transform(tmp_path: Path) -> None:
    mesh_path = tmp_path / "negative_x.msh"
    _write_triangle_mesh(
        mesh_path,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )
    mesh_config = MeshConfig(name="shifted", file=str(mesh_path), scale_factor=1.0, translation_m=(-0.1, 0.0, 0.0))

    with pytest.raises(SymmetryValidationError, match="positive X fundamental domain"):
        validate_reduced_mesh_config(mesh_config, "x")


def test_x_symmetry_accepts_scaled_single_precision_plane_noise(tmp_path: Path) -> None:
    mesh_path = tmp_path / "rounded_plane.msh"
    _write_triangle_mesh(
        mesh_path,
        [
            [-1.22e-5, 0.0, -800.0],
            [100.0, 0.0, -800.0],
            [0.0, 100.0, -800.0],
        ],
    )

    validate_reduced_mesh_config(MeshConfig(name="mesh", file=str(mesh_path), scale_factor=0.001), "x")


def test_x_symmetry_accepts_sawmod_scale_plane_noise(tmp_path: Path) -> None:
    mesh_path = tmp_path / "sawmod_rounded_plane.msh"
    _write_triangle_mesh(
        mesh_path,
        [
            [-8.515e-5, 12.7, 1.2],
            [318.0, 0.0, -274.0],
            [0.0, 145.5, 125.0],
        ],
    )

    validate_reduced_mesh_config(MeshConfig(name="mesh", file=str(mesh_path), scale_factor=0.001), "x")


def test_symmetry_plane_snapping_is_exact_and_limited_to_active_axes() -> None:
    points = np.asarray(
        [
            [-1.2e-8, -1.2e-8, 0.8],
            [2.0e-8, 2.0e-8, -0.8],
            [2.0e-5, -2.0e-5, 0.0],
        ]
    )

    tolerance = symmetry_plane_tolerance_m(points)
    snapped = snap_points_to_symmetry_planes(points, "x")

    assert tolerance == pytest.approx(1.6e-6)
    assert np.array_equal(snapped[:, 0], np.asarray([0.0, 0.0, 2.0e-5]))
    assert np.array_equal(snapped[:, 1], points[:, 1])
    assert np.array_equal(points[0], np.asarray([-1.2e-8, -1.2e-8, 0.8]))


def test_xy_symmetry_rejects_negative_y_vertices(tmp_path: Path) -> None:
    mesh_path = tmp_path / "negative_y.msh"
    _write_triangle_mesh(
        mesh_path,
        [
            [0.0, 0.0, 0.0],
            [1.0, -0.5, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )

    with pytest.raises(SymmetryValidationError, match="positive Y fundamental domain"):
        validate_reduced_mesh_config(MeshConfig(name="mesh", file=str(mesh_path), scale_factor=1.0), "xy")


def test_off_symmetry_skips_side_validation(tmp_path: Path) -> None:
    mesh_path = tmp_path / "full.msh"
    _write_triangle_mesh(
        mesh_path,
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )

    validate_reduced_mesh_config(MeshConfig(name="mesh", file=str(mesh_path), scale_factor=1.0), "off")


def test_effective_symmetry_for_backend_preserves_julia_modes() -> None:
    assert effective_symmetry_for_backend("x", "beat_cuda") == "x"
    assert effective_symmetry_for_backend("x", "beat_cpu") == "x"
    assert effective_symmetry_for_backend("xy", "beat_cpu") == "xy"
    assert effective_symmetry_for_backend("xy", "local_julia") == "xy"


def test_effective_symmetry_for_backend_disables_unsupported_modes() -> None:
    assert effective_symmetry_for_backend("x", "local") == "off"
    assert effective_symmetry_for_backend("xy", "bempp_local") == "off"
    assert effective_symmetry_for_backend("off", "local") == "off"
