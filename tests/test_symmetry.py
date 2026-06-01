from pathlib import Path

import meshio
import numpy as np
import pytest

from blab.config import MeshConfig
from blab.symmetry import SymmetryValidationError, effective_symmetry_for_backend, validate_reduced_mesh_config


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
    assert effective_symmetry_for_backend("x", "julia_local") == "x"
    assert effective_symmetry_for_backend("xy", "local_julia") == "xy"


def test_effective_symmetry_for_backend_disables_unsupported_modes() -> None:
    assert effective_symmetry_for_backend("x", "local") == "off"
    assert effective_symmetry_for_backend("xy", "bempp_local") == "off"
    assert effective_symmetry_for_backend("off", "local") == "off"
