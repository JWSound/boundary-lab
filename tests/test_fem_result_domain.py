from __future__ import annotations

import meshio
import numpy as np
import pytest

from blab.solve_results.fem import _selected_tetrahedra


def _mesh(cell_blocks, physical_blocks) -> meshio.Mesh:
    return meshio.Mesh(
        points=np.zeros((10, 3), dtype=float),
        cells=cell_blocks,
        cell_data={"gmsh:physical": physical_blocks},
    )


def test_selected_tetrahedra_preserve_quadratic_nodes() -> None:
    cells = np.arange(10, dtype=np.int64).reshape(1, 10)
    mesh = _mesh([("tetra10", cells)], [np.asarray([7])])

    selected = _selected_tetrahedra(mesh, {7})

    np.testing.assert_array_equal(selected, cells)


def test_selected_tetrahedra_reject_mixed_orders() -> None:
    mesh = _mesh(
        [
            ("tetra", np.asarray([[0, 1, 2, 3]], dtype=np.int64)),
            ("tetra10", np.arange(10, dtype=np.int64).reshape(1, 10)),
        ],
        [np.asarray([7]), np.asarray([7])],
    )

    with pytest.raises(ValueError, match="mixes first- and second-order"):
        _selected_tetrahedra(mesh, {7})
