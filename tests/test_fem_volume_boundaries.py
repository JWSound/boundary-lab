from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import meshio
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QComboBox

from blab.fem_topology import selected_volume_surface_tags
from blab.physical_compiler import PhysicalModelCompileError, PhysicalSystemCompiler
from blab.physical_model import Boundary, BoundaryKind, PhysicalGroupRef
from blab.ui.dialogs import MeshDialogEntry
from blab.ui.system_config import SystemConfigDialog, inspect_system_meshes

_APP = QApplication.instance() or QApplication([])


def _two_volume_mesh() -> meshio.Mesh:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [3.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    triangles = np.asarray(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
            [4, 5, 6],
            [4, 5, 7],
            [4, 6, 7],
            [5, 6, 7],
        ],
        dtype=np.int32,
    )
    tetrahedra = np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    triangle_tags = np.asarray([1, 3, 7, 7, 2, 4, 7, 7], dtype=np.int32)
    volume_tags = np.asarray([1, 2], dtype=np.int32)
    return meshio.Mesh(
        points=points,
        cells=[("triangle", triangles), ("tetra", tetrahedra)],
        cell_data={
            "gmsh:physical": [triangle_tags, volume_tags],
            "gmsh:geometrical": [triangle_tags, volume_tags],
        },
        field_data={
            "source_A": np.asarray([1, 2], dtype=np.int32),
            "source_B": np.asarray([2, 2], dtype=np.int32),
            "exit_A": np.asarray([3, 2], dtype=np.int32),
            "exit_B": np.asarray([4, 2], dtype=np.int32),
            "walls": np.asarray([7, 2], dtype=np.int32),
            "volume_A": np.asarray([1, 3], dtype=np.int32),
            "volume_B": np.asarray([2, 3], dtype=np.int32),
        },
    )


def _write_two_volume_mesh(path: Path) -> None:
    meshio.write(path, _two_volume_mesh(), file_format="gmsh22", binary=False)


def _boundary_names_by_region(dialog: SystemConfigDialog) -> dict[str, set[str]]:
    names: dict[str, set[str]] = {}
    for row in range(dialog.boundaries_table.rowCount()):
        region_name = dialog.boundaries_table.item(row, 0).text()
        names.setdefault(region_name, set()).add(dialog.boundaries_table.item(row, 2).text())
    return names


def test_selected_volume_surface_tags_follow_tetrahedron_adjacency() -> None:
    mesh = _two_volume_mesh()

    assert selected_volume_surface_tags(mesh, {1}) == {1, 3, 7}
    assert selected_volume_surface_tags(mesh, {2}) == {2, 4, 7}
    assert selected_volume_surface_tags(mesh, {1, 2}) == {1, 2, 3, 4, 7}


def test_selected_volume_surface_tags_accept_quadratic_elements() -> None:
    linear = _two_volume_mesh()
    triangles = np.asarray(linear.cells[0].data, dtype=np.int32)
    tetrahedra = np.asarray(linear.cells[1].data, dtype=np.int32)
    quadratic = meshio.Mesh(
        points=linear.points,
        cells=[
            ("triangle6", np.column_stack((triangles, triangles))),
            ("tetra10", np.column_stack((tetrahedra, tetrahedra, tetrahedra[:, :2]))),
        ],
        cell_data=linear.cell_data,
        field_data=linear.field_data,
    )

    assert selected_volume_surface_tags(quadratic, {1}) == {1, 3, 7}
    assert selected_volume_surface_tags(quadratic, {2}) == {2, 4, 7}


def test_system_dialog_filters_surfaces_for_each_volume_group(tmp_path: Path) -> None:
    mesh_path = tmp_path / "two-volume.msh"
    _write_two_volume_mesh(mesh_path)
    (available,) = inspect_system_meshes(
        (MeshDialogEntry(name="Channels", source_file=str(mesh_path), scale_factor=1.0),)
    )
    dialog = SystemConfigDialog((available,), None, ("main",))

    assert available.surface_groups_for_volume("volume_A") == ("exit_A", "source_A", "walls")
    assert _boundary_names_by_region(dialog) == {"Interior Air 1": {"exit_A", "source_A", "walls"}}

    dialog._add_default_region()
    volume_combo = dialog.regions_table.cellWidget(1, 3)
    assert isinstance(volume_combo, QComboBox)
    volume_combo.setCurrentIndex(volume_combo.findData("volume_B"))
    dialog._refresh_boundaries()

    assert _boundary_names_by_region(dialog) == {
        "Interior Air 1": {"exit_A", "source_A", "walls"},
        "Interior Air 2": {"exit_B", "source_B", "walls"},
    }
    PhysicalSystemCompiler().compile(dialog.physical_system())


def test_compiler_rejects_surface_assignment_outside_selected_volume(tmp_path: Path) -> None:
    mesh_path = tmp_path / "two-volume.msh"
    _write_two_volume_mesh(mesh_path)
    (available,) = inspect_system_meshes(
        (MeshDialogEntry(name="Channels", source_file=str(mesh_path), scale_factor=1.0),)
    )
    dialog = SystemConfigDialog((available,), None, ("main",))
    system = dialog.physical_system()
    region = system.regions[0]
    resource = system.meshes[0]
    invalid = replace(
        system,
        boundaries=(
            *system.boundaries,
            Boundary(
                id="boundary:wrong-volume",
                name="Source B on volume A",
                region_id=region.id,
                group=PhysicalGroupRef(mesh_id=resource.id, dimension=2, name="source_B"),
                kind=BoundaryKind.RIGID,
            ),
        ),
    )

    with pytest.raises(PhysicalModelCompileError, match="outside its selected volume groups.*source_B"):
        PhysicalSystemCompiler().compile(invalid)


def test_dialog_moves_existing_misassigned_boundary_to_its_adjacent_region(tmp_path: Path) -> None:
    mesh_path = tmp_path / "two-volume.msh"
    _write_two_volume_mesh(mesh_path)
    (available,) = inspect_system_meshes(
        (MeshDialogEntry(name="Channels", source_file=str(mesh_path), scale_factor=1.0),)
    )
    seed_dialog = SystemConfigDialog((available,), None, ("main",))
    seed_system = seed_dialog.physical_system()
    region_a = seed_system.regions[0]
    source_b = Boundary(
        id="boundary:legacy-source-b",
        name="source_B",
        region_id=region_a.id,
        group=PhysicalGroupRef(mesh_id=seed_system.meshes[0].id, dimension=2, name="source_B"),
        kind=BoundaryKind.MOVING,
    )
    legacy_system = replace(seed_system, boundaries=(*seed_system.boundaries, source_b))
    dialog = SystemConfigDialog((available,), legacy_system, ("main",))

    dialog._add_default_region()
    volume_combo = dialog.regions_table.cellWidget(1, 3)
    assert isinstance(volume_combo, QComboBox)
    volume_combo.setCurrentIndex(volume_combo.findData("volume_B"))
    dialog._refresh_boundaries()

    moved_row = next(
        row
        for row in range(dialog.boundaries_table.rowCount())
        if dialog.boundaries_table.item(row, 0).text() == "Interior Air 2"
        and dialog.boundaries_table.item(row, 2).text() == "source_B"
    )
    assignment = dialog.boundaries_table.cellWidget(moved_row, 3)
    assert isinstance(assignment, QComboBox)
    assert assignment.property("boundary_id") == source_b.id
    assert assignment.currentData() == BoundaryKind.MOVING
