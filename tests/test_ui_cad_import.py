"""The CAD Import editor inside the main window's design dock."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from time import monotonic, sleep
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QPalette
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QDockWidget, QHeaderView, QLabel, QScrollArea, QSplitter

from blab.generators import catalog as catalog_module
from blab.physical_compiler import PhysicalSystemCompiler
from blab.ui.settings import load_gui_preferences, save_gui_preferences
from cad_fixtures import bored_ball, strip_names, vented_box, waveguide
from cad_plugin import inspect_step
from conftest import _stub_qt_interactor_class
from test_cad_import_provider import PROVIDER, ROOT
from test_ui_settings import MemorySettings

VENTED = "blab.vented_enclosure"


def wait(qapp, condition, timeout=30):
    deadline = monotonic() + timeout
    while not condition() and monotonic() < deadline:
        qapp.processEvents()
        sleep(0.005)
    assert condition()


class _StubActor:
    def GetAddressAsString(self, _type_name):
        return hex(id(self))


def _cad_qt_interactor_class():
    """The plotter stub plus what the face picking uses: actors from ``add_mesh``, whose meshes it records."""

    class CadQtInteractor(_stub_qt_interactor_class()):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.meshes = []  # (mesh, add_mesh keywords) since the last clear
            self.labels = []  # point label texts since the last clear

        @property
        def colours(self):
            return [kwargs["color"] for mesh, kwargs in self.meshes if not mesh.n_lines and "name" not in kwargs]

        def clear(self):
            self.calls.append("clear")
            self.meshes = []
            self.labels = []

        def add_point_labels(self, points, labels, **kwargs):
            self.labels += labels

        def add_mesh(self, mesh, **kwargs):
            # pyvista resets the camera unless told not to while it counts the camera as unset, which here is always.
            self.calls.append("add_mesh" if kwargs.get("reset_camera") is False else "add_mesh_resetting_camera")
            self.meshes.append((mesh, kwargs))
            return _StubActor()

    return CadQtInteractor


@pytest.fixture
def blank_cad_editor(main_window, monkeypatch):
    """A new CAD Import design in the dock, without a STEP file."""
    import pyvistaqt

    monkeypatch.setattr(pyvistaqt, "QtInteractor", _cad_qt_interactor_class())
    catalog = catalog_module.ProviderCatalog([ROOT], enabled=[PROVIDER])
    catalog.scan()
    monkeypatch.setattr(catalog_module, "_catalog", catalog)
    main_window.preferences.default_geometry_provider = PROVIDER
    main_window.add_generator_document()
    return main_window._provider_editors[main_window.active_generator_document().id]


@pytest.fixture
def cad_editor(blank_cad_editor, qapp, tmp_path):
    """A CAD Import design in the dock, with the vented box loaded and previewed."""
    editor, host = blank_cad_editor
    editor.set_step_path(str(vented_box(tmp_path / "box.step")))
    wait(qapp, lambda: not editor.loading)
    return editor, host


def body_items(editor):
    return {
        editor.tree.topLevelItem(i).text(0): editor.tree.topLevelItem(i) for i in range(editor.tree.topLevelItemCount())
    }


def children(item):
    return [item.child(i).text(0) for i in range(item.childCount())]


def faces_row(body):
    """The Faces row of a body row."""
    return next(body.child(i) for i in range(body.childCount()) if body.child(i).text(0) == "Faces")


def test_loaded_step_lists_bodies_and_faces_with_size_and_symmetry(cad_editor):
    editor, host = cad_editor

    assert host.snapshot().source["step_path"].endswith("box.step")
    bodies = body_items(editor)
    assert list(bodies) == ["Chamber", "Port", "Cabinet"]
    assert children(bodies["Chamber"]) == ["Volume", "Faces"]
    assert children(bodies["Cabinet"]) == ["Faces"]  # an exterior surface body has no volume
    assert "Driver" in children(faces_row(bodies["Chamber"]))  # STEP face name
    assert all(name.startswith("Face ") for name in children(faces_row(bodies["Cabinet"])))
    assert {name: item.text(2) for name, item in bodies.items()} == {
        "Chamber": "Air body",
        "Port": "Air body",
        "Cabinet": "Exterior surface",
    }
    assert not editor.kind_combos  # a dropdown only on closed shells
    assert "120.0 x 120.0 x 210.0 mm" in editor.info.text()
    assert "Symmetry Off" in editor.info.text()
    assert editor.visible_faces() == {face["id"] for face in editor.geometry["faces"]}


def test_tree_gives_spare_width_to_names_and_fits_the_size_and_kind_columns(cad_editor):
    editor, _ = cad_editor
    header = editor.tree.header()

    assert [editor.tree.headerItem().text(c) for c in range(3)] == ["Body / face", "Mesh size", "Kind"]
    assert not header.stretchLastSection()
    assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Stretch
    assert header.sectionResizeMode(1) == QHeaderView.ResizeMode.ResizeToContents
    assert header.sectionResizeMode(2) == QHeaderView.ResizeMode.ResizeToContents


def test_tree_hides_bodies_and_faces_and_show_all_restores_them(cad_editor, qapp):
    editor, _ = cad_editor
    faces = {face["id"]: face for face in editor.geometry["faces"]}
    bodies = body_items(editor)

    bodies["Cabinet"].setCheckState(0, Qt.CheckState.Unchecked)
    assert editor.visible_faces() == {f for f, face in faces.items() if "Cabinet" not in face["bodies"]}
    chamber = bodies["Chamber"]
    driver = next(item for item in children_items(faces_row(chamber)) if item.text(0) == "Driver")
    driver.setCheckState(0, Qt.CheckState.Unchecked)
    assert driver.data(0, Qt.ItemDataRole.UserRole) not in editor.visible_faces()
    assert chamber.checkState(0) == Qt.CheckState.PartiallyChecked

    editor.show_all_button.click()
    assert editor.visible_faces() == set(faces)
    assert all(item.checkState(0) == Qt.CheckState.Checked for item in bodies.values())
    qapp.processEvents()
    assert "add_mesh" in editor.viewport.plotter.calls


def test_a_face_shared_by_two_bodies_hides_under_both(cad_editor, qapp, tmp_path):
    editor, _ = cad_editor
    # A cabinet front without a hole covers the port mouth, so Port and Cabinet share that face.
    editor.set_step_path(str(vented_box(tmp_path / "covered.step", covered=True)))
    wait(qapp, lambda: not editor.loading)
    assert "Warning: " in editor.warnings.text() and "covers a face" in editor.warnings.text()
    assert "covers a face" not in editor.info.text()  # the info line holds only Size / Symmetry
    items = {}
    for body in body_items(editor).values():
        for item in children_items(faces_row(body)):
            items.setdefault(item_face(item), []).append(item)
    shared = next(face_items for face_items in items.values() if len(face_items) > 1)  # port mouth: Port + Cabinet

    shared[0].setCheckState(0, Qt.CheckState.Unchecked)

    assert all(item.checkState(0) == Qt.CheckState.Unchecked for item in shared)
    assert shared[0].data(0, Qt.ItemDataRole.UserRole) not in editor.visible_faces()


def test_sizing_settings_edit_the_source_and_show_the_target_size(cad_editor, main_window, qapp):
    editor, host = cad_editor
    main_window.freq_max_spin.setValue(3430)
    editor.spins["elements_per_wavelength"].setValue(10)
    wait(qapp, lambda: "3430" in editor.target_label.text())

    assert host.snapshot().source["elements_per_wavelength"] == 10
    assert "Target element size FEM 10.0 mm = c / 3430 Hz / 10, BEM 20.0 mm" in editor.target_label.text()
    editor.spins["growth_rate"].setValue(1.2)
    assert host.snapshot().source["growth_rate"] == 1.2

    editor.set_operation_state(SimpleNamespace(active=True))
    assert not editor.step_path.isEnabled() and not editor.spins["growth_rate"].isEnabled()
    assert not editor.fineness.isEnabled()
    editor.set_operation_state(SimpleNamespace(active=False))
    assert editor.step_path.isEnabled() and editor.spins["growth_rate"].isEnabled()


def test_shown_sizes_use_the_project_sound_speed(cad_editor, main_window, qapp):
    from blab.physical_model import AcousticRegion, AcousticRegionKind, PhysicalSystem

    editor, _ = cad_editor
    main_window.freq_max_spin.setValue(3430)
    exterior = AcousticRegion("air", "Exterior air", AcousticRegionKind.UNBOUNDED_AIR, (), sound_speed_m_per_s=686.0)
    main_window.project.physical_system = PhysicalSystem("system", "System", (), (exterior,), ())
    editor.spins["elements_per_wavelength"].setValue(10)  # an edit fetches the project context
    wait(qapp, lambda: "c = 686 m/s" in editor.target_label.text())

    assert "Target element size FEM 20.0 mm = c / 3430 Hz" in editor.target_label.text()
    face = cabinet_faces(editor, lambda f: True)[0]
    walls = size_combo(editor, face_item(editor, face))
    assert walls.currentText() == "Global (BEM Medium)" and "(40 mm)" in walls.toolTip()

    main_window.project.physical_system = None  # no exterior region: the default
    editor.spins["elements_per_wavelength"].setValue(5)
    wait(qapp, lambda: "c = 343 m/s" in editor.target_label.text())
    assert "Target element size FEM 20.0 mm" in editor.target_label.text()


def test_host_generate_accepts_locked_meshes_and_system_entries(cad_editor, main_window, monkeypatch, qapp, tmp_path):
    from blab.ui.main_window import geometry_workflow

    editor, host = cad_editor
    monkeypatch.setattr(geometry_workflow, "GENERATED_GEOMETRY_ROOT", tmp_path / "runs")
    errors = []
    monkeypatch.setattr(main_window, "show_error", lambda *args: errors.append(args))
    main_window.freq_max_spin.setValue(3430)

    host.generate(expected_revision=host.snapshot().revision)
    assert not editor.spins["growth_rate"].isEnabled()  # busy while the host generates
    wait(qapp, lambda: not main_window.geometry_controller.active)

    assert not errors
    document = main_window.active_generator_document()
    assert [m.id for m in document.artifact.meshes] == ["chamber", "exterior"]
    entries = main_window._mesh_config_dialog_entries()
    assert len(entries) == 2 and all(e.locked and e.assembly_id == document.id for e in entries)
    assert len(PhysicalSystemCompiler().compile(main_window.project.physical_system).interfaces) == 1
    assert "accepted" in editor.status.text()
    assert main_window.editor_dock.features() & QDockWidget.DockWidgetFeature.DockWidgetClosable


def test_host_cancel_stops_meshing_without_meshes(cad_editor, main_window, monkeypatch, qapp, tmp_path):
    from blab.ui.main_window import geometry_workflow

    editor, host = cad_editor
    monkeypatch.setattr(geometry_workflow, "GENERATED_GEOMETRY_ROOT", tmp_path / "runs")
    monkeypatch.setattr(main_window, "show_error", lambda *args: None)

    host.generate(expected_revision=host.snapshot().revision)
    main_window.cancel_geometry_generation()
    wait(qapp, lambda: not main_window.geometry_controller.active)

    assert main_window.active_generator_document().artifact is None
    assert editor.spins["growth_rate"].isEnabled()


def test_clear_shows_everything_and_clears_the_selection(cad_editor):
    editor, host = cad_editor
    source = host.snapshot().source
    bodies = body_items(editor)
    bodies["Port"].setCheckState(0, Qt.CheckState.Unchecked)
    faces_row(bodies["Chamber"]).child(0).setSelected(True)
    assert editor.tree.selectedItems()

    editor.clear_button.click()

    assert editor.visible_faces() == {face["id"] for face in editor.geometry["faces"]}
    assert editor.tree.selectedItems() == []
    assert host.snapshot().source == source


def test_closed_shell_gets_its_guessed_kind_unasked_and_the_kind_dropdown_changes_it(cad_editor, qapp, tmp_path):
    editor, host = cad_editor

    editor.set_step_path(str(waveguide(tmp_path / "wg.step", surface_model=True)))
    wait(qapp, lambda: not editor.loading)

    assert host.snapshot().source["closed_shells"] == {}  # a guess is no choice
    assert list(body_items(editor)) == ["Closed shell 1", "Exterior surface 1"]
    assert list(editor.kind_combos) == ["Closed shell 1"]  # only closed shells have a kind
    combo = editor.kind_combos["Closed shell 1"]
    assert [combo.itemText(i) for i in range(combo.count())] == ["Air body", "Exterior surface"]
    assert combo.currentData() == "air"

    combo.setCurrentIndex(combo.findData("exterior"))
    wait(qapp, lambda: not editor.loading)

    assert host.snapshot().source["closed_shells"] == {"Closed shell 1": "exterior"}
    assert {name: item.toolTip(0) for name, item in body_items(editor).items()} == {
        "Closed shell 1": "Exterior surface body",
        "Exterior surface 1": "Exterior surface body",
    }
    assert editor.kind_combos["Closed shell 1"].currentData() == "exterior"
    editor.set_operation_state(SimpleNamespace(active=True))
    assert not editor.kind_combos["Closed shell 1"].isEnabled()


def test_preview_is_smooth_shaded_with_face_outlines_and_no_triangle_edges(cad_editor):
    editor, _ = cad_editor
    meshes = [(mesh, kwargs) for mesh, kwargs in editor.viewport.plotter.meshes if "name" not in kwargs]  # no guides
    faces = [kwargs for mesh, kwargs in meshes if not mesh.n_lines]
    outlines = [kwargs for mesh, kwargs in meshes if mesh.n_lines]

    assert faces and all(kwargs.get("smooth_shading") and not kwargs.get("show_edges") for kwargs in faces)
    assert len(outlines) == 1 and outlines[0]["pickable"] is False


def test_face_outlines_are_the_edges_between_faces(blank_cad_editor):
    editor, _ = blank_cad_editor
    preview = {
        "points": np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]], dtype=float),
        "triangles": np.array([[0, 1, 2], [0, 2, 3], [1, 4, 2]]),  # a square face and a triangular face beside it
        "face_ids": np.array([1, 1, 2]),
    }

    editor.viewport.show_faces(preview, {1, 2})

    (outlines,) = [mesh for mesh, kwargs in editor.viewport.plotter.meshes if mesh.n_lines]
    edges = outlines.lines.reshape(-1, 3)[:, 1:]
    # Includes the shared face boundary (1, 2), but not the square's triangle diagonal (0, 2).
    assert sorted(map(tuple, edges)) == [(0, 1), (0, 3), (1, 2), (1, 4), (2, 3), (2, 4)]


def cabinet_faces(editor, predicate):
    return [f["id"] for f in editor.geometry["faces"] if list(f["bodies"]) == ["Cabinet"] and predicate(f)]


def test_picked_faces_become_a_face_group_that_generates_a_rigid_boundary(
    cad_editor, main_window, monkeypatch, qapp, tmp_path
):
    from blab.ui.main_window import geometry_workflow

    editor, host = cad_editor
    sides = cabinet_faces(editor, lambda f: abs(abs(f["normal"][0]) - 1) < 1e-9)  # the two X sides
    editor.pick_face(sides[0])
    editor.pick_face(sides[1], add=True)
    assert editor.selected_faces() == set(sides)

    editor.group_selection("Sides")

    (group,) = host.snapshot().source["face_groups"]
    assert group["name"] == "Sides"
    assert sorted(abs(face["centre"][0]) for face in group["faces"]) == pytest.approx([0.06, 0.06])
    assert {face["surface_type"] for face in group["faces"]} == {"Plane"}
    assert [editor.groups_table.item(r, 0).text() for r in range(editor.groups_table.rowCount())] == [
        "walls",
        "Interface 1 Port",  # the interface
        "Sides",
        "Driver",  # from the STEP file
    ]
    monkeypatch.setattr(geometry_workflow, "GENERATED_GEOMETRY_ROOT", tmp_path / "runs")
    errors = []
    monkeypatch.setattr(main_window, "show_error", lambda *args: errors.append(args))
    host.generate(expected_revision=host.snapshot().revision)
    wait(qapp, lambda: not main_window.geometry_controller.active)

    assert not errors
    sides_boundary = next(b for b in main_window.project.physical_system.boundaries if b.group.name == "Sides")
    assert sides_boundary.kind == "rigid"


def all_rows(editor):
    table = editor.groups_table
    return {table.item(r, 0).text(): int(table.item(r, 1).text()) for r in range(table.rowCount())}


def group_rows(editor):
    """The face group rows, without walls and the interface groups."""
    interfaces = {i["name"] for i in editor.geometry["interfaces"]}
    return {name: count for name, count in all_rows(editor).items() if name not in {"walls", *interfaces}}


def test_placement_fields_edit_the_source_and_move_the_display_without_reading(cad_editor, monkeypatch):
    editor, host = cad_editor
    editor.pick_face(chamber_back(editor))
    editor.group_selection("Back")
    reads = record_reads(editor, monkeypatch)
    before = face_points(editor)

    editor.rotation_spins[1].setValue(90)  # about Y: (x, y, z) -> (z, y, -x)
    editor.translation_spins[0].setValue(25)

    source = host.snapshot().source
    assert (source["rotation_deg"], source["translation_mm"]) == ([0.0, 90.0, 0.0], [25.0, 0.0, 0.0])
    assert reads == [] and not editor.loading
    after = face_points(editor)
    assert np.allclose(after, before[:, [2, 1, 0]] * [1, 1, -1] + [0.025, 0, 0], atol=1e-9)
    assert "210.0 x 120.0 x 120.0 mm" in editor.info.text()
    assert editor.groups_needing_review() == [] and group_rows(editor)["Back"] == 1
    editor.translation_spins[0].setValue(0)
    editor.rotation_spins[1].setValue(0)
    assert np.allclose(face_points(editor), before, atol=1e-9)


def face_points(editor):
    """The points of the first face mesh drawn."""
    return next(mesh for mesh, kwargs in editor.viewport.plotter.meshes if "name" not in kwargs).points.copy()


def guides(editor):
    """``{name: mesh}`` of the axes and symmetry planes drawn."""
    return {kwargs["name"]: mesh for mesh, kwargs in editor.viewport.plotter.meshes if "name" in kwargs}


def test_labelled_axes_and_symmetry_planes_reach_past_the_model_and_one_toggle_hides_them(cad_editor):
    editor, _ = cad_editor

    shown = guides(editor)
    assert set(shown) == {"axis X", "axis Y", "axis Z", "plane X=0", "plane Y=0"}
    kwargs = {k: kw for _, kw in editor.viewport.plotter.meshes for k in [kw.get("name")] if k}
    assert all(kw["pickable"] is False for kw in kwargs.values())
    assert editor.viewport.plotter.labels == ["X · Horizontal", "Y · Vertical", "Z · On Axis"]
    plane = shown["plane X=0"].points
    assert np.allclose(plane[:, 0], 0)
    # The model spans x, y -60..60 mm and z -210..0 mm; axes and planes reach past it by a fifth of each span.
    assert plane[:, 1].min() == pytest.approx(-0.084) and plane[:, 2].min() == pytest.approx(-0.252)
    assert plane[:, 2].max() == pytest.approx(0.042)  # at least a tenth of the largest size past the origin
    # The axes reach half the span further, so their labels at the tips sit clear of the model.
    z_axis = shown["axis Z"].points
    assert z_axis[:, 2].min() == pytest.approx(-0.315) and z_axis[:, 2].max() == pytest.approx(0.105)
    assert shown["axis X"].points[:, 0].max() == pytest.approx(0.12)
    # Both planes in one green, not the Y axis's.
    colours = {name: kw["color"] for name, kw in kwargs.items()}
    assert colours["plane X=0"] == colours["plane Y=0"] != colours["axis Y"]

    editor.guides_check.setChecked(False)
    assert guides(editor) == {} and face_points(editor) is not None
    editor.guides_check.setChecked(True)
    assert len(guides(editor)) == 5


def record_reads(editor, monkeypatch):
    """The STEP file each read of ``editor`` reads, in order."""
    editor_module = sys.modules[type(editor).__module__]
    read = editor_module.preview_in_worker
    reads = []

    def recorded(step_path, *args, **kwargs):
        reads.append(Path(step_path).name)
        return read(step_path, *args, **kwargs)

    monkeypatch.setattr(editor_module, "preview_in_worker", recorded)
    return reads


def test_opening_a_step_file_with_a_closed_shell_reads_it_once(cad_editor, monkeypatch, qapp, tmp_path):
    editor, _ = cad_editor
    reads = record_reads(editor, monkeypatch)

    editor.set_step_path(str(waveguide(tmp_path / "wg.step", surface_model=True)))
    wait(qapp, lambda: not editor.loading)

    assert reads == ["wg.step"]


def test_the_previous_preview_stays_while_reading_and_with_the_error_when_reading_fails(
    cad_editor, monkeypatch, qapp, tmp_path
):
    editor, _ = cad_editor
    bodies = list(body_items(editor))
    editor_module = sys.modules[type(editor).__module__]

    def failing(*args, **kwargs):
        raise ValueError("Could not read STEP file broken.step")

    monkeypatch.setattr(editor_module, "preview_in_worker", failing)
    broken = tmp_path / "broken.step"
    broken.write_text("not STEP", encoding="utf-8")
    editor.set_step_path(str(broken))

    assert editor.loading  # the result arrives through the event loop
    assert list(body_items(editor)) == bodies and "Reading broken.step" in editor.info.text()
    wait(qapp, lambda: not editor.loading)
    assert list(body_items(editor)) == bodies and editor.geometry is not None
    assert "Could not read STEP file broken.step" in editor.info.text()
    assert face_points(editor) is not None  # still drawn


@pytest.mark.parametrize(
    ("name", "pick", "message"),
    [
        ("Interface 1 Port", "cabinet", "Face group name 'Interface 1 Port' is reserved"),
        ("Mouth", "interface", "Face .* stays in interface 'Interface 1 Port'"),
    ],
)
def test_reserved_names_and_interface_faces_cannot_be_grouped(cad_editor, name, pick, message):
    editor, host = cad_editor
    if pick == "interface":
        face = editor.geometry["interfaces"][0]["face_ids"][0]
    else:
        face = cabinet_faces(editor, lambda f: True)[0]
    editor.pick_face(face)

    editor.group_selection(name)

    assert host.snapshot().source.get("face_groups", []) == []
    assert re.search(message, editor.status.text())


FINE = {"elements_per_wavelength": 6}
COARSE = {"elements_per_wavelength": 2}
SIZE_ENTRIES = [
    "Global (FEM Fine)",
    "FEM Fine",
    "FEM Medium",
    "FEM Coarse",
    "BEM Fine",
    "BEM Medium",
    "BEM Coarse",
    "Custom…",
]


def size_combo(editor, row):
    """The Mesh size dropdown of a tree item, or of the face-group list row named ``row``."""
    if isinstance(row, str):
        return editor.groups_table.cellWidget(row_of(editor, row), 2)
    return editor.tree.itemWidget(row, 1)


def size_entries(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def has_size_entries(combo):
    """The combo offers the Mesh size choices, Global named by the mesh fineness that sizes its row."""
    entries = size_entries(combo)[: len(SIZE_ENTRIES)]
    return entries[0].startswith("Global") and entries[1:] == SIZE_ENTRIES[1:]


def choose_size(combo, text, qapp):
    """Choose ``text`` in a Mesh size dropdown, as a user does."""
    assert has_size_entries(combo)
    combo.setCurrentIndex(combo.findText(text))
    combo.activated.emit(combo.currentIndex())
    qapp.processEvents()  # applied once the dropdown has closed


def child_row(body, text):
    return next(item for item in children_items(body) if item.text(0) == text)


def tree_rows(editor):
    """Every body, Volume, Faces and face row of the tree."""
    rows = []
    for body in body_items(editor).values():
        rows.append(body)
        for child in children_items(body):
            rows += [child, *children_items(child)]
    return rows


def body_faces(editor, name):
    return {f["id"] for f in editor.geometry["faces"] if name in f["bodies"]}


def test_every_row_has_a_mesh_size_dropdown_showing_global_its_own_size_or_mixed(cad_editor, qapp):
    editor, host = cad_editor
    rows = tree_rows(editor)

    assert all(has_size_entries(size_combo(editor, row)) for row in rows)
    levels = {"Global (FEM Fine)", "Global (BEM Medium)", "Global"}  # what sizes each row: see the next test
    assert {size_combo(editor, row).currentText() for row in rows} <= levels
    assert {size_combo(editor, name).currentText() for name in all_rows(editor)} <= levels

    side = cabinet_faces(editor, lambda f: True)[0]
    editor.set_face_size({side}, COARSE)

    cabinet = body_items(editor)["Cabinet"]
    assert size_combo(editor, face_item(editor, side)).currentText() == "FEM Coarse"
    for row in (cabinet, faces_row(cabinet), "walls"):
        combo = size_combo(editor, row)
        assert combo.currentText() == "Mixed"
        assert not combo.model().item(combo.currentIndex()).isEnabled()  # shown, never chosen
    assert size_combo(editor, "Driver").currentText() == "Global (FEM Fine)"


def test_a_body_size_sets_its_volume_and_faces_and_a_volume_size_only_the_volume(cad_editor, monkeypatch, qapp):
    editor, host = cad_editor
    chamber = body_items(editor)["Chamber"]
    volume, faces = child_row(chamber, "Volume"), faces_row(chamber)
    assert volume.flags() & Qt.ItemFlag.ItemIsUserCheckable == Qt.ItemFlag.NoItemFlags  # a volume is not drawn

    choose_size(size_combo(editor, chamber), "FEM Medium", qapp)

    source = host.snapshot().source
    assert source["volume_sizes"] == {"Chamber": {"elements_per_wavelength": 4}}
    assert len(source["face_sizes"]) == len(body_faces(editor, "Chamber"))
    assert "body_sizes" not in source and "face_group_sizes" not in source
    assert size_combo(editor, "Driver").currentText() == "FEM Medium"  # a chamber face
    assert size_combo(editor, "walls").currentText() == "Mixed"  # chamber and cabinet walls

    choose_size(size_combo(editor, volume), "FEM Coarse", qapp)
    assert host.snapshot().source["volume_sizes"] == {"Chamber": COARSE}
    assert size_combo(editor, chamber).currentText() == "Mixed"
    assert size_combo(editor, faces).currentText() == "FEM Medium"

    monkeypatch.setattr(editor, "ask_size_mm", lambda: 25.0)
    choose_size(size_combo(editor, faces), "Custom…", qapp)
    assert {e["size_m"] for e in host.snapshot().source["face_sizes"]} == {0.025}
    assert size_combo(editor, faces).currentText() == "25 mm"
    monkeypatch.setattr(editor, "ask_size_mm", lambda: None)  # cancelled
    choose_size(size_combo(editor, chamber), "Custom…", qapp)
    assert host.snapshot().source["volume_sizes"] == {"Chamber": COARSE}

    choose_size(size_combo(editor, chamber), "Global (FEM Fine)", qapp)
    assert (host.snapshot().source["volume_sizes"], host.snapshot().source["face_sizes"]) == ({}, [])


def test_a_size_chosen_in_a_row_dropdown_shows_in_the_dropdown(cad_editor, qapp):
    editor, host = cad_editor
    side = cabinet_faces(editor, lambda f: True)[0]
    volume = child_row(body_items(editor)["Chamber"], "Volume")

    choose_size(size_combo(editor, "walls"), "FEM Coarse", qapp)
    choose_size(size_combo(editor, face_item(editor, side)), "BEM Coarse", qapp)
    choose_size(size_combo(editor, volume), "BEM Fine", qapp)

    assert host.snapshot().source["face_sizes"]
    assert size_combo(editor, face_item(editor, side)).currentText() == "BEM Coarse"
    assert size_combo(editor, volume).currentText() == "BEM Fine"
    assert size_combo(editor, "walls").currentText() == "Mixed"  # one of its faces has its own size


def face_item(editor, face_id):
    return next(
        item
        for body in body_items(editor).values()
        for item in children_items(faces_row(body))
        if item_face(item) == face_id
    )


def children_items(item):
    return [item.child(i) for i in range(item.childCount())]


def item_face(item):
    return item.data(0, Qt.ItemDataRole.UserRole)


def test_mesh_size_from_the_face_menu_sets_a_face_override_on_the_selection(cad_editor, monkeypatch):
    editor, host = cad_editor
    first, second = cabinet_faces(editor, lambda f: abs(abs(f["normal"][0]) - 1) < 1e-9)
    interface_face = editor.geometry["interfaces"][0]["face_ids"][0]
    editor.pick_face(first)
    editor.pick_face(second, add=True)

    menu = editor.face_menu(second)
    assert list(menu_entries(menu)) == ['Move to "Driver"', "New group…", "Mesh size"]  # both in walls
    sizes = menu_entries(menu_entries(menu)["Mesh size"].menu())
    assert list(sizes) == SIZE_ENTRIES
    sizes["FEM Fine"].trigger()

    entries = host.snapshot().source["face_sizes"]
    assert [e["elements_per_wavelength"] for e in entries] == [6, 6]
    assert {round(abs(e["face"]["centre"][0]), 3) for e in entries} == {0.06}  # stored like face group faces
    assert size_combo(editor, face_item(editor, first)).currentText() == "FEM Fine"

    interface_entries = menu_entries(editor.face_menu(interface_face))
    assert list(interface_entries) == ["Rename interface…", "Mesh size"]
    menu_entries(interface_entries["Mesh size"].menu())["FEM Coarse"].trigger()  # the face, outside the selection
    assert len(host.snapshot().source["face_sizes"]) == 3
    menu_entries(menu_entries(editor.face_menu(first))["Mesh size"].menu())["Global (FEM Fine)"].trigger()
    assert len(host.snapshot().source["face_sizes"]) == 1


def test_designs_with_no_body_or_face_group_sizes_are_left_as_they_are(
    cad_editor, main_window, monkeypatch, qapp, tmp_path
):
    editor, host = cad_editor
    main_window.freq_max_spin.setValue(3430)
    editor.spins["curvature_elements_per_circle"].setValue(13)  # an edit fetches the project context
    wait(qapp, lambda: "3430" in editor.target_label.text())
    empty = host.snapshot().source | {"body_sizes": {}, "face_group_sizes": {}}  # as earlier slices saved them
    snapshot = host.update_source(empty, expected_revision=host.snapshot().revision)
    update_source = host.update_source
    stored = []
    monkeypatch.setattr(
        host, "update_source", lambda source, **kw: stored.append(source) or update_source(source, **kw)
    )

    editor.apply_source(snapshot.source, snapshot.revision)
    editor.set_step_path(str(vented_box(tmp_path / "other.step")))  # the one edit: reads and draws the file
    wait(qapp, lambda: not editor.loading)
    editor.spins["curvature_elements_per_circle"].setValue(14)  # and another, which fetches the context again
    qapp.processEvents()

    assert len(stored) == 2
    assert "sizes are now set" not in editor.status.text()


def test_existing_designs_get_the_face_and_volume_sizes_that_applied_before(cad_editor, main_window, qapp, tmp_path):
    from test_cad_import_provider import generate, make_project, median_edge, tet_count

    editor, host = cad_editor
    main_window.freq_max_spin.setValue(3430)
    editor.spins["curvature_elements_per_circle"].setValue(13)  # an edit fetches the project context
    wait(qapp, lambda: "3430" in editor.target_label.text())
    legacy = {
        "elements_per_wavelength": 5.0,  # 20 mm
        "body_sizes_m": {"Chamber": 0.04},
        "face_group_sizes_m": {"Driver": 0.008},
        "body_sizes": {"Port": {"size_m": 0.03}},
        "face_group_sizes": {},
    }
    old = {k: v for k, v in host.snapshot().source.items() if k not in ("face_sizes", "volume_sizes")} | legacy

    snapshot = host.update_source(old, expected_revision=host.snapshot().revision)
    editor.apply_source(snapshot.source, snapshot.revision)

    new = host.snapshot().source
    assert not {"body_sizes", "face_group_sizes", "body_sizes_m", "face_group_sizes_m"} & new.keys()
    assert new["volume_sizes"] == {"Chamber": {"size_m": 0.04}, "Port": {"size_m": 0.03}}
    driver = next(f["id"] for f in editor.geometry["faces"] if f["step_name"] == "Driver")
    assert size_combo(editor, face_item(editor, driver)).currentText() == "8 mm"
    chamber = body_faces(editor, "Chamber") - {driver}
    assert {size_combo(editor, face_item(editor, f)).currentText() for f in chamber} == {"40 mm"}
    port = body_faces(editor, "Port")  # its interface face too: no face group size
    assert {size_combo(editor, face_item(editor, f)).currentText() for f in port} == {"30 mm"}
    cabinet = body_faces(editor, "Cabinet")
    assert {size_combo(editor, face_item(editor, f)).currentText() for f in cabinet} == {"Global (BEM Medium)"}
    assert editor.fineness.currentText() == "Custom" and not editor.spins["elements_per_wavelength"].isHidden()

    catalog = catalog_module._catalog
    step = Path(new["step_path"])
    _, before = generate(catalog.factory(PROVIDER, "backend")(), make_project(catalog, step, **old), tmp_path / "a")
    _, after = generate(catalog.factory(PROVIDER, "backend")(), make_project(catalog, step, **new), tmp_path / "b")
    assert tet_count(after) == pytest.approx(tet_count(before), rel=0.02)
    for mesh_id, group in (("chamber", "Driver"), ("chamber", "walls"), ("exterior", "walls")):
        assert median_edge(after, mesh_id, group) == pytest.approx(median_edge(before, mesh_id, group), rel=0.02)


def test_mesh_fineness_levels_set_elements_per_wavelength_and_name_the_frequency_they_hold(
    cad_editor, main_window, qapp
):
    editor, host = cad_editor
    main_window.freq_max_spin.setValue(3000)
    editor.spins["curvature_elements_per_circle"].setValue(13)  # an edit fetches the project context
    wait(qapp, lambda: "3000" in editor.target_label.text())
    fineness, custom = editor.fineness, editor.spins["elements_per_wavelength"]

    labels = {label.text() for label in editor.sizing_box.findChildren(QLabel)}
    assert "FEM mesh fineness" in labels and "Mesh fineness" not in labels
    assert fineness.accessibleName() == "FEM mesh fineness"
    assert [fineness.itemText(i) for i in range(fineness.count())] == ["FEM Fine", "FEM Medium", "FEM Coarse", "Custom"]
    assert fineness.currentText() == "FEM Fine" and custom.isHidden()  # new designs
    assert "6 per wavelength up to 3000 Hz" in fineness.itemData(0, Qt.ItemDataRole.ToolTipRole)
    assert "up to 2000 Hz" in fineness.itemData(1, Qt.ItemDataRole.ToolTipRole)
    assert "6 per wavelength up to 1000 Hz" in fineness.itemData(2, Qt.ItemDataRole.ToolTipRole)

    fineness.setCurrentIndex(1)
    fineness.activated.emit(1)
    assert host.snapshot().source["elements_per_wavelength"] == 4
    fineness.setCurrentIndex(3)
    fineness.activated.emit(3)
    assert not custom.isHidden() and fineness.currentText() == "Custom"
    custom.setValue(4)  # a level's number keeps Custom while chosen
    assert fineness.currentText() == "Custom"
    custom.setValue(9)
    assert host.snapshot().source["elements_per_wavelength"] == 9
    assert "Target element size" in editor.target_label.text()


def test_global_names_the_mesh_fineness_that_sizes_the_row(cad_editor, qapp):
    editor, host = cad_editor
    cabinet, chamber = body_items(editor)["Cabinet"], body_items(editor)["Chamber"]
    exterior = set(cabinet_faces(editor, lambda f: True))
    interface = {f for i in editor.geometry["interfaces"] for f in i["face_ids"]}

    def shown(row):
        return size_combo(editor, row).currentText()

    assert interface
    assert {shown(face_item(editor, f)) for f in exterior} == {"Global (BEM Medium)"}
    assert {shown(face_item(editor, f)) for f in interface} == {"Global (FEM Fine)"}  # the finer on an interface face
    assert shown(child_row(chamber, "Volume")) == shown(chamber) == shown(faces_row(chamber)) == "Global (FEM Fine)"
    assert shown(cabinet) == shown(faces_row(cabinet)) == "Global (BEM Medium)"
    assert shown("walls") == "Global"  # FEM and BEM faces

    port = body_items(editor)["Port"]
    editor.bem_fineness.setCurrentIndex(0)
    editor.bem_fineness.activated.emit(0)  # BEM Fine, finer than FEM Fine
    assert {shown(face_item(editor, f)) for f in exterior | interface} == {"Global (BEM Fine)"}
    assert shown(cabinet) == shown(faces_row(cabinet)) == "Global (BEM Fine)"
    assert shown(chamber) == "Global (FEM Fine)"
    assert shown(port) == shown(faces_row(port)) == "Global"  # its interface face follows BEM, its others FEM


def test_bem_mesh_fineness_levels_write_elements_per_wavelength_and_name_their_edge_length(
    cad_editor, main_window, qapp
):
    editor, host = cad_editor
    main_window.freq_max_spin.setValue(3430)
    editor.spins["curvature_elements_per_circle"].setValue(13)  # an edit fetches the project context
    wait(qapp, lambda: "3430" in editor.target_label.text())
    bem, custom = editor.bem_fineness, editor.spins["bem_elements_per_wavelength"]

    labels = [label.text() for label in editor.sizing_box.findChildren(QLabel)]
    assert labels.index("BEM mesh fineness") == labels.index("FEM mesh fineness") + 1
    assert bem.accessibleName() == "BEM mesh fineness"
    assert [bem.itemText(i) for i in range(bem.count())] == ["BEM Fine", "BEM Medium", "BEM Coarse", "Custom"]
    assert bem.currentText() == "BEM Medium" and custom.isHidden()  # new designs
    assert "(8.33 mm)" in bem.itemData(0, Qt.ItemDataRole.ToolTipRole)  # c / fmax / 12 = 0.1 m / 12
    assert "(20 mm)" in bem.itemData(1, Qt.ItemDataRole.ToolTipRole)
    assert "(333 mm)" in bem.itemData(2, Qt.ItemDataRole.ToolTipRole)
    assert "(16.7 mm)" in editor.fineness.itemData(0, Qt.ItemDataRole.ToolTipRole)
    assert "BEM 20.0 mm = c / 3430 Hz / 5" in editor.target_label.text()

    bem.setCurrentIndex(2)
    bem.activated.emit(2)
    assert host.snapshot().source["bem_elements_per_wavelength"] == 0.3
    assert "BEM 333.3 mm = c / 3430 Hz / 0.3" in editor.target_label.text()
    bem.setCurrentIndex(3)
    bem.activated.emit(3)
    assert not custom.isHidden() and bem.currentText() == "Custom"
    custom.setValue(7)
    assert host.snapshot().source["bem_elements_per_wavelength"] == 7
    assert host.snapshot().source["elements_per_wavelength"] == 6  # the FEM one is untouched


def test_a_design_without_a_bem_fineness_shows_its_fem_value_in_the_bem_dropdown(cad_editor):
    editor, host = cad_editor
    bem, custom = editor.bem_fineness, editor.spins["bem_elements_per_wavelength"]

    def reload(**changes):
        source = {k: v for k, v in host.snapshot().source.items() if k != "bem_elements_per_wavelength"} | changes
        snapshot = host.update_source(source, expected_revision=host.snapshot().revision)
        editor.apply_source(snapshot.source, snapshot.revision)

    reload(elements_per_wavelength=5.0)
    assert bem.currentText() == "BEM Medium" and custom.isHidden()
    reload(elements_per_wavelength=4.0)
    assert bem.currentText() == "Custom" and custom.value() == 4.0
    assert "bem_elements_per_wavelength" not in host.snapshot().source  # shown, not written

    snapshot = host.update_source(
        host.snapshot().source | {"bem_elements_per_wavelength": 12.0}, expected_revision=host.snapshot().revision
    )
    editor.apply_source(snapshot.source, snapshot.revision)  # a host revision
    assert bem.currentText() == "BEM Fine"


def chamber_back(editor):
    """The chamber's 100 x 100 mm back face."""
    (face,) = [
        f["id"] for f in editor.geometry["faces"] if abs(f["normal"][2]) > 0.999 and f["area"] == pytest.approx(0.01)
    ]
    return face


def test_group_whose_faces_are_gone_is_listed_as_needing_review(cad_editor, qapp, tmp_path):
    editor, host = cad_editor
    editor.set_step_path(str(strip_names(vented_box(tmp_path / "named.step"), tmp_path / "stripped.step")))
    wait(qapp, lambda: not editor.loading)
    editor.pick_face(chamber_back(editor))
    editor.group_selection("Back")
    assert editor.groups_needing_review() == []

    editor.set_step_path(str(strip_names(vented_box(tmp_path / "named.step", depth=230), tmp_path / "deep.step")))
    wait(qapp, lambda: not editor.loading)

    assert editor.groups_needing_review() == ["Back"]
    row = next(r for r in range(editor.groups_table.rowCount()) if editor.groups_table.item(r, 0).text() == "Back")
    assert "needs review" in editor.groups_table.item(row, 1).text()


@pytest.mark.parametrize("answer", [True, False])
def test_changed_step_file_offers_to_re_mesh_when_the_application_regains_focus(
    cad_editor, monkeypatch, qapp, tmp_path, answer
):
    editor, host = cad_editor
    asked, generated = [], []
    monkeypatch.setattr(editor, "ask_remesh", lambda path: asked.append(path) or answer)
    monkeypatch.setattr(host, "generate", lambda *, expected_revision: generated.append(expected_revision))

    def regain_focus():
        qapp.applicationStateChanged.emit(Qt.ApplicationState.ApplicationActive)

    regain_focus()
    assert asked == []

    vented_box(tmp_path / "box.step", depth=181)
    regain_focus()
    regain_focus()

    assert asked == [host.snapshot().source["step_path"]]  # asked once per change
    assert generated == ([host.snapshot().revision] if answer else [])


def regain_focus(qapp):
    qapp.applicationStateChanged.emit(Qt.ApplicationState.ApplicationActive)


def test_design_without_a_step_file_never_offers_to_re_mesh(blank_cad_editor, monkeypatch, qapp, tmp_path):
    editor, _ = blank_cad_editor
    asked = []
    monkeypatch.setattr(editor, "ask_remesh", lambda path: asked.append(path))

    monkeypatch.chdir(tmp_path)  # an empty path must not stand for the working directory
    regain_focus(qapp)

    assert asked == []


def test_step_file_edited_while_it_loads_offers_to_re_mesh(cad_editor, monkeypatch, qapp, tmp_path):
    editor, host = cad_editor
    asked = []
    monkeypatch.setattr(editor, "ask_remesh", lambda path: asked.append(path))
    editor_module = sys.modules[type(editor).__module__]
    read = editor_module.preview_in_worker

    def edited_during_load(step_path, *args, **kwargs):
        stat = os.stat(step_path)
        os.utime(step_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
        return read(step_path, *args, **kwargs)

    monkeypatch.setattr(editor_module, "preview_in_worker", edited_during_load)
    editor.set_step_path(str(vented_box(tmp_path / "other.step")))
    wait(qapp, lambda: not editor.loading)
    regain_focus(qapp)

    assert asked == [host.snapshot().source["step_path"]]


def mouse(widget, button, press, release=None, modifier=Qt.KeyboardModifier.NoModifier):
    """Press at ``press`` and release at ``release`` (default: same point) through the widget's real events."""
    QTest.mousePress(widget, button, modifier, QPoint(*press))
    if release is not None:
        QTest.mouseMove(widget, QPoint(*release))
    QTest.mouseRelease(widget, button, modifier, QPoint(*(release or press)))


@pytest.fixture
def faces_under_mouse(cad_editor, monkeypatch):
    """The VTK pick (offscreen it hits nothing) answers ``faces[0]`` and records the display positions."""
    editor, _ = cad_editor
    faces, positions = [], []
    monkeypatch.setattr(editor.viewport, "face_at", lambda x, y: positions.append((x, y)) or faces[0])
    return faces, positions


def test_a_click_in_the_view_picks_the_face_and_a_drag_does_not(cad_editor, faces_under_mouse, qapp):
    editor, _ = cad_editor
    faces, positions = faces_under_mouse
    first, second = cabinet_faces(editor, lambda f: abs(abs(f["normal"][0]) - 1) < 1e-9)
    plotter = editor.viewport.plotter
    plotter.resize(200, 100)
    ratio = plotter.devicePixelRatioF()

    faces.append(first)
    mouse(plotter, Qt.MouseButton.LeftButton, (20, 30))
    assert editor.selected_faces() == {first}
    assert positions == [(round(20 * ratio), round(100 * ratio) - round(30 * ratio) - 1)]  # VTK display: y up

    faces[0] = second
    mouse(plotter, Qt.MouseButton.LeftButton, (20, 30), (60, 50))  # a drag rotates the view
    assert editor.selected_faces() == {first}
    mouse(plotter, Qt.MouseButton.LeftButton, (5, 5), modifier=Qt.KeyboardModifier.ControlModifier)
    assert editor.selected_faces() == {first, second}
    mouse(plotter, Qt.MouseButton.LeftButton, (5, 5), modifier=Qt.KeyboardModifier.ShiftModifier)
    assert editor.selected_faces() == {first}
    mouse(plotter, Qt.MouseButton.LeftButton, (5, 5), (5, 40))  # also lets go of the modifier for later tests
    assert editor.selected_faces() == {first}
    assert "1 face selected" in editor.hint.text()


def row_of(editor, name):
    table = editor.groups_table
    return next(r for r in range(table.rowCount()) if table.item(r, 0).text() == name)


def test_walls_row_comes_first_then_the_interface_rows_then_the_face_groups(cad_editor, qapp):
    editor, _ = cad_editor
    (interface,) = editor.geometry["interfaces"]
    driver = [f["id"] for f in editor.geometry["faces"] if f["step_name"] == "Driver"]
    walls = {f["id"] for f in editor.geometry["faces"]} - set(driver) - set(interface["face_ids"])

    assert list(all_rows(editor).items()) == [
        ("walls", len(walls)),
        ("Interface 1 Port", len(interface["face_ids"])),
        ("Driver", 1),
    ]
    editor.groups_table.selectRow(row_of(editor, "walls"))
    assert editor.selected_faces() == walls
    assert editor.group_name.text() == "walls"
    assert editor.group_button.text() == 'Add to "walls"'

    row = row_of(editor, "Interface 1 Port")
    editor.groups_table.selectRow(row)
    assert editor.selected_faces() == set(interface["face_ids"])
    assert editor.group_name.text() == "walls"  # an interface group is never the current group
    assert not editor.groups_table.item(row, 0).flags() & Qt.ItemFlag.ItemIsEditable


def test_single_faces_are_added_to_the_current_group_and_removed_back_to_walls(cad_editor):
    editor, host = cad_editor
    first, second = cabinet_faces(editor, lambda f: abs(abs(f["normal"][0]) - 1) < 1e-9)
    editor.pick_face(first)
    editor.group_name.setText("Sides")
    assert editor.group_button.text() == "New group"
    editor.group_button.click()
    editor.groups_table.selectRow(row_of(editor, "Sides"))
    assert editor.group_name.text() == "Sides" and editor.group_button.text() == 'Add to "Sides"'

    editor.pick_face(second)
    editor.group_button.click()
    assert group_rows(editor)["Sides"] == 2
    editor.pick_face(first)
    editor.remove_button.click()
    assert [len(g["faces"]) for g in host.snapshot().source["face_groups"]] == [1]

    editor.pick_face(second)
    editor.set_current_group("walls")
    editor.group_button.click()  # adding to walls removes from the group
    assert host.snapshot().source["face_groups"] == []
    assert "Sides" not in group_rows(editor)


def test_each_face_group_is_drawn_in_its_own_colour_with_a_swatch(cad_editor, qapp):
    editor, _ = cad_editor
    first, second = cabinet_faces(editor, lambda f: abs(abs(f["normal"][0]) - 1) < 1e-9)
    editor.add_to_group("A", {first})
    editor.add_to_group("B", {second})
    editor.tree.clearSelection()
    qapp.processEvents()

    table = editor.groups_table
    swatches = {
        table.item(r, 0).text(): table.item(r, 0).data(Qt.ItemDataRole.DecorationRole).name()
        for r in range(table.rowCount())
    }
    assert list(swatches) == ["walls", "Interface 1 Port", "A", "B", "Driver"]
    assert len(set(swatches.values())) == len(swatches)
    assert swatches["walls"] == "#c8ccd2"  # the plain face colour
    wait(qapp, lambda: sorted(editor.viewport.plotter.colours) == sorted(swatches.values()))

    editor.pick_face(first)
    wait(qapp, lambda: "#e8a33d" in editor.viewport.plotter.colours)
    assert "#e8a33d" in editor.viewport.plotter.colours  # the selection on top
    assert swatches["A"] not in editor.viewport.plotter.colours


def test_the_face_under_the_mouse_is_pale_orange_or_dark_orange_when_selected(cad_editor, faces_under_mouse, qapp):
    editor, _ = cad_editor
    first, second = cabinet_faces(editor, lambda f: abs(abs(f["normal"][0]) - 1) < 1e-9)
    faces, _ = faces_under_mouse
    plotter = editor.viewport.plotter
    editor.pick_face(first)

    faces.append(second)
    hover(plotter, (20, 30))
    qapp.processEvents()
    assert editor.hovered_face() == second
    assert {"#e8a33d", "#f6cf98"} <= set(plotter.colours)  # selected orange, hovered pale orange

    faces[0] = first
    hover(plotter, (25, 30))
    qapp.processEvents()
    assert "#b8620f" in plotter.colours and "#f6cf98" not in plotter.colours  # hovered and selected: dark orange

    faces[0] = None
    hover(plotter, (30, 30))
    qapp.processEvents()
    assert editor.hovered_face() is None and "#b8620f" not in plotter.colours


def hover(widget, position):
    """Move the mouse over ``widget`` with no button pressed (QTest's moves do not arrive offscreen)."""
    point = QPointF(*position)
    event = QMouseEvent(
        QEvent.Type.MouseMove,
        point,
        widget.mapToGlobal(point),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(widget, event)


@pytest.mark.parametrize("row", ["body", "Volume", "Faces"])
def test_clicking_a_body_volume_or_faces_row_selects_all_the_body_faces(cad_editor, row):
    editor, _ = cad_editor
    chamber = body_items(editor)["Chamber"]

    editor.tree.itemClicked.emit(chamber if row == "body" else child_row(chamber, row), 0)

    assert editor.selected_faces() == body_faces(editor, "Chamber")


def test_placement_change_updates_the_interfaces_and_their_face_groups(cad_editor, qapp, tmp_path):
    editor, _ = cad_editor
    # A half model written 30 mm off X=0: the port mouth only opens into the cabinet once moved onto X=0.
    editor.set_step_path(str(vented_box(tmp_path / "half.step", symmetry="x", shift_mm=(30, 0, 0))))
    wait(qapp, lambda: not editor.loading)
    assert "Interface 1 Port" not in all_rows(editor)

    editor.translation_spins[0].setValue(-30)

    assert not editor.loading
    assert [i["name"] for i in editor.geometry["interfaces"]] == ["Interface 1 Port"]
    assert list(all_rows(editor))[:2] == ["walls", "Interface 1 Port"]


def menu_entries(menu):
    """``{text: action}`` of ``menu``."""
    return {action.text(): action for action in menu.actions()}


def test_right_click_menu_moves_a_face_to_a_group_and_back_to_walls(cad_editor, monkeypatch):
    editor, host = cad_editor
    first, second = cabinet_faces(editor, lambda f: abs(abs(f["normal"][0]) - 1) < 1e-9)

    entries = menu_entries(editor.face_menu(first))
    assert list(entries) == ['Move to "Driver"', "New group…", "Mesh size"]  # in walls; no interface group
    entries['Move to "Driver"'].trigger()
    assert [(g["name"], len(g["faces"])) for g in host.snapshot().source["face_groups"]] == [("Driver", 1)]
    assert list(menu_entries(editor.face_menu(first))) == ["New group…", "Move to walls", "Mesh size"]

    monkeypatch.setattr(editor, "ask_group_name", lambda: None)  # cancelled
    status = editor.status.text()
    menu_entries(editor.face_menu(second))["New group…"].trigger()
    assert editor.status.text() == status
    monkeypatch.setattr(editor, "ask_group_name", lambda: "Sides")
    editor.pick_face(second)
    menu_entries(editor.face_menu(second))["New group…"].trigger()
    assert group_rows(editor) == {"Driver": 2, "Sides": 1}

    menu_entries(editor.face_menu(second))["Move to walls"].trigger()  # second is selected: acts on the selection
    assert group_rows(editor) == {"Driver": 2}
    menu_entries(editor.face_menu(first))["Move to walls"].trigger()  # first is not: acts on the face
    assert group_rows(editor) == {"Driver": 1}

    editor.add_to_group("Driver", {first})
    editor.pick_face(first)
    editor.pick_face(second, add=True)
    entries = menu_entries(editor.face_menu(second))  # one in Driver, one in walls: every move does something
    assert list(entries) == ['Move to "Driver"', "New group…", "Move to walls", "Mesh size"]
    entries['Move to "Driver"'].trigger()
    assert group_rows(editor) == {"Driver": 3}


def test_right_click_outside_the_selection_selects_the_face_names_its_group_and_points_at_its_row(
    cad_editor, faces_under_mouse, monkeypatch, qapp
):
    editor, _ = cad_editor
    interface_face = editor.geometry["interfaces"][0]["face_ids"][0]
    assert [a.text() for a in editor.face_menu(interface_face).actions()][0] == "Rename interface…"
    first, second = cabinet_faces(editor, lambda f: abs(abs(f["normal"][0]) - 1) < 1e-9)
    editor.pick_face(first)
    editor.pick_face(second, add=True)

    faces, _ = faces_under_mouse
    seen = []

    def opened(_pos):
        rows = {editor.groups_table.item(i.row(), 0).text() for i in editor.groups_table.selectedItems()}
        opacities = {kw.get("opacity", 1.0) for mesh, kw in editor.viewport.plotter.meshes if "name" not in kw}
        seen.append((editor.selected_faces(), editor.status.text(), rows, editor.viewport.plotter.colours, opacities))

    menu = SimpleNamespace(exec=opened, deleteLater=lambda: None)
    monkeypatch.setattr(editor, "face_menu", lambda face: menu)
    faces.append(second)
    mouse(editor.viewport.plotter, Qt.MouseButton.RightButton, (20, 30))  # on a selected face: the selection stays
    wait(qapp, lambda: seen)
    faces[0] = interface_face
    mouse(editor.viewport.plotter, Qt.MouseButton.RightButton, (20, 30), (60, 30))  # a drag pans the view
    mouse(editor.viewport.plotter, Qt.MouseButton.RightButton, (20, 30))  # outside it: the face becomes it
    wait(qapp, lambda: len(seen) == 2)

    (kept, *_), (selected, status, rows, colours, opacities) = seen
    assert kept == {first, second}
    assert selected == {interface_face}
    assert status == f"Face {interface_face} · Group 'Interface 1 Port'"
    assert rows == {"Interface 1 Port"}
    assert "#e8a33d" in colours and opacities == {1.0}  # the selection in orange, nothing see-through
    assert editor.selected_faces() == {interface_face}  # the row only points


def test_an_interface_is_renamed_from_its_face_menu_and_keeps_its_size(cad_editor, monkeypatch):
    editor, host = cad_editor
    interface_face = editor.geometry["interfaces"][0]["face_ids"][0]
    editor.set_face_group_size("Interface 1 Port", FINE)

    def rename(new_name):
        monkeypatch.setattr(editor, "ask_interface_name", lambda name: new_name)
        menu_entries(editor.face_menu(interface_face))["Rename interface…"].trigger()

    rename("Port mouth")
    source = host.snapshot().source
    assert source["interface_names"] == {"Interface 1 Port": "Port mouth"}
    assert "Port mouth" in all_rows(editor) and "Interface 1 Port" not in all_rows(editor)
    assert size_combo(editor, "Port mouth").currentText() == "FEM Fine"  # the size is on its faces

    rename("Driver")
    assert "'Driver' is taken" in editor.status.text()
    assert host.snapshot().source["interface_names"] == {"Interface 1 Port": "Port mouth"}


def group_the_future_interface_face(editor, qapp, tmp_path):
    """Read a waveguide whose closed shell is exterior and group the shell's mouth face as "Interface"; that face
    becomes an interface face once the shell is an air body."""
    step = waveguide(tmp_path / "wg.step", surface_model=True)
    editor.set_step_path(str(step))
    wait(qapp, lambda: not editor.loading)
    combo = editor.kind_combos["Closed shell 1"]
    combo.setCurrentIndex(combo.findData("exterior"))
    wait(qapp, lambda: not editor.loading)
    assert not editor.geometry["interfaces"]
    as_air = inspect_step(step, closed_shells={"Closed shell 1": "air"})
    (mouth,) = [f.centre for f in as_air.faces if f.id in as_air.interfaces[0].face_ids]
    # The shell's mouth face, which the air volume will share with the exterior.
    shell = editor.geometry["closed_shells"][0]["face_ids"]
    (face,) = [f["id"] for f in editor.geometry["faces"] if f["id"] in shell and np.allclose(f["centre"], mouth)]
    editor.pick_face(face)
    editor.group_selection("Interface")


def test_faces_grouped_before_they_became_interface_faces_stay_in_the_interface_while_it_is_detected(
    cad_editor, monkeypatch, qapp, tmp_path
):
    editor, host = cad_editor
    group_the_future_interface_face(editor, qapp, tmp_path)

    editor.kind_combos["Closed shell 1"].setCurrentIndex(0)  # air body
    wait(qapp, lambda: not editor.loading)

    (interface,) = editor.geometry["interfaces"]
    assert interface["name"] == "Interface 1 Closed shell 1"
    assert group_rows(editor)["Interface"] == 0  # its only face is in the interface
    assert f"stays in interface '{interface['name']}'" in editor.status.text()
    assert [g["name"] for g in host.snapshot().source["face_groups"]] == ["Interface"]  # stored as picked

    editor.kind_combos["Closed shell 1"].setCurrentIndex(1)  # exterior again: no interface
    wait(qapp, lambda: not editor.loading)
    assert group_rows(editor)["Interface"] == 1


def test_the_interface_note_shows_once_per_change_not_on_every_read(cad_editor, qapp, tmp_path):
    editor, _ = cad_editor
    group_the_future_interface_face(editor, qapp, tmp_path)
    editor.kind_combos["Closed shell 1"].setCurrentIndex(0)  # air body
    wait(qapp, lambda: not editor.loading)
    assert "stays in interface" in editor.status.text()
    editor.status.setText("")

    editor._load(editor._key)  # read again, as after the file changed on disk
    wait(qapp, lambda: not editor.loading)

    assert "stays in interface" not in editor.status.text()


def test_a_read_dropping_a_size_and_holding_an_interface_face_shows_both_notes(cad_editor, qapp, tmp_path):
    editor, host = cad_editor
    group_the_future_interface_face(editor, qapp, tmp_path)
    gone = {"step_name": None, "centre": (9.0, 9.0, 9.0), "area": 1.0, "normal": (0.0, 0.0, 1.0)}
    entry = {"face": gone | {"surface_type": "Plane"}, "size_m": 0.01}
    snapshot = host.update_source(
        host.snapshot().source | {"face_sizes": [entry]}, expected_revision=host.snapshot().revision
    )
    editor.apply_source(snapshot.source, snapshot.revision)

    editor.kind_combos["Closed shell 1"].setCurrentIndex(0)  # air body: a read
    wait(qapp, lambda: not editor.loading)

    assert "1 face size override dropped" in editor.status.text()
    assert "stays in interface" in editor.status.text()


def test_an_invalid_stored_face_size_stays_shown_beside_the_interface_note(cad_editor, qapp, tmp_path):
    editor, host = cad_editor
    group_the_future_interface_face(editor, qapp, tmp_path)
    face = {"step_name": None, "centre": (0.0, 0.0, 0.0), "area": 1.0, "normal": (0.0, 0.0, 1.0)}
    entry = {"face": face | {"surface_type": "Plane"}, "size_m": -1}
    snapshot = host.update_source(
        host.snapshot().source | {"face_sizes": [entry]}, expected_revision=host.snapshot().revision
    )
    editor.apply_source(snapshot.source, snapshot.revision)

    editor.kind_combos["Closed shell 1"].setCurrentIndex(0)  # air body: a read with an interface note
    wait(qapp, lambda: not editor.loading)

    assert "Invalid face size: Size override edge length must be a positive number." in editor.status.text()
    assert "stays in interface" in editor.status.text()


def test_picking_and_grouping_keep_the_camera(cad_editor, faces_under_mouse, qapp):
    editor, _ = cad_editor
    faces, _ = faces_under_mouse
    face = cabinet_faces(editor, lambda f: True)[0]
    faces.append(face)
    plotter = editor.viewport.plotter
    plotter.calls.clear()

    mouse(plotter, Qt.MouseButton.LeftButton, (20, 30))
    qapp.processEvents()
    menu_entries(editor.face_menu(face))['Move to "Driver"'].trigger()
    qapp.processEvents()
    editor.rotation_spins[1].setValue(90)  # placement changes keep the camera too
    qapp.processEvents()

    assert "add_mesh" in plotter.calls
    assert not {"reset_camera", "view_xy", "add_mesh_resetting_camera"} & set(plotter.calls)


def test_the_view_opens_looking_along_minus_z_and_resets_on_loading_a_file(cad_editor, qapp, tmp_path):
    editor, _ = cad_editor
    plotter = editor.viewport.plotter
    assert "view_xy" in plotter.calls  # Z towards the viewer, Y up, X to the right
    plotter.calls.clear()

    editor.set_step_path(str(vented_box(tmp_path / "other.step", depth=150)))
    wait(qapp, lambda: not editor.loading)

    assert "view_xy" in plotter.calls


def unmeshable_rows(editor):
    return [editor.unmeshable_list.item(i).text() for i in range(editor.unmeshable_list.count())]


@pytest.fixture
def ball_editor(cad_editor, qapp, tmp_path):
    """The CAD Import design with the bored ball loaded: its torus face cannot be meshed."""
    editor, host = cad_editor
    editor.set_step_path(str(bored_ball(tmp_path / "ball.step")))
    wait(qapp, lambda: not editor.loading)
    return editor, host


def torus_face(editor):
    (face,) = [f["id"] for f in editor.geometry["faces"] if f["surface_type"] == "Torus"]
    return face


def listed_position(editor):
    """The position, mm, of the one Unmeshable faces row."""
    (row,) = unmeshable_rows(editor)
    return np.array([int(v) for v in re.search(r"near \((-?\d+), (-?\d+), (-?\d+)\) mm", row).groups()])


def test_a_good_file_shows_no_unmeshable_faces_list(cad_editor):
    editor, _ = cad_editor

    assert editor.unmeshable_box.isHidden() and unmeshable_rows(editor) == []
    assert editor.warnings.isHidden()


def test_unmeshable_faces_are_listed_with_their_name_type_body_position_and_reason(ball_editor):
    editor, _ = ball_editor
    torus = torus_face(editor)
    reason = f"Impossible to mesh periodic surface {torus}; area reads as negative"

    assert not editor.unmeshable_box.isHidden()
    (row,) = unmeshable_rows(editor)
    assert re.fullmatch(rf"Face {torus} · Torus · Ball · near \(-?\d+, -?\d+, -?\d+\) mm · {re.escape(reason)}", row)
    # The ball (r 12.7 mm) sits at the origin: its face centre lies within it.
    centre = next(f["centre"] for f in editor.geometry["faces"] if f["id"] == torus)
    assert np.all(np.abs(listed_position(editor)) <= 12)
    assert np.allclose(listed_position(editor), np.array(centre) * 1000, atol=0.5)
    assert not editor.warnings.isHidden()
    assert "1 face cannot be meshed; see the list." in editor.warnings.text()
    assert "cannot be meshed" not in editor.info.text()
    item = face_item(editor, torus)
    assert item.toolTip(0) == f"Cannot be meshed: {reason}" and not item.icon(0).isNull()
    other = face_item(editor, cabinet_faces(editor, lambda f: True)[0])
    assert other.toolTip(0) == "" and other.icon(0).isNull()


def test_the_listed_position_follows_placement_without_reading(ball_editor, monkeypatch):
    editor, _ = ball_editor
    reads = record_reads(editor, monkeypatch)
    before = listed_position(editor)

    editor.translation_spins[0].setValue(100)

    assert reads == [] and not editor.loading
    assert np.array_equal(listed_position(editor), before + [100, 0, 0])


def test_re_reading_a_fixed_file_empties_and_hides_the_list(ball_editor, qapp, tmp_path):
    editor, _ = ball_editor

    status = editor.status.text(), editor.status.palette().color(QPalette.ColorRole.WindowText)
    editor.set_step_path(str(vented_box(tmp_path / "fixed.step")))
    wait(qapp, lambda: not editor.loading)

    assert editor.unmeshable_box.isHidden() and unmeshable_rows(editor) == []
    assert editor.warnings.isHidden()
    assert (editor.status.text(), editor.status.palette().color(QPalette.ColorRole.WindowText)) == status


def laid_out(widget):
    """The widgets of ``widget``'s layout, sub-layouts included, top to bottom and left to right."""

    def walk(layout):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget() is not None:
                yield item.widget()
            elif item.layout() is not None:
                yield from walk(item.layout())

    return list(walk(widget.layout()))


def in_order(widgets, expected):
    return [w for w in widgets if any(w is e for e in expected)] == expected


def test_the_view_comes_first_then_warnings_unmeshable_faces_placement_and_the_tree_below_its_splitter(ball_editor):
    editor, _ = ball_editor
    splitter = editor.viewport.parentWidget()

    assert isinstance(splitter, QSplitter) and splitter.widget(0) is editor.viewport
    assert in_order(
        laid_out(editor.widget),
        [editor.step_path, editor.info, splitter, editor.groups_box, editor.sizing_box, editor.status],
    )
    panel = splitter.widget(1)
    assert not isinstance(panel, QScrollArea)  # no outer scroll box: each list scrolls on its own
    assert in_order(
        laid_out(panel),
        [editor.warnings, editor.unmeshable_box, editor.placement_box, editor.tree_box],
    )
    assert in_order(laid_out(editor.tree_box), [editor.hint, editor.tree, editor.show_all_button])
    assert editor.fineness in laid_out(editor.sizing_box)


def shown(editor, qapp, height=1330):
    """Show ``editor``'s window ``height`` pixels tall and let it lay out."""
    window = editor.widget.window()
    window.resize(1900, height)
    window.show()
    for _ in range(20):
        qapp.processEvents()


def test_the_face_groups_table_takes_only_its_rows_and_the_tree_the_spare_height(ball_editor, qapp):
    editor, _ = ball_editor
    table, tree = editor.groups_table, editor.tree

    shown(editor, qapp)

    rows = sum(table.rowHeight(r) for r in range(table.rowCount()))
    assert rows <= table.viewport().height() <= rows + table.rowHeight(0)  # a scroll area keeps a minimum height
    assert tree.viewport().height() >= 4 * tree.sizeHintForRow(0)


def test_every_unmeshable_face_row_shows(ball_editor, qapp):
    editor, _ = ball_editor
    listed = editor.unmeshable_list
    for _ in range(5):  # more rows than a list shows unless sized to them
        listed.addItem(listed.item(0).text())

    shown(editor, qapp)

    assert listed.viewport().height() >= listed.count() * listed.sizeHintForRow(0)


@pytest.mark.parametrize("long", ["unmeshable", "groups"])
def test_a_long_unmeshable_list_or_face_groups_table_shows_8_rows_then_scrolls(ball_editor, qapp, long):
    editor, _ = ball_editor
    listed, table = editor.unmeshable_list, editor.groups_table
    if long == "unmeshable":
        for _ in range(19):
            listed.addItem(listed.item(0).text())
        view, row = listed, listed.sizeHintForRow(0)
    else:
        table.setRowCount(20)
        view, row = table, table.rowHeight(0)

    shown(editor, qapp)

    assert 8 * row <= view.viewport().height() < 9 * row
    assert view.verticalScrollBar().maximum() > 0
    if long == "unmeshable":  # 8 taller face group rows leave the tree its minimum in this window
        assert editor.tree.viewport().height() >= 4 * editor.tree.sizeHintForRow(0)


def test_size_columns_fit_their_dropdowns_and_face_group_names_take_the_spare_width(ball_editor, qapp, tmp_path):
    editor, _ = ball_editor
    shown(editor, qapp)
    # Read while shown: a dropdown shown before Global is named would keep the width it had without the name.
    editor.set_step_path(str(bored_ball(tmp_path / "again.step")))
    wait(qapp, lambda: not editor.loading)
    shown(editor, qapp)

    def fitted(combo):
        unshown = QComboBox()
        unshown.addItems([combo.itemText(i) for i in range(combo.count())])
        return unshown.sizeHint().width()

    assert editor.tree.columnWidth(1) >= fitted(editor.tree.itemWidget(editor.tree.topLevelItem(0), 1))
    table = editor.groups_table
    assert table.horizontalHeader().sectionResizeMode(0) == QHeaderView.ResizeMode.Stretch
    assert table.columnWidth(1) >= table.sizeHintForColumn(1)
    assert table.columnWidth(2) >= fitted(table.cellWidget(0, 2))


def test_warnings_are_red_in_both_themes_and_the_status_line_is_not(ball_editor, qapp):
    from blab.ui.theme import apply_application_theme

    editor, _ = ball_editor
    try:
        for theme in ("light", "dark"):
            apply_application_theme(theme)
            background = qapp.palette().color(QPalette.ColorRole.Window)
            red = editor.warnings.palette().color(QPalette.ColorRole.WindowText)
            text = editor.status.palette().color(QPalette.ColorRole.WindowText)
            assert red.red() > 2 * max(red.green(), red.blue()), f"{theme}: {red.name()} is not red"
            assert abs(red.lightness() - background.lightness()) > 80, f"{theme}: {red.name()} is not legible"
            assert abs(text.red() - text.blue()) < 40, f"{theme}: the status line is {text.name()}"
    finally:
        apply_application_theme("system")


def test_other_faces_are_grouped_and_sized_while_unmeshable_faces_are_listed(ball_editor):
    editor, host = ball_editor
    face = cabinet_faces(editor, lambda f: True)[0]

    editor.pick_face(face)
    editor.group_selection("Front")
    editor.set_face_size({face}, FINE)

    source = host.snapshot().source
    assert [(g["name"], len(g["faces"])) for g in source["face_groups"]] == [("Front", 1)]
    assert [e["elements_per_wavelength"] for e in source["face_sizes"]] == [6]
    assert editor.selected_faces() == {face} and len(unmeshable_rows(editor)) == 1


def checked_faces(editor):
    """The faces whose every tree row is checked."""
    rows = {}
    for body in body_items(editor).values():
        for item in children_items(faces_row(body)):
            rows.setdefault(item_face(item), []).append(item.checkState(0) == Qt.CheckState.Checked)
    return {face for face, checks in rows.items() if all(checks)}


def drawn_faces(editor):
    """``(faces, opacity, colour)`` of each face mesh drawn, its faces read back from its triangles."""
    preview = editor._preview
    face_of = {tuple(t): f for t, f in zip(preview["triangles"].tolist(), preview["face_ids"].tolist(), strict=True)}
    return [
        (
            {face_of[tuple(t)] for t in mesh.faces.reshape(-1, 4)[:, 1:].tolist()},
            kwargs.get("opacity", 1.0),
            kwargs["color"],
        )
        for mesh, kwargs in editor.viewport.plotter.meshes
        if not mesh.n_lines and "name" not in kwargs
    ]


def grey_faces(editor):
    """The faces drawn translucent but not red: the context of a shown face (grey unless selected or hovered)."""
    return set().union(*(faces for faces, opacity, _ in drawn_faces(editor) if opacity < 1)) - red_faces(editor)


def red_faces(editor):
    """The faces drawn translucent red: unmeshable faces with fill triangles."""
    red = sys.modules[type(editor).__module__].UNMESHABLE_COLOUR
    return set().union(*(faces for faces, opacity, colour in drawn_faces(editor) if opacity < 1 and colour == red))


def opaque_faces(editor):
    return set().union(*(faces for faces, opacity, _ in drawn_faces(editor) if opacity == 1))


def test_showing_an_unmeshable_face_greys_the_other_bodies_and_selects_and_marks_it(ball_editor):
    editor, _ = ball_editor
    every = {f["id"] for f in editor.geometry["faces"]}
    torus = torus_face(editor)

    editor.show_unmeshable(torus)

    assert editor.visible_faces() == every == checked_faces(editor)
    assert grey_faces(editor) == body_faces(editor, "Cabinet") - body_faces(editor, "Ball")
    assert opaque_faces(editor) == body_faces(editor, "Ball") - {torus}  # the torus has no triangles
    assert len(unmeshable_outlines(editor)) == 1
    assert editor.selected_faces() == {torus}
    assert editor.status.text() == f"Face {torus} · Group 'walls'"
    assert {i.row() for i in editor.groups_table.selectedItems()} == {row_of(editor, "walls")}


def test_showing_a_face_shared_by_an_air_body_and_the_exterior_greys_only_the_other_bodies(cad_editor, qapp, tmp_path):
    editor, _ = cad_editor
    editor.set_step_path(str(vented_box(tmp_path / "covered.step", covered=True)))
    wait(qapp, lambda: not editor.loading)
    (mouth,) = [f["id"] for f in editor.geometry["faces"] if set(f["bodies"]) == {"Port", "Cabinet"}]

    editor.show_unmeshable(mouth)

    shown = body_faces(editor, "Port") | body_faces(editor, "Cabinet")
    assert grey_faces(editor) == body_faces(editor, "Chamber") - shown
    assert opaque_faces(editor) == shown
    assert editor.selected_faces() == {mouth}


def test_showing_another_face_moves_the_grey_and_show_all_and_clear_end_it(ball_editor):
    editor, _ = ball_editor
    every = {f["id"] for f in editor.geometry["faces"]}
    side = cabinet_faces(editor, lambda f: True)[0]
    editor.show_unmeshable(torus_face(editor))

    editor.show_unmeshable(side)
    # The torus has no triangles to grey.
    assert grey_faces(editor) == body_faces(editor, "Ball") - body_faces(editor, "Cabinet") - {torus_face(editor)}
    assert editor.selected_faces() == {side}

    editor.show_all_button.click()
    assert grey_faces(editor) == set() and editor.visible_faces() == every == checked_faces(editor)
    assert editor.selected_faces() == {side}

    editor.show_unmeshable(torus_face(editor))
    editor.clear_button.click()
    assert grey_faces(editor) == set() and editor.visible_faces() == every == checked_faces(editor)
    assert editor.selected_faces() == set()


def test_picking_a_grey_face_selects_it_and_keeps_the_grey(ball_editor, qapp):
    editor, _ = ball_editor
    editor.show_unmeshable(torus_face(editor))
    grey = grey_faces(editor)
    side = cabinet_faces(editor, lambda f: f["id"] in grey)[0]

    editor.pick_face(side)
    qapp.processEvents()  # the render timer

    assert editor.selected_faces() == {side}
    assert side in grey_faces(editor) and grey_faces(editor) == grey


def test_unchecked_faces_stay_hidden_not_grey(ball_editor):
    editor, _ = ball_editor
    row = children_items(faces_row(body_items(editor)["Cabinet"]))[0]
    side = item_face(row)
    row.setCheckState(0, Qt.CheckState.Unchecked)

    editor.show_unmeshable(torus_face(editor))

    assert side not in editor.visible_faces() and side not in grey_faces(editor)
    assert grey_faces(editor) == body_faces(editor, "Cabinet") - body_faces(editor, "Ball") - {side}


def test_clicking_an_unmeshable_faces_row_shows_its_face(ball_editor):
    editor, _ = ball_editor
    torus = torus_face(editor)
    row = editor.unmeshable_list.item(0)

    editor.unmeshable_list.itemClicked.emit(row)

    assert grey_faces(editor) == body_faces(editor, "Cabinet") - body_faces(editor, "Ball")
    assert editor.selected_faces() == {torus}
    assert editor.status.text() == f"Face {torus} · Group 'walls'"


def unmeshable_outlines(editor):
    """The orange outlines drawn for selected faces without triangles."""
    return [mesh for mesh, kwargs in editor.viewport.plotter.meshes if mesh.n_lines and kwargs.get("line_width") == 4]


def test_showing_an_unmeshable_face_without_curves_neither_outlines_nor_frames_it(ball_editor, monkeypatch):
    editor, _ = ball_editor
    torus = torus_face(editor)
    editor._preview["edges"][torus] = np.empty((0, 9, 3))
    monkeypatch.setattr(editor.viewport, "frame", lambda points: pytest.fail("framed nothing"))

    editor.show_unmeshable(torus)

    assert editor.selected_faces() == {torus} and unmeshable_outlines(editor) == []


def fill_torus(editor):
    """Give the ball's torus one fill triangle (m), as reading a face the fill can approximate does."""
    preview, torus = editor._preview, torus_face(editor)
    start = len(preview["points"])
    preview["points"] = np.vstack([preview["points"], [[0.0, 0.0, 0.0127], [0.001, 0.0, 0.0127], [0.0, 0.001, 0.0127]]])
    preview["triangles"] = np.vstack([preview["triangles"], [[start, start + 1, start + 2]]])
    preview["face_ids"] = np.append(preview["face_ids"], torus)
    return torus


def test_an_unmeshable_face_with_fill_triangles_is_drawn_red_and_outlined_when_selected(ball_editor, qapp):
    editor, _ = ball_editor
    torus = fill_torus(editor)

    editor.show_all()
    assert red_faces(editor) == {torus} and unmeshable_outlines(editor) == []
    assert torus not in opaque_faces(editor)

    editor.pick_face(torus)
    qapp.processEvents()  # the render timer
    assert red_faces(editor) == {torus} and len(unmeshable_outlines(editor)) == 1

    editor.show_unmeshable(torus)
    assert red_faces(editor) == {torus} and len(unmeshable_outlines(editor)) == 1
    assert grey_faces(editor) == body_faces(editor, "Cabinet") - body_faces(editor, "Ball")


def test_an_unmeshable_face_around_a_shown_face_stays_red_not_grey(ball_editor):
    editor, _ = ball_editor
    torus = fill_torus(editor)

    editor.show_unmeshable(cabinet_faces(editor, lambda f: True)[0])

    assert red_faces(editor) == {torus}
    assert grey_faces(editor) == body_faces(editor, "Ball") - body_faces(editor, "Cabinet") - {torus}


def test_showing_an_unmeshable_face_without_curves_frames_its_fill(ball_editor, monkeypatch):
    editor, _ = ball_editor
    torus = fill_torus(editor)
    editor._preview["edges"][torus] = np.empty((0, 9, 3))
    framed = []
    monkeypatch.setattr(editor.viewport, "frame", framed.append)

    editor.show_unmeshable(torus)

    (points,) = framed
    assert np.allclose(np.reshape(points, (-1, 3)), [[0.0, 0.0, 0.0127], [0.001, 0.0, 0.0127], [0.0, 0.001, 0.0127]])


@pytest.fixture
def menus(cad_editor, monkeypatch):
    """The entries of each menu opened."""
    shown = []
    patch_menu_exec(cad_editor[0], monkeypatch, lambda menu: shown.append(list(menu_entries(menu))))
    return shown


def patch_menu_exec(editor, monkeypatch, opened):
    """Call ``opened(menu)`` instead of showing each menu the editor opens."""
    monkeypatch.setattr(sys.modules[type(editor).__module__], "_exec", lambda menu, _pos: opened(menu))


def test_right_clicking_a_face_faces_or_face_group_row_opens_the_face_menu_on_its_faces(cad_editor, menus):
    editor, _ = cad_editor
    first, second = cabinet_faces(editor, lambda f: abs(abs(f["normal"][0]) - 1) < 1e-9)
    editor.pick_face(first)
    editor.pick_face(second, add=True)

    editor.show_row_menu(face_item(editor, second), QPoint())  # in the selection: it stays
    assert editor.selected_faces() == {first, second}
    editor.show_row_menu(faces_row(body_items(editor)["Port"]), QPoint())
    assert editor.selected_faces() == body_faces(editor, "Port")
    editor.show_group_menu(row_of(editor, "Driver"), QPoint())
    driver = next(f["id"] for f in editor.geometry["faces"] if f["step_name"] == "Driver")
    assert editor.selected_faces() == {driver}
    editor.show_group_menu(row_of(editor, "Interface 1 Port"), QPoint())
    editor.show_row_menu(child_row(body_items(editor)["Chamber"], "Volume"), QPoint())
    editor.show_row_menu(body_items(editor)["Chamber"], QPoint())  # a body row has no menu

    assert menus == [
        ['Move to "Driver"', "New group…", "Mesh size"],
        ['Move to "Driver"', "New group…", "Mesh size"],  # its interface face is skipped, the rest are walls
        ["New group…", "Move to walls", "Mesh size"],
        ["Rename interface…", "Mesh size"],
        ["Mesh size"],
    ]


def test_volume_row_menu_sizes_the_volume(cad_editor, monkeypatch):
    editor, host = cad_editor

    def choose_coarse(menu):
        menu_entries(menu_entries(menu)["Mesh size"].menu())["FEM Coarse"].trigger()

    patch_menu_exec(editor, monkeypatch, choose_coarse)
    editor.show_row_menu(child_row(body_items(editor)["Port"], "Volume"), QPoint())

    assert host.snapshot().source["volume_sizes"] == {"Port": COARSE}
    assert host.snapshot().source["face_sizes"] == []


def test_open_step_starts_in_the_folder_last_used(blank_cad_editor, monkeypatch, qapp, tmp_path):
    from blab.ui import file_dialogs

    editor, host = blank_cad_editor
    (tmp_path / "models").mkdir()
    step = vented_box(tmp_path / "models" / "box.step")
    starts = []

    def get_open_file_name(parent, title, start, file_filter):
        starts.append(start)
        return str(step), file_filter

    monkeypatch.setattr(file_dialogs.QFileDialog, "getOpenFileName", get_open_file_name)
    editor.browse_button.click()
    wait(qapp, lambda: not editor.loading)
    assert host.snapshot().source["step_path"] == str(step)

    main_window = editor.widget.window()
    main_window.add_generator_document()  # another CAD design starts where the last file was
    other, _ = main_window._provider_editors[main_window.active_generator_document().id]
    other.browse_button.click()

    assert Path(starts[1]) == step.parent


def save_and_reopen(main_window, qapp, path):
    """Save the project to ``path``, start a new one, open ``path`` again and return the reopened CAD editor."""
    document_id = main_window.active_generator_document().id
    assert main_window.project_workflow._save_project_to_path(path)
    main_window.new_project()
    assert all(d.id != document_id for d in main_window.project.generator_documents)
    main_window.project_workflow._load_project_from_path(path)
    wait(qapp, lambda: main_window.active_generator_document().id == document_id)
    editor, host = main_window._provider_editors[document_id]
    wait(qapp, lambda: not editor.loading and editor.geometry is not None)
    return editor, host


def test_a_saved_project_reopens_with_the_design_and_its_meshes(cad_editor, main_window, monkeypatch, qapp, tmp_path):
    from blab.ui.main_window import geometry_workflow

    editor, host = cad_editor
    monkeypatch.setattr(geometry_workflow, "GENERATED_GEOMETRY_ROOT", tmp_path / "runs")
    errors = []
    monkeypatch.setattr(main_window, "show_error", lambda *args: errors.append(args))
    side = cabinet_faces(editor, lambda f: abs(abs(f["normal"][0]) - 1) < 1e-9)[0]
    editor.pick_face(side)
    editor.group_selection("Side")
    editor.set_face_size({side}, COARSE)
    editor.rename_interface("Interface 1 Port", "Port mouth")
    editor.rotation_spins[1].setValue(90)
    editor.translation_spins[0].setValue(25)
    host.generate(expected_revision=host.snapshot().revision)
    wait(qapp, lambda: not main_window.geometry_controller.active)
    assert not errors
    source = host.snapshot().source
    rows = all_rows(editor)
    meshes = [m.mesh_path for m in main_window.active_generator_document().artifact.meshes]

    editor, host = save_and_reopen(main_window, qapp, tmp_path / "speaker.blab.json")

    assert not errors
    assert host.snapshot().source == source
    assert all_rows(editor) == rows
    assert [s.value() for s in editor.rotation_spins] == [0, 90, 0]
    assert [s.value() for s in editor.translation_spins] == [25, 0, 0]
    assert size_combo(editor, "Side").currentText() == "FEM Coarse"
    assert [m.mesh_path for m in main_window.active_generator_document().artifact.meshes] == meshes
    entries = main_window._mesh_config_dialog_entries()
    assert len(entries) == 2 and all(e.locked for e in entries)
    assert len(PhysicalSystemCompiler().compile(main_window.project.physical_system).interfaces) == 1
    assert not main_window.geometry_controller.active  # restored, not re-meshed


def test_a_saved_project_whose_meshes_are_gone_reopens_with_the_design_and_generate_rebuilds_them(
    cad_editor, main_window, monkeypatch, qapp, tmp_path
):
    from blab.ui.main_window import geometry_workflow

    editor, host = cad_editor
    monkeypatch.setattr(geometry_workflow, "GENERATED_GEOMETRY_ROOT", tmp_path / "runs")
    errors = []
    monkeypatch.setattr(main_window, "show_error", lambda *args: errors.append(args))
    editor.translation_spins[0].setValue(25)
    host.generate(expected_revision=host.snapshot().revision)
    wait(qapp, lambda: not main_window.geometry_controller.active)
    assert not errors
    source = host.snapshot().source
    for mesh in main_window.active_generator_document().artifact.meshes:
        Path(mesh.mesh_path).unlink()

    editor, host = save_and_reopen(main_window, qapp, tmp_path / "speaker.blab.json")

    assert host.snapshot().source == source
    host.generate(expected_revision=host.snapshot().revision)
    wait(qapp, lambda: not main_window.geometry_controller.active)
    assert not errors
    assert all(Path(m.mesh_path).is_file() for m in main_window.active_generator_document().artifact.meshes)
    assert len(PhysicalSystemCompiler().compile(main_window.project.physical_system).interfaces) == 1


def test_bundled_cad_import_is_enabled_once_also_for_existing_installs():
    settings = MemorySettings({"providers/enabled": [VENTED], "providers/vented_enclosure_introduced": True})
    assert load_gui_preferences(settings).enabled_geometry_providers == (VENTED, PROVIDER)
    preferences = load_gui_preferences(settings)
    preferences.enabled_geometry_providers = (VENTED,)
    save_gui_preferences(settings, preferences)
    assert load_gui_preferences(settings).enabled_geometry_providers == (VENTED,)
