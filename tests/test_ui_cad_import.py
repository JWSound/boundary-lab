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
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDockWidget, QHeaderView

from blab.cad_import import inspect_step
from blab.generators import catalog as catalog_module
from blab.physical_compiler import PhysicalSystemCompiler
from blab.ui.settings import load_gui_preferences, save_gui_preferences
from cad_fixtures import named_sides, strip_names, vented_box, waveguide
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


def test_loaded_step_lists_bodies_and_faces_with_size_and_symmetry(cad_editor):
    editor, host = cad_editor

    assert host.snapshot().source["step_path"].endswith("box.step")
    bodies = body_items(editor)
    assert list(bodies) == ["Chamber", "Port", "Cabinet"]
    assert "Driver" in children(bodies["Chamber"])  # STEP face name
    assert all(name.startswith("Face ") for name in children(bodies["Cabinet"]))
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
    driver = next(chamber.child(i) for i in range(chamber.childCount()) if chamber.child(i).text(0) == "Driver")
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
    assert "covers a face" in editor.info.text()
    items = {}
    for body in body_items(editor).values():
        for i in range(body.childCount()):
            items.setdefault(body.child(i).data(0, Qt.ItemDataRole.UserRole), []).append(body.child(i))
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
    assert "Target element size 10.0 mm" in editor.target_label.text()
    editor.spins["growth_rate"].setValue(1.2)
    assert host.snapshot().source["growth_rate"] == 1.2

    editor.set_operation_state(SimpleNamespace(active=True))
    assert not editor.step_path.isEnabled() and not editor.spins["growth_rate"].isEnabled()
    assert not editor.fineness.isEnabled()
    editor.set_operation_state(SimpleNamespace(active=False))
    assert editor.step_path.isEnabled() and editor.spins["growth_rate"].isEnabled()


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
    bodies["Chamber"].child(0).setSelected(True)
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
    outline_edges = sys.modules[type(editor).__module__].outline_edges  # the provider package loads with the editor
    triangles = np.array([[0, 1, 2], [0, 2, 3], [1, 4, 2]])  # face 1: a square of two triangles; face 2 beside it

    edges = outline_edges(triangles, np.array([1, 1, 2]))

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


def test_picking_into_a_group_moves_faces_and_groups_can_be_deleted(cad_editor):
    editor, host = cad_editor
    first, second = cabinet_faces(editor, lambda f: abs(abs(f["normal"][0]) - 1) < 1e-9)
    editor.pick_face(first)
    editor.group_selection("A")
    editor.pick_face(second)
    editor.group_selection("A")  # an existing name adds the selection
    editor.pick_face(first)
    editor.group_selection("B")

    groups = {g["name"]: len(g["faces"]) for g in host.snapshot().source["face_groups"]}
    assert groups == {"A": 1, "B": 1}
    editor.set_face_group_size("B", FINE)
    editor.delete_face_group("B")
    assert [g["name"] for g in host.snapshot().source["face_groups"]] == ["A"]
    assert host.snapshot().source["face_group_sizes"] == {}


def all_rows(editor):
    table = editor.groups_table
    return {table.item(r, 0).text(): int(table.item(r, 1).text()) for r in range(table.rowCount())}


def group_rows(editor):
    """The face group rows, without walls and the interface groups."""
    interfaces = {i["name"] for i in editor.geometry["interfaces"]}
    return {name: count for name, count in all_rows(editor).items() if name not in {"walls", *interfaces}}


def test_picking_one_of_several_faces_sharing_a_step_name_groups_only_that_face(cad_editor, qapp, tmp_path):
    editor, host = cad_editor
    editor.set_step_path(str(vented_box(tmp_path / "sides.step", face_names=named_sides)))
    wait(qapp, lambda: not editor.loading)
    left, right = cabinet_faces(editor, lambda f: abs(abs(f["normal"][0]) - 1) < 1e-9)

    editor.pick_face(left)
    editor.group_selection("Left")
    editor.pick_face(right)
    editor.group_selection("Right")

    assert {g["name"]: len(g["faces"]) for g in host.snapshot().source["face_groups"]} == {"Left": 1, "Right": 1}
    assert group_rows(editor) == {"Left": 1, "Right": 1, "Side": 2, "Driver": 1}


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
    run_worker = editor_module.run_worker
    reads = []

    def recorded(job, output, *args, **kwargs):
        reads.append(Path(job["step_path"]).name)
        run_worker(job, output, *args, **kwargs)

    monkeypatch.setattr(editor_module, "run_worker", recorded)
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

    def failing(job, output, *args, **kwargs):
        raise ValueError("Could not read STEP file broken.step")

    monkeypatch.setattr(editor_module, "run_worker", failing)
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


def test_step_names_that_are_reserved_form_no_face_group(cad_editor, qapp, tmp_path):
    editor, _ = cad_editor
    names = {-210: "walls", -200: "Driver"}
    editor.set_step_path(str(vented_box(tmp_path / "walls.step", face_names=lambda c: names.get(round(c[2])))))
    wait(qapp, lambda: not editor.loading)

    assert group_rows(editor) == {"Driver": 1}


def test_deleting_a_picked_group_keeps_the_size_of_a_step_named_group_of_that_name(cad_editor):
    editor, host = cad_editor
    editor.pick_face(cabinet_faces(editor, lambda f: True)[0])
    editor.group_selection("Driver")
    editor.set_face_group_size("Driver", FINE)

    editor.delete_face_group("Driver")

    assert host.snapshot().source["face_groups"] == []
    assert host.snapshot().source["face_group_sizes"] == {"Driver": FINE}
    assert group_rows(editor) == {"Driver": 1}


FINE = {"elements_per_wavelength": 6}


def choose_size(view, index, text, qapp):
    """Choose ``text`` in the Mesh size dropdown of cell ``index`` of ``view``, as a user does."""
    view.edit(index)
    combo = view.indexWidget(index)
    assert [combo.itemText(i) for i in range(combo.count())] == ["", "Coarse", "Medium", "Fine", "Custom…"]
    combo.setCurrentIndex(combo.findText(text))
    combo.activated.emit(combo.currentIndex())
    qapp.processEvents()  # applied once the dropdown has closed


def size_cell(editor, name):
    """The Mesh size cell of body ``name`` in the tree, or of face group ``name`` in the list."""
    bodies = body_items(editor)
    if name in bodies:
        return editor.tree, editor.tree.indexFromItem(bodies[name], 1)
    return editor.groups_table, editor.groups_table.model().index(row_of(editor, name), 2)


def test_mesh_size_dropdowns_set_levels_and_custom_sizes_on_bodies_and_face_groups(cad_editor, monkeypatch, qapp):
    editor, host = cad_editor
    monkeypatch.setattr(editor, "ask_size_mm", lambda: 25.0)

    choose_size(*size_cell(editor, "Driver"), "Medium", qapp)
    choose_size(*size_cell(editor, "Chamber"), "Custom…", qapp)

    source = host.snapshot().source
    assert source["face_group_sizes"] == {"Driver": {"elements_per_wavelength": 4}}
    assert source["body_sizes"] == {"Chamber": {"size_m": 0.025}}
    assert editor.groups_table.item(row_of(editor, "Driver"), 2).text() == "Medium"
    assert body_items(editor)["Chamber"].text(1) == "25 mm"

    monkeypatch.setattr(editor, "ask_size_mm", lambda: None)  # cancelled
    choose_size(*size_cell(editor, "Chamber"), "Custom…", qapp)
    choose_size(*size_cell(editor, "Driver"), "", qapp)  # empty: inherited again
    source = host.snapshot().source
    assert source["face_group_sizes"] == {}
    assert source["body_sizes"] == {"Chamber": {"size_m": 0.025}}


def face_item(editor, face_id):
    return next(
        item for body in body_items(editor).values() for item in children_items(body) if item_face(item) == face_id
    )


def children_items(item):
    return [item.child(i) for i in range(item.childCount())]


def item_face(item):
    return item.data(0, Qt.ItemDataRole.UserRole)


def test_face_rows_show_their_own_size_or_where_the_inherited_one_comes_from(cad_editor, main_window, qapp):
    editor, _ = cad_editor
    main_window.freq_max_spin.setValue(3430)
    wait(qapp, lambda: "3430" in editor.target_label.text())
    driver = next(f["id"] for f in editor.geometry["faces"] if f["step_name"] == "Driver")
    side, other = cabinet_faces(editor, lambda f: True)[:2]

    editor.set_face_group_size("Driver", {"elements_per_wavelength": 4})
    editor.set_body_size("Cabinet", {"size_m": 0.03})
    editor.set_face_size({side}, {"elements_per_wavelength": 2})

    assert face_item(editor, side).text(1) == "Coarse"
    assert face_item(editor, driver).text(1) == ""
    assert face_item(editor, driver).toolTip(1) == "Inherits face group 'Driver': Medium (25 mm)"
    assert face_item(editor, other).toolTip(1) == "Inherits body 'Cabinet': 30 mm"
    assert body_items(editor)["Chamber"].toolTip(1) == "Inherits the mesh fineness: Fine (16.7 mm)"


def test_mesh_size_from_the_face_menu_sets_a_face_override_on_the_selection(cad_editor, monkeypatch):
    editor, host = cad_editor
    first, second = cabinet_faces(editor, lambda f: abs(abs(f["normal"][0]) - 1) < 1e-9)
    interface_face = editor.geometry["interfaces"][0]["face_ids"][0]
    editor.pick_face(first)
    editor.pick_face(second, add=True)

    menu = editor.face_menu(second)
    assert list(menu_entries(menu)) == ['Add to "Driver"', "New group…", "Move to walls", "Mesh size"]
    sizes = menu_entries(menu_entries(menu)["Mesh size"].menu())
    assert list(sizes) == ["Coarse", "Medium", "Fine", "Custom…", "Clear"]
    sizes["Fine"].trigger()

    entries = host.snapshot().source["face_sizes"]
    assert [e["elements_per_wavelength"] for e in entries] == [6, 6]
    assert {round(abs(e["face"]["centre"][0]), 3) for e in entries} == {0.06}  # stored like face group faces
    assert face_item(editor, first).text(1) == "Fine"

    interface_entries = menu_entries(editor.face_menu(interface_face))
    assert list(interface_entries) == ["Rename interface…", "Mesh size"]
    menu_entries(interface_entries["Mesh size"].menu())["Coarse"].trigger()  # the face, outside the selection
    assert len(host.snapshot().source["face_sizes"]) == 3
    menu_entries(menu_entries(editor.face_menu(first))["Mesh size"].menu())["Clear"].trigger()
    assert len(host.snapshot().source["face_sizes"]) == 1


def test_face_overrides_whose_faces_are_gone_are_dropped_with_a_note(cad_editor, qapp, tmp_path):
    editor, host = cad_editor
    editor.set_step_path(str(strip_names(vented_box(tmp_path / "named.step"), tmp_path / "stripped.step")))
    wait(qapp, lambda: not editor.loading)
    editor.set_face_size({chamber_back(editor)}, FINE)

    editor.set_step_path(str(strip_names(vented_box(tmp_path / "named.step", depth=230), tmp_path / "deep.step")))
    wait(qapp, lambda: not editor.loading)

    assert host.snapshot().source["face_sizes"] == []
    assert "1 face size override dropped" in editor.status.text()


def test_existing_designs_keep_their_mm_overrides_and_elements_per_wavelength(cad_editor, qapp):
    editor, host = cad_editor
    legacy = {
        "elements_per_wavelength": 5.0,
        "body_sizes_m": {"Chamber": 0.04},
        "face_group_sizes_m": {"Driver": 0.008},
    }
    snapshot = host.update_source(host.snapshot().source | legacy, expected_revision=host.snapshot().revision)
    editor.apply_source(snapshot.source, snapshot.revision)

    assert editor.fineness.currentText() == "Custom" and not editor.spins["elements_per_wavelength"].isHidden()
    assert body_items(editor)["Chamber"].text(1) == "40 mm"
    assert editor.groups_table.item(row_of(editor, "Driver"), 2).text() == "8 mm"

    editor.set_body_size("Port", FINE)

    source = host.snapshot().source
    assert source["body_sizes"] == {"Chamber": {"size_m": 0.04}, "Port": FINE}
    assert "body_sizes_m" not in source
    assert source["face_group_sizes_m"] == {"Driver": 0.008}  # untouched until a face group size changes


def test_mesh_fineness_levels_set_elements_per_wavelength_and_name_the_frequency_they_hold(
    cad_editor, main_window, qapp
):
    editor, host = cad_editor
    main_window.freq_max_spin.setValue(3000)
    editor.spins["curvature_elements_per_circle"].setValue(13)  # an edit fetches the project context
    wait(qapp, lambda: "3000" in editor.target_label.text())
    fineness, custom = editor.fineness, editor.spins["elements_per_wavelength"]

    assert [fineness.itemText(i) for i in range(fineness.count())] == ["Coarse", "Medium", "Fine", "Custom"]
    assert fineness.currentText() == "Fine" and custom.isHidden()  # new designs
    assert "6 per wavelength up to 1000 Hz" in fineness.itemData(0, Qt.ItemDataRole.ToolTipRole)
    assert "up to 2000 Hz" in fineness.itemData(1, Qt.ItemDataRole.ToolTipRole)

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


def chamber_back(editor):
    """The chamber's 100 x 100 mm back face."""
    (face,) = [
        f["id"] for f in editor.geometry["faces"] if abs(f["normal"][2]) > 0.999 and f["area"] == pytest.approx(0.01)
    ]
    return face


def test_group_whose_faces_are_gone_needs_review_and_re_picking_repairs_it(cad_editor, qapp, tmp_path):
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
    editor.pick_face(chamber_back(editor))
    editor.group_selection("Back")

    assert editor.groups_needing_review() == []
    assert [len(g["faces"]) for g in host.snapshot().source["face_groups"]] == [1]
    assert group_rows(editor)["Back"] == 1


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
    run_worker = editor_module.run_worker

    def edited_during_load(job, *args, **kwargs):
        stat = os.stat(job["step_path"])
        os.utime(job["step_path"], ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
        return run_worker(job, *args, **kwargs)

    monkeypatch.setattr(editor_module, "run_worker", edited_during_load)
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
    choose_size(*size_cell(editor, "Interface 1 Port"), "Fine", qapp)
    choose_size(*size_cell(editor, "walls"), "Coarse", qapp)
    assert editor.source["face_group_sizes"] == {"Interface 1 Port": FINE, "walls": {"elements_per_wavelength": 2}}


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


def test_clicking_a_body_row_selects_all_its_faces(cad_editor):
    editor, _ = cad_editor
    chamber = body_items(editor)["Chamber"]

    editor.tree.itemClicked.emit(chamber, 0)

    assert editor.selected_faces() == {face["id"] for face in editor.geometry["faces"] if "Chamber" in face["bodies"]}


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


def test_right_click_menu_adds_a_face_to_a_group_and_removes_it(cad_editor, monkeypatch):
    editor, host = cad_editor
    first, second = cabinet_faces(editor, lambda f: abs(abs(f["normal"][0]) - 1) < 1e-9)

    entries = menu_entries(editor.face_menu(first))
    assert list(entries) == ['Add to "Driver"', "New group…", "Move to walls", "Mesh size"]  # no interface group
    entries['Add to "Driver"'].trigger()
    assert [(g["name"], len(g["faces"])) for g in host.snapshot().source["face_groups"]] == [("Driver", 1)]

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

    editor.pick_face(first)
    editor.pick_face(second, add=True)
    menu_entries(editor.face_menu(second))['Add to "Driver"'].trigger()
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
    assert source["face_group_sizes"] == {"Port mouth": FINE}
    assert "Port mouth" in all_rows(editor) and "Interface 1 Port" not in all_rows(editor)

    rename("Driver")
    assert "'Driver' is taken" in editor.status.text()
    assert host.snapshot().source["interface_names"] == {"Interface 1 Port": "Port mouth"}

    rename("")  # back to the detected name
    assert host.snapshot().source["interface_names"] == {}
    assert "Interface 1 Port" in all_rows(editor)


def test_faces_grouped_before_they_became_interface_faces_move_to_the_interface(
    cad_editor, monkeypatch, qapp, tmp_path
):
    editor, host = cad_editor
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

    editor.kind_combos["Closed shell 1"].setCurrentIndex(0)  # air body
    wait(qapp, lambda: not editor.loading)

    (interface,) = editor.geometry["interfaces"]
    assert interface["name"] == "Interface 1 Closed shell 1"
    assert host.snapshot().source["face_groups"] == []  # emptied, so gone
    assert "Interface" not in group_rows(editor)
    assert f"stays in interface '{interface['name']}'" in editor.status.text()


def test_picking_and_grouping_keep_the_camera(cad_editor, faces_under_mouse, qapp):
    editor, _ = cad_editor
    faces, _ = faces_under_mouse
    face = cabinet_faces(editor, lambda f: True)[0]
    faces.append(face)
    plotter = editor.viewport.plotter
    plotter.calls.clear()

    mouse(plotter, Qt.MouseButton.LeftButton, (20, 30))
    qapp.processEvents()
    menu_entries(editor.face_menu(face))['Add to "Driver"'].trigger()
    qapp.processEvents()

    assert "add_mesh" in plotter.calls
    assert not {"reset_camera", "add_mesh_resetting_camera"} & set(plotter.calls)


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


def test_bundled_cad_import_is_enabled_once_also_for_existing_installs():
    settings = MemorySettings({"providers/enabled": [VENTED], "providers/vented_enclosure_introduced": True})
    assert load_gui_preferences(settings).enabled_geometry_providers == (VENTED, PROVIDER)
    preferences = load_gui_preferences(settings)
    preferences.enabled_geometry_providers = (VENTED,)
    save_gui_preferences(settings, preferences)
    assert load_gui_preferences(settings).enabled_geometry_providers == (VENTED,)
