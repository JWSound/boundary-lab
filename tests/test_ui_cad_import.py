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
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QHeaderView

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

        @property
        def colours(self):
            return [kwargs["color"] for mesh, kwargs in self.meshes if not mesh.n_lines]

        def clear(self):
            self.calls.append("clear")
            self.meshes = []

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


def test_tree_gives_spare_width_to_names_and_fits_the_size_column(cad_editor):
    editor, _ = cad_editor
    header = editor.tree.header()

    assert not header.stretchLastSection()
    assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Stretch
    assert header.sectionResizeMode(1) == QHeaderView.ResizeMode.ResizeToContents


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


def test_closed_shell_is_asked_once_and_stays_editable(cad_editor, qapp, monkeypatch, tmp_path):
    editor, host = cad_editor
    asked = []
    monkeypatch.setattr(editor, "ask_closed_shell", lambda name: asked.append(name) or "air")

    editor.set_step_path(str(waveguide(tmp_path / "wg.step", surface_model=True)))
    wait(qapp, lambda: not editor.loading)

    assert asked == ["Closed shell 1"]
    assert host.snapshot().source["closed_shells"] == {"Closed shell 1": "air"}
    assert list(body_items(editor)) == ["Closed shell 1", "Exterior surface 1"]
    combo = editor.closed_shell_combos["Closed shell 1"]
    assert combo.currentData() == "air"

    combo.setCurrentIndex(combo.findData("exterior"))
    wait(qapp, lambda: not editor.loading)

    assert host.snapshot().source["closed_shells"] == {"Closed shell 1": "exterior"}
    assert "Closed shell 1" not in body_items(editor)
    assert editor.closed_shell_combos["Closed shell 1"].currentData() == "exterior"
    assert asked == ["Closed shell 1"]
    editor.set_operation_state(SimpleNamespace(active=True))
    assert not editor.closed_shell_combos["Closed shell 1"].isEnabled()


def test_closed_shell_is_highlighted_and_framed_while_asked(cad_editor, qapp, monkeypatch, tmp_path):
    editor, _ = cad_editor
    plotter = editor.viewport.plotter
    plotter.camera_position = "before"
    seen = []

    def ask(name):
        (shell,) = editor.geometry["closed_shells"]
        opacities = [kwargs.get("opacity", 1.0) for mesh, kwargs in plotter.meshes if not mesh.n_lines]
        seen.append((editor.highlighted_faces(), set(shell["face_ids"]), opacities, "reset_camera" in plotter.calls))
        plotter.camera_position = "framed"
        return "exterior"

    monkeypatch.setattr(editor, "ask_closed_shell", ask)
    editor.set_step_path(str(waveguide(tmp_path / "wg.step", surface_model=True)))
    wait(qapp, lambda: not editor.loading)

    ((highlighted, shell_faces, opacities, framed),) = seen
    assert highlighted == shell_faces and shell_faces
    assert min(opacities) < 1 and max(opacities) == 1  # the rest see-through
    assert framed
    assert editor.highlighted_faces() == set()
    assert all(kwargs.get("opacity", 1.0) == 1 for _, kwargs in plotter.meshes)
    assert plotter.camera_position == "before"


def test_preview_is_smooth_shaded_with_face_outlines_and_no_triangle_edges(cad_editor):
    editor, _ = cad_editor
    meshes = editor.viewport.plotter.meshes
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
    return [f["id"] for f in editor.geometry["faces"] if f["bodies"] == ["Cabinet"] and predicate(f)]


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
        "Sides",
        "Driver",  # from the STEP file
        "Port opening",  # the interface
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
    editor.set_face_group_size("B", 0.008)
    editor.delete_face_group("B")
    assert [g["name"] for g in host.snapshot().source["face_groups"]] == ["A"]
    assert host.snapshot().source["face_group_sizes_m"] == {}


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


def test_grouping_waits_for_the_reload_after_an_axis_change(cad_editor, qapp):
    editor, host = cad_editor
    editor.pick_face(cabinet_faces(editor, lambda f: True)[0])

    editor.forward_combo.setCurrentIndex(editor.forward_combo.findData("-Y"))
    editor.up_combo.setCurrentIndex(editor.up_combo.findData("+Z"))
    assert editor.loading
    editor.group_selection("Early")

    assert host.snapshot().source.get("face_groups", []) == []
    wait(qapp, lambda: not editor.loading)
    assert "Driver" in group_rows(editor)


@pytest.mark.parametrize(
    ("name", "pick", "message"),
    [
        ("Port opening", "cabinet", "Face group name 'Port opening' is reserved"),
        ("Mouth", "interface", "Face group 'Mouth' contains face .* of interface 'Port opening'"),
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
    editor.set_face_group_size("Driver", 0.008)

    editor.delete_face_group("Driver")

    assert host.snapshot().source["face_groups"] == []
    assert host.snapshot().source["face_group_sizes_m"] == {"Driver": 0.008}
    assert group_rows(editor) == {"Driver": 1}


@pytest.mark.parametrize("text", ["inf", "nan", "-5"])
def test_size_override_must_be_finite_and_positive(cad_editor, text):
    editor, host = cad_editor

    body_items(editor)["Chamber"].setText(1, text)

    assert host.snapshot().source.get("body_sizes_m", {}) == {}
    assert "positive number" in editor.status.text()
    assert body_items(editor)["Chamber"].text(1) == ""


def test_size_overrides_for_bodies_and_face_groups_edit_the_source(cad_editor):
    editor, host = cad_editor

    editor.set_face_group_size("Driver", 0.008)
    editor.set_body_size("Chamber", 0.04)

    source = host.snapshot().source
    assert source["face_group_sizes_m"] == {"Driver": 0.008}
    assert source["body_sizes_m"] == {"Chamber": 0.04}
    driver_row = next(
        r for r in range(editor.groups_table.rowCount()) if editor.groups_table.item(r, 0).text() == "Driver"
    )
    assert editor.groups_table.item(driver_row, 2).text() == "8"
    assert body_items(editor)["Chamber"].text(1) == "40"

    editor.groups_table.item(driver_row, 2).setText("")  # typing in the table clears the override
    body_items(editor)["Chamber"].setText(1, "25")
    source = host.snapshot().source
    assert source["face_group_sizes_m"] == {}
    assert source["body_sizes_m"] == {"Chamber": 0.025}


def test_orientation_pickers_reorient_the_preview(cad_editor, qapp, tmp_path):
    editor, host = cad_editor
    editor.set_step_path(str(vented_box(tmp_path / "turned.step", cad_axes=("-Y", "+Z"))))
    wait(qapp, lambda: not editor.loading)
    assert "120.0 x 210.0 x 120.0 mm" in editor.info.text()

    editor.up_combo.setCurrentIndex(editor.up_combo.findData("+Z"))  # +Z up while forward is +Z: invalid
    wait(qapp, lambda: not editor.loading)
    assert "perpendicular" in editor.info.text()
    editor.forward_combo.setCurrentIndex(editor.forward_combo.findData("-Y"))
    wait(qapp, lambda: not editor.loading)

    assert (host.snapshot().source["forward"], host.snapshot().source["up"]) == ("-Y", "+Z")
    assert "120.0 x 120.0 x 210.0 mm" in editor.info.text()


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
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QTest

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


def test_walls_row_comes_first_and_interface_rows_last(cad_editor):
    editor, _ = cad_editor
    (interface,) = editor.geometry["interfaces"]
    driver = [f["id"] for f in editor.geometry["faces"] if f["step_name"] == "Driver"]
    walls = {f["id"] for f in editor.geometry["faces"]} - set(driver) - set(interface["face_ids"])

    assert list(all_rows(editor).items()) == [
        ("walls", len(walls)),
        ("Driver", 1),
        ("Port opening", len(interface["face_ids"])),
    ]
    editor.groups_table.selectRow(row_of(editor, "walls"))
    assert editor.selected_faces() == walls
    assert editor.group_name.text() == "walls"
    assert editor.group_button.text() == 'Add to "walls"'

    row = row_of(editor, "Port opening")
    editor.groups_table.selectRow(row)
    assert editor.selected_faces() == set(interface["face_ids"])
    assert editor.group_name.text() == "walls"  # an interface group is never the current group
    assert not editor.groups_table.item(row, 0).flags() & Qt.ItemFlag.ItemIsEditable
    editor.groups_table.item(row, 2).setText("4")
    editor.groups_table.item(row_of(editor, "walls"), 2).setText("30")
    assert editor.source["face_group_sizes_m"] == {"Port opening": 0.004, "walls": 0.03}


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
    assert list(swatches) == ["walls", "A", "B", "Driver", "Port opening"]
    assert len(set(swatches.values())) == len(swatches)
    assert swatches["walls"] == "#c8ccd2"  # the plain face colour
    assert sorted(editor.viewport.plotter.colours) == sorted(swatches.values())

    editor.pick_face(first)
    qapp.processEvents()
    assert "#e8a33d" in editor.viewport.plotter.colours  # the selection on top
    assert swatches["A"] not in editor.viewport.plotter.colours


def menu_entries(menu):
    """``{text: action}`` of ``menu``, submenu entries as ``"Add to/<text>"``."""
    entries = {}
    for action in menu.actions():
        if action.menu() is not None:
            entries |= {f"{action.text()}/{a.text()}": a for a in action.menu().actions()}
        else:
            entries[action.text()] = action
    return entries


def test_right_click_menu_adds_a_face_to_a_group_and_removes_it(cad_editor, monkeypatch):
    editor, host = cad_editor
    first, second = cabinet_faces(editor, lambda f: abs(abs(f["normal"][0]) - 1) < 1e-9)

    entries = menu_entries(editor.face_menu(first))
    assert list(entries) == ["Add to/walls", "Add to/Driver", "Add to/New group…"]  # no interface group
    entries["Add to/Driver"].trigger()
    assert [(g["name"], len(g["faces"])) for g in host.snapshot().source["face_groups"]] == [("Driver", 1)]

    monkeypatch.setattr(editor, "ask_group_name", lambda: None)  # cancelled
    status = editor.status.text()
    menu_entries(editor.face_menu(second))["Add to/New group…"].trigger()
    assert editor.status.text() == status
    monkeypatch.setattr(editor, "ask_group_name", lambda: "Sides")
    editor.pick_face(second)
    menu_entries(editor.face_menu(second))["Add to/New group…"].trigger()
    assert group_rows(editor) == {"Driver": 2, "Sides": 1}

    entries = menu_entries(editor.face_menu(first))  # first is not selected: the menu acts on it alone
    assert 'Remove from "Driver"' in entries
    entries['Remove from "Driver"'].trigger()
    assert group_rows(editor) == {"Driver": 1, "Sides": 1}

    editor.pick_face(first)
    editor.pick_face(second, add=True)
    menu_entries(editor.face_menu(second))["Add to/Driver"].trigger()  # second is selected: acts on the selection
    assert group_rows(editor) == {"Driver": 3}


def test_interface_faces_get_no_menu_and_a_right_click_opens_the_face_menu(
    cad_editor, faces_under_mouse, monkeypatch, qapp
):
    editor, _ = cad_editor
    interface_face = editor.geometry["interfaces"][0]["face_ids"][0]
    assert editor.face_menu(interface_face).actions() == []

    faces, _ = faces_under_mouse
    faces.append(interface_face)
    asked = []
    monkeypatch.setattr(editor, "face_menu", lambda face: asked.append(face) or SimpleNamespace(actions=lambda: []))
    mouse(editor.viewport.plotter, Qt.MouseButton.RightButton, (20, 30), (60, 30))  # a drag pans the view
    mouse(editor.viewport.plotter, Qt.MouseButton.RightButton, (20, 30))
    wait(qapp, lambda: asked)
    assert asked == [interface_face]


def test_picking_and_grouping_keep_the_camera(cad_editor, faces_under_mouse, qapp):
    editor, _ = cad_editor
    faces, _ = faces_under_mouse
    face = cabinet_faces(editor, lambda f: True)[0]
    faces.append(face)
    plotter = editor.viewport.plotter
    plotter.calls.clear()

    mouse(plotter, Qt.MouseButton.LeftButton, (20, 30))
    qapp.processEvents()
    menu_entries(editor.face_menu(face))["Add to/Driver"].trigger()
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
