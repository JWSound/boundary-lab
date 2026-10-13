"""The CAD Import editor inside the main window's design dock."""

from __future__ import annotations

from time import monotonic, sleep
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget

from blab.generators import catalog as catalog_module
from blab.physical_compiler import PhysicalSystemCompiler
from blab.ui.settings import load_gui_preferences, save_gui_preferences
from cad_fixtures import vented_box
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


@pytest.fixture
def cad_editor(main_window, monkeypatch, qapp, tmp_path):
    """A CAD Import design in the dock, with the vented box loaded and previewed."""
    import pyvistaqt

    monkeypatch.setattr(pyvistaqt, "QtInteractor", _stub_qt_interactor_class())
    catalog = catalog_module.ProviderCatalog([ROOT], enabled=[PROVIDER])
    catalog.scan()
    monkeypatch.setattr(catalog_module, "_catalog", catalog)
    main_window.preferences.default_geometry_provider = PROVIDER
    main_window.add_generator_document()
    editor, host = main_window._provider_editors[main_window.active_generator_document().id]
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
    assert not editor.mesh_button.isEnabled() and not editor.spins["growth_rate"].isEnabled()
    editor.set_operation_state(SimpleNamespace(active=False))
    assert editor.mesh_button.isEnabled() and not editor.cancel_button.isEnabled()


def test_mesh_from_the_dock_accepts_locked_meshes_and_system_entries(
    cad_editor, main_window, monkeypatch, qapp, tmp_path
):
    from blab.ui.main_window import geometry_workflow

    editor, _ = cad_editor
    monkeypatch.setattr(geometry_workflow, "GENERATED_GEOMETRY_ROOT", tmp_path / "runs")
    errors = []
    monkeypatch.setattr(main_window, "show_error", lambda *args: errors.append(args))
    main_window.freq_max_spin.setValue(3430)

    editor.mesh_button.click()
    assert editor.cancel_button.isEnabled()
    wait(qapp, lambda: not main_window.geometry_controller.active)

    assert not errors
    document = main_window.active_generator_document()
    assert [m.id for m in document.artifact.meshes] == ["chamber", "exterior"]
    entries = main_window._mesh_config_dialog_entries()
    assert len(entries) == 2 and all(e.locked and e.assembly_id == document.id for e in entries)
    assert len(PhysicalSystemCompiler().compile(main_window.project.physical_system).interfaces) == 1
    assert "accepted" in editor.status.text()
    assert main_window.editor_dock.features() & QDockWidget.DockWidgetFeature.DockWidgetClosable


def test_cancel_stops_meshing_without_meshes(cad_editor, main_window, monkeypatch, qapp, tmp_path):
    from blab.ui.main_window import geometry_workflow

    editor, _ = cad_editor
    monkeypatch.setattr(geometry_workflow, "GENERATED_GEOMETRY_ROOT", tmp_path / "runs")
    monkeypatch.setattr(main_window, "show_error", lambda *args: None)

    editor.mesh_button.click()
    editor.cancel_button.click()
    wait(qapp, lambda: not main_window.geometry_controller.active)

    assert main_window.active_generator_document().artifact is None


def test_clear_shows_everything_and_clears_the_selection(cad_editor):
    editor, host = cad_editor
    source = host.snapshot().source
    bodies = body_items(editor)
    bodies["Port"].setCheckState(0, Qt.CheckState.Unchecked)
    bodies["Chamber"].child(0).setSelected(True)
    assert editor.selected_faces()

    editor.clear_button.click()

    assert editor.visible_faces() == {face["id"] for face in editor.geometry["faces"]}
    assert editor.selected_faces() == set()
    assert host.snapshot().source == source


def test_bundled_cad_import_is_enabled_once_also_for_existing_installs():
    settings = MemorySettings({"providers/enabled": [VENTED], "providers/vented_enclosure_introduced": True})
    assert load_gui_preferences(settings).enabled_geometry_providers == (VENTED, PROVIDER)
    preferences = load_gui_preferences(settings)
    preferences.enabled_geometry_providers = (VENTED,)
    save_gui_preferences(settings, preferences)
    assert load_gui_preferences(settings).enabled_geometry_providers == (VENTED,)
