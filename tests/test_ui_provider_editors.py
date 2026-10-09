from dataclasses import replace
from pathlib import Path
from time import monotonic, sleep

import pytest

from blab.generators import catalog as catalog_module
from blab.generators.catalog import ProviderCatalog
from blab.generators.host import ProviderHostError
from blab.ui.provider_editor import DocumentHost
from blab.ui.settings import GuiPreferences, load_gui_preferences, save_gui_preferences


@pytest.fixture(autouse=True)
def isolated_provider_preferences():
    from blab.ui.settings import application_settings

    settings = application_settings()
    settings.remove("providers")
    yield
    settings.remove("providers")


@pytest.fixture
def box_catalog(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "examples" / "geometry_providers"
    catalog = ProviderCatalog([root], enabled=("example.box",))
    catalog.scan()
    monkeypatch.setattr(catalog_module, "_catalog", catalog)
    return catalog


def test_custom_editor_updates_source_and_default_does_not_change_existing_design(main_window, box_catalog):
    old = main_window.active_generator_document()
    main_window.preferences.default_geometry_provider = "example.box"
    main_window.add_generator_document()
    document = main_window.active_generator_document()
    assert document.provider_id == "example.box"
    assert document.mesh_scale_factor == 1.0
    assert main_window.generator_documents[0] == old
    editor, host = main_window._provider_editors[document.id]
    revision = host.snapshot().revision
    editor.spins["width_m"].setValue(0.25)
    assert main_window.active_generator_document().source["width_m"] == 0.25
    with pytest.raises(ProviderHostError, match="Source changed"):
        host.update_source({}, expected_revision=revision)
    assert main_window.project_workflow.has_unsaved_project_changes()


def test_dispose_suppresses_pending_subscription_and_invalidates_source_handle(main_window, qapp):
    document = main_window.active_generator_document()
    host = DocumentHost(main_window, document)
    delivered = []
    host.services.subscribe(delivered.append)
    host.dispose()
    qapp.processEvents()
    qapp.processEvents()
    assert not main_window.provider_host.service._subscriptions
    assert delivered == []
    with pytest.raises(ProviderHostError):
        host.snapshot()
    assert host.services.context().exception().code == "closed"


def test_rebuild_disposes_custom_editor_and_missing_provider_preserves_source(main_window, box_catalog, qapp):
    main_window.preferences.default_geometry_provider = "example.box"
    main_window.add_generator_document()
    document = main_window.active_generator_document()
    _editor, host = main_window._provider_editors[document.id]
    box_catalog.enabled.clear()
    main_window.rebuild_generator_document_tabs()
    qapp.processEvents()
    qapp.processEvents()
    assert main_window.active_generator_document() == document
    assert "disabled" in main_window.editor_tabs.currentWidget().toPlainText()
    assert not main_window.generate_button.isEnabled()
    assert host.services.context().exception().code == "closed"
    assert not main_window.provider_host.service._subscriptions


def test_source_schema_mismatch_does_not_create_editor(main_window, box_catalog):
    main_window.preferences.default_geometry_provider = "example.box"
    main_window.add_generator_document()
    document = replace(main_window.active_generator_document(), provider_schema_version=2)
    main_window.generator_documents = (document,)
    main_window.rebuild_generator_document_tabs()
    assert "Incompatible" in main_window.editor_tabs.currentWidget().toPlainText()
    assert not main_window.generate_button.isEnabled()
    assert main_window.active_generator_document().source == document.source


def test_document_generate_selects_its_own_document_and_correlates_request(main_window, monkeypatch):
    first = main_window.active_generator_document()
    host = DocumentHost(main_window, first)
    main_window.add_generator_document()
    calls = []

    def generate():
        calls.append(main_window.active_generator_document().id)
        main_window.geometry_workflow._pending_request_id = "request"

    monkeypatch.setattr(main_window.geometry_workflow, "generate_geometry", generate)
    assert host.generate(expected_revision=host.snapshot().revision) == "request"
    assert calls == [first.id]


def test_provider_preferences_roundtrip_and_new_project_default(main_window, box_catalog, tmp_path):
    from PySide6.QtCore import QSettings

    settings = QSettings(str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat)
    preferences = GuiPreferences(default_geometry_provider="example.box", enabled_geometry_providers=("example.box",))
    save_gui_preferences(settings, preferences)
    restored = load_gui_preferences(settings)
    assert restored.default_geometry_provider == "example.box"
    assert restored.enabled_geometry_providers == ("example.box",)
    main_window.preferences = restored
    main_window.project_workflow.confirm_unsaved_project_changes = lambda _action: True
    main_window.new_project()
    assert main_window.active_generator_document().provider_id == "example.box"


def test_custom_editor_generates_through_worker_and_accepts_memory_mesh(main_window, box_catalog, qapp, monkeypatch):
    errors = []
    monkeypatch.setattr(main_window, "show_error", lambda *args: errors.append(args))
    main_window.preferences.default_geometry_provider = "example.box"
    main_window.add_generator_document()
    document = main_window.active_generator_document()
    _editor, host = main_window._provider_editors[document.id]
    events = []
    host.services.subscribe(events.append)
    qapp.processEvents()
    request_id = host.generate(expected_revision=host.snapshot().revision)
    deadline = monotonic() + 10
    while main_window.geometry_controller.active and monotonic() < deadline:
        qapp.processEvents()
        sleep(0.001)
    assert not main_window.geometry_controller.active
    assert not errors
    artifact = main_window.active_generator_document().artifact
    assert artifact is not None and artifact.mesh_data is not None
    assert main_window.project.physical_system.meshes[0].scale_to_m == 1.0
    assert any(event.kind == "geometry_accepted" and event.generation_request_id == request_id for event in events)


def test_failed_editor_factory_preserves_document_and_releases_subscription(
    main_window, box_catalog, qapp, monkeypatch
):
    def factory(parent, host):
        host.services.subscribe(lambda event: None)
        raise RuntimeError("Broken editor")

    monkeypatch.setattr(box_catalog, "factory", lambda *_args: factory)
    main_window.preferences.default_geometry_provider = "example.box"
    main_window.add_generator_document()
    assert "Broken editor" in main_window.editor_tabs.currentWidget().toPlainText()
    assert main_window.active_generator_document().source == box_catalog.source_defaults("example.box")
    qapp.processEvents()
    qapp.processEvents()
    assert not main_window.provider_host.service._subscriptions


def test_plugin_manager_requires_available_default_and_cancel_preserves_catalog(qapp, box_catalog):
    from PySide6.QtWidgets import QDialog, QDialogButtonBox

    from blab.ui.provider_preferences import ProviderPackagesDialog

    dialog = ProviderPackagesDialog(("example.box",), default_provider="example.box")
    assert dialog.table.item(0, 3).text() == "Built-in"
    assert not dialog.table.cellWidget(0, 0).isEnabled()
    dialog.table.cellWidget(1, 0).setChecked(False)
    assert not dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    dialog.accept()
    assert dialog.result() != QDialog.Accepted
    dialog.default_provider_combo.setCurrentIndex(0)
    assert dialog.default_provider == "ath"
    assert dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    dialog.reject()
    assert box_catalog.enabled == {"example.box"}
    dialog.deleteLater()


def test_plugin_manager_rescan_retains_staged_choices(qapp, box_catalog):
    from blab.ui.provider_preferences import ProviderPackagesDialog

    dialog = ProviderPackagesDialog((), default_provider="ath")
    dialog.table.cellWidget(1, 0).setChecked(True)
    dialog.default_provider_combo.setCurrentIndex(1)
    dialog.rescan()
    assert dialog.default_provider == "example.box"
    assert dialog.table.cellWidget(1, 0).isChecked()
    dialog.deleteLater()


def test_plugin_manager_missing_default_and_restart_status(qapp, box_catalog, monkeypatch):
    from PySide6.QtWidgets import QDialogButtonBox

    from blab.ui.provider_preferences import ProviderPackagesDialog

    package = box_catalog.packages[0]
    package.error = "Package changed after loading; restart Boundary Lab."
    monkeypatch.setattr(box_catalog, "scan", lambda: box_catalog.packages)
    dialog = ProviderPackagesDialog(("example.box",), default_provider="example.box")
    assert "restart" in dialog.table.item(1, 3).text()
    assert not dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    dialog.deleteLater()
    missing = ProviderPackagesDialog((), default_provider="missing.plugin")
    assert "unavailable" in missing.default_provider_combo.currentText()
    assert not missing.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    missing.deleteLater()


def test_preferences_preserve_plugin_settings_without_manager_controls(qapp):
    from blab.ui.dialogs import PreferencesDialog

    preferences = GuiPreferences(default_geometry_provider="example.box", enabled_geometry_providers=("example.box",))
    dialog = PreferencesDialog(preferences)
    assert not hasattr(dialog, "default_provider_combo")
    assert not hasattr(dialog, "provider_packages_button")
    saved = dialog.preferences()
    assert saved.default_geometry_provider == "example.box"
    assert saved.enabled_geometry_providers == ("example.box",)
    dialog.deleteLater()


@pytest.mark.parametrize("accepted", [True, False])
def test_plugin_manager_commit_preserves_design_artifacts_and_results(
    main_window, box_catalog, monkeypatch, accepted
):
    from PySide6.QtWidgets import QDialog

    from blab.ui.provider_preferences import ProviderPackagesDialog

    main_window.preferences.default_geometry_provider = "example.box"
    main_window.preferences.enabled_geometry_providers = ("example.box",)
    main_window.add_generator_document()
    editor, _host = main_window._provider_editors[main_window.active_generator_document_id]
    editor.spins["width_m"].setValue(0.25)
    documents = main_window.generator_documents
    selected = main_window.active_generator_document_id
    artifacts = dict(main_window.generated_geometry_by_document_id)
    invalidations = []
    main_window.solve_results_invalidated.connect(invalidations.append)

    def execute(dialog):
        dialog.table.cellWidget(1, 0).setChecked(False)
        dialog.default_provider_combo.setCurrentIndex(0)
        return QDialog.Accepted if accepted else QDialog.Rejected

    monkeypatch.setattr(ProviderPackagesDialog, "exec", execute)
    main_window.open_generator_plugins()
    assert main_window.generator_documents == documents
    assert main_window.active_generator_document_id == selected
    assert main_window.generated_geometry_by_document_id == artifacts
    assert not invalidations
    assert main_window.preferences.default_geometry_provider == ("ath" if accepted else "example.box")
    assert box_catalog.enabled == (set() if accepted else {"example.box"})
    if accepted:
        assert "Edit > Generator Plugins" in main_window.editor_tabs.currentWidget().toPlainText()
        restored = load_gui_preferences(main_window.settings)
        assert restored.default_geometry_provider == "ath"
        assert restored.enabled_geometry_providers == ()
        box_catalog.enabled.add("example.box")
        main_window.rebuild_generator_document_tabs()
        editor, _host = main_window._provider_editors[selected]
        assert editor.spins["width_m"].value() == 0.25


def test_plugin_manager_cannot_apply_during_background_work(qapp, box_catalog, monkeypatch):
    from PySide6.QtWidgets import QDialog, QMessageBox

    from blab.ui.provider_preferences import ProviderPackagesDialog

    messages = []
    monkeypatch.setattr(QMessageBox, "information", lambda *args: messages.append(args))
    dialog = ProviderPackagesDialog((), is_busy=lambda: True)
    dialog.accept()
    assert messages
    assert dialog.result() != QDialog.Accepted
    dialog.deleteLater()


def test_generator_menu_and_layout_compatibility(main_window, monkeypatch):
    from blab.ui.provider_preferences import ProviderPackagesDialog

    calls = []
    monkeypatch.setattr(ProviderPackagesDialog, "exec", lambda _dialog: calls.append(True))
    menu_actions = main_window.menuBar().actions()
    edit = next(action for action in menu_actions if action.text() == "Edit").menu()
    action = next(action for action in edit.actions() if action.text() == "Generator Plugins...")
    action.trigger()
    assert calls == [True]
    assert main_window.editor_dock.windowTitle() == "Generator"
    assert main_window.editor_dock.objectName() == "ath_editor_dock"
    assert main_window.panel_view_actions["editor"].text() == "Generator"
    state = main_window.workspace.saveState()
    main_window.editor_dock.hide()
    assert main_window.workspace.restoreState(state)
    assert not main_window.editor_dock.isHidden()
    from blab.ui.application_state import OperationPhase

    main_window.set_workflow_phase(OperationPhase.RUNNING)
    assert not action.isEnabled()
    main_window.set_workflow_phase(OperationPhase.IDLE)
    assert action.isEnabled()


def test_ath_import_keeps_custom_design_and_export_follows_active_plugin(main_window, box_catalog, tmp_path):
    main_window.preferences.default_geometry_provider = "example.box"
    main_window.add_generator_document()
    custom = main_window.active_generator_document()
    assert custom.name.startswith("design")
    assert not main_window.export_ath_design_action.isEnabled()
    path = tmp_path / "horn.cfg"
    path.write_text("Throat.Diameter = 25\n")
    main_window.import_config_path(path)
    assert custom in main_window.generator_documents
    assert main_window.active_generator_document().provider_id == "ath"
    assert main_window.active_generator_document().source["text"] == path.read_text()
    assert main_window.export_ath_design_action.isEnabled()
