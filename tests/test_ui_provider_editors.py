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


def test_failed_editor_factory_preserves_document_and_releases_subscription(main_window, box_catalog, qapp, monkeypatch):
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
