from time import monotonic, sleep
from types import SimpleNamespace

from blab.generators import catalog as catalog_module
from blab.physical_compiler import PhysicalSystemCompiler
from blab.ui.settings import load_gui_preferences, save_gui_preferences
from test_ui_settings import MemorySettings
from test_vented_enclosure_generator import PROVIDER
from test_vented_enclosure_generator import plugin as plugin


def test_bundled_plugin_is_enabled_once_and_keeps_existing_default():
    settings = MemorySettings({"providers/default": "ath", "providers/enabled": ["custom.plugin", {}]})
    preferences = load_gui_preferences(settings)
    assert preferences.default_geometry_provider == "ath"
    assert preferences.enabled_geometry_providers == ("custom.plugin", PROVIDER)
    preferences.enabled_geometry_providers = ("custom.plugin",)
    save_gui_preferences(settings, preferences)
    assert load_gui_preferences(settings).enabled_geometry_providers == ("custom.plugin",)


def test_slider_editor_uses_host_revisions_and_busy_state(main_window, plugin, monkeypatch, qapp):
    catalog, _, _ = plugin
    monkeypatch.setattr(catalog_module, "_catalog", catalog)
    main_window.preferences.default_geometry_provider = PROVIDER
    main_window.add_generator_document()
    document = main_window.active_generator_document()
    editor, host = main_window._provider_editors[document.id]
    assert document.mesh_scale_factor == 1
    editor.spins["width_m"].setValue(350)
    assert host.snapshot().source["width_m"] == 0.35
    assert host.snapshot().source["port_area_m2"] == document.source["port_area_m2"]
    assert "Generate" in editor.status.text()
    assert editor.sketch.source["width_m"] == 0.35
    editor.sliders["port_nfr"].setValue(200)
    assert host.snapshot().source["port_nfr"] == 0.16
    editor.spins["port_y_m"].setValue(100)
    assert "overlap" in editor.summary.text()
    assert editor.sketch.source is None
    editor.set_operation_state(SimpleNamespace(active=True))
    assert not editor.spins["width_m"].isEnabled()
    editor.set_operation_state(SimpleNamespace(active=False))
    assert editor.spins["width_m"].isEnabled()
    # Exercise the custom painter at a typical dock width.
    editor.spins["port_y_m"].setValue(-130)
    editor.widget.resize(380, 700)
    qapp.processEvents()
    assert not editor.widget.grab().isNull()
    assert not main_window.generated_geometry_by_document_id


def test_generate_from_dock_accepts_coupled_assembly(main_window, plugin, monkeypatch, qapp, tmp_path):
    from blab.ui.main_window import geometry_workflow

    catalog, _, _ = plugin
    monkeypatch.setattr(catalog_module, "_catalog", catalog)
    monkeypatch.setattr(geometry_workflow, "GENERATED_GEOMETRY_ROOT", tmp_path)
    errors = []
    monkeypatch.setattr(main_window, "show_error", lambda *args: errors.append(args))
    main_window.preferences.default_geometry_provider = PROVIDER
    main_window.add_generator_document()
    document = main_window.active_generator_document()
    editor, host = main_window._provider_editors[document.id]
    host.generate(expected_revision=host.snapshot().revision)
    deadline = monotonic() + 30
    while main_window.geometry_controller.active and monotonic() < deadline:
        qapp.processEvents()
        sleep(0.005)
    assert not main_window.geometry_controller.active
    assert not errors
    assert len(main_window.active_generator_document().artifact.meshes) == 2
    assert len(PhysicalSystemCompiler().compile(main_window.project.physical_system).interfaces) == 1
    assert "accepted" in editor.status.text()
