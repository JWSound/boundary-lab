from dataclasses import replace

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QDialog, QLabel, QTabWidget

from blab.ui.dialogs import PreferencesDialog, ProjectConfigurationDialog
from blab.ui.settings import load_gui_preferences


def test_project_configuration_scope_and_explicit_defaults(main_window, monkeypatch, tmp_path):
    window = main_window
    window.settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    initial = window.new_project_defaults
    distance = initial.polar_observation_distance_m + 1
    application = replace(window.preferences, theme="dark", field_cache_size_mb=768)
    window.preferences = application

    def edit(dialog):
        tabs = dialog.findChild(QTabWidget)
        assert [tabs.tabText(i) for i in range(tabs.count())] == ["Observation Sampling", "Plot Presentation"]
        assert any("Impose Channel Normalization" in label.toolTip() for label in dialog.findChildren(QLabel))
        dialog.spherical_sampling_check.setChecked(False)
        assert dialog.balloon_angle_precision_spin.isHidden()
        dialog.spherical_sampling_check.setChecked(True)
        assert not dialog.balloon_angle_precision_spin.isHidden()
        dialog.polar_distance_spin.setValue(distance)
        return QDialog.Accepted

    monkeypatch.setattr(ProjectConfigurationDialog, "exec", edit)
    window.project_configuration_action.trigger()
    assert window.project_workflow.has_unsaved_project_changes()
    assert window.stitch_imported_meshes
    window.stitch_imported_meshes = False
    path = tmp_path / "distance.blab.json"
    assert window.project_workflow._save_project_to_path(path)
    window._save_preferences()
    assert not window.settings.contains("preferences/polar_observation_distance_m")
    window.new_project()
    assert window.project.project_preferences == initial
    assert window.stitch_imported_meshes
    window.project_workflow._preparations = None
    window.project_workflow.open_recent_project(path)
    assert not window.stitch_imported_meshes
    assert window.preferences.polar_observation_distance_m == distance
    assert window.preferences.theme == "dark" and window.preferences.field_cache_size_mb == 768
    assert not window.project_workflow.has_unsaved_project_changes()

    def adopt(dialog):
        dialog.use_as_defaults.setChecked(True)
        return QDialog.Accepted

    monkeypatch.setattr(ProjectConfigurationDialog, "exec", adopt)
    window.open_project_configuration()
    assert not window.project_workflow.has_unsaved_project_changes()
    window.new_project()
    assert window.preferences.polar_observation_distance_m == distance
    assert load_gui_preferences(window.settings).polar_observation_distance_m == distance

    # Partial legacy files inherit only the fields they omitted.
    window.project_workflow._apply_project_payload({"project_preferences": {"spl_min_db": -42, "freq_count": 1}})
    assert window.preferences.polar_observation_distance_m == distance
    assert window.preferences.spl_min_db == -42
    assert window.freq_count_spin.value() == 1
    monkeypatch.setattr(window.project_workflow, "confirm_unsaved_project_changes", lambda _: True)
    window.new_project()
    assert window.preferences.spl_min_db == initial.spl_min_db


def test_presentation_updates_synthesis_and_sampling_confirmation_is_atomic(main_window, monkeypatch):
    from unittest.mock import Mock

    window = main_window
    initial = window.project.project_preferences
    dataset = Mock(has_balloon_data=False)
    window.live_dataset = dataset
    monkeypatch.setattr(window, "_refresh_channel_responses", Mock())
    confirm = Mock(return_value=False)
    monkeypatch.setattr(window, "_confirm_clear_solved_data", confirm)

    def presentation(dialog):
        dialog.horizontal_norm_angle_spin.setValue(30)
        return QDialog.Accepted

    monkeypatch.setattr(ProjectConfigurationDialog, "exec", presentation)
    window.open_project_configuration()
    confirm.assert_not_called()
    dataset.set_channel_synthesis.assert_called_once_with(window.channel_configs(), flat_target_reference_angle_deg=30)
    window._refresh_channel_responses.assert_called_once()
    assert window.project.project_preferences == replace(initial, horizontal_normalization_angle=30)

    def sampling(dialog):
        dialog.polar_distance_spin.setValue(initial.polar_observation_distance_m + 1)
        dialog.use_as_defaults.setChecked(True)
        return QDialog.Accepted

    monkeypatch.setattr(ProjectConfigurationDialog, "exec", sampling)
    window.open_project_configuration()
    confirm.assert_called_once()
    assert window.project.project_preferences.polar_observation_distance_m == initial.polar_observation_distance_m
    assert window.new_project_defaults == initial
    assert window.live_dataset is dataset

    preferences = PreferencesDialog(window.preferences)
    assert not hasattr(preferences, "polar_distance_spin")
    assert not hasattr(preferences, "horizontal_norm_angle_spin")
    preferences.close()


def test_application_preferences_do_not_dirty_project(main_window, monkeypatch):
    window = main_window
    before = window.project_workflow.project_payload()

    def edit(dialog):
        dialog.field_cache_size_spin.setValue(768)
        return QDialog.Accepted

    monkeypatch.setattr(PreferencesDialog, "exec", edit)
    window.open_preferences()
    assert window.preferences.field_cache_size_mb == 768
    assert window.project_workflow.project_payload() == before
    assert not window.project_workflow.has_unsaved_project_changes()
