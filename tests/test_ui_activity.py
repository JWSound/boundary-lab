"""Activity lifetimes, delayed presentation, and application integration."""

import pytest
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QLabel, QMessageBox

from blab.ui.activity import ActivityController, ActivityStatusBar
from blab.ui.application_state import OperationPhase, OperationState
from test_ui_preparation_worker import wait_until


def test_overlapping_activities_finish_independently(qapp):
    activities = ActivityController()
    first = activities.start("Opening project")
    second = activities.start("Inspecting meshes")
    first.update("Preparing project")
    assert activities.message == "Inspecting meshes"
    second.finish()
    assert activities.active
    assert activities.message == "Preparing project"
    second.finish()
    first.finish()
    first.update("Too late")
    assert not activities.active
    assert activities.message == ""


def test_finishing_older_activity_does_not_clear_newer_one(qapp):
    activities = ActivityController()
    first = activities.start("First")
    second = activities.start("Second")
    first.finish()
    assert activities.message == "Second"
    second.finish()


def test_scoped_activity_cleans_up_on_failure(qapp):
    activities = ActivityController()
    with pytest.raises(ValueError), activities.start("Loading"):
        raise ValueError("Failed")
    assert not activities.active


def test_quick_operations_never_show_spinner(qapp):
    activities = ActivityController(delay_ms=30)
    visibility = QSignalSpy(activities.indicator_changed)
    with activities.start("Quick operation"):
        assert not activities.indicator_visible
    QTest.qWait(60)
    assert visibility.count() == 0


def test_delayed_spinner_survives_overlapping_work_and_stops(qapp):
    activities = ActivityController(delay_ms=20)
    visibility = QSignalSpy(activities.indicator_changed)
    first = activities.start("First")
    assert visibility.wait(500)
    assert activities.indicator_visible
    second = activities.start("Second")
    first.finish()
    second.update("Still working")
    assert activities.indicator_visible
    assert visibility.count() == 1
    second.finish()
    assert not activities.indicator_visible
    assert visibility.count() == 2


def test_clear_invalidates_old_handles_without_affecting_new_activity(qapp):
    activities = ActivityController()
    old = activities.start("Old")
    activities.clear()
    current = activities.start("Current")
    old.finish()
    old.update("Stale")
    assert activities.message == "Current"
    current.finish()


def test_status_row_preserves_completion_text_and_space(qapp):
    activities = ActivityController(delay_ms=10)
    status = QLabel("Ready")
    bar = ActivityStatusBar(activities, status)
    bar.resize(500, 30)
    bar.show()
    qapp.processEvents()
    idle_position = status.pos()
    idle_size = bar.sizeHint()
    try:
        activity = activities.start("Working")
        visibility = QSignalSpy(activities.indicator_changed)
        assert visibility.wait(500)
        assert bar.indicator.isVisible()
        assert bar.indicator._animation.isActive()
        assert bar.activity_label.isVisible()
        status.setText("Complete")
        activity.finish()
        qapp.processEvents()
        assert status.isVisible()
        assert status.text() == "Complete"
        assert not bar.indicator.isVisible()
        assert not bar.indicator._animation.isActive()
        assert status.pos() == idle_position
        assert bar.sizeHint().height() == idle_size.height()
    finally:
        bar.close()


@pytest.mark.parametrize("phase", [OperationPhase.COMPLETED, OperationPhase.CANCELLED, OperationPhase.FAILED])
def test_background_operation_states_update_shared_indicator(main_window, phase):
    window = main_window
    window.geometry_controller.state_changed.emit(OperationState(OperationPhase.RUNNING, "Generating"))
    window.solve_controller.state_changed.emit(OperationState(OperationPhase.RUNNING, "Solving"))
    window.geometry_controller.state_changed.emit(OperationState(phase, "Generation ended"))
    assert window.activities.active
    assert window.activities.message == "Solving"
    window.solve_controller.state_changed.emit(OperationState(OperationPhase.CANCELLING, "Stopping solve"))
    assert window.activities.message == "Stopping solve"
    window.solve_controller.state_changed.emit(OperationState(phase, "Solve ended"))
    assert not window.activities.active
    assert not window._operation_activities


def test_project_failure_releases_activity_before_error_dialog(main_window, monkeypatch, tmp_path):
    import blab.ui.main_window.project_workflow as workflow

    def read(_path):
        assert main_window.activities.active
        raise ValueError("Broken project")

    errors = []

    def show_error(title, message):
        assert not main_window.activities.active
        errors.append(message)

    monkeypatch.setattr(workflow, "read_project_file", read)
    monkeypatch.setattr(main_window, "show_error", show_error)
    main_window.project_workflow._load_project_from_path(tmp_path / "broken.blab.json")
    wait_until(lambda: not main_window.preparations.active)
    assert errors == ["Broken project"]


def test_system_activity_ends_before_modal_interaction(main_window, monkeypatch):
    from types import SimpleNamespace

    class Dialog:
        systemApplied = SimpleNamespace(connect=lambda _slot: None)

        def exec(self):
            assert not main_window.activities.active

    def prepare(activity, _inventory):
        assert main_window.activities.active
        activity.update("Building dialog")
        return Dialog()

    monkeypatch.setattr(main_window, "_prepare_system_config_dialog", prepare)
    main_window.open_system_config()
    wait_until(lambda: not main_window.preparations.active)
    assert not main_window.activities.active


def test_system_failure_releases_activity(main_window, monkeypatch):
    errors = []

    def prepare(_activity, _inventory):
        raise ValueError("Missing mesh")

    def critical(*args):
        assert not main_window.activities.active
        errors.append(args[-1])

    monkeypatch.setattr(main_window, "_prepare_system_config_dialog", prepare)
    monkeypatch.setattr(QMessageBox, "critical", critical)
    main_window.open_system_config()
    wait_until(lambda: not main_window.preparations.active)
    assert "Missing mesh" in errors[0]
