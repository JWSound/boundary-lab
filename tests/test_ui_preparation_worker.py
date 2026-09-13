from threading import Event, get_ident
from time import monotonic

from PySide6.QtTest import QTest

from blab.ui.activity import ActivityController
from blab.ui.preparation_worker import PreparationController


def wait_until(predicate, timeout=3):
    deadline = monotonic() + timeout
    while not predicate() and monotonic() < deadline:
        QTest.qWait(10)
    assert predicate()


def test_worker_does_not_block_gui_and_delivers_only_latest_result(qapp):
    activities = ActivityController(delay_ms=10)
    controller = PreparationController(None, activities)
    gate = Event()
    started = Event()
    results, errors, threads = [], [], []
    gui_thread = get_ident()

    def work():
        threads.append(get_ident())
        started.set()
        gate.wait(3)
        raise ValueError("Obsolete failure")

    try:
        controller.submit("preview", "Old", work, results.append, errors.append)
        wait_until(started.is_set)
        wait_until(lambda: activities.indicator_visible)
        assert threads[0] != gui_thread
        controller.submit(
            "preview", "New", lambda: 42, lambda value: results.append((get_ident(), value)), errors.append
        )
        assert activities.active
        gate.set()
        wait_until(lambda: not controller.active)
        assert results == [(gui_thread, 42)]
        assert not errors
        assert not activities.active
    finally:
        gate.set()
        controller.close()


def test_cancelled_work_and_failed_work_release_handles(qapp):
    activities = ActivityController()
    controller = PreparationController(None, activities)
    gate = Event()
    results, errors = [], []
    try:
        controller.submit("system", "Inspect", lambda: gate.wait(3), results.append, errors.append)
        controller.cancel("system")
        gate.set()
        wait_until(lambda: not controller.active)
        assert not activities.active

        def fail():
            raise ValueError("Missing file")

        controller.submit("project", "Read", fail, results.append, errors.append)
        wait_until(lambda: not controller.active)
        assert not results
        assert str(errors[0]) == "Missing file"
        assert not activities.active
    finally:
        gate.set()
        controller.close()


def test_project_edit_while_loading_discards_completion(main_window, monkeypatch, tmp_path):
    from blab.ui.project_state import replace_generator_document

    gate = Event()
    started = Event()
    previous = main_window.project

    def read(_path):
        started.set()
        gate.wait(3)
        return {}

    monkeypatch.setattr("blab.ui.main_window.project_workflow.read_project_file", read)
    try:
        main_window.project_workflow._load_project_from_path(tmp_path / "other.blab.json")
        wait_until(started.is_set)
        document = main_window.generator_documents[0]
        main_window.generator_documents = replace_generator_document(
            main_window.generator_documents,
            document.id,
            name="Edited while loading",
        )
        gate.set()
        wait_until(lambda: not main_window.preparations.active)
        assert main_window.project is previous
        assert main_window.generator_documents[0].name == "Edited while loading"
        assert "discarded" in main_window.status_label.text()
    finally:
        gate.set()


def test_authored_project_skips_legacy_mesh_preparation(main_window, monkeypatch):
    from blab.physical_model import PhysicalSystem, physical_system_to_dict

    def unexpected():
        calls.append("read")
        raise AssertionError("Authored projects must not prepare meshes to restore legacy sources")

    monkeypatch.setattr(main_window, "surface_tags_for_meshes", unexpected)
    # Do not let the old exception-swallowing path conceal this regression.
    calls = []
    monkeypatch.setattr(main_window, "apply_saved_imported_source_config", lambda _tags: calls.append(True))
    system = PhysicalSystem("system", "Authored", (), (), ())
    main_window.project_workflow._apply_project_payload({"physical_system": physical_system_to_dict(system)})
    assert not calls
    assert main_window.project.physical_system == system


def test_preview_completion_cannot_replace_a_new_project(main_window, monkeypatch):
    from blab.ui.project_state import ImportedMeshState

    pending = {}
    main_window.project.imported_meshes = (ImportedMeshState("mesh", "unused.msh"),)
    original = main_window.project

    def submit(key, message, work, complete, failed):
        pending.update(complete=complete, failed=failed)

    monkeypatch.setattr(main_window.preparations, "submit", submit)
    main_window._refresh_mesh_preview()
    main_window.project_workflow.confirm_unsaved_project_changes = lambda _action: True
    main_window.new_project()
    pending["complete"](None)  # Stale result must be rejected before it is unpacked.
    pending["failed"](ValueError("Obsolete"))
    assert main_window.project is not original
    assert not main_window.project.imported_meshes
    assert "Obsolete" not in main_window.status_label.text()


def test_stop_keeps_controls_locked_until_running_preparation_finishes(main_window):
    gate = Event()
    started = Event()
    results = []

    def work():
        started.set()
        gate.wait(3)
        return "Should not publish"

    try:
        main_window.preparations.submit("preview", "Preparing", work, results.append, results.append)
        wait_until(started.is_set)
        assert not main_window.solve_button.isEnabled()
        assert not main_window.generate_button.isEnabled()
        assert main_window.cancel_button.isEnabled()
        main_window.cancel_current_operation()
        assert main_window.preparations.active
        assert not main_window.solve_button.isEnabled()
        gate.set()
        wait_until(lambda: not main_window.preparations.active)
        assert not results
        assert main_window.generate_button.isEnabled()
        assert not main_window.cancel_button.isEnabled()
    finally:
        gate.set()


def test_opening_project_supersedes_pending_preview(main_window, monkeypatch, tmp_path):
    gate = Event()
    started = Event()
    old_results, loaded = [], []

    def old_preview():
        started.set()
        gate.wait(3)
        return "Old preview"

    monkeypatch.setattr("blab.ui.main_window.project_workflow.read_project_file", lambda _path: {})
    monkeypatch.setattr(main_window.project_workflow, "_finish_project_load", lambda *args: loaded.append(args))
    try:
        main_window.preparations.submit("preview", "Old preview", old_preview, old_results.append, old_results.append)
        wait_until(started.is_set)
        main_window.project_workflow._load_project_from_path(tmp_path / "new.blab.json")
        gate.set()
        wait_until(lambda: not main_window.preparations.active)
        assert not old_results
        assert len(loaded) == 1
    finally:
        gate.set()


def test_cancelled_worker_progress_cannot_overwrite_current_activity(qapp):
    activities = ActivityController()
    controller = PreparationController(None, activities)
    gate, started, reported = Event(), Event(), Event()
    release = Event()
    results, errors = [], []

    def work(report):
        started.set()
        gate.wait(3)
        report("Obsolete progress")
        reported.set()
        release.wait(3)

    try:
        controller.submit("solve", "Preparing solve...", work, results.append, errors.append,
                          report_progress=True)
        wait_until(started.is_set)
        controller.cancel("solve")
        gate.set()
        wait_until(reported.is_set)
        QTest.qWait(20)
        assert activities.message == "Stopping preparation..."
        release.set()
        wait_until(lambda: not controller.active)
        assert not results and not errors and not activities.active
    finally:
        gate.set()
        release.set()
        controller.close()
