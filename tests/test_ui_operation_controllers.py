import time
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QCoreApplication, QObject, Signal, Slot

import blab.ui.operation_controllers as controller_module
from blab.system_contract import SystemFrequencyResult
from blab.ui.application_state import OperationPhase
from blab.ui.operation_controllers import GeometryController, SolveController
from blab.ui.system_solve import SystemSolveWorker
from blab.warm_up_history import WarmUpHistory, WarmUpKey


class _SolveWorkerStub(QObject):
    result = SystemFrequencyResult(
        freq_hz=1000.0,
        quantities=(),
        excitation_port_ids=(),
        diagnostics={"timings": {"assembly_s": 1.2, "solve_s": 0.3, "field_s": 0.04}},
    )
    initialized = Signal(object, object, object)
    result_ready = Signal(object)
    status = Signal(str)
    warm_up_key = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, *_args, **_kwargs):
        super().__init__()
        self.stopped = False

    @Slot()
    def run(self) -> None:
        self.status.emit("Preparing backend")
        self.initialized.emit(np.array([0.0]), np.array(["driver"]), None)
        self.result_ready.emit(self.result)
        self.finished.emit()

    @Slot()
    def stop(self) -> None:
        self.stopped = True


def test_rejected_projection_stops_worker_and_cannot_complete_successfully(qapp) -> None:
    controller = SolveController()
    worker = _SolveWorkerStub()
    controller._worker = worker
    controller._begin_run(1)
    controller._set_state(OperationPhase.RUNNING, "Solving")
    stopped_at_notification = []
    controller.failed.connect(lambda _message: stopped_at_notification.append(worker.stopped))
    controller.result_ready.connect(lambda _result: controller.reject_result("Invalid pressure shape"))

    controller._on_result(_SolveWorkerStub.result)
    controller._on_finished()

    assert worker.stopped
    assert stopped_at_notification == [True]
    assert controller.last_error == "Invalid pressure shape"
    assert controller.last_completion.phase == OperationPhase.FAILED
    assert controller.last_completion.solved_count == 0
    assert not controller.last_completion.completed


def test_solve_controller_owns_worker_thread_and_completion_state(qapp, monkeypatch) -> None:
    app = QCoreApplication.instance() or QCoreApplication([])
    monkeypatch.setattr(controller_module, "SystemSolveWorker", _SolveWorkerStub)
    controller = SolveController()
    statuses = []
    controller.status.connect(statuses.append)
    results = []
    completions = []
    controller.result_ready.connect(results.append)
    controller.finished.connect(completions.append)

    started = controller.start(SimpleNamespace(request=SimpleNamespace(frequencies_hz=(1000.0,))))
    thread = controller._thread
    assert thread is not None
    thread_state_at_completion = []
    controller.finished.connect(
        lambda _completion: thread_state_at_completion.append((thread.isRunning(), thread.isFinished()))
    )
    deadline = time.monotonic() + 2.0
    while not completions and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    for _ in range(5):
        app.processEvents()

    assert started is True
    assert results == [_SolveWorkerStub.result]
    assert statuses == [
        "Preparing backend · 0:00",
        "Solving 1/1 · 0:00 elapsed",
    ]
    assert completions[0].phase == OperationPhase.COMPLETED
    assert completions[0].completed is True
    assert thread_state_at_completion == [(False, True)]
    assert controller.last_completion == completions[0]
    assert controller.last_error is None
    assert controller.active is False


def test_streaming_status_updates_only_for_completed_frequencies(qapp):
    controller = SolveController()
    controller.clock = lambda: 0.0
    controller._begin_run(2)
    controller._set_state(OperationPhase.RUNNING, "Initializing solver")
    statuses = []
    controller.status.connect(statuses.append)
    controller._on_status("Loading backend")
    controller._on_initialized(np.array([0.0]), np.array(["driver"]), None)
    assert controller.state.message == "Solving... · 0:00"
    # BEAT reports its warm-up phases after the worker announces the session.
    controller._on_status("Preparing BEAT CPU geometry caches")
    assert controller.state.message == "Preparing BEAT CPU geometry caches · 0:00"
    controller._on_result(_SolveWorkerStub.result)
    expected = "Solving 1/2 · 0:00 elapsed"
    assert controller.state.message == expected
    controller._on_status("Computing field for next frequency")
    assert controller.state.message == expected
    assert statuses == ["Loading backend · 0:00", "Preparing BEAT CPU geometry caches · 0:00", expected]
    controller._on_result(
        SystemFrequencyResult(
            freq_hz=2000.0, quantities=(), excitation_port_ids=(), diagnostics=_SolveWorkerStub.result.diagnostics
        )
    )
    assert controller.state.message == "Solving 2/2 · 0:00 elapsed"
    controller.cancel()
    controller._on_status("Backend still stopping")
    controller._on_result(_SolveWorkerStub.result)
    assert controller.state.phase == OperationPhase.CANCELLING
    assert controller.state.message == "Stopping solve"
    controller._on_failed("Backend failed")
    controller._on_status("Late backend detail")
    assert controller.state.phase == OperationPhase.FAILED
    assert controller.state.message == "Backend failed"


def test_streaming_status_shows_elapsed_and_remaining_time(qapp):
    now = [100.0]
    controller = SolveController()
    controller.clock = lambda: now[0]
    controller._worker = _SolveWorkerStub()
    controller._begin_run(10)
    statuses = []
    controller.status.connect(statuses.append)
    for arrival_s in (130.0, 140.0, 142.0, 146.0):
        now[0] = arrival_s
        controller._on_result(_SolveWorkerStub.result)

    assert statuses == [
        "Solving 1/10 · 0:30 elapsed",
        "Solving 2/10 · 0:40 elapsed",
        "Solving 3/10 · 0:42 elapsed",
        "Solving 4/10 · 0:46 elapsed · ~0:20 left",
    ]
    controller.cancel()
    controller._on_finished()
    assert controller.state.message == "Solve finished"


def test_warm_up_status_ticks_elapsed_time_until_first_result(qapp):
    now = [100.0]
    controller = SolveController()
    controller.clock = lambda: now[0]
    controller._begin_run(10)
    assert controller.state.message == "Initializing solver · 0:00"
    assert controller._ticker.isActive() and controller._ticker.interval() == 1000
    statuses = []
    controller.status.connect(statuses.append)

    now[0] = 142.0
    controller._on_status("Preparing BEAT CPU geometry caches")
    now[0] = 150.0
    controller._ticker.timeout.emit()
    now[0] = 160.0
    controller._on_result(_SolveWorkerStub.result)

    assert statuses == [
        "Preparing BEAT CPU geometry caches · 0:42",
        "Preparing BEAT CPU geometry caches · 0:50",
        "Solving 1/10 · 1:00 elapsed",
    ]
    assert not controller._ticker.isActive()


def test_warm_up_shows_prediction_and_records_duration_at_first_result(qapp, tmp_path):
    from blab.warm_up_history import WarmUpKey

    key = WarmUpKey("cpu", "0.4.0", "8", True)
    history = WarmUpHistory(tmp_path / "history.json")
    history.record(key, 90.0)
    now = [100.0]
    controller = SolveController()
    controller.clock = lambda: now[0]
    controller.warm_up_history = history
    controller._begin_run(10)
    statuses = []
    controller.status.connect(statuses.append)

    now[0] = 142.0
    controller._on_warm_up_key(key)
    now[0] = 195.0
    controller._on_status("Preparing BEAT CPU geometry caches")
    now[0] = 225.0
    controller._ticker.timeout.emit()
    now[0] = 230.0
    controller._on_result(_SolveWorkerStub.result)
    controller._on_result(_SolveWorkerStub.result)

    assert statuses[:3] == [
        "Initializing solver · 0:42 / ~2:00",
        "Preparing BEAT CPU geometry caches · 1:35 / ~2:00",
        "Preparing BEAT CPU geometry caches · 2:05 / longer than usual",
    ]
    assert history.durations_s(key) == [90.0, 130.0]


def test_warm_up_cancelled_before_first_result_is_not_recorded(qapp, tmp_path):
    from blab.warm_up_history import WarmUpKey

    key = WarmUpKey("cpu", "0.4.0", "8", True)
    controller = SolveController()
    controller.warm_up_history = WarmUpHistory(tmp_path / "history.json")
    controller._begin_run(10)
    controller._on_warm_up_key(key)
    controller.cancel()
    controller._on_result(_SolveWorkerStub.result)
    controller._on_finished()

    assert controller.warm_up_history.durations_s(key) == []


def test_warm_up_ticking_stops_on_cancel_and_failure(qapp):
    controller = SolveController()
    controller._begin_run(10)
    controller.cancel()
    assert not controller._ticker.isActive()

    controller._begin_run(10)
    controller._on_failed("Backend failed")
    assert not controller._ticker.isActive()

    controller._begin_run(10)
    controller._on_finished()
    assert not controller._ticker.isActive()


def test_frequency_status_is_visible_in_activity_overlay(main_window):
    controller = main_window.solve_controller
    controller.clock = lambda: 0.0
    controller._begin_run(2)
    controller._set_state(OperationPhase.RUNNING, "Solving...")
    controller._on_result(_SolveWorkerStub.result)
    expected = "Solving 1/2 · 0:00 elapsed"
    assert main_window.activities.message == expected
    assert main_window.activity_status.activity_label.text() == expected
    assert main_window.status_label.text() == expected
    controller._on_status("Assembling next frequency")
    assert main_window.activity_status.activity_label.text() == expected
    controller._set_state(OperationPhase.COMPLETED, "Solve finished")


def test_controllers_preserve_latest_failure_for_diagnostics(qapp) -> None:
    app = QCoreApplication.instance() or QCoreApplication([])
    geometry_controller = GeometryController()
    solve_controller = SolveController()

    geometry_controller._on_failed("geometry failed")
    solve_controller._on_failed("solve failed")

    assert geometry_controller.last_error == "geometry failed"
    assert app is not None
    assert geometry_controller.state.phase == OperationPhase.FAILED
    assert solve_controller.last_error == "solve failed"
    assert solve_controller.state.phase == OperationPhase.FAILED


def test_solve_worker_logs_and_emits_backend_status(monkeypatch, caplog) -> None:
    class Session:
        metadata = SimpleNamespace(
            polar_angle_deg=np.array([0.0]),
            radiator_names=np.array(["driver"]),
            sphere_metadata=None,
        )

        def __init__(self, request):
            self.request = request

        def warm_up_key(self):
            return WarmUpKey("cpu", "test", "1", True)

        def solve_stream(self, *, stop_requested=None):
            del stop_requested
            self.request.status_callback("assembling backend detail")
            return iter(())

        def stop(self) -> None:
            pass

    class Backend:
        def create_system_session(self, request):
            request.status_callback("initializing backend detail")
            return Session(request)

    monkeypatch.setattr("blab.ui.system_solve.PhysicalSystemProductionBackend", lambda *_args, **_kwargs: Backend())
    from blab.system_contract import SystemSolveRequest

    worker = SystemSolveWorker(
        SimpleNamespace(
            request=SystemSolveRequest(compiled_system=None, frequencies_hz=(1000.0,), excitation_port_ids=()),
            backend_id="beat_cpu",
            cuda_worker_reuse=False,
            polar_angle_deg=np.array([0.0]),
            excitation_component_names=np.array(["driver"]),
            sphere_metadata=None,
        )
    )
    statuses = []
    worker.status.connect(statuses.append)

    with caplog.at_level("INFO", logger="blab.ui.system_solve"):
        worker.run()

    assert statuses == ["initializing backend detail", "assembling backend detail"]
    assert "initializing backend detail" in caplog.text
    assert "assembling backend detail" in caplog.text
