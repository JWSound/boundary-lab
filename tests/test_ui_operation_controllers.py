import time
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QCoreApplication, QObject, Signal, Slot

import blab.ui.operation_controllers as controller_module
from blab.solvers.base import FrequencySolveTimings
from blab.ui.application_state import OperationPhase
from blab.ui.operation_controllers import GeometryController, SolveController
from blab.ui.system_solve import SystemSolveWorker


class _SolveWorkerStub(QObject):
    result = SimpleNamespace(freq_hz=1000.0, timings=FrequencySolveTimings(assembly_s=1.2, solve_s=0.3, field_s=0.04))
    initialized = Signal(object, object, object)
    result_ready = Signal(object)
    system_result_ready = Signal(object)
    status = Signal(str)
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


def test_solve_controller_owns_worker_thread_and_completion_state(qapp, monkeypatch) -> None:
    app = QCoreApplication.instance() or QCoreApplication([])
    monkeypatch.setattr(controller_module, "SystemSolveWorker", _SolveWorkerStub)
    controller = SolveController()
    controller._streaming = True  # A new run must restore startup messages.
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
        "Preparing backend",
        "Solved 1/1 (1000.0 Hz) | Assembly 1.20s | Solve 0.30s | Field 0.04s",
    ]
    assert completions[0].phase == OperationPhase.COMPLETED
    assert completions[0].completed is True
    assert thread_state_at_completion == [(False, True)]
    assert controller.last_completion == completions[0]
    assert controller.last_error is None
    assert controller.active is False


def test_streaming_status_updates_only_for_completed_frequencies(qapp):
    controller = SolveController()
    controller._expected_count = 2
    controller._set_state(OperationPhase.RUNNING, "Initializing solver")
    statuses = []
    controller.status.connect(statuses.append)
    controller._on_status("Loading backend")
    controller._on_initialized(np.array([0.0]), np.array(["driver"]), None)
    controller._on_status("Assembling first frequency")
    assert controller.state.message == "Solving..."
    controller._on_result(_SolveWorkerStub.result)
    expected = "Solved 1/2 (1000.0 Hz) | Assembly 1.20s | Solve 0.30s | Field 0.04s"
    assert controller.state.message == expected
    controller._on_status("Computing field for next frequency")
    assert controller.state.message == expected
    assert statuses == ["Loading backend", expected]
    controller._on_result(SimpleNamespace(freq_hz=2000.0, timings=_SolveWorkerStub.result.timings))
    assert controller.state.message.startswith("Solved 2/2 (2000.0 Hz)")
    controller.cancel()
    controller._on_status("Backend still stopping")
    controller._on_result(_SolveWorkerStub.result)
    assert controller.state.phase == OperationPhase.CANCELLING
    assert controller.state.message == "Stopping solve"
    controller._on_failed("Backend failed")
    controller._on_status("Late backend detail")
    assert controller.state.phase == OperationPhase.FAILED
    assert controller.state.message == "Backend failed"


def test_frequency_status_is_visible_in_activity_overlay(main_window):
    controller = main_window.solve_controller
    controller._expected_count = 2
    controller._streaming = True
    controller._set_state(OperationPhase.RUNNING, "Solving...")
    controller._on_result(_SolveWorkerStub.result)
    expected = "Solved 1/2 (1000.0 Hz) | Assembly 1.20s | Solve 0.30s | Field 0.04s"
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
