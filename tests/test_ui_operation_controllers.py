import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QObject, Signal, Slot

import blab.ui.operation_controllers as controller_module
from blab.config import SimulationConfig
from blab.ui.application_state import OperationPhase
from blab.ui.operation_controllers import GeometryController, SolveController, SolveRequest
from blab.ui.solve_worker import SolveWorker


class _SolveWorkerStub(QObject):
    initialized = Signal(object, object, object)
    result_ready = Signal(object)
    status = Signal(str)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, *_args, **_kwargs):
        super().__init__()
        self.stopped = False

    @Slot()
    def run(self) -> None:
        self.initialized.emit(np.array([0.0]), np.array(["driver"]), None)
        self.result_ready.emit("frequency-result")
        self.finished.emit()

    @Slot()
    def stop(self) -> None:
        self.stopped = True


def test_solve_controller_owns_worker_thread_and_completion_state(monkeypatch) -> None:
    app = QCoreApplication.instance() or QCoreApplication([])
    monkeypatch.setattr(controller_module, "SolveWorker", _SolveWorkerStub)
    controller = SolveController()
    results = []
    completions = []
    controller.result_ready.connect(results.append)
    controller.finished.connect(completions.append)

    started = controller.start(
        SolveRequest(
            config=SimulationConfig(mesh_file="speaker.msh"),
            ordered_frequencies=np.array([1000.0]),
            backend_id="local",
            server_url="http://127.0.0.1:8765",
        )
    )
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
    assert results == ["frequency-result"]
    assert completions[0].phase == OperationPhase.COMPLETED
    assert completions[0].completed is True
    assert thread_state_at_completion == [(False, True)]
    assert controller.last_completion == completions[0]
    assert controller.last_error is None
    assert controller.active is False


def test_controllers_preserve_latest_failure_for_diagnostics() -> None:
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


def test_solve_worker_logs_backend_detail_without_emitting_visible_status(monkeypatch, caplog) -> None:
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
        def create_session(self, request):
            request.status_callback("initializing backend detail")
            return Session(request)

    monkeypatch.setattr("blab.ui.solve_worker.create_backend", lambda *_args, **_kwargs: Backend())
    worker = SolveWorker(
        SimulationConfig(mesh_file="speaker.msh"),
        np.array([1000.0]),
    )
    statuses = []
    worker.status.connect(statuses.append)

    with caplog.at_level("INFO", logger="blab.ui.solve_worker"):
        worker.run()

    assert statuses == []
    assert "initializing backend detail" in caplog.text
    assert "assembling backend detail" in caplog.text
