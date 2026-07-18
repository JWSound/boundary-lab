import time

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QObject, Signal, Slot

import blab.ui.operation_controllers as controller_module
from blab.config import SimulationConfig
from blab.ui.application_state import OperationPhase
from blab.ui.operation_controllers import SolveController, SolveRequest


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
    assert controller.active is False
