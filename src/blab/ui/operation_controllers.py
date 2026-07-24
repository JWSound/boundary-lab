"""Qt controllers that own background geometry and solve operation lifecycles."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from blab.config import SimulationConfig
from blab.generators.base import GeneratedGeometry, GenerationCompleted, GenerationRequest
from blab.ui.application_state import OperationPhase, OperationState, SolveCompletion
from blab.ui.generator_worker import GeneratorWorker
from blab.ui.solve_worker import SolveWorker


@dataclass(frozen=True)
class SolveRequest:
    config: SimulationConfig
    ordered_frequencies: np.ndarray
    backend_id: str
    server_url: str
    server_access_token: str = ""
    worker_count: int = 1


class GeometryController(QObject):
    completed = Signal(object)
    status = Signal(str)
    failed = Signal(str)
    cancelled = Signal()
    finished = Signal()
    state_changed = Signal(object)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.state = OperationState()
        self._request: GenerationRequest | None = None
        self._thread: QThread | None = None
        self._worker: GeneratorWorker | None = None
        self._failed = False
        self._cancelled = False
        self.last_error: str | None = None

    @property
    def active(self) -> bool:
        return self.state.active

    def start(self, request: GenerationRequest) -> bool:
        if self.active:
            return False
        self._request = request
        self._failed = False
        self._cancelled = False
        self.last_error = None
        self._set_state(OperationPhase.RUNNING, "Generating geometry")
        thread = QThread(self)
        worker = GeneratorWorker(request)
        self._thread = thread
        self._worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.generated.connect(self._on_generated)
        worker.status.connect(self._on_status)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_finished)
        thread.finished.connect(thread.deleteLater)
        thread.start()
        return True

    def cancel(self) -> None:
        if not self.active:
            return
        self._set_state(OperationPhase.CANCELLING, "Stopping geometry generation")
        if self._worker is not None:
            self._worker.stop()

    @Slot(object)
    def _on_generated(self, result: GeneratedGeometry) -> None:
        if self._request is not None:
            self.completed.emit(GenerationCompleted(self._request, result))

    @Slot(str)
    def _on_status(self, message: str) -> None:
        self._set_state(self.state.phase, message)
        self.status.emit(message)

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self._failed = True
        self.last_error = message
        self._set_state(OperationPhase.FAILED, message)
        self.failed.emit(message)

    @Slot()
    def _on_cancelled(self) -> None:
        self._cancelled = True
        self._set_state(OperationPhase.CANCELLED, "Geometry generation stopped")
        self.cancelled.emit()

    @Slot()
    def _on_finished(self) -> None:
        if not self._failed and not self._cancelled:
            self._set_state(OperationPhase.COMPLETED, "Geometry generation complete")
        self._worker = None
        self._thread = None
        self._request = None
        self.finished.emit()

    def _set_state(self, phase: OperationPhase, message: str) -> None:
        self.state = OperationState(phase=phase, message=message)
        self.state_changed.emit(self.state)


class SolveController(QObject):
    initialized = Signal(object, object, object)
    result_ready = Signal(object)
    status = Signal(str)
    failed = Signal(str)
    finished = Signal(object)
    state_changed = Signal(object)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.state = OperationState()
        self._thread: QThread | None = None
        self._worker: SolveWorker | None = None
        self._started_at: float | None = None
        self._expected_count = 0
        self._solved_count = 0
        self._failed = False
        self._cancelled = False
        self.last_error: str | None = None
        self.last_completion: SolveCompletion | None = None

    @property
    def active(self) -> bool:
        return self.state.active

    def start(self, request: SolveRequest) -> bool:
        if self.active:
            return False
        self._expected_count = int(request.ordered_frequencies.size)
        self._solved_count = 0
        self._failed = False
        self._cancelled = False
        self.last_error = None
        self.last_completion = None
        self._started_at = time.perf_counter()
        self._set_state(OperationPhase.RUNNING, "Initializing solver")
        thread = QThread(self)
        worker = SolveWorker(
            request.config,
            request.ordered_frequencies,
            worker_count=request.worker_count,
            backend_id=request.backend_id,
            server_url=request.server_url,
            server_access_token=request.server_access_token,
        )
        self._thread = thread
        self._worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.initialized.connect(self.initialized)
        worker.result_ready.connect(self._on_result)
        worker.status.connect(self._on_status)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_finished)
        thread.finished.connect(thread.deleteLater)
        thread.start()
        return True

    def cancel(self) -> None:
        if not self.active:
            return
        self._cancelled = True
        self._set_state(OperationPhase.CANCELLING, "Stopping solve")
        if self._worker is not None:
            self._worker.stop()

    @Slot(object)
    def _on_result(self, result: object) -> None:
        self._solved_count += 1
        self.result_ready.emit(result)

    @Slot(str)
    def _on_status(self, message: str) -> None:
        self._set_state(self.state.phase, message)
        self.status.emit(message)

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self._failed = True
        self.last_error = message
        self._set_state(OperationPhase.FAILED, message)
        self.failed.emit(message)

    @Slot()
    def _on_finished(self) -> None:
        elapsed_s = 0.0 if self._started_at is None else time.perf_counter() - self._started_at
        if self._cancelled:
            phase = OperationPhase.CANCELLED
        elif self._failed:
            phase = OperationPhase.FAILED
        else:
            phase = OperationPhase.COMPLETED
        self._set_state(phase, "Solve finished")
        completion = SolveCompletion(
            phase=phase,
            solved_count=self._solved_count,
            expected_count=self._expected_count,
            elapsed_s=elapsed_s,
        )
        self.last_completion = completion
        self._worker = None
        self._thread = None
        self._started_at = None
        self.finished.emit(completion)

    def _set_state(self, phase: OperationPhase, message: str) -> None:
        self.state = OperationState(phase=phase, message=message)
        self.state_changed.emit(self.state)
