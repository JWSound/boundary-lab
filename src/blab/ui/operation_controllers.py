"""Qt controllers that own background geometry and solve operation lifecycles."""

from __future__ import annotations

import time

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot

from blab.generators.base import GenerationCompleted, GenerationRequest
from blab.solve_eta import SolveProgress, format_solve_status, format_warm_up_status
from blab.ui.application_state import OperationPhase, OperationState, SolveCompletion
from blab.ui.generator_worker import GeneratorWorker
from blab.ui.system_solve import PreparedSystemSolve, SystemSolveWorker
from blab.warm_up_history import WarmUpHistory, WarmUpKey


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
        self.host_factory = None
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
        worker = (
            GeneratorWorker(request)
            if self.host_factory is None
            else GeneratorWorker(request, host=self.host_factory(request.document_id, request.provider_id))
        )
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
    def _on_generated(self, completed: GenerationCompleted) -> None:
        if self._request is not None and completed.request.request_id == self._request.request_id:
            self.completed.emit(completed)

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
    failed = Signal(str, str)
    finished = Signal(object)
    state_changed = Signal(object)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.state = OperationState()
        self._thread: QThread | None = None
        self._worker: SystemSolveWorker | None = None
        self.clock = time.perf_counter
        self.warm_up_history = WarmUpHistory()
        self._progress: SolveProgress | None = None
        self._expected_count = 0
        self._solved_count = 0
        self._warm_up_message = ""
        # Warm-up can pass long stretches without worker events, so elapsed time ticks on its own.
        self._ticker = QTimer(self)
        self._ticker.setInterval(1000)
        self._ticker.timeout.connect(self._show_warm_up)
        self._failed = False
        self._cancelled = False
        self.last_error: str | None = None
        self.last_completion: SolveCompletion | None = None

    @property
    def active(self) -> bool:
        return self.state.active

    def start(self, request: PreparedSystemSolve) -> bool:
        if self.active:
            return False
        self._begin_run(len(request.request.frequencies_hz))
        thread = QThread(self)
        worker = SystemSolveWorker(request)
        self._thread = thread
        self._worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.initialized.connect(self._on_initialized)
        worker.result_ready.connect(self._on_result)
        worker.status.connect(self._on_status)
        worker.warm_up_key.connect(self._on_warm_up_key)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_finished)
        thread.finished.connect(thread.deleteLater)
        thread.start()
        return True

    def _begin_run(self, expected_count: int) -> None:
        self._expected_count = expected_count
        self._solved_count = 0
        self._progress = SolveProgress(expected_count, clock=self.clock, history=self.warm_up_history)
        self._failed = False
        self._cancelled = False
        self.last_error = None
        self.last_completion = None
        self._warm_up_message = "Initializing solver"
        self._set_state(OperationPhase.RUNNING, self._warm_up_text())
        self._ticker.start()

    def _warm_up_text(self) -> str:
        progress = self._progress
        return format_warm_up_status(self._warm_up_message, progress.elapsed_s(), progress.predicted_warm_up_s)

    @Slot()
    def _show_warm_up(self) -> None:
        if self.state.phase != OperationPhase.RUNNING:
            return
        message = self._warm_up_text()
        self._set_state(OperationPhase.RUNNING, message)
        self.status.emit(message)

    @Slot(object)
    def _on_warm_up_key(self, key: WarmUpKey) -> None:
        self._progress.start_warm_up(key)
        if self._progress.warming_up:
            self._show_warm_up()

    def cancel(self) -> None:
        if not self.active:
            return
        self._ticker.stop()
        self._cancelled = True
        self._set_state(OperationPhase.CANCELLING, "Stopping solve")
        if self._worker is not None:
            self._worker.stop()

    @Slot(object)
    def _on_result(self, result: object) -> None:
        if self._failed:
            return
        self.result_ready.emit(result)
        if self._failed:
            return
        self._ticker.stop()
        self._solved_count += 1
        # Results arriving while cancelling neither count as a finished warm-up nor need an ETA.
        if self.state.phase == OperationPhase.RUNNING:
            timing = self._progress.frequency_fields()
            message = format_solve_status(
                self._solved_count, self._expected_count, timing["elapsed_s"], timing["eta_s"]
            )
            self._set_state(OperationPhase.RUNNING, message)
            self.status.emit(message)

    def reject_result(self, message: str) -> None:
        """Stop a run whose canonical result could not be accumulated or projected."""
        self._failed = True
        if self._worker is not None:
            self._worker.stop()
        self._on_failed(message)

    @Slot(object, object, object)
    def _on_initialized(self, angles, radiator_names, sphere_metadata) -> None:
        self._warm_up_message = "Solving..."
        if self.state.phase == OperationPhase.RUNNING and self._progress.warming_up:
            self._set_state(OperationPhase.RUNNING, self._warm_up_text())
        self.initialized.emit(angles, radiator_names, sphere_metadata)

    @Slot(str)
    def _on_status(self, message: str) -> None:
        # Backend detail remains in the worker's logs. Warm-up phases are shown;
        # after the first result, the last completed frequency stays until the next.
        if self._solved_count or self.state.phase != OperationPhase.RUNNING:
            return
        self._warm_up_message = message
        self._show_warm_up()

    @Slot(str, str)
    def _on_failed(self, message: str, details: str = "") -> None:
        self._ticker.stop()
        self._failed = True
        self.last_error = message
        self._set_state(OperationPhase.FAILED, message)
        self.failed.emit(message, details)

    @Slot()
    def _on_finished(self) -> None:
        self._ticker.stop()
        elapsed_s = 0.0 if self._progress is None else self._progress.elapsed_s()
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
        self._progress = None
        self.finished.emit(completion)

    def _set_state(self, phase: OperationPhase, message: str) -> None:
        self.state = OperationState(phase=phase, message=message)
        self.state_changed.emit(self.state)
