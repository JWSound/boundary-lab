"""Qt worker wrapper for cancellable geometry generation."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from blab.generators.base import GenerationCancelledError, GenerationRequest, GeneratorSession
from blab.generators.registry import create_generator


class GeneratorWorker(QObject):
    generated = Signal(object)
    status = Signal(str)
    failed = Signal(str)
    cancelled = Signal()
    finished = Signal()

    def __init__(self, request: GenerationRequest):
        super().__init__()
        self.request = request
        self._session: GeneratorSession | None = None
        self._stop = False

    @Slot()
    def run(self) -> None:
        try:
            backend = create_generator(self.request.provider_id, **dict(self.request.provider_options))
            self._session = backend.create_session(self.request)
            if self._stop:
                self._session.stop()
                self.cancelled.emit()
                return
            result = self._session.generate(
                status_callback=self.status.emit,
                stop_requested=lambda: self._stop,
            )
            if self._stop:
                self.cancelled.emit()
                return
            self.generated.emit(result)
        except GenerationCancelledError:
            self.cancelled.emit()
        except Exception as exc:
            if self._stop:
                self.cancelled.emit()
            else:
                self.failed.emit(str(exc))
        finally:
            self.finished.emit()

    @Slot()
    def stop(self) -> None:
        self._stop = True
        self.status.emit("Stopping geometry generation...")
        if self._session is not None:
            self._session.stop()
