"""Qt worker wrapper for cancellable geometry generation."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from blab.generators.base import GenerationCancelledError, GenerationRequest, GeneratorSession, complete_generation
from blab.generators.registry import create_generator


class GeneratorWorker(QObject):
    generated = Signal(object)
    status = Signal(str)
    failed = Signal(str)
    cancelled = Signal()
    finished = Signal()

    def __init__(self, request: GenerationRequest, *, host=None):
        super().__init__()
        self.request = request
        self.host = host
        self._session: GeneratorSession | None = None
        self._stop = False

    @Slot()
    def run(self) -> None:
        try:
            backend = create_generator(self.request.provider_id, **dict(self.request.provider_options))
            if self.host is not None and callable(getattr(backend, "bind_host", None)):
                backend.bind_host(self.host)
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
            self.generated.emit(complete_generation(self.request, result))
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
