"""Qt worker for background solve-server health probes."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from blab.solvers.http_server import query_server_health


class ServerHealthCheckWorker(QObject):
    succeeded = Signal(str, object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, server_url: str, *, timeout_s: float = 5.0):
        super().__init__()
        self.server_url = (server_url or "http://127.0.0.1:8765").rstrip("/")
        self.timeout_s = timeout_s

    @Slot()
    def run(self) -> None:
        try:
            payload = query_server_health(self.server_url, timeout_s=self.timeout_s)
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(self.server_url, payload)
        finally:
            self.finished.emit()
