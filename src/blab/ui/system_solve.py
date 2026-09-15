"""Qt worker adapter for physical-system solves and existing live plots."""

from __future__ import annotations

import logging
import os
from dataclasses import replace

from PySide6.QtCore import QObject, Signal, Slot

from blab.solvers.coupled_backend import PhysicalSystemProductionBackend
from blab.system_solve import (
    PreparedSystemSolve,
    canonicalize_observation_result,
    prepare_coupled_ui_solve,
    prepare_system_solve,
    supports_exterior_system_protocol,
)

LOGGER = logging.getLogger(__name__)


class SystemSolveWorker(QObject):
    """Run the selected physical-system backend and emit canonical results."""

    initialized = Signal(object, object, object)
    result_ready = Signal(object)
    status = Signal(str)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, prepared: PreparedSystemSolve):
        super().__init__()
        self.prepared = prepared
        self._stop = False
        self._session = None

    @Slot()
    def run(self) -> None:
        try:
            request = replace(self.prepared.request, status_callback=self._log_backend_status)
            self._run_physical_system(request)
        except Exception as exc:
            if not self._stop:
                self.failed.emit(str(exc))
        finally:
            self.finished.emit()

    def _run_physical_system(self, request) -> None:
        bem_backend = self.prepared.backend_id.removeprefix("beat_")
        if self.prepared.backend_id == "beat_remote":
            from blab.remote import RemoteBackend

            options = self.prepared.remote_options or {}
            backend = RemoteBackend(
                options.get("url", "http://127.0.0.1:8765"),
                token=options.get("access_key") or None,
            )
            self.status.emit("Connecting to server…")
        else:
            backend = PhysicalSystemProductionBackend(
                bem_backend=bem_backend,
                julia_executable=os.environ.get("BLAB_JULIA_EXE", "julia"),
            )
        if self._stop:
            return
        session = backend.create_system_session(request)
        self._session = session
        self.initialized.emit(
            self.prepared.polar_angle_deg,
            self.prepared.excitation_component_names,
            self.prepared.sphere_metadata,
        )
        for result in session.solve_stream(stop_requested=lambda: self._stop):
            self.result_ready.emit(canonicalize_observation_result(self.prepared, result))

    @Slot()
    def stop(self) -> None:
        self._stop = True
        if self._session is not None and self.prepared.backend_id != "beat_remote":
            self._session.stop()

    def _log_backend_status(self, message: str) -> None:
        self.status.emit(message)
        LOGGER.info("Physical-system solver backend status: %s", message)


__all__ = [
    "SystemSolveWorker",
    "PreparedSystemSolve",
    "prepare_system_solve",
    "supports_exterior_system_protocol",
    "CoupledSolveWorker",
    "CoupledUiSolveRequest",
    "prepare_coupled_ui_solve",
]


CoupledUiSolveRequest = PreparedSystemSolve
CoupledSolveWorker = SystemSolveWorker
