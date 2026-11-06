"""Capabilities of the selected physical-system solver backend."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable

from PySide6.QtCore import QObject, QProcess, QTimer, Signal

from blab.solvers.beat_engine_runtime import METAL_STARTUP_TIMEOUT_S
from blab.solvers.engine_distribution import engine_paths
from blab.solvers.registry import backend_info, packaged_backend_available
from blab.ui.settings import GuiPreferences


class BackendHealthController(QObject):
    capability_changed = Signal(str)
    availability_failed = Signal(str)

    def __init__(self, parent: QObject | None, *, preferences: Callable[[], GuiPreferences]) -> None:
        super().__init__(parent)
        self._read_preferences = preferences
        self._probe = None
        self._metal_available = False
        self._output = ""
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._timeout)

    def check_selected_backend(self) -> None:
        """Check Metal asynchronously; successful checks never change the UI."""
        if (
            sys.platform != "darwin"
            or self._read_preferences().solve_backend != "beat_metal"
            or not packaged_backend_available("beat_metal")
            or self._metal_available
            or self._probe is not None
        ):
            return
        program, arguments = metal_probe_command()
        process = QProcess(self)
        self._probe = process
        self._output = ""
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._drain)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._error)
        self._timer.start(int(METAL_STARTUP_TIMEOUT_S * 1000))
        process.start(program, arguments)

    def _drain(self) -> None:
        if self._probe is not None:
            self._output = (self._output + bytes(self._probe.readAllStandardOutput()).decode(errors="replace"))[-16000:]

    def _finished(self, code, status) -> None:
        self._drain()
        self._complete(code == 0 and status == QProcess.NormalExit)

    def _error(self, error) -> None:
        if error == QProcess.FailedToStart and self._probe is not None:
            self._output += self._probe.errorString()
            self._complete(False)

    def _timeout(self) -> None:
        self._output += "\nMetal availability check timed out."
        self.close()
        self._report_failure()

    def _complete(self, available: bool) -> None:
        self._timer.stop()
        process, self._probe = self._probe, None
        if process is None:
            return
        process.deleteLater()
        self._metal_available = available
        if not available:
            self._report_failure()

    def _report_failure(self) -> None:
        logging.getLogger(__name__).warning("Metal availability check failed: %s", self._output)
        if self._read_preferences().solve_backend == "beat_metal":
            self.availability_failed.emit("Metal is unavailable. Select CPU in Preferences to continue.")

    def close(self) -> None:
        self._timer.stop()
        process, self._probe = self._probe, None
        if process is not None:
            process.kill()
            process.waitForFinished(1000)
            process.deleteLater()

    def supports_symmetry(self, backend_id: str) -> bool:
        return backend_info(backend_id).capabilities.supports_symmetry

    def selected_backend_supports_symmetry(self) -> bool:
        return self.supports_symmetry(self._read_preferences().solve_backend)

    def effective_symmetry(self, symmetry: str, preferences: GuiPreferences) -> str:
        return symmetry if self.supports_symmetry(preferences.solve_backend) else "off"


def metal_probe_command() -> tuple[str, list[str]]:
    return os.environ.get("BLAB_JULIA_EXE", "julia"), [
        "--startup-file=no",
        "--threads=2",
        f"--project={engine_paths('metal').project}",
        "-e",
        'using Metal; Metal.functional() || error("Metal unavailable")',
    ]
