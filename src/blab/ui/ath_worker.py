"""Qt worker wrapper for cancellable Ath geometry generation."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from blab.ath import AthCancelledError, AthProcessRunner, clean_ath_mesh_output


class AthGenerationWorker(QObject):
    generated = Signal(object)
    status = Signal(str)
    failed = Signal(str)
    cancelled = Signal()
    finished = Signal()

    def __init__(
        self,
        *,
        ath_exe: Path,
        config_text: str,
        run_root: Path,
        case_name: str,
    ):
        super().__init__()
        self.ath_exe = ath_exe
        self.config_text = config_text
        self.run_root = run_root
        self.case_name = case_name
        self._runner = AthProcessRunner()
        self._stop = False

    @Slot()
    def run(self) -> None:
        try:
            self.status.emit("Running Ath...")
            raw_result = self._runner.run(
                ath_exe=self.ath_exe,
                config_text=self.config_text,
                run_root=self.run_root,
                case_name=self.case_name,
            )
            if self._stop:
                self.cancelled.emit()
                return
            self.status.emit("Cleaning generated mesh...")
            result = clean_ath_mesh_output(raw_result)
            if self._stop:
                self.cancelled.emit()
                return
            self.generated.emit(result)
        except AthCancelledError:
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
        self.status.emit("Stopping Ath generation...")
        self._runner.stop()
