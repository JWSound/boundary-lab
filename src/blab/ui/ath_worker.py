"""Qt worker wrapper for cancellable Ath geometry generation."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from blab.ath import (
    AthCancelledError,
    AthProcessRunner,
    HornlabMesherRunner,
    clean_ath_mesh_output,
    generate_waveguide_mesh,
)


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
        self._mesher_runner = HornlabMesherRunner()
        self._stop = False

    @Slot()
    def run(self) -> None:
        try:
            # Prefer the fast native HornLab mesher; fall back to Ath (Ath.exe via
            # Wine on macOS/Linux) for geometries the mesher can't handle. The runner
            # is threaded through so cancellation (stop()) still reaches an Ath run.
            raw_result = generate_waveguide_mesh(
                ath_exe=self.ath_exe,
                config_text=self.config_text,
                run_root=self.run_root,
                case_name=self.case_name,
                runner=self._runner,
                mesher_runner=self._mesher_runner,
                on_status=self.status.emit,
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
        self.status.emit("Stopping mesh generation...")
        self._mesher_runner.stop()
        self._runner.stop()
