"""Desktop GUI entrypoint for Ath4 waveguide generation and live BEM solving."""

from __future__ import annotations

import faulthandler
import multiprocessing as mp
import os
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path


def _missing_gui_dependency_message(exc: ImportError) -> str:
    missing = getattr(exc, "name", None) or "a GUI dependency"
    if missing == "_cl":
        return (
            "pyopencl is installed incorrectly or is missing its compiled _cl extension. "
            "Reinstall it with: python -m pip install --force-reinstall --no-cache-dir pyopencl"
        )
    return (
        f"{missing} is required for the GUI. Reinstall the GUI extra with: "
        'python -m pip install -e ".[gui]"'
    )


try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QFont, QIcon, QPixmap
    from PySide6.QtWidgets import QApplication, QMessageBox, QSplashScreen
except ImportError as exc:  # pragma: no cover - exercised only by manual GUI launch
    raise SystemExit(
        "PySide6 is required for the GUI. Install the GUI extra with: "
        'python -m pip install -e ".[gui]"'
    ) from exc


APP_ROOT = Path(__file__).resolve().parents[2]
SPLASH_PATH = APP_ROOT / "assets" / "splash.png"
ICON_PATHS = tuple(APP_ROOT / "assets" / f"{size}.ico" for size in (32, 64, 128, 256))
STARTUP_TIMEOUT_SECONDS = 45
SPLASH_TEXT_COLOR = QColor("#1c1c1c")
SPLASH_TEXT_FONT = QFont("Courier New", 10)
SPLASH_TEXT_BOTTOM_MARGIN_PX = 15


class StartupReporter:
    def __init__(self, app: QApplication, splash: QSplashScreen | None):
        self.app = app
        self.splash = splash
        self.stage = "Starting"
        self.finished = threading.Event()
        self.log_path = _startup_log_path()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path.open("a", encoding="utf-8")
        self._log("Boundary Lab GUI startup")
        if self.splash is not None:
            self.splash.setFont(SPLASH_TEXT_FONT)

    def close(self) -> None:
        self.finished.set()
        faulthandler.cancel_dump_traceback_later()
        self._log("Startup completed")
        self._log_file.close()

    def update(self, stage: str) -> None:
        self.stage = stage
        self._log(stage)
        if self.splash is not None:
            self.splash.showMessage(
                stage,
                Qt.AlignBottom | Qt.AlignHCenter,
                SPLASH_TEXT_COLOR,
            )
            self.app.processEvents()

    def exception(self, title: str, exc: BaseException) -> None:
        details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self._log(f"{title}: {exc}\n{details}")
        QMessageBox.critical(None, title, f"{exc}\n\nStartup log:\n{self.log_path}")

    def start_watchdog(self) -> None:
        faulthandler.dump_traceback_later(
            STARTUP_TIMEOUT_SECONDS,
            repeat=False,
            file=self._log_file,
        )
        thread = threading.Thread(target=self._watchdog, name="BoundaryLabStartupWatchdog", daemon=True)
        thread.start()

    def _watchdog(self) -> None:
        if self.finished.wait(STARTUP_TIMEOUT_SECONDS):
            return
        message = (
            "Boundary Lab is still starting after "
            f"{STARTUP_TIMEOUT_SECONDS} seconds.\n\n"
            f"Current startup step: {self.stage}\n\n"
            f"Startup log:\n{self.log_path}"
        )
        self._log(message)
        _show_native_error_box("Boundary Lab startup stalled", message)

    def _log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log_file.write(f"[{timestamp}] {message}\n")
        self._log_file.flush()


class BoundaryLabSplashScreen(QSplashScreen):
    def drawContents(self, painter) -> None:  # noqa: N802 - Qt override
        painter.setPen(SPLASH_TEXT_COLOR)
        painter.setFont(SPLASH_TEXT_FONT)
        painter.drawText(
            self.rect().adjusted(0, 0, 0, -SPLASH_TEXT_BOTTOM_MARGIN_PX),
            int(Qt.AlignBottom | Qt.AlignHCenter),
            self.message(),
        )


def _startup_log_path() -> Path:
    root = os.environ.get("LOCALAPPDATA")
    if root:
        base = Path(root) / "Boundary Lab" / "logs"
    else:
        base = Path.home() / ".boundary-lab" / "logs"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return base / f"startup-{timestamp}.log"


def _show_native_error_box(title: str, message: str) -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, title, 0x10)
    except Exception:
        pass


def _set_windows_app_user_model_id() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("BoundaryLab.Beta")
    except Exception:
        pass


def _create_app_icon() -> QIcon:
    icon = QIcon()
    for path in ICON_PATHS:
        if path.exists():
            icon.addFile(str(path))
    return icon


def _create_splash_screen() -> QSplashScreen | None:
    if not SPLASH_PATH.exists():
        return None
    pixmap = QPixmap(str(SPLASH_PATH))
    if pixmap.isNull():
        return None
    return BoundaryLabSplashScreen(pixmap, Qt.WindowStaysOnTopHint)


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    del argv, prog
    mp.freeze_support()
    _set_windows_app_user_model_id()
    app = QApplication(sys.argv)
    app_icon = _create_app_icon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)

    splash = _create_splash_screen()
    if splash is not None:
        splash.show()
        app.processEvents()

    reporter = StartupReporter(app, splash)
    reporter.start_watchdog()
    try:
        reporter.update("Loading main window modules...")
        from blab.ui.main_window import MainWindow
    except ImportError as exc:  # pragma: no cover - exercised only by manual GUI launch
        if splash is not None:
            splash.close()
        message = _missing_gui_dependency_message(exc)
        reporter.exception("Boundary Lab could not start", RuntimeError(message))
        reporter.close()
        raise SystemExit(message) from exc
    except Exception as exc:  # pragma: no cover - exercised only by manual GUI launch
        if splash is not None:
            splash.close()
        reporter.exception("Boundary Lab could not load", exc)
        reporter.close()
        raise

    try:
        reporter.update("Creating main window...")
        window = MainWindow(startup_status=reporter.update)
        if not app_icon.isNull():
            window.setWindowIcon(app_icon)
        reporter.update("Showing main window...")
        window.show()
        if splash is not None:
            splash.finish(window)
        reporter.close()
    except Exception as exc:  # pragma: no cover - exercised only by manual GUI launch
        if splash is not None:
            splash.close()
        reporter.exception("Boundary Lab could not open", exc)
        reporter.close()
        raise

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
