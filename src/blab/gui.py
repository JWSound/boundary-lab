"""Desktop GUI entrypoint for waveguide generation and live BEM solving."""

from __future__ import annotations

import faulthandler
import logging
import multiprocessing as mp
import os
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

from blab.paths import APP_ROOT


def _missing_gui_dependency_message(exc: ImportError) -> str:
    missing = getattr(exc, "name", None) or "a GUI dependency"
    if missing == "_cl":
        return (
            "pyopencl is installed incorrectly or is missing its compiled _cl extension. "
            "Reinstall it with: python -m pip install --force-reinstall --no-cache-dir pyopencl"
        )
    return f'{missing} is required for the GUI. Reinstall the GUI extra with: python -m pip install -e ".[gui]"'


try:
    from PySide6.QtCore import QRect, Qt
    from PySide6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPixmap
    from PySide6.QtWidgets import QApplication, QMessageBox, QSplashScreen
except ImportError as exc:  # pragma: no cover - exercised only by manual GUI launch
    raise SystemExit(
        'PySide6 is required for the GUI. Install the GUI extra with: python -m pip install -e ".[gui]"'
    ) from exc


SPLASH_PATH = APP_ROOT / "assets" / "splash.png"
ICON_PATHS = tuple(APP_ROOT / "assets" / f"{size}.ico" for size in (32, 64, 128, 256))
STARTUP_TIMEOUT_SECONDS = 45
SPLASH_TEXT_COLOR = QColor("#1c1c1c")
SPLASH_TEXT_FONT = QFont("Courier New", 10)
SPLASH_TEXT_PANEL_COLOR = QColor(250, 250, 250, 225)
SPLASH_TEXT_PANEL_BORDER_COLOR = QColor(28, 28, 28, 55)
SPLASH_TEXT_CONTENT_INSET_PX = 15
SPLASH_TEXT_HORIZONTAL_PADDING_PX = 10
SPLASH_TEXT_VERTICAL_PADDING_PX = 5
SPLASH_ALPHA_VISIBILITY_THRESHOLD = 32


class StartupReporter:
    def __init__(self, app: QApplication, splash: QSplashScreen | None):
        self.app = app
        self.splash = splash
        self.stage = "Starting"
        self.finished = threading.Event()
        self.log_path = _startup_log_path()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path.open("a", encoding="utf-8", buffering=1)
        self._closed = False
        self._previous_faulthandler_enabled = faulthandler.is_enabled()
        self._faulthandler_enabled = False
        try:
            faulthandler.enable(file=self._log_file, all_threads=True)
            self._faulthandler_enabled = True
        except (OSError, RuntimeError):
            pass
        self._runtime_logger = logging.getLogger("blab.runtime")
        self._previous_runtime_log_level = self._runtime_logger.level
        self._previous_runtime_log_propagate = self._runtime_logger.propagate
        self._runtime_handler = logging.StreamHandler(self._log_file)
        self._runtime_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        self._runtime_logger.addHandler(self._runtime_handler)
        self._runtime_logger.setLevel(logging.INFO)
        self._runtime_logger.propagate = False
        self._log("Boundary Lab GUI startup")
        if self.splash is not None:
            self.splash.setFont(SPLASH_TEXT_FONT)

    def finish_startup(self) -> None:
        if self.finished.is_set():
            return
        self.finished.set()
        faulthandler.cancel_dump_traceback_later()
        self._log("Startup completed")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.finished.set()
        faulthandler.cancel_dump_traceback_later()
        self._log("Boundary Lab GUI session ended")
        self._runtime_logger.removeHandler(self._runtime_handler)
        self._runtime_handler.close()
        self._runtime_logger.setLevel(self._previous_runtime_log_level)
        self._runtime_logger.propagate = self._previous_runtime_log_propagate
        if self._faulthandler_enabled:
            if self._previous_faulthandler_enabled:
                faulthandler.enable(all_threads=True)
            else:
                faulthandler.disable()
        self._log_file.close()

    def update(self, stage: str) -> None:
        self.stage = stage
        self._log(stage)
        if self.splash is not None:
            self.splash.showMessage(
                stage,
                Qt.AlignBottom | Qt.AlignLeft,
                SPLASH_TEXT_COLOR,
            )
            self.app.processEvents()

    def log(self, message: str) -> None:
        self._log(message)

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


def _splash_message_anchor_rect(pixmap: QPixmap) -> QRect:
    """Return the lowest substantial opaque span in a splash pixmap.

    Splash artwork may include transparent padding. Anchoring status text to
    the widget rectangle would place it in that padding, where it can disappear
    against the desktop. The lowest visible horizontal run is a useful proxy
    for the bottom edge of the artwork without coupling the UI to one asset's
    dimensions.
    """
    fallback = pixmap.rect()
    image = pixmap.toImage()
    if image.isNull() or not image.hasAlphaChannel():
        return fallback

    image = image.convertToFormat(QImage.Format_RGBA8888)
    width = image.width()
    height = image.height()
    if width <= 0 or height <= 0:
        return fallback

    pixels = image.constBits()
    bytes_per_line = image.bytesPerLine()
    minimum_run_width = max(8, width // 20)

    for y in range(height - 1, -1, -1):
        row_offset = y * bytes_per_line
        longest_start = -1
        longest_width = 0
        run_start = -1

        for x in range(width):
            alpha = pixels[row_offset + x * 4 + 3]
            if alpha >= SPLASH_ALPHA_VISIBILITY_THRESHOLD:
                if run_start < 0:
                    run_start = x
                continue
            if run_start >= 0:
                run_width = x - run_start
                if run_width > longest_width:
                    longest_start = run_start
                    longest_width = run_width
                run_start = -1

        if run_start >= 0:
            run_width = width - run_start
            if run_width > longest_width:
                longest_start = run_start
                longest_width = run_width

        if longest_width >= minimum_run_width:
            return QRect(longest_start, 0, longest_width, y + 1)

    return fallback


class BoundaryLabSplashScreen(QSplashScreen):
    def __init__(self, pixmap: QPixmap, flags=Qt.WindowFlags()):
        super().__init__(pixmap, flags)
        self._message_anchor_rect = _splash_message_anchor_rect(pixmap)
        self._last_message_panel_rect = QRect()

    def drawContents(self, painter) -> None:  # noqa: N802 - Qt override
        message = self.message()
        if not message:
            return

        painter.setPen(SPLASH_TEXT_COLOR)
        painter.setFont(SPLASH_TEXT_FONT)
        metrics = painter.fontMetrics()
        available_width = max(1, self._message_anchor_rect.width() - 2 * SPLASH_TEXT_CONTENT_INSET_PX)
        panel_width = min(
            metrics.horizontalAdvance(message) + 2 * SPLASH_TEXT_HORIZONTAL_PADDING_PX,
            available_width,
        )
        panel_height = metrics.height() + 2 * SPLASH_TEXT_VERTICAL_PADDING_PX
        panel_left = self._message_anchor_rect.left() + SPLASH_TEXT_CONTENT_INSET_PX
        panel_bottom = self._message_anchor_rect.bottom() - SPLASH_TEXT_CONTENT_INSET_PX
        panel_rect = QRect(
            panel_left,
            panel_bottom - panel_height + 1,
            panel_width,
            panel_height,
        )
        self._last_message_panel_rect = panel_rect

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(SPLASH_TEXT_PANEL_BORDER_COLOR)
        painter.setBrush(SPLASH_TEXT_PANEL_COLOR)
        painter.drawRoundedRect(panel_rect, 4, 4)
        painter.setPen(SPLASH_TEXT_COLOR)
        text_rect = panel_rect.adjusted(
            SPLASH_TEXT_HORIZONTAL_PADDING_PX,
            0,
            -SPLASH_TEXT_HORIZONTAL_PADDING_PX,
            0,
        )
        elided_message = metrics.elidedText(message, Qt.ElideRight, text_rect.width())
        painter.drawText(text_rect, int(Qt.AlignLeft | Qt.AlignVCenter), elided_message)
        painter.restore()


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
        reporter.update("Checking Python dependencies...")
        from blab.startup_checks import check_python_dependency_versions, format_startup_check_results

        for line in format_startup_check_results(check_python_dependency_versions()):
            reporter.log(f"Dependency {line}")

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
        reporter.finish_startup()
    except Exception as exc:  # pragma: no cover - exercised only by manual GUI launch
        if splash is not None:
            splash.close()
        reporter.exception("Boundary Lab could not open", exc)
        reporter.close()
        raise

    try:
        exit_code = app.exec()
    finally:
        reporter.close()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
