"""Desktop GUI entrypoint for Ath4 waveguide generation and live BEM solving."""

from __future__ import annotations

import multiprocessing as mp
import sys
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
    from PySide6.QtGui import QIcon, QPixmap
    from PySide6.QtWidgets import QApplication, QSplashScreen
except ImportError as exc:  # pragma: no cover - exercised only by manual GUI launch
    raise SystemExit(
        "PySide6 is required for the GUI. Install the GUI extra with: "
        'python -m pip install -e ".[gui]"'
    ) from exc


APP_ROOT = Path(__file__).resolve().parents[2]
SPLASH_PATH = APP_ROOT / "assets" / "splash.png"
ICON_PATHS = tuple(APP_ROOT / "assets" / f"{size}.ico" for size in (32, 64, 128, 256))


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
    return QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)


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

    try:
        from blab.ui.main_window import MainWindow
    except ImportError as exc:  # pragma: no cover - exercised only by manual GUI launch
        if splash is not None:
            splash.close()
        raise SystemExit(_missing_gui_dependency_message(exc)) from exc

    window = MainWindow()
    if not app_icon.isNull():
        window.setWindowIcon(app_icon)
    window.show()
    if splash is not None:
        splash.finish(window)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
