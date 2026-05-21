"""Desktop GUI entrypoint for Ath4 waveguide generation and live BEM solving."""

from __future__ import annotations

import multiprocessing as mp
import sys


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
    from PySide6.QtWidgets import QApplication
except ImportError as exc:  # pragma: no cover - exercised only by manual GUI launch
    raise SystemExit(
        "PySide6 is required for the GUI. Install the GUI extra with: "
        'python -m pip install -e ".[gui]"'
    ) from exc

try:
    from blab.ui.main_window import MainWindow
except ImportError as exc:  # pragma: no cover - exercised only by manual GUI launch
    raise SystemExit(_missing_gui_dependency_message(exc)) from exc


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    del argv, prog
    mp.freeze_support()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
