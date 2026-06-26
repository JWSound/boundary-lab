"""Small support widgets and records for the main window."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QDockWidget, QHBoxLayout, QLabel, QPlainTextEdit, QToolButton, QWidget

from blab.live import FrequencyResult
from blab.ui.drag_drop import local_drop_paths


@dataclass(frozen=True)
class PlotEntry:
    plot_id: str
    title: str
    default_filename: str
    widget: QWidget
    update: Callable[[dict[str, np.ndarray]], None]


def format_frequency_solve_timings(result: FrequencyResult) -> str:
    timings = result.timings
    return f"Assembly {timings.assembly_s:.2f}s | Solve {timings.solve_s:.2f}s | Field {timings.field_s:.2f}s"


class AthScriptEditor(QPlainTextEdit):
    configDropped = Signal(object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event) -> None:
        if self._cfg_drop_path(event) is not None:
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if self._cfg_drop_path(event) is not None:
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        path = self._cfg_drop_path(event)
        if path is None:
            super().dropEvent(event)
            return
        event.acceptProposedAction()
        self.configDropped.emit(path)

    @staticmethod
    def _cfg_drop_path(event) -> Path | None:
        for path in local_drop_paths(event):
            if path.suffix.lower() == ".cfg":
                return path
        return None


class DockTitleBar(QWidget):
    def __init__(
        self,
        title: str,
        dock: QDockWidget,
        *,
        save_action: QAction | None = None,
        tool_actions: tuple[QAction, ...] = (),
    ):
        super().__init__(dock)
        self.dock = dock
        self.tool_buttons: list[QToolButton] = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 2, 2)
        layout.setSpacing(4)
        label = QLabel(title)
        for action in (*(() if save_action is None else (save_action,)), *tool_actions):
            button = QToolButton()
            button.setAutoRaise(True)
            button.setDefaultAction(action)
            button.setToolTip(action.toolTip())
            self.tool_buttons.append(button)
        close_button = QToolButton()
        close_button.setAutoRaise(True)
        close_button.setText("x")
        close_button.setToolTip(f"Close {title}")
        close_button.clicked.connect(dock.close)
        layout.addWidget(label, 1)
        for button in self.tool_buttons:
            layout.addWidget(button)
        layout.addWidget(close_button)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.dock.setFloating(not self.dock.isFloating())
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        event.ignore()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt override
        event.ignore()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt override
        event.ignore()


__all__ = [
    "AthScriptEditor",
    "DockTitleBar",
    "PlotEntry",
    "format_frequency_solve_timings",
]
