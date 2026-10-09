"""Error dialog whose details scroll instead of outgrowing the screen."""

from __future__ import annotations

from PySide6.QtGui import QFontDatabase, QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

MAX_SCREEN_HEIGHT_FRACTION = 0.6


class ErrorDialog(QDialog):
    def __init__(self, title: str, message: str, details: str | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setSizeGripEnabled(True)
        body = "\n\n".join(part for part in (message, details) if part)
        self.copy_text = f"{title}\n\n{body}"

        summary = message.splitlines()[0] if message else title
        label = QLabel(summary)
        label.setWordWrap(True)

        self.details_view = QPlainTextEdit(body)
        self.details_view.setReadOnly(True)
        self.details_view.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))

        buttons = QDialogButtonBox()
        copy_button = QPushButton("Copy && Close")
        close_button = QPushButton("Close")
        buttons.addButton(copy_button, QDialogButtonBox.AcceptRole)
        buttons.addButton(close_button, QDialogButtonBox.RejectRole)
        close_button.setDefault(True)
        copy_button.clicked.connect(self._copy_and_close)
        close_button.clicked.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(label)
        layout.addWidget(self.details_view, 1)
        layout.addWidget(buttons)

        screen = (parent.screen() if parent is not None else None) or QGuiApplication.primaryScreen()
        available = screen.availableGeometry()
        self.setMaximumHeight(int(available.height() * MAX_SCREEN_HEIGHT_FRACTION))
        self.resize(min(720, available.width()), self.maximumHeight())

    def _copy_and_close(self) -> None:
        QGuiApplication.clipboard().setText(self.copy_text)
        self.accept()
