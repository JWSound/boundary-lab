from __future__ import annotations

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QPushButton

from blab.ui.error_dialog import MAX_SCREEN_HEIGHT_FRACTION, ErrorDialog


def test_long_error_stays_within_the_screen(qapp) -> None:
    details = "\n".join(f"line {index}" for index in range(2000))
    dialog = ErrorDialog("Solve failed", "boom", details)
    dialog.show()

    available = QGuiApplication.primaryScreen().availableGeometry().height()
    assert dialog.height() <= int(available * MAX_SCREEN_HEIGHT_FRACTION)
    assert dialog.details_view.toPlainText() == f"boom\n\n{details}"
    dialog.close()


def test_copy_and_close_copies_title_message_and_details(qapp) -> None:
    dialog = ErrorDialog("Solve failed", "boom", "Traceback ...")
    copy_button = next(button for button in dialog.findChildren(QPushButton) if button.text() == "Copy && Close")
    copy_button.click()

    assert QGuiApplication.clipboard().text() == "Solve failed\n\nboom\n\nTraceback ..."
    assert dialog.result() == ErrorDialog.Accepted
