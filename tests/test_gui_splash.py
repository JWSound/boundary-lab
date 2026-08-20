from __future__ import annotations

import logging

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QPainter, QPixmap

from blab.gui import SPLASH_TEXT_COLOR, BoundaryLabSplashScreen


def test_splash_status_is_drawn_inside_transparent_padded_artwork(qapp) -> None:
    pixmap = QPixmap(240, 160)
    pixmap.fill(Qt.transparent)
    artwork_rect = QRect(40, 20, 160, 110)
    painter = QPainter(pixmap)
    painter.fillRect(artwork_rect, QColor(100, 100, 100))
    painter.end()

    splash = BoundaryLabSplashScreen(pixmap)
    try:
        splash.show()
        splash.showMessage("Loading solver modules...", Qt.AlignBottom | Qt.AlignLeft, SPLASH_TEXT_COLOR)
        qapp.processEvents()

        panel_rect = splash._last_message_panel_rect
        assert not panel_rect.isEmpty()
        assert panel_rect.left() >= artwork_rect.left()
        assert panel_rect.right() <= artwork_rect.right()
        assert panel_rect.bottom() <= artwork_rect.bottom()

        rendered = splash.grab().toImage()
        panel_background = rendered.pixelColor(panel_rect.right() - 7, panel_rect.top() + 2)
        assert panel_background.red() > 200
        assert panel_background.green() > 200
        assert panel_background.blue() > 200
    finally:
        splash.close()
        splash.deleteLater()
        qapp.processEvents()


def test_runtime_log_remains_open_after_startup_and_records_events(qapp, tmp_path, monkeypatch) -> None:
    import blab.gui as gui_module

    log_path = tmp_path / "session.log"
    monkeypatch.setattr(gui_module, "_startup_log_path", lambda: log_path)
    reporter = gui_module.StartupReporter(qapp, None)
    try:
        reporter.finish_startup()
        assert not reporter._log_file.closed
        logging.getLogger("blab.runtime").info("Exterior field regression breadcrumb")
        reporter._runtime_handler.flush()
        contents = log_path.read_text(encoding="utf-8")
        assert "Startup completed" in contents
        assert "Exterior field regression breadcrumb" in contents
        assert "session ended" not in contents
    finally:
        reporter.close()

    assert reporter._log_file.closed
    assert "Boundary Lab GUI session ended" in log_path.read_text(encoding="utf-8")
