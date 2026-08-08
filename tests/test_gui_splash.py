from __future__ import annotations

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
