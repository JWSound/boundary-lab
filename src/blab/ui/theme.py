"""Application theme dispatcher.

System preserves Qt's native application appearance. Light and Dark use the
classic Boundary Lab style with an explicit color variant.
"""

from __future__ import annotations

from PySide6.QtGui import QFont, QPalette
from PySide6.QtWidgets import QApplication, QStyleFactory

from blab.ui.classic_theme import classic_font, classic_palette, classic_stylesheet
from blab.ui.settings import normalize_theme
from blab.ui.win32_chrome import (
    ClassicChromeEventFilter,
    apply_classic_window_chrome,
    apply_system_window_chrome,
)

DARK_THEME_CONTENT_BACKGROUND_COLOR = "#293134"

_CLASSIC_CHROME_FILTER: ClassicChromeEventFilter | None = None
_SYSTEM_STYLE_NAME: str | None = None
_SYSTEM_FONT: QFont | None = None
_SYSTEM_PALETTE: QPalette | None = None


def apply_application_theme(theme: object) -> None:
    app = QApplication.instance()
    if app is None:
        return

    theme = normalize_theme(theme)
    _remember_system_appearance(app)
    app.setStyleSheet("")

    if theme == "system":
        _apply_system_appearance(app)
        _set_classic_chrome_enabled(app, enabled=False)
        _refresh_theme_widgets(app)
        return

    style = QStyleFactory.create("Windows") or QStyleFactory.create("Fusion")
    if style is not None:
        app.setStyle(style)
    app.setFont(classic_font())

    app.setPalette(classic_palette(dark=theme == "dark"))
    app.setStyleSheet(classic_stylesheet())
    _set_classic_chrome_enabled(app, enabled=True)

    _refresh_theme_widgets(app)


def themed_content_background(palette: QPalette) -> str:
    """Dark canvas color, or the palette's input background."""
    if palette.color(QPalette.Window).lightness() < 128:
        return DARK_THEME_CONTENT_BACKGROUND_COLOR
    return palette.color(QPalette.Base).name()


def _refresh_theme_widgets(app: QApplication) -> None:
    # QApplication propagates StyleChange and PaletteChange when the style,
    # palette, or application stylesheet changes. Manually unpolishing every
    # widget is redundant and can touch native widgets that are pending
    # deletion, which is unsafe during a global theme switch.
    for widget in app.topLevelWidgets():
        widget.update()
    app.processEvents()


def _remember_system_appearance(app: QApplication) -> None:
    global _SYSTEM_STYLE_NAME, _SYSTEM_FONT, _SYSTEM_PALETTE
    if _SYSTEM_FONT is not None and _SYSTEM_PALETTE is not None:
        return
    style = app.style()
    style_name = style.name() if hasattr(style, "name") else style.objectName()
    _SYSTEM_STYLE_NAME = str(style_name or "") or None
    _SYSTEM_FONT = QFont(app.font())
    _SYSTEM_PALETTE = QPalette(app.palette())


def _apply_system_appearance(app: QApplication) -> None:
    if _SYSTEM_STYLE_NAME:
        style = QStyleFactory.create(_SYSTEM_STYLE_NAME)
        if style is not None:
            app.setStyle(style)
    if _SYSTEM_FONT is not None:
        app.setFont(QFont(_SYSTEM_FONT))
    if _SYSTEM_PALETTE is not None:
        app.setPalette(QPalette(_SYSTEM_PALETTE))


def _set_classic_chrome_enabled(app: QApplication, *, enabled: bool) -> None:
    global _CLASSIC_CHROME_FILTER
    if enabled and _CLASSIC_CHROME_FILTER is None:
        _CLASSIC_CHROME_FILTER = ClassicChromeEventFilter(app)
        app.installEventFilter(_CLASSIC_CHROME_FILTER)
    if _CLASSIC_CHROME_FILTER is not None:
        _CLASSIC_CHROME_FILTER.set_enabled(enabled)
    for widget in app.topLevelWidgets():
        if enabled:
            apply_classic_window_chrome(widget)
        else:
            apply_system_window_chrome(widget)
