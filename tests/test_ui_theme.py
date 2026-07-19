from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor, QPalette

from blab.ui.theme import (
    DARK_THEME_BUTTON_COLOR,
    DARK_THEME_CONTENT_BACKGROUND_COLOR,
    DARK_THEME_TEXT_COLOR,
    DARK_THEME_WINDOW_COLOR,
    LIGHT_THEME_BUTTON_BORDER_COLOR,
    LIGHT_THEME_BUTTON_COLOR,
    LIGHT_THEME_MENU_BAR_COLOR,
    LIGHT_THEME_SLIDER_FILLED_COLOR,
    LIGHT_THEME_SLIDER_UNFILLED_COLOR,
    _theme_stylesheet,
    themed_content_background,
)


def test_dark_theme_uses_requested_color_palette() -> None:
    assert DARK_THEME_WINDOW_COLOR == "#303020"
    assert DARK_THEME_TEXT_COLOR == "#e0e2e4"
    assert DARK_THEME_CONTENT_BACKGROUND_COLOR == "#293134"
    assert DARK_THEME_BUTTON_COLOR == "#333333"

    source = Path("src/blab/ui/theme.py").read_text(encoding="utf-8")
    assert "window_color = QColor(DARK_THEME_WINDOW_COLOR)" in source
    assert "text_color = QColor(DARK_THEME_TEXT_COLOR)" in source
    assert "palette.setColor(QPalette.Button, QColor(DARK_THEME_BUTTON_COLOR))" in source


def test_content_background_uses_dark_canvas_color_only_for_dark_palettes() -> None:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(DARK_THEME_WINDOW_COLOR))
    palette.setColor(QPalette.Base, QColor("#101010"))
    assert themed_content_background(palette) == DARK_THEME_CONTENT_BACKGROUND_COLOR

    palette.setColor(QPalette.Window, QColor("#f5f5f5"))
    palette.setColor(QPalette.Base, QColor("#ffffff"))
    assert themed_content_background(palette) == "#ffffff"


def test_ath_editor_tracks_the_application_theme() -> None:
    widget_source = Path("src/blab/ui/main_window_widgets.py").read_text(encoding="utf-8")
    theme_source = Path("src/blab/ui/theme.py").read_text(encoding="utf-8")

    assert "class AthScriptEditor(QPlainTextEdit):" in widget_source
    assert 'self.setObjectName("athScriptEditor")' in widget_source
    assert "def changeEvent" not in widget_source
    assert "QPlainTextEdit#athScriptEditor" in theme_source
    assert "background-color: {editor_background};" in theme_source

    dark_stylesheet = _theme_stylesheet(QColor("#e0e2e4"), QColor("#303020"), QColor("#1e1e1e"))
    light_stylesheet = _theme_stylesheet(QColor("#1e1e1e"), QColor("#f5f5f5"), QColor("#ffffff"))
    assert "QPlainTextEdit#athScriptEditor {\n            background-color: #293134;" in dark_stylesheet
    assert "QPlainTextEdit#athScriptEditor {\n            background-color: #ffffff;" in light_stylesheet


def test_light_theme_controls_have_contrasting_button_and_slider_styles() -> None:
    assert LIGHT_THEME_BUTTON_COLOR == "#f9f9f9"
    assert LIGHT_THEME_BUTTON_BORDER_COLOR == "#a0a0a0"
    assert LIGHT_THEME_MENU_BAR_COLOR == "#c9ced1"
    assert LIGHT_THEME_SLIDER_FILLED_COLOR == "#879499"
    assert LIGHT_THEME_SLIDER_UNFILLED_COLOR == "#c9ced1"

    stylesheet = _theme_stylesheet(
        QColor("#1e1e1e"),
        QColor("#f5f5f5"),
        QColor("#ffffff"),
        light_controls=True,
    )
    assert "QPushButton {" in stylesheet
    assert "QMenuBar, QMenuBar::item {" in stylesheet
    assert "background-color: #c9ced1;" in stylesheet
    assert "background-color: #f9f9f9;" in stylesheet
    assert "border: 1px solid #a0a0a0;" in stylesheet
    assert "QSlider::sub-page:horizontal" in stylesheet
    assert "background-color: #879499;" in stylesheet
    assert "QSlider::add-page:horizontal" in stylesheet
    assert "background-color: #c9ced1;" in stylesheet

    dark_stylesheet = _theme_stylesheet(QColor("#e0e2e4"), QColor("#303020"), QColor("#1e1e1e"))
    assert "QPushButton {" not in dark_stylesheet
    assert "QMenuBar, QMenuBar::item {" not in dark_stylesheet
    assert "QSlider::add-page:horizontal" not in dark_stylesheet
