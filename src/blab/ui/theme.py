"""Application theme palette and stylesheet helpers."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from blab.paths import APP_ROOT
from blab.ui.settings import normalize_theme

DARK_THEME_WINDOW_COLOR = "#303020"
DARK_THEME_TEXT_COLOR = "#e0e2e4"
DARK_THEME_CONTENT_BACKGROUND_COLOR = "#293134"
DARK_THEME_BUTTON_COLOR = "#333333"
LIGHT_THEME_BUTTON_COLOR = "#f9f9f9"
LIGHT_THEME_BUTTON_BORDER_COLOR = "#a0a0a0"
LIGHT_THEME_MENU_BAR_COLOR = "#c9ced1"
LIGHT_THEME_SLIDER_FILLED_COLOR = "#879499"
LIGHT_THEME_SLIDER_UNFILLED_COLOR = "#c9ced1"


def apply_application_theme(theme: object) -> None:
    app = QApplication.instance()
    if app is None:
        return

    theme = normalize_theme(theme)
    app.setStyleSheet("")
    dark_text = QColor(30, 30, 30)
    light_text = QColor(245, 245, 245)
    if theme == "system":
        palette = app.style().standardPalette()
        window_color = palette.color(QPalette.Window)
        base_color = palette.color(QPalette.Base)
        text_color = dark_text if window_color.lightness() >= 128 else light_text
        _set_palette_text_colors(palette, text_color)
        app.setPalette(palette)
        app.setStyleSheet(_theme_stylesheet(text_color, window_color, base_color))
    elif theme == "dark":
        palette = app.style().standardPalette()
        window_color = QColor(DARK_THEME_WINDOW_COLOR)
        text_color = QColor(DARK_THEME_TEXT_COLOR)
        base_color = QColor(30, 30, 30)
        palette.setColor(QPalette.Window, window_color)
        palette.setColor(QPalette.Base, base_color)
        palette.setColor(QPalette.AlternateBase, window_color)
        palette.setColor(QPalette.ToolTipBase, QColor(30, 30, 30))
        palette.setColor(QPalette.Button, QColor(DARK_THEME_BUTTON_COLOR))
        palette.setColor(QPalette.BrightText, QColor(255, 80, 80))
        palette.setColor(QPalette.Highlight, QColor(61, 126, 154))
        palette.setColor(QPalette.HighlightedText, text_color)
        _set_palette_text_colors(palette, text_color)
        app.setPalette(palette)
        app.setStyleSheet(_theme_stylesheet(text_color, window_color, base_color))
    else:
        palette = app.style().standardPalette()
        window_color = QColor(245, 245, 245)
        base_color = QColor(255, 255, 255)
        palette.setColor(QPalette.Window, window_color)
        palette.setColor(QPalette.Base, Qt.white)
        palette.setColor(QPalette.AlternateBase, QColor(240, 240, 240))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.Button, QColor(LIGHT_THEME_BUTTON_COLOR))
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
        palette.setColor(QPalette.HighlightedText, Qt.white)
        _set_palette_text_colors(palette, dark_text)
        app.setPalette(palette)
        app.setStyleSheet(_theme_stylesheet(dark_text, window_color, base_color, light_controls=True))

    _refresh_theme_widgets(app)


def themed_content_background(palette: QPalette) -> str:
    """Return the dark canvas color or the active light/system input background."""
    if palette.color(QPalette.Window).lightness() < 128:
        return DARK_THEME_CONTENT_BACKGROUND_COLOR
    return palette.color(QPalette.Base).name()


def _set_palette_text_colors(palette: QPalette, color: QColor) -> None:
    roles = (
        QPalette.WindowText,
        QPalette.Text,
        QPalette.ButtonText,
        QPalette.ToolTipText,
    )
    if hasattr(QPalette, "PlaceholderText"):
        roles = (*roles, QPalette.PlaceholderText)

    disabled_color = QColor(color)
    disabled_color.setAlpha(140)
    for group, group_color in (
        (QPalette.Active, color),
        (QPalette.Inactive, color),
        (QPalette.Disabled, disabled_color),
    ):
        for role in roles:
            palette.setColor(group, role, group_color)


def _refresh_theme_widgets(app: QApplication) -> None:
    style = app.style()
    for widget in app.allWidgets():
        style.unpolish(widget)
        style.polish(widget)
        widget.update()
    app.processEvents()


def _theme_stylesheet(
    text_color: QColor,
    window_color: QColor,
    base_color: QColor,
    *,
    light_controls: bool = False,
) -> str:
    text = text_color.name()
    window = window_color.name()
    base = base_color.name()
    editor_background = DARK_THEME_CONTENT_BACKGROUND_COLOR if window_color.lightness() < 128 else base_color.name()
    border = QColor(85, 85, 85).name() if text_color.lightness() > 128 else QColor(190, 190, 190).name()
    selected = QColor(61, 126, 154).name() if text_color.lightness() > 128 else QColor(0, 120, 215).name()
    selected_text = QColor(255, 255, 255).name()
    hover = QColor(65, 65, 68).name() if text_color.lightness() > 128 else QColor(225, 225, 225).name()
    disabled = QColor(text_color)
    disabled.setAlpha(150)
    disabled_css = f"rgba({disabled.red()}, {disabled.green()}, {disabled.blue()}, {disabled.alpha()})"
    arrow_variant = "light" if text_color.lightness() > 128 else "dark"
    spin_arrow_up = (APP_ROOT / "assets" / f"spin_arrow_up_{arrow_variant}.svg").as_posix()
    spin_arrow_down = (APP_ROOT / "assets" / f"spin_arrow_down_{arrow_variant}.svg").as_posix()
    light_control_styles = ""
    if light_controls:
        light_control_styles = f"""
        QMenuBar, QMenuBar::item {{
            background-color: {LIGHT_THEME_MENU_BAR_COLOR};
        }}
        QPushButton {{
            background-color: {LIGHT_THEME_BUTTON_COLOR};
            border: 1px solid {LIGHT_THEME_BUTTON_BORDER_COLOR};
            border-radius: 3px;
            padding: 3px 8px;
        }}
        QPushButton:hover {{
            background-color: #ffffff;
            border-color: #808080;
        }}
        QPushButton:pressed {{
            background-color: #e5e5e5;
        }}
        QPushButton:disabled {{
            background-color: #eeeeee;
            border-color: #c6c6c6;
        }}
        QSlider::groove:horizontal {{
            height: 4px;
            background-color: {LIGHT_THEME_SLIDER_UNFILLED_COLOR};
            border: 1px solid #9ba3a6;
            border-radius: 3px;
        }}
        QSlider::sub-page:horizontal {{
            background-color: {LIGHT_THEME_SLIDER_FILLED_COLOR};
            border-radius: 2px;
        }}
        QSlider::add-page:horizontal {{
            background-color: {LIGHT_THEME_SLIDER_UNFILLED_COLOR};
            border-radius: 2px;
        }}
        QSlider::handle:horizontal {{
            width: 12px;
            margin: -5px 0;
            background-color: #555b5e;
            border: 1px solid #404649;
            border-radius: 6px;
        }}
        """

    return f"""
        QWidget {{
            color: {text};
        }}
        QMenuBar, QMenuBar::item, QMenu {{
            background-color: {window};
            color: {text};
        }}
        QMenuBar::item:selected, QMenu::item:selected {{
            background-color: {hover};
            color: {text};
        }}
        QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox,
        QTableWidget, QTableView, QListView, QTreeView {{
            background-color: {base};
            color: {text};
            border: 1px solid {border};
            selection-background-color: {selected};
            selection-color: {selected_text};
        }}
        QPlainTextEdit#athScriptEditor {{
            background-color: {editor_background};
        }}
        {light_control_styles}
        QSpinBox, QDoubleSpinBox {{
            padding-right: 20px;
        }}
        QSpinBox::up-button, QDoubleSpinBox::up-button {{
            subcontrol-origin: border;
            subcontrol-position: top right;
            width: 18px;
            border-left: 1px solid {border};
            border-bottom: 1px solid {border};
        }}
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            subcontrol-origin: border;
            subcontrol-position: bottom right;
            width: 18px;
            border-left: 1px solid {border};
        }}
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
            image: url("{spin_arrow_up}");
            width: 8px;
            height: 8px;
        }}
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
            image: url("{spin_arrow_down}");
            width: 8px;
            height: 8px;
        }}
        QHeaderView::section {{
            background-color: {window};
            color: {text};
            border: 1px solid {border};
        }}
        QWidget:disabled {{
            color: {disabled_css};
        }}
        QToolTip {{
            background-color: {base};
            color: {text};
            border: 1px solid {border};
        }}
    """
