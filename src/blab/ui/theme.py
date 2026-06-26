"""Application theme palette and stylesheet helpers."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from blab.ui.settings import normalize_theme

APP_ROOT = Path(__file__).resolve().parents[3]


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
        window_color = QColor(45, 45, 48)
        base_color = QColor(30, 30, 30)
        palette.setColor(QPalette.Window, QColor(45, 45, 48))
        palette.setColor(QPalette.Base, base_color)
        palette.setColor(QPalette.AlternateBase, QColor(45, 45, 48))
        palette.setColor(QPalette.ToolTipBase, QColor(30, 30, 30))
        palette.setColor(QPalette.Button, QColor(45, 45, 48))
        palette.setColor(QPalette.BrightText, QColor(255, 80, 80))
        palette.setColor(QPalette.Highlight, QColor(61, 126, 154))
        palette.setColor(QPalette.HighlightedText, light_text)
        _set_palette_text_colors(palette, light_text)
        app.setPalette(palette)
        app.setStyleSheet(_theme_stylesheet(light_text, window_color, base_color))
    else:
        palette = app.style().standardPalette()
        window_color = QColor(245, 245, 245)
        base_color = QColor(255, 255, 255)
        palette.setColor(QPalette.Window, window_color)
        palette.setColor(QPalette.Base, Qt.white)
        palette.setColor(QPalette.AlternateBase, QColor(240, 240, 240))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.Button, QColor(245, 245, 245))
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
        palette.setColor(QPalette.HighlightedText, Qt.white)
        _set_palette_text_colors(palette, dark_text)
        app.setPalette(palette)
        app.setStyleSheet(_theme_stylesheet(dark_text, window_color, base_color))

    _refresh_theme_widgets(app)


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


def _theme_stylesheet(text_color: QColor, window_color: QColor, base_color: QColor) -> str:
    text = text_color.name()
    window = window_color.name()
    base = base_color.name()
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
