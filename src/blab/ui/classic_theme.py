"""Windows 2000 "Classic" theme palette, font, and stylesheet.

The "Windows" QStyle draws the 3D bevel; a QSS `background`/`border` rule flattens it.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette


@dataclass(frozen=True)
class ClassicColors:
    face: str  # 3DFACE - buttons, menus, window background
    highlight_edge: str  # 3DHILIGHT - bevel top/left
    midlight_edge: str  # derived, between face and highlight_edge
    shadow_edge: str  # 3DSHADOW - bevel bottom/right
    dark_shadow_edge: str  # 3DDKSHADOW - outer bevel
    window: str  # text-entry background (Base)
    text: str


CLASSIC_LIGHT_COLORS = ClassicColors(
    face="#D4D0C8",
    highlight_edge="#FFFFFF",
    midlight_edge="#E0DDD5",
    shadow_edge="#808080",
    dark_shadow_edge="#404040",
    window="#FFFFFF",
    text="#000000",
)

CLASSIC_DARK_COLORS = ClassicColors(
    face="#3C3C3C",
    highlight_edge="#6E6E6E",
    midlight_edge="#525252",
    shadow_edge="#1E1E1E",
    dark_shadow_edge="#0A0A0A",
    window="#262626",
    text="#E8E8E8",
)

CLASSIC_HIGHLIGHT = "#0A246A"  # selection + active caption, shared across variants
CLASSIC_HIGHLIGHT_TEXT = "#FFFFFF"
CLASSIC_DISABLED_TEXT = "#808080"
CLASSIC_TOOLTIP = "#FFFFE1"  # INFOBK

CLASSIC_FONT_FAMILIES = ("Tahoma", "MS Sans Serif", "DejaVu Sans")


def classic_palette(*, dark: bool = False) -> QPalette:
    colors = CLASSIC_DARK_COLORS if dark else CLASSIC_LIGHT_COLORS
    palette = QPalette()

    face = QColor(colors.face)
    highlight_edge = QColor(colors.highlight_edge)
    midlight_edge = QColor(colors.midlight_edge)
    shadow_edge = QColor(colors.shadow_edge)
    dark_shadow_edge = QColor(colors.dark_shadow_edge)
    window_base = QColor(colors.window)
    text = QColor(colors.text)
    highlight = QColor(CLASSIC_HIGHLIGHT)
    highlight_text = QColor(CLASSIC_HIGHLIGHT_TEXT)
    tooltip = QColor(CLASSIC_TOOLTIP)

    for group in (QPalette.Active, QPalette.Inactive):
        palette.setColor(group, QPalette.Window, face)
        palette.setColor(group, QPalette.Button, face)
        palette.setColor(group, QPalette.Base, window_base)
        palette.setColor(group, QPalette.AlternateBase, midlight_edge)
        palette.setColor(group, QPalette.Light, highlight_edge)
        palette.setColor(group, QPalette.Midlight, midlight_edge)
        palette.setColor(group, QPalette.Dark, shadow_edge)
        palette.setColor(group, QPalette.Shadow, dark_shadow_edge)
        palette.setColor(group, QPalette.WindowText, text)
        palette.setColor(group, QPalette.ButtonText, text)
        palette.setColor(group, QPalette.Text, text)
        palette.setColor(group, QPalette.BrightText, QColor("#FF0000"))
        palette.setColor(group, QPalette.Highlight, highlight)
        palette.setColor(group, QPalette.HighlightedText, highlight_text)
        palette.setColor(group, QPalette.ToolTipBase, tooltip)
        palette.setColor(group, QPalette.ToolTipText, QColor(CLASSIC_LIGHT_COLORS.text))
        if hasattr(QPalette, "PlaceholderText"):
            placeholder = QColor(text)
            placeholder.setAlpha(140)
            palette.setColor(group, QPalette.PlaceholderText, placeholder)

    disabled_text = QColor(CLASSIC_DISABLED_TEXT)
    palette.setColor(QPalette.Disabled, QPalette.Window, face)
    palette.setColor(QPalette.Disabled, QPalette.Button, face)
    palette.setColor(QPalette.Disabled, QPalette.Base, face)
    palette.setColor(QPalette.Disabled, QPalette.AlternateBase, face)
    palette.setColor(QPalette.Disabled, QPalette.Light, highlight_edge)
    palette.setColor(QPalette.Disabled, QPalette.Midlight, midlight_edge)
    palette.setColor(QPalette.Disabled, QPalette.Dark, shadow_edge)
    palette.setColor(QPalette.Disabled, QPalette.Shadow, dark_shadow_edge)
    palette.setColor(QPalette.Disabled, QPalette.WindowText, disabled_text)
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, disabled_text)
    palette.setColor(QPalette.Disabled, QPalette.Text, disabled_text)
    palette.setColor(QPalette.Disabled, QPalette.Highlight, shadow_edge)
    palette.setColor(QPalette.Disabled, QPalette.HighlightedText, disabled_text)
    palette.setColor(QPalette.Disabled, QPalette.ToolTipBase, tooltip)
    palette.setColor(QPalette.Disabled, QPalette.ToolTipText, disabled_text)

    return palette


def classic_font() -> QFont:
    available = set(QFontDatabase.families())
    for family in CLASSIC_FONT_FAMILIES:
        if family in available:
            return QFont(family, 8)
    return QFont(CLASSIC_FONT_FAMILIES[0], 8)


def classic_stylesheet() -> str:
    """Near-empty: only widgets with no bevel to lose (tooltips, labels).

    `background`/`border` on a beveled widget hands box painting to QStyleSheetStyle.
    """
    return f"""
        QToolTip {{
            background-color: {CLASSIC_TOOLTIP};
            color: {CLASSIC_LIGHT_COLORS.text};
            border: 1px solid #000000;
        }}
        QWidget#dockTitleBar QLabel {{
            font-weight: bold;
        }}
    """


__all__ = [
    "CLASSIC_DARK_COLORS",
    "CLASSIC_DISABLED_TEXT",
    "CLASSIC_FONT_FAMILIES",
    "CLASSIC_HIGHLIGHT",
    "CLASSIC_HIGHLIGHT_TEXT",
    "CLASSIC_LIGHT_COLORS",
    "CLASSIC_TOOLTIP",
    "ClassicColors",
    "classic_font",
    "classic_palette",
    "classic_stylesheet",
]
