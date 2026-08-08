"""DWM title bar tinting for the classic theme.

Qt can't style the OS-drawn title bar. Needs Windows 11 22000+; older builds
stay untinted.
"""

from __future__ import annotations

import sys

from PySide6.QtCore import QEvent, QObject

from blab.ui.classic_theme import CLASSIC_HIGHLIGHT, CLASSIC_HIGHLIGHT_TEXT

DWMWA_WINDOW_CORNER_PREFERENCE = 33
DWMWCP_DEFAULT = 0
DWMWCP_DONOTROUND = 1

DWMWA_BORDER_COLOR = 34
DWMWA_CAPTION_COLOR = 35
DWMWA_TEXT_COLOR = 36
DWMWA_COLOR_DEFAULT = 0xFFFFFFFF


def _color_ref(hex_color: str) -> int:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b << 16) | (g << 8) | r


def _set_dwm_attribute(hwnd: int, attribute: int, value: int) -> bool:
    try:
        import ctypes

        c_value = ctypes.c_uint(value)
        result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
            ctypes.c_void_p(hwnd),
            ctypes.c_uint(attribute),
            ctypes.byref(c_value),
            ctypes.sizeof(c_value),
        )
        return result == 0
    except Exception:
        return False


def apply_classic_window_chrome(widget) -> bool:
    """Tint one top-level widget's native title bar for the classic theme."""
    if sys.platform != "win32":
        return False
    try:
        hwnd = int(widget.winId())
    except Exception:
        return False

    ok = True
    ok &= _set_dwm_attribute(hwnd, DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_DONOTROUND)
    ok &= _set_dwm_attribute(hwnd, DWMWA_CAPTION_COLOR, _color_ref(CLASSIC_HIGHLIGHT))
    ok &= _set_dwm_attribute(hwnd, DWMWA_TEXT_COLOR, _color_ref(CLASSIC_HIGHLIGHT_TEXT))
    ok &= _set_dwm_attribute(hwnd, DWMWA_BORDER_COLOR, _color_ref(CLASSIC_HIGHLIGHT))
    return bool(ok)


def apply_system_window_chrome(widget) -> bool:
    """Restore one top-level widget's native DWM title-bar appearance."""
    if sys.platform != "win32":
        return False
    try:
        hwnd = int(widget.winId())
    except Exception:
        return False

    ok = True
    ok &= _set_dwm_attribute(hwnd, DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_DEFAULT)
    ok &= _set_dwm_attribute(hwnd, DWMWA_CAPTION_COLOR, DWMWA_COLOR_DEFAULT)
    ok &= _set_dwm_attribute(hwnd, DWMWA_TEXT_COLOR, DWMWA_COLOR_DEFAULT)
    ok &= _set_dwm_attribute(hwnd, DWMWA_BORDER_COLOR, DWMWA_COLOR_DEFAULT)
    return bool(ok)


class ClassicChromeEventFilter(QObject):
    """Tints each top-level window's title bar the moment it's shown."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._enabled = True

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt override
        if self._enabled and event.type() == QEvent.Type.Show:
            is_top_level = getattr(watched, "isWindow", None)
            if callable(is_top_level) and is_top_level():
                apply_classic_window_chrome(watched)
        return False


__all__ = [
    "ClassicChromeEventFilter",
    "apply_classic_window_chrome",
    "apply_system_window_chrome",
]
