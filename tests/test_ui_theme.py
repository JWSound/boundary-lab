import sys

import pytest
from PySide6.QtCore import QEvent
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QWidget

from blab.ui import win32_chrome as win32_chrome_module
from blab.ui.classic_theme import (
    CLASSIC_DARK_COLORS,
    CLASSIC_DISABLED_TEXT,
    CLASSIC_HIGHLIGHT,
    CLASSIC_LIGHT_COLORS,
    CLASSIC_TOOLTIP,
    classic_font,
    classic_stylesheet,
)
from blab.ui.settings import normalize_theme
from blab.ui.theme import DARK_THEME_CONTENT_BACKGROUND_COLOR, apply_application_theme, themed_content_background
from blab.ui.win32_chrome import (
    ClassicChromeEventFilter,
    apply_classic_window_chrome,
    apply_system_window_chrome,
)


@pytest.fixture(autouse=True)
def _restore_theme_after_test(qapp):
    """qapp is session-scoped; without this a theme here leaks into later tests."""
    yield
    apply_application_theme("system")


def _underlying_style_class_name(style) -> str:
    """Real QStyle class name, unwrapping the QStyleSheetStyle proxy any stylesheet adds."""
    meta = style.metaObject()
    while meta.className() == "QStyleSheetStyle":
        meta = meta.superClass()
    return meta.className()


def test_normalize_theme_falls_back_to_system_for_unknown_values() -> None:
    assert normalize_theme("system") == "system"
    assert normalize_theme("Light") == "light"
    assert normalize_theme("DARK") == "dark"
    assert normalize_theme("not-a-real-theme") == "system"


def test_content_background_uses_dark_canvas_color_only_for_dark_palettes() -> None:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(CLASSIC_DARK_COLORS.face))
    palette.setColor(QPalette.Base, QColor("#101010"))
    assert themed_content_background(palette) == DARK_THEME_CONTENT_BACKGROUND_COLOR

    palette.setColor(QPalette.Window, QColor("#f5f5f5"))
    palette.setColor(QPalette.Base, QColor("#ffffff"))
    assert themed_content_background(palette) == "#ffffff"


@pytest.mark.parametrize(
    ("theme_value", "expected_colors"),
    [("light", CLASSIC_LIGHT_COLORS), ("dark", CLASSIC_DARK_COLORS)],
)
def test_light_and_dark_pin_their_classic_color_variant(qapp, theme_value, expected_colors) -> None:
    apply_application_theme(theme_value)

    assert _underlying_style_class_name(qapp.style()) in {"QWindowsStyle", "QFusionStyle"}
    assert qapp.font().family() == classic_font().family()

    palette = qapp.palette()
    assert palette.color(QPalette.Button).name() == expected_colors.face.lower()
    assert palette.color(QPalette.Base).name() == expected_colors.window.lower()
    assert palette.color(QPalette.Highlight).name() == CLASSIC_HIGHLIGHT.lower()

    disabled_button_text = palette.color(QPalette.Disabled, QPalette.ButtonText)
    assert disabled_button_text.name() == CLASSIC_DISABLED_TEXT.lower()
    assert disabled_button_text.alpha() == 255


def test_system_theme_restores_native_application_appearance(qapp) -> None:
    apply_application_theme("system")
    native_style = _underlying_style_class_name(qapp.style())
    native_font = QFont(qapp.font())
    native_palette = QPalette(qapp.palette())

    apply_application_theme("dark")
    assert qapp.palette().color(QPalette.Button).name() == CLASSIC_DARK_COLORS.face.lower()

    apply_application_theme("system")
    assert _underlying_style_class_name(qapp.style()) == native_style
    assert qapp.font() == native_font
    assert qapp.palette() == native_palette


def test_light_and_dark_force_the_classic_style_and_font(qapp) -> None:
    for theme_value in ("light", "dark"):
        apply_application_theme(theme_value)
        assert _underlying_style_class_name(qapp.style()) in {"QWindowsStyle", "QFusionStyle"}
        assert qapp.font().pointSize() == classic_font().pointSize()


def test_classic_stylesheet_never_touches_beveled_widget_chrome() -> None:
    stylesheet = classic_stylesheet()
    beveled_widgets = (
        "QPushButton",
        "QComboBox",
        "QLineEdit",
        "QSpinBox",
        "QDoubleSpinBox",
        "QSlider",
        "QHeaderView",
        "QTabBar",
        "dockTitleBar",
    )
    for widget_name in beveled_widgets:
        for line in stylesheet.splitlines():
            if widget_name not in line:
                continue
            assert "background" not in line, f"{widget_name} rule sets background, flattening its bevel"
            assert "border" not in line, f"{widget_name} rule sets border, flattening its bevel"

    assert CLASSIC_TOOLTIP in stylesheet
    assert "qlineargradient" not in stylesheet


def test_dock_title_bar_label_has_no_forced_color() -> None:
    """The label rule is bold text only, so it inherits the normal palette text color."""
    stylesheet = classic_stylesheet()
    label_block = stylesheet.split("QWidget#dockTitleBar QLabel {", 1)[1].split("}", 1)[0]
    assert "color:" not in label_block
    assert "font-weight: bold;" in label_block


def test_win32_chrome_is_a_noop_off_windows(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert apply_classic_window_chrome(object()) is False
    assert apply_system_window_chrome(object()) is False


def test_classic_chrome_filter_can_be_disabled(monkeypatch, qapp) -> None:
    calls = []
    monkeypatch.setattr(win32_chrome_module, "apply_classic_window_chrome", lambda widget: calls.append(widget))
    event_filter = ClassicChromeEventFilter(qapp)
    window = QWidget()
    show_event = QEvent(QEvent.Type.Show)

    event_filter.set_enabled(False)
    event_filter.eventFilter(window, show_event)
    assert calls == []

    event_filter.set_enabled(True)
    event_filter.eventFilter(window, show_event)
    assert calls == [window]
