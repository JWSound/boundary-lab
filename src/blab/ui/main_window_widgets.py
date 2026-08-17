"""Small support widgets and records for the main window."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import (
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QStyle,
    QStyleOptionToolButton,
    QStylePainter,
    QToolButton,
    QWidget,
)

from blab.live import FrequencyResult
from blab.ui.result_projection import VisualizationProjection


@dataclass(frozen=True)
class PlotEntry:
    plot_id: str
    title: str
    default_filename: str
    widget: QWidget
    update: Callable[[VisualizationProjection], None]


def format_frequency_solve_timings(result: FrequencyResult) -> str:
    timings = result.timings
    return f"Assembly {timings.assembly_s:.2f}s | Solve {timings.solve_s:.2f}s | Field {timings.field_s:.2f}s"


def _dock_close_button_glyph() -> tuple[str, QFont | None]:
    """The classic Win2000 close mark, or a plain "x" when Marlett is absent."""
    if "Marlett" in QFontDatabase.families():
        return "r", QFont("Marlett")
    return "x", None


# Fixed so the button count can't change the bar height.
DOCK_TITLE_BAR_HEIGHT = 26

TAB_CLOSE_GLYPH_WEIGHT = 0.55
TAB_CLOSE_BUTTON_FALLBACK_PX = 16


def dimmed_button_text_color(palette: QPalette, weight: float = TAB_CLOSE_GLYPH_WEIGHT) -> QColor:
    """Blend button text toward the button face."""
    text = palette.color(QPalette.ButtonText)
    face = palette.color(QPalette.Button)
    return QColor(
        round(face.red() + (text.red() - face.red()) * weight),
        round(face.green() + (text.green() - face.green()) * weight),
        round(face.blue() + (text.blue() - face.blue()) * weight),
    )


class TabCloseButton(QToolButton):
    """Tab close control, using the classic dock glyph.

    Replaces Qt's ``SP_TabCloseButton``, a fixed red-orange bitmap that
    ignores the palette.
    """

    def __init__(self, tooltip: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("tabCloseButton")
        self.setAutoRaise(True)
        self.setToolTip(tooltip)
        self.setFocusPolicy(Qt.NoFocus)
        glyph, glyph_font = _dock_close_button_glyph()
        self.setText(glyph)
        if glyph_font is not None:
            self.setFont(glyph_font)
        # Qt's own size, so the swap keeps the tab height.
        style = self.style()
        width = style.pixelMetric(QStyle.PM_TabCloseIndicatorWidth, None, self) or TAB_CLOSE_BUTTON_FALLBACK_PX
        height = style.pixelMetric(QStyle.PM_TabCloseIndicatorHeight, None, self) or TAB_CLOSE_BUTTON_FALLBACK_PX
        self.setFixedSize(width, height)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QStylePainter(self)
        option = QStyleOptionToolButton()
        self.initStyleOption(option)
        # Recoloured per paint, so theme switches need no rewiring.
        option.palette.setColor(QPalette.ButtonText, dimmed_button_text_color(self.palette()))
        painter.drawComplexControl(QStyle.CC_ToolButton, option)


class DockTitleBar(QFrame):
    def __init__(
        self,
        title: str,
        dock: QDockWidget,
        *,
        save_action: QAction | None = None,
        tool_actions: tuple[QAction, ...] = (),
    ):
        super().__init__(dock)
        self.setObjectName("dockTitleBar")
        # QFrame paints the bevel; a QSS rule only ever showed under grab().
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setFixedHeight(DOCK_TITLE_BAR_HEIGHT)
        self.dock = dock
        self.tool_buttons: list[QToolButton] = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 2, 2)
        layout.setSpacing(4)
        label = QLabel(title)
        label.setObjectName("dockTitleBarLabel")
        for action in (*(() if save_action is None else (save_action,)), *tool_actions):
            button = QToolButton()
            button.setAutoRaise(True)
            if action.menu() is not None:
                button.setText(action.text())
                button.setMenu(action.menu())
                button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
                _sync_menu_tool_button(button, action)
                action.changed.connect(
                    lambda button=button, action=action: _sync_menu_tool_button(button, action)
                )
            else:
                button.setDefaultAction(action)
                if not action.icon().isNull():
                    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            button.setToolTip(action.toolTip())
            self.tool_buttons.append(button)
        close_button = QToolButton()
        close_button.setObjectName("dockTitleBarCloseButton")
        close_button.setAutoRaise(True)
        glyph, glyph_font = _dock_close_button_glyph()
        close_button.setText(glyph)
        if glyph_font is not None:
            close_button.setFont(glyph_font)
        close_button.setToolTip(f"Close {title}")
        close_button.clicked.connect(dock.close)
        layout.addWidget(label, 1)
        for button in self.tool_buttons:
            layout.addWidget(button)
        layout.addWidget(close_button)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        # QDockWidget lays out using sizeHint(), not setFixedHeight(); override both.
        hint = super().sizeHint()
        return QSize(hint.width(), DOCK_TITLE_BAR_HEIGHT)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.dock.setFloating(not self.dock.isFloating())
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        event.ignore()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt override
        event.ignore()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt override
        event.ignore()


def _sync_menu_tool_button(button: QToolButton, action: QAction) -> None:
    button.setEnabled(action.isEnabled())
    button.setVisible(action.isVisible())
    button.setText(action.text())
    button.setToolTip(action.toolTip())


__all__ = [
    "TAB_CLOSE_GLYPH_WEIGHT",
    "DockTitleBar",
    "PlotEntry",
    "TabCloseButton",
    "dimmed_button_text_color",
    "format_frequency_solve_timings",
]
