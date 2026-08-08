"""Ath source editor with line numbers and syntax colouring."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPainter, QPalette
from PySide6.QtWidgets import QPlainTextEdit, QWidget

from blab.ui.ath_syntax import AthSyntaxHighlighter
from blab.ui.drag_drop import local_drop_paths

EDITOR_FONT_FAMILIES = ("Consolas", "DejaVu Sans Mono", "Courier New")
EDITOR_FONT_POINT_SIZE = 10
EDITOR_TAB_WIDTH_CHARS = 4

GUTTER_MARGIN_PX = 6
# Steady gutter width under 10 lines.
GUTTER_MIN_DIGITS = 2
# Dims against either palette.
GUTTER_TEXT_ALPHA = 170


def editor_font() -> QFont:
    """Fixed-pitch font, falling back when Consolas is missing."""
    available = set(QFontDatabase.families())
    family = next((name for name in EDITOR_FONT_FAMILIES if name in available), EDITOR_FONT_FAMILIES[0])
    font = QFont(family, EDITOR_FONT_POINT_SIZE)
    font.setStyleHint(QFont.Monospace)
    font.setFixedPitch(True)
    return font


def gutter_text_color(palette: QPalette) -> QColor:
    color = QColor(palette.color(QPalette.WindowText))
    color.setAlpha(GUTTER_TEXT_ALPHA)
    return color


class LineNumberGutter(QWidget):
    """Line-number margin. The editor does the painting."""

    def __init__(self, editor: AthScriptEditor):
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return QSize(self.editor.line_number_width(), 0)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.editor.paint_line_numbers(event)


class AthScriptEditor(QPlainTextEdit):
    configDropped = Signal(object)

    def __init__(self, parent: QWidget | None = None, *, highlight_syntax: bool = True):
        super().__init__(parent)
        self.setObjectName("athScriptEditor")
        self.setAcceptDrops(True)
        # Both must exist before setFont() fires changeEvent().
        self.line_number_gutter = LineNumberGutter(self)
        self.syntax_highlighter = AthSyntaxHighlighter(
            self.document(),
            self.palette(),
            enabled=highlight_syntax,
        )
        self.setFont(editor_font())
        self.setTabStopDistance(EDITOR_TAB_WIDTH_CHARS * self.fontMetrics().horizontalAdvance(" "))
        self.blockCountChanged.connect(self._reserve_gutter_width)
        self.updateRequest.connect(self._repaint_gutter)
        self._reserve_gutter_width()

    @property
    def syntax_highlighting_enabled(self) -> bool:
        return self.syntax_highlighter.enabled

    def set_syntax_highlighting_enabled(self, enabled: bool) -> None:
        self.syntax_highlighter.set_enabled(enabled)

    def line_number_width(self) -> int:
        digits = max(GUTTER_MIN_DIGITS, len(str(max(1, self.blockCount()))))
        return 2 * GUTTER_MARGIN_PX + digits * self.fontMetrics().horizontalAdvance("9")

    def paint_line_numbers(self, event) -> None:
        gutter = self.line_number_gutter
        palette = self.palette()
        painter = QPainter(gutter)
        painter.fillRect(event.rect(), palette.color(QPalette.Window))
        painter.setPen(palette.color(QPalette.Dark))
        painter.drawLine(gutter.width() - 1, event.rect().top(), gutter.width() - 1, event.rect().bottom())
        painter.setFont(self.font())
        painter.setPen(gutter_text_color(palette))

        block = self.firstVisibleBlock()
        number = block.blockNumber()
        top = self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
        line_height = self.fontMetrics().height()
        while block.isValid() and top <= event.rect().bottom():
            bottom = top + self.blockBoundingRect(block).height()
            if block.isVisible() and bottom >= event.rect().top():
                # First visual line only: a wrapped block keeps one number.
                painter.drawText(
                    0,
                    int(top),
                    gutter.width() - GUTTER_MARGIN_PX,
                    line_height,
                    Qt.AlignRight | Qt.AlignVCenter,
                    str(number + 1),
                )
            block = block.next()
            top = bottom
            number += 1

    def _reserve_gutter_width(self) -> None:
        self.setViewportMargins(self.line_number_width(), 0, 0, 0)

    def _repaint_gutter(self, rect, dy: int) -> None:
        if dy:
            self.line_number_gutter.scroll(0, dy)
        else:
            self.line_number_gutter.update(0, rect.y(), self.line_number_gutter.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._reserve_gutter_width()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        contents = self.contentsRect()
        self.line_number_gutter.setGeometry(
            QRect(contents.left(), contents.top(), self.line_number_width(), contents.height())
        )

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:
            self.syntax_highlighter.refresh_colors(self.palette())
        if event.type() in {QEvent.Type.PaletteChange, QEvent.Type.FontChange}:
            self._reserve_gutter_width()
            self.line_number_gutter.update()

    def dragEnterEvent(self, event) -> None:
        if self._cfg_drop_path(event) is not None:
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if self._cfg_drop_path(event) is not None:
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        path = self._cfg_drop_path(event)
        if path is None:
            super().dropEvent(event)
            return
        event.acceptProposedAction()
        self.configDropped.emit(path)

    @staticmethod
    def _cfg_drop_path(event) -> Path | None:
        for path in local_drop_paths(event):
            if path.suffix.lower() == ".cfg":
                return path
        return None


__all__ = [
    "EDITOR_FONT_FAMILIES",
    "GUTTER_MARGIN_PX",
    "GUTTER_MIN_DIGITS",
    "AthScriptEditor",
    "LineNumberGutter",
    "editor_font",
    "gutter_text_color",
]
