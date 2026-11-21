"""Logarithmic frequency range with exact, editable endpoints."""

from math import log10

from PySide6.QtCore import QPoint, QRect, QSignalBlocker, Qt, Signal
from PySide6.QtGui import QPainter, QPalette, QPolygon
from PySide6.QtWidgets import QApplication, QHBoxLayout, QSpinBox, QWidget, qDrawWinButton, qDrawWinPanel
from superqt import QDoubleRangeSlider

from blab.plot_style import AUDIO_FREQ_MAX_HZ, AUDIO_FREQ_MIN_HZ


class FrequencyGripSlider(QDoubleRangeSlider):
    """Two outward-facing grips whose inner edges mark the frequencies."""

    highlighted = Signal(int)  # Endpoint index, or -1 for neither.
    grip_width = 14
    grip_height = 18

    def __init__(self):
        super().__init__(Qt.Horizontal)
        self.setMinimumSize(3 * self.grip_width, self.grip_height)
        self.setMouseTracking(True)
        self.setSingleStep(0)
        self._highlight = -1
        self._press_x = 0
        self._offset = 0

    def _span(self):
        return self.width() - 2 * self.grip_width - 1

    def _anchors(self):
        scale = self._span() / (self.maximum() - self.minimum())
        return tuple(self.grip_width + round((v - self.minimum()) * scale) for v in self.value())

    def _grip(self, index, anchor):
        left = anchor - self.grip_width if index == 0 else anchor + 1
        return QRect(left, (self.height() - self.grip_height) // 2, self.grip_width, self.grip_height)

    def paintEvent(self, event):
        painter = QPainter(self)
        palette = self.palette()
        low, high = self._anchors()
        y = self.height() // 2
        groove = QRect(self.grip_width - 1, y - 3, self._span() + 3, 6)
        qDrawWinPanel(painter, groove, palette, True, palette.mid())
        painter.fillRect(low, y - 1, high - low + 1, 2, palette.highlight())
        for index, anchor in enumerate((low, high)):
            rect = self._grip(index, anchor)
            active = index == self._highlight
            pressed = active and self.isSliderDown()
            face = palette.highlight() if active else palette.button()
            qDrawWinButton(painter, rect, palette, pressed, face)
            direction = -1 if index == 0 else 1
            center = rect.center() + QPoint(int(pressed), int(pressed))
            arrow = QPolygon([center + QPoint(direction * dx, dy) for dx, dy in ((2, 0), (-1, -3), (-1, 3))])
            painter.setPen(Qt.NoPen)
            painter.setBrush(palette.highlightedText() if active else palette.buttonText())
            painter.drawPolygon(arrow)

    def _set_highlight(self, index):
        if index != self._highlight:
            self._highlight = index
            self.highlighted.emit(index)
            self.update()

    def _hit(self, point):
        for index, anchor in enumerate(self._anchors()):
            if self._grip(index, anchor).contains(point):
                return index
        return -1

    def mousePressEvent(self, event):
        if event.button() not in (Qt.LeftButton, Qt.MiddleButton):
            event.ignore()
            return
        point = event.position().toPoint()
        self._press_x = point.x()
        low, high = self._anchors()
        index = self._hit(point)
        self._offset = 0
        self.setSliderDown(True)
        if index >= 0:
            self._offset = self._press_x - (low, high)[index]
        elif high - low <= 1 and low <= self._press_x <= high:
            # The shared seam stays neutral until the gesture has a direction.
            self._set_highlight(-1)
            return
        else:
            index = int(self._press_x > (low + high) / 2)
            self._move_endpoint(index, self._press_x)
        self._set_highlight(index)
        self.update()

    def _move_endpoint(self, index, x):
        fraction = (x - self._offset - self.grip_width) / self._span()
        self.setSliderPosition(self.minimum() + fraction * (self.maximum() - self.minimum()), index=index)

    def mouseMoveEvent(self, event):
        point = event.position().toPoint()
        if not self.isSliderDown():
            self._set_highlight(self._hit(point))
            return
        if self._highlight < 0:
            delta = point.x() - self._press_x
            if abs(delta) < QApplication.startDragDistance():
                return
            self._set_highlight(int(delta > 0))
        self._move_endpoint(self._highlight, point.x())

    def mouseReleaseEvent(self, event):
        if self.isSliderDown() and not event.buttons():
            self.setSliderDown(False)
            self._set_highlight(self._hit(event.position().toPoint()))
            self.update()

    def leaveEvent(self, event):
        if not self.isSliderDown():
            self._set_highlight(-1)
        super().leaveEvent(event)


class FrequencyRangeControl(QWidget):
    edited = Signal()

    def __init__(self, low: int, high: int):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.minimum_spin = QSpinBox()
        self.maximum_spin = QSpinBox()
        self.slider = FrequencyGripSlider()
        self.slider.setAccessibleName("Frequency range")
        for spin, label in ((self.minimum_spin, "Minimum frequency"), (self.maximum_spin, "Maximum frequency")):
            spin.setAccessibleName(label)
            spin.setToolTip(label)
            spin.setSuffix(" Hz")
            spin.setKeyboardTracking(False)
        layout.addWidget(self.minimum_spin)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.maximum_spin)
        self.set_range(low, high)
        self.minimum_spin.valueChanged.connect(self._minimum_edited)
        self.maximum_spin.valueChanged.connect(self._maximum_edited)
        self.slider.valueChanged.connect(self._slider_changed)
        self.slider.sliderReleased.connect(self._commit)
        self.slider.highlighted.connect(self._highlight_endpoint)

    def _highlight_endpoint(self, index: int) -> None:
        for i, spin in enumerate((self.minimum_spin, self.maximum_spin)):
            palette = QPalette()
            if i == index:
                palette.setBrush(QPalette.Base, self.palette().highlight())
                palette.setBrush(QPalette.Text, self.palette().highlightedText())
            spin.setPalette(palette)

    def set_range(self, low: int, high: int) -> None:
        """Restore project values, including legacy single-frequency sweeps."""
        low, high = sorted((max(1, low), max(1, high)))
        floor, ceiling = min(AUDIO_FREQ_MIN_HZ, low), max(AUDIO_FREQ_MAX_HZ, high)
        with QSignalBlocker(self.minimum_spin), QSignalBlocker(self.maximum_spin), QSignalBlocker(self.slider):
            for spin in (self.minimum_spin, self.maximum_spin):
                spin.setRange(floor, ceiling)
            self.slider.setRange(log10(floor), log10(ceiling))
            self._show_values(low, high)
        self._committed = (low, high)

    def _show_values(self, low: int, high: int) -> None:
        with QSignalBlocker(self.minimum_spin), QSignalBlocker(self.maximum_spin), QSignalBlocker(self.slider):
            self.minimum_spin.setValue(low)
            self.maximum_spin.setValue(high)
            self.slider.setValue((log10(self.minimum_spin.value()), log10(self.maximum_spin.value())))

    def _minimum_edited(self, value: int) -> None:
        high = self.maximum_spin.value()
        self._show_values(min(value, high - 1), high)
        self._commit()

    def _maximum_edited(self, value: int) -> None:
        low = self.minimum_spin.value()
        self._show_values(low, max(value, low + 1))
        self._commit()

    def _slider_changed(self, positions: tuple[float, float]) -> None:
        low, high = (round(10**position) for position in positions)
        if low != self.minimum_spin.value() and high == self.maximum_spin.value():
            low = min(low, high - 1)
        else:
            high = max(high, low + 1)
            low = min(low, self.maximum_spin.maximum() - 1)
        self._show_values(low, high)
        # A groove press can change the value before Qt marks the slider down.
        if not self.slider.isSliderDown() and not QApplication.mouseButtons():
            self._commit()

    def _commit(self) -> None:
        values = self.minimum_spin.value(), self.maximum_spin.value()
        if values != self._committed:
            self._committed = values
            self.edited.emit()
