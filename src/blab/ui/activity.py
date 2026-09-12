"""Shared, GUI-thread activity lifetimes and an animated status indicator.

Use ``with activities.start(message): ...`` for scoped work, or retain the
returned handle until an asynchronous operation finishes. Worker signals must
be delivered to GUI-thread slots before updating handles. This controller does
not run work or pump Qt events; synchronous work still blocks animation.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QRectF, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QPainter, QPalette, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QStackedWidget, QWidget


class ActivityHandle:
    """An independently finishable activity; finishing twice is harmless."""

    def __init__(self, controller: ActivityController, token: int):
        self._controller = controller
        self._token = token
        self._finished = False

    def update(self, message: str) -> None:
        if not self._finished:
            self._controller._update(self._token, message)

    def finish(self) -> None:
        if not self._finished:
            self._finished = True
            self._controller._finish(self._token)

    def __enter__(self) -> ActivityHandle:
        return self

    def __exit__(self, *_exc) -> None:
        self.finish()


class ActivityController(QObject):
    """Track overlapping activities; display the most recently started one.

    The delay applies to a continuous busy interval, not individual updates.
    ``changed`` reports activity immediately; ``indicator_changed`` suppresses
    the spinner for short operations. Completion status belongs to the caller.
    """

    changed = Signal(bool, str)
    indicator_changed = Signal(bool)

    def __init__(self, parent: QObject | None = None, *, delay_ms: int = 200):
        super().__init__(parent)
        self._messages: dict[int, str] = {}
        self._next_token = 0
        self._indicator_visible = False
        self._delay = QTimer(self)
        self._delay.setSingleShot(True)
        self._delay.setInterval(max(0, delay_ms))
        self._delay.timeout.connect(self._show_indicator)

    @property
    def active(self) -> bool:
        return bool(self._messages)

    @property
    def message(self) -> str:
        return next(reversed(self._messages.values()), "")

    @property
    def indicator_visible(self) -> bool:
        return self._indicator_visible

    def start(self, message: str) -> ActivityHandle:
        was_active = self.active
        self._next_token += 1
        self._messages[self._next_token] = message
        if not was_active:
            self._delay.start()
        handle = ActivityHandle(self, self._next_token)
        self.changed.emit(True, self.message)
        return handle

    def _update(self, token: int, message: str) -> None:
        if token in self._messages:
            self._messages[token] = message
            self.changed.emit(True, self.message)

    def _finish(self, token: int) -> None:
        if token not in self._messages:
            return
        del self._messages[token]
        if not self.active:
            self._delay.stop()
            self._set_indicator_visible(False)
        self.changed.emit(self.active, self.message)

    @Slot()
    def clear(self) -> None:
        """Reset presentation on shutdown; this does not cancel any work."""
        self._messages.clear()
        self._delay.stop()
        self._set_indicator_visible(False)
        self.changed.emit(False, "")

    @Slot()
    def _show_indicator(self) -> None:
        self._set_indicator_visible(self.active)

    def _set_indicator_visible(self, visible: bool) -> None:
        if visible != self._indicator_visible:
            self._indicator_visible = visible
            self.indicator_changed.emit(visible)


class BusyIndicator(QWidget):
    """Palette-aware, resolution-independent spinner with reserved idle space."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedSize(20, 20)
        policy = self.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.setSizePolicy(policy)
        self.setAccessibleName("Activity in progress")
        self._angle = 0
        self._running = False
        self._animation = QTimer(self)
        self._animation.setInterval(40)
        self._animation.timeout.connect(self._advance)
        self.hide()

    @Slot(bool)
    def set_running(self, running: bool) -> None:
        self._running = running
        self.setVisible(running)
        if running and self.isVisible():
            self._animation.start()
        else:
            self._animation.stop()
            self._angle = 0
        self.update()

    @Slot()
    def _advance(self) -> None:
        self._angle = (self._angle + 18) % 360
        self.update()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        if self._running:
            self._animation.start()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._animation.stop()
        super().hideEvent(event)

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        if not self._running:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(self.palette().color(QPalette.Highlight), 2.0)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawArc(QRectF(3, 3, 14, 14), -self._angle * 16, 270 * 16)


class ActivityStatusBar(QFrame):
    """Show activity text without overwriting the application's last status."""

    def __init__(self, activities: ActivityController, status_label: QLabel, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Sunken)
        self.indicator = BusyIndicator(self)
        self.activity_label = QLabel()
        self.activity_label.setAccessibleName("Current activity")
        self._messages = QStackedWidget()
        self._messages.addWidget(status_label)
        self._messages.addWidget(self.activity_label)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)
        layout.addWidget(self.indicator)
        layout.addWidget(self._messages, 1)
        activities.changed.connect(self._activity_changed)
        activities.indicator_changed.connect(self.indicator.set_running)
        self._activity_changed(activities.active, activities.message)
        self.indicator.set_running(activities.indicator_visible)

    @Slot(bool, str)
    def _activity_changed(self, active: bool, message: str) -> None:
        self.activity_label.setText(message)
        self.indicator.setToolTip(message)
        self.indicator.setAccessibleDescription(message)
        self._messages.setCurrentIndex(1 if active else 0)
