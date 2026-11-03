"""Shared frequency Marker drawn on every plot canvas.

``PlotMarker`` holds the one marker state (on/off, locked, frequency) that all
plots share, plus the checkable toolbar action bound to it. ``PlotMarkerOverlay``
connects one canvas to that state: it turns hover and click events into marker
updates and draws the vertical line and frequency label with blitting.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.lines import Line2D
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QAction

from blab.ui.plots import (
    CROSSHAIR_COLOR,
    CROSSHAIR_LABEL_EDGE,
    CROSSHAIR_LABEL_FACE,
    PLOT_TICK_SIZE,
    InteractivePlotCanvas,
    format_plot_frequency,
)

MARKER_LINE_GID = "plot_marker_line"
MARKER_LABEL_GID = "plot_marker_label"
# Below the x-axis, where the click/drag crosshair puts its frequency label.
MARKER_LABEL_AXES_Y = -0.075


class PlotMarker(QObject):
    """Marker state shared by all plots; ``changed`` fires on every update."""

    changed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._enabled = False
        self._locked = False
        self._freq_hz: float | None = None
        self.action = QAction("Marker", self)
        self.action.setToolTip("Show a frequency marker on all plots; click a plot to lock or unlock it")
        self.action.setCheckable(True)
        self.action.toggled.connect(self.set_enabled)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def locked(self) -> bool:
        return self._locked

    @property
    def frequency_hz(self) -> float | None:
        return self._freq_hz

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._enabled:
            return
        self._enabled = enabled
        self._locked = False
        self._freq_hz = None
        self.action.setChecked(enabled)
        self.changed.emit()

    def hover(self, freq_hz: float) -> None:
        """Follow the mouse unless the marker is off or locked."""
        if not self._enabled or self._locked:
            return
        self._freq_hz = float(freq_hz)
        self.changed.emit()

    def click(self, freq_hz: float) -> None:
        """Lock the marker at ``freq_hz``, or unlock it if it is locked."""
        if not self._enabled:
            return
        self._locked = not self._locked
        self._freq_hz = float(freq_hz)
        self.changed.emit()


class PlotMarkerOverlay:
    """Draws the shared marker on one canvas and feeds its mouse input back.

    The canvas maps frequencies through its marker contract
    (``marker_frequency_at``, ``marker_frequency_range``, ``frequency_to_axis_x``)
    and extends the drawing through ``update_marker_readout``, which returns
    extra artists to blit along with the line and frequency label.
    """

    def __init__(self, canvas: InteractivePlotCanvas, marker: PlotMarker) -> None:
        self._canvas = canvas
        self._marker = marker
        self._line: Line2D | None = None
        self._label = None
        self._background = None
        self._was_enabled = marker.enabled
        canvas.mpl_connect("motion_notify_event", self._on_motion)
        canvas.mpl_connect("button_press_event", self._on_press)
        canvas.mpl_connect("draw_event", self._on_draw)
        marker.changed.connect(self._on_marker_changed)
        canvas.set_marker_overlay(self)

    @property
    def enabled(self) -> bool:
        return self._marker.enabled

    def _on_motion(self, event) -> None:
        if not self._marker.enabled:
            return
        freq_hz = self._canvas.marker_frequency_at(event)
        if freq_hz is not None:
            self._marker.hover(freq_hz)

    def _on_press(self, event) -> None:
        if not self._marker.enabled or event.button != MouseButton.LEFT:
            return
        if self._canvas._event_hits_series_legend(event):
            return
        freq_hz = self._canvas.marker_frequency_at(event)
        if freq_hz is not None:
            self._marker.click(freq_hz)

    def _on_marker_changed(self) -> None:
        if self._marker.enabled != self._was_enabled:
            # The crosshair hides while the marker is on; a full redraw shows or hides both.
            self._was_enabled = self._marker.enabled
            self._canvas._redraw_crosshair()
            self._refresh_artists()
            self._canvas.draw_idle()
            return
        artists = self._refresh_artists()
        if not self._blit(artists):
            self._canvas.draw_idle()

    def _on_draw(self, event) -> None:
        if event.canvas is not self._canvas:
            return
        try:
            self._background = self._canvas.copy_from_bbox(self._canvas.figure.bbox)
        except Exception:
            self._background = None
        artists = self._refresh_artists()
        if artists:
            self._blit(artists)

    def _blit(self, artists: tuple) -> bool:
        canvas = self._canvas
        if self._background is None or not canvas._is_render_visible():
            return False
        try:
            canvas.restore_region(self._background)
            for artist in artists:
                artist.axes.draw_artist(artist)
            canvas.blit(canvas.figure.bbox)
            return True
        except Exception:
            self._background = None
            return False

    def _marker_frequency(self) -> float | None:
        freq_hz = self._marker.frequency_hz
        if not self._marker.enabled or freq_hz is None:
            return None
        span = self._canvas.marker_frequency_range()
        if span is None:
            return None
        return float(np.clip(freq_hz, *span))

    def _refresh_artists(self) -> tuple:
        """Move or hide the marker artists; return those to draw."""
        freq_hz = self._marker_frequency()
        if freq_hz is None:
            for artist in (self._line, self._label):
                if artist is not None:
                    artist.set_visible(False)
            self._canvas.update_marker_readout(None)
            return ()
        self._ensure_artists()
        axis_x = self._canvas.frequency_to_axis_x(freq_hz)
        self._line.set_xdata([axis_x, axis_x])
        self._line.set_linestyle("-" if self._marker.locked else "--")
        self._label.set_position((axis_x, MARKER_LABEL_AXES_Y))
        self._label.set_text(format_plot_frequency(freq_hz))
        self._line.set_visible(True)
        readout = tuple(self._canvas.update_marker_readout(freq_hz))
        self._label.set_visible(not readout)
        if readout:
            return (self._line, *readout)
        return (self._line, self._label)

    def _ensure_artists(self) -> None:
        axes = self._canvas.axes
        children = axes.get_children()
        # Clearing the axes for new data drops our artists; recreate them on demand.
        if self._line is None or self._line not in children:
            self._line = Line2D(
                [0.0, 0.0],
                [0.0, 1.0],
                transform=axes.get_xaxis_transform(),
                color=CROSSHAIR_COLOR,
                linewidth=0.9,
                alpha=0.9,
                zorder=9,
                gid=MARKER_LINE_GID,
                animated=True,
            )
            # add_artist, unlike axvline, leaves the data limits and autoscaling alone.
            axes.add_artist(self._line)
        if self._label is None or self._label not in children:
            self._label = axes.text(
                0.0,
                MARKER_LABEL_AXES_Y,
                "",
                transform=axes.get_xaxis_transform(),
                ha="center",
                va="top",
                color=CROSSHAIR_LABEL_EDGE,
                fontsize=PLOT_TICK_SIZE,
                bbox={
                    "boxstyle": "square,pad=0.12",
                    "facecolor": CROSSHAIR_LABEL_FACE,
                    "edgecolor": CROSSHAIR_LABEL_EDGE,
                    "linewidth": 0.8,
                },
                clip_on=False,
                zorder=10,
                gid=MARKER_LABEL_GID,
                animated=True,
            )
