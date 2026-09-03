"""Source-referenced on-axis group-delay plot canvas."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator, LogLocator, MaxNLocator
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu

from blab.ui.plots import (
    GRID_LINE_ALPHA,
    MINOR_GRID_LINE_ALPHA,
    PLOT_TITLE_PAD,
    SINGLE_AXIS_LAYOUT,
    RawCoordinatePlotCanvas,
    apply_audio_frequency_axis,
    apply_compact_plot_text,
    clear_plot_axes,
)


class GroupDelayCanvas(RawCoordinatePlotCanvas):
    """Plot source-referenced group delay for the sum and each channel."""

    def __init__(self) -> None:
        self.figure = Figure(figsize=(6.5, 3.0), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self._lines: dict[str, Any] = {}
        self._series_labels: tuple[str, ...] = ()
        self._series_visibility: dict[str, bool] = {}
        self._series_actions: dict[str, QAction] = {}
        self._channel_colors: dict[str, Any] = {}
        self._plot_state = None
        super().__init__(self.figure, "Group Delay")
        self.trace_filter_menu = QMenu("Traces", self)
        self.trace_filter_action = QAction("Traces", self)
        self.trace_filter_action.setToolTip("Choose visible group-delay traces")
        self.trace_filter_action.setMenu(self.trace_filter_menu)
        self.set_layout_profile(SINGLE_AXIS_LAYOUT)
        self._draw_empty()

    def _configure_axes(self) -> None:
        self.axes.set_title(self.title, pad=PLOT_TITLE_PAD)
        self.axes.set_xlabel("Frequency (Hz)")
        self.axes.set_ylabel("Group Delay (ms)")
        apply_audio_frequency_axis(self.axes)
        self.axes.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2.0, 10.0)))
        self.axes.yaxis.set_major_locator(MaxNLocator(nbins=8, min_n_ticks=4))
        self.axes.yaxis.set_minor_locator(AutoMinorLocator(2))
        self.axes.grid(which="major", color="#808080", linewidth=0.8, alpha=GRID_LINE_ALPHA)
        self.axes.grid(
            which="minor",
            axis="both",
            color="#808080",
            linewidth=0.5,
            alpha=MINOR_GRID_LINE_ALPHA,
        )
        self.axes.axhline(0.0, color="#606060", linewidth=0.8, alpha=0.8)
        apply_compact_plot_text(self.axes)

    def _draw_empty(self) -> None:
        clear_plot_axes(self.axes)
        self._lines = {}
        self._series_labels = ()
        self._plot_state = None
        self._reset_crosshair_artists()
        self._reset_comparison_interaction()
        self._configure_axes()
        self._sync_trace_filter_actions(())
        self._apply_manual_axis_limits()
        self._redraw_crosshair()
        self.draw_idle()

    def _format_crosshair_y(self, value: float) -> str:
        return f"{value:.3g} ms"

    def set_comparison_plot(
        self,
        freqs_hz: np.ndarray,
        trace_names: np.ndarray,
        group_delay_ms: np.ndarray,
    ) -> None:
        self._set_comparison_plot_state(self._normalized_plot_state(freqs_hz, trace_names, group_delay_ms))

    @staticmethod
    def _normalized_plot_state(
        freqs_hz: np.ndarray,
        trace_names: np.ndarray,
        group_delay_ms: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frequencies = np.asarray(freqs_hz, dtype=np.float32).copy()
        names = np.asarray(trace_names).copy()
        values = np.asarray(group_delay_ms, dtype=np.float32).copy()
        if values.shape != (names.size, frequencies.size):
            raise ValueError("Group-delay values must have shape (trace, frequency).")
        return frequencies, names, values

    def _current_plot_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if self._plot_state is None:
            return None
        return tuple(values.copy() for values in self._plot_state)

    def _apply_plot_state(self, state: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        self.update_plot(*state)

    def update_plot(
        self,
        freqs_hz: np.ndarray,
        trace_names: np.ndarray,
        group_delay_ms: np.ndarray,
    ) -> None:
        state = self._normalized_plot_state(freqs_hz, trace_names, group_delay_ms)
        if self._defer_current_plot_state(state):
            return
        freqs_hz, trace_names, group_delay_ms = state
        self._plot_state = state
        self._invalidate_crosshair_background()
        labels = tuple(str(value) for value in trace_names.tolist())
        if labels != self._series_labels:
            for line in self._lines.values():
                line.remove()
            self._lines = {}
            for index, label in enumerate(labels):
                self._lines[label] = self.axes.plot(
                    [],
                    [],
                    linewidth=2.0 if label == "Sum" else 1.3,
                    color=self._series_color(label, index),
                    label=label,
                )[0]
                self._series_visibility.setdefault(label, True)
            self._series_labels = labels
            self._sync_trace_filter_actions(labels)
        for index, label in enumerate(labels):
            self._lines[label].set_data(freqs_hz, group_delay_ms[index])
        self._apply_series_visibility()
        self._update_y_limits()
        self._apply_manual_axis_limits()
        apply_compact_plot_text(self.axes)
        if self._crosshair_visible:
            self._redraw_crosshair()
        self.draw_idle()

    def _series_color(self, label: str, index: int) -> Any:
        if label == "Sum":
            return "#000000"
        if label not in self._channel_colors:
            cycle = rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
            self._channel_colors[label] = cycle[(index - 1) % len(cycle)]
        return self._channel_colors[label]

    def _sync_trace_filter_actions(self, labels: tuple[str, ...]) -> None:
        self.trace_filter_menu.clear()
        self._series_actions = {}
        for label in labels:
            action = self.trace_filter_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(self._series_visibility.get(label, True))
            action.toggled.connect(lambda checked, series_label=label: self.set_series_visible(series_label, checked))
            self._series_actions[label] = action
        self.trace_filter_action.setEnabled(bool(labels))

    def set_series_visible(self, label: str, visible: bool) -> None:
        if label not in self._series_labels:
            return
        self._series_visibility[label] = bool(visible)
        self._apply_series_visibility()
        self._update_y_limits()
        self._apply_manual_axis_limits()
        self._invalidate_crosshair_background()
        self.draw_idle()

    def _apply_series_visibility(self) -> None:
        for label, line in self._lines.items():
            line.set_visible(self._series_visibility.get(label, True))
        legend = self.axes.get_legend()
        if legend is not None:
            legend.remove()
        visible = [line for line in self._lines.values() if line.get_visible()]
        if visible:
            self.axes.legend(visible, [line.get_label() for line in visible], loc="best")

    def _update_y_limits(self) -> None:
        finite_rows = []
        for line in self._lines.values():
            if not line.get_visible():
                continue
            values = np.asarray(line.get_ydata(), dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size:
                finite_rows.append(finite)
        if not finite_rows:
            self.axes.set_ylim(-0.1, 0.1)
            return
        finite = np.concatenate(finite_rows)
        minimum = min(0.0, float(np.min(finite)))
        maximum = max(0.0, float(np.max(finite)))
        span = maximum - minimum
        padding = max(0.08 * span, 0.05)
        self.axes.set_ylim(minimum - padding, maximum + padding)


__all__ = ["GroupDelayCanvas"]
