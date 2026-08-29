"""Per-channel electrical input-impedance plot canvas."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator, LogLocator, MaxNLocator
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu

from blab.ui.plots import (
    DUAL_AXIS_LAYOUT,
    GRID_LINE_ALPHA,
    MINOR_GRID_LINE_ALPHA,
    PLOT_TITLE_PAD,
    RawCoordinatePlotCanvas,
    _phase_curve_with_wrap_breaks,
    apply_audio_frequency_axis,
    apply_compact_plot_text,
    clear_plot_axes,
)


class ElectricalImpedanceCanvas(RawCoordinatePlotCanvas):
    """Plot parallel electrical load magnitude and optionally wrapped phase."""

    def __init__(self) -> None:
        self.figure = Figure(figsize=(6.5, 3.0), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.phase_axes = self.axes.twinx()
        self._magnitude_lines: dict[str, Any] = {}
        self._phase_lines: dict[str, Any] = {}
        self._lines: list[Any] = []
        self._series_labels: tuple[str, ...] = ()
        self._series_visibility: dict[str, bool] = {}
        self._series_actions: dict[str, QAction] = {}
        self._phase_available = False
        self._plot_state = None
        super().__init__(self.figure, "Electrical Impedance", secondary_axes=self.phase_axes)
        self.trace_filter_menu = QMenu("Traces", self)
        self.trace_filter_action = QAction("Traces", self)
        self.trace_filter_action.setToolTip("Choose visible electrical impedance traces")
        self.trace_filter_action.setMenu(self.trace_filter_menu)
        self.show_phase_action = QAction("Phase", self)
        self.show_phase_action.setToolTip("Show electrical impedance phase traces")
        self.show_phase_action.setCheckable(True)
        self.show_phase_action.setEnabled(False)
        self.show_phase_action.toggled.connect(self._set_phase_visible)
        self.set_layout_profile(DUAL_AXIS_LAYOUT)
        self.phase_axes.yaxis.set_label_position("right")
        self.phase_axes.yaxis.tick_right()
        self._draw_empty()

    def _configure_axes(self) -> None:
        self.axes.set_title(self.title, pad=PLOT_TITLE_PAD)
        self.axes.set_xlabel("Frequency (Hz)")
        self.axes.set_ylabel("Impedance (Ω)")
        self.phase_axes.set_ylabel("Phase (deg)")
        self.phase_axes.yaxis.set_label_position("right")
        self.phase_axes.yaxis.tick_right()
        apply_audio_frequency_axis(self.axes)
        self.phase_axes.set_ylim(-180.0, 600.0)
        self.phase_axes.set_yticks(np.arange(-180.0, 601.0, 90.0))
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
        apply_compact_plot_text(self.axes)
        apply_compact_plot_text(self.phase_axes)
        self.phase_axes.set_visible(False)

    def _draw_empty(self) -> None:
        clear_plot_axes(self.axes)
        clear_plot_axes(self.phase_axes)
        self._magnitude_lines = {}
        self._phase_lines = {}
        self._lines = []
        self._series_labels = ()
        self._phase_available = False
        self._plot_state = None
        self._reset_crosshair_artists()
        self._reset_comparison_interaction()
        self._configure_axes()
        self._sync_trace_filter_actions(())
        self.show_phase_action.setEnabled(False)
        self._apply_manual_axis_limits()
        self._redraw_crosshair()
        self.draw_idle()

    def _format_crosshair_y(self, value: float) -> str:
        return f"{value:.3g} Ω"

    def _format_secondary_crosshair_y(self, value: float) -> str:
        return f"{value:.1f} deg"

    def set_comparison_plot(
        self,
        freqs_hz: np.ndarray,
        channel_names: np.ndarray,
        magnitude_ohm: np.ndarray,
        phase_deg: np.ndarray,
    ) -> None:
        self._set_comparison_plot_state(
            self._normalized_plot_state(freqs_hz, channel_names, magnitude_ohm, phase_deg)
        )

    @staticmethod
    def _normalized_plot_state(
        freqs_hz: np.ndarray,
        channel_names: np.ndarray,
        magnitude_ohm: np.ndarray,
        phase_deg: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        frequencies = np.asarray(freqs_hz, dtype=np.float32).copy()
        names = np.asarray(channel_names).copy()
        magnitude = np.asarray(magnitude_ohm, dtype=np.float32).copy()
        phase = np.asarray(phase_deg, dtype=np.float32).copy()
        expected_shape = (names.size, frequencies.size)
        if magnitude.shape != expected_shape or phase.shape != expected_shape:
            raise ValueError("Electrical impedance arrays must have shape (channel, frequency).")
        return frequencies, names, magnitude, phase

    def _current_plot_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        if self._plot_state is None:
            return None
        return tuple(values.copy() for values in self._plot_state)

    def _apply_plot_state(
        self,
        state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        self.update_plot(*state)

    def update_plot(
        self,
        freqs_hz: np.ndarray,
        channel_names: np.ndarray,
        magnitude_ohm: np.ndarray,
        phase_deg: np.ndarray,
    ) -> None:
        state = self._normalized_plot_state(freqs_hz, channel_names, magnitude_ohm, phase_deg)
        if self._defer_current_plot_state(state):
            return
        freqs_hz, channel_names, magnitude_ohm, phase_deg = state
        self._plot_state = state
        self._invalidate_crosshair_background()
        labels = tuple(str(value) for value in channel_names.tolist())
        if labels != self._series_labels:
            for line in (*self._magnitude_lines.values(), *self._phase_lines.values()):
                line.remove()
            self._magnitude_lines = {}
            self._phase_lines = {}
            for label in labels:
                magnitude_line = self.axes.plot([], [], linewidth=1.5, label=label)[0]
                self._magnitude_lines[label] = magnitude_line
                self._phase_lines[label] = self.phase_axes.plot(
                    [],
                    [],
                    linewidth=1.2,
                    linestyle=":",
                    color=magnitude_line.get_color(),
                )[0]
                self._series_visibility.setdefault(label, True)
            self._lines = list(self._magnitude_lines.values())
            self._series_labels = labels
            self._sync_trace_filter_actions(labels)

        for index, label in enumerate(labels):
            self._magnitude_lines[label].set_data(freqs_hz, magnitude_ohm[index])
            phase_freqs, wrapped_phase = _phase_curve_with_wrap_breaks(
                freqs_hz,
                phase_deg[index],
            )
            self._phase_lines[label].set_data(phase_freqs, wrapped_phase)
        self._phase_available = bool(labels) and bool(np.any(np.isfinite(phase_deg)))
        self.show_phase_action.setEnabled(self._phase_available)
        self._apply_series_visibility()
        self._update_magnitude_limits()
        self._apply_manual_axis_limits()
        apply_compact_plot_text(self.axes)
        apply_compact_plot_text(self.phase_axes)
        if self._crosshair_visible:
            self._redraw_crosshair()
        self.draw_idle()

    def _sync_trace_filter_actions(self, labels: tuple[str, ...]) -> None:
        self.trace_filter_menu.clear()
        self._series_actions = {}
        for label in labels:
            action = self.trace_filter_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(self._series_visibility.get(label, True))
            action.toggled.connect(
                lambda checked, series_label=label: self.set_series_visible(series_label, checked)
            )
            self._series_actions[label] = action
        self.trace_filter_action.setEnabled(bool(labels))

    def set_series_visible(self, label: str, visible: bool) -> None:
        if label not in self._series_labels:
            return
        self._series_visibility[label] = bool(visible)
        self._apply_series_visibility()
        self._update_magnitude_limits()
        self._apply_manual_axis_limits()
        self._invalidate_crosshair_background()
        self.draw_idle()

    def _set_phase_visible(self, _visible: bool) -> None:
        self._apply_series_visibility()
        self._invalidate_crosshair_background()
        self.draw_idle()

    def _apply_series_visibility(self) -> None:
        show_phase = self.show_phase_action.isChecked() and self._phase_available
        any_phase_visible = False
        for label in self._series_labels:
            visible = self._series_visibility.get(label, True)
            self._magnitude_lines[label].set_visible(visible)
            phase_line = self._phase_lines[label]
            phase_visible = visible and show_phase and np.asarray(phase_line.get_ydata()).size > 0
            phase_line.set_visible(phase_visible)
            any_phase_visible = any_phase_visible or phase_visible
        self.phase_axes.set_visible(any_phase_visible)
        legend = self.axes.get_legend()
        if legend is not None:
            legend.remove()
        visible_lines = [
            self._magnitude_lines[label]
            for label in self._series_labels
            if self._series_visibility.get(label, True)
        ]
        if visible_lines:
            self.axes.legend(visible_lines, [line.get_label() for line in visible_lines], loc="best")

    def _update_magnitude_limits(self) -> None:
        finite_rows = []
        for label in self._series_labels:
            line = self._magnitude_lines[label]
            if not line.get_visible():
                continue
            values = np.asarray(line.get_ydata(), dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size:
                finite_rows.append(finite)
        maximum = float(np.max(np.concatenate(finite_rows))) if finite_rows else 0.0
        self.axes.set_ylim(0.0, max(1.0e-3, maximum * 1.08))


__all__ = ["ElectricalImpedanceCanvas"]
