"""Matplotlib canvases used by the live solver GUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib import rcParams
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import LogLocator, MultipleLocator
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu

from blab.plotting import VisualizerConfig
from blab.spinorama import SpinoramaCurves, compute_spinorama_from_planes

AUDIO_FREQ_MIN_HZ = 20
AUDIO_FREQ_MAX_HZ = 20000
AUDIO_AXIS_TICKS = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
AUDIO_AXIS_LABELS = ["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"]
FREQ_SLIDER_STEPS = 1000
ISOBAR_COLORBAR_MIN_TICK_STEP_DB = 3.0
LIVE_ISOBAR_ANGLE_SAMPLES = 180
LIVE_ISOBAR_FREQ_SAMPLES = 90
FINAL_ISOBAR_ANGLE_SAMPLES = 1000
FINAL_ISOBAR_FREQ_SAMPLES = 500
LIVE_ISOBAR_SHADING = "nearest"
FINAL_ISOBAR_SHADING = "gouraud"
PLOT_TITLE_SIZE = 11
PLOT_LABEL_SIZE = 9
PLOT_TICK_SIZE = 9
PLOT_LEGEND_SIZE = 9
PLOT_TITLE_PAD = 7
GRID_LINE_ALPHA = 0.6
MINOR_GRID_LINE_ALPHA = 0.3
ISOBAR_COLORBAR_GAP_PT = 7.0
ISOBAR_COLORBAR_WIDTH_PT = 10.0
ON_AXIS_DB_SPAN = 50.0
SPINORAMA_SPL_LIMITS = (-40.0, 10.0)
SPINORAMA_DI_LIMITS = (-5.0, 45.0)
# Legend baseline, in figure coordinates. Axes-relative placement scales with
# the axes height and falls off short panels.
SPINORAMA_LEGEND_BOTTOM = 0.006
ISOBAR_CROSSHAIR_COLOR = "#101214"
ISOBAR_CROSSHAIR_LABEL_FACE = "#f8fbff"
ISOBAR_CROSSHAIR_LABEL_EDGE = "#2868ff"
ISOBAR_CROSSHAIR_DB_FACE = "#101214"


@dataclass(frozen=True)
class PlotLayoutProfile:
    """Physical padding around plot axes, independent of dock dimensions."""

    left_pt: float
    right_pt: float
    top_pt: float = 22.0
    bottom_pt: float = 36.0
    min_axes_width_pt: float = 110.0
    min_axes_height_pt: float = 72.0


@dataclass(frozen=True)
class PlotAxisLimits:
    """User-domain limits for a plot's primary x and y axes."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def validated(self) -> PlotAxisLimits:
        values = np.asarray((self.x_min, self.x_max, self.y_min, self.y_max), dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Plot limits must be finite numbers.")
        if self.x_min <= 0.0:
            raise ValueError("The lower X limit must be greater than zero for frequency plots.")
        if self.x_min >= self.x_max:
            raise ValueError("The lower X limit must be less than the upper X limit.")
        if self.y_min >= self.y_max:
            raise ValueError("The lower Y limit must be less than the upper Y limit.")
        return self


SINGLE_AXIS_LAYOUT = PlotLayoutProfile(left_pt=52.0, right_pt=14.0)
DUAL_AXIS_LAYOUT = PlotLayoutProfile(left_pt=52.0, right_pt=52.0)
ISOBAR_LAYOUT = PlotLayoutProfile(left_pt=50.0, right_pt=58.0)
SPINORAMA_LAYOUT = PlotLayoutProfile(left_pt=52.0, right_pt=52.0, bottom_pt=58.0)


def apply_figure_layout(figure: Figure, profile: PlotLayoutProfile) -> None:
    """Apply point-based margins, scaling them only when a figure is exceptionally small."""

    width_pt, height_pt = np.asarray(figure.get_size_inches(), dtype=float) * 72.0
    left, right = _layout_axis_edges(
        width_pt,
        profile.left_pt,
        profile.right_pt,
        profile.min_axes_width_pt,
    )
    bottom, top = _layout_axis_edges(
        height_pt,
        profile.bottom_pt,
        profile.top_pt,
        profile.min_axes_height_pt,
    )
    figure.subplots_adjust(left=left, right=right, top=top, bottom=bottom)


def figure_point_fraction(figure: Figure, points: float, *, axis: str) -> float:
    """Convert physical points into a normalized figure-coordinate distance."""

    dimension_inches = figure.get_figwidth() if axis == "x" else figure.get_figheight()
    return float(points) / max(float(dimension_inches) * 72.0, 1.0)


def _layout_axis_edges(
    total_pt: float,
    leading_pt: float,
    trailing_pt: float,
    minimum_content_pt: float,
) -> tuple[float, float]:
    total = max(float(total_pt), 1.0)
    leading = max(float(leading_pt), 0.0)
    trailing = max(float(trailing_pt), 0.0)
    requested = leading + trailing
    available = max(total - min(float(minimum_content_pt), total * 0.72), total * 0.08)
    if requested > available and requested > 0.0:
        scale = available / requested
        leading *= scale
        trailing *= scale
    return leading / total, 1.0 - trailing / total


def apply_audio_frequency_axis(axes) -> None:
    axes.set_xscale("log")
    axes.set_xlim(AUDIO_FREQ_MIN_HZ, AUDIO_FREQ_MAX_HZ)
    axes.set_xticks(AUDIO_AXIS_TICKS)
    axes.set_xticklabels(AUDIO_AXIS_LABELS)


def apply_log_image_frequency_axis(axes) -> None:
    if axes.get_xscale() != "linear":
        axes.set_xscale("linear")
    axes.set_xlim(np.log10(AUDIO_FREQ_MIN_HZ), np.log10(AUDIO_FREQ_MAX_HZ))
    axes.set_xticks(np.log10(AUDIO_AXIS_TICKS))
    axes.set_xticklabels(AUDIO_AXIS_LABELS)


def apply_compact_plot_text(axes) -> None:
    axes.title.set_fontsize(PLOT_TITLE_SIZE)
    axes.xaxis.label.set_size(PLOT_LABEL_SIZE)
    axes.yaxis.label.set_size(PLOT_LABEL_SIZE)
    axes.tick_params(axis="both", which="major", labelsize=PLOT_TICK_SIZE)
    legend = axes.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(PLOT_LEGEND_SIZE)


def clear_plot_axes(axes) -> None:
    if axes.get_xscale() != "linear":
        axes.set_xscale("linear")
    axes.clear()


def frequency_to_slider_value(freq_hz: int | float) -> int:
    clamped = min(max(float(freq_hz), AUDIO_FREQ_MIN_HZ), AUDIO_FREQ_MAX_HZ)
    fraction = (np.log10(clamped) - np.log10(AUDIO_FREQ_MIN_HZ)) / (
        np.log10(AUDIO_FREQ_MAX_HZ) - np.log10(AUDIO_FREQ_MIN_HZ)
    )
    return int(round(fraction * FREQ_SLIDER_STEPS))


def slider_value_to_frequency(value: int) -> int:
    fraction = float(value) / FREQ_SLIDER_STEPS
    log_freq = np.log10(AUDIO_FREQ_MIN_HZ) + fraction * (np.log10(AUDIO_FREQ_MAX_HZ) - np.log10(AUDIO_FREQ_MIN_HZ))
    return int(round(10.0**log_freq))


def _format_isobar_crosshair_frequency(freq_hz: float) -> str:
    if freq_hz >= 1000.0:
        return f"{freq_hz / 1000.0:.2f}".rstrip("0").rstrip(".") + " kHz"
    return f"{freq_hz:.0f} Hz"


def _grid_bracket(values: np.ndarray, target: float) -> tuple[int, int, float]:
    axis = np.asarray(values, dtype=float)
    if axis.size <= 1:
        return 0, 0, 0.0
    if axis[0] > axis[-1]:
        mirrored_left, mirrored_right, fraction = _grid_bracket(axis[::-1], target)
        last = axis.size - 1
        return last - mirrored_left, last - mirrored_right, fraction
    if target <= axis[0]:
        return 0, 0, 0.0
    if target >= axis[-1]:
        last = axis.size - 1
        return last, last, 0.0
    right = int(np.searchsorted(axis, target, side="right"))
    left = max(0, right - 1)
    span = axis[right] - axis[left]
    fraction = 0.0 if np.isclose(span, 0.0) else float((target - axis[left]) / span)
    return left, right, float(np.clip(fraction, 0.0, 1.0))


class InteractivePlotCanvas(FigureCanvas):
    """Shared mouse lifecycle for plot crosshairs and solve comparisons."""

    def __init__(self, figure: Figure, title: str) -> None:
        self.title = title
        self._manual_axis_limits: PlotAxisLimits | None = None
        self._automatic_axis_limits_snapshot: PlotAxisLimits | None = None
        self._layout_profile: PlotLayoutProfile | None = None
        self._comparison_plot: Any | None = None
        self._comparison_restore_plot: Any | None = None
        self._comparison_active = False
        self._crosshair_visible = False
        self._crosshair_dragging = False
        self._crosshair_background = None
        super().__init__(figure)
        self._connect_interaction_events()

    @property
    def automatic_axis_limits(self) -> bool:
        return self._manual_axis_limits is None

    def displayed_axis_limits(self) -> PlotAxisLimits:
        x_min, x_max = self._x_limits_to_user_domain(self.axes.get_xlim())
        y_min, y_max = self.axes.get_ylim()
        return PlotAxisLimits(float(x_min), float(x_max), float(y_min), float(y_max))

    def set_axis_limits(self, limits: PlotAxisLimits | None) -> None:
        if limits is None:
            automatic_limits = self._automatic_axis_limits_snapshot
            self._manual_axis_limits = None
            self._automatic_axis_limits_snapshot = None
            if automatic_limits is not None:
                self._apply_axis_limits(automatic_limits)
            state = self._current_plot_state()
            if state is None:
                self._draw_empty()
            else:
                self._apply_plot_state(state)
            return
        if self._manual_axis_limits is None:
            self._automatic_axis_limits_snapshot = self.displayed_axis_limits()
        self._manual_axis_limits = limits.validated()
        self._apply_manual_axis_limits()
        self._invalidate_crosshair_background()
        if self._crosshair_visible:
            self._redraw_crosshair()
        self.draw_idle()

    def _x_limits_to_user_domain(self, limits: tuple[float, float]) -> tuple[float, float]:
        return float(limits[0]), float(limits[1])

    def _x_limits_from_user_domain(self, limits: tuple[float, float]) -> tuple[float, float]:
        return float(limits[0]), float(limits[1])

    def _apply_manual_axis_limits(self) -> None:
        limits = self._manual_axis_limits
        if limits is None:
            return
        self._apply_axis_limits(limits)

    def _apply_axis_limits(self, limits: PlotAxisLimits) -> None:
        self.axes.set_xlim(*self._x_limits_from_user_domain((limits.x_min, limits.x_max)))
        self.axes.set_ylim(limits.y_min, limits.y_max)

    def set_layout_profile(self, profile: PlotLayoutProfile) -> None:
        self._layout_profile = profile
        self._apply_layout_profile()

    def _apply_layout_profile(self) -> None:
        if self._layout_profile is None:
            return
        apply_figure_layout(self.figure, self._layout_profile)
        self._after_layout_profile_applied()

    def _apply_layout(self) -> None:
        """Re-apply the layout after an artist outside the axes comes or goes.

        ``_remove_colorbar`` calls this on every canvas, so the base class has to
        answer it. ``SpinoramaCanvas`` overrides it to also re-anchor its
        right-hand DI axis; nothing else needs more than the layout profile.
        """
        self._apply_layout_profile()

    def _after_layout_profile_applied(self) -> None:
        """Reposition manually managed artists after the axes rectangle changes."""

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._apply_layout_profile()
        self.draw_idle()

    def _connect_interaction_events(self) -> None:
        self.mpl_connect("button_press_event", self._on_crosshair_button_press)
        self.mpl_connect("button_release_event", self._on_crosshair_button_release)
        self.mpl_connect("motion_notify_event", self._on_crosshair_motion)
        self.mpl_connect("draw_event", self._on_crosshair_draw)
        self.mpl_connect("button_press_event", self._on_comparison_button_press)
        self.mpl_connect("button_release_event", self._on_comparison_button_release)
        self.mpl_connect("figure_leave_event", self._on_figure_leave)

    def _interactive_axes(self) -> tuple[object, ...]:
        return (self.axes,)

    def _event_is_in_plot(self, event) -> bool:
        return any(event.inaxes is axes for axes in self._interactive_axes())

    def _set_comparison_plot_state(self, state: Any) -> None:
        self._comparison_plot = state
        self.setToolTip("Hold the right mouse button over the plot to view the previous completed solve.")

    def clear_comparison_plot(self) -> None:
        self._end_comparison()
        self._comparison_plot = None
        self._comparison_restore_plot = None
        self.setToolTip("")

    def _defer_current_plot_state(self, state: Any) -> bool:
        if not self._comparison_active:
            return False
        self._comparison_restore_plot = state
        return True

    def _reset_comparison_interaction(self) -> None:
        self._comparison_restore_plot = None
        self._comparison_active = False

    def _current_plot_state(self) -> Any | None:
        raise NotImplementedError

    def _apply_plot_state(self, state: Any) -> None:
        raise NotImplementedError

    def _on_comparison_button_press(self, event) -> None:
        if not self._event_is_in_plot(event) or event.button != MouseButton.RIGHT:
            return
        if self._comparison_active or self._comparison_plot is None:
            return
        current_plot = self._current_plot_state()
        if current_plot is None:
            return
        self._comparison_restore_plot = current_plot
        self._apply_plot_state(self._comparison_plot)
        self._comparison_active = True
        self.axes.set_title(f"{self.title} - Previous Solve", pad=PLOT_TITLE_PAD)
        self._invalidate_crosshair_background()
        self.draw_idle()

    def _on_comparison_button_release(self, event) -> None:
        if event.button == MouseButton.RIGHT:
            self._end_comparison()

    def _end_comparison(self) -> None:
        if not self._comparison_active:
            return
        restore_plot = self._comparison_restore_plot
        self._comparison_restore_plot = None
        self._comparison_active = False
        if restore_plot is not None:
            self._apply_plot_state(restore_plot)
        self.axes.set_title(self.title, pad=PLOT_TITLE_PAD)
        self._invalidate_crosshair_background()
        self.draw_idle()

    def _on_figure_leave(self, event) -> None:
        del event
        self._crosshair_dragging = False
        self._end_comparison()

    def _on_crosshair_button_press(self, event) -> None:
        if not self._event_is_in_plot(event) or event.button != MouseButton.LEFT:
            return
        if getattr(event, "dblclick", False):
            self._hide_crosshair()
            return
        if not self._has_crosshair_data():
            return
        self._crosshair_dragging = True
        self._set_crosshair_from_event(event)

    def _on_crosshair_button_release(self, event) -> None:
        if event.button == MouseButton.LEFT:
            self._crosshair_dragging = False

    def _on_crosshair_motion(self, event) -> None:
        if not self._crosshair_dragging or not self._event_is_in_plot(event):
            return
        self._set_crosshair_from_event(event)

    def _on_crosshair_draw(self, event) -> None:
        if event.canvas is not self:
            return
        try:
            self._crosshair_background = self.copy_from_bbox(self.figure.bbox)
        except Exception:
            self._crosshair_background = None
        if self._crosshair_visible:
            self._draw_crosshair_overlay()

    def _invalidate_crosshair_background(self) -> None:
        self._crosshair_background = None

    def _draw_crosshair_overlay(self) -> bool:
        if self._crosshair_background is None:
            return False
        try:
            self.restore_region(self._crosshair_background)
            for artist in self._crosshair_artists():
                if artist is not None and artist.get_visible():
                    artist.axes.draw_artist(artist)
            self.blit(self.figure.bbox)
            self.flush_events()
            return True
        except Exception:
            self._crosshair_background = None
            return False

    def _hide_crosshair(self) -> None:
        self._crosshair_visible = False
        self._crosshair_dragging = False
        self._set_crosshair_artist_visibility(False)
        if not self._draw_crosshair_overlay():
            self.draw_idle()

    def _set_crosshair_artist_visibility(self, visible: bool) -> None:
        for artist in self._crosshair_artists():
            if artist is not None:
                artist.set_visible(visible)

    def _has_crosshair_data(self) -> bool:
        raise NotImplementedError

    def _set_crosshair_from_event(self, event) -> None:
        raise NotImplementedError

    def _crosshair_artists(self) -> tuple[object, ...]:
        raise NotImplementedError

    def _redraw_crosshair(self) -> None:
        raise NotImplementedError


class IsobarCanvas(InteractivePlotCanvas):
    def __init__(
        self,
        title: str,
        *,
        left_margin: float | None = None,
        right_margin: float | None = None,
        show_colorbar: bool = True,
    ):
        self.figure = Figure(figsize=(5.5, 2.8), dpi=100)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure, title)
        self.show_colorbar = bool(show_colorbar)
        profile = ISOBAR_LAYOUT if self.show_colorbar else SINGLE_AXIS_LAYOUT
        if left_margin is not None or right_margin is not None:
            figure_width_pt = self.figure.get_figwidth() * 72.0
            profile = PlotLayoutProfile(
                left_pt=(
                    profile.left_pt
                    if left_margin is None
                    else float(left_margin) * figure_width_pt
                ),
                right_pt=(
                    profile.right_pt
                    if right_margin is None
                    else (1.0 - float(right_margin)) * figure_width_pt
                ),
            )
        self.colors = VisualizerConfig.custom_colors
        self._colormap = LinearSegmentedColormap.from_list("boundary_lab_isobar", list(self.colors), N=256)
        self._mesh_artist = None
        self._image_artist = None
        self._line_artist = None
        self._contour_artist = None
        self._colorbar = None
        self._colorbar_axes = None
        self._colorbar_mappable = None
        self._colorbar_state: tuple[float, float, float] | None = None
        self._mesh_freqs_hz: np.ndarray | None = None
        self._mesh_angles_deg: np.ndarray | None = None
        self._mesh_values_db: np.ndarray | None = None
        self._mesh_clip: tuple[float, float] | None = None
        self._mesh_shading: str | None = None
        self._mesh_render_mode: str | None = None
        self._mesh_contour_step_db: float | None = None
        self._x_axis_mode = "frequency"
        self._axis_configuration_mode: str | None = None
        self._captured_contours: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self._crosshair_freq_hz: float | None = None
        self._crosshair_angle_deg: float | None = None
        self._crosshair_vline = None
        self._crosshair_hline = None
        self._crosshair_freq_label = None
        self._crosshair_angle_label = None
        self._crosshair_db_label = None
        self.set_layout_profile(profile)
        self._draw_empty()

    def _configure_axes(self) -> None:
        self._apply_layout_profile()
        self.axes.set_title(self.title, pad=PLOT_TITLE_PAD)
        self.axes.set_xlabel("Frequency (Hz)")
        self.axes.set_ylabel("Angle (deg)")
        if self._x_axis_mode == "log_image":
            apply_log_image_frequency_axis(self.axes)
        else:
            apply_audio_frequency_axis(self.axes)
        self.axes.set_ylim(-180, 180)
        self.axes.set_yticks(np.arange(-180, 181, 45))
        self.axes.grid(which="major", color="#808080", linewidth=0.8, alpha=GRID_LINE_ALPHA)
        apply_compact_plot_text(self.axes)
        self._axis_configuration_mode = self._x_axis_mode

    def _x_limits_to_user_domain(self, limits: tuple[float, float]) -> tuple[float, float]:
        if self._x_axis_mode == "log_image":
            return float(10.0 ** limits[0]), float(10.0 ** limits[1])
        return super()._x_limits_to_user_domain(limits)

    def _x_limits_from_user_domain(self, limits: tuple[float, float]) -> tuple[float, float]:
        if self._x_axis_mode == "log_image":
            return float(np.log10(limits[0])), float(np.log10(limits[1]))
        return super()._x_limits_from_user_domain(limits)

    def _draw_empty(self) -> None:
        clear_plot_axes(self.axes)
        self._mesh_artist = None
        self._image_artist = None
        self._line_artist = None
        self._contour_artist = None
        self._crosshair_vline = None
        self._crosshair_hline = None
        self._crosshair_freq_label = None
        self._crosshair_angle_label = None
        self._crosshair_db_label = None
        self._crosshair_background = None
        self._reset_comparison_interaction()
        self._remove_colorbar()
        self._mesh_freqs_hz = None
        self._mesh_angles_deg = None
        self._mesh_values_db = None
        self._mesh_clip = None
        self._mesh_shading = None
        self._mesh_render_mode = None
        self._x_axis_mode = "frequency"
        self._configure_axes()
        self._apply_manual_axis_limits()
        self._redraw_captured_contours()
        self._redraw_crosshair()
        self.draw_idle()

    def set_comparison_plot(
        self,
        freqs_hz: np.ndarray,
        angles_deg: np.ndarray,
        values_db: np.ndarray,
        clip_min_db: float,
        clip_max_db: float,
        *,
        shading: str = FINAL_ISOBAR_SHADING,
        contour_step_db: float = 3.0,
    ) -> None:
        self._set_comparison_plot_state(
            (
                np.asarray(freqs_hz, dtype=np.float32).copy(),
                np.asarray(angles_deg, dtype=np.float32).copy(),
                np.asarray(values_db, dtype=np.float32).copy(),
                float(clip_min_db),
                float(clip_max_db),
                str(shading or FINAL_ISOBAR_SHADING),
                max(0.0, float(contour_step_db)),
            )
        )

    def _current_plot_state(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, str, float] | None:
        if (
            self._mesh_freqs_hz is None
            or self._mesh_angles_deg is None
            or self._mesh_values_db is None
            or self._mesh_clip is None
            or self._mesh_shading is None
        ):
            return None
        return (
            self._mesh_freqs_hz.copy(),
            self._mesh_angles_deg.copy(),
            self._mesh_values_db.copy(),
            self._mesh_clip[0],
            self._mesh_clip[1],
            self._mesh_shading,
            3.0 if self._mesh_contour_step_db is None else self._mesh_contour_step_db,
        )

    def _apply_plot_state(
        self,
        state: tuple[np.ndarray, np.ndarray, np.ndarray, float, float, str, float],
    ) -> None:
        freqs_hz, angles_deg, values_db, clip_min_db, clip_max_db, shading, contour_step_db = state
        self.update_plot(
            freqs_hz,
            angles_deg,
            values_db,
            clip_min_db,
            clip_max_db,
            shading=shading,
            contour_step_db=contour_step_db,
        )

    def _crosshair_artists(self) -> tuple[object, ...]:
        return (
            self._crosshair_vline,
            self._crosshair_hline,
            self._crosshair_freq_label,
            self._crosshair_angle_label,
            self._crosshair_db_label,
        )

    def _has_crosshair_data(self) -> bool:
        return (
            self._mesh_freqs_hz is not None
            and self._mesh_angles_deg is not None
            and self._mesh_values_db is not None
            and self._mesh_freqs_hz.size > 0
            and self._mesh_angles_deg.size > 0
        )

    def _set_crosshair_from_event(self, event) -> None:
        if event.xdata is None or event.ydata is None:
            return
        freq_hz = self._axis_x_to_frequency(float(event.xdata))
        angle_deg = float(event.ydata)
        self._set_crosshair_position(freq_hz, angle_deg)

    def _axis_x_to_frequency(self, x_value: float) -> float:
        if self._x_axis_mode == "log_image":
            return float(10.0**x_value)
        return float(x_value)

    def _frequency_to_axis_x(self, freq_hz: float) -> float:
        freq = max(float(freq_hz), np.finfo(float).tiny)
        if self._x_axis_mode == "log_image":
            return float(np.log10(freq))
        return freq

    def _clamp_crosshair_position(self, freq_hz: float, angle_deg: float) -> tuple[float, float]:
        if self._mesh_freqs_hz is None or self._mesh_angles_deg is None:
            return float(freq_hz), float(angle_deg)
        freq_min = float(np.nanmin(self._mesh_freqs_hz))
        freq_max = float(np.nanmax(self._mesh_freqs_hz))
        angle_min = float(np.nanmin(self._mesh_angles_deg))
        angle_max = float(np.nanmax(self._mesh_angles_deg))
        return (
            float(np.clip(freq_hz, freq_min, freq_max)),
            float(np.clip(angle_deg, angle_min, angle_max)),
        )

    def _set_crosshair_position(self, freq_hz: float, angle_deg: float) -> None:
        if not self._has_crosshair_data():
            return
        self._crosshair_freq_hz, self._crosshair_angle_deg = self._clamp_crosshair_position(freq_hz, angle_deg)
        self._crosshair_visible = True
        self._redraw_crosshair()
        if not self._draw_crosshair_overlay():
            self.draw_idle()

    def _ensure_crosshair_artists(self) -> None:
        if self._crosshair_vline is None:
            self._crosshair_vline = self.axes.axvline(
                AUDIO_FREQ_MIN_HZ,
                color=ISOBAR_CROSSHAIR_COLOR,
                linewidth=0.9,
                alpha=0.9,
                zorder=9,
                visible=False,
            )
        if self._crosshair_hline is None:
            self._crosshair_hline = self.axes.axhline(
                0.0,
                color=ISOBAR_CROSSHAIR_COLOR,
                linewidth=0.9,
                alpha=0.9,
                zorder=9,
                visible=False,
            )
        if self._crosshair_freq_label is None:
            self._crosshair_freq_label = self.axes.text(
                AUDIO_FREQ_MIN_HZ,
                -0.075,
                "",
                transform=self.axes.get_xaxis_transform(),
                ha="center",
                va="top",
                color=ISOBAR_CROSSHAIR_LABEL_EDGE,
                fontsize=PLOT_TICK_SIZE,
                bbox={
                    "boxstyle": "square,pad=0.12",
                    "facecolor": ISOBAR_CROSSHAIR_LABEL_FACE,
                    "edgecolor": ISOBAR_CROSSHAIR_LABEL_EDGE,
                    "linewidth": 0.8,
                },
                clip_on=False,
                zorder=10,
                visible=False,
            )
        if self._crosshair_angle_label is None:
            self._crosshair_angle_label = self.axes.text(
                -0.01,
                0.0,
                "",
                transform=self.axes.get_yaxis_transform(),
                ha="right",
                va="center",
                color=ISOBAR_CROSSHAIR_LABEL_EDGE,
                fontsize=PLOT_TICK_SIZE,
                bbox={
                    "boxstyle": "square,pad=0.12",
                    "facecolor": ISOBAR_CROSSHAIR_LABEL_FACE,
                    "edgecolor": ISOBAR_CROSSHAIR_LABEL_EDGE,
                    "linewidth": 0.8,
                },
                clip_on=False,
                zorder=10,
                visible=False,
            )
        if self._crosshair_db_label is None:
            self._crosshair_db_label = self.axes.annotate(
                "",
                xy=(AUDIO_FREQ_MIN_HZ, 0.0),
                xytext=(8, 8),
                textcoords="offset points",
                ha="left",
                va="bottom",
                color="#ffffff",
                fontsize=PLOT_TICK_SIZE,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": ISOBAR_CROSSHAIR_DB_FACE,
                    "edgecolor": "none",
                    "linewidth": 0.0,
                    "alpha": 0.75,
                },
                zorder=10,
                visible=False,
            )

        for artist in self._crosshair_artists():
            if artist is not None:
                artist.set_animated(True)

    def _redraw_crosshair(self) -> None:
        if not self._crosshair_visible or self._crosshair_freq_hz is None or self._crosshair_angle_deg is None:
            self._set_crosshair_artist_visibility(False)
            return
        if not self._has_crosshair_data():
            self._set_crosshair_artist_visibility(False)
            return
        self._crosshair_freq_hz, self._crosshair_angle_deg = self._clamp_crosshair_position(
            self._crosshair_freq_hz,
            self._crosshair_angle_deg,
        )
        axis_x = self._frequency_to_axis_x(self._crosshair_freq_hz)
        db_value = self._crosshair_value_db(self._crosshair_freq_hz, self._crosshair_angle_deg)
        self._ensure_crosshair_artists()
        self._crosshair_vline.set_xdata([axis_x, axis_x])
        self._crosshair_hline.set_ydata([self._crosshair_angle_deg, self._crosshair_angle_deg])
        self._crosshair_freq_label.set_position((axis_x, -0.075))
        self._crosshair_freq_label.set_text(_format_isobar_crosshair_frequency(self._crosshair_freq_hz))
        self._crosshair_angle_label.set_position((-0.01, self._crosshair_angle_deg))
        self._crosshair_angle_label.set_text(f"{int(round(self._crosshair_angle_deg)):+d} deg")
        self._crosshair_db_label.xy = (axis_x, self._crosshair_angle_deg)
        self._crosshair_db_label.set_text("" if db_value is None else f"{db_value:.1f} dB")
        self._set_crosshair_artist_visibility(True)

    def _crosshair_value_db(self, freq_hz: float, angle_deg: float) -> float | None:
        if not self._has_crosshair_data():
            return None
        freqs = np.asarray(self._mesh_freqs_hz, dtype=float)
        angles = np.asarray(self._mesh_angles_deg, dtype=float)
        values = np.asarray(self._mesh_values_db, dtype=float)
        if values.shape != (angles.size, freqs.size):
            return None
        log_freqs = np.log10(np.maximum(freqs, np.finfo(float).tiny))
        log_freq = float(np.log10(max(float(freq_hz), np.finfo(float).tiny)))
        angle_left, angle_right, angle_fraction = _grid_bracket(angles, float(angle_deg))
        freq_left, freq_right, freq_fraction = _grid_bracket(log_freqs, log_freq)
        top_left = values[angle_left, freq_left]
        top_right = values[angle_left, freq_right]
        bottom_left = values[angle_right, freq_left]
        bottom_right = values[angle_right, freq_right]
        top = top_left + freq_fraction * (top_right - top_left)
        bottom = bottom_left + freq_fraction * (bottom_right - bottom_left)
        interpolated = top + angle_fraction * (bottom - top)
        return float(interpolated) if np.isfinite(interpolated) else None

    def _remove_artist(self, name: str) -> None:
        artist = getattr(self, name)
        if artist is None:
            return
        try:
            artist.remove()
        except ValueError:
            pass
        setattr(self, name, None)

    def _remove_mesh_artists(self) -> None:
        self._remove_artist("_mesh_artist")
        self._remove_artist("_image_artist")

    def _remove_colorbar(self) -> None:
        if self._colorbar is None and self._colorbar_axes is None:
            self._colorbar_mappable = None
            self._colorbar_state = None
            return
        if self._colorbar is not None:
            try:
                self._colorbar.remove()
            except (AttributeError, KeyError, ValueError):
                pass
        elif self._colorbar_axes is not None:
            try:
                self._colorbar_axes.remove()
            except (AttributeError, KeyError, ValueError):
                pass
        self._colorbar = None
        self._colorbar_axes = None
        self._colorbar_mappable = None
        self._colorbar_state = None
        self._apply_layout()

    def _remove_contour_artist(self) -> None:
        if self._contour_artist is None:
            return
        try:
            self._contour_artist.remove()
        except (AttributeError, NotImplementedError, ValueError):
            for collection in getattr(self._contour_artist, "collections", ()):
                try:
                    collection.remove()
                except (NotImplementedError, ValueError):
                    pass
        self._contour_artist = None

    @property
    def has_captured_contours(self) -> bool:
        return self._captured_contours is not None

    def _current_contour_levels(
        self,
        clip: tuple[float, float],
        values_db: np.ndarray,
        contour_step_db: float,
    ) -> np.ndarray:
        if contour_step_db <= 0.0:
            return np.asarray([], dtype=np.float32)
        clip_min_db, clip_max_db = clip
        finite = values_db[np.isfinite(values_db)]
        if finite.size == 0:
            return np.asarray([], dtype=np.float32)
        data_min = float(np.min(finite))
        data_max = float(np.max(finite))
        levels = np.arange(
            np.ceil(clip_min_db / contour_step_db) * contour_step_db,
            clip_max_db + 0.5 * contour_step_db,
            contour_step_db,
            dtype=np.float32,
        )
        return levels[(levels > clip_min_db) & (levels < clip_max_db) & (levels > data_min) & (levels < data_max)]

    def _redraw_captured_contours(self) -> None:
        self._remove_contour_artist()
        if self._captured_contours is None:
            return
        freqs_hz, angles_deg, values_db, levels = self._captured_contours
        if freqs_hz.size < 2 or angles_deg.size < 2 or levels.size == 0:
            return
        x_values = np.log10(freqs_hz) if self._x_axis_mode == "log_image" else freqs_hz
        self._contour_artist = self.axes.contour(
            x_values,
            angles_deg,
            values_db,
            levels=levels,
            colors="white",
            linewidths=0.9,
            linestyles="solid",
            alpha=0.85,
            zorder=5,
        )

    def capture_contours(self) -> bool:
        if (
            self._mesh_freqs_hz is None
            or self._mesh_angles_deg is None
            or self._mesh_values_db is None
            or self._mesh_clip is None
            or self._mesh_freqs_hz.size < 2
            or self._mesh_angles_deg.size < 2
        ):
            return False
        levels = self._current_contour_levels(
            self._mesh_clip,
            self._mesh_values_db,
            3.0 if self._mesh_contour_step_db is None else self._mesh_contour_step_db,
        )
        if levels.size == 0:
            return False
        self._captured_contours = (
            self._mesh_freqs_hz.copy(),
            self._mesh_angles_deg.copy(),
            self._mesh_values_db.copy(),
            levels.copy(),
        )
        self._redraw_captured_contours()
        self._invalidate_crosshair_background()
        self.draw_idle()
        return True

    def clear_contours(self) -> None:
        self._captured_contours = None
        self._remove_contour_artist()
        self._invalidate_crosshair_background()
        self.draw_idle()

    def _mesh_matches(
        self,
        freqs_hz: np.ndarray,
        angles_deg: np.ndarray,
        clip: tuple[float, float],
        shading: str,
        contour_step_db: float,
    ) -> bool:
        return (
            (self._mesh_artist is not None or self._image_artist is not None)
            and self._mesh_freqs_hz is not None
            and self._mesh_angles_deg is not None
            and self._mesh_clip == clip
            and self._mesh_shading == shading
            and self._mesh_contour_step_db == contour_step_db
            and self._mesh_render_mode is not None
            and self._mesh_freqs_hz.shape == freqs_hz.shape
            and self._mesh_angles_deg.shape == angles_deg.shape
            and np.array_equal(self._mesh_freqs_hz, freqs_hz)
            and np.array_equal(self._mesh_angles_deg, angles_deg)
        )

    def _isobar_colormap(self):
        return self._colormap

    def _color_boundaries(self, clip_min_db: float, clip_max_db: float, contour_step_db: float) -> np.ndarray:
        if contour_step_db <= 0.0 or clip_max_db <= clip_min_db:
            return np.asarray([], dtype=np.float32)
        first = np.ceil(clip_min_db / contour_step_db) * contour_step_db
        inner = np.arange(first, clip_max_db + 0.5 * contour_step_db, contour_step_db, dtype=np.float32)
        inner = inner[(inner > clip_min_db) & (inner < clip_max_db)]
        return np.concatenate(
            (
                np.asarray([clip_min_db], dtype=np.float32),
                inner,
                np.asarray([clip_max_db], dtype=np.float32),
            )
        )

    def _color_mapping(self, clip_min_db: float, clip_max_db: float, contour_step_db: float):
        cmap = self._isobar_colormap()
        boundaries = self._color_boundaries(clip_min_db, clip_max_db, contour_step_db)
        if boundaries.size >= 2:
            return cmap, BoundaryNorm(boundaries, cmap.N)
        return cmap, Normalize(vmin=clip_min_db, vmax=clip_max_db)

    def _colorbar_ticks(self, clip_min_db: float, clip_max_db: float, contour_step_db: float) -> np.ndarray:
        tick_step_db = max(ISOBAR_COLORBAR_MIN_TICK_STEP_DB, contour_step_db)
        start = np.ceil(clip_min_db / tick_step_db) * tick_step_db
        end = np.floor(clip_max_db / tick_step_db) * tick_step_db
        if end < start:
            return np.asarray([clip_min_db, clip_max_db], dtype=np.float32)
        return np.arange(start, end + 0.5 * tick_step_db, tick_step_db, dtype=np.float32)

    def _update_colorbar(
        self,
        clip_min_db: float,
        clip_max_db: float,
        contour_step_db: float,
        cmap,
        norm,
    ) -> None:
        if not self.show_colorbar:
            if self._colorbar is not None or self._colorbar_axes is not None:
                self._remove_colorbar()
            return
        state = (float(clip_min_db), float(clip_max_db), float(contour_step_db))
        if self._colorbar is not None and self._colorbar_mappable is not None:
            if self._colorbar_state == state:
                return
            self._colorbar_mappable.set_cmap(cmap)
            self._colorbar_mappable.set_norm(norm)
            self._colorbar.update_normal(self._colorbar_mappable)
        else:
            self._colorbar_mappable = ScalarMappable(norm=norm, cmap=cmap)
            self._colorbar_mappable.set_array([])
            self._colorbar_axes = self.figure.add_axes(self._colorbar_bounds())
            self._colorbar = self.figure.colorbar(
                self._colorbar_mappable,
                cax=self._colorbar_axes,
            )
        self._colorbar.set_ticks(self._colorbar_ticks(clip_min_db, clip_max_db, contour_step_db))
        self._colorbar.ax.tick_params(labelsize=PLOT_TICK_SIZE)
        self._colorbar_state = state

    def _after_layout_profile_applied(self) -> None:
        if self._colorbar_axes is not None:
            self._colorbar_axes.set_position(self._colorbar_bounds())

    def _colorbar_bounds(self) -> list[float]:
        axes_position = self.axes.get_position()
        gap = figure_point_fraction(self.figure, ISOBAR_COLORBAR_GAP_PT, axis="x")
        width = figure_point_fraction(self.figure, ISOBAR_COLORBAR_WIDTH_PT, axis="x")
        return [
            axes_position.x1 + gap,
            axes_position.y0,
            width,
            axes_position.height,
        ]

    def _image_from_values(
        self,
        clipped: np.ndarray,
        clip_min_db: float,
        clip_max_db: float,
        contour_step_db: float,
    ) -> np.ndarray:
        cmap, norm = self._color_mapping(clip_min_db, clip_max_db, contour_step_db)
        return np.asarray(cmap(norm(clipped)), dtype=np.float32)

    def update_plot(
        self,
        freqs_hz: np.ndarray,
        angles_deg: np.ndarray,
        values_db: np.ndarray,
        clip_min_db: float,
        clip_max_db: float,
        *,
        shading: str = LIVE_ISOBAR_SHADING,
        contour_step_db: float = 3.0,
    ) -> None:
        freqs_hz = np.asarray(freqs_hz, dtype=np.float32)
        angles_deg = np.asarray(angles_deg, dtype=np.float32)
        clipped = np.clip(np.asarray(values_db, dtype=np.float32), clip_min_db, clip_max_db)
        clip = (float(clip_min_db), float(clip_max_db))
        shading = str(shading or LIVE_ISOBAR_SHADING)
        contour_step_db = max(0.0, float(contour_step_db))
        render_mode = "image" if shading == FINAL_ISOBAR_SHADING else "mesh"
        state = (
            freqs_hz.copy(),
            angles_deg.copy(),
            clipped.copy(),
            float(clip_min_db),
            float(clip_max_db),
            shading,
            contour_step_db,
        )
        if self._defer_current_plot_state(state):
            return
        self._invalidate_crosshair_background()

        if freqs_hz.size >= 2 and angles_deg.size >= 2:
            self._remove_artist("_line_artist")
            cmap, norm = self._color_mapping(clip_min_db, clip_max_db, contour_step_db)
            self._update_colorbar(clip_min_db, clip_max_db, contour_step_db, cmap, norm)
            if render_mode == "image":
                self._x_axis_mode = "log_image"
                image = np.asarray(cmap(norm(clipped)), dtype=np.float32)
                if (
                    self._mesh_matches(freqs_hz, angles_deg, clip, shading, contour_step_db)
                    and self._image_artist is not None
                ):
                    self._image_artist.set_data(image)
                else:
                    self._remove_mesh_artists()
                    log_freqs = np.log10(freqs_hz)
                    self._image_artist = self.axes.imshow(
                        image,
                        aspect="auto",
                        extent=(
                            float(log_freqs[0]),
                            float(log_freqs[-1]),
                            float(angles_deg[0]),
                            float(angles_deg[-1]),
                        ),
                        interpolation="bilinear",
                        origin="lower",
                        zorder=1,
                    )
                    self._mesh_freqs_hz = freqs_hz.copy()
                    self._mesh_angles_deg = angles_deg.copy()
                    self._mesh_clip = clip
                    self._mesh_shading = shading
                    self._mesh_render_mode = render_mode
                    self._mesh_contour_step_db = contour_step_db
            else:
                self._x_axis_mode = "frequency"
                if (
                    self._mesh_matches(freqs_hz, angles_deg, clip, shading, contour_step_db)
                    and self._mesh_artist is not None
                ):
                    self._mesh_artist.set_array(clipped.ravel())
                else:
                    self._remove_mesh_artists()
                    self._mesh_artist = self.axes.pcolormesh(
                        freqs_hz,
                        angles_deg,
                        clipped,
                        cmap=cmap,
                        norm=norm,
                        shading=shading,
                    )
                    self._mesh_freqs_hz = freqs_hz.copy()
                    self._mesh_angles_deg = angles_deg.copy()
                    self._mesh_clip = clip
                    self._mesh_shading = shading
                    self._mesh_render_mode = render_mode
                    self._mesh_contour_step_db = contour_step_db
            self._mesh_values_db = clipped.copy()
        elif freqs_hz.size == 1:
            self._remove_mesh_artists()
            self._x_axis_mode = "frequency"
            self._mesh_freqs_hz = None
            self._mesh_angles_deg = None
            self._mesh_values_db = None
            self._mesh_clip = None
            self._mesh_shading = None
            self._mesh_render_mode = None
            self._mesh_contour_step_db = None
            self._remove_colorbar()
            x_values = np.full_like(angles_deg, float(freqs_hz[0]))
            if self._line_artist is None:
                (self._line_artist,) = self.axes.plot(
                    x_values,
                    angles_deg,
                    color="#1f77b4",
                    linewidth=1.5,
                )
            else:
                self._line_artist.set_data(x_values, angles_deg)
        else:
            self._remove_mesh_artists()
            self._remove_artist("_line_artist")
            self._x_axis_mode = "frequency"
            self._mesh_freqs_hz = None
            self._mesh_angles_deg = None
            self._mesh_values_db = None
            self._mesh_clip = None
            self._mesh_shading = None
            self._mesh_render_mode = None
            self._mesh_contour_step_db = None
            self._remove_colorbar()

        if self._axis_configuration_mode != self._x_axis_mode:
            self._configure_axes()
            self._redraw_captured_contours()
        self._apply_manual_axis_limits()
        if self._crosshair_visible:
            self._redraw_crosshair()
        self.draw_idle()


class RawCoordinatePlotCanvas(InteractivePlotCanvas):
    """Shared raw-coordinate crosshair overlay for Cartesian line plots."""

    def __init__(self, figure: Figure, title: str, *, secondary_axes=None) -> None:
        self.secondary_axes = secondary_axes
        self._crosshair_freq_hz: float | None = None
        self._crosshair_y_value: float | None = None
        self._crosshair_vline = None
        self._crosshair_hline = None
        self._crosshair_freq_label = None
        self._crosshair_y_label = None
        self._crosshair_secondary_label = None
        super().__init__(figure, title)

    def _interactive_axes(self) -> tuple[object, ...]:
        if self.secondary_axes is None:
            return (self.axes,)
        return (self.axes, self.secondary_axes)

    def _has_crosshair_data(self) -> bool:
        state = getattr(self, "_plot_state", None)
        if state is None:
            return False
        freqs_hz = state.freq_hz if isinstance(state, SpinoramaCurves) else state[0]
        return np.asarray(freqs_hz).size > 0

    def _set_crosshair_from_event(self, event) -> None:
        if event.xdata is None or event.ydata is None or event.inaxes is None:
            return
        display_x, display_y = event.inaxes.transData.transform((float(event.xdata), float(event.ydata)))
        freq_hz, y_value = self.axes.transData.inverted().transform((display_x, display_y))
        self._set_crosshair_position(float(freq_hz), float(y_value))

    def _clamp_crosshair_position(self, freq_hz: float, y_value: float) -> tuple[float, float]:
        x_min, x_max = sorted(float(value) for value in self.axes.get_xlim())
        y_min, y_max = sorted(float(value) for value in self.axes.get_ylim())
        return float(np.clip(freq_hz, x_min, x_max)), float(np.clip(y_value, y_min, y_max))

    def _set_crosshair_position(self, freq_hz: float, y_value: float) -> None:
        if not self._has_crosshair_data():
            return
        self._crosshair_freq_hz, self._crosshair_y_value = self._clamp_crosshair_position(freq_hz, y_value)
        self._crosshair_visible = True
        self._redraw_crosshair()
        if not self._draw_crosshair_overlay():
            self.draw_idle()

    def _crosshair_artists(self) -> tuple[object, ...]:
        return (
            self._crosshair_vline,
            self._crosshair_hline,
            self._crosshair_freq_label,
            self._crosshair_y_label,
            self._crosshair_secondary_label,
        )

    def _reset_crosshair_artists(self) -> None:
        self._crosshair_vline = None
        self._crosshair_hline = None
        self._crosshair_freq_label = None
        self._crosshair_y_label = None
        self._crosshair_secondary_label = None
        self._crosshair_background = None

    def _ensure_crosshair_artists(self) -> None:
        if self._crosshair_vline is None:
            self._crosshair_vline = self.axes.axvline(
                AUDIO_FREQ_MIN_HZ,
                color=ISOBAR_CROSSHAIR_COLOR,
                linewidth=0.9,
                alpha=0.9,
                zorder=9,
                visible=False,
            )
        if self._crosshair_hline is None:
            self._crosshair_hline = self.axes.axhline(
                0.0,
                color=ISOBAR_CROSSHAIR_COLOR,
                linewidth=0.9,
                alpha=0.9,
                zorder=9,
                visible=False,
            )
        if self._crosshair_freq_label is None:
            self._crosshair_freq_label = self.axes.text(
                AUDIO_FREQ_MIN_HZ,
                -0.075,
                "",
                transform=self.axes.get_xaxis_transform(),
                ha="center",
                va="top",
                color=ISOBAR_CROSSHAIR_LABEL_EDGE,
                fontsize=PLOT_TICK_SIZE,
                bbox={
                    "boxstyle": "square,pad=0.12",
                    "facecolor": ISOBAR_CROSSHAIR_LABEL_FACE,
                    "edgecolor": ISOBAR_CROSSHAIR_LABEL_EDGE,
                    "linewidth": 0.8,
                },
                clip_on=False,
                zorder=10,
                visible=False,
            )
        if self._crosshair_y_label is None:
            self._crosshair_y_label = self.axes.text(
                -0.01,
                0.0,
                "",
                transform=self.axes.get_yaxis_transform(),
                ha="right",
                va="center",
                color=ISOBAR_CROSSHAIR_LABEL_EDGE,
                fontsize=PLOT_TICK_SIZE,
                bbox={
                    "boxstyle": "square,pad=0.12",
                    "facecolor": ISOBAR_CROSSHAIR_LABEL_FACE,
                    "edgecolor": ISOBAR_CROSSHAIR_LABEL_EDGE,
                    "linewidth": 0.8,
                },
                clip_on=False,
                zorder=10,
                visible=False,
            )
        if self.secondary_axes is not None and self._crosshair_secondary_label is None:
            self._crosshair_secondary_label = self.secondary_axes.text(
                1.01,
                0.0,
                "",
                transform=self.secondary_axes.get_yaxis_transform(),
                ha="left",
                va="center",
                color=ISOBAR_CROSSHAIR_LABEL_EDGE,
                fontsize=PLOT_TICK_SIZE,
                bbox={
                    "boxstyle": "square,pad=0.12",
                    "facecolor": ISOBAR_CROSSHAIR_LABEL_FACE,
                    "edgecolor": ISOBAR_CROSSHAIR_LABEL_EDGE,
                    "linewidth": 0.8,
                },
                clip_on=False,
                zorder=10,
                visible=False,
            )
        for artist in self._crosshair_artists():
            if artist is not None:
                artist.set_animated(True)

    def _redraw_crosshair(self) -> None:
        if not self._crosshair_visible or self._crosshair_freq_hz is None or self._crosshair_y_value is None:
            self._set_crosshair_artist_visibility(False)
            return
        if not self._has_crosshair_data():
            self._set_crosshair_artist_visibility(False)
            return
        freq_hz, y_value = self._clamp_crosshair_position(
            self._crosshair_freq_hz,
            self._crosshair_y_value,
        )
        self._ensure_crosshair_artists()
        self._crosshair_vline.set_xdata([freq_hz, freq_hz])
        self._crosshair_hline.set_ydata([y_value, y_value])
        self._crosshair_freq_label.set_position((freq_hz, -0.075))
        self._crosshair_freq_label.set_text(_format_isobar_crosshair_frequency(freq_hz))
        self._crosshair_y_label.set_position((-0.01, y_value))
        self._crosshair_y_label.set_text(self._format_crosshair_y(y_value))
        if self._crosshair_secondary_label is not None:
            secondary_y = self._secondary_y_value(freq_hz, y_value)
            self._crosshair_secondary_label.set_position((1.01, secondary_y))
            self._crosshair_secondary_label.set_text(self._format_secondary_crosshair_y(secondary_y))
        self._set_crosshair_artist_visibility(True)

    def _secondary_y_value(self, freq_hz: float, y_value: float) -> float:
        if self.secondary_axes is None:
            return y_value
        display_point = self.axes.transData.transform((freq_hz, y_value))
        return float(self.secondary_axes.transData.inverted().transform(display_point)[1])

    def _format_crosshair_y(self, value: float) -> str:
        return f"{value:.3g}"

    def _format_secondary_crosshair_y(self, value: float) -> str:
        return f"{value:.1f}"


class ImpedanceCanvas(RawCoordinatePlotCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(6.5, 3.0), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self._lines: list[object] = []
        self._series_labels: tuple[str, ...] = ()
        self._series_visibility: dict[str, bool] = {}
        self._series_actions: dict[str, QAction] = {}
        self._plot_state = None
        super().__init__(self.figure, "Acoustic Impedance")
        self.trace_filter_menu = QMenu("Traces", self)
        self.trace_filter_action = QAction("Traces", self)
        self.trace_filter_action.setToolTip("Choose visible acoustic load impedance traces")
        self.trace_filter_action.setMenu(self.trace_filter_menu)
        self.set_layout_profile(SINGLE_AXIS_LAYOUT)
        self._draw_empty()

    def _configure_axes(self) -> None:
        self.axes.set_title(self.title, pad=PLOT_TITLE_PAD)
        self.axes.set_xlabel("Frequency (Hz)")
        self.axes.set_ylabel("Normalized Acoustic Impedance (Z / ρcSd)")
        apply_audio_frequency_axis(self.axes)
        self.axes.grid(which="major", color="#808080", linewidth=0.8, alpha=GRID_LINE_ALPHA)
        apply_compact_plot_text(self.axes)

    def _draw_empty(self) -> None:
        clear_plot_axes(self.axes)
        self._lines = []
        self._series_labels = ()
        self._plot_state = None
        self._reset_crosshair_artists()
        self._reset_comparison_interaction()
        self._configure_axes()
        self._sync_trace_filter_actions(())
        self._apply_manual_axis_limits()
        self._redraw_crosshair()
        self.draw_idle()

    def set_comparison_plot(
        self,
        freqs_hz: np.ndarray,
        radiator_names: np.ndarray,
        impedance_real: np.ndarray,
        impedance_imag: np.ndarray,
    ) -> None:
        self._set_comparison_plot_state(
            self._normalized_plot_state(freqs_hz, radiator_names, impedance_real, impedance_imag)
        )

    def _normalized_plot_state(
        self,
        freqs_hz: np.ndarray,
        radiator_names: np.ndarray,
        impedance_real: np.ndarray,
        impedance_imag: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        real = np.asarray(impedance_real, dtype=np.float32)
        imaginary = np.asarray(impedance_imag, dtype=np.float32)
        if real.ndim == 1:
            real = real[np.newaxis, :]
            imaginary = imaginary[np.newaxis, :]
        return (
            np.asarray(freqs_hz, dtype=np.float32).copy(),
            np.asarray(radiator_names).copy(),
            real.copy(),
            imaginary.copy(),
        )

    def _current_plot_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        if self._plot_state is None:
            return None
        return tuple(values.copy() for values in self._plot_state)

    def _apply_plot_state(self, state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
        self.update_plot(*state)

    def update_plot(
        self,
        freqs_hz: np.ndarray,
        radiator_names: np.ndarray,
        impedance_real: np.ndarray,
        impedance_imag: np.ndarray,
    ) -> None:
        state = self._normalized_plot_state(freqs_hz, radiator_names, impedance_real, impedance_imag)
        if self._defer_current_plot_state(state):
            return
        freqs_hz, radiator_names, impedance_real, impedance_imag = state
        self._plot_state = state
        self._invalidate_crosshair_background()

        series_labels = tuple(
            str(radiator_names[index]) if index < radiator_names.size else f"Radiator {index + 1}"
            for index in range(impedance_real.shape[0])
        )
        if series_labels != self._series_labels:
            for line in self._lines:
                line.remove()
            self._lines = []
            for name in series_labels:
                (real_line,) = self.axes.plot([], [], linewidth=1.5, label=f"{name} Z real")
                (imag_line,) = self.axes.plot([], [], linewidth=1.5, linestyle="--", label=f"{name} Z imag")
                self._lines.extend((real_line, imag_line))
            self._series_labels = series_labels
            for label in series_labels:
                self._series_visibility.setdefault(label, True)
            self._sync_trace_filter_actions(series_labels)

        for index in range(impedance_real.shape[0]):
            self._lines[index * 2].set_data(freqs_hz, impedance_real[index])
            self._lines[index * 2 + 1].set_data(freqs_hz, impedance_imag[index])
        self._apply_series_visibility()
        self._update_y_limits()
        self._apply_manual_axis_limits()
        apply_compact_plot_text(self.axes)
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
        self._update_y_limits()
        self._apply_manual_axis_limits()
        self._invalidate_crosshair_background()
        self.draw_idle()

    def _apply_series_visibility(self) -> None:
        visible_lines = []
        for index, label in enumerate(self._series_labels):
            visible = self._series_visibility.get(label, True)
            pair = self._lines[index * 2 : index * 2 + 2]
            for line in pair:
                line.set_visible(visible)
                if visible:
                    visible_lines.append(line)
        legend = self.axes.get_legend()
        if legend is not None:
            legend.remove()
        if visible_lines:
            self.axes.legend(
                visible_lines,
                [line.get_label() for line in visible_lines],
                loc="best",
            )

    def _update_y_limits(self) -> None:
        finite_rows = []
        for line in self._lines:
            if not line.get_visible():
                continue
            values = np.asarray(line.get_ydata(), dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size:
                finite_rows.append(finite)
        if not finite_rows:
            self.axes.set_ylim(-0.1, 0.1)
            return
        values = np.concatenate(finite_rows)
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        span = maximum - minimum
        padding = max(0.08 * span, 1.0e-3)
        self.axes.set_ylim(minimum - padding, maximum + padding)


class OnAxisResponseCanvas(RawCoordinatePlotCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(6.5, 3.0), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.phase_axes = self.axes.twinx()
        self._lines: list[object] = []
        self._magnitude_lines: dict[str, object] = {}
        self._phase_lines: dict[str, object] = {}
        self._series_labels: tuple[str, ...] = ()
        self._series_visibility: dict[str, bool] = {}
        self._channel_colors: dict[str, Any] = {}
        self._series_actions: dict[str, QAction] = {}
        self._phase_available = False
        self._plot_state = None
        super().__init__(self.figure, "On-Axis Frequency Response", secondary_axes=self.phase_axes)
        self.trace_filter_menu = QMenu("Traces", self)
        self.trace_filter_action = QAction("Traces", self)
        self.trace_filter_action.setToolTip("Choose visible on-axis response traces")
        self.trace_filter_action.setMenu(self.trace_filter_menu)
        self.show_phase_action = QAction("Phase", self)
        self.show_phase_action.setToolTip("Show phase traces")
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
        self.axes.set_ylabel("SPL (dB)")
        self.phase_axes.set_ylabel("Phase (deg)")
        self.phase_axes.yaxis.set_label_position("right")
        self.phase_axes.yaxis.tick_right()
        apply_audio_frequency_axis(self.axes)
        self.phase_axes.set_ylim(-180.0, 600.0)
        self.phase_axes.set_yticks(np.arange(-180.0, 601.0, 90.0))
        self.axes.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2.0, 10.0)))
        self.axes.yaxis.set_major_locator(MultipleLocator(10.0))
        self.axes.yaxis.set_minor_locator(MultipleLocator(5.0))
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
        self._lines = []
        self._magnitude_lines = {}
        self._phase_lines = {}
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
        return f"{value:.1f} dB"

    def _format_secondary_crosshair_y(self, value: float) -> str:
        return f"{value:.1f} deg"

    def set_comparison_plot(
        self,
        freqs_hz: np.ndarray,
        angles_deg: np.ndarray,
        horizontal_spl_db: np.ndarray,
        channel_names: np.ndarray | None = None,
        channel_on_axis_spl_db: np.ndarray | None = None,
        on_axis_phase_deg: np.ndarray | None = None,
        channel_on_axis_phase_deg: np.ndarray | None = None,
    ) -> None:
        self._set_comparison_plot_state(
            self._normalized_plot_state(
                freqs_hz,
                angles_deg,
                horizontal_spl_db,
                channel_names,
                channel_on_axis_spl_db,
                on_axis_phase_deg,
                channel_on_axis_phase_deg,
            )
        )

    def _normalized_plot_state(
        self,
        freqs_hz: np.ndarray,
        angles_deg: np.ndarray,
        horizontal_spl_db: np.ndarray,
        channel_names: np.ndarray | None,
        channel_on_axis_spl_db: np.ndarray | None,
        on_axis_phase_deg: np.ndarray | None,
        channel_on_axis_phase_deg: np.ndarray | None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        return (
            np.asarray(freqs_hz, dtype=np.float32).copy(),
            np.asarray(angles_deg, dtype=np.float32).copy(),
            np.asarray(horizontal_spl_db, dtype=np.float32).copy(),
            None if channel_names is None else np.asarray(channel_names).copy(),
            None if channel_on_axis_spl_db is None else np.asarray(channel_on_axis_spl_db, dtype=np.float32).copy(),
            None if on_axis_phase_deg is None else np.asarray(on_axis_phase_deg, dtype=np.float32).copy(),
            (
                None
                if channel_on_axis_phase_deg is None
                else np.asarray(channel_on_axis_phase_deg, dtype=np.float32).copy()
            ),
        )

    def _current_plot_state(
        self,
    ) -> (
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray | None,
            np.ndarray | None,
            np.ndarray | None,
            np.ndarray | None,
        ]
        | None
    ):
        if self._plot_state is None:
            return None
        return tuple(None if values is None else values.copy() for values in self._plot_state)

    def _apply_plot_state(
        self,
        state: tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray | None,
            np.ndarray | None,
            np.ndarray | None,
            np.ndarray | None,
        ],
    ) -> None:
        self.update_plot(*state)

    def update_plot(
        self,
        freqs_hz: np.ndarray,
        angles_deg: np.ndarray,
        horizontal_spl_db: np.ndarray,
        channel_names: np.ndarray | None = None,
        channel_on_axis_spl_db: np.ndarray | None = None,
        on_axis_phase_deg: np.ndarray | None = None,
        channel_on_axis_phase_deg: np.ndarray | None = None,
    ) -> None:
        state = self._normalized_plot_state(
            freqs_hz,
            angles_deg,
            horizontal_spl_db,
            channel_names,
            channel_on_axis_spl_db,
            on_axis_phase_deg,
            channel_on_axis_phase_deg,
        )
        if self._defer_current_plot_state(state):
            return
        (
            freqs_hz,
            angles_deg,
            horizontal_spl_db,
            channel_names,
            channel_on_axis_spl_db,
            on_axis_phase_deg,
            channel_on_axis_phase_deg,
        ) = state
        self._plot_state = state
        self._invalidate_crosshair_background()
        if freqs_hz.size:
            on_axis = np.asarray(
                [
                    np.interp(
                        0.0,
                        angles_deg.astype(float),
                        row.astype(float),
                    )
                    for row in horizontal_spl_db
                ],
                dtype=float,
            )
            series: list[tuple[str, np.ndarray]] = [("Sum", on_axis)]
            phase_series: dict[str, np.ndarray] = {}
            if on_axis_phase_deg is not None and np.asarray(on_axis_phase_deg).shape == freqs_hz.shape:
                phase_series["Sum"] = np.asarray(on_axis_phase_deg, dtype=float)
            if channel_names is not None and channel_on_axis_spl_db is not None:
                names = np.asarray(channel_names)
                channel_curves = np.asarray(channel_on_axis_spl_db, dtype=float)
                for index, name in enumerate(names):
                    if index >= channel_curves.shape[0]:
                        continue
                    label = str(name)
                    series.append((label, channel_curves[index]))
                    if channel_on_axis_phase_deg is not None and index < np.asarray(channel_on_axis_phase_deg).shape[0]:
                        phase = np.asarray(channel_on_axis_phase_deg[index], dtype=float)
                        if phase.shape == freqs_hz.shape:
                            phase_series[label] = phase
            labels = tuple(name for name, _values in series)
            if labels != self._series_labels:
                for line in (*self._magnitude_lines.values(), *self._phase_lines.values()):
                    line.remove()
                self._magnitude_lines = {}
                self._phase_lines = {}
                for index, label in enumerate(labels):
                    color = self._series_color(label, index)
                    self._magnitude_lines[label] = self.axes.plot(
                        [],
                        [],
                        linewidth=2.0 if label == "Sum" else 1.3,
                        linestyle="-",
                        color=color,
                        label=label,
                    )[0]
                    self._phase_lines[label] = self.phase_axes.plot(
                        [],
                        [],
                        linewidth=1.2,
                        linestyle=":",
                        color=color,
                    )[0]
                    self._series_visibility.setdefault(label, True)
                self._lines = list(self._magnitude_lines.values())
                self._series_labels = labels
                self._sync_trace_filter_actions(labels)

            for label, values in series:
                self._magnitude_lines[label].set_data(freqs_hz, values)
                if label in phase_series:
                    phase_freqs, wrapped_phase = _phase_curve_with_wrap_breaks(freqs_hz, phase_series[label])
                    self._phase_lines[label].set_data(phase_freqs, wrapped_phase)
                else:
                    self._phase_lines[label].set_data([], [])

            self._phase_available = bool(phase_series)
            self.show_phase_action.setEnabled(self._phase_available)
            self._apply_series_visibility()
            self._update_magnitude_limits()

        self._apply_manual_axis_limits()
        apply_compact_plot_text(self.axes)
        apply_compact_plot_text(self.phase_axes)
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
        self._refresh_magnitude_legend()

    def _refresh_magnitude_legend(self) -> None:
        legend = self.axes.get_legend()
        if legend is not None:
            legend.remove()
        visible_lines = [
            self._magnitude_lines[label] for label in self._series_labels if self._series_visibility.get(label, True)
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
        if finite_rows:
            ymax = float(np.ceil(np.max(np.concatenate(finite_rows)) / 5.0) * 5.0)
            self.axes.set_ylim(ymax - ON_AXIS_DB_SPAN, ymax)


def _phase_curve_with_wrap_breaks(
    freqs_hz: np.ndarray,
    phase_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.asarray(freqs_hz, dtype=float)
    phase = ((np.asarray(phase_deg, dtype=float) + 180.0) % 360.0) - 180.0
    if freqs.shape != phase.shape:
        raise ValueError("Phase data must match the on-axis frequency array.")
    if freqs.size < 2:
        return freqs.copy(), phase.copy()
    wrapped_freqs: list[float] = []
    wrapped_phase: list[float] = []
    for index, (frequency, value) in enumerate(zip(freqs, phase, strict=True)):
        wrapped_freqs.append(float(frequency))
        wrapped_phase.append(float(value))
        if index + 1 >= phase.size:
            continue
        next_value = phase[index + 1]
        if np.isfinite(value) and np.isfinite(next_value) and abs(float(next_value - value)) > 180.0:
            wrapped_freqs.append(float("nan"))
            wrapped_phase.append(float("nan"))
    return np.asarray(wrapped_freqs), np.asarray(wrapped_phase)


def _snapshot_spinorama_curves(curves: SpinoramaCurves) -> SpinoramaCurves:
    return SpinoramaCurves(
        freq_hz=np.asarray(curves.freq_hz).copy(),
        on_axis_db=np.asarray(curves.on_axis_db).copy(),
        listening_window_db=np.asarray(curves.listening_window_db).copy(),
        early_reflections_db=np.asarray(curves.early_reflections_db).copy(),
        sound_power_db=np.asarray(curves.sound_power_db).copy(),
        estimated_in_room_db=np.asarray(curves.estimated_in_room_db).copy(),
        early_reflections_di_db=np.asarray(curves.early_reflections_di_db).copy(),
        sound_power_di_db=np.asarray(curves.sound_power_di_db).copy(),
        sound_power_di_label=curves.sound_power_di_label,
    )


class SpinoramaCanvas(RawCoordinatePlotCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(6.5, 3.9), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.di_axes = self.axes.twinx()
        self._spl_lines: dict[str, object] = {}
        self._di_lines: dict[str, object] = {}
        self._series_labels: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
        self._plot_state = None
        super().__init__(self.figure, "Spinorama", secondary_axes=self.di_axes)
        self.set_layout_profile(SPINORAMA_LAYOUT)
        self._draw_empty()

    def _apply_layout(self) -> None:
        self._apply_layout_profile()
        self.di_axes.yaxis.set_label_position("right")
        self.di_axes.yaxis.tick_right()

    def _draw_empty(self) -> None:
        clear_plot_axes(self.axes)
        clear_plot_axes(self.di_axes)
        self._spl_lines = {}
        self._di_lines = {}
        self._series_labels = ((), ())
        self._plot_state = None
        self._reset_crosshair_artists()
        self._reset_comparison_interaction()
        self._apply_layout()
        self.axes.set_title(self.title, pad=PLOT_TITLE_PAD)
        self.axes.set_xlabel("Frequency (Hz)")
        self.axes.set_ylabel("SPL (dB)")
        self.di_axes.set_ylabel("DI (dB)")
        apply_audio_frequency_axis(self.axes)
        self.axes.set_ylim(*SPINORAMA_SPL_LIMITS)
        self.di_axes.set_ylim(*SPINORAMA_DI_LIMITS)
        self.axes.grid(which="major", color="#808080", linewidth=0.8, alpha=GRID_LINE_ALPHA)
        apply_compact_plot_text(self.axes)
        apply_compact_plot_text(self.di_axes)
        self._apply_manual_axis_limits()
        self._redraw_crosshair()
        self.draw_idle()

    def _format_crosshair_y(self, value: float) -> str:
        return f"{value:.1f} dB SPL"

    def _format_secondary_crosshair_y(self, value: float) -> str:
        return f"{value:.1f} dB DI"

    def set_comparison_plot(
        self,
        freqs_hz: np.ndarray,
        angles_deg: np.ndarray,
        horizontal_spl_db: np.ndarray,
        vertical_spl_db: np.ndarray,
        *,
        horizontal_reference_angle_deg: float = 0.0,
        vertical_reference_angle_deg: float = 0.0,
    ) -> None:
        curves = compute_spinorama_from_planes(
            freq_hz=freqs_hz,
            polar_angle_deg=angles_deg,
            horizontal_spl_db=horizontal_spl_db,
            vertical_spl_db=vertical_spl_db,
            horizontal_reference_angle_deg=horizontal_reference_angle_deg,
            vertical_reference_angle_deg=vertical_reference_angle_deg,
        )
        self._set_comparison_plot_state(_snapshot_spinorama_curves(curves))

    def set_comparison_curves(self, curves: SpinoramaCurves) -> None:
        self._set_comparison_plot_state(_snapshot_spinorama_curves(curves))

    def _current_plot_state(self) -> SpinoramaCurves | None:
        return None if self._plot_state is None else _snapshot_spinorama_curves(self._plot_state)

    def _apply_plot_state(self, state: SpinoramaCurves) -> None:
        self.update_curves(state)

    def update_plot(
        self,
        freqs_hz: np.ndarray,
        angles_deg: np.ndarray,
        horizontal_spl_db: np.ndarray,
        vertical_spl_db: np.ndarray,
        *,
        horizontal_reference_angle_deg: float = 0.0,
        vertical_reference_angle_deg: float = 0.0,
    ) -> None:
        curves = compute_spinorama_from_planes(
            freq_hz=freqs_hz,
            polar_angle_deg=angles_deg,
            horizontal_spl_db=horizontal_spl_db,
            vertical_spl_db=vertical_spl_db,
            horizontal_reference_angle_deg=horizontal_reference_angle_deg,
            vertical_reference_angle_deg=vertical_reference_angle_deg,
        )
        self.update_curves(curves)

    def update_curves(self, curves: SpinoramaCurves) -> None:
        state = _snapshot_spinorama_curves(curves)
        if self._defer_current_plot_state(state):
            return
        curves = state
        self._plot_state = state
        self._invalidate_crosshair_background()
        colors = {
            "On Axis": "#1f77b4",
            "Listen. Wind.": "#2ca02c",
            "Early Reflections": "#ff7f0e",
            "Sound Power": "#d62728",
            "PIR": "#9467bd",
            "ERDI": "#8c564b",
            "SPDI": "#17becf",
        }
        spl_curves = curves.spl_curves()
        di_curves = curves.di_curves()
        labels = (tuple(name for name, _values in spl_curves), tuple(name for name, _values in di_curves))
        if labels != self._series_labels:
            for line in (*self._spl_lines.values(), *self._di_lines.values()):
                line.remove()
            self._spl_lines = {
                name: self.axes.plot([], [], linewidth=1.5, label=name, color=colors.get(name))[0]
                for name, _values in spl_curves
            }
            self._di_lines = {
                name: self.di_axes.plot(
                    [],
                    [],
                    linewidth=1.2,
                    linestyle="--",
                    label=name,
                    color=colors.get(name),
                )[0]
                for name, _values in di_curves
            }
            legend = self.axes.get_legend()
            if legend is not None:
                legend.remove()
            lines = [*self._spl_lines.values(), *self._di_lines.values()]
            self.axes.legend(
                lines,
                [*labels[0], *labels[1]],
                loc="lower center",
                bbox_to_anchor=(0.5, SPINORAMA_LEGEND_BOTTOM),
                bbox_transform=self.figure.transFigure,
                ncols=4,
                borderaxespad=0.0,
                frameon=False,
                # Two rows in a fixed-fraction margin; tight spacing suits
                # short panels.
                borderpad=0.0,
                labelspacing=0.25,
                handlelength=1.6,
                handletextpad=0.5,
            )
            self._series_labels = labels

        for name, values in spl_curves:
            self._spl_lines[name].set_data(curves.freq_hz, values)
        for name, values in di_curves:
            self._di_lines[name].set_data(curves.freq_hz, values)

        self.axes.set_ylim(*SPINORAMA_SPL_LIMITS)
        self.di_axes.set_ylim(*SPINORAMA_DI_LIMITS)
        self._apply_manual_axis_limits()

        apply_compact_plot_text(self.axes)
        apply_compact_plot_text(self.di_axes)
        if self._crosshair_visible:
            self._redraw_crosshair()
        self.draw_idle()
