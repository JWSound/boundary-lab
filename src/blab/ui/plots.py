"""Matplotlib canvases used by the live solver GUI."""

from __future__ import annotations

import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure

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
PLOT_TITLE_PAD = 1
GRID_LINE_ALPHA = 0.6
ON_AXIS_DB_SPAN = 50.0
SPINORAMA_SPL_LIMITS = (-40.0, 10.0)
SPINORAMA_DI_LIMITS = (-5.0, 45.0)
ISOBAR_CROSSHAIR_COLOR = "#101214"
ISOBAR_CROSSHAIR_LABEL_FACE = "#f8fbff"
ISOBAR_CROSSHAIR_LABEL_EDGE = "#2868ff"
ISOBAR_CROSSHAIR_DB_FACE = "#101214"


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


class IsobarCanvas(FigureCanvas):
    def __init__(
        self,
        title: str,
        *,
        left_margin: float = 0.14,
        right_margin: float = 0.88,
        show_colorbar: bool = True,
    ):
        self.figure = Figure(figsize=(5.5, 2.8), dpi=100)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.title = title
        self.left_margin = float(left_margin)
        self.right_margin = float(right_margin)
        self.show_colorbar = bool(show_colorbar)
        self.colors = VisualizerConfig.custom_colors
        self._mesh_artist = None
        self._image_artist = None
        self._line_artist = None
        self._contour_artist = None
        self._colorbar = None
        self._colorbar_axes = None
        self._colorbar_mappable = None
        self._mesh_freqs_hz: np.ndarray | None = None
        self._mesh_angles_deg: np.ndarray | None = None
        self._mesh_values_db: np.ndarray | None = None
        self._mesh_clip: tuple[float, float] | None = None
        self._mesh_shading: str | None = None
        self._mesh_render_mode: str | None = None
        self._mesh_contour_step_db: float | None = None
        self._x_axis_mode = "frequency"
        self._captured_contours: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self._comparison_plot: tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            float,
            float,
            str,
            float,
        ] | None = None
        self._comparison_restore_plot: tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            float,
            float,
            str,
            float,
        ] | None = None
        self._comparison_active = False
        self._crosshair_visible = False
        self._crosshair_dragging = False
        self._crosshair_freq_hz: float | None = None
        self._crosshair_angle_deg: float | None = None
        self._crosshair_vline = None
        self._crosshair_hline = None
        self._crosshair_freq_label = None
        self._crosshair_angle_label = None
        self._crosshair_db_label = None
        self._crosshair_background = None
        self._apply_layout()
        self._connect_crosshair_events()
        self._connect_comparison_events()
        self._draw_empty()

    def _apply_layout(self) -> None:
        self.figure.subplots_adjust(left=self.left_margin, right=self.right_margin, top=0.91, bottom=0.2)

    def _configure_axes(self) -> None:
        self._apply_layout()
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
        self._comparison_active = False
        self._comparison_restore_plot = None
        self._remove_colorbar()
        self._mesh_freqs_hz = None
        self._mesh_angles_deg = None
        self._mesh_values_db = None
        self._mesh_clip = None
        self._mesh_shading = None
        self._mesh_render_mode = None
        self._x_axis_mode = "frequency"
        self._configure_axes()
        self._redraw_captured_contours()
        self._redraw_crosshair()
        self.draw_idle()

    def _connect_crosshair_events(self) -> None:
        self.mpl_connect("button_press_event", self._on_crosshair_button_press)
        self.mpl_connect("button_release_event", self._on_crosshair_button_release)
        self.mpl_connect("motion_notify_event", self._on_crosshair_motion)
        self.mpl_connect("draw_event", self._on_crosshair_draw)

    def _connect_comparison_events(self) -> None:
        self.mpl_connect("button_press_event", self._on_comparison_button_press)
        self.mpl_connect("button_release_event", self._on_comparison_button_release)

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
        self._comparison_plot = (
            np.asarray(freqs_hz, dtype=np.float32).copy(),
            np.asarray(angles_deg, dtype=np.float32).copy(),
            np.asarray(values_db, dtype=np.float32).copy(),
            float(clip_min_db),
            float(clip_max_db),
            str(shading or FINAL_ISOBAR_SHADING),
            max(0.0, float(contour_step_db)),
        )
        self.setToolTip("Hold the right mouse button over the plot to view the previous completed solve.")

    def clear_comparison_plot(self) -> None:
        self._comparison_plot = None
        self._comparison_restore_plot = None
        self._comparison_active = False
        self.setToolTip("")

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

    def _on_comparison_button_press(self, event) -> None:
        if event.inaxes is not self.axes or event.button != MouseButton.RIGHT:
            return
        if self._comparison_active or self._comparison_plot is None:
            return
        current_plot = self._current_plot_state()
        if current_plot is None:
            return
        self._comparison_restore_plot = current_plot
        self._comparison_active = True
        self._apply_plot_state(self._comparison_plot)
        self.axes.set_title(f"{self.title} - Previous Solve", pad=PLOT_TITLE_PAD)
        self.draw_idle()

    def _on_comparison_button_release(self, event) -> None:
        if event.button != MouseButton.RIGHT or not self._comparison_active:
            return
        restore_plot = self._comparison_restore_plot
        self._comparison_restore_plot = None
        self._comparison_active = False
        if restore_plot is not None:
            self._apply_plot_state(restore_plot)

    def _on_crosshair_draw(self, event) -> None:
        if event.canvas is not self:
            return
        try:
            self._crosshair_background = self.copy_from_bbox(self.figure.bbox)
        except Exception:
            self._crosshair_background = None
        if self._crosshair_visible:
            self._draw_crosshair_overlay()

    def _crosshair_artists(self) -> tuple[object, ...]:
        return (
            self._crosshair_vline,
            self._crosshair_hline,
            self._crosshair_freq_label,
            self._crosshair_angle_label,
            self._crosshair_db_label,
        )

    def _invalidate_crosshair_background(self) -> None:
        self._crosshair_background = None

    def _draw_crosshair_overlay(self) -> bool:
        if self._crosshair_background is None:
            return False
        try:
            self.restore_region(self._crosshair_background)
            for artist in self._crosshair_artists():
                if artist is not None and artist.get_visible():
                    self.axes.draw_artist(artist)
            self.blit(self.figure.bbox)
            self.flush_events()
            return True
        except Exception:
            self._crosshair_background = None
            return False

    def _has_crosshair_data(self) -> bool:
        return (
            self._mesh_freqs_hz is not None
            and self._mesh_angles_deg is not None
            and self._mesh_values_db is not None
            and self._mesh_freqs_hz.size > 0
            and self._mesh_angles_deg.size > 0
        )

    def _on_crosshair_button_press(self, event) -> None:
        if event.inaxes is not self.axes or event.button != MouseButton.LEFT:
            return
        if getattr(event, "dblclick", False):
            self._hide_crosshair()
            return
        if not self._has_crosshair_data():
            return
        self._crosshair_dragging = True
        self._set_crosshair_from_event(event)

    def _on_crosshair_button_release(self, event) -> None:
        del event
        self._crosshair_dragging = False

    def _on_crosshair_motion(self, event) -> None:
        if not self._crosshair_dragging or event.inaxes is not self.axes:
            return
        self._set_crosshair_from_event(event)

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

    def _hide_crosshair(self) -> None:
        self._crosshair_visible = False
        self._crosshair_dragging = False
        self._set_crosshair_artist_visibility(False)
        if not self._draw_crosshair_overlay():
            self.draw_idle()

    def _set_crosshair_artist_visibility(self, visible: bool) -> None:
        for artist in (
            self._crosshair_vline,
            self._crosshair_hline,
            self._crosshair_freq_label,
            self._crosshair_angle_label,
            self._crosshair_db_label,
        ):
            if artist is not None:
                artist.set_visible(visible)

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
        return LinearSegmentedColormap.from_list("boundary_lab_isobar", list(self.colors), N=256)

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
            self._remove_colorbar()
            return
        self._remove_colorbar()
        self._colorbar_mappable = ScalarMappable(norm=norm, cmap=cmap)
        self._colorbar_mappable.set_array([])
        self._colorbar_axes = self.figure.add_axes([self.right_margin + 0.025, 0.2, 0.025, 0.71])
        self._colorbar = self.figure.colorbar(
            self._colorbar_mappable,
            cax=self._colorbar_axes,
        )
        self._colorbar.set_ticks(self._colorbar_ticks(clip_min_db, clip_max_db, contour_step_db))
        self._colorbar.ax.tick_params(labelsize=PLOT_TICK_SIZE)

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

        if self._colorbar is None:
            self._configure_axes()
        else:
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
        self._redraw_captured_contours()
        if self._crosshair_visible:
            self._redraw_crosshair()
        self.draw_idle()


class ImpedanceCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(6.5, 3.0), dpi=100, tight_layout=True)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self._draw_empty()

    def _draw_empty(self) -> None:
        clear_plot_axes(self.axes)
        self.axes.set_title("Acoustic Impedance", pad=PLOT_TITLE_PAD)
        self.axes.set_xlabel("Frequency (Hz)")
        self.axes.set_ylabel("Acoustic Impedance (Pa*s/m^3)")
        apply_audio_frequency_axis(self.axes)
        self.axes.grid(which="major", color="#808080", linewidth=0.8, alpha=GRID_LINE_ALPHA)
        apply_compact_plot_text(self.axes)
        self.draw_idle()

    def update_plot(
        self,
        freqs_hz: np.ndarray,
        radiator_names: np.ndarray,
        impedance_real: np.ndarray,
        impedance_imag: np.ndarray,
    ) -> None:
        clear_plot_axes(self.axes)
        self.axes.set_title("Acoustic Impedance", pad=PLOT_TITLE_PAD)
        self.axes.set_xlabel("Frequency (Hz)")
        self.axes.set_ylabel("Acoustic Impedance (Pa*s/m^3)")
        apply_audio_frequency_axis(self.axes)
        self.axes.grid(which="major", color="#808080", linewidth=0.8, alpha=GRID_LINE_ALPHA)

        if impedance_real.ndim == 1:
            impedance_real = impedance_real[np.newaxis, :]
            impedance_imag = impedance_imag[np.newaxis, :]

        for index in range(impedance_real.shape[0]):
            name = str(radiator_names[index]) if index < radiator_names.size else f"Radiator {index + 1}"
            self.axes.plot(freqs_hz, impedance_real[index], linewidth=1.5, label=f"{name} Z real")
            self.axes.plot(freqs_hz, impedance_imag[index], linewidth=1.5, linestyle="--", label=f"{name} Z imag")

        self.axes.legend(loc="best")
        apply_compact_plot_text(self.axes)
        self.draw_idle()


class OnAxisResponseCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(6.5, 3.0), dpi=100, tight_layout=True)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self._draw_empty()

    def _draw_empty(self) -> None:
        clear_plot_axes(self.axes)
        self.axes.set_title("On-Axis Frequency Response", pad=PLOT_TITLE_PAD)
        self.axes.set_xlabel("Frequency (Hz)")
        self.axes.set_ylabel("SPL (dB)")
        apply_audio_frequency_axis(self.axes)
        self.axes.grid(which="major", color="#808080", linewidth=0.8, alpha=GRID_LINE_ALPHA)
        apply_compact_plot_text(self.axes)
        self.draw_idle()

    def update_plot(
        self,
        freqs_hz: np.ndarray,
        angles_deg: np.ndarray,
        horizontal_spl_db: np.ndarray,
        channel_names: np.ndarray | None = None,
        channel_on_axis_spl_db: np.ndarray | None = None,
    ) -> None:
        clear_plot_axes(self.axes)
        self.axes.set_title("On-Axis Frequency Response", pad=PLOT_TITLE_PAD)
        self.axes.set_xlabel("Frequency (Hz)")
        self.axes.set_ylabel("SPL (dB)")
        apply_audio_frequency_axis(self.axes)
        self.axes.grid(which="major", color="#808080", linewidth=0.8, alpha=GRID_LINE_ALPHA)

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
            self.axes.plot(freqs_hz, on_axis, linewidth=2.0, color="#1f77b4", label="Sum")
            if channel_names is not None and channel_on_axis_spl_db is not None:
                names = np.asarray(channel_names)
                channel_curves = np.asarray(channel_on_axis_spl_db, dtype=float)
                for index, name in enumerate(names):
                    if index >= channel_curves.shape[0]:
                        continue
                    self.axes.plot(
                        freqs_hz,
                        channel_curves[index],
                        linewidth=1.1,
                        linestyle="--",
                        alpha=0.85,
                        label=str(name),
                    )
            finite = on_axis[np.isfinite(on_axis)]
            if channel_on_axis_spl_db is not None:
                channel_finite = np.asarray(channel_on_axis_spl_db, dtype=float)
                channel_finite = channel_finite[np.isfinite(channel_finite)]
                if channel_finite.size:
                    finite = np.concatenate([finite, channel_finite])
            if finite.size:
                ymax = float(np.ceil(np.max(finite) / 5.0) * 5.0)
                self.axes.set_ylim(ymax - ON_AXIS_DB_SPAN, ymax)
            self.axes.legend(loc="best")

        apply_compact_plot_text(self.axes)
        self.draw_idle()


class SpinoramaCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(6.5, 3.9), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.di_axes = self.axes.twinx()
        super().__init__(self.figure)
        self._apply_layout()
        self._draw_empty()

    def _apply_layout(self) -> None:
        self.figure.subplots_adjust(left=0.14, right=0.9, top=0.91, bottom=0.28)
        self.di_axes.yaxis.set_label_position("right")
        self.di_axes.yaxis.tick_right()

    def _draw_empty(self) -> None:
        clear_plot_axes(self.axes)
        clear_plot_axes(self.di_axes)
        self._apply_layout()
        # self.axes.set_title("Spinorama", pad=PLOT_TITLE_PAD)
        self.axes.set_xlabel("Frequency (Hz)")
        self.axes.set_ylabel("SPL (dB)")
        self.di_axes.set_ylabel("DI (dB)")
        apply_audio_frequency_axis(self.axes)
        self.axes.grid(which="major", color="#808080", linewidth=0.8, alpha=GRID_LINE_ALPHA)
        apply_compact_plot_text(self.axes)
        apply_compact_plot_text(self.di_axes)
        self.draw_idle()

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
        clear_plot_axes(self.axes)
        clear_plot_axes(self.di_axes)
        self._apply_layout()
        # self.axes.set_title("Spinorama", pad=PLOT_TITLE_PAD)
        self.axes.set_xlabel("Frequency (Hz)")
        self.axes.set_ylabel("SPL (dB)")
        self.di_axes.set_ylabel("DI (dB)")
        apply_audio_frequency_axis(self.axes)
        self.axes.grid(which="major", color="#808080", linewidth=0.8, alpha=GRID_LINE_ALPHA)

        colors = {
            "On Axis": "#1f77b4",
            "Listen. Wind.": "#2ca02c",
            "Early Reflections": "#ff7f0e",
            "Sound Power": "#d62728",
            "PIR": "#9467bd",
            "ERDI": "#8c564b",
            "SPDI": "#17becf",
        }
        for name, values in curves.spl_curves():
            self.axes.plot(curves.freq_hz, values, linewidth=1.5, label=name, color=colors.get(name))

        for name, values in curves.di_curves():
            self.di_axes.plot(
                curves.freq_hz,
                values,
                linewidth=1.2,
                linestyle="--",
                label=name,
                color=colors.get(name),
            )

        self.axes.set_ylim(*SPINORAMA_SPL_LIMITS)
        self.di_axes.set_ylim(*SPINORAMA_DI_LIMITS)

        lines, labels = self.axes.get_legend_handles_labels()
        di_lines, di_labels = self.di_axes.get_legend_handles_labels()
        self.axes.legend(
            lines + di_lines,
            labels + di_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncols=4,
            borderaxespad=0.0,
            frameon=False,
        )
        apply_compact_plot_text(self.axes)
        apply_compact_plot_text(self.di_axes)
        self.draw_idle()
