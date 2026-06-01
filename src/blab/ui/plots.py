"""Matplotlib canvases used by the live solver GUI."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure

from blab.plotting import VisualizerConfig
from blab.spinorama import SpinoramaCurves, compute_spinorama_from_planes


AUDIO_FREQ_MIN_HZ = 20
AUDIO_FREQ_MAX_HZ = 20000
AUDIO_AXIS_TICKS = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
AUDIO_AXIS_LABELS = ["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"]
FREQ_SLIDER_STEPS = 1000
LIVE_ISOBAR_ANGLE_SAMPLES = 250
LIVE_ISOBAR_FREQ_SAMPLES = 500
PLOT_TITLE_SIZE = 11
PLOT_LABEL_SIZE = 9
PLOT_TICK_SIZE = 9
PLOT_LEGEND_SIZE = 9
PLOT_TITLE_PAD = 1
ON_AXIS_DB_SPAN = 50.0
SPINORAMA_SPL_LIMITS = (-40.0, 10.0)
SPINORAMA_DI_LIMITS = (-5.0, 45.0)


def apply_audio_frequency_axis(axes) -> None:
    axes.set_xscale("log")
    axes.set_xlim(AUDIO_FREQ_MIN_HZ, AUDIO_FREQ_MAX_HZ)
    axes.set_xticks(AUDIO_AXIS_TICKS)
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
    fraction = (
        np.log10(clamped) - np.log10(AUDIO_FREQ_MIN_HZ)
    ) / (np.log10(AUDIO_FREQ_MAX_HZ) - np.log10(AUDIO_FREQ_MIN_HZ))
    return int(round(fraction * FREQ_SLIDER_STEPS))


def slider_value_to_frequency(value: int) -> int:
    fraction = float(value) / FREQ_SLIDER_STEPS
    log_freq = np.log10(AUDIO_FREQ_MIN_HZ) + fraction * (
        np.log10(AUDIO_FREQ_MAX_HZ) - np.log10(AUDIO_FREQ_MIN_HZ)
    )
    return int(round(10.0**log_freq))


class IsobarCanvas(FigureCanvas):
    def __init__(self, title: str, *, left_margin: float = 0.14, right_margin: float = 0.98):
        self.figure = Figure(figsize=(5.5, 2.8), dpi=100)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.title = title
        self.left_margin = float(left_margin)
        self.right_margin = float(right_margin)
        self.colors = VisualizerConfig.custom_colors
        self._mesh_artist = None
        self._line_artist = None
        self._mesh_freqs_hz: np.ndarray | None = None
        self._mesh_angles_deg: np.ndarray | None = None
        self._mesh_clip: tuple[float, float] | None = None
        self._apply_layout()
        self._draw_empty()

    def _apply_layout(self) -> None:
        self.figure.subplots_adjust(left=self.left_margin, right=self.right_margin, top=0.91, bottom=0.2)

    def _configure_axes(self) -> None:
        self._apply_layout()
        self.axes.set_title(self.title, pad=PLOT_TITLE_PAD)
        self.axes.set_xlabel("Frequency (Hz)")
        self.axes.set_ylabel("Angle (deg)")
        apply_audio_frequency_axis(self.axes)
        self.axes.set_ylim(-180, 180)
        self.axes.set_yticks(np.arange(-180, 181, 30))
        self.axes.grid(which="major", color="#808080", linewidth=0.8)
        apply_compact_plot_text(self.axes)

    def _draw_empty(self) -> None:
        clear_plot_axes(self.axes)
        self._mesh_artist = None
        self._line_artist = None
        self._mesh_freqs_hz = None
        self._mesh_angles_deg = None
        self._mesh_clip = None
        self._configure_axes()
        self.draw_idle()

    def _remove_artist(self, name: str) -> None:
        artist = getattr(self, name)
        if artist is None:
            return
        try:
            artist.remove()
        except ValueError:
            pass
        setattr(self, name, None)

    def _mesh_matches(self, freqs_hz: np.ndarray, angles_deg: np.ndarray, clip: tuple[float, float]) -> bool:
        return (
            self._mesh_artist is not None
            and self._mesh_freqs_hz is not None
            and self._mesh_angles_deg is not None
            and self._mesh_clip == clip
            and self._mesh_freqs_hz.shape == freqs_hz.shape
            and self._mesh_angles_deg.shape == angles_deg.shape
            and np.array_equal(self._mesh_freqs_hz, freqs_hz)
            and np.array_equal(self._mesh_angles_deg, angles_deg)
        )

    def update_plot(
        self,
        freqs_hz: np.ndarray,
        angles_deg: np.ndarray,
        values_db: np.ndarray,
        clip_min_db: float,
        clip_max_db: float,
    ) -> None:
        freqs_hz = np.asarray(freqs_hz, dtype=np.float32)
        angles_deg = np.asarray(angles_deg, dtype=np.float32)
        clipped = np.clip(np.asarray(values_db, dtype=np.float32), clip_min_db, clip_max_db)
        clip = (float(clip_min_db), float(clip_max_db))

        if freqs_hz.size >= 2 and angles_deg.size >= 2:
            self._remove_artist("_line_artist")
            if self._mesh_matches(freqs_hz, angles_deg, clip):
                self._mesh_artist.set_array(clipped.ravel())
            else:
                self._remove_artist("_mesh_artist")
                boundaries = np.linspace(clip_min_db, clip_max_db, len(self.colors) + 1)
                cmap = ListedColormap(list(self.colors))
                norm = BoundaryNorm(boundaries, cmap.N)
                self._mesh_artist = self.axes.pcolormesh(
                    freqs_hz,
                    angles_deg,
                    clipped,
                    cmap=cmap,
                    norm=norm,
                    shading="gouraud",
                )
                self._mesh_freqs_hz = freqs_hz.copy()
                self._mesh_angles_deg = angles_deg.copy()
                self._mesh_clip = clip
        elif freqs_hz.size == 1:
            self._remove_artist("_mesh_artist")
            self._mesh_freqs_hz = None
            self._mesh_angles_deg = None
            self._mesh_clip = None
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
            self._remove_artist("_mesh_artist")
            self._remove_artist("_line_artist")
            self._mesh_freqs_hz = None
            self._mesh_angles_deg = None
            self._mesh_clip = None

        self._configure_axes()
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
        self.axes.grid(which="major", color="#808080", linewidth=0.8)
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
        self.axes.grid(which="major", color="#808080", linewidth=0.8)

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
        self.axes.grid(which="major", color="#808080", linewidth=0.8)
        apply_compact_plot_text(self.axes)
        self.draw_idle()

    def update_plot(
        self,
        freqs_hz: np.ndarray,
        angles_deg: np.ndarray,
        horizontal_spl_db: np.ndarray,
    ) -> None:
        clear_plot_axes(self.axes)
        self.axes.set_title("On-Axis Frequency Response", pad=PLOT_TITLE_PAD)
        self.axes.set_xlabel("Frequency (Hz)")
        self.axes.set_ylabel("SPL (dB)")
        apply_audio_frequency_axis(self.axes)
        self.axes.grid(which="major", color="#808080", linewidth=0.8)

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
            self.axes.plot(freqs_hz, on_axis, linewidth=1.8, color="#1f77b4")
            finite = on_axis[np.isfinite(on_axis)]
            if finite.size:
                ymax = float(np.ceil(np.max(finite) / 5.0) * 5.0)
                self.axes.set_ylim(ymax - ON_AXIS_DB_SPAN, ymax)

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
        self.axes.grid(which="major", color="#808080", linewidth=0.8)
        apply_compact_plot_text(self.axes)
        apply_compact_plot_text(self.di_axes)
        self.draw_idle()

    def update_plot(
        self,
        freqs_hz: np.ndarray,
        angles_deg: np.ndarray,
        horizontal_spl_db: np.ndarray,
        vertical_spl_db: np.ndarray,
    ) -> None:
        curves = compute_spinorama_from_planes(
            freq_hz=freqs_hz,
            polar_angle_deg=angles_deg,
            horizontal_spl_db=horizontal_spl_db,
            vertical_spl_db=vertical_spl_db,
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
        self.axes.grid(which="major", color="#808080", linewidth=0.8)

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
