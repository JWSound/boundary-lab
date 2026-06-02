"""Generates PNG plots from preprocessed directivity visualization data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from blab.defaults import FORMATTED_OUTPUT_NPZ, PLOT_OUTPUT_DIR
from blab.spinorama import SpinoramaCurves, compute_spinorama_from_planes


PLOT_TITLE_SIZE = 11
PLOT_LABEL_SIZE = 9
PLOT_TICK_SIZE = 9
PLOT_LEGEND_SIZE = 9
PLOT_TITLE_PAD = 1
ON_AXIS_DB_SPAN = 50.0
SPINORAMA_SPL_LIMITS = (-40.0, 10.0)
SPINORAMA_DI_LIMITS = (-5.0, 45.0)


@dataclass
class VisualizerConfig:
    input_npz: Path = FORMATTED_OUTPUT_NPZ
    output_dir: Path = PLOT_OUTPUT_DIR
    colorbar_tick_step_db: float = 3.0
    figure_width_in: float = 11.0
    figure_height_in: float = 6.0
    figure_dpi: int = 160
    isobar_interp_angle_factor: int = 2
    isobar_interp_freq_factor: int = 3
    custom_colors: tuple[str, ...] = (
        "#00008F",
        "#0000FF",
        "#006FFF",
        "#00DFFF",
        "#4FFFBF",
        "#BFFF4F",
        "#FFDF00",
        "#FF6F00",
        "#FF0000",
        "#8F0000",
    )


def load_data(npz_path: Path) -> dict[str, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"File not found: {npz_path}")

    with np.load(npz_path) as data:
        required = {
            "isobar_angle_deg",
            "isobar_freq_hz",
            "horizontal_isobar_db",
            "vertical_isobar_db",
            "freq_hz",
            "polar_angle_deg",
            "horizontal_spl_norm_db",
            "impedance_freq_hz",
            "impedance_real",
            "impedance_imag",
            "clip_min_db",
            "clip_max_db",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Visualization NPZ missing arrays: {sorted(missing)}")

        out = {k: data[k] for k in data.files}
        if "horizontal_spl_db" not in out:
            out["horizontal_spl_db"] = out["horizontal_spl_norm_db"]
        return out


def _build_db_tick_values(min_db: float, max_db: float, step_db: float) -> np.ndarray:
    if step_db <= 0:
        raise ValueError("colorbar_tick_step_db must be > 0")

    start = np.ceil(min_db / step_db) * step_db
    end = np.floor(max_db / step_db) * step_db

    if end < start:
        return np.array([min_db, max_db], dtype=float)

    return np.arange(start, end + 0.5 * step_db, step_db, dtype=float)


def _setup_log_frequency_axis(ax: plt.Axes) -> None:
    tickvals = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    ticktext = ["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"]

    ax.set_xscale("log")
    ax.set_xlim(200, 20000)
    ax.set_xticks(tickvals)
    ax.set_xticklabels(ticktext)


def _apply_compact_plot_text(ax: plt.Axes) -> None:
    ax.title.set_fontsize(PLOT_TITLE_SIZE)
    ax.xaxis.label.set_size(PLOT_LABEL_SIZE)
    ax.yaxis.label.set_size(PLOT_LABEL_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=PLOT_TICK_SIZE)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(PLOT_LEGEND_SIZE)


def _upsample_isobar_grid(
    angle_deg: np.ndarray,
    freqs_hz: np.ndarray,
    spl_matrix: np.ndarray,
    angle_factor: int,
    freq_factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if angle_factor < 1 or freq_factor < 1:
        raise ValueError("Interpolation factors must be >= 1")

    if angle_factor == 1 and freq_factor == 1:
        return angle_deg, freqs_hz, spl_matrix

    n_angles = angle_deg.shape[0]
    n_freqs = freqs_hz.shape[0]

    dense_angle_count = (n_angles - 1) * angle_factor + 1
    dense_freq_count = (n_freqs - 1) * freq_factor + 1

    dense_angles = np.linspace(float(angle_deg[0]), float(angle_deg[-1]), dense_angle_count)
    dense_log_freqs = np.linspace(np.log10(float(freqs_hz[0])), np.log10(float(freqs_hz[-1])), dense_freq_count)
    source_log_freqs = np.log10(freqs_hz)

    freq_upsampled = np.empty((n_angles, dense_freq_count), dtype=float)
    for row_idx in range(n_angles):
        freq_upsampled[row_idx, :] = np.interp(dense_log_freqs, source_log_freqs, spl_matrix[row_idx, :])

    dense_spl = np.empty((dense_angle_count, dense_freq_count), dtype=float)
    for col_idx in range(dense_freq_count):
        dense_spl[:, col_idx] = np.interp(dense_angles, angle_deg, freq_upsampled[:, col_idx])

    return dense_angles, np.power(10.0, dense_log_freqs), dense_spl


def _save_isobar_plot(
    output_path: Path,
    angle_deg: np.ndarray,
    freqs_hz: np.ndarray,
    spl_matrix: np.ndarray,
    title: str,
    colors: tuple[str, ...],
    clip_min_db: float,
    clip_max_db: float,
    colorbar_tick_step_db: float,
    figure_width_in: float,
    figure_height_in: float,
    figure_dpi: int,
    isobar_interp_angle_factor: int,
    isobar_interp_freq_factor: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    boundaries = np.linspace(clip_min_db, clip_max_db, len(colors) + 1)
    cmap = ListedColormap(list(colors))
    norm = BoundaryNorm(boundaries, cmap.N)
    cbar_ticks = _build_db_tick_values(clip_min_db, clip_max_db, colorbar_tick_step_db)

    plot_angles, plot_freqs, plot_spl = _upsample_isobar_grid(
        angle_deg=angle_deg,
        freqs_hz=freqs_hz,
        spl_matrix=spl_matrix,
        angle_factor=isobar_interp_angle_factor,
        freq_factor=isobar_interp_freq_factor,
    )
    plot_spl = np.clip(plot_spl, clip_min_db, clip_max_db)

    fig, ax = plt.subplots(figsize=(figure_width_in, figure_height_in), dpi=figure_dpi)
    mesh = ax.pcolormesh(plot_freqs, plot_angles, plot_spl, cmap=cmap, norm=norm, shading="gouraud")

    _setup_log_frequency_axis(ax)
    ax.set_ylim(-180, 180)
    ax.set_yticks(np.arange(-180, 181, 30))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title(title, pad=PLOT_TITLE_PAD)
    ax.grid(which="major", color="#808080", linewidth=0.8)

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Normalized SPL (dB)")
    cbar.set_ticks(cbar_ticks)
    cbar.ax.yaxis.label.set_size(PLOT_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=PLOT_TICK_SIZE)

    _apply_compact_plot_text(ax)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _save_impedance_plot(
    output_path: Path,
    impedance_freq_hz: np.ndarray,
    radiator_names: np.ndarray,
    impedance_real: np.ndarray,
    impedance_imag: np.ndarray,
    figure_width_in: float,
    figure_height_in: float,
    figure_dpi: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(figure_width_in, figure_height_in), dpi=figure_dpi)

    if impedance_real.ndim == 1:
        impedance_real = impedance_real[np.newaxis, :]
        impedance_imag = impedance_imag[np.newaxis, :]

    for i in range(impedance_real.shape[0]):
        name = str(radiator_names[i]) if i < radiator_names.size else f"Radiator {i + 1}"
        ax.plot(impedance_freq_hz, impedance_real[i], linewidth=2, label=f"{name} Z real")
        ax.plot(
            impedance_freq_hz,
            impedance_imag[i],
            linewidth=2,
            linestyle="--",
            label=f"{name} Z imag",
        )

    _setup_log_frequency_axis(ax)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Acoustic Impedance (Pa*s/m^3)")
    ax.set_title("Acoustic Impedance", pad=PLOT_TITLE_PAD)
    ax.grid(which="major", color="#808080", linewidth=0.8)
    ax.legend()

    _apply_compact_plot_text(ax)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _save_on_axis_plot(
    output_path: Path,
    freqs_hz: np.ndarray,
    angle_deg: np.ndarray,
    horizontal_spl: np.ndarray,
    figure_width_in: float,
    figure_height_in: float,
    figure_dpi: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    on_axis = np.asarray(
        [
            np.interp(0.0, angle_deg.astype(float), row.astype(float))
            for row in horizontal_spl
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(figure_width_in, figure_height_in), dpi=figure_dpi)
    ax.plot(freqs_hz, on_axis, linewidth=2, color="#1f77b4")
    _setup_log_frequency_axis(ax)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized SPL (dB)")
    ax.set_title("On-Axis Frequency Response", pad=PLOT_TITLE_PAD)
    ax.grid(which="major", color="#808080", linewidth=0.8)

    finite = on_axis[np.isfinite(on_axis)]
    if finite.size:
        ymax = float(np.ceil(np.max(finite) / 5.0) * 5.0)
        ax.set_ylim(ymax - ON_AXIS_DB_SPAN, ymax)

    _apply_compact_plot_text(ax)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _save_spinorama_plot(
    output_path: Path,
    curves: SpinoramaCurves,
    figure_width_in: float,
    figure_height_in: float,
    figure_dpi: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(figure_width_in, figure_height_in), dpi=figure_dpi)
    fig.subplots_adjust(left=0.14, right=0.9, top=0.91, bottom=0.28)
    di_ax = ax.twinx()
    di_ax.yaxis.set_label_position("right")
    di_ax.yaxis.tick_right()
    colors = {
        "On Axis": "#1f77b4",
        "Listen. Wind.": "#2ca02c",
        "Early Refl.": "#ff7f0e",
        "Sound Power": "#d62728",
        "PIR": "#9467bd",
        "ERDI": "#8c564b",
        "SPDI": "#17becf",
    }
    for name, values in curves.spl_curves():
        ax.plot(curves.freq_hz, values, linewidth=1.8, label=name, color=colors.get(name))
    for name, values in curves.di_curves():
        di_ax.plot(
            curves.freq_hz,
            values,
            linewidth=1.4,
            linestyle="--",
            label=name,
            color=colors.get(name),
        )

    _setup_log_frequency_axis(ax)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("SPL (dB)")
    di_ax.set_ylabel("DI (dB)")
    ax.set_title("Spin Estimate", pad=PLOT_TITLE_PAD)
    ax.grid(which="major", color="#808080", linewidth=0.8)

    ax.set_ylim(*SPINORAMA_SPL_LIMITS)
    di_ax.set_ylim(*SPINORAMA_DI_LIMITS)

    lines, labels = ax.get_legend_handles_labels()
    di_lines, di_labels = di_ax.get_legend_handles_labels()
    ax.legend(
        lines + di_lines,
        labels + di_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncols=4,
        borderaxespad=0.0,
        frameon=False,
    )

    _apply_compact_plot_text(ax)
    _apply_compact_plot_text(di_ax)
    fig.savefig(output_path)
    plt.close(fig)


def generate_plots(dataset: dict[str, np.ndarray], cfg: VisualizerConfig) -> dict[str, str]:
    output_horizontal_png = cfg.output_dir / "horizontal_isobar.png"
    output_vertical_png = cfg.output_dir / "vertical_isobar.png"
    output_impedance_png = cfg.output_dir / "acoustic_impedance.png"
    output_on_axis_png = cfg.output_dir / "on_axis_frequency_response.png"
    output_spinorama_png = cfg.output_dir / "spinorama.png"

    angle_deg = dataset["isobar_angle_deg"].astype(float)
    freqs_hz = dataset["isobar_freq_hz"].astype(float)
    horizontal_spl = dataset["horizontal_isobar_db"].astype(float)
    vertical_spl = dataset["vertical_isobar_db"].astype(float)
    response_angle_deg = dataset["polar_angle_deg"].astype(float)
    horizontal_response_spl = dataset["horizontal_spl_db"].astype(float)
    impedance_freq_hz = dataset["impedance_freq_hz"].astype(float)
    impedance_real = dataset["impedance_real"].astype(float)
    impedance_imag = dataset["impedance_imag"].astype(float)
    radiator_names = dataset.get("impedance_radiator_names", np.asarray(["Radiator"]))

    clip_min_db = float(dataset["clip_min_db"])
    clip_max_db = float(dataset["clip_max_db"])

    _save_isobar_plot(
        output_path=output_horizontal_png,
        angle_deg=angle_deg,
        freqs_hz=freqs_hz,
        spl_matrix=horizontal_spl,
        title="Horizontal Isobar",
        colors=cfg.custom_colors,
        clip_min_db=clip_min_db,
        clip_max_db=clip_max_db,
        colorbar_tick_step_db=cfg.colorbar_tick_step_db,
        figure_width_in=cfg.figure_width_in,
        figure_height_in=cfg.figure_height_in,
        figure_dpi=cfg.figure_dpi,
        isobar_interp_angle_factor=cfg.isobar_interp_angle_factor,
        isobar_interp_freq_factor=cfg.isobar_interp_freq_factor,
    )

    _save_isobar_plot(
        output_path=output_vertical_png,
        angle_deg=angle_deg,
        freqs_hz=freqs_hz,
        spl_matrix=vertical_spl,
        title="Vertical Isobar",
        colors=cfg.custom_colors,
        clip_min_db=clip_min_db,
        clip_max_db=clip_max_db,
        colorbar_tick_step_db=cfg.colorbar_tick_step_db,
        figure_width_in=cfg.figure_width_in,
        figure_height_in=cfg.figure_height_in,
        figure_dpi=cfg.figure_dpi,
        isobar_interp_angle_factor=cfg.isobar_interp_angle_factor,
        isobar_interp_freq_factor=cfg.isobar_interp_freq_factor,
    )

    _save_impedance_plot(
        output_path=output_impedance_png,
        impedance_freq_hz=impedance_freq_hz,
        radiator_names=radiator_names,
        impedance_real=impedance_real,
        impedance_imag=impedance_imag,
        figure_width_in=cfg.figure_width_in,
        figure_height_in=cfg.figure_height_in,
        figure_dpi=cfg.figure_dpi,
    )

    _save_on_axis_plot(
        output_path=output_on_axis_png,
        freqs_hz=dataset["freq_hz"].astype(float),
        angle_deg=response_angle_deg,
        horizontal_spl=horizontal_response_spl,
        figure_width_in=cfg.figure_width_in,
        figure_height_in=cfg.figure_height_in,
        figure_dpi=cfg.figure_dpi,
    )

    spinorama = compute_spinorama_from_planes(
        freq_hz=dataset["freq_hz"].astype(float),
        polar_angle_deg=response_angle_deg,
        horizontal_spl_db=dataset["horizontal_spl_db"].astype(float),
        vertical_spl_db=dataset["vertical_spl_db"].astype(float),
        horizontal_reference_angle_deg=float(
            dataset.get("spin_horizontal_reference_angle_deg", dataset.get("horizontal_reference_angle_deg", 0.0))
        ),
        vertical_reference_angle_deg=float(
            dataset.get("spin_vertical_reference_angle_deg", dataset.get("vertical_reference_angle_deg", 0.0))
        ),
    )
    _save_spinorama_plot(
        output_path=output_spinorama_png,
        curves=spinorama,
        figure_width_in=cfg.figure_width_in,
        figure_height_in=cfg.figure_height_in,
        figure_dpi=cfg.figure_dpi,
    )

    return {
        "horizontal_isobar_png": str(output_horizontal_png),
        "vertical_isobar_png": str(output_vertical_png),
        "acoustic_impedance_png": str(output_impedance_png),
        "on_axis_frequency_response_png": str(output_on_axis_png),
        "spinorama_png": str(output_spinorama_png),
    }


def _build_arg_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Generate directivity/impedance PNG plots.")
    parser.add_argument(
        "input_npz",
        nargs="?",
        type=Path,
        default=VisualizerConfig.input_npz,
        help="Path to pressure_data_formatted.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VisualizerConfig.output_dir,
        help="Directory for generated plot PNG files",
    )
    return parser


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    args = _build_arg_parser(prog=prog).parse_args(argv)
    cfg = VisualizerConfig(
        input_npz=args.input_npz,
        output_dir=args.output_dir,
    )

    dataset = load_data(cfg.input_npz)
    outputs = generate_plots(dataset, cfg)
    print("Generated PNG plots:")
    for name, path in outputs.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
