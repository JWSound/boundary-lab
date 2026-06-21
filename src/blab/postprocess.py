"""
Formats BEM solver output for visualization or analysis in downstream modules.

    - Ingests pressure_data_raw.npz from the solver package output (horizontal/vertical polar data)
    - Creates pressure_data_formatted.npz containing plot-ready arrays:
        * clipped polar SPL matrices
        * Fractional octave smoothed horizontal and vertical isobar matrices
        * Real + Imaginary Impedance arrays
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from blab.defaults import FORMATTED_OUTPUT_NPZ, SOLVER_OUTPUT_NPZ


@dataclass
class PrepConfig:
    input_polar_npz: Path = SOLVER_OUTPUT_NPZ
    output_npz: Path = FORMATTED_OUTPUT_NPZ

    min_db: float = -30.0  # minimum dB for clipping SPL data
    max_db: float = 0.0  # maximum dB for clipping SPL data

    angle_samples: int = 250
    freq_samples: int = 500
    octave_smoothing: int | float | None = 24  # fractional octave smoothing for plots
    hor_ref_angle: float = 10  # normalization angle for horizontal plane
    vert_ref_angle: float = 10  # normalization angle for vertical plane
    spin_hor_ref_angle: float = 0.0
    spin_vert_ref_angle: float = 0.0
    normalize_polar: bool = True
    auto_db_span: bool = False


def _load_polar_npz(file_path: Path) -> dict[str, np.ndarray]:
    if not file_path.exists():
        raise FileNotFoundError(f"Polar file not found: {file_path}")

    with np.load(file_path) as data:
        required = {
            "freq_hz",
            "polar_angle_deg",
            "horizontal_spl_norm_db",
            "vertical_spl_norm_db",
            "impedance_freq_hz",
            "impedance_real",
            "impedance_imag",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Polar NPZ missing arrays: {sorted(missing)}")

        freq_hz = np.asarray(data["freq_hz"], dtype=float)
        angles_deg = np.asarray(data["polar_angle_deg"], dtype=float)
        horizontal = np.asarray(data["horizontal_spl_norm_db"], dtype=float)
        vertical = np.asarray(data["vertical_spl_norm_db"], dtype=float)
        horizontal_raw = (
            np.asarray(data["horizontal_spl_db"], dtype=float) if "horizontal_spl_db" in data.files else horizontal
        )
        vertical_raw = np.asarray(data["vertical_spl_db"], dtype=float) if "vertical_spl_db" in data.files else vertical
        z_freq = np.asarray(data["impedance_freq_hz"], dtype=float)
        z_real = np.asarray(data["impedance_real"], dtype=float)
        z_imag = np.asarray(data["impedance_imag"], dtype=float)
        z_names = np.asarray(
            data["impedance_radiator_names"] if "impedance_radiator_names" in data.files else ["Radiator"],
        )

    if horizontal.ndim != 2 or vertical.ndim != 2:
        raise ValueError("horizontal/vertical SPL arrays must be 2D (n_freq, n_angles).")
    if horizontal.shape != vertical.shape:
        raise ValueError("horizontal_spl_norm_db and vertical_spl_norm_db must have matching shapes.")
    if horizontal_raw.shape != horizontal.shape or vertical_raw.shape != vertical.shape:
        raise ValueError("Raw and normalized SPL arrays must have matching shapes.")
    if horizontal.shape[0] != freq_hz.size:
        raise ValueError("horizontal/vertical SPL first axis must match freq_hz length.")
    if horizontal.shape[1] != angles_deg.size:
        raise ValueError("polar_angle_deg length must match SPL second axis.")
    if z_real.ndim == 1:
        z_real = z_real[np.newaxis, :]
        z_imag = z_imag[np.newaxis, :]
    if z_real.ndim != 2 or z_imag.ndim != 2:
        raise ValueError("Impedance real/imag arrays must be 1D or 2D.")
    if z_real.shape != z_imag.shape:
        raise ValueError("Impedance real/imag arrays must have matching shapes.")
    if z_real.shape[1] != z_freq.size:
        raise ValueError("Impedance arrays second axis must match impedance_freq_hz length.")
    if z_names.size != z_real.shape[0]:
        raise ValueError("impedance_radiator_names length must match impedance array first axis.")

    return {
        "freq_hz": freq_hz,
        "polar_angle_deg": angles_deg,
        "horizontal_spl_db": horizontal_raw,
        "vertical_spl_db": vertical_raw,
        "horizontal_spl_norm_db": horizontal,
        "vertical_spl_norm_db": vertical,
        "impedance_freq_hz": z_freq,
        "impedance_radiator_names": z_names,
        "impedance_real": z_real,
        "impedance_imag": z_imag,
    }


def _fractional_octave_smooth(
    spl_matrix: np.ndarray,
    freqs: np.ndarray,
    fraction: int | float | None,
) -> np.ndarray:
    if not fraction or fraction <= 0:
        return spl_matrix

    if freqs.ndim != 1 or freqs.size < 2:
        return spl_matrix

    log2_freqs = np.log2(freqs)
    half_band = 1.0 / (2.0 * float(fraction))

    smoothed = np.empty_like(spl_matrix)
    for i in range(freqs.size):
        mask = np.abs(log2_freqs - log2_freqs[i]) <= half_band
        smoothed[:, i] = np.mean(spl_matrix[:, mask], axis=1)
    return smoothed


def _normalize_plane_to_reference_angle(
    spl_matrix: np.ndarray,
    angles_deg: np.ndarray,
    reference_angle_deg: float,
) -> np.ndarray:
    if spl_matrix.ndim != 2:
        raise ValueError("spl_matrix must be 2D with shape (n_angles, n_freq).")

    if angles_deg.ndim != 1 or angles_deg.size != spl_matrix.shape[0]:
        raise ValueError("angles_deg must be 1D and match spl_matrix first axis.")

    if angles_deg.size < 2:
        return spl_matrix

    reference_wrapped = ((float(reference_angle_deg) + 180.0) % 360.0) - 180.0

    if np.isclose(angles_deg[0], -180.0) and np.isclose(angles_deg[-1], 180.0):
        interp_angles = angles_deg[:-1]
        interp_matrix = spl_matrix[:-1, :]
    else:
        interp_angles = angles_deg
        interp_matrix = spl_matrix

    if interp_angles.size < 2:
        return spl_matrix

    angles_ext = np.concatenate([interp_angles - 360.0, interp_angles, interp_angles + 360.0])
    out = np.empty_like(spl_matrix)

    for i in range(spl_matrix.shape[1]):
        values = interp_matrix[:, i]
        values_ext = np.concatenate([values, values, values])
        ref_db = np.interp(reference_wrapped, angles_ext, values_ext)
        out[:, i] = spl_matrix[:, i] - ref_db

    return out


def _interpolate_isobar_heatmap(
    angles_deg: np.ndarray,
    freqs: np.ndarray,
    spl_matrix: np.ndarray,
    angle_samples: int | None,
    freq_samples: int | None,
    fill_db: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if angle_samples is None and freq_samples is None:
        return angles_deg, freqs, spl_matrix

    if angles_deg.size < 2 or freqs.size < 2:
        return angles_deg, freqs, spl_matrix

    log_freqs = np.log10(freqs)
    target_angles = np.linspace(angles_deg.min(), angles_deg.max(), angle_samples or angles_deg.size)
    target_log_freqs = np.linspace(log_freqs.min(), log_freqs.max(), freq_samples or freqs.size)

    interpolator = RegularGridInterpolator(
        (angles_deg, log_freqs),
        spl_matrix,
        bounds_error=False,
        fill_value=fill_db,
    )

    angle_grid, log_freq_grid = np.meshgrid(target_angles, target_log_freqs, indexing="ij")
    query = np.column_stack([angle_grid.ravel(), log_freq_grid.ravel()])
    spl_interp = interpolator(query).reshape(angle_grid.shape)

    return target_angles, np.power(10.0, target_log_freqs), spl_interp.astype(np.float32, copy=False)


def prepare_visualization_data(cfg: PrepConfig):
    polar = _load_polar_npz(cfg.input_polar_npz)

    dataset = prepare_visualization_data_from_arrays(
        freq_hz=polar["freq_hz"],
        polar_angle_deg=polar["polar_angle_deg"],
        horizontal_spl_norm_db=polar["horizontal_spl_norm_db"],
        vertical_spl_norm_db=polar["vertical_spl_norm_db"],
        horizontal_spl_db=polar["horizontal_spl_db"],
        vertical_spl_db=polar["vertical_spl_db"],
        impedance_freq_hz=polar["impedance_freq_hz"],
        impedance_radiator_names=polar["impedance_radiator_names"],
        impedance_real=polar["impedance_real"],
        impedance_imag=polar["impedance_imag"],
        cfg=cfg,
    )

    np.savez_compressed(cfg.output_npz, **dataset)


def prepare_visualization_data_from_arrays(
    *,
    freq_hz: np.ndarray,
    polar_angle_deg: np.ndarray,
    horizontal_spl_norm_db: np.ndarray,
    vertical_spl_norm_db: np.ndarray,
    horizontal_spl_db: np.ndarray | None = None,
    vertical_spl_db: np.ndarray | None = None,
    impedance_freq_hz: np.ndarray,
    impedance_radiator_names: np.ndarray,
    impedance_real: np.ndarray,
    impedance_imag: np.ndarray,
    cfg: PrepConfig,
) -> dict[str, np.ndarray]:
    freq_hz = np.asarray(freq_hz, dtype=np.float32)
    base_angles_deg = np.asarray(polar_angle_deg, dtype=np.float32)

    horizontal_unclipped = np.asarray(horizontal_spl_norm_db, dtype=float)
    vertical_unclipped = np.asarray(vertical_spl_norm_db, dtype=float)
    raw_horizontal = np.asarray(
        horizontal_unclipped if horizontal_spl_db is None else horizontal_spl_db,
        dtype=float,
    )
    raw_vertical = np.asarray(
        vertical_unclipped if vertical_spl_db is None else vertical_spl_db,
        dtype=float,
    )

    if horizontal_unclipped.ndim != 2 or vertical_unclipped.ndim != 2:
        raise ValueError("horizontal/vertical SPL arrays must be 2D (n_freq, n_angles).")
    if horizontal_unclipped.shape != vertical_unclipped.shape:
        raise ValueError("horizontal and vertical SPL arrays must have matching shapes.")
    if horizontal_unclipped.shape[0] != freq_hz.size:
        raise ValueError("SPL first axis must match freq_hz length.")
    if horizontal_unclipped.shape[1] != base_angles_deg.size:
        raise ValueError("SPL second axis must match polar_angle_deg length.")
    if raw_horizontal.shape != horizontal_unclipped.shape or raw_vertical.shape != vertical_unclipped.shape:
        raise ValueError("Raw and normalized SPL arrays must have matching shapes.")

    horizontal = horizontal_unclipped.T
    vertical = vertical_unclipped.T

    if cfg.normalize_polar:
        horizontal = _normalize_plane_to_reference_angle(
            horizontal,
            base_angles_deg,
            cfg.hor_ref_angle,
        )
        vertical = _normalize_plane_to_reference_angle(
            vertical,
            base_angles_deg,
            cfg.vert_ref_angle,
        )

    horizontal = _fractional_octave_smooth(horizontal, freq_hz, cfg.octave_smoothing)
    vertical = _fractional_octave_smooth(vertical, freq_hz, cfg.octave_smoothing)
    clip_min_db, clip_max_db = _resolve_db_span(horizontal, vertical, cfg)
    horizontal = np.clip(horizontal, clip_min_db, clip_max_db)
    vertical = np.clip(vertical, clip_min_db, clip_max_db)

    isobar_angles, isobar_freqs, horizontal_interp = _interpolate_isobar_heatmap(
        base_angles_deg,
        freq_hz,
        horizontal,
        cfg.angle_samples,
        cfg.freq_samples,
        clip_min_db,
    )
    _, _, vertical_interp = _interpolate_isobar_heatmap(
        base_angles_deg,
        freq_hz,
        vertical,
        cfg.angle_samples,
        cfg.freq_samples,
        clip_min_db,
    )

    z_freq = np.asarray(impedance_freq_hz, dtype=np.float32)
    z_real = np.asarray(impedance_real, dtype=np.float32)
    z_imag = np.asarray(impedance_imag, dtype=np.float32)
    z_names = np.asarray(impedance_radiator_names)

    return {
        "freq_hz": freq_hz,
        "polar_angle_deg": base_angles_deg,
        "horizontal_spl_db": raw_horizontal.astype(np.float32, copy=False),
        "vertical_spl_db": raw_vertical.astype(np.float32, copy=False),
        "horizontal_spl_norm_db": horizontal.T.astype(np.float32, copy=False),
        "vertical_spl_norm_db": vertical.T.astype(np.float32, copy=False),
        "isobar_angle_deg": isobar_angles.astype(np.float32, copy=False),
        "isobar_freq_hz": isobar_freqs.astype(np.float32, copy=False),
        "horizontal_isobar_db": horizontal_interp,
        "vertical_isobar_db": vertical_interp,
        "impedance_freq_hz": z_freq,
        "impedance_radiator_names": z_names,
        "impedance_real": z_real,
        "impedance_imag": z_imag,
        "clip_min_db": np.float32(clip_min_db),
        "clip_max_db": np.float32(clip_max_db),
        "horizontal_normalization_angle_deg": np.float32(cfg.hor_ref_angle),
        "vertical_normalization_angle_deg": np.float32(cfg.vert_ref_angle),
        "spin_horizontal_reference_angle_deg": np.float32(cfg.spin_hor_ref_angle),
        "spin_vertical_reference_angle_deg": np.float32(cfg.spin_vert_ref_angle),
        "polar_normalization_enabled": np.asarray(cfg.normalize_polar),
    }


def _resolve_db_span(horizontal: np.ndarray, vertical: np.ndarray, cfg: PrepConfig) -> tuple[float, float]:
    if not cfg.auto_db_span:
        return float(cfg.min_db), float(cfg.max_db)

    combined = np.concatenate([horizontal.ravel(), vertical.ravel()])
    finite = combined[np.isfinite(combined)]
    if finite.size == 0:
        return float(cfg.min_db), float(cfg.max_db)

    min_db = float(np.floor(np.min(finite)))
    max_db = float(np.ceil(np.max(finite)))
    if np.isclose(min_db, max_db):
        max_db = min_db + 1.0
    return min_db, max_db


def _build_arg_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Prepare solver output for visualization.")
    parser.add_argument(
        "input_polar_npz",
        nargs="?",
        type=Path,
        default=PrepConfig.input_polar_npz,
        help="Path to the input solver NPZ file",
    )
    parser.add_argument(
        "output_npz",
        nargs="?",
        type=Path,
        default=PrepConfig.output_npz,
        help="Path to the output formatted NPZ file",
    )
    parser.add_argument(
        "--min-db",
        type=float,
        default=PrepConfig.min_db,
        help="Minimum dB clipping value",
    )
    parser.add_argument(
        "--max-db",
        type=float,
        default=PrepConfig.max_db,
        help="Maximum dB clipping value",
    )
    parser.add_argument(
        "--octave-smoothing",
        type=float,
        default=PrepConfig.octave_smoothing,
        help="Fractional-octave smoothing denominator; use 0 to disable",
    )
    parser.add_argument(
        "--hor-ref-angle",
        type=float,
        default=PrepConfig.hor_ref_angle,
        help="Horizontal reference angle for normalization",
    )
    parser.add_argument(
        "--vert-ref-angle",
        type=float,
        default=PrepConfig.vert_ref_angle,
        help="Vertical reference angle for normalization",
    )
    parser.add_argument(
        "--spin-hor-ref-angle",
        type=float,
        default=PrepConfig.spin_hor_ref_angle,
        help="Horizontal reference axis angle for spinorama curves",
    )
    parser.add_argument(
        "--spin-vert-ref-angle",
        type=float,
        default=PrepConfig.spin_vert_ref_angle,
        help="Vertical reference axis angle for spinorama curves",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> PrepConfig:
    octave_fraction = args.octave_smoothing
    if octave_fraction == 0:
        octave_fraction = None

    return PrepConfig(
        input_polar_npz=args.input_polar_npz,
        output_npz=args.output_npz,
        min_db=args.min_db,
        max_db=args.max_db,
        octave_smoothing=octave_fraction,
        hor_ref_angle=args.hor_ref_angle,
        vert_ref_angle=args.vert_ref_angle,
        spin_hor_ref_angle=args.spin_hor_ref_angle,
        spin_vert_ref_angle=args.spin_vert_ref_angle,
    )


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    args = _build_arg_parser(prog=prog).parse_args(argv)
    config = _config_from_args(args)
    prepare_visualization_data(config)
    print(f"Saved {config.output_npz}")


if __name__ == "__main__":
    main()
