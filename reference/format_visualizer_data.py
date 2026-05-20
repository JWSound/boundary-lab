"""
Prepares BEM pressure data for visualization.

This module loads the raw solver bundle written by bemppsolver.py,
applies clipping and smoothing used by the report workflow, and writes a
pressure_data_formatted.npz file containing precomputed arrays for the
visualizer.
"""

from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata


# ==========================================
# Configuration
# ==========================================
INPUT_NPZ = Path("pressure_data_raw.npz")
OUTPUT_NPZ = Path("pressure_data_formatted.npz")
MIN_DB = -30.0
MAX_DB = 0.0
GRID_RES_THETA = 40
GRID_RES_PHI = 80
ISOBAR_ANGLE_SAMPLES = 361
ISOBAR_ANGLE_SAMPLES_SMOOTH = 250
ISOBAR_FREQ_SAMPLES_SMOOTH = 500
ISOBAR_OCTAVE_SMOOTH_FRACTION = 24


def load_pressure_data(npz_path: Path) -> dict[str, np.ndarray]:
    required = {
        "freq_hz",
        "r_distance_m",
        "theta_polar_rad",
        "phi_azimuth_rad",
        "spl_norm",
        "impedance_real",
        "impedance_imag",
    }
    with np.load(npz_path) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"NPZ missing keys: {sorted(missing)}")

        bundle = {key: data[key] for key in required}

    freq_hz = bundle["freq_hz"]
    theta = bundle["theta_polar_rad"]
    phi = bundle["phi_azimuth_rad"]
    spl = bundle["spl_norm"]

    if freq_hz.ndim != 1:
        raise ValueError("freq_hz must be a 1D array.")
    if theta.ndim != 1 or phi.ndim != 1:
        raise ValueError("theta_polar_rad and phi_azimuth_rad must be 1D arrays.")
    if spl.shape != (freq_hz.size, theta.size):
        raise ValueError(
            "spl_norm must have shape (len(freq_hz), len(theta_polar_rad))."
        )

    return bundle


def _wrap_seam_data(
    points_theta: np.ndarray,
    points_phi: np.ndarray,
    points_spl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask_start = points_phi < np.pi
    mask_end = points_phi > np.pi

    theta_end = points_theta[mask_start]
    phi_end = points_phi[mask_start] + 2.0 * np.pi
    spl_end = points_spl[mask_start]

    theta_start = points_theta[mask_end]
    phi_start = points_phi[mask_end] - 2.0 * np.pi
    spl_start = points_spl[mask_end]

    points_theta = np.concatenate([points_theta, theta_end, theta_start])
    points_phi = np.concatenate([points_phi, phi_end, phi_start])
    points_spl = np.concatenate([points_spl, spl_end, spl_start])
    return points_theta, points_phi, points_spl


def _interpolate_spl_line(
    points_theta: np.ndarray,
    points_phi: np.ndarray,
    points_spl: np.ndarray,
    query_theta: np.ndarray,
    query_phi: np.ndarray,
) -> np.ndarray:
    values = griddata(
        (points_theta, points_phi),
        points_spl,
        (query_theta, query_phi),
        method="linear",
    )
    if np.isnan(values).any():
        nearest = griddata(
            (points_theta, points_phi),
            points_spl,
            (query_theta, query_phi),
            method="nearest",
        )
        values = np.where(np.isnan(values), nearest, values)
    return np.nan_to_num(values, nan=MIN_DB)


def _grid_spl_surface(
    points_theta: np.ndarray,
    points_phi: np.ndarray,
    points_spl: np.ndarray,
    theta_mesh: np.ndarray,
    phi_mesh: np.ndarray,
) -> np.ndarray:
    wrapped_theta, wrapped_phi, wrapped_spl = _wrap_seam_data(
        points_theta,
        points_phi,
        points_spl,
    )

    grid_spl = griddata(
        (wrapped_theta, wrapped_phi),
        wrapped_spl,
        (theta_mesh, phi_mesh),
        method="linear",
    )

    if np.isnan(grid_spl).any():
        nearest = griddata(
            (wrapped_theta, wrapped_phi),
            wrapped_spl,
            (theta_mesh, phi_mesh),
            method="nearest",
        )
        grid_spl = np.where(np.isnan(grid_spl), nearest, grid_spl)

    return np.nan_to_num(grid_spl, nan=MIN_DB)


def _balloon_xyz(
    grid_spl: np.ndarray,
    theta_mesh: np.ndarray,
    phi_mesh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_balloon = np.maximum(grid_spl - MIN_DB, 0.0)
    x = r_balloon * np.sin(theta_mesh) * np.cos(phi_mesh)
    y = r_balloon * np.sin(theta_mesh) * np.sin(phi_mesh)
    z = r_balloon * np.cos(theta_mesh)
    return x, y, z


def build_isobar_matrix(
    theta_polar_rad: np.ndarray,
    phi_azimuth_rad: np.ndarray,
    spl_norm: np.ndarray,
    angles_deg: np.ndarray,
    plane: str,
) -> np.ndarray:
    angles_rad = np.deg2rad(angles_deg)
    if plane == "horizontal":
        x = np.sin(angles_rad)
        z = np.cos(angles_rad)
        query_theta = np.arccos(z)
        query_phi = np.where(x >= 0.0, 0.0, np.pi)
    elif plane == "vertical":
        y = np.sin(angles_rad)
        z = np.cos(angles_rad)
        query_theta = np.arccos(z)
        query_phi = np.where(y >= 0.0, np.pi / 2.0, 3.0 * np.pi / 2.0)
    else:
        raise ValueError("plane must be 'horizontal' or 'vertical'")

    matrix = np.zeros((angles_deg.size, spl_norm.shape[0]), dtype=float)
    for idx in range(spl_norm.shape[0]):
        wrapped_theta, wrapped_phi, wrapped_spl = _wrap_seam_data(
            theta_polar_rad,
            phi_azimuth_rad,
            spl_norm[idx],
        )
        matrix[:, idx] = _interpolate_spl_line(
            wrapped_theta,
            wrapped_phi,
            wrapped_spl,
            query_theta,
            query_phi,
        )
    return matrix


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
    for idx in range(freqs.size):
        mask = np.abs(log2_freqs - log2_freqs[idx]) <= half_band
        smoothed[:, idx] = np.mean(spl_matrix[:, mask], axis=1)
    return smoothed


def _interpolate_isobar_heatmap(
    angles_deg: np.ndarray,
    freqs: np.ndarray,
    spl_matrix: np.ndarray,
    angle_samples: int | None,
    freq_samples: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if angle_samples is None and freq_samples is None:
        return angles_deg, freqs, spl_matrix

    log_freqs = np.log10(freqs)
    target_angles = np.linspace(
        angles_deg.min(),
        angles_deg.max(),
        angle_samples or angles_deg.size,
    )
    target_log_freqs = np.linspace(
        log_freqs.min(),
        log_freqs.max(),
        freq_samples or freqs.size,
    )

    interpolator = RegularGridInterpolator(
        (angles_deg, log_freqs),
        spl_matrix,
        bounds_error=False,
        fill_value=MIN_DB,
    )

    angle_grid, log_freq_grid = np.meshgrid(target_angles, target_log_freqs, indexing="ij")
    samples = np.column_stack([angle_grid.ravel(), log_freq_grid.ravel()])
    spl_interp = interpolator(samples).reshape(angle_grid.shape)
    return target_angles, np.power(10.0, target_log_freqs), spl_interp


def build_formatted_bundle(raw_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    freq_hz = raw_data["freq_hz"].astype(float, copy=False)
    theta_polar_rad = raw_data["theta_polar_rad"].astype(float, copy=False)
    phi_azimuth_rad = raw_data["phi_azimuth_rad"].astype(float, copy=False)
    spl_norm = np.clip(raw_data["spl_norm"].astype(float, copy=False), MIN_DB, MAX_DB)

    theta_grid = np.linspace(0.0, np.pi, GRID_RES_THETA)
    phi_grid = np.linspace(0.0, 2.0 * np.pi, GRID_RES_PHI)
    theta_mesh, phi_mesh = np.meshgrid(theta_grid, phi_grid)

    surface_shape = (freq_hz.size,) + theta_mesh.shape
    balloon_surface_spl = np.empty(surface_shape, dtype=float)
    balloon_x = np.empty(surface_shape, dtype=float)
    balloon_y = np.empty(surface_shape, dtype=float)
    balloon_z = np.empty(surface_shape, dtype=float)

    for idx in range(freq_hz.size):
        grid_spl = _grid_spl_surface(
            theta_polar_rad,
            phi_azimuth_rad,
            spl_norm[idx],
            theta_mesh,
            phi_mesh,
        )
        x, y, z = _balloon_xyz(grid_spl, theta_mesh, phi_mesh)
        balloon_surface_spl[idx] = grid_spl
        balloon_x[idx] = x
        balloon_y[idx] = y
        balloon_z[idx] = z

    base_angle_deg = np.linspace(-180.0, 180.0, ISOBAR_ANGLE_SAMPLES)
    horizontal_spl = build_isobar_matrix(
        theta_polar_rad,
        phi_azimuth_rad,
        spl_norm,
        base_angle_deg,
        "horizontal",
    )
    vertical_spl = build_isobar_matrix(
        theta_polar_rad,
        phi_azimuth_rad,
        spl_norm,
        base_angle_deg,
        "vertical",
    )

    horizontal_spl = _fractional_octave_smooth(
        horizontal_spl,
        freq_hz,
        ISOBAR_OCTAVE_SMOOTH_FRACTION,
    )
    vertical_spl = _fractional_octave_smooth(
        vertical_spl,
        freq_hz,
        ISOBAR_OCTAVE_SMOOTH_FRACTION,
    )

    isobar_angle_deg, isobar_freq_hz, horizontal_isobar_spl = _interpolate_isobar_heatmap(
        base_angle_deg,
        freq_hz,
        horizontal_spl,
        ISOBAR_ANGLE_SAMPLES_SMOOTH,
        ISOBAR_FREQ_SAMPLES_SMOOTH,
    )
    _, _, vertical_isobar_spl = _interpolate_isobar_heatmap(
        base_angle_deg,
        freq_hz,
        vertical_spl,
        ISOBAR_ANGLE_SAMPLES_SMOOTH,
        ISOBAR_FREQ_SAMPLES_SMOOTH,
    )

    return {
        "freq_hz": freq_hz,
        "theta_grid_rad": theta_mesh,
        "phi_grid_rad": phi_mesh,
        "balloon_surface_spl": balloon_surface_spl,
        "balloon_x": balloon_x,
        "balloon_y": balloon_y,
        "balloon_z": balloon_z,
        "isobar_angle_deg": isobar_angle_deg,
        "isobar_freq_hz": isobar_freq_hz,
        "horizontal_isobar_spl": horizontal_isobar_spl,
        "vertical_isobar_spl": vertical_isobar_spl,
        "impedance_real": raw_data["impedance_real"].astype(float, copy=False),
        "impedance_imag": raw_data["impedance_imag"].astype(float, copy=False),
        "min_db": np.array(MIN_DB, dtype=float),
        "max_db": np.array(MAX_DB, dtype=float),
    }


def save_formatted_data(input_npz: Path = INPUT_NPZ, output_npz: Path = OUTPUT_NPZ):
    raw_data = load_pressure_data(input_npz)
    formatted_bundle = build_formatted_bundle(raw_data)
    np.savez(output_npz, **formatted_bundle)
    print(f"Saved {output_npz}")


if __name__ == "__main__":
    save_formatted_data()