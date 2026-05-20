"""Prepare spherical directivity data for 3D balloon plotting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import griddata


DEFAULT_BALLOON_GRID_THETA = 40
DEFAULT_BALLOON_GRID_PHI = 80


@dataclass(frozen=True)
class BalloonPrepConfig:
    min_db: float = -30.0
    max_db: float = 0.0
    theta_samples: int = DEFAULT_BALLOON_GRID_THETA
    phi_samples: int = DEFAULT_BALLOON_GRID_PHI


def prepare_balloon_data(
    raw_data: dict[str, np.ndarray],
    cfg: BalloonPrepConfig | None = None,
) -> dict[str, np.ndarray]:
    prep_cfg = cfg or BalloonPrepConfig()
    freq_hz = np.asarray(raw_data["freq_hz"], dtype=np.float32)
    theta_polar_rad = np.asarray(raw_data["theta_polar_rad"], dtype=np.float32)
    phi_azimuth_rad = np.asarray(raw_data["phi_azimuth_rad"], dtype=np.float32)
    spl_norm = np.clip(np.asarray(raw_data["spl_norm"], dtype=np.float32), prep_cfg.min_db, prep_cfg.max_db)

    if freq_hz.ndim != 1:
        raise ValueError("freq_hz must be a 1D array.")
    if theta_polar_rad.ndim != 1 or phi_azimuth_rad.ndim != 1:
        raise ValueError("theta_polar_rad and phi_azimuth_rad must be 1D arrays.")
    if theta_polar_rad.shape != phi_azimuth_rad.shape:
        raise ValueError("theta_polar_rad and phi_azimuth_rad must have matching shapes.")
    if spl_norm.shape != (freq_hz.size, theta_polar_rad.size):
        raise ValueError("spl_norm must have shape (len(freq_hz), len(theta_polar_rad)).")
    if prep_cfg.theta_samples < 2 or prep_cfg.phi_samples < 3:
        raise ValueError("Balloon grid resolution is too small.")

    theta_grid = np.linspace(0.0, np.pi, prep_cfg.theta_samples, dtype=np.float32)
    phi_grid = np.linspace(0.0, 2.0 * np.pi, prep_cfg.phi_samples, dtype=np.float32)
    theta_mesh, phi_mesh = np.meshgrid(theta_grid, phi_grid, indexing="ij")

    surface_shape = (freq_hz.size,) + theta_mesh.shape
    balloon_surface_spl = np.empty(surface_shape, dtype=np.float32)
    balloon_x = np.empty(surface_shape, dtype=np.float32)
    balloon_y = np.empty(surface_shape, dtype=np.float32)
    balloon_z = np.empty(surface_shape, dtype=np.float32)

    for index in range(freq_hz.size):
        grid_spl = _grid_spl_surface(
            theta_polar_rad,
            phi_azimuth_rad,
            spl_norm[index],
            theta_mesh,
            phi_mesh,
            prep_cfg.min_db,
        ).astype(np.float32, copy=False)
        x, y, z = _balloon_xyz(grid_spl, theta_mesh, phi_mesh, prep_cfg.min_db)
        balloon_surface_spl[index] = grid_spl
        balloon_x[index] = x
        balloon_y[index] = y
        balloon_z[index] = z

    return {
        "freq_hz": freq_hz,
        "theta_grid_rad": theta_mesh,
        "phi_grid_rad": phi_mesh,
        "balloon_surface_spl": balloon_surface_spl,
        "balloon_x": balloon_x,
        "balloon_y": balloon_y,
        "balloon_z": balloon_z,
        "min_db": np.asarray(prep_cfg.min_db, dtype=np.float32),
        "max_db": np.asarray(prep_cfg.max_db, dtype=np.float32),
    }


def _wrap_periodic_phi_data(
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

    return (
        np.concatenate([points_theta, theta_end, theta_start]),
        np.concatenate([points_phi, phi_end, phi_start]),
        np.concatenate([points_spl, spl_end, spl_start]),
    )


def _grid_spl_surface(
    points_theta: np.ndarray,
    points_phi: np.ndarray,
    points_spl: np.ndarray,
    theta_mesh: np.ndarray,
    phi_mesh: np.ndarray,
    fallback_db: float,
) -> np.ndarray:
    wrapped_theta, wrapped_phi, wrapped_spl = _wrap_periodic_phi_data(points_theta, points_phi, points_spl)
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

    return np.nan_to_num(grid_spl, nan=fallback_db)


def _balloon_xyz(
    grid_spl: np.ndarray,
    theta_mesh: np.ndarray,
    phi_mesh: np.ndarray,
    min_db: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius = np.maximum(grid_spl - float(min_db), 0.0)
    x = radius * np.sin(theta_mesh) * np.cos(phi_mesh)
    y = radius * np.sin(theta_mesh) * np.sin(phi_mesh)
    z = radius * np.cos(theta_mesh)
    return x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)
