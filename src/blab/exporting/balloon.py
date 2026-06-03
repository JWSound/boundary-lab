"""Export prepared balloon surfaces for external visualization tools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


BALLOON_EXPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BalloonExportResult:
    output_dir: Path
    files: tuple[Path, ...]
    frequency_count: int
    point_count: int
    quad_count: int


def export_balloon_data(
    prepared: dict[str, np.ndarray],
    output_dir: str | Path,
) -> BalloonExportResult:
    """Write a compact fixed-topology balloon export folder.

    The flattened point order is row-major over ``(theta, phi)``. Per-frequency
    arrays keep their grid dimensions so external tools can consume either
    textures/arrays or the ready-made flattened vertex positions.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    freq_hz = _required_array(prepared, "freq_hz", np.float32)
    theta_grid = _required_array(prepared, "theta_grid_rad", np.float32)
    phi_grid = _required_array(prepared, "phi_grid_rad", np.float32)
    spl_db = _required_array(prepared, "balloon_surface_spl", np.float32)
    x = _required_array(prepared, "balloon_x", np.float32)
    y = _required_array(prepared, "balloon_y", np.float32)
    z = _required_array(prepared, "balloon_z", np.float32)
    min_db = float(np.asarray(prepared["min_db"], dtype=np.float32))
    max_db = float(np.asarray(prepared["max_db"], dtype=np.float32))

    _validate_prepared_shapes(freq_hz, theta_grid, phi_grid, spl_db, x, y, z)

    directions = _direction_vectors(theta_grid, phi_grid)
    quad_indices = _structured_grid_quads(theta_grid.shape)
    radius_norm = _normalized_radius(spl_db, min_db, max_db)
    positions = np.stack((x, y, z), axis=-1).astype(np.float32, copy=False)

    metadata = {
        "schema": "boundary-lab-balloon-export",
        "schema_version": BALLOON_EXPORT_SCHEMA_VERSION,
        "description": "Fixed-topology Boundary Lab balloon data for realtime displacement and color mapping.",
        "frequency_count": int(freq_hz.size),
        "theta_samples": int(theta_grid.shape[0]),
        "phi_samples": int(theta_grid.shape[1]),
        "point_count": int(theta_grid.size),
        "quad_count": int(quad_indices.shape[0]),
        "point_order": "row-major theta, then phi; flat_index = theta_index * phi_samples + phi_index",
        "coordinate_system": {
            "x": "horizontal right for phi=0",
            "y": "vertical-side axis for phi=90deg",
            "z": "on-axis forward",
            "theta_rad": "polar angle from +z",
            "phi_rad": "azimuth angle around +z from +x toward +y",
        },
        "db_range": {"min_db": min_db, "max_db": max_db},
        "radius_mapping": {
            "radius_db_units": "max(spl_db - min_db, 0)",
            "radius_norm": "clip((spl_db - min_db) / (max_db - min_db), 0, 1)",
        },
        "files": {
            "topology": "topology.npz",
            "spl_db": "spl_db.npy",
            "radius_norm": "radius_norm.npy",
            "positions_xyz": "positions_xyz.npy",
        },
        "array_shapes": {
            "spl_db": ["frequency", "theta", "phi"],
            "radius_norm": ["frequency", "theta", "phi"],
            "positions_xyz": ["frequency", "theta", "phi", "xyz"],
            "directions_xyz": ["point", "xyz"],
            "quad_indices": ["quad", "corner"],
        },
    }

    metadata_path = output_path / "metadata.json"
    topology_path = output_path / "topology.npz"
    spl_path = output_path / "spl_db.npy"
    radius_path = output_path / "radius_norm.npy"
    positions_path = output_path / "positions_xyz.npy"

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    np.savez_compressed(
        topology_path,
        freq_hz=freq_hz.astype(np.float32, copy=False),
        theta_grid_rad=theta_grid.astype(np.float32, copy=False),
        phi_grid_rad=phi_grid.astype(np.float32, copy=False),
        directions_xyz=directions,
        quad_indices=quad_indices,
    )
    np.save(spl_path, spl_db.astype(np.float32, copy=False))
    np.save(radius_path, radius_norm)
    np.save(positions_path, positions)

    return BalloonExportResult(
        output_dir=output_path,
        files=(metadata_path, topology_path, spl_path, radius_path, positions_path),
        frequency_count=int(freq_hz.size),
        point_count=int(theta_grid.size),
        quad_count=int(quad_indices.shape[0]),
    )


def _required_array(prepared: dict[str, Any], key: str, dtype) -> np.ndarray:
    if key not in prepared:
        raise ValueError(f"Prepared balloon data is missing {key!r}.")
    return np.asarray(prepared[key], dtype=dtype)


def _validate_prepared_shapes(
    freq_hz: np.ndarray,
    theta_grid: np.ndarray,
    phi_grid: np.ndarray,
    spl_db: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> None:
    if freq_hz.ndim != 1:
        raise ValueError("freq_hz must be a 1D array.")
    if theta_grid.ndim != 2 or phi_grid.ndim != 2 or theta_grid.shape != phi_grid.shape:
        raise ValueError("theta_grid_rad and phi_grid_rad must be matching 2D arrays.")
    expected_surface_shape = (freq_hz.size,) + theta_grid.shape
    for name, array in (("balloon_surface_spl", spl_db), ("balloon_x", x), ("balloon_y", y), ("balloon_z", z)):
        if array.shape != expected_surface_shape:
            raise ValueError(f"{name} must have shape {expected_surface_shape}.")


def _direction_vectors(theta_grid: np.ndarray, phi_grid: np.ndarray) -> np.ndarray:
    directions = np.stack(
        (
            np.sin(theta_grid) * np.cos(phi_grid),
            np.sin(theta_grid) * np.sin(phi_grid),
            np.cos(theta_grid),
        ),
        axis=-1,
    )
    return directions.reshape(-1, 3).astype(np.float32, copy=False)


def _structured_grid_quads(surface_shape: tuple[int, int]) -> np.ndarray:
    theta_count, phi_count = surface_shape
    quads = []
    for theta_index in range(theta_count - 1):
        row = theta_index * phi_count
        next_row = (theta_index + 1) * phi_count
        for phi_index in range(phi_count - 1):
            quads.append(
                (
                    row + phi_index,
                    row + phi_index + 1,
                    next_row + phi_index + 1,
                    next_row + phi_index,
                )
            )
    return np.asarray(quads, dtype=np.int32)


def _normalized_radius(spl_db: np.ndarray, min_db: float, max_db: float) -> np.ndarray:
    span = max(float(max_db) - float(min_db), 1e-6)
    return np.clip((spl_db.astype(np.float32, copy=False) - float(min_db)) / span, 0.0, 1.0).astype(
        np.float32,
        copy=False,
    )
