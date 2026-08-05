"""Export direct triangular balloon surfaces for external visualization tools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

BALLOON_EXPORT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class BalloonExportResult:
    output_dir: Path
    files: tuple[Path, ...]
    frequency_count: int
    point_count: int
    triangle_count: int


def export_balloon_data(
    prepared: dict[str, np.ndarray],
    output_dir: str | Path,
) -> BalloonExportResult:
    """Write solved Fibonacci vertices with one shared triangular topology."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    freq_hz = _required_array(prepared, "freq_hz", np.float32)
    theta = _required_array(prepared, "theta_polar_rad", np.float32)
    phi = _required_array(prepared, "phi_azimuth_rad", np.float32)
    directions = _required_array(prepared, "directions_xyz", np.float32)
    triangle_indices = _required_array(prepared, "triangle_indices", np.int32)
    spl_db = _required_array(prepared, "balloon_surface_spl", np.float32)
    min_db = float(np.asarray(prepared["min_db"], dtype=np.float32))
    max_db = float(np.asarray(prepared["max_db"], dtype=np.float32))

    _validate_prepared_shapes(freq_hz, theta, phi, directions, triangle_indices, spl_db)
    radius_norm = _normalized_radius(spl_db, min_db, max_db)

    metadata = {
        "schema": "boundary-lab-balloon-export",
        "schema_version": BALLOON_EXPORT_SCHEMA_VERSION,
        "description": "Direct Fibonacci-sampled Boundary Lab balloon data with shared triangular topology.",
        "frequency_count": int(freq_hz.size),
        "point_count": int(directions.shape[0]),
        "triangle_count": int(triangle_indices.shape[0]),
        "point_order": "original solver Fibonacci sample order",
        "coordinate_system": {
            "x": "horizontal right for phi=0",
            "y": "vertical-side axis for phi=90deg",
            "z": "on-axis forward",
            "theta_rad": "polar angle from +z",
            "phi_rad": "azimuth around +z from +x toward +y",
        },
        "db_range": {"min_db": min_db, "max_db": max_db},
        "radius_mapping": {
            "radius_db_units": "max(spl_db - min_db, 0)",
            "radius_norm": "clip((spl_db - min_db) / (max_db - min_db), 0, 1)",
            "positions_xyz": "directions_xyz * radius_db_units[..., newaxis]",
        },
        "files": {
            "topology": "topology.npz",
            "spl_db": "spl_db.npy",
            "radius_norm": "radius_norm.npy",
        },
        "array_shapes": {
            "spl_db": ["frequency", "point"],
            "radius_norm": ["frequency", "point"],
            "directions_xyz": ["point", "xyz"],
            "triangle_indices": ["triangle", "corner"],
        },
    }

    metadata_path = output_path / "metadata.json"
    topology_path = output_path / "topology.npz"
    spl_path = output_path / "spl_db.npy"
    radius_path = output_path / "radius_norm.npy"

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    topology_arrays: dict[str, np.ndarray] = {
        "freq_hz": freq_hz.astype(np.float32, copy=False),
        "theta_polar_rad": theta.astype(np.float32, copy=False),
        "phi_azimuth_rad": phi.astype(np.float32, copy=False),
        "directions_xyz": directions.astype(np.float32, copy=False),
        "triangle_indices": triangle_indices.astype(np.int32, copy=False),
    }
    if "r_distance_m" in prepared:
        topology_arrays["r_distance_m"] = np.asarray(prepared["r_distance_m"], dtype=np.float32)
    np.savez_compressed(topology_path, **topology_arrays)
    np.save(spl_path, spl_db.astype(np.float32, copy=False))
    np.save(radius_path, radius_norm)

    return BalloonExportResult(
        output_dir=output_path,
        files=(metadata_path, topology_path, spl_path, radius_path),
        frequency_count=int(freq_hz.size),
        point_count=int(directions.shape[0]),
        triangle_count=int(triangle_indices.shape[0]),
    )


def _required_array(prepared: dict[str, Any], key: str, dtype) -> np.ndarray:
    if key not in prepared:
        raise ValueError(f"Prepared balloon data is missing {key!r}.")
    return np.asarray(prepared[key], dtype=dtype)


def _validate_prepared_shapes(
    freq_hz: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    directions: np.ndarray,
    triangle_indices: np.ndarray,
    spl_db: np.ndarray,
) -> None:
    if freq_hz.ndim != 1:
        raise ValueError("freq_hz must be a 1D array.")
    if theta.ndim != 1 or phi.ndim != 1 or theta.shape != phi.shape:
        raise ValueError("theta_polar_rad and phi_azimuth_rad must be matching 1D arrays.")
    point_count = theta.size
    if directions.shape != (point_count, 3):
        raise ValueError(f"directions_xyz must have shape ({point_count}, 3).")
    if triangle_indices.ndim != 2 or triangle_indices.shape[1] != 3:
        raise ValueError("triangle_indices must have shape (triangle, 3).")
    if triangle_indices.size and (np.min(triangle_indices) < 0 or np.max(triangle_indices) >= point_count):
        raise ValueError("triangle_indices references a point outside directions_xyz.")
    if spl_db.shape != (freq_hz.size, point_count):
        raise ValueError(f"balloon_surface_spl must have shape ({freq_hz.size}, {point_count}).")


def _normalized_radius(spl_db: np.ndarray, min_db: float, max_db: float) -> np.ndarray:
    span = max(float(max_db) - float(min_db), 1e-6)
    return np.clip((spl_db.astype(np.float32, copy=False) - float(min_db)) / span, 0.0, 1.0).astype(
        np.float32,
        copy=False,
    )
