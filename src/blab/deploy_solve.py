"""Qt-free preparation for Boundary Lab Deploy Level 2 solves."""

from __future__ import annotations

import io
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from blab.speaker_package import validate_speaker_package

DEPLOY_SOLVE_SCHEMA = "boundary_lab_deploy_solve"
DEPLOY_SOLVE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DeploySourcePlacement:
    position_x_m: float
    position_height_m: float
    position_z_m: float
    yaw_deg: float
    level_db: float
    delay_ms: float
    polarity: int

    @classmethod
    def from_payload(cls, raw: object) -> "DeploySourcePlacement":
        if not isinstance(raw, dict):
            raise ValueError("Deploy source must be an object.")
        polarity = int(raw.get("polarity", 1))
        if polarity not in (-1, 1):
            raise ValueError("Deploy source polarity must be -1 or +1.")
        values = cls(
            position_x_m=float(raw.get("positionX", 0.0)),
            position_height_m=float(raw.get("positionHeightM", 0.0)),
            position_z_m=float(raw.get("positionZ", 0.0)),
            yaw_deg=float(raw.get("yawDeg", 0.0)),
            level_db=float(raw.get("levelDb", 0.0)),
            delay_ms=float(raw.get("delayMs", 0.0)),
            polarity=polarity,
        )
        if not all(math.isfinite(value) for value in values.__dict__.values()):
            raise ValueError("Deploy source values must be finite.")
        return values


@dataclass(frozen=True)
class DeployObservationPlane:
    width_m: float
    depth_m: float
    near_m: float
    height_m: float
    columns: int
    rows: int

    @classmethod
    def from_payload(cls, raw: object) -> "DeployObservationPlane":
        if not isinstance(raw, dict):
            raise ValueError("Deploy observation plane must be an object.")
        value = cls(
            width_m=float(raw.get("widthM", 0.0)),
            depth_m=float(raw.get("depthM", 0.0)),
            near_m=float(raw.get("nearM", 0.0)),
            height_m=float(raw.get("heightM", 0.0)),
            columns=int(raw.get("columns", 0)),
            rows=int(raw.get("rows", 0)),
        )
        if not all(
            math.isfinite(item)
            for item in (value.width_m, value.depth_m, value.near_m, value.height_m)
        ):
            raise ValueError("Deploy observation plane values must be finite.")
        if value.width_m <= 0.0 or value.depth_m <= 0.0:
            raise ValueError("Deploy observation plane width and depth must be positive.")
        if value.columns < 2 or value.rows < 2:
            raise ValueError("Deploy observation plane must contain at least two rows and columns.")
        if value.columns * value.rows > 250_000:
            raise ValueError("Deploy observation plane exceeds the 250,000 point limit.")
        return value

    def points(self) -> np.ndarray:
        x = np.linspace(-self.width_m / 2.0, self.width_m / 2.0, self.columns, dtype=np.float32)
        z = np.linspace(self.near_m, self.near_m + self.depth_m, self.rows, dtype=np.float32)
        xx, zz = np.meshgrid(x, z, indexing="xy")
        return np.column_stack(
            (
                xx.reshape(-1),
                np.full(xx.size, self.height_m, dtype=np.float32),
                zz.reshape(-1),
            )
        ).astype(np.float32, copy=False)


def prepare_deploy_solve_request(payload: object, work_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    """Validate a renderer request and stage one fixed-source package for BEAT."""

    if not isinstance(payload, dict):
        raise ValueError("Deploy solve request must be an object.")
    package_path = Path(str(payload.get("packagePath", ""))).expanduser().resolve()
    if package_path.suffix.lower() != ".blabsp" or not package_path.is_file():
        raise ValueError("Deploy Level 2 requires an existing .blabsp package path.")

    manifest = validate_speaker_package(package_path)
    if int(manifest.get("fidelity_level", 0)) < 2:
        raise ValueError("Deploy Level 2 requires a package containing fixed distributed sources.")

    requested_frequency = float(payload.get("frequencyHz", 0.0))
    if not math.isfinite(requested_frequency) or requested_frequency <= 0.0:
        raise ValueError("Deploy solve frequency must be finite and positive.")
    frequencies = np.asarray(manifest.get("frequencies_hz", ()), dtype=np.float64)
    if frequencies.size == 0:
        raise ValueError("Speaker package contains no frequencies.")
    frequency_index = int(np.argmin(np.abs(frequencies - requested_frequency)))
    frequency_hz = float(frequencies[frequency_index])
    tolerance_hz = max(1e-4, abs(frequency_hz) * 1e-6)
    if abs(frequency_hz - requested_frequency) > tolerance_hz:
        raise ValueError("Deploy Level 2 initially requires an exact exported package frequency.")

    source = DeploySourcePlacement.from_payload(payload.get("source"))
    observation = DeployObservationPlane.from_payload(payload.get("observation"))
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)

    fixed_file = manifest.get("files", {}).get("fixed_sources", {})
    fixed_path = str(fixed_file.get("path", ""))
    geometry_path = str(fixed_file.get("geometry_mesh", ""))
    if not fixed_path or not geometry_path:
        raise ValueError("Speaker package does not declare fixed-source data and geometry.")

    with zipfile.ZipFile(package_path, "r") as archive:
        try:
            fixed_bytes = archive.read(fixed_path)
            geometry_bytes = archive.read(geometry_path)
        except KeyError as exc:
            raise ValueError(f"Speaker package is missing {exc.args[0]!r}.") from exc

    with np.load(io.BytesIO(fixed_bytes), allow_pickle=False) as fixed:
        triangles = np.asarray(fixed["triangles"], dtype=np.int64)
        points = np.asarray(fixed["points_m"], dtype=np.float64)
        pressure = np.asarray(fixed["pressure_pa"])
        normal = np.asarray(fixed["normal_derivative_pa_per_m"])
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Fixed-source points must have shape (node, 3).")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("Fixed-source triangles must have shape (face, 3).")
    if normal.ndim != 3 or normal.shape[0] != frequencies.size or normal.shape[2] != triangles.shape[0]:
        raise ValueError("Fixed-source Neumann traces do not align with package frequencies and faces.")
    if pressure.ndim != 3 or pressure.shape[0] != frequencies.size or pressure.shape[2] != points.shape[0]:
        raise ValueError("Fixed-source pressure traces do not align with package frequencies and nodes.")
    if normal.shape[1] < 1:
        raise ValueError("Fixed-source package contains no excitation ports.")
    if pressure.shape[1] != normal.shape[1]:
        raise ValueError("Fixed-source pressure and Neumann traces have different excitation counts.")

    phase = 2.0 * math.pi * frequency_hz * source.delay_ms / 1000.0
    gain = source.polarity * 10.0 ** (source.level_db / 20.0) * np.exp(1j * phase)
    q_neumann = np.asarray(normal[frequency_index, 0] * gain, dtype=np.complex64)
    reference_pressure = np.asarray(pressure[frequency_index, 0] * gain, dtype=np.complex64)
    if not np.all(np.isfinite(q_neumann)) or not np.all(np.isfinite(reference_pressure)):
        raise ValueError("Fixed-source boundary traces contain non-finite values.")

    staged_mesh = work_path / "exterior.msh"
    staged_mesh.write_bytes(geometry_bytes)
    points_m = observation.points()
    medium = manifest.get("medium", {})
    request: dict[str, Any] = {
        "schema": DEPLOY_SOLVE_SCHEMA,
        "schema_version": DEPLOY_SOLVE_SCHEMA_VERSION,
        "beat_engine_backend": str(payload.get("backend", "cuda")),
        "frequency_hz": frequency_hz,
        "mesh_file": str(staged_mesh),
        "mesh_scale_factor": 1.0,
        "source_transform": {
            "position_m": [source.position_x_m, source.position_height_m, source.position_z_m],
            "yaw_deg": source.yaw_deg,
        },
        "boundary_neumann": {
            "real": q_neumann.real.tolist(),
            "imag": q_neumann.imag.tolist(),
        },
        "reference_boundary_pressure": {
            "real": reference_pressure.real.tolist(),
            "imag": reference_pressure.imag.tolist(),
        },
        "observation_points_m": points_m.tolist(),
        "observation_shape": [observation.rows, observation.columns],
        "density_kg_per_m3": float(medium.get("density_kg_per_m3", 1.21)),
        "sound_speed_m_per_s": float(medium.get("sound_speed_m_per_s", 343.0)),
        "quadrature_order": int(payload.get("quadratureOrder", 2)),
        "singular_order": int(payload.get("singularOrder", 3)),
        "provenance": {
            "package_path": str(package_path),
            "package_name": str(manifest.get("name", package_path.stem)),
            "frequency_index": frequency_index,
            "node_count": int(points.shape[0]),
            "face_count": int(triangles.shape[0]),
            "excitation_index": 0,
        },
    }
    request_path = work_path / "request.json"
    request_path.write_text(json.dumps(request, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    return request_path, request
