"""Export solved polar response data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from blab.channel_synthesis import complex_reference_pressure
from blab.live import LiveSolveDataset
from blab.phasor import solver_phase_deg


def export_polar_text_files(
    dataset: LiveSolveDataset,
    output_dir: str | Path,
    *,
    include_phase: bool = True,
    relative_phase: bool = True,
    planes: tuple[str, ...] = ("H", "V"),
    reference_angles_deg: dict[str, float] | None = None,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    freqs, angles, horizontal, vertical = dataset.as_polar_export_arrays()
    horizontal_phase = None
    vertical_phase = None
    requested_planes = tuple(str(value).upper() for value in planes)
    if not requested_planes or any(value not in {"H", "V"} for value in requested_planes):
        raise ValueError("Polar export planes must contain H, V, or both.")
    if include_phase and dataset.supports_channel_resynthesis:
        _, _, horizontal_complex, vertical_complex = dataset.as_complex_polar_export_arrays()
        horizontal_phase = _polar_phase_deg(
            horizontal_complex,
            angles,
            reference_angle_deg=_plane_reference_angle(reference_angles_deg, "H"),
            relative=relative_phase,
        )
        vertical_phase = _polar_phase_deg(
            vertical_complex,
            angles,
            reference_angle_deg=_plane_reference_angle(reference_angles_deg, "V"),
            relative=relative_phase,
        )

    written = []
    for prefix, matrix, phase_matrix in (
        ("H", horizontal, horizontal_phase),
        ("V", vertical, vertical_phase),
    ):
        if prefix not in requested_planes:
            continue
        if reference_angles_deg is not None:
            matrix = _relative_magnitude_db(
                matrix,
                angles,
                _plane_reference_angle(reference_angles_deg, prefix),
            )
        for angle_index, angle in enumerate(angles):
            file_path = output_path / f"{prefix} {_format_angle_for_filename(float(angle))}.txt"
            with file_path.open("w", encoding="utf-8", newline="\n") as handle:
                if phase_matrix is None:
                    for freq, spl in zip(freqs, matrix[:, angle_index]):
                        handle.write(f"{float(freq):.6f}\t{float(spl):.3f}\n")
                else:
                    for freq, spl, phase in zip(freqs, matrix[:, angle_index], phase_matrix[:, angle_index]):
                        handle.write(f"{float(freq):.6f}\t{float(spl):.3f}\t{float(phase):.3f}\n")
            written.append(file_path)
    return written


def _polar_phase_deg(
    pressure: np.ndarray,
    angles_deg: np.ndarray,
    *,
    reference_angle_deg: float,
    relative: bool,
) -> np.ndarray:
    pressure = np.asarray(pressure, dtype=np.complex64)
    if not relative:
        return solver_phase_deg(pressure)

    reference = np.asarray(
        [complex_reference_pressure(row, angles_deg, reference_angle_deg) for row in pressure],
        dtype=np.complex64,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_pressure = np.where(
            np.abs(reference[:, np.newaxis]) > 1e-12,
            pressure / reference[:, np.newaxis],
            pressure,
        )
    return solver_phase_deg(relative_pressure)


def _relative_magnitude_db(
    values_db: np.ndarray,
    angles_deg: np.ndarray,
    reference_angle_deg: float,
) -> np.ndarray:
    values = np.asarray(values_db, dtype=np.float32)
    reference = np.asarray(
        [np.interp(float(reference_angle_deg), angles_deg.astype(float), row.astype(float)) for row in values],
        dtype=np.float32,
    )
    return values - reference[:, np.newaxis]


def _plane_reference_angle(reference_angles_deg: dict[str, float] | None, plane: str) -> float:
    if reference_angles_deg is None:
        return 0.0
    return float(reference_angles_deg.get(plane, 0.0))


def _format_angle_for_filename(angle_deg: float) -> str:
    if np.isclose(angle_deg, round(angle_deg)):
        return str(int(round(angle_deg)))
    return f"{angle_deg:g}"
