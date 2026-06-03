"""Export solved polar response data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from blab.live import LiveSolveDataset


def export_polar_text_files(
    dataset: LiveSolveDataset,
    output_dir: str | Path,
    *,
    include_phase: bool = True,
    relative_phase: bool = True,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    freqs, angles, horizontal, vertical = dataset.as_polar_export_arrays()
    horizontal_phase = None
    vertical_phase = None
    if include_phase:
        _, _, horizontal_complex, vertical_complex = dataset.as_complex_polar_export_arrays()
        reference_index = int(np.argmin(np.abs(angles)))
        horizontal_phase = _polar_phase_deg(horizontal_complex, reference_index=reference_index, relative=relative_phase)
        vertical_phase = _polar_phase_deg(vertical_complex, reference_index=reference_index, relative=relative_phase)

    written = []
    for prefix, matrix, phase_matrix in (
        ("H", horizontal, horizontal_phase),
        ("V", vertical, vertical_phase),
    ):
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


def _polar_phase_deg(pressure: np.ndarray, *, reference_index: int, relative: bool) -> np.ndarray:
    pressure = np.asarray(pressure, dtype=np.complex64)
    if not relative:
        return np.rad2deg(np.angle(pressure)).astype(np.float32, copy=False)

    reference = pressure[:, int(reference_index)]
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_pressure = np.where(
            np.abs(reference[:, np.newaxis]) > 1e-12,
            pressure / reference[:, np.newaxis],
            pressure,
        )
    return np.rad2deg(np.angle(relative_pressure)).astype(np.float32, copy=False)


def _format_angle_for_filename(angle_deg: float) -> str:
    if np.isclose(angle_deg, round(angle_deg)):
        return str(int(round(angle_deg)))
    return f"{angle_deg:g}"
