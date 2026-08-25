"""Export solved per-channel on-axis frequency responses."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from blab.live import LiveSolveDataset

_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def export_on_axis_text_files(
    dataset: LiveSolveDataset,
    output_target: str | Path,
    *,
    include_sum: bool = False,
) -> list[Path]:
    """Write REW-compatible frequency, SPL, and propagation-aligned phase columns.

    ``output_target`` is the exact file path for one output trace and an output
    directory for multiple traces. Requesting the synthesized sum therefore
    always uses a directory when at least one channel is present.
    """
    freqs, channel_names, spl_db, phase_deg = dataset.as_channel_on_axis_export_arrays()
    if include_sum:
        summed_spl_db, summed_phase_deg = dataset.as_summed_on_axis_export_arrays()
        channel_names = np.concatenate((np.asarray(["Sum"]), channel_names.astype(str)))
        spl_db = np.vstack((summed_spl_db, spl_db))
        phase_deg = np.vstack((summed_phase_deg, phase_deg))
    target = Path(output_target)

    if channel_names.size == 1:
        file_path = target if target.suffix else target.with_suffix(".txt")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        _write_response(file_path, freqs, spl_db[0], phase_deg[0])
        return [file_path]

    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    used_names: set[str] = set()
    for channel_index, channel_name in enumerate(channel_names):
        filename = _unique_channel_filename(str(channel_name), used_names)
        file_path = target / filename
        _write_response(file_path, freqs, spl_db[channel_index], phase_deg[channel_index])
        written.append(file_path)
    return written


def default_on_axis_filename(channel_name: str) -> str:
    """Return a filesystem-safe default filename for one channel."""
    safe_name = _safe_filename_component(channel_name)
    return f"on_axis_{safe_name}.txt"


def _write_response(file_path: Path, freqs, spl_db, phase_deg) -> None:
    with file_path.open("w", encoding="utf-8", newline="\n") as handle:
        for freq, spl, phase in zip(freqs, spl_db, phase_deg, strict=True):
            handle.write(f"{float(freq):.6f}\t{float(spl):.3f}\t{float(phase):.3f}\n")


def _unique_channel_filename(channel_name: str, used_names: set[str]) -> str:
    stem = f"on_axis_{_safe_filename_component(channel_name)}"
    candidate = f"{stem}.txt"
    suffix = 2
    while candidate.casefold() in used_names:
        candidate = f"{stem}_{suffix}.txt"
        suffix += 1
    used_names.add(candidate.casefold())
    return candidate


def _safe_filename_component(value: str) -> str:
    sanitized = _INVALID_FILENAME_CHARS.sub("_", str(value)).strip().rstrip(". ")
    return sanitized or "channel"
