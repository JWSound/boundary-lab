"""Generic tabular exports for frequency-domain plot data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TraceQuantity:
    """One value type arranged as ``(trace, frequency)``."""

    label: str
    unit: str
    values: np.ndarray


def export_frequency_trace_table(
    output_path: str | Path,
    *,
    title: str,
    frequency_hz: np.ndarray,
    trace_names: np.ndarray,
    quantities: tuple[TraceQuantity, ...],
) -> Path:
    """Write a self-describing tab-separated table for one or more traces."""

    path = Path(output_path)
    if path.suffix == "":
        path = path.with_suffix(".txt")
    path.parent.mkdir(parents=True, exist_ok=True)

    frequencies = np.asarray(frequency_hz, dtype=float)
    names = np.asarray(trace_names).astype(str)
    if frequencies.ndim != 1:
        raise ValueError("Export frequencies must be a one-dimensional array.")
    if names.ndim != 1 or not names.size:
        raise ValueError("At least one export trace is required.")
    if not quantities:
        raise ValueError("At least one exported quantity is required.")

    normalized: list[tuple[TraceQuantity, np.ndarray]] = []
    expected_shape = (names.size, frequencies.size)
    for quantity in quantities:
        values = np.asarray(quantity.values, dtype=float)
        if values.shape != expected_shape:
            raise ValueError(
                f"{quantity.label} values have shape {values.shape}, expected {expected_shape}."
            )
        normalized.append((quantity, values))

    headers = ["Frequency (Hz)"]
    columns: list[np.ndarray] = [frequencies]
    for trace_index, raw_name in enumerate(names.tolist()):
        name = _clean_header_text(raw_name)
        for quantity, values in normalized:
            label = _clean_header_text(quantity.label)
            unit = _clean_header_text(quantity.unit)
            suffix = f" ({unit})" if unit else ""
            headers.append(f"{name} {label}{suffix}")
            columns.append(values[trace_index])

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# Boundary Lab - {_clean_header_text(title)}\n")
        handle.write("# " + "\t".join(headers) + "\n")
        for row in zip(*columns, strict=True):
            handle.write("\t".join(_format_number(value) for value in row) + "\n")
    return path


def _clean_header_text(value: object) -> str:
    return " ".join(str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").split())


def _format_number(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return f"{float(value):.9g}"


__all__ = ["TraceQuantity", "export_frequency_trace_table"]
