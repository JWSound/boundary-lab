"""In-memory helpers for live GUI solving and plotting."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from blab.config import SimulationConfig
from blab.postprocess import PrepConfig, prepare_visualization_data_from_arrays
from blab.solvers.base import FrequencyResult, SolveRequest
from blab.solvers.bempp_local import BemppLocalBackend


@dataclass
class LiveSolveDataset:
    polar_angle_deg: np.ndarray
    radiator_names: np.ndarray = field(default_factory=lambda: np.asarray(["Radiator"]))
    sphere_r_distance_m: np.ndarray | None = None
    sphere_theta_polar_rad: np.ndarray | None = None
    sphere_phi_azimuth_rad: np.ndarray | None = None
    results: dict[float, FrequencyResult] = field(default_factory=dict)

    def add(self, result: FrequencyResult) -> None:
        self.results[float(result.freq_hz)] = result

    def ordered_results(self) -> list[FrequencyResult]:
        return [self.results[key] for key in sorted(self.results)]

    @property
    def solved_count(self) -> int:
        return len(self.results)

    @property
    def solved_frequencies(self) -> np.ndarray:
        return np.asarray([result.freq_hz for result in self.ordered_results()], dtype=np.float32)

    def as_polar_export_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.results:
            raise ValueError("No solved polar data available.")

        ordered = self.ordered_results()
        freqs = np.asarray([item.freq_hz for item in ordered], dtype=np.float32)
        horizontal = np.vstack([item.horizontal_spl_norm_db for item in ordered]).astype(np.float32, copy=False)
        vertical = np.vstack([item.vertical_spl_norm_db for item in ordered]).astype(np.float32, copy=False)
        return freqs, self.polar_angle_deg.astype(np.float32, copy=False), horizontal, vertical

    def as_raw_polar_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.results:
            raise ValueError("No solved polar data available.")

        ordered = self.ordered_results()
        freqs = np.asarray([item.freq_hz for item in ordered], dtype=np.float32)
        horizontal = np.vstack(
            [
                item.horizontal_spl_db
                if item.horizontal_spl_db is not None
                else item.horizontal_spl_norm_db
                for item in ordered
            ]
        ).astype(np.float32, copy=False)
        vertical = np.vstack(
            [
                item.vertical_spl_db
                if item.vertical_spl_db is not None
                else item.vertical_spl_norm_db
                for item in ordered
            ]
        ).astype(np.float32, copy=False)
        return freqs, self.polar_angle_deg.astype(np.float32, copy=False), horizontal, vertical

    def as_visualization_dataset(self, cfg: PrepConfig | None = None) -> dict[str, np.ndarray] | None:
        if not self.results:
            return None

        prep_cfg = cfg or PrepConfig()
        freqs, angles, horizontal, vertical = self.as_polar_export_arrays()
        _, _, raw_horizontal, raw_vertical = self.as_raw_polar_arrays()
        ordered = self.ordered_results()
        impedance = np.stack([item.impedance for item in ordered], axis=1)

        return prepare_visualization_data_from_arrays(
            freq_hz=freqs,
            polar_angle_deg=angles,
            horizontal_spl_norm_db=horizontal,
            vertical_spl_norm_db=vertical,
            horizontal_spl_db=raw_horizontal,
            vertical_spl_db=raw_vertical,
            impedance_freq_hz=freqs,
            impedance_radiator_names=self.radiator_names,
            impedance_real=impedance[:, :, 0],
            impedance_imag=impedance[:, :, 1],
            cfg=prep_cfg,
        )

    def as_balloon_raw_bundle(self) -> dict[str, np.ndarray] | None:
        if (
            not self.results
            or self.sphere_r_distance_m is None
            or self.sphere_theta_polar_rad is None
            or self.sphere_phi_azimuth_rad is None
        ):
            return None

        ordered = self.ordered_results()
        freqs = np.asarray([item.freq_hz for item in ordered], dtype=np.float32)
        if any(item.sphere_spl_norm_db is None for item in ordered):
            return None

        return {
            "freq_hz": freqs,
            "r_distance_m": np.asarray(self.sphere_r_distance_m, dtype=np.float32),
            "theta_polar_rad": np.asarray(self.sphere_theta_polar_rad, dtype=np.float32),
            "phi_azimuth_rad": np.asarray(self.sphere_phi_azimuth_rad, dtype=np.float32),
            "spl_norm": np.vstack([item.sphere_spl_norm_db for item in ordered]).astype(np.float32, copy=False),
        }


def export_polar_text_files(dataset: LiveSolveDataset, output_dir: str | Path) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    freqs, angles, horizontal, vertical = dataset.as_polar_export_arrays()
    written = []
    for prefix, matrix in (("H", horizontal), ("V", vertical)):
        for angle_index, angle in enumerate(angles):
            file_path = output_path / f"{prefix} {_format_angle_for_filename(float(angle))}.txt"
            with file_path.open("w", encoding="utf-8", newline="\n") as handle:
                for freq, spl in zip(freqs, matrix[:, angle_index]):
                    handle.write(f"{float(freq):.6f}\t{float(spl):.3f}\n")
            written.append(file_path)
    return written


def _format_angle_for_filename(angle_deg: float) -> str:
    if np.isclose(angle_deg, round(angle_deg)):
        return str(int(round(angle_deg)))
    return f"{angle_deg:g}"


def build_log_frequencies(freq_min: float, freq_max: float, freq_count: int) -> np.ndarray:
    if freq_min <= 0 or freq_max <= 0:
        raise ValueError("Frequency limits must be positive for log spacing.")
    if freq_max < freq_min:
        raise ValueError("freq_max must be greater than or equal to freq_min.")
    if freq_count < 1:
        raise ValueError("freq_count must be at least 1.")
    if freq_count == 1:
        return np.asarray([freq_min], dtype=np.float32)
    return np.logspace(np.log10(freq_min), np.log10(freq_max), freq_count).astype(np.float32)


def order_frequencies_for_live_plotting(
    frequencies: Iterable[float],
    *,
    vdc_base: int = 2,
) -> np.ndarray:
    freqs = np.asarray(list(frequencies), dtype=np.float32)
    if freqs.size <= 2:
        return freqs

    sorted_freqs = np.unique(np.sort(freqs))
    endpoint_indices = [0, sorted_freqs.size - 1]
    interior_indices = _van_der_corput_index_order(sorted_freqs.size - 2, base=vdc_base) + 1
    ordered_indices = np.concatenate([endpoint_indices, interior_indices])
    return sorted_freqs[ordered_indices]


def _van_der_corput_index_order(count: int, *, base: int = 2) -> np.ndarray:
    if count <= 0:
        return np.asarray([], dtype=np.int64)
    if base < 2:
        raise ValueError("Van der Corput base must be >= 2.")

    sequence = np.asarray([_van_der_corput(i, base=base) for i in range(1, count + 1)])
    return np.argsort(sequence, kind="stable").astype(np.int64, copy=False)


def _van_der_corput(index: int, *, base: int = 2) -> float:
    value = 0.0
    denominator = float(base)
    while index:
        index, remainder = divmod(index, base)
        value += remainder / denominator
        denominator *= base
    return value


def split_frequency_order_for_workers(frequencies: Iterable[float], worker_count: int) -> list[np.ndarray]:
    freqs = np.asarray(list(frequencies), dtype=np.float32)
    if worker_count < 1:
        raise ValueError("worker_count must be >= 1.")
    worker_count = min(worker_count, max(1, freqs.size))
    return [freqs[index::worker_count] for index in range(worker_count) if freqs[index::worker_count].size]


def solve_frequency_worker_process(config: SimulationConfig, frequencies, stop_event, output_queue, worker_id: int) -> None:
    try:
        t_start = time.perf_counter()
        live_solver = LiveSolver(config)
        output_queue.put(
            (
                "initialized",
                worker_id,
                (
                    live_solver.polar_angle_deg,
                    live_solver.radiator_names,
                    live_solver.sphere_metadata,
                    time.perf_counter() - t_start,
                ),
            )
        )
        for result in live_solver.solve_stream(
            frequencies,
            stop_requested=stop_event.is_set,
        ):
            output_queue.put(("result", worker_id, result))
    except Exception as exc:
        output_queue.put(("error", worker_id, str(exc)))
    finally:
        output_queue.put(("done", worker_id, None))


class LiveSolver:
    """Thin warm-solver facade for GUI code."""

    def __init__(self, config: SimulationConfig):
        self._frequencies = np.asarray([], dtype=np.float32)
        self.session = BemppLocalBackend().create_session(SolveRequest(config, self._frequencies))

    @property
    def polar_angle_deg(self) -> np.ndarray:
        return self.session.metadata.polar_angle_deg

    @property
    def radiator_names(self) -> np.ndarray:
        return self.session.metadata.radiator_names

    @property
    def sphere_metadata(self) -> dict[str, np.ndarray] | None:
        return self.session.metadata.sphere_metadata

    def solve_stream(
        self,
        frequencies: Iterable[float],
        *,
        stop_requested: Callable[[], bool] | None = None,
    ):
        self.session.request = SolveRequest(self.session.request.config, np.asarray(frequencies, dtype=np.float32))
        yield from self.session.solve_stream(stop_requested=stop_requested)
