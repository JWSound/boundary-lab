"""In-memory helpers for live GUI solving and plotting."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from blab.channel_synthesis import (
    channel_drive,
    complex_reference_pressure,
    flat_target_corrections,
    pressure_to_spl,
    synthesize_channel_basis_spl,
)
from blab.config import ChannelConfig, SimulationConfig
from blab.postprocess import PrepConfig, prepare_visualization_data_from_arrays
from blab.solvers.base import FrequencyResult, SolveRequest


@dataclass
class LiveSolveDataset:
    polar_angle_deg: np.ndarray
    radiator_names: np.ndarray = field(default_factory=lambda: np.asarray(["Radiator"]))
    channel_configs: tuple[ChannelConfig, ...] = ()
    flat_target_normalization_enabled: bool = True
    flat_target_reference_angle_deg: float = 0.0
    sphere_r_distance_m: np.ndarray | None = None
    sphere_theta_polar_rad: np.ndarray | None = None
    sphere_phi_azimuth_rad: np.ndarray | None = None
    results: dict[float, FrequencyResult] = field(default_factory=dict)

    def add(self, result: FrequencyResult) -> None:
        self.results[float(result.freq_hz)] = result

    def set_channel_synthesis(
        self,
        channels: tuple[ChannelConfig, ...],
        *,
        flat_target_reference_angle_deg: float | None = None,
    ) -> None:
        self.channel_configs = tuple(channels)
        if flat_target_reference_angle_deg is not None:
            self.flat_target_reference_angle_deg = float(flat_target_reference_angle_deg)

    def ordered_results(self) -> list[FrequencyResult]:
        return [self.results[key] for key in sorted(self.results)]

    @property
    def solved_count(self) -> int:
        return len(self.results)

    @property
    def solved_frequencies(self) -> np.ndarray:
        return np.asarray([result.freq_hz for result in self.ordered_results()], dtype=np.float32)

    @property
    def supports_channel_resynthesis(self) -> bool:
        return bool(self.results) and all(result.has_channel_basis for result in self.results.values())

    def as_polar_export_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.results:
            raise ValueError("No solved polar data available.")

        ordered = self.ordered_results()
        freqs = np.asarray([item.freq_hz for item in ordered], dtype=np.float32)
        horizontal = np.vstack([self._synthesized_arrays(item)[0] for item in ordered]).astype(np.float32, copy=False)
        vertical = np.vstack([self._synthesized_arrays(item)[1] for item in ordered]).astype(np.float32, copy=False)
        return freqs, self.polar_angle_deg.astype(np.float32, copy=False), horizontal, vertical

    def as_raw_polar_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.results:
            raise ValueError("No solved polar data available.")

        ordered = self.ordered_results()
        freqs = np.asarray([item.freq_hz for item in ordered], dtype=np.float32)
        horizontal = np.vstack([self._synthesized_arrays(item)[2] for item in ordered]).astype(np.float32, copy=False)
        vertical = np.vstack([self._synthesized_arrays(item)[3] for item in ordered]).astype(np.float32, copy=False)
        return freqs, self.polar_angle_deg.astype(np.float32, copy=False), horizontal, vertical

    def as_complex_polar_export_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.results:
            raise ValueError("No solved polar data available.")
        if not self.supports_channel_resynthesis:
            raise ValueError("Phase export requires channel-basis pressure data.")

        ordered = self.ordered_results()
        freqs = np.asarray([item.freq_hz for item in ordered], dtype=np.float32)
        horizontal = np.vstack(
            [self._synthesized_complex_pressures(item)[0] for item in ordered]
        ).astype(np.complex64, copy=False)
        vertical = np.vstack(
            [self._synthesized_complex_pressures(item)[1] for item in ordered]
        ).astype(np.complex64, copy=False)
        return freqs, self.polar_angle_deg.astype(np.float32, copy=False), horizontal, vertical

    def as_visualization_dataset(self, cfg: PrepConfig | None = None) -> dict[str, np.ndarray] | None:
        if not self.results:
            return None

        prep_cfg = cfg or PrepConfig()
        self.set_channel_synthesis(
            self.channel_configs,
            flat_target_reference_angle_deg=prep_cfg.hor_ref_angle,
        )
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
        ) | self._channel_on_axis_dataset(freqs)

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
        if any(self._synthesized_sphere(item) is None for item in ordered):
            return None

        return {
            "freq_hz": freqs,
            "r_distance_m": np.asarray(self.sphere_r_distance_m, dtype=np.float32),
            "theta_polar_rad": np.asarray(self.sphere_theta_polar_rad, dtype=np.float32),
            "phi_azimuth_rad": np.asarray(self.sphere_phi_azimuth_rad, dtype=np.float32),
            "spl_norm": np.vstack([self._synthesized_sphere(item) for item in ordered]).astype(np.float32, copy=False),
        }

    def _synthesized_arrays(
        self,
        result: FrequencyResult,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if result.has_channel_basis:
            synthesized = synthesize_channel_basis_spl(
                freq_hz=float(result.freq_hz),
                polar_angle_deg=self.polar_angle_deg,
                channel_names=result.channel_names,
                horizontal_pressure=result.horizontal_pressure,
                vertical_pressure=result.vertical_pressure,
                sphere_pressure=result.sphere_pressure,
                channel_configs=self.channel_configs,
                flat_target_reference_angle_deg=self.flat_target_reference_angle_deg,
                flat_target_enabled=self.flat_target_normalization_enabled,
            )
            return (
                synthesized["horizontal_spl_norm_db"],
                synthesized["vertical_spl_norm_db"],
                synthesized["horizontal_spl_db"],
                synthesized["vertical_spl_db"],
            )

        return (
            result.horizontal_spl_norm_db,
            result.vertical_spl_norm_db,
            result.horizontal_spl_db if result.horizontal_spl_db is not None else result.horizontal_spl_norm_db,
            result.vertical_spl_db if result.vertical_spl_db is not None else result.vertical_spl_norm_db,
        )

    def _synthesized_complex_pressures(self, result: FrequencyResult) -> tuple[np.ndarray, np.ndarray]:
        if (
            result.channel_names is None
            or result.horizontal_pressure is None
            or result.vertical_pressure is None
        ):
            raise ValueError("Phase export requires channel-basis pressure data.")

        weights = self._channel_basis_weights(result)
        horizontal = np.sum(np.asarray(result.horizontal_pressure) * weights[:, np.newaxis], axis=0)
        vertical = np.sum(np.asarray(result.vertical_pressure) * weights[:, np.newaxis], axis=0)
        return (
            horizontal.astype(np.complex64, copy=False),
            vertical.astype(np.complex64, copy=False),
        )

    def _channel_basis_weights(self, result: FrequencyResult) -> np.ndarray:
        if result.channel_names is None or result.horizontal_pressure is None:
            raise ValueError("Channel-basis pressure data is unavailable.")

        channel_configs_by_name = {channel.name: channel for channel in self.channel_configs}
        angles = np.asarray(self.polar_angle_deg, dtype=np.float32)
        corrections = flat_target_corrections(
            result.horizontal_pressure,
            angles,
            self.flat_target_reference_angle_deg,
            enabled=self.flat_target_normalization_enabled,
        )
        return np.asarray(
            [
                channel_drive(
                    channel_configs_by_name.get(str(channel_name), ChannelConfig(name=str(channel_name))),
                    float(result.freq_hz),
                )
                * float(corrections[index])
                for index, channel_name in enumerate(np.asarray(result.channel_names).tolist())
            ],
            dtype=np.complex64,
        )

    def _synthesized_sphere(self, result: FrequencyResult) -> np.ndarray | None:
        if result.has_channel_basis and result.sphere_pressure is not None:
            synthesized = synthesize_channel_basis_spl(
                freq_hz=float(result.freq_hz),
                polar_angle_deg=self.polar_angle_deg,
                channel_names=result.channel_names,
                horizontal_pressure=result.horizontal_pressure,
                vertical_pressure=result.vertical_pressure,
                sphere_pressure=result.sphere_pressure,
                channel_configs=self.channel_configs,
                flat_target_reference_angle_deg=self.flat_target_reference_angle_deg,
                flat_target_enabled=self.flat_target_normalization_enabled,
            )
            return synthesized["sphere_spl_norm_db"]
        return result.sphere_spl_norm_db

    def _channel_on_axis_dataset(self, freqs: np.ndarray) -> dict[str, np.ndarray]:
        if not self.supports_channel_resynthesis:
            return {}

        ordered = self.ordered_results()
        first = ordered[0]
        if first.channel_names is None:
            return {}

        channel_names = np.asarray(first.channel_names)
        curves = np.empty((channel_names.size, len(ordered)), dtype=np.float32)
        angles = np.asarray(self.polar_angle_deg, dtype=np.float32)

        for freq_index, result in enumerate(ordered):
            if (
                result.channel_names is None
                or result.horizontal_pressure is None
                or np.asarray(result.channel_names).size != channel_names.size
            ):
                return {}
            weights = self._channel_basis_weights(result)
            for channel_index in range(np.asarray(result.channel_names).size):
                pressure = complex_reference_pressure(
                    np.asarray(result.horizontal_pressure[channel_index]) * weights[channel_index],
                    angles,
                    0.0,
                )
                curves[channel_index, freq_index] = float(pressure_to_spl(np.asarray([pressure]))[0])

        return {
            "channel_on_axis_names": channel_names,
            "channel_on_axis_spl_db": curves,
            "channel_on_axis_freq_hz": freqs,
        }


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
        from blab.solvers.bempp_local import BemppLocalBackend

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
