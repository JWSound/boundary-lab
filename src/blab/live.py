"""In-memory helpers for live GUI solving and plotting."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from blab.channel_synthesis import (
    channel_drive,
    channel_voltage_gain,
    complex_reference_pressure,
    flat_target_corrections,
    pressure_to_spl,
    synthesize_channel_basis_spl,
)
from blab.config import ChannelConfig, SimulationConfig
from blab.phasor import solver_phase_deg, solver_to_standard_phasor
from blab.postprocess import PrepConfig, prepare_visualization_data_from_arrays
from blab.solve_results.model import DIAPHRAGM_VELOCITY_ID, VOICE_COIL_CURRENT_ID
from blab.solvers.base import FrequencyResult, SolveRequest
from blab.system_contract import SystemFrequencyResult

GROUP_DELAY_VALID_RELATIVE_DB = -40.0
ACOUSTIC_LOAD_MAX_VELOCITY_CONDITION = 1.0e6
ACOUSTIC_LOAD_MIN_VELOCITY_M_PER_S = 1.0e-12


@dataclass
class LiveSolveDataset:
    polar_angle_deg: np.ndarray
    radiator_names: np.ndarray = field(default_factory=lambda: np.asarray(["Radiator"]))
    channel_configs: tuple[ChannelConfig, ...] = ()
    flat_target_normalization_enabled: bool = True
    flat_target_reference_angle_deg: float = 0.0
    polar_observation_distance_m: float = 0.0
    exterior_sound_speed_m_per_s: float = 343.0
    sphere_r_distance_m: np.ndarray | None = None
    sphere_theta_polar_rad: np.ndarray | None = None
    sphere_phi_azimuth_rad: np.ndarray | None = None
    voltage_channel_names: frozenset[str] = frozenset()
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
        freqs, angles, horizontal, vertical, _raw_horizontal, _raw_vertical = self._polar_export_arrays()
        return freqs, angles, horizontal, vertical

    def as_raw_polar_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        freqs, angles, _horizontal, _vertical, raw_horizontal, raw_vertical = self._polar_export_arrays()
        return freqs, angles, raw_horizontal, raw_vertical

    def _polar_export_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.results:
            raise ValueError("No solved polar data available.")

        ordered = self.ordered_results()
        synthesized = [self._synthesized_arrays(item) for item in ordered]
        freqs = np.asarray([item.freq_hz for item in ordered], dtype=np.float32)
        angles = self.polar_angle_deg.astype(np.float32, copy=False)
        horizontal = np.vstack([row[0] for row in synthesized]).astype(np.float32, copy=False)
        vertical = np.vstack([row[1] for row in synthesized]).astype(np.float32, copy=False)
        raw_horizontal = np.vstack([row[2] for row in synthesized]).astype(np.float32, copy=False)
        raw_vertical = np.vstack([row[3] for row in synthesized]).astype(np.float32, copy=False)
        return freqs, angles, horizontal, vertical, raw_horizontal, raw_vertical

    def as_complex_polar_export_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.results:
            raise ValueError("No solved polar data available.")
        if not self.supports_channel_resynthesis:
            raise ValueError("Phase export requires channel-basis pressure data.")

        ordered = self.ordered_results()
        freqs = np.asarray([item.freq_hz for item in ordered], dtype=np.float32)
        horizontal = np.vstack([self._synthesized_complex_pressures(item)[0] for item in ordered]).astype(
            np.complex64, copy=False
        )
        vertical = np.vstack([self._synthesized_complex_pressures(item)[1] for item in ordered]).astype(
            np.complex64, copy=False
        )
        return freqs, self.polar_angle_deg.astype(np.float32, copy=False), horizontal, vertical

    def as_channel_on_axis_export_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return frequency, channel name, SPL, and phase arrays at zero degrees."""
        freqs, channel_names, pressures = self._channel_on_axis_complex_pressures()
        spl_db = pressure_to_spl(pressures).astype(np.float32, copy=False)
        phase_deg = self._propagation_aligned_phase_deg(pressures, freqs)
        return freqs, channel_names, spl_db, phase_deg

    def _channel_on_axis_complex_pressures(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return raw complex on-axis pressure for each synthesized channel."""

        if not self.results:
            raise ValueError("No solved on-axis data available.")
        if not self.supports_channel_resynthesis:
            raise ValueError("On-axis phase export requires channel-basis pressure data.")

        ordered = self.ordered_results()
        first_names = ordered[0].channel_names
        if first_names is None:
            raise ValueError("Channel names are unavailable.")
        channel_names = np.asarray(first_names).astype(str)
        channel_count = channel_names.size
        if channel_count == 0:
            raise ValueError("No solved channels are available for on-axis export.")
        pressures = np.empty((channel_count, len(ordered)), dtype=np.complex64)
        angles = np.asarray(self.polar_angle_deg, dtype=np.float32)

        for freq_index, result in enumerate(ordered):
            if result.channel_names is None or result.horizontal_pressure is None:
                raise ValueError("Channel-basis pressure data is incomplete.")
            result_names = np.asarray(result.channel_names).astype(str)
            horizontal = np.asarray(result.horizontal_pressure, dtype=np.complex64)
            if not np.array_equal(result_names, channel_names):
                raise ValueError("Solved channel names or ordering changed between frequencies.")
            if horizontal.ndim != 2 or horizontal.shape != (channel_count, angles.size):
                raise ValueError("Channel-basis pressure dimensions are inconsistent with the polar samples.")

            weights = self._channel_basis_weights(result)
            for channel_index in range(channel_count):
                pressures[channel_index, freq_index] = complex_reference_pressure(
                    horizontal[channel_index] * weights[channel_index],
                    angles,
                    0.0,
                )

        freqs = np.asarray([item.freq_hz for item in ordered], dtype=np.float32)
        return freqs, channel_names, pressures

    def as_group_delay_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return source-referenced group delay for the sum and each channel."""

        if not self.supports_channel_resynthesis or self.solved_count < 3:
            return None
        try:
            freqs, channel_names, pressures = self._channel_on_axis_complex_pressures()
        except ValueError:
            return None
        aligned_pressures = solver_to_standard_phasor(
            self._propagation_aligned_values(pressures, freqs)
        )
        configs_by_name = {channel.name: channel for channel in self.channel_configs}
        configured_delay_s = np.asarray(
            [
                configs_by_name.get(str(name), ChannelConfig(name=str(name))).delay_ms / 1000.0
                for name in channel_names.tolist()
            ],
            dtype=np.float64,
        )
        channel_group_delay_ms, summed_group_delay_ms = group_delay_from_channel_pressures(
            freqs,
            aligned_pressures,
            configured_delay_s=configured_delay_s,
        )
        labels = np.concatenate((np.asarray(["Sum"]), channel_names.astype(str)))
        values = np.vstack((summed_group_delay_ms, channel_group_delay_ms)).astype(
            np.float32,
            copy=False,
        )
        return freqs, labels, values

    def as_visualization_dataset(self, cfg: PrepConfig | None = None) -> dict[str, np.ndarray] | None:
        if not self.results:
            return None

        prep_cfg = cfg or PrepConfig()
        self.set_channel_synthesis(
            self.channel_configs,
            flat_target_reference_angle_deg=prep_cfg.hor_ref_angle,
        )
        freqs, angles, horizontal, vertical, raw_horizontal, raw_vertical = self._polar_export_arrays()
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
        if not self.has_balloon_data:
            return None

        ordered = self.ordered_results()
        freqs = np.asarray([item.freq_hz for item in ordered], dtype=np.float32)
        sphere_rows: list[np.ndarray] = []
        for item in ordered:
            sphere = self._synthesized_sphere(item)
            if sphere is None:
                return None
            sphere_rows.append(sphere)

        return {
            "freq_hz": freqs,
            "r_distance_m": np.asarray(self.sphere_r_distance_m, dtype=np.float32),
            "theta_polar_rad": np.asarray(self.sphere_theta_polar_rad, dtype=np.float32),
            "phi_azimuth_rad": np.asarray(self.sphere_phi_azimuth_rad, dtype=np.float32),
            "spl_norm": np.vstack(sphere_rows).astype(np.float32, copy=False),
        }

    @property
    def has_balloon_data(self) -> bool:
        if (
            not self.results
            or self.sphere_r_distance_m is None
            or self.sphere_theta_polar_rad is None
            or self.sphere_phi_azimuth_rad is None
        ):
            return False
        return all(
            (result.has_channel_basis and result.sphere_pressure is not None) or result.sphere_spl_norm_db is not None
            for result in self.results.values()
        )

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
                channel_configs=self.channel_configs,
                flat_target_reference_angle_deg=self.flat_target_reference_angle_deg,
                flat_target_enabled=self.flat_target_normalization_enabled,
                voltage_channel_names=self.voltage_channel_names,
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
        if result.channel_names is None or result.horizontal_pressure is None or result.vertical_pressure is None:
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
                * (
                    channel_voltage_gain(
                        channel_configs_by_name.get(str(channel_name), ChannelConfig(name=str(channel_name)))
                    )
                    if str(channel_name) in self.voltage_channel_names
                    else 1.0
                )
                * float(corrections[index])
                for index, channel_name in enumerate(np.asarray(result.channel_names).tolist())
            ],
            dtype=np.complex64,
        )

    def channel_basis_weights(self, result: FrequencyResult) -> np.ndarray:
        """Return the complex drive applied to each grouped channel basis."""

        return self._channel_basis_weights(result)

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
                voltage_channel_names=self.voltage_channel_names,
            )
            return synthesized["sphere_spl_norm_db"]
        return result.sphere_spl_norm_db

    def _channel_on_axis_dataset(self, freqs: np.ndarray) -> dict[str, np.ndarray]:
        if not self.supports_channel_resynthesis:
            return {}

        try:
            export_freqs, channel_names, curves, channel_phase_deg = self.as_channel_on_axis_export_arrays()
        except ValueError:
            return {}
        if not np.array_equal(export_freqs, np.asarray(freqs, dtype=np.float32)):
            return {}

        angles = np.asarray(self.polar_angle_deg, dtype=np.float32)
        summed_pressures = np.asarray(
            [
                complex_reference_pressure(
                    self._synthesized_complex_pressures(result)[0],
                    angles,
                    0.0,
                )
                for result in self.ordered_results()
            ],
            dtype=np.complex64,
        )
        summed_phase_deg = self._propagation_aligned_phase_deg(summed_pressures, freqs)

        return {
            "channel_on_axis_names": channel_names,
            "channel_on_axis_spl_db": curves,
            "on_axis_phase_deg": summed_phase_deg,
            "channel_on_axis_phase_deg": channel_phase_deg,
            "channel_on_axis_freq_hz": freqs,
        }

    def _propagation_aligned_phase_deg(
        self,
        pressures: np.ndarray,
        freqs_hz: np.ndarray,
    ) -> np.ndarray:
        values = self._propagation_aligned_values(pressures, freqs_hz)
        return solver_phase_deg(values)

    def _propagation_aligned_values(
        self,
        pressures: np.ndarray,
        freqs_hz: np.ndarray,
    ) -> np.ndarray:
        distance_m = float(self.polar_observation_distance_m)
        sound_speed_m_per_s = float(self.exterior_sound_speed_m_per_s)
        if not np.isfinite(distance_m) or distance_m < 0.0:
            raise ValueError("Polar observation distance must be finite and non-negative.")
        if not np.isfinite(sound_speed_m_per_s) or sound_speed_m_per_s <= 0.0:
            raise ValueError("Exterior sound speed must be finite and greater than zero.")

        freqs = np.asarray(freqs_hz, dtype=np.float64)
        values = np.asarray(pressures)
        if values.shape[-1] != freqs.size:
            raise ValueError("Pressure frequency axis must match freqs_hz.")
        reference_delay_s = distance_m / sound_speed_m_per_s
        reference_rotation = np.exp(-1j * 2.0 * np.pi * freqs * reference_delay_s)
        return values * reference_rotation


def group_delay_from_channel_pressures(
    freqs_hz: np.ndarray,
    channel_pressures: np.ndarray,
    *,
    configured_delay_s: np.ndarray | None = None,
    valid_relative_db: float = GROUP_DELAY_VALID_RELATIVE_DB,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive channel and summed group delay from standard-audio responses.

    Known pure channel delays are removed before phase unwrapping and added
    back analytically. The summed derivative is assembled from the channel
    derivatives, avoiding a second unwrap of the potentially cancelling sum.
    """

    frequencies = np.asarray(freqs_hz, dtype=np.float64)
    pressures = np.asarray(channel_pressures, dtype=np.complex128)
    if frequencies.ndim != 1 or frequencies.size < 3:
        raise ValueError("Group delay requires at least three frequency samples.")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("Group-delay frequencies must be finite and greater than zero.")
    if np.any(np.diff(frequencies) <= 0.0):
        raise ValueError("Group-delay frequencies must be strictly increasing.")
    if pressures.ndim != 2 or pressures.shape[1] != frequencies.size:
        raise ValueError("Channel pressures must have shape (channel, frequency).")
    delays = (
        np.zeros(pressures.shape[0], dtype=np.float64)
        if configured_delay_s is None
        else np.asarray(configured_delay_s, dtype=np.float64)
    )
    if delays.shape != (pressures.shape[0],) or not np.all(np.isfinite(delays)):
        raise ValueError("Configured channel delays must be finite and match the channel count.")

    omega = 2.0 * np.pi * frequencies
    channel_group_delay_s = np.full(pressures.shape, np.nan, dtype=np.float64)
    channel_derivatives = np.zeros(pressures.shape, dtype=np.complex128)
    for channel_index, (pressure, delay_s) in enumerate(zip(pressures, delays, strict=True)):
        amplitude = np.abs(pressure)
        residual = pressure * np.exp(1j * omega * delay_s)
        residual_phase = np.unwrap(np.angle(residual))
        full_phase = residual_phase - omega * delay_s
        amplitude_derivative = np.gradient(amplitude, omega, edge_order=2)
        phase_derivative = np.gradient(full_phase, omega, edge_order=2)
        channel_derivatives[channel_index] = np.exp(1j * full_phase) * (
            amplitude_derivative + 1j * amplitude * phase_derivative
        )
        channel_group_delay_s[channel_index] = -phase_derivative

    summed_pressure = np.sum(pressures, axis=0)
    summed_derivative = np.sum(channel_derivatives, axis=0)
    summed_log_derivative = np.full(frequencies.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(
            summed_derivative,
            summed_pressure,
            out=summed_log_derivative,
            where=np.abs(summed_pressure) > np.finfo(np.float64).tiny,
        )
    summed_group_delay_s = -np.imag(summed_log_derivative)

    for channel_index, pressure in enumerate(pressures):
        channel_group_delay_s[channel_index, ~_group_delay_valid_mask(pressure, valid_relative_db)] = np.nan
    summed_group_delay_s[~_group_delay_valid_mask(summed_pressure, valid_relative_db)] = np.nan
    return channel_group_delay_s * 1000.0, summed_group_delay_s * 1000.0


def _group_delay_valid_mask(values: np.ndarray, valid_relative_db: float) -> np.ndarray:
    magnitude = np.abs(np.asarray(values))
    finite = np.isfinite(magnitude)
    maximum = float(np.max(magnitude[finite])) if np.any(finite) else 0.0
    if maximum <= np.finfo(float).tiny:
        return np.zeros(magnitude.shape, dtype=bool)
    threshold = maximum * 10.0 ** (float(valid_relative_db) / 20.0)
    return finite & (magnitude >= threshold)


@dataclass
class TransducerMotionDataset:
    """Small live cache of the canonical transducer-velocity result rows."""

    excitation_channel_names: np.ndarray
    transducer_names: np.ndarray
    results: dict[float, np.ndarray] = field(default_factory=dict)

    def add(self, result: SystemFrequencyResult) -> None:
        quantity = next((item for item in result.quantities if item.id == DIAPHRAGM_VELOCITY_ID), None)
        if quantity is None:
            return
        if quantity.axes != ("excitation", "transducer"):
            raise ValueError("Diaphragm velocity must use excitation and transducer axes.")
        values = np.asarray(quantity.values, dtype=np.complex64)
        expected_shape = (self.excitation_channel_names.size, self.transducer_names.size)
        if values.shape != expected_shape:
            raise ValueError(
                f"Diaphragm velocity has shape {values.shape}, expected {expected_shape}."
            )
        self.results[float(result.freq_hz)] = values.copy()

    def as_excursion_arrays(
        self,
        acoustic: LiveSolveDataset,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return frequency, transducer names, and synthesized excursion in mm."""

        frequencies = sorted(set(self.results).intersection(acoustic.results))
        if not frequencies:
            return None
        rows: list[np.ndarray] = []
        excitation_channels = [str(value) for value in self.excitation_channel_names.tolist()]
        for frequency in frequencies:
            live_result = acoustic.results[frequency]
            if live_result.channel_names is None:
                return None
            grouped_names = [str(value) for value in np.asarray(live_result.channel_names).tolist()]
            grouped_weights = acoustic.channel_basis_weights(live_result)
            if grouped_weights.shape != (len(grouped_names),):
                raise ValueError("Channel synthesis weights do not match the grouped channel names.")
            weight_by_name = dict(zip(grouped_names, grouped_weights, strict=True))
            try:
                excitation_weights = np.asarray(
                    [weight_by_name[name] for name in excitation_channels],
                    dtype=np.complex64,
                )
            except KeyError as exc:
                raise ValueError(f"No synthesized channel basis is available for {exc.args[0]!r}.") from exc
            velocity = self.results[frequency]
            synthesized_velocity = np.sum(velocity * excitation_weights[:, np.newaxis], axis=0)
            excursion_mm = np.abs(synthesized_velocity / (-1j * 2.0 * np.pi * frequency)) * 1000.0
            rows.append(excursion_mm.astype(np.float32, copy=False))
        return (
            np.asarray(frequencies, dtype=np.float32),
            np.asarray(self.transducer_names).copy(),
            np.vstack(rows).T.astype(np.float32, copy=False),
        )


@dataclass
class ElectricalImpedanceDataset:
    """Live per-channel electrical load derived from voltage-basis currents."""

    excitation_port_ids: tuple[str, ...]
    excitation_channel_names: np.ndarray
    excitation_component_ids: np.ndarray
    transducer_component_ids: np.ndarray
    physical_driver_orbit_counts: np.ndarray
    channel_names: np.ndarray
    results: dict[float, np.ndarray] = field(default_factory=dict)

    def add(self, result: SystemFrequencyResult) -> None:
        quantity = next((item for item in result.quantities if item.id == VOICE_COIL_CURRENT_ID), None)
        if quantity is None:
            return
        if quantity.axes != ("excitation", "transducer"):
            raise ValueError("Voice-coil current must use excitation and transducer axes.")

        result_port_ids = tuple(str(value) for value in result.excitation_port_ids)
        if set(result_port_ids) != set(self.excitation_port_ids):
            raise ValueError("Voice-coil current excitation ports do not match the prepared solve.")
        row_by_port_id = {port_id: index for index, port_id in enumerate(result_port_ids)}
        row_order = [row_by_port_id[port_id] for port_id in self.excitation_port_ids]

        expected_component_ids = tuple(str(value) for value in self.transducer_component_ids.tolist())
        result_component_ids = tuple(str(value) for value in quantity.metadata.get("component_ids", ()))
        if not result_component_ids:
            result_component_ids = expected_component_ids
        if set(result_component_ids) != set(expected_component_ids):
            raise ValueError("Voice-coil current transducers do not match the prepared solve.")
        column_by_component_id = {
            component_id: index for index, component_id in enumerate(result_component_ids)
        }
        column_order = [column_by_component_id[component_id] for component_id in expected_component_ids]

        values = np.asarray(quantity.values, dtype=np.complex64)
        expected_shape = (len(result_port_ids), len(result_component_ids))
        if values.shape != expected_shape:
            raise ValueError(f"Voice-coil current has shape {values.shape}, expected {expected_shape}.")
        values = values[np.ix_(row_order, column_order)]

        try:
            reference_voltage_v = float(result.diagnostics["transducer_reference_voltage_v"])
        except (KeyError, TypeError, ValueError):
            reference_voltage_v = np.nan
        if not np.isfinite(reference_voltage_v) or reference_voltage_v <= 0.0:
            reference_voltage_v = np.nan

        excitation_channels = np.asarray(self.excitation_channel_names).astype(str)
        excitation_components = np.asarray(self.excitation_component_ids).astype(str)
        transducer_components = np.asarray(self.transducer_component_ids).astype(str)
        orbit_counts = np.asarray(self.physical_driver_orbit_counts, dtype=np.float64)
        impedances = np.full(self.channel_names.size, np.nan + 1j * np.nan, dtype=np.complex64)
        for channel_index, channel_name_value in enumerate(self.channel_names.tolist()):
            channel_name = str(channel_name_value)
            excitation_indices = np.flatnonzero(excitation_channels == channel_name)
            driven_component_ids = set(excitation_components[excitation_indices].tolist())
            transducer_indices = np.asarray(
                [
                    index
                    for index, component_id in enumerate(transducer_components.tolist())
                    if component_id in driven_component_ids
                ],
                dtype=np.int64,
            )
            if not excitation_indices.size or not transducer_indices.size:
                continue
            channel_currents = values[np.ix_(excitation_indices, transducer_indices)]
            total_current = np.sum(
                channel_currents * orbit_counts[transducer_indices][np.newaxis, :],
                dtype=np.complex128,
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                impedances[channel_index] = reference_voltage_v / total_current
        self.results[float(result.freq_hz)] = impedances

    def as_impedance_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Return frequency, channel, magnitude, and wrapped phase arrays."""

        if not self.results or not self.channel_names.size:
            return None
        ordered = sorted(self.results.items())
        frequencies = np.asarray([frequency for frequency, _values in ordered], dtype=np.float32)
        complex_impedance = np.vstack([values for _frequency, values in ordered]).T
        magnitude_ohm = np.abs(complex_impedance).astype(np.float32, copy=False)
        phase_deg = solver_phase_deg(complex_impedance)
        return frequencies, np.asarray(self.channel_names).copy(), magnitude_ohm, phase_deg


@dataclass
class AcousticLoadImpedanceDataset:
    """Intrinsic transducer acoustic loads recovered from a voltage-basis solve.

    The coupled solve returns diaphragm velocity and voice-coil current for every
    independent voltage excitation. Mechanical equilibrium gives the acoustic
    load force as ``Bl * current - Zm * velocity``. Solving that force matrix
    against the velocity matrix isolates the self impedance of each transducer
    with all other generalized transducer velocities held at zero.
    """

    excitation_port_ids: tuple[str, ...]
    excitation_port_kinds: np.ndarray
    excitation_component_ids: np.ndarray
    transducer_component_ids: np.ndarray
    transducer_names: np.ndarray
    bl_n_per_a: np.ndarray
    mmd_kg: np.ndarray
    cms_m_per_n: np.ndarray
    rms_n_s_per_m: np.ndarray
    results: dict[float, np.ndarray] = field(default_factory=dict)
    velocity_condition_numbers: dict[float, float] = field(default_factory=dict)

    def add(self, result: SystemFrequencyResult) -> None:
        velocity_quantity = next(
            (item for item in result.quantities if item.id == DIAPHRAGM_VELOCITY_ID),
            None,
        )
        current_quantity = next(
            (item for item in result.quantities if item.id == VOICE_COIL_CURRENT_ID),
            None,
        )
        if velocity_quantity is None or current_quantity is None:
            return
        expected_axes = ("excitation", "transducer")
        if velocity_quantity.axes != expected_axes or current_quantity.axes != expected_axes:
            raise ValueError(
                "Coupled acoustic load recovery requires excitation and transducer axes."
            )

        result_port_ids = tuple(str(value) for value in result.excitation_port_ids)
        if set(result_port_ids) != set(self.excitation_port_ids):
            raise ValueError("Coupled acoustic load excitation ports do not match the prepared solve.")
        row_by_port_id = {port_id: index for index, port_id in enumerate(result_port_ids)}
        row_order = [row_by_port_id[port_id] for port_id in self.excitation_port_ids]

        expected_component_ids = tuple(str(value) for value in self.transducer_component_ids.tolist())
        velocity_component_ids = tuple(
            str(value) for value in velocity_quantity.metadata.get("component_ids", ())
        ) or expected_component_ids
        current_component_ids = tuple(
            str(value) for value in current_quantity.metadata.get("component_ids", ())
        ) or expected_component_ids
        if set(velocity_component_ids) != set(expected_component_ids):
            raise ValueError("Diaphragm-velocity transducers do not match the prepared solve.")
        if set(current_component_ids) != set(expected_component_ids):
            raise ValueError("Voice-coil-current transducers do not match the prepared solve.")

        velocity = self._ordered_quantity_values(
            velocity_quantity.values,
            result_port_ids,
            velocity_component_ids,
            row_order,
            expected_component_ids,
            "Diaphragm velocity",
        )
        current = self._ordered_quantity_values(
            current_quantity.values,
            result_port_ids,
            current_component_ids,
            row_order,
            expected_component_ids,
            "Voice-coil current",
        )

        excitation_kinds = np.asarray(self.excitation_port_kinds).astype(str)
        excitation_components = np.asarray(self.excitation_component_ids).astype(str)
        voltage_rows: list[int] = []
        for component_id in expected_component_ids:
            candidates = np.flatnonzero(
                (excitation_kinds == "voltage") & (excitation_components == component_id)
            )
            if candidates.size != 1:
                self._store_unavailable(float(result.freq_hz))
                return
            voltage_rows.append(int(candidates[0]))

        velocity_basis = velocity[voltage_rows, :].T.astype(np.complex128, copy=False)
        current_basis = current[voltage_rows, :].T.astype(np.complex128, copy=False)
        singular_values = np.linalg.svd(velocity_basis, compute_uv=False)
        maximum_singular = float(singular_values[0]) if singular_values.size else 0.0
        minimum_singular = float(singular_values[-1]) if singular_values.size else 0.0
        condition = (
            maximum_singular / minimum_singular
            if minimum_singular > 0.0
            else float("inf")
        )
        self.velocity_condition_numbers[float(result.freq_hz)] = condition
        if (
            not np.isfinite(condition)
            or condition > ACOUSTIC_LOAD_MAX_VELOCITY_CONDITION
            or maximum_singular <= ACOUSTIC_LOAD_MIN_VELOCITY_M_PER_S
        ):
            self._store_unavailable(float(result.freq_hz), record_condition=False)
            return

        omega = 2.0 * np.pi * float(result.freq_hz)
        mechanical_impedance = np.asarray(self.rms_n_s_per_m, dtype=np.float64) + 1j * (
            1.0 / (omega * np.asarray(self.cms_m_per_n, dtype=np.float64))
            - omega * np.asarray(self.mmd_kg, dtype=np.float64)
        )
        load_force = (
            np.asarray(self.bl_n_per_a, dtype=np.float64)[:, np.newaxis] * current_basis
            - mechanical_impedance[:, np.newaxis] * velocity_basis
        )
        try:
            impedance_matrix = np.linalg.solve(velocity_basis.T, load_force.T).T
        except np.linalg.LinAlgError:
            self._store_unavailable(float(result.freq_hz), record_condition=False)
            return
        diagonal = np.diag(impedance_matrix)
        diagonal = np.where(np.isfinite(diagonal), diagonal, np.nan + 1j * np.nan)
        self.results[float(result.freq_hz)] = diagonal.astype(np.complex64, copy=False)

    @staticmethod
    def _ordered_quantity_values(
        raw_values: np.ndarray,
        result_port_ids: tuple[str, ...],
        result_component_ids: tuple[str, ...],
        row_order: list[int],
        expected_component_ids: tuple[str, ...],
        label: str,
    ) -> np.ndarray:
        values = np.asarray(raw_values, dtype=np.complex64)
        expected_shape = (len(result_port_ids), len(result_component_ids))
        if values.shape != expected_shape:
            raise ValueError(f"{label} has shape {values.shape}, expected {expected_shape}.")
        column_by_component_id = {
            component_id: index for index, component_id in enumerate(result_component_ids)
        }
        column_order = [column_by_component_id[value] for value in expected_component_ids]
        return values[np.ix_(row_order, column_order)]

    def _store_unavailable(self, frequency_hz: float, *, record_condition: bool = True) -> None:
        if record_condition:
            self.velocity_condition_numbers[frequency_hz] = float("inf")
        self.results[frequency_hz] = np.full(
            self.transducer_names.size,
            np.nan + 1j * np.nan,
            dtype=np.complex64,
        )

    def as_impedance_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Return standard-convention real and imaginary acoustic load in N*s/m."""

        if not self.results or not self.transducer_names.size:
            return None
        ordered = sorted(self.results.items())
        frequencies = np.asarray([frequency for frequency, _values in ordered], dtype=np.float32)
        native_impedance = np.vstack([values for _frequency, values in ordered]).T
        display_impedance = solver_to_standard_phasor(native_impedance)
        return (
            frequencies,
            np.asarray(self.transducer_names).copy(),
            display_impedance.real.astype(np.float32, copy=False),
            display_impedance.imag.astype(np.float32, copy=False),
        )


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


def solve_frequency_worker_process(
    config: SimulationConfig, frequencies, stop_event, output_queue, worker_id: int
) -> None:
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
