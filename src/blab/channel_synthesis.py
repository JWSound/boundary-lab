"""Complex-pressure synthesis for post-solve channel edits."""

from __future__ import annotations

import numpy as np
from scipy import signal

from blab.config import ChannelConfig, CrossoverConfig


REFERENCE_PRESSURE_PA = 20e-6


def crossover_response(crossover: CrossoverConfig, freq_hz: float) -> complex:
    crossover_type = crossover.type.lower()
    if crossover_type == "none":
        return 1.0 + 0.0j

    filter_name = crossover.filter.lower()
    if filter_name == "linkwitz_riley":
        section_order = crossover.order // 2
        section = butterworth_response(crossover_type, section_order, crossover.frequency_hz, freq_hz)
        return section * section

    return butterworth_response(crossover_type, crossover.order, crossover.frequency_hz, freq_hz)


def butterworth_response(crossover_type: str, order: int, cutoff_hz: float, freq_hz: float) -> complex:
    b, a = signal.butter(
        order,
        2.0 * np.pi * cutoff_hz,
        btype="lowpass" if crossover_type == "lowpass" else "highpass",
        analog=True,
    )
    _, h = signal.freqs(b, a, worN=[2.0 * np.pi * freq_hz])
    return complex(h[0])


def channel_drive(channel: ChannelConfig, freq_hz: float) -> complex:
    omega = 2.0 * np.pi * freq_hz
    level = 10.0 ** (channel.level_db / 20.0)
    delay = np.exp(-1j * omega * (channel.delay_ms / 1000.0))
    crossover = 1.0 + 0.0j
    for crossover_config in (channel.hpf, channel.lpf):
        if crossover_config.type.lower() != "none":
            crossover *= crossover_response(crossover_config, freq_hz)
    return complex(level * channel.polarity * delay * crossover)


def flat_target_corrections(
    horizontal_pressure: np.ndarray,
    angles_deg: np.ndarray,
    reference_angle_deg: float,
    *,
    enabled: bool = True,
    floor_pa: float = 1e-12,
) -> np.ndarray:
    pressure = np.asarray(horizontal_pressure)
    if pressure.ndim != 2:
        raise ValueError("horizontal_pressure must have shape (channel, angle).")
    if not enabled:
        return np.ones(pressure.shape[0], dtype=np.float32)

    reference_pressures = np.asarray(
        [
            _interpolate_complex_reference(channel_pressure, angles_deg, reference_angle_deg)
            for channel_pressure in pressure
        ]
    )
    magnitudes = np.abs(reference_pressures)
    valid = np.isfinite(magnitudes) & (magnitudes > floor_pa)
    corrections = np.ones(pressure.shape[0], dtype=np.float32)
    corrections[valid] = (1.0 / magnitudes[valid]).astype(np.float32, copy=False)
    return corrections


def synthesize_channel_basis_spl(
    *,
    freq_hz: float,
    polar_angle_deg: np.ndarray,
    channel_names: np.ndarray,
    horizontal_pressure: np.ndarray,
    vertical_pressure: np.ndarray,
    channel_configs: tuple[ChannelConfig, ...],
    flat_target_reference_angle_deg: float,
    flat_target_enabled: bool = True,
    sphere_pressure: np.ndarray | None = None,
) -> dict[str, np.ndarray | None]:
    names = [str(name) for name in np.asarray(channel_names).tolist()]
    channels_by_name = {channel.name: channel for channel in channel_configs}
    horizontal = np.asarray(horizontal_pressure, dtype=np.complex64)
    vertical = np.asarray(vertical_pressure, dtype=np.complex64)
    angles = np.asarray(polar_angle_deg, dtype=np.float32)
    corrections = flat_target_corrections(
        horizontal,
        angles,
        flat_target_reference_angle_deg,
        enabled=flat_target_enabled,
    )
    weights = np.asarray(
        [
            channel_drive(channels_by_name.get(name, ChannelConfig(name=name)), freq_hz) * corrections[index]
            for index, name in enumerate(names)
        ],
        dtype=np.complex64,
    )

    horizontal_summed = np.sum(horizontal * weights[:, np.newaxis], axis=0)
    vertical_summed = np.sum(vertical * weights[:, np.newaxis], axis=0)
    horizontal_spl = pressure_to_spl(horizontal_summed)
    vertical_spl = pressure_to_spl(vertical_summed)
    on_axis_idx = int(np.argmin(np.abs(angles)))
    on_axis_ref = horizontal_spl[on_axis_idx]
    sphere_spl_norm = None
    if sphere_pressure is not None:
        sphere = np.asarray(sphere_pressure, dtype=np.complex64)
        sphere_summed = np.sum(sphere * weights[:, np.newaxis], axis=0)
        sphere_spl_norm = (pressure_to_spl(sphere_summed) - on_axis_ref).astype(np.float32, copy=False)

    return {
        "horizontal_spl_db": horizontal_spl.astype(np.float32, copy=False),
        "vertical_spl_db": vertical_spl.astype(np.float32, copy=False),
        "horizontal_spl_norm_db": (horizontal_spl - on_axis_ref).astype(np.float32, copy=False),
        "vertical_spl_norm_db": (vertical_spl - on_axis_ref).astype(np.float32, copy=False),
        "sphere_spl_norm_db": sphere_spl_norm,
        "channel_corrections": corrections,
    }


def pressure_to_spl(pressure: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.abs(pressure) / REFERENCE_PRESSURE_PA)


def complex_reference_pressure(
    values: np.ndarray,
    angles_deg: np.ndarray,
    reference_angle_deg: float,
) -> complex:
    return _interpolate_complex_reference(values, angles_deg, reference_angle_deg)


def _interpolate_complex_reference(
    values: np.ndarray,
    angles_deg: np.ndarray,
    reference_angle_deg: float,
) -> complex:
    angles = np.asarray(angles_deg, dtype=float)
    complex_values = np.asarray(values, dtype=np.complex64)
    if angles.ndim != 1 or complex_values.ndim != 1 or angles.size != complex_values.size:
        raise ValueError("angles_deg and values must be 1D arrays with matching length.")
    if angles.size == 0:
        return 0.0 + 0.0j
    if angles.size == 1:
        return complex(complex_values[0])

    reference_wrapped = ((float(reference_angle_deg) + 180.0) % 360.0) - 180.0
    if np.isclose(angles[0], -180.0) and np.isclose(angles[-1], 180.0):
        interp_angles = angles[:-1]
        interp_values = complex_values[:-1]
    else:
        interp_angles = angles
        interp_values = complex_values

    angles_ext = np.concatenate([interp_angles - 360.0, interp_angles, interp_angles + 360.0])
    values_ext = np.concatenate([interp_values, interp_values, interp_values])
    real = np.interp(reference_wrapped, angles_ext, values_ext.real)
    imag = np.interp(reference_wrapped, angles_ext, values_ext.imag)
    return complex(real, imag)
