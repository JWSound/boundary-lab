"""Complex-pressure synthesis for post-solve channel edits."""

from __future__ import annotations

import numpy as np
from scipy import signal

from blab.config import DEFAULT_CHANNEL_VOLTAGE_V, ChannelConfig, CrossoverConfig
from blab.phasor import standard_to_solver_phasor

REFERENCE_PRESSURE_PA = 20e-6


def channel_configs_from_payload(raw: object) -> tuple[ChannelConfig, ...]:
    raw = raw if isinstance(raw, dict) else {}
    if not raw:
        return (ChannelConfig(name="main"),)
    channels = []
    for name, payload in sorted(raw.items()):
        payload = payload if isinstance(payload, dict) else {}
        channels.append(
            ChannelConfig(
                name=str(name),
                voltage_v=float(payload.get("voltage_v", DEFAULT_CHANNEL_VOLTAGE_V)),
                level_db=float(payload.get("level_db", 0.0)),
                polarity=int(payload.get("polarity", 1)),
                delay_ms=float(payload.get("delay_ms", 0.0)),
                hpf=saved_crossover(payload.get("hpf"), crossover_type="highpass"),
                lpf=saved_crossover(payload.get("lpf"), crossover_type="lowpass"),
            )
        )
    return tuple(channels) or (ChannelConfig(name="main"),)


def saved_crossover(raw: object, *, crossover_type: str) -> CrossoverConfig:
    if not isinstance(raw, dict) or raw.get("frequency_hz") is None:
        return CrossoverConfig()
    return CrossoverConfig(
        type=crossover_type,
        filter=str(raw.get("filter", "butterworth")).lower(),
        order=int(raw.get("order", 1)),
        frequency_hz=float(raw["frequency_hz"]),
    )


def crossover_response(crossover: CrossoverConfig, freq_hz: float) -> complex:
    """Return a crossover coefficient in the solver's ``exp(+i omega t)`` convention."""

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
    return complex(standard_to_solver_phasor(h[0]))


def channel_drive(channel: ChannelConfig, freq_hz: float) -> complex:
    """Return channel DSP in the solver's ``exp(+i omega t)`` convention."""

    omega = 2.0 * np.pi * freq_hz
    level = 10.0 ** (channel.level_db / 20.0)
    delay = np.exp(-1j * omega * (channel.delay_ms / 1000.0))
    crossover = 1.0 + 0.0j
    for crossover_config in (channel.hpf, channel.lpf):
        if crossover_config.type.lower() != "none":
            crossover *= crossover_response(crossover_config, freq_hz)
    return complex(level * channel.polarity * delay * crossover)


def channel_voltage_gain(channel: ChannelConfig) -> float:
    """Return a channel voltage relative to the fixed transducer solve basis."""

    voltage = float(channel.voltage_v)
    if not np.isfinite(voltage) or voltage < 0.0:
        raise ValueError("Channel voltage must be finite and non-negative.")
    return voltage / DEFAULT_CHANNEL_VOLTAGE_V


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


def excitation_weights(
    *,
    solve_index: int,
    frequency_hz: float,
    response_id: str,
    excitation_ids: tuple[str, ...],
    channel_names_by_excitation: tuple[str, ...],
    channel_configs: tuple[ChannelConfig, ...],
    voltage_channel_names: frozenset[str],
    horizontal_pressure_by_frequency: np.ndarray | None,
    horizontal_angles_deg: np.ndarray | None,
    flat_target_enabled: bool,
    flat_target_reference_angle_deg: float,
) -> np.ndarray:
    configs = {channel.name: channel for channel in channel_configs}
    channel_names = tuple(dict.fromkeys(channel_names_by_excitation))
    correction_by_channel = {name: 1.0 for name in channel_names}
    if flat_target_enabled and horizontal_pressure_by_frequency is not None and horizontal_angles_deg is not None:
        horizontal_basis = np.asarray(horizontal_pressure_by_frequency[solve_index])
        grouped = np.vstack(
            [
                np.sum(
                    horizontal_basis[
                        [index for index, candidate in enumerate(channel_names_by_excitation) if candidate == name]
                    ],
                    axis=0,
                )
                for name in channel_names
            ]
        )
        corrections = flat_target_corrections(
            grouped,
            horizontal_angles_deg,
            flat_target_reference_angle_deg,
            enabled=True,
        )
        correction_by_channel.update(
            {name: float(correction) for name, correction in zip(channel_names, corrections, strict=True)}
        )

    selected_channel = response_id.removeprefix("channel:") if response_id.startswith("channel:") else None
    if selected_channel not in channel_names:
        selected_channel = None
    weights = np.zeros(len(excitation_ids), dtype=np.complex64)
    for index, channel_name in enumerate(channel_names_by_excitation):
        if selected_channel is not None and channel_name != selected_channel:
            continue
        config = configs.get(channel_name, ChannelConfig(name=channel_name))
        voltage_gain = channel_voltage_gain(config) if channel_name in voltage_channel_names else 1.0
        weights[index] = correction_by_channel[channel_name] * channel_drive(config, frequency_hz) * voltage_gain
    return weights


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
    voltage_channel_names: frozenset[str] = frozenset(),
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
            channel_drive(channels_by_name.get(name, ChannelConfig(name=name)), freq_hz)
            * (
                channel_voltage_gain(channels_by_name.get(name, ChannelConfig(name=name)))
                if name in voltage_channel_names
                else 1.0
            )
            * corrections[index]
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
    with np.errstate(divide="ignore"):
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
