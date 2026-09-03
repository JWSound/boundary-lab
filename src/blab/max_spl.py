"""Post-solve maximum-SPL limits for voltage-driven transducer channels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from blab.config import DEFAULT_CHANNEL_VOLTAGE_V


@dataclass(frozen=True)
class MaxSplLimit:
    """Shared ratings applied to every electrodynamic component on a channel."""

    xmax_mm: float
    pmax_w: float

    @property
    def enabled(self) -> bool:
        return self.xmax_mm > 0.0 and self.pmax_w > 0.0

    def validated(self) -> MaxSplLimit:
        xmax_mm = float(self.xmax_mm)
        pmax_w = float(self.pmax_w)
        if not np.isfinite(xmax_mm) or xmax_mm < 0.0:
            raise ValueError("Xmax must be finite and non-negative.")
        if not np.isfinite(pmax_w) or pmax_w < 0.0:
            raise ValueError("Pmax must be finite and non-negative.")
        if (xmax_mm == 0.0) != (pmax_w == 0.0):
            raise ValueError("Set both Xmax and Pmax to zero to disable a channel, or set both above zero.")
        return MaxSplLimit(xmax_mm=xmax_mm, pmax_w=pmax_w)


def max_spl_limits_from_payload(payload: object) -> dict[str, MaxSplLimit]:
    """Read valid project-persisted limits, ignoring malformed stale entries."""

    if not isinstance(payload, dict):
        return {}
    limits: dict[str, MaxSplLimit] = {}
    for raw_name, raw_limit in payload.items():
        name = str(raw_name).strip()
        if not name or not isinstance(raw_limit, dict):
            continue
        try:
            limit = MaxSplLimit(
                xmax_mm=float(raw_limit["xmax_mm"]),
                pmax_w=float(raw_limit["pmax_w"]),
            ).validated()
        except (KeyError, TypeError, ValueError):
            continue
        limits[name] = limit
    return limits


def max_spl_limits_payload(limits: Mapping[str, MaxSplLimit]) -> dict[str, dict[str, float]]:
    """Serialize channel ratings into the project-file representation."""

    return {
        str(name): {
            "xmax_mm": float(limit.validated().xmax_mm),
            "pmax_w": float(limit.validated().pmax_w),
        }
        for name, limit in limits.items()
        if str(name).strip()
    }


def transducer_rated_resistance_ohm(parameters: Mapping[str, object]) -> float:
    """Return the resistance used to convert a rated wattage to RMS voltage."""

    semi_inductance = parameters.get("semi_inductance")
    if isinstance(semi_inductance, dict) and semi_inductance.get("enabled") is True:
        value = semi_inductance.get("re_prime_ohm")
    else:
        value = parameters.get("re_ohm")
    resistance = float(value)
    if not np.isfinite(resistance) or resistance <= 0.0:
        raise ValueError("Transducer resistance must be finite and greater than zero.")
    return resistance


def calculate_max_spl_curves(
    *,
    frequencies_hz: np.ndarray,
    channel_names: np.ndarray,
    on_axis_pressure_pa: np.ndarray,
    excitation_channel_names: np.ndarray,
    transducer_channel_names: np.ndarray,
    transducer_resistance_ohm: np.ndarray,
    diaphragm_velocity_m_per_s: np.ndarray,
    limits_by_channel: Mapping[str, MaxSplLimit],
    reference_voltage_v: np.ndarray | float = DEFAULT_CHANNEL_VOLTAGE_V,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate isolated maximum-SPL curves from the raw voltage basis.

    ``diaphragm_velocity_m_per_s`` has axes frequency, excitation, transducer.
    The 2.83 V solve basis is interpreted as RMS; Xmax is one-way peak.
    """

    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    channels = np.asarray(channel_names).astype(str)
    pressure = np.asarray(on_axis_pressure_pa, dtype=np.complex128)
    excitation_channels = np.asarray(excitation_channel_names).astype(str)
    transducer_channels = np.asarray(transducer_channel_names).astype(str)
    resistance = np.asarray(transducer_resistance_ohm, dtype=np.float64)
    velocity = np.asarray(diaphragm_velocity_m_per_s, dtype=np.complex128)
    reference_voltage = np.broadcast_to(np.asarray(reference_voltage_v, dtype=np.float64), frequencies.shape)

    if frequencies.ndim != 1 or not frequencies.size or np.any(frequencies <= 0.0):
        raise ValueError("Maximum SPL requires positive frequency samples.")
    if pressure.shape != (channels.size, frequencies.size):
        raise ValueError("On-axis pressure must have shape (channel, frequency).")
    expected_velocity_shape = (frequencies.size, excitation_channels.size, transducer_channels.size)
    if velocity.shape != expected_velocity_shape:
        raise ValueError("Diaphragm velocity must have shape (frequency, excitation, transducer).")
    if resistance.shape != (transducer_channels.size,):
        raise ValueError("Transducer resistance must align with the transducer axis.")
    if np.any(~np.isfinite(reference_voltage)) or np.any(reference_voltage <= 0.0):
        raise ValueError("Reference voltage must be finite and greater than zero.")

    pressure_by_channel = {name: index for index, name in enumerate(channels.tolist())}
    valid_names = [
        name
        for name in channels.tolist()
        if name in limits_by_channel
        and limits_by_channel[name].validated().enabled
        and np.any(excitation_channels == name)
        and np.any(transducer_channels == name)
    ]
    curves: list[np.ndarray] = []
    for name in valid_names:
        limit = limits_by_channel[name].validated()
        excitation_indices = np.flatnonzero(excitation_channels == name)
        transducer_indices = np.flatnonzero(transducer_channels == name)
        channel_velocity = np.sum(velocity[:, excitation_indices, :], axis=1)
        assigned_velocity = channel_velocity[:, transducer_indices]
        peak_excursion_mm = (
            np.sqrt(2.0) * np.abs(assigned_velocity / (-1j * 2.0 * np.pi * frequencies[:, np.newaxis])) * 1000.0
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            excursion_gain = np.min(limit.xmax_mm / peak_excursion_mm, axis=1)

        voltage_limits = np.sqrt(limit.pmax_w * resistance[transducer_indices])
        power_gain = float(np.min(voltage_limits)) / reference_voltage
        limiting_gain = np.minimum(excursion_gain, power_gain)
        limited_pressure = np.abs(pressure[pressure_by_channel[name]]) * limiting_gain
        with np.errstate(divide="ignore", invalid="ignore"):
            spl_db = 20.0 * np.log10(limited_pressure / 20.0e-6)
        spl_db[~np.isfinite(spl_db)] = np.nan
        curves.append(spl_db.astype(np.float32, copy=False))

    return (
        frequencies.astype(np.float32, copy=False),
        np.asarray(valid_names),
        np.vstack(curves).astype(np.float32, copy=False)
        if curves
        else np.empty((0, frequencies.size), dtype=np.float32),
    )


__all__ = [
    "MaxSplLimit",
    "calculate_max_spl_curves",
    "max_spl_limits_from_payload",
    "max_spl_limits_payload",
    "transducer_rated_resistance_ohm",
]
