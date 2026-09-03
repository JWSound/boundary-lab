from __future__ import annotations

import numpy as np
import pytest

from blab.config import ChannelConfig
from blab.live import LiveSolveDataset, TransducerMotionDataset
from blab.max_spl import (
    MaxSplLimit,
    calculate_max_spl_curves,
    max_spl_limits_from_payload,
    max_spl_limits_payload,
    transducer_rated_resistance_ohm,
)
from blab.solve_results import DIAPHRAGM_VELOCITY_ID
from blab.solvers.base import FrequencyResult
from blab.system_contract import QuantityResult, SystemFrequencyResult


def test_max_spl_applies_peak_xmax_and_rated_power_voltage_limit() -> None:
    frequency_hz = 100.0
    rms_excursion_m = 0.001
    velocity = 1j * 2.0 * np.pi * frequency_hz * rms_excursion_m
    pmax_w = 5.66**2 / 4.0

    frequencies, channels, curves = calculate_max_spl_curves(
        frequencies_hz=np.asarray([frequency_hz]),
        channel_names=np.asarray(["low"]),
        on_axis_pressure_pa=np.asarray([[2.0 + 0.0j]]),
        excitation_channel_names=np.asarray(["low"]),
        transducer_channel_names=np.asarray(["low"]),
        transducer_resistance_ohm=np.asarray([4.0]),
        diaphragm_velocity_m_per_s=np.asarray([[[velocity]]]),
        limits_by_channel={"low": MaxSplLimit(xmax_mm=2.0 * np.sqrt(2.0), pmax_w=pmax_w)},
        reference_voltage_v=2.83,
    )

    assert frequencies.tolist() == [frequency_hz]
    assert channels.tolist() == ["low"]
    assert curves[0, 0] == pytest.approx(20.0 * np.log10(4.0 / 20.0e-6), abs=1e-5)


def test_max_spl_parallel_channel_uses_the_most_restrictive_component() -> None:
    frequency_hz = 100.0
    velocities = 1j * 2.0 * np.pi * frequency_hz * np.asarray([0.0005, 0.001])

    _frequencies, _channels, curves = calculate_max_spl_curves(
        frequencies_hz=np.asarray([frequency_hz]),
        channel_names=np.asarray(["main"]),
        on_axis_pressure_pa=np.asarray([[1.0 + 0.0j]]),
        excitation_channel_names=np.asarray(["main", "main"]),
        transducer_channel_names=np.asarray(["main", "main"]),
        transducer_resistance_ohm=np.asarray([8.0, 8.0]),
        diaphragm_velocity_m_per_s=np.asarray([[velocities, np.zeros(2)]], dtype=np.complex128),
        limits_by_channel={"main": MaxSplLimit(xmax_mm=np.sqrt(2.0), pmax_w=1000.0)},
        reference_voltage_v=2.83,
    )

    # The second component reaches one-way peak Xmax at unity gain.
    assert curves[0, 0] == pytest.approx(20.0 * np.log10(1.0 / 20.0e-6), abs=1e-5)


def test_max_spl_limits_round_trip_and_ignore_invalid_entries() -> None:
    limits = {
        "low": MaxSplLimit(xmax_mm=5.5, pmax_w=450.0),
        "high": MaxSplLimit(xmax_mm=0.0, pmax_w=0.0),
    }

    payload = max_spl_limits_payload(limits)
    loaded = max_spl_limits_from_payload(payload | {"stale": {"xmax_mm": 0.0, "pmax_w": 1.0}})

    assert loaded == limits


def test_rated_resistance_uses_enabled_semi_inductance_series_resistance() -> None:
    assert transducer_rated_resistance_ohm(
        {
            "re_ohm": 6.0,
            "semi_inductance": {"enabled": True, "re_prime_ohm": 4.5},
        }
    ) == pytest.approx(4.5)


def test_live_max_spl_uses_raw_basis_not_channel_synthesis_correction() -> None:
    acoustic = LiveSolveDataset(
        polar_angle_deg=np.asarray([0.0]),
        channel_configs=(ChannelConfig(name="main", voltage_v=5.66, level_db=12.0),),
        flat_target_normalization_enabled=True,
        voltage_channel_names=frozenset({"main"}),
    )
    acoustic.add(
        FrequencyResult(
            freq_hz=100.0,
            horizontal_spl_norm_db=np.zeros(1),
            vertical_spl_norm_db=np.zeros(1),
            impedance=np.zeros((1, 2)),
            channel_names=np.asarray(["main"]),
            horizontal_pressure=np.asarray([[2.0 + 0.0j]]),
            vertical_pressure=np.asarray([[2.0 + 0.0j]]),
        )
    )
    motion = TransducerMotionDataset(
        excitation_channel_names=np.asarray(["main"]),
        transducer_names=np.asarray(["Woofer"]),
        transducer_channel_names=np.asarray(["main"]),
        transducer_resistance_ohm=np.asarray([4.0]),
    )
    motion.add(
        SystemFrequencyResult(
            freq_hz=100.0,
            excitation_port_ids=("port:woofer",),
            quantities=(
                QuantityResult(
                    id=DIAPHRAGM_VELOCITY_ID,
                    quantity="diaphragm_velocity",
                    unit="m/s",
                    axes=("excitation", "transducer"),
                    values=np.asarray([[1.0e-6j]]),
                ),
            ),
            diagnostics={"transducer_reference_voltage_v": 2.83},
        )
    )

    _frequencies, channels, curves = motion.as_max_spl_arrays(
        acoustic,
        {"main": MaxSplLimit(xmax_mm=100.0, pmax_w=2.83**2 / 4.0)},
        frozenset({"main"}),
    )

    assert channels.tolist() == ["main"]
    assert curves[0, 0] == pytest.approx(20.0 * np.log10(2.0 / 20.0e-6), abs=1e-5)
