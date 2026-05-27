import numpy as np
from types import SimpleNamespace

from blab.config import ChannelConfig, CrossoverConfig, RadiatorConfig
from blab.solver import (
    HornBEMSolver,
    _flat_target_correction,
    _split_frequencies_evenly,
    build_fibonacci_sphere_observation_points,
)


def test_split_frequencies_evenly_preserves_all_points() -> None:
    freqs = np.array([100.0, 200.0, 400.0, 800.0, 1600.0])
    chunks = _split_frequencies_evenly(freqs, worker_count=3)

    assert [len(chunk) for chunk in chunks] == [2, 2, 1]
    assert np.concatenate(chunks).tolist() == freqs.tolist()


def test_fibonacci_sphere_observation_points_shape_and_radius() -> None:
    points, theta, phi, r_distance = build_fibonacci_sphere_observation_points(
        32,
        distance_m=2.0,
        axial_offset_m=0.25,
    )

    assert points.shape == (3, 32)
    assert theta.shape == (32,)
    assert phi.shape == (32,)
    assert r_distance.shape == (32,)
    assert np.all((theta >= 0.0) & (theta <= np.pi))
    assert np.all((phi >= 0.0) & (phi < 2.0 * np.pi))
    centered = points.copy()
    centered[2, :] -= 0.25
    assert np.allclose(np.linalg.norm(centered, axis=0), 2.0)


def test_fibonacci_sphere_rejects_invalid_point_count() -> None:
    with np.testing.assert_raises(ValueError):
        build_fibonacci_sphere_observation_points(0, distance_m=2.0)


def test_linkwitz_riley_response_is_complex_and_bounded() -> None:
    crossover = CrossoverConfig(
        type="lowpass",
        filter="linkwitz_riley",
        order=4,
        frequency_hz=1200.0,
    )

    response = HornBEMSolver._crossover_response(None, crossover, 1200.0)

    assert isinstance(response, complex)
    assert 0.0 < abs(response) <= 1.0


def test_sixth_order_crossover_responses_are_supported() -> None:
    for filter_name in ("butterworth", "linkwitz_riley"):
        crossover = CrossoverConfig(
            type="lowpass",
            filter=filter_name,
            order=6,
            frequency_hz=1200.0,
        )

        HornBEMSolver._validate_crossover_config("HF", crossover)
        response = HornBEMSolver._crossover_response(None, crossover, 1200.0)

        assert isinstance(response, complex)
        assert 0.0 < abs(response) <= 1.0


def test_radiator_drive_multiplies_highpass_and_lowpass() -> None:
    radiator = RadiatorConfig(
        name="bandpass",
        tag=2,
        hpf=CrossoverConfig(type="highpass", filter="butterworth", order=2, frequency_hz=500.0),
        lpf=CrossoverConfig(type="lowpass", filter="butterworth", order=2, frequency_hz=5000.0),
    )

    drive = HornBEMSolver._radiator_drive(None, radiator, 1000.0)
    hpf = HornBEMSolver._crossover_response(None, radiator.hpf, 1000.0)
    lpf = HornBEMSolver._crossover_response(None, radiator.lpf, 1000.0)

    assert np.isclose(drive, hpf * lpf)


def test_radiator_drive_uses_channel_with_velocity_offset() -> None:
    channel = ChannelConfig(name="HF", level_db=-6.0)
    radiator = RadiatorConfig(name="dome", tag=2, channel="HF", velocity_offset_db=-3.0)
    solver = SimpleNamespace(channel_configs={"HF": channel})

    drive = HornBEMSolver._radiator_drive(solver, radiator, 1000.0)

    assert np.isclose(abs(drive), 10.0 ** (-9.0 / 20.0))


def test_radiator_validation_resolves_channels_before_attribute_is_set() -> None:
    solver = HornBEMSolver.__new__(HornBEMSolver)
    solver.cfg = SimpleNamespace(
        channels=(ChannelConfig(name="HF"),),
    )
    solver.mesh_names = ("mesh",)

    HornBEMSolver._validate_radiator_configs(
        solver,
        (RadiatorConfig(name="dome", mesh="mesh", tag=2, channel="HF"),),
    )


def test_flat_target_correction_is_inverse_on_axis_pressure_magnitude() -> None:
    assert np.isclose(_flat_target_correction(2.0 + 0.0j), 0.5)
    assert _flat_target_correction(0.0 + 0.0j) == 1.0
    assert _flat_target_correction(np.nan + 0.0j) == 1.0
