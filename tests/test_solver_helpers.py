import numpy as np

from blab.config import CrossoverConfig, RadiatorConfig
from blab.solver import HornBEMSolver, _split_frequencies_evenly, build_fibonacci_sphere_observation_points


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
