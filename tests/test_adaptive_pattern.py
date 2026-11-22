import numpy as np
import pytest

from blab.adaptive_pattern import interpolate_pressure, pattern_errors, train_pattern


def test_reference_delay_linear_complex_response_and_silent_port():
    delays = np.array([1, 2, 3]) / 343

    def direct(frequencies):
        f = np.array(frequencies)[:, None]
        p = (1 + 0.02j * f) * np.exp(-2j * np.pi * f * delays)
        return np.stack((p, p * 0), axis=1)

    knots, samples, report = train_pattern((20, 500), delays, direct)
    assert len(knots) == 9
    assert report["direct_frequency_count"] == 25
    for f in [20, 162.7, 499.9, 500]:
        i = np.clip(np.searchsorted(knots, f) - 1, 0, len(knots) - 2)
        predicted = interpolate_pressure(f, *knots[i : i + 2], *samples[i : i + 2], delays)
        np.testing.assert_allclose(predicted, direct([f])[0], atol=1e-12)
    with pytest.raises(ValueError):
        interpolate_pressure(501, *knots[-2:], *samples[-2:], delays)


def test_adapts_to_quiet_port_resonance_and_enforces_budget():
    delay = np.array([0.01, 0.02])

    def solve(frequencies):
        f = np.array(frequencies)[:, None]
        common = np.exp(-2j * np.pi * f * delay)
        return np.stack((common, common * 1e-6 / (1 + 1j * (f - 160) / 5)), axis=1)

    knots, _, report = train_pattern((100, 250), delay, solve)
    assert sum((knots > 150) & (knots < 170)) > 5
    assert all(row["peak_relative"] <= 0.02 for row in report["audit_checks"])
    assert set(knots).isdisjoint(row["frequency_hz"] for row in report["audit_checks"])
    assert np.isinf(pattern_errors(np.zeros((1, 2)), np.ones((1, 2)))).all()
    with pytest.raises(ValueError, match="budget"):
        train_pattern((100, 250), delay, solve, max_solves=10)
