"""Adaptive frequency sampling of complex pressure on the equal-area Fibonacci sphere."""

from __future__ import annotations

import numpy as np

PATTERN_EVALUATOR = "delay_compensated_linear_v1"


def interpolate_pressure(frequency, lower, upper, pressure_lower, pressure_upper, delay_s):
    """Evaluate positive-time pressure at the original observation radii."""
    if not lower <= frequency <= upper or lower >= upper:
        raise ValueError("Frequency must lie inside a nonempty interpolation interval.")
    mix = (frequency - lower) / (upper - lower)
    phase = 2j * np.pi * delay_s
    return (1 - mix) * pressure_lower * np.exp(phase * (lower - frequency)) + mix * pressure_upper * np.exp(
        phase * (upper - frequency)
    )


def pattern_errors(actual, predicted):
    """Worst per-port relative sphere RMS and peak-normalized directional error."""
    delta = np.abs(actual - predicted)
    magnitude = np.abs(actual)

    # Equal weights on the Fibonacci sphere. A silent port must also predict silence.
    def relative(numerator, denominator):
        ratios = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        ratios[(denominator == 0) & (numerator > 0)] = np.inf
        return float(ratios.max())

    return (
        relative(np.linalg.norm(delta, axis=1), np.linalg.norm(magnitude, axis=1)),
        relative(delta.max(axis=1), magnitude.max(axis=1)),
    )


def train_pattern(band, delay_s, solve_batch, *, tolerance=0.02, max_solves=512):
    """Retain adaptive knots and their final midpoints, then audit the interpolant off-centre."""
    lo, hi = map(float, band)
    if not np.isfinite([lo, hi, tolerance]).all() or not 0 < lo < hi or not 0 < tolerance < 1:
        raise ValueError("Expected a positive increasing frequency band and a fractional tolerance.")
    delay_s = np.asarray(delay_s, dtype=float)
    if delay_s.ndim != 1 or not delay_s.size or not np.isfinite(delay_s).all() or np.any(delay_s <= 0):
        raise ValueError("Invalid sphere propagation delays.")
    cache = {}

    def fetch(frequencies):
        missing = sorted(set(frequencies).difference(cache))
        if len(cache) + len(missing) > max_solves:
            raise ValueError(f"Adaptive pattern exceeded its budget of {max_solves} direct frequencies.")
        if missing:
            values = np.asarray(solve_batch(missing))
            if (
                values.ndim != 3
                or values.shape[0] != len(missing)
                or values.shape[2] != delay_s.size
                or not np.iscomplexobj(values)
                or not np.isfinite(values).all()
            ):
                raise ValueError("Invalid complex sphere pressure from direct solve.")
            cache.update(zip(missing, values, strict=True))

    def check_intervals(knots, *, audit):
        intervals = list(zip(knots[:-1], knots[1:], strict=True))
        probes = []
        for i, (left, right) in enumerate(intervals):
            fraction = 0.5
            if audit:
                fraction = 0.37 if i % 2 == 0 else 0.63
            probe = left * (right / left) ** fraction
            if not left < probe < right:
                raise ValueError("Frequency interval is too narrow for distinct pattern checks.")
            probes.append(probe)
        fetch(probes)
        failures = []
        checks = []
        for (left, right), f in zip(intervals, probes, strict=True):
            if audit:
                # The exported interval is split at the directly solved midpoint.
                midpoint = left * (right / left) ** 0.5
                if f < midpoint:
                    right = midpoint
                else:
                    left = midpoint
            prediction = interpolate_pressure(f, left, right, cache[left], cache[right], delay_s)
            rms, peak = pattern_errors(cache[f], prediction)
            checks.append(dict(frequency_hz=f, rms_relative=rms, peak_relative=peak))
            if max(rms, peak) > tolerance:
                failures.append(f)
        return failures, checks

    knots = np.geomspace(lo, hi, 9).tolist()
    fetch(knots)
    while True:
        failures, midpoint_checks = check_intervals(knots, audit=False)
        if not failures:
            failures, audit_checks = check_intervals(knots, audit=True)
            if not failures:
                break
        knots = sorted([*knots, *failures])
    knots = sorted([*knots, *(row["frequency_hz"] for row in midpoint_checks)])
    return (
        np.asarray(knots),
        np.stack([cache[f] for f in knots]),
        {
            "evaluator": PATTERN_EVALUATOR,
            "frequency_band_hz": [lo, hi],
            "relative_tolerance": tolerance,
            "direct_frequency_count": len(cache),
            "midpoint_checks": midpoint_checks,
            "audit_checks": audit_checks,
        },
    )
