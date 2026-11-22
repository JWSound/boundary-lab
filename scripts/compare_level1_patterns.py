"""Compare an adaptive Level 1 archive with an independently solved discrete export."""

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from blab.adaptive_pattern import PATTERN_EVALUATOR, interpolate_pressure, pattern_errors
from blab.phasor import convert_phasor
from blab.speaker_package import validate_speaker_package


def read_package(path):
    manifest = validate_speaker_package(path)
    with zipfile.ZipFile(path) as archive:
        with np.load(io.BytesIO(archive.read(manifest["files"]["patterns"]["path"]))) as data:
            return manifest, dict(data)


def compare(adaptive, discrete):
    model, data = read_package(adaptive)
    reference, truth = read_package(discrete)
    descriptor = model["files"]["patterns"]
    if descriptor.get("evaluator") != PATTERN_EVALUATOR or "evaluator" in reference["files"]["patterns"]:
        raise ValueError("Expected an adaptive export followed by a discrete export.")
    for key in ("medium", "coordinate_system", "excitation_port_ids"):
        if model[key] != reference[key]:
            raise ValueError(f"Package mismatch in {key}.")
    for key in ("points_m", "directions_xyz", "radius_m"):
        np.testing.assert_array_equal(data[key], truth[key])
    frequencies = data["frequencies_hz"]
    pressure = convert_phasor(data["pressure_pa"], model["phasor_convention"])
    direct = convert_phasor(truth["pressure_pa"], reference["phasor_convention"])
    delay = data["radius_m"] / model["medium"]["sound_speed_m_per_s"]
    records = []
    for f, actual in zip(truth["frequencies_hz"], direct, strict=True):
        i = int(np.clip(np.searchsorted(frequencies, f) - 1, 0, frequencies.size - 2))
        lo, hi = frequencies[i : i + 2]
        mix = (f - lo) / (hi - lo)
        a, b = pressure[i : i + 2]
        predicted = interpolate_pressure(f, lo, hi, a, b, delay)
        rms, peak = pattern_errors(actual, predicted)
        plain_rms, plain_peak = pattern_errors(actual, (1 - mix) * a + mix * b)
        records.append(dict(frequency_hz=f, rms=rms, peak=peak, plain_rms=plain_rms, plain_peak=plain_peak))
    return {
        "direct_frequency_count": descriptor["direct_frequency_count"],
        "knots": len(frequencies),
        "directions": len(delay),
        "ports": pressure.shape[1],
        "reference_samples": len(records),
        "off_knot_samples": sum(not np.isclose(frequencies, r["frequency_hz"], rtol=1e-10).any() for r in records),
        "adaptive_bytes": adaptive.stat().st_size,
        "discrete_bytes": discrete.stat().st_size,
        "worst": {
            metric: max(records, key=lambda r: r[metric]) for metric in ("rms", "peak", "plain_rms", "plain_peak")
        },
        "records": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adaptive", type=Path)
    parser.add_argument("discrete", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.adaptive, args.discrete), indent=2))
