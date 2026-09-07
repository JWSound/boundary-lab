from types import SimpleNamespace

import numpy as np
import pytest

from blab.deploy_acoustic_loading import normalized_acoustic_loading


def fixture():
    # Deliberately reversed reference order; off-diagonal terms are essential.
    reference = np.array([[8 - 4j, 2 - 1j], [2 - 1j, 4 - 2j]])
    package = SimpleNamespace(
        manifest={
            "medium": {"density_kg_per_m3": 2, "sound_speed_m_per_s": 100},
            "physical_system": {
                "components": [
                    {"id": name, "kind": "electrodynamic_transducer", "parameters": {
                        "bl_n_per_a": 2, "rms_n_s_per_m": 3, "cms_m_per_n": 0.001, "mmd_kg": 0.1,
                    }} for name in ("a", "b")
                ],
                "metadata": {"acoustic_impedance_normalization": {
                    "a": {"effective_area_m2": 0.01}, "b": {"effective_area_m2": 0.02},
                }},
            },
            "files": {"isolated_acoustic_impedance": {"transducer_ids": ["b", "a"]}},
        },
        isolated_acoustic_impedance={
            "frequencies_hz": np.array([100.0]),
            "available_frequency_mask": np.array([True]),
            "acoustic_impedance_n_s_per_m": reference[None],
            "effective_area_m2": np.array([0.02, 0.01]),
        },
    )
    velocity = np.array([1 + 1j, 2 - 1j])
    active = np.array([-2 - 4j, 12 + 8j])
    omega = 2 * np.pi * 100
    zm = 3 + 1j * (1 / (omega * 0.001) - omega * 0.1)
    current = (active + zm) * velocity / 2
    result = {"diagnostics": {
        "transducer_velocity": [{"real": velocity.real.tolist(), "imag": velocity.imag.tolist()}],
        "transducer_current": [{"real": current.real.tolist(), "imag": current.imag.tolist()}],
    }}
    request = {"transducers": [{"id": "source:a"}, {"id": "source:b"}]}
    return package, request, result, velocity


def test_normalization_sign_and_matched_motion_reference():
    package, request, result, velocity = fixture()
    actual = normalized_acoustic_loading(package, request, result, 100)
    assert actual["resistance"] == pytest.approx([-1, 3])
    assert actual["reactance"] == pytest.approx([2, -2])
    isolated = (np.array([[4 - 2j, 2 - 1j], [2 - 1j, 8 - 4j]]) @ velocity) / velocity / [2, 4]
    assert actual["isolated_resistance"] == pytest.approx(isolated.real)
    assert actual["isolated_reactance"] == pytest.approx(-isolated.imag)
    assert not np.allclose(actual["isolated_resistance"], [2, 2])  # not just the diagonal


@pytest.mark.parametrize("velocity", [0.0, 1e-14, float("nan")])
def test_undefined_velocity_is_a_gap(velocity):
    package, request, result, _ = fixture()
    result["diagnostics"]["transducer_velocity"][0] = {"real": [velocity, 1], "imag": [0, 0]}
    actual = normalized_acoustic_loading(package, request, result, 100)
    assert all(values[0] is None for values in actual.values())


def test_legacy_package_can_plot_array_without_reference():
    package, request, result, _ = fixture()
    package.isolated_acoustic_impedance = None
    actual = normalized_acoustic_loading(package, request, result, 100)
    assert actual["resistance"] == pytest.approx([-1, 3])
    assert actual["isolated_resistance"] == [None, None]


def test_unavailable_frequency_and_missing_area():
    package, request, result, _ = fixture()
    package.isolated_acoustic_impedance["available_frequency_mask"][0] = False
    assert normalized_acoustic_loading(package, request, result, 100)["isolated_resistance"] == [None, None]
    assert normalized_acoustic_loading(package, request, result, 200)["isolated_resistance"] == [None, None]
    package.isolated_acoustic_impedance = None
    package.manifest["physical_system"]["metadata"] = {}
    assert normalized_acoustic_loading(package, request, result, 100)["resistance"] == [None, None]


def test_multiple_cabinets_keep_scene_order():
    package, request, result, _ = fixture()
    request["transducers"] += [{"id": "other:a"}, {"id": "other:b"}]
    for key in ("transducer_velocity", "transducer_current"):
        result["diagnostics"][key] *= 2
    actual = normalized_acoustic_loading(package, request, result, 100)
    assert actual["resistance"] == pytest.approx([-1, 3, -1, 3])
