from types import SimpleNamespace

import numpy as np
import pytest

from blab.deploy_acoustic_loading import normalized_acoustic_loading


def fixture():
    package = SimpleNamespace(
        manifest={
            "medium": {"density_kg_per_m3": 2, "sound_speed_m_per_s": 100},
            "physical_system": {
                "components": [
                    {
                        "id": name,
                        "kind": "electrodynamic_transducer",
                        "parameters": {
                            "bl_n_per_a": 2,
                            "rms_n_s_per_m": 3,
                            "cms_m_per_n": 0.001,
                            "mmd_kg": 0.1,
                        },
                    }
                    for name in ("a", "b")
                ],
                "metadata": {
                    "acoustic_impedance_normalization": {
                        "a": {"effective_area_m2": 0.01},
                        "b": {"effective_area_m2": 0.02},
                    }
                },
            },
        },
    )
    velocity = np.array([1 + 1j, 2 - 1j])
    active = np.array([-2 - 4j, 12 + 8j])
    omega = 2 * np.pi * 100
    zm = 3 + 1j * (1 / (omega * 0.001) - omega * 0.1)
    current = (active + zm) * velocity / 2
    result = {
        "diagnostics": {
            "transducer_velocity": [{"real": velocity.real.tolist(), "imag": velocity.imag.tolist()}],
            "transducer_current": [{"real": current.real.tolist(), "imag": current.imag.tolist()}],
        }
    }
    request = {"transducers": [{"id": "source:a"}, {"id": "source:b"}]}
    return package, request, result, velocity


def test_normalization_and_sign():
    package, request, result, _ = fixture()
    actual = normalized_acoustic_loading(package, request, result, 100)
    assert actual["resistance"] == pytest.approx([-1, 3])
    assert actual["reactance"] == pytest.approx([2, -2])


@pytest.mark.parametrize("velocity", [0.0, 1e-14, float("nan")])
def test_undefined_velocity_is_a_gap(velocity):
    package, request, result, _ = fixture()
    result["diagnostics"]["transducer_velocity"][0] = {"real": [velocity, 1], "imag": [0, 0]}
    actual = normalized_acoustic_loading(package, request, result, 100)
    assert all(actual[key][0] is None for key in ("resistance", "reactance"))
    assert (actual["pressure_real_pa"][0] is not None) == bool(np.isfinite(velocity))


def test_differential_pressure_is_complex_force_over_area():
    package, request, result, velocity = fixture()
    actual = normalized_acoustic_loading(package, request, result, 100)
    pressure = np.array([-2 - 4j, 12 + 8j]) * velocity / [0.01, 0.02]
    assert actual["pressure_real_pa"] == pytest.approx(pressure.real)
    assert actual["pressure_imag_pa"] == pytest.approx(-pressure.imag)


def test_stationary_driver_pressure_survives_without_medium():
    package, request, result, _ = fixture()
    package.manifest["medium"] = {}
    result["diagnostics"]["transducer_velocity"][0] = {"real": [0, 1], "imag": [0, 0]}
    result["diagnostics"]["transducer_current"][0] = {"real": [3, 0], "imag": [4, 0]}
    actual = normalized_acoustic_loading(package, request, result, 100)
    assert actual["pressure_real_pa"][0] == pytest.approx(600)
    assert actual["pressure_imag_pa"][0] == pytest.approx(-800)
    assert actual["resistance"][0] is None


def test_legacy_package_can_plot_array_without_reference():
    package, request, result, _ = fixture()
    actual = normalized_acoustic_loading(package, request, result, 100)
    assert actual["resistance"] == pytest.approx([-1, 3])


def test_missing_area():
    package, request, result, _ = fixture()
    package.manifest["physical_system"]["metadata"] = {}
    actual = normalized_acoustic_loading(package, request, result, 100)
    assert all(values == [None, None] for values in actual.values())


def test_multiple_cabinets_keep_scene_order():
    package, request, result, _ = fixture()
    request["transducers"] += [{"id": "other:a"}, {"id": "other:b"}]
    for key in ("transducer_velocity", "transducer_current"):
        result["diagnostics"][key] *= 2
    actual = normalized_acoustic_loading(package, request, result, 100)
    assert actual["resistance"] == pytest.approx([-1, 3, -1, 3])
