import pytest

from blab.acoustic_materials import (
    miki_wall_impedance_parameters,
    region_bulk_loss_factor,
    wall_impedance_parameters,
)


def test_region_bulk_loss_factor_defaults_and_validates() -> None:
    assert region_bulk_loss_factor({}) == 0.0
    assert region_bulk_loss_factor({"bulk_loss_factor": 0.02}) == pytest.approx(0.02)
    with pytest.raises(ValueError, match="between 0 and 1"):
        region_bulk_loss_factor({"bulk_loss_factor": 1.01})


def test_miki_wall_parameters_use_polyfill_defaults_and_round_trip() -> None:
    parameters = miki_wall_impedance_parameters()
    treatment = wall_impedance_parameters(parameters)

    assert treatment is not None
    assert treatment["model"] == "miki"
    assert treatment["thickness_m"] == pytest.approx(0.03)
    assert treatment["flow_resistivity_pa_s_per_m2"] == pytest.approx(5000.0)


@pytest.mark.parametrize(
    "parameters",
    (
        {"wall_impedance": {"model": "unknown"}},
        {"wall_impedance": {"thickness_m": 0.0}},
        {"wall_impedance": {"flow_resistivity_pa_s_per_m2": -1.0}},
    ),
)
def test_wall_impedance_parameters_reject_invalid_values(parameters) -> None:
    with pytest.raises(ValueError):
        wall_impedance_parameters(parameters)
