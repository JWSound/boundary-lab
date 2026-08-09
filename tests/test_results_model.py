from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from blab.live import LiveSolveDataset
from blab.physical_model import ExcitationPortKind
from blab.solve_results import (
    HORIZONTAL_POLAR_PRESSURE_ID,
    VOICE_COIL_CURRENT_ID,
    ResultDomain,
    SolvedQuantity,
    SolvedSystem,
    SolvedSystemBuilder,
    SolveProvenance,
    electrical_input_impedance,
    legacy_result_domains,
    legacy_result_to_system_result,
    velocity_to_excursion,
)
from blab.solvers.base import FrequencyResult
from blab.system_contract import QuantityResult, SystemFrequencyResult


def _pressure_result(frequency_hz: float, values: list[complex]) -> SystemFrequencyResult:
    return SystemFrequencyResult(
        freq_hz=frequency_hz,
        excitation_port_ids=("port:driver",),
        quantities=(
            QuantityResult(
                id=HORIZONTAL_POLAR_PRESSURE_ID,
                quantity="exterior_pressure",
                unit="Pa",
                target_id="observation:horizontal-polar",
                axes=("excitation", "observation"),
                values=np.asarray([values], dtype=np.complex64),
            ),
        ),
        diagnostics={"relative_residual": frequency_hz / 1e6},
    )


def test_builder_stacks_out_of_order_results_on_the_requested_frequency_axis() -> None:
    builder = SolvedSystemBuilder(
        frequencies_hz=(100.0, 1000.0),
        excitation_ids=("port:driver",),
        provenance=SolveProvenance(backend_id="beat_cpu", solve_kind="coupled_bem_fem"),
        domains=(
            ResultDomain(
                id="observation:horizontal-polar",
                kind="polar_observation",
                dimensions=("observation",),
                coordinates={"angle_deg": np.asarray([-90.0, 0.0])},
            ),
        ),
    )

    builder.add(_pressure_result(1000.0, [3.0 + 4.0j, 5.0 + 6.0j]))
    builder.add(_pressure_result(100.0, [1.0 + 2.0j, 2.0 + 3.0j]))
    solved = builder.snapshot(status="completed")

    pressure = solved.quantity(HORIZONTAL_POLAR_PRESSURE_ID)
    assert solved.complete
    assert solved.solved_count == 2
    assert pressure.dimensions == ("frequency", "excitation", "observation")
    assert pressure.values.shape == (2, 1, 2)
    assert pressure.values[0, 0, 0] == 1.0 + 2.0j
    assert pressure.values[1, 0, 0] == 3.0 + 4.0j
    assert not pressure.values.flags.writeable
    assert solved.diagnostics_by_frequency[0]["relative_residual"] == pytest.approx(0.0001)


def test_builder_rejects_quantity_shape_changes_between_frequencies() -> None:
    builder = SolvedSystemBuilder(
        frequencies_hz=(100.0, 1000.0),
        excitation_ids=("port:driver",),
        provenance=SolveProvenance(backend_id="beat_cpu", solve_kind="coupled_bem_fem"),
    )
    builder.add(_pressure_result(100.0, [1.0 + 0.0j]))

    with pytest.raises(ValueError, match="changed shape or semantics"):
        builder.add(_pressure_result(1000.0, [1.0 + 0.0j, 2.0 + 0.0j]))


def test_legacy_result_adapter_preserves_complex_pressure_and_impedance() -> None:
    dataset = LiveSolveDataset(
        polar_angle_deg=np.asarray([-90.0, 0.0, 90.0]),
        radiator_names=np.asarray(["woofer"]),
    )
    result = FrequencyResult(
        freq_hz=1000.0,
        horizontal_spl_norm_db=np.zeros(3),
        vertical_spl_norm_db=np.zeros(3),
        impedance=np.asarray([[2.0, -3.0]], dtype=np.float32),
        channel_names=np.asarray(["main"]),
        horizontal_pressure=np.ones((1, 3), dtype=np.complex64),
        vertical_pressure=np.full((1, 3), 2.0j, dtype=np.complex64),
    )

    canonical = legacy_result_to_system_result(result)
    domains = {domain.id: domain for domain in legacy_result_domains(dataset)}
    quantities = {quantity.id: quantity for quantity in canonical.quantities}

    assert canonical.excitation_port_ids == ("main",)
    assert quantities[HORIZONTAL_POLAR_PRESSURE_ID].values.shape == (1, 3)
    assert quantities["acoustic:radiation-impedance"].values.tolist() == [2.0 - 3.0j]
    assert domains["observation:horizontal-polar"].coordinates["angle_deg"].tolist() == [-90.0, 0.0, 90.0]


def test_velocity_to_excursion_uses_the_solver_phasor_convention() -> None:
    velocity = SolvedQuantity(
        id="mechanical:diaphragm-velocity",
        quantity="diaphragm_velocity",
        unit="m/s",
        dimensions=("frequency", "excitation", "transducer"),
        values=np.asarray([[[1.0j]]], dtype=np.complex64),
        available_frequency_mask=np.asarray([True]),
    )

    excursion = velocity_to_excursion(velocity, np.asarray([1.0 / (2.0 * np.pi)]))

    assert excursion.unit == "m"
    assert excursion.values[0, 0, 0] == pytest.approx(-1.0 + 0.0j)


def test_electrical_impedance_uses_the_matching_voltage_basis_current() -> None:
    current_values = np.asarray(
        [
            [[0.5 + 0.0j, 9.0 + 0.0j], [8.0 + 0.0j, 0.25 + 0.0j]],
            [[1.0 + 0.0j, 9.0 + 0.0j], [8.0 + 0.0j, 0.5 + 0.0j]],
        ],
        dtype=np.complex64,
    )
    current = SolvedQuantity(
        id=VOICE_COIL_CURRENT_ID,
        quantity="voice_coil_current",
        unit="A",
        dimensions=("frequency", "excitation", "transducer"),
        values=current_values,
        metadata={"component_ids": ["component:a", "component:b"]},
        available_frequency_mask=np.asarray([True, True]),
    )
    ports = (
        SimpleNamespace(id="port:a", component_id="component:a", kind=ExcitationPortKind.VOLTAGE),
        SimpleNamespace(id="port:b", component_id="component:b", kind=ExcitationPortKind.VOLTAGE),
    )
    solved = SolvedSystem(
        run_id="run",
        provenance=SolveProvenance(backend_id="beat_cpu", solve_kind="coupled_bem_fem"),
        frequencies_hz=np.asarray([100.0, 200.0]),
        excitation_ids=("port:a", "port:b"),
        domains={},
        quantities={VOICE_COIL_CURRENT_ID: current},
        completion_mask=np.asarray([True, True]),
        diagnostics_by_frequency=(
            {"transducer_reference_voltage_v": 2.0},
            {"transducer_reference_voltage_v": 2.0},
        ),
        status="completed",
        compiled_system=SimpleNamespace(excitation_ports=ports),
    )

    impedance = electrical_input_impedance(solved)

    assert impedance.metadata["excitation_port_ids"] == ["port:a", "port:b"]
    assert impedance.values.tolist() == [[4.0 + 0.0j, 8.0 + 0.0j], [2.0 + 0.0j, 4.0 + 0.0j]]
