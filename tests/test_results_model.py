from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from blab.config import ChannelConfig
from blab.live import LiveSolveDataset, TransducerMotionDataset
from blab.physical_model import AcousticRegionKind, ExcitationPortKind, MeshPurpose
from blab.solve_results import (
    BEM_BOUNDARY_DOMAIN_ID,
    BEM_BOUNDARY_NEUMANN_ID,
    BEM_BOUNDARY_PRESSURE_ID,
    DIAPHRAGM_VELOCITY_ID,
    HORIZONTAL_POLAR_PRESSURE_ID,
    VOICE_COIL_CURRENT_ID,
    ResultDomain,
    SolvedQuantity,
    SolvedSystem,
    SolvedSystemBuilder,
    SolveProvenance,
    bem_boundary_result_domain,
    bem_boundary_traces_from_solved_system,
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


def test_builder_finalization_freezes_and_transfers_quantity_buffers() -> None:
    builder = SolvedSystemBuilder(
        frequencies_hz=(100.0,),
        excitation_ids=("port:driver",),
        provenance=SolveProvenance(backend_id="beat_cpu", solve_kind="coupled_bem_fem"),
    )
    builder.add(_pressure_result(100.0, [1.0 + 2.0j]))
    source_buffer = builder._quantities[HORIZONTAL_POLAR_PRESSURE_ID].values

    solved = builder.finalize(status="completed")

    assert np.shares_memory(solved.quantity(HORIZONTAL_POLAR_PRESSURE_ID).values, source_buffer)
    assert not source_buffer.flags.writeable
    with pytest.raises(RuntimeError, match="finalized"):
        builder.add(_pressure_result(100.0, [2.0 + 3.0j]))


def test_bem_boundary_domain_matches_solver_mesh_concatenation(tmp_path) -> None:
    mesh_a = tmp_path / "a.msh"
    mesh_b = tmp_path / "b.msh"
    mesh_a.write_text(
        """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
3
10 0 0 0
20 1 0 0
30 0 1 0
$EndNodes
$Elements
1
1 2 2 7 1 10 20 30
$EndElements
""",
        encoding="utf-8",
    )
    mesh_b.write_text(
        """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
3
4 -0.000000012 0 0
8 0 1 0
12 0 0 1
$EndNodes
$Elements
1
1 2 2 9 1 12 4 8
$EndElements
""",
        encoding="utf-8",
    )
    system = SimpleNamespace(
        regions=(
            SimpleNamespace(
                id="region:exterior",
                kind=AcousticRegionKind.UNBOUNDED_AIR,
                mesh_ids=("mesh:a", "mesh:b"),
            ),
        ),
        meshes=(
            SimpleNamespace(
                id="mesh:a",
                purpose=MeshPurpose.BEM_SURFACE,
                file=str(mesh_a),
                scale_to_m=2.0,
                translation_m=(1.0, 0.0, 0.0),
            ),
            SimpleNamespace(
                id="mesh:b",
                purpose=MeshPurpose.BEM_SURFACE,
                file=str(mesh_b),
                scale_to_m=1.0,
                translation_m=(0.0, 2.0, 0.0),
            ),
        ),
    )

    domain = bem_boundary_result_domain(system, symmetry="xy")

    assert domain.id == BEM_BOUNDARY_DOMAIN_ID
    assert domain.metadata["mesh_ids"] == ["mesh:a", "mesh:b"]
    assert domain.metadata["node_offsets"] == [0, 3]
    assert domain.metadata["face_offsets"] == [0, 1]
    assert domain.metadata["node_counts"] == [3, 3]
    assert domain.metadata["face_counts"] == [1, 1]
    assert domain.metadata["symmetry"] == "xy"
    assert domain.topology["triangles"].tolist() == [[0, 1, 2], [5, 3, 4]]
    assert domain.topology["source_physical_tag"].tolist() == [7, 9]
    np.testing.assert_allclose(domain.coordinates["points_m"][0], [1.0, 0.0, 0.0])
    assert domain.coordinates["points_m"][3, 0] == 0.0
    np.testing.assert_allclose(domain.coordinates["points_m"][4], [0.0, 3.0, 0.0])


def test_bem_boundary_trace_view_validates_and_sorts_available_results() -> None:
    domain = ResultDomain(
        id=BEM_BOUNDARY_DOMAIN_ID,
        kind="bem_boundary",
        dimensions=("bem_node", "bem_face"),
        coordinates={"points_m": np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])},
        topology={"triangles": np.asarray([[0, 1, 0]])},
        metadata={"symmetry": "x"},
    )
    builder = SolvedSystemBuilder(
        frequencies_hz=(1000.0, 100.0),
        excitation_ids=("port:a", "port:b"),
        provenance=SolveProvenance(backend_id="beat_cpu", solve_kind="coupled_bem_fem"),
        domains=(domain,),
    )
    for frequency, value in ((100.0, 1.0), (1000.0, 10.0)):
        builder.add(
            SystemFrequencyResult(
                freq_hz=frequency,
                excitation_port_ids=("port:a", "port:b"),
                quantities=(
                    QuantityResult(
                        id=BEM_BOUNDARY_PRESSURE_ID,
                        quantity="bem_boundary_pressure",
                        unit="Pa",
                        target_id=BEM_BOUNDARY_DOMAIN_ID,
                        axes=("excitation", "bem_node"),
                        values=np.full((2, 2), value, dtype=np.complex64),
                    ),
                    QuantityResult(
                        id=BEM_BOUNDARY_NEUMANN_ID,
                        quantity="bem_boundary_neumann",
                        unit="Pa/m",
                        target_id=BEM_BOUNDARY_DOMAIN_ID,
                        axes=("excitation", "bem_face"),
                        values=np.full((2, 1), value * 1j, dtype=np.complex64),
                    ),
                ),
            )
        )

    solved = builder.finalize(status="completed")
    traces = bem_boundary_traces_from_solved_system(solved)

    assert traces is not None
    assert traces.frequencies_hz.tolist() == [100.0, 1000.0]
    assert traces.frequency_indices.tolist() == [1, 0]
    assert traces.symmetry == "x"
    pressure, normal_derivative = traces.excitation_basis(100.0)
    np.testing.assert_allclose(pressure, 1.0)
    np.testing.assert_allclose(normal_derivative, 1.0j)
    assert traces.trace_storage_nbytes == 96


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


def test_live_transducer_excursion_uses_normalized_channel_drive_before_magnitude() -> None:
    frequency_hz = 1.0 / (2.0 * np.pi)
    acoustic = LiveSolveDataset(
        polar_angle_deg=np.asarray([0.0], dtype=np.float32),
        channel_configs=(ChannelConfig(name="main"),),
        flat_target_normalization_enabled=True,
    )
    acoustic.add(
        FrequencyResult(
            freq_hz=frequency_hz,
            horizontal_spl_norm_db=np.zeros(1),
            vertical_spl_norm_db=np.zeros(1),
            impedance=np.zeros((1, 2)),
            channel_names=np.asarray(["main"]),
            horizontal_pressure=np.asarray([[2.0 + 0.0j]], dtype=np.complex64),
            vertical_pressure=np.asarray([[2.0 + 0.0j]], dtype=np.complex64),
        )
    )
    motion = TransducerMotionDataset(
        excitation_channel_names=np.asarray(["main"]),
        transducer_names=np.asarray(["Woofer"]),
    )
    motion.add(
        SystemFrequencyResult(
            freq_hz=frequency_hz,
            excitation_port_ids=("port:woofer",),
            quantities=(
                QuantityResult(
                    id=DIAPHRAGM_VELOCITY_ID,
                    quantity="diaphragm_velocity",
                    unit="m/s",
                    axes=("excitation", "transducer"),
                    values=np.asarray([[2.0j]], dtype=np.complex64),
                ),
            ),
        )
    )

    freqs, names, excursion_mm = motion.as_excursion_arrays(acoustic)

    assert freqs.tolist() == pytest.approx([frequency_hz])
    assert names.tolist() == ["Woofer"]
    # The 2 Pa on-axis channel basis applies a 0.5 normalized correction.
    assert excursion_mm[0, 0] == pytest.approx(1000.0)


def test_live_transducer_excursion_sums_complex_excitation_bases_before_magnitude() -> None:
    acoustic = LiveSolveDataset(
        polar_angle_deg=np.asarray([0.0], dtype=np.float32),
        channel_configs=(ChannelConfig(name="main"),),
        flat_target_normalization_enabled=False,
    )
    acoustic.add(
        FrequencyResult(
            freq_hz=100.0,
            horizontal_spl_norm_db=np.zeros(1),
            vertical_spl_norm_db=np.zeros(1),
            impedance=np.zeros((2, 2)),
            channel_names=np.asarray(["main"]),
            horizontal_pressure=np.asarray([[1.0 + 0.0j]], dtype=np.complex64),
            vertical_pressure=np.asarray([[1.0 + 0.0j]], dtype=np.complex64),
        )
    )
    motion = TransducerMotionDataset(
        excitation_channel_names=np.asarray(["main", "main"]),
        transducer_names=np.asarray(["Woofer"]),
    )
    motion.add(
        SystemFrequencyResult(
            freq_hz=100.0,
            excitation_port_ids=("port:a", "port:b"),
            quantities=(
                QuantityResult(
                    id=DIAPHRAGM_VELOCITY_ID,
                    quantity="diaphragm_velocity",
                    unit="m/s",
                    axes=("excitation", "transducer"),
                    values=np.asarray([[1.0j], [-1.0j]], dtype=np.complex64),
                ),
            ),
        )
    )

    _freqs, _names, excursion_mm = motion.as_excursion_arrays(acoustic)

    assert excursion_mm[0, 0] == pytest.approx(0.0)


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
