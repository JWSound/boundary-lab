import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from blab.physical_compiler import PhysicalModelCompileError, PhysicalSystemCompiler
from blab.physical_model import (
    AcousticInterface,
    AcousticRegion,
    AcousticRegionKind,
    AssumptionStatus,
    Boundary,
    BoundaryKind,
    ComponentKind,
    ExcitationPort,
    ExcitationPortKind,
    MeshPurpose,
    MeshResource,
    PhysicalComponent,
    PhysicalGroupRef,
    PhysicalSystem,
    physical_system_from_dict,
    physical_system_to_dict,
)
from blab.solvers.beat_engine_backend import DEFAULT_BEAT_ENGINE_CUDA_PROJECT
from blab.solvers.coupled_backend import (
    DEFAULT_TRANSDUCER_REFERENCE_VOLTAGE_V,
    CoupledProductionBackend,
    CoupledReferenceBackend,
)
from blab.system_contract import (
    OutputRequest,
    QuantityResult,
    SystemFrequencyResult,
    SystemSolveRequest,
    compiled_system_from_dict,
    compiled_system_to_dict,
    system_frequency_result_from_dict,
    system_frequency_result_to_dict,
    system_solve_request_from_dict,
    system_solve_request_to_dict,
)

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"
FEM_FIXTURE = FIXTURE_ROOT / "femvolume.msh"
BEM_FIXTURE = FIXTURE_ROOT / "exterior_conforming.msh"


def test_compiler_resolves_fixture_physics_and_interface_topology() -> None:
    compiled = PhysicalSystemCompiler().compile(_fixture_system())

    assert compiled.contract_version == 1
    regions = {region.id: region for region in compiled.regions}
    assert regions["region:interior"].volume_groups[0].tag == 1

    boundaries = {boundary.id: boundary for boundary in compiled.boundaries}
    assert boundaries["boundary:radiator"].group.tag == 2

    topology = compiled.interfaces[0].topology
    assert len(topology.fem_vertex_indices) == 106
    assert len(topology.fem_to_bem_vertex_indices) == 106
    assert len(topology.fem_face_indices) == 180
    assert len(topology.bem_face_indices) == 180
    assert set(topology.normal_sign) <= {-1, 1}
    assert topology.max_coordinate_error <= 1e-12
    assert topology.fem_facets_on_tetra_boundary == 180
    assert topology.bem_boundary_edges == 0

    assumptions = {(item.status, item.statement) for item in compiled.assumptions}
    assert (
        AssumptionStatus.INCLUDED,
        "Conforming bidirectional FEM-BEM interfaces",
    ) in assumptions
    assert (AssumptionStatus.EXCLUDED, "Acoustic loss models") in assumptions


def test_compiler_rejects_unassigned_physical_surface_group() -> None:
    system = _fixture_system()
    incomplete = PhysicalSystem(
        id=system.id,
        name=system.name,
        meshes=system.meshes,
        regions=system.regions,
        boundaries=tuple(boundary for boundary in system.boundaries if boundary.id != "boundary:wall"),
        interfaces=system.interfaces,
        components=system.components,
        excitation_ports=system.excitation_ports,
    )

    with pytest.raises(PhysicalModelCompileError, match="unassigned physical surface groups"):
        PhysicalSystemCompiler().compile(incomplete)


def test_compiled_system_and_request_round_trip_through_versioned_contract() -> None:
    compiled = PhysicalSystemCompiler().compile(_fixture_system())
    compiled_wire = compiled_system_to_dict(compiled)
    restored = compiled_system_from_dict(compiled_wire)
    assert restored == compiled
    assert "signals" not in compiled_wire
    assert compiled_wire["excitation_ports"][0]["kind"] == "normal_velocity"

    request = SystemSolveRequest(
        compiled_system=compiled,
        frequencies_hz=(200.0, 1000.0),
        excitation_port_ids=("excitation:radiator",),
        outputs=(
            OutputRequest(
                id="output:exterior-pressure",
                quantity="acoustic_pressure",
                target_ids=("region:exterior",),
                options={"sampling": "polar"},
            ),
            OutputRequest(
                id="output:diaphragm-velocity",
                quantity="normal_velocity",
                target_ids=("component:radiator",),
            ),
        ),
        solver_options={"precision": "float64", "coupling": "direct_reference"},
    )
    restored_request = system_solve_request_from_dict(system_solve_request_to_dict(request))
    assert restored_request == request


def test_coupled_reference_backend_exposes_system_metadata_without_starting_julia() -> None:
    compiled = PhysicalSystemCompiler().compile(_fixture_system())
    request = SystemSolveRequest(
        compiled_system=compiled,
        frequencies_hz=(500.0,),
        excitation_port_ids=("excitation:radiator",),
        outputs=(OutputRequest(id="output:fem", quantity="fem_nodal_pressure"),),
    )

    session = CoupledReferenceBackend().create_system_session(request)

    assert session.metadata.system_id == compiled.id
    assert session.metadata.excitation_port_ids == request.excitation_port_ids
    assert session.metadata.available_quantity_ids == ("output:fem",)
    assert session.request.solver_options["precision"] == "float64"
    assert session.request.solver_options["bem_backend"] == "cpu"


def test_coupled_production_backend_forces_fp32_and_selects_cuda_project() -> None:
    compiled = PhysicalSystemCompiler().compile(_fixture_system())
    request = SystemSolveRequest(
        compiled_system=compiled,
        frequencies_hz=(500.0,),
        excitation_port_ids=("excitation:radiator",),
        outputs=(OutputRequest(id="output:fem", quantity="fem_nodal_pressure"),),
        solver_options={"precision": "float64", "bem_backend": "cpu"},
    )

    backend = CoupledProductionBackend(bem_backend="cuda")
    session = backend.create_system_session(request)

    assert session.request.solver_options["precision"] == "float32"
    assert session.request.solver_options["bem_backend"] == "cuda"
    assert session.julia_project == DEFAULT_BEAT_ENGINE_CUDA_PROJECT.resolve()
    assert session.julia_threads == 4
    assert CoupledProductionBackend(bem_backend="cpu").create_system_session(request).julia_threads == 8


def test_coupled_backend_rejects_unsupported_physical_roles_before_starting_julia() -> None:
    compiled = PhysicalSystemCompiler().compile(_fixture_system())
    wall = next(boundary for boundary in compiled.boundaries if boundary.id == "boundary:wall")
    unsupported = replace(
        compiled,
        boundaries=tuple(
            replace(boundary, kind=BoundaryKind.IMPEDANCE)
            if boundary.id == wall.id
            else boundary
            for boundary in compiled.boundaries
        ),
    )
    request = SystemSolveRequest(
        compiled_system=unsupported,
        frequencies_hz=(500.0,),
        excitation_port_ids=("excitation:radiator",),
    )

    with pytest.raises(ValueError, match="does not support the boundary assignments"):
        CoupledProductionBackend(bem_backend="cpu").create_system_session(request)


def test_coupled_backend_accepts_mmd_electrodynamic_component_and_voltage_port() -> None:
    compiled = PhysicalSystemCompiler().compile(_electrodynamic_fixture_system())
    request = SystemSolveRequest(
        compiled_system=compiled,
        frequencies_hz=(500.0,),
        excitation_port_ids=("excitation:radiator",),
        outputs=(
            OutputRequest(id="output:velocity", quantity="diaphragm_velocity"),
            OutputRequest(id="output:current", quantity="voice_coil_current"),
        ),
        solver_options={"static_condensation": True},
    )

    session = CoupledProductionBackend(bem_backend="cuda").create_system_session(request)

    assert session.request.solver_options["static_condensation"] is False
    assert compiled.excitation_ports[0].kind == ExcitationPortKind.VOLTAGE
    assert DEFAULT_TRANSDUCER_REFERENCE_VOLTAGE_V == pytest.approx(2.83)
    assumptions = {item.statement for item in compiled.assumptions}
    assert "Linear rigid-piston electrodynamic transducers with dry moving mass" in assumptions


@pytest.mark.parametrize(
    ("parameters", "message"),
    (
        (
            {
                "re_ohm": 6.0,
                "le_h": 0.0005,
                "bl_n_per_a": 7.0,
                "cms_m_per_n": 0.0005,
                "rms_n_s_per_m": 1.0,
            },
            "mmd_kg",
        ),
        (
            {
                "re_ohm": 6.0,
                "le_h": 0.0005,
                "bl_n_per_a": 7.0,
                "mmd_kg": 0.015,
                "mms_kg": 0.016,
                "cms_m_per_n": 0.0005,
                "rms_n_s_per_m": 1.0,
            },
            "mms_kg",
        ),
    ),
)
def test_coupled_backend_rejects_incomplete_or_mms_transducer_parameters(
    parameters: dict[str, float],
    message: str,
) -> None:
    system = _electrodynamic_fixture_system()
    component = replace(system.components[0], parameters=parameters)
    compiled = PhysicalSystemCompiler().compile(
        replace(system, components=(component,))
    )
    request = SystemSolveRequest(
        compiled_system=compiled,
        frequencies_hz=(500.0,),
        excitation_port_ids=("excitation:radiator",),
    )

    with pytest.raises(ValueError, match=message):
        CoupledProductionBackend(bem_backend="cpu").create_system_session(request)


def test_coupled_backend_rejects_symmetry_for_electrodynamic_component() -> None:
    compiled = PhysicalSystemCompiler().compile(_electrodynamic_fixture_system())
    request = SystemSolveRequest(
        compiled_system=compiled,
        frequencies_hz=(500.0,),
        excitation_port_ids=("excitation:radiator",),
        solver_options={"symmetry": "x"},
    )

    with pytest.raises(ValueError, match="require symmetry to be off"):
        CoupledProductionBackend(bem_backend="cpu").create_system_session(request)


@pytest.mark.skipif(
    os.environ.get("BLAB_RUN_COUPLED_REFERENCE") != "1",
    reason="Set BLAB_RUN_COUPLED_REFERENCE=1 to run the Julia dense reference integration.",
)
def test_coupled_reference_backend_solves_fixture_and_returns_basis_quantities() -> None:
    compiled = PhysicalSystemCompiler().compile(_fixture_system())
    request = SystemSolveRequest(
        compiled_system=compiled,
        frequencies_hz=(500.0,),
        excitation_port_ids=("excitation:radiator",),
        outputs=(
            OutputRequest(id="output:fem", quantity="fem_nodal_pressure"),
            OutputRequest(id="output:bem", quantity="bem_boundary_pressure"),
            OutputRequest(id="output:interface", quantity="interface_normal_derivative"),
            OutputRequest(
                id="output:field",
                quantity="exterior_pressure",
                options={"points_m": [[0.0, 0.0, 0.2]]},
            ),
        ),
        solver_options={"quadrature_order": 1, "singular_order": 1},
    )
    julia_executable = os.environ.get("BLAB_JULIA_EXE", "julia")
    results = list(
        CoupledReferenceBackend(julia_executable=julia_executable).create_system_session(request).solve_stream()
    )

    assert len(results) == 1
    result = results[0]
    assert result.excitation_port_ids == request.excitation_port_ids
    quantities = {quantity.id: quantity for quantity in result.quantities}
    assert quantities["output:fem"].values.shape == (1, 842)
    assert quantities["output:bem"].values.shape == (1, 1214)
    assert quantities["output:interface"].values.shape == (1, 106)
    assert quantities["output:field"].values.shape == (1, 1)
    assert quantities["output:fem"].values.dtype == np.complex128
    assert result.diagnostics["relative_residual"] < 1e-8
    assert result.diagnostics["pressure_continuity_error"] < 1e-8
    assert result.diagnostics["flux_conservation_error"] < 1e-10
    assert result.diagnostics["all_bem_replay_error"] < 1e-8


@pytest.mark.skipif(
    os.environ.get("BLAB_RUN_COUPLED_REFERENCE") != "1",
    reason="Set BLAB_RUN_COUPLED_REFERENCE=1 to run the Julia electrodynamic integration.",
)
def test_coupled_reference_backend_solves_bidirectional_electrodynamic_fixture() -> None:
    compiled = PhysicalSystemCompiler().compile(
        _bidirectional_electrodynamic_fixture_system()
    )
    request = SystemSolveRequest(
        compiled_system=compiled,
        frequencies_hz=(500.0,),
        excitation_port_ids=("excitation:radiator",),
        outputs=(
            OutputRequest(id="output:fem", quantity="fem_nodal_pressure"),
            OutputRequest(id="output:bem", quantity="bem_boundary_pressure"),
            OutputRequest(id="output:velocity", quantity="diaphragm_velocity"),
            OutputRequest(id="output:current", quantity="voice_coil_current"),
            OutputRequest(
                id="output:field",
                quantity="exterior_pressure",
                options={"points_m": [[0.0, 0.0, 0.2]]},
            ),
        ),
        solver_options={"quadrature_order": 1, "singular_order": 1},
    )
    julia_executable = os.environ.get("BLAB_JULIA_EXE", "julia")

    (result,) = tuple(
        CoupledReferenceBackend(
            julia_executable=julia_executable,
            persistent_worker=False,
        )
        .create_system_session(request)
        .solve_stream()
    )

    quantities = {quantity.id: quantity for quantity in result.quantities}
    velocity = quantities["output:velocity"].values[0, 0]
    current = quantities["output:current"].values[0, 0]
    electrical_impedance = 6.0 - 1j * 2.0 * np.pi * 500.0 * 0.0005
    electrical_residual = abs(
        electrical_impedance * current + 7.0 * velocity - DEFAULT_TRANSDUCER_REFERENCE_VOLTAGE_V
    ) / DEFAULT_TRANSDUCER_REFERENCE_VOLTAGE_V

    assert quantities["output:velocity"].unit == "m/s"
    assert quantities["output:current"].unit == "A"
    assert quantities["output:velocity"].metadata["component_ids"] == ["component:radiator"]
    assert np.isfinite(velocity)
    assert np.isfinite(current)
    assert abs(velocity) > 0.0
    assert abs(current) > 0.0
    assert electrical_residual < 1e-8
    assert result.diagnostics["transducer_count"] == 1
    assert result.diagnostics["transducer_reference_voltage_v"] == pytest.approx(2.83)
    assert result.diagnostics["formulation"] == "monolithic"
    assert result.diagnostics["relative_residual"] < 1e-8
    assert result.diagnostics["all_bem_replay_error"] < 1e-8


@pytest.mark.skipif(
    os.environ.get("BLAB_RUN_COUPLED_CUDA") != "1",
    reason="Set BLAB_RUN_COUPLED_CUDA=1 to run electrodynamic CPU/CUDA parity.",
)
def test_coupled_electrodynamic_cuda_matches_cpu() -> None:
    compiled = PhysicalSystemCompiler().compile(
        _bidirectional_electrodynamic_fixture_system()
    )
    request = SystemSolveRequest(
        compiled_system=compiled,
        frequencies_hz=(500.0,),
        excitation_port_ids=("excitation:radiator",),
        outputs=(
            OutputRequest(id="output:velocity", quantity="diaphragm_velocity"),
            OutputRequest(id="output:current", quantity="voice_coil_current"),
            OutputRequest(
                id="output:field",
                quantity="exterior_pressure",
                options={"points_m": [[0.0, 0.0, 0.2]]},
            ),
        ),
        solver_options={
            "quadrature_order": 1,
            "singular_order": 1,
            "validation_diagnostics": False,
            "static_condensation": True,
        },
    )
    julia_executable = os.environ.get("BLAB_JULIA_EXE", "julia")
    cpu_backend = CoupledProductionBackend(
        bem_backend="cpu",
        julia_executable=julia_executable,
        persistent_worker=False,
    )
    cuda_backend = CoupledProductionBackend(
        bem_backend="cuda",
        julia_executable=julia_executable,
        persistent_worker=True,
    )

    (cpu_result,) = tuple(cpu_backend.create_system_session(request).solve_stream())
    (cuda_result,) = tuple(cuda_backend.create_system_session(request).solve_stream())

    cpu_quantities = {quantity.id: quantity.values for quantity in cpu_result.quantities}
    cuda_quantities = {quantity.id: quantity.values for quantity in cuda_result.quantities}
    for quantity_id in cpu_quantities:
        scale = max(float(np.linalg.norm(cpu_quantities[quantity_id])), np.finfo(float).eps)
        relative_error = float(
            np.linalg.norm(cuda_quantities[quantity_id] - cpu_quantities[quantity_id])
        ) / scale
        assert relative_error < 5e-3
    assert cpu_result.diagnostics["linear_backend"] == "cpu"
    assert cuda_result.diagnostics["linear_backend"] == "cuda"
    assert cpu_result.diagnostics["formulation"] == "monolithic"
    assert cuda_result.diagnostics["formulation"] == "monolithic"
    assert cuda_result.diagnostics["static_condensation_requested"] is False


def test_editable_physical_system_round_trips_for_project_persistence() -> None:
    system = _fixture_system()

    restored = physical_system_from_dict(physical_system_to_dict(system))

    assert restored == system


def test_generalized_frequency_result_preserves_complex_double_precision() -> None:
    result = SystemFrequencyResult(
        freq_hz=1000.0,
        quantities=(
            QuantityResult(
                id="pressure:interface",
                quantity="acoustic_pressure",
                unit="Pa",
                target_id="interface:port",
                axes=("excitation", "interface_vertex"),
                values=np.asarray([[1.0 + 2.0j, 3.0 - 4.0j]], dtype=np.complex128),
            ),
        ),
        excitation_port_ids=("excitation:radiator",),
        diagnostics={"coupled_residual": 1e-10},
    )

    restored = system_frequency_result_from_dict(system_frequency_result_to_dict(result))

    assert restored.freq_hz == result.freq_hz
    assert restored.quantities[0].values.dtype == np.complex128
    assert np.array_equal(restored.quantities[0].values, result.quantities[0].values)
    assert restored.diagnostics == result.diagnostics


def test_system_contract_rejects_unknown_ports_and_misaligned_excitation_axes() -> None:
    compiled = PhysicalSystemCompiler().compile(_fixture_system())
    bad_request = SystemSolveRequest(
        compiled_system=compiled,
        frequencies_hz=(1000.0,),
        excitation_port_ids=("excitation:unknown",),
    )
    with pytest.raises(ValueError, match="unknown excitation port"):
        system_solve_request_to_dict(bad_request)

    bad_result = SystemFrequencyResult(
        freq_hz=1000.0,
        quantities=(
            QuantityResult(
                id="pressure:field",
                quantity="acoustic_pressure",
                unit="Pa",
                axes=("excitation", "observation"),
                values=np.ones((2, 3), dtype=np.complex128),
            ),
        ),
        excitation_port_ids=("excitation:radiator",),
    )
    with pytest.raises(ValueError, match="excitation axis has length 2"):
        system_frequency_result_to_dict(bad_result)


def test_compiler_requires_one_physical_input_port_for_an_active_component() -> None:
    system = _fixture_system()
    without_port = PhysicalSystem(
        id=system.id,
        name=system.name,
        meshes=system.meshes,
        regions=system.regions,
        boundaries=system.boundaries,
        interfaces=system.interfaces,
        components=system.components,
    )

    with pytest.raises(PhysicalModelCompileError, match="requires 1 excitation port"):
        PhysicalSystemCompiler().compile(without_port)


def _electrodynamic_fixture_system() -> PhysicalSystem:
    system = _fixture_system()
    component = replace(
        system.components[0],
        name="Linear electrodynamic radiator",
        kind=ComponentKind.ELECTRODYNAMIC_TRANSDUCER,
        parameters={
            "re_ohm": 6.0,
            "le_h": 0.0005,
            "bl_n_per_a": 7.0,
            "mmd_kg": 0.015,
            "cms_m_per_n": 0.0005,
            "rms_n_s_per_m": 1.0,
            "motion_profile": "uniform",
        },
    )
    port = replace(
        system.excitation_ports[0],
        name="Radiator 2.83 V reference",
        kind=ExcitationPortKind.VOLTAGE,
    )
    return replace(system, components=(component,), excitation_ports=(port,))


def _bidirectional_electrodynamic_fixture_system() -> PhysicalSystem:
    system = _electrodynamic_fixture_system()
    exterior_boundary = next(
        boundary
        for boundary in system.boundaries
        if boundary.id == "boundary:exterior"
    )
    moving_exterior = replace(exterior_boundary, kind=BoundaryKind.MOVING)
    component = replace(
        system.components[0],
        boundary_ids=("boundary:radiator", "boundary:exterior"),
    )
    return replace(
        system,
        boundaries=tuple(
            moving_exterior if boundary.id == moving_exterior.id else boundary
            for boundary in system.boundaries
        ),
        components=(component,),
    )


def _fixture_system() -> PhysicalSystem:
    return PhysicalSystem(
        id="system:coupled-fixture",
        name="Coupled box and port fixture",
        meshes=(
            MeshResource(
                id="mesh:fem",
                name="Interior volume",
                file=str(FEM_FIXTURE),
                purpose=MeshPurpose.FEM_VOLUME,
                scale_to_m=0.001,
            ),
            MeshResource(
                id="mesh:bem",
                name="Exterior boundary",
                file=str(BEM_FIXTURE),
                purpose=MeshPurpose.BEM_SURFACE,
                scale_to_m=0.001,
            ),
        ),
        regions=(
            AcousticRegion(
                id="region:interior",
                name="Interior air",
                kind=AcousticRegionKind.BOUNDED_AIR,
                mesh_ids=("mesh:fem",),
                volume_groups=(PhysicalGroupRef(mesh_id="mesh:fem", dimension=3, name="Volume"),),
            ),
            AcousticRegion(
                id="region:exterior",
                name="Exterior air",
                kind=AcousticRegionKind.UNBOUNDED_AIR,
                mesh_ids=("mesh:bem",),
            ),
        ),
        boundaries=(
            Boundary(
                id="boundary:radiator",
                name="Radiator",
                region_id="region:interior",
                group=PhysicalGroupRef(mesh_id="mesh:fem", dimension=2, name="Radiator"),
                kind=BoundaryKind.MOVING,
            ),
            Boundary(
                id="boundary:wall",
                name="Interior hard walls",
                region_id="region:interior",
                group=PhysicalGroupRef(mesh_id="mesh:fem", dimension=2, name="Volume_boundary"),
                kind=BoundaryKind.RIGID,
            ),
            Boundary(
                id="boundary:fem-interface",
                name="Interior port side",
                region_id="region:interior",
                group=PhysicalGroupRef(mesh_id="mesh:fem", dimension=2, name="Interface"),
                kind=BoundaryKind.INTERFACE,
            ),
            Boundary(
                id="boundary:exterior",
                name="Exterior box",
                region_id="region:exterior",
                group=PhysicalGroupRef(mesh_id="mesh:bem", dimension=2, name="ExteriorBox"),
                kind=BoundaryKind.RIGID,
            ),
            Boundary(
                id="boundary:bem-interface",
                name="Exterior port side",
                region_id="region:exterior",
                group=PhysicalGroupRef(mesh_id="mesh:bem", dimension=2, name="Interface"),
                kind=BoundaryKind.INTERFACE,
            ),
        ),
        interfaces=(
            AcousticInterface(
                id="interface:port",
                name="Port",
                bounded_boundary_id="boundary:fem-interface",
                unbounded_boundary_id="boundary:bem-interface",
            ),
        ),
        components=(
            PhysicalComponent(
                id="component:radiator",
                name="Ideal radiator",
                kind=ComponentKind.IDEAL_VELOCITY_SOURCE,
                boundary_ids=("boundary:radiator",),
                parameters={
                    "motion_profile": "uniform",
                },
            ),
        ),
        excitation_ports=(
            ExcitationPort(
                id="excitation:radiator",
                name="Radiator unit normal velocity",
                component_id="component:radiator",
                kind=ExcitationPortKind.NORMAL_VELOCITY,
            ),
        ),
    )
