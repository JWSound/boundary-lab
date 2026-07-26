import os
from pathlib import Path

import numpy as np
import pytest

from blab.config import ChannelConfig, RadiatorConfig, SimulationConfig
from blab.legacy_physical_adapter import physical_system_from_legacy_config
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
from blab.solvers.coupled_reference_backend import CoupledReferenceBackend
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
    meshes = {mesh.id: mesh for mesh in compiled.meshes}
    assert meshes["mesh:fem"].point_count == 842
    assert meshes["mesh:fem"].tetrahedron_count == 2925
    assert meshes["mesh:bem"].point_count == 1214
    assert meshes["mesh:bem"].triangle_count == 2424
    regions = {region.id: region for region in compiled.regions}
    assert regions["region:interior"].volume_groups[0].tag == 1
    assert regions["region:interior"].volume_groups[0].element_count == 2925

    boundaries = {boundary.id: boundary for boundary in compiled.boundaries}
    assert boundaries["boundary:radiator"].group.tag == 2
    assert boundaries["boundary:fem-interface"].group.element_count == 180
    assert boundaries["boundary:bem-interface"].group.element_count == 180

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
    assert result.diagnostics["relative_residual"] < 1e-8
    assert result.diagnostics["pressure_continuity_error"] < 1e-8
    assert result.diagnostics["flux_conservation_error"] < 1e-10
    assert result.diagnostics["all_bem_replay_error"] < 1e-8


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


def test_legacy_bem_config_adapts_and_compiles_without_changing_legacy_protocol() -> None:
    legacy = SimulationConfig(
        mesh_file=str(BEM_FIXTURE),
        scale_factor=0.001,
        radiators=(
            RadiatorConfig(
                name="Legacy radiator",
                tag=2,
                mesh="mesh",
                channel="main",
            ),
        ),
        channels=(ChannelConfig(name="main", level_db=-3.0),),
    )

    physical = physical_system_from_legacy_config(legacy)
    compiled = PhysicalSystemCompiler().compile(physical)

    assert len(compiled.regions) == 1
    assert compiled.regions[0].kind == AcousticRegionKind.UNBOUNDED_AIR
    assert len(compiled.interfaces) == 0
    assert compiled.components[0].kind == ComponentKind.IDEAL_VELOCITY_SOURCE
    assert compiled.excitation_ports[0].kind == ExcitationPortKind.NORMAL_VELOCITY
    assert not hasattr(compiled, "signals")
    assert compiled.metadata["source"] == "legacy_simulation_config"


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
