import os
import sys
from dataclasses import replace
from pathlib import Path

import meshio
import numpy as np
import pytest

from blab.acoustic_materials import miki_wall_impedance_parameters
from blab.interface_conform import conform_bem_interface_to_fem
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
from blab.solvers.beat_engine_backend import (
    DEFAULT_BEAT_ENGINE_CUDA_PROJECT,
    shutdown_beat_engine_workers,
)
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
from repo_paths import REPO_ROOT

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"
FEM_FIXTURE = FIXTURE_ROOT / "femvolume.msh"
BEM_FIXTURE = FIXTURE_ROOT / "exterior_conforming.msh"
SKRAM_FIXTURE_ROOT = FIXTURE_ROOT / "SKRAM"
SIMPLE_SEALED_FIXTURE_ROOT = REPO_ROOT / "examples" / "Simple_Sealed"


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
    assert (
        AssumptionStatus.EXCLUDED,
        "Region-specific acoustic material loss models",
    ) in assumptions


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


def test_legacy_unused_boundary_deserializes_as_rigid() -> None:
    payload = physical_system_to_dict(_fixture_system())
    payload["boundaries"][0]["kind"] = "unused"

    restored = physical_system_from_dict(payload)

    assert restored.boundaries[0].kind == BoundaryKind.RIGID


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


def test_coupled_backend_accepts_disconnected_exterior_mesh_resources() -> None:
    system = _fixture_system()
    exterior_mesh = next(mesh for mesh in system.meshes if mesh.id == "mesh:bem")
    second_exterior_mesh = replace(
        exterior_mesh,
        id="mesh:bem-phase-plug",
        name="Disconnected phase plug",
        translation_m=(0.5, 0.0, 0.0),
    )
    second_mesh_boundaries = tuple(
        replace(
            boundary,
            id=f"{boundary.id}-phase-plug",
            name=f"{boundary.name} phase plug",
            group=replace(boundary.group, mesh_id=second_exterior_mesh.id),
            kind=BoundaryKind.RIGID,
        )
        for boundary in system.boundaries
        if boundary.region_id == "region:exterior"
    )
    configured = replace(
        system,
        meshes=(*system.meshes, second_exterior_mesh),
        regions=tuple(
            replace(region, mesh_ids=("mesh:bem", second_exterior_mesh.id))
            if region.kind == AcousticRegionKind.UNBOUNDED_AIR
            else region
            for region in system.regions
        ),
        boundaries=(*system.boundaries, *second_mesh_boundaries),
    )
    compiled = PhysicalSystemCompiler().compile(configured)
    request = SystemSolveRequest(
        compiled_system=compiled,
        frequencies_hz=(500.0,),
        excitation_port_ids=("excitation:radiator",),
    )

    session = CoupledProductionBackend(bem_backend="cpu").create_system_session(request)

    exterior = next(
        region for region in session.request.compiled_system.regions if region.kind == AcousticRegionKind.UNBOUNDED_AIR
    )
    assert exterior.mesh_ids == ("mesh:bem", "mesh:bem-phase-plug")


def test_coupled_cancel_keeps_persistent_worker_warm(tmp_path: Path) -> None:
    starts_path = tmp_path / "coupled_cancel_starts.txt"
    fake_solver = tmp_path / "fake_coupled_cancel_worker.py"
    fake_solver.write_text(
        f"""
import json
import pathlib
import sys
import time

starts_path = pathlib.Path({str(starts_path)!r})
starts = int(starts_path.read_text(encoding="utf-8")) if starts_path.exists() else 0
starts_path.write_text(str(starts + 1), encoding="utf-8")

if "--worker" not in sys.argv:
    raise SystemExit("expected --worker")

print(json.dumps({{"type": "ready"}}), flush=True)
for line in sys.stdin:
    submission = json.loads(line)
    with open(submission["request"], "r", encoding="utf-8") as handle:
        request = json.load(handle)
    frequency = request["frequencies_hz"][0]
    if frequency == 500.0:
        print(json.dumps({{"type": "status", "message": "started"}}), flush=True)
        deadline = time.monotonic() + 2.0
        cancel_path = pathlib.Path(request["cancel_path"])
        while not cancel_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not cancel_path.exists():
            print(json.dumps({{"type": "failed", "error": "cancel file was not created"}}), flush=True)
            continue
        print(json.dumps({{"type": "cancelled", "solved_count": 0}}), flush=True)
        continue
    print(json.dumps({{
        "type": "result",
        "result": {{
            "schema_version": 1,
            "freq_hz": frequency,
            "excitation_port_ids": request["excitation_port_ids"],
            "quantities": [],
            "diagnostics": {{}},
        }},
    }}), flush=True)
    print(json.dumps({{"type": "completed", "solved_count": 1}}), flush=True)
""".strip(),
        encoding="utf-8",
    )
    compiled = PhysicalSystemCompiler().compile(_fixture_system())

    def request(frequency: float) -> SystemSolveRequest:
        return SystemSolveRequest(
            compiled_system=compiled,
            frequencies_hz=(frequency,),
            excitation_port_ids=("excitation:radiator",),
        )

    try:
        backend = CoupledReferenceBackend(
            julia_executable=sys.executable,
            solver_script=fake_solver,
            julia_project=None,
            persistent_worker=True,
        )
        cancelled_session = backend.create_system_session(request(500.0))
        assert list(cancelled_session.solve_stream(stop_requested=lambda: True)) == []

        completed_session = backend.create_system_session(request(1000.0))
        assert len(list(completed_session.solve_stream())) == 1
        assert starts_path.read_text(encoding="utf-8") == "1"
    finally:
        shutdown_beat_engine_workers()


@pytest.mark.parametrize("loss_factor", (-0.001, 1.001, "invalid"))
def test_compiler_rejects_invalid_region_fem_bulk_loss_factor(loss_factor) -> None:
    system = _fixture_system()
    invalid = replace(
        system,
        regions=tuple(
            replace(region, loss_model={"bulk_loss_factor": loss_factor})
            if region.kind == AcousticRegionKind.BOUNDED_AIR
            else region
            for region in system.regions
        ),
    )

    with pytest.raises(PhysicalModelCompileError, match="FEM bulk loss factor"):
        PhysicalSystemCompiler().compile(invalid)


def test_coupled_backend_accepts_region_fem_bulk_loss_and_wall_impedance() -> None:
    system = _fixture_system()
    configured = replace(
        system,
        regions=tuple(
            replace(region, loss_model={"bulk_loss_factor": 0.01})
            if region.kind == AcousticRegionKind.BOUNDED_AIR
            else region
            for region in system.regions
        ),
        boundaries=tuple(
            replace(boundary, parameters=miki_wall_impedance_parameters())
            if boundary.id == "boundary:wall"
            else boundary
            for boundary in system.boundaries
        ),
    )
    compiled = PhysicalSystemCompiler().compile(configured)
    request = SystemSolveRequest(
        compiled_system=compiled,
        frequencies_hz=(500.0,),
        excitation_port_ids=("excitation:radiator",),
    )

    session = CoupledProductionBackend(bem_backend="cpu").create_system_session(request)

    interior = next(
        region for region in session.request.compiled_system.regions if region.kind == AcousticRegionKind.BOUNDED_AIR
    )
    wall = next(boundary for boundary in session.request.compiled_system.boundaries if boundary.id == "boundary:wall")
    assert interior.loss_model["bulk_loss_factor"] == pytest.approx(0.01)
    assert wall.parameters["wall_impedance"]["thickness_m"] == pytest.approx(0.03)
    assert wall.parameters["wall_impedance"]["flow_resistivity_pa_s_per_m2"] == pytest.approx(5000.0)
    assumptions = {item.statement for item in compiled.assumptions}
    assert "Homogeneous per-region FEM bulk loss" in assumptions
    assert "Locally reacting rigid-backed Miki porous wall treatments" in assumptions


def test_coupled_backend_rejects_unsupported_physical_roles_before_starting_julia() -> None:
    compiled = PhysicalSystemCompiler().compile(_fixture_system())
    wall = next(boundary for boundary in compiled.boundaries if boundary.id == "boundary:wall")
    unsupported = replace(
        compiled,
        boundaries=tuple(
            replace(boundary, kind=BoundaryKind.IMPEDANCE) if boundary.id == wall.id else boundary
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

    assert session.request.solver_options["static_condensation"] is True
    assert compiled.excitation_ports[0].kind == ExcitationPortKind.VOLTAGE
    assert DEFAULT_TRANSDUCER_REFERENCE_VOLTAGE_V == pytest.approx(2.83)
    assumptions = {item.statement for item in compiled.assumptions}
    assert "Linear single-axis rigid-body electrodynamic transducers with dry moving mass" in assumptions


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
                "motion_axis": [0.0, 0.0, 1.0],
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
                "motion_axis": [0.0, 0.0, 1.0],
            },
            "mms_kg",
        ),
    ),
)
def test_coupled_backend_rejects_incomplete_or_mms_transducer_parameters(
    parameters: dict[str, object],
    message: str,
) -> None:
    system = _electrodynamic_fixture_system()
    component = replace(system.components[0], parameters=parameters)
    compiled = PhysicalSystemCompiler().compile(replace(system, components=(component,)))
    request = SystemSolveRequest(
        compiled_system=compiled,
        frequencies_hz=(500.0,),
        excitation_port_ids=("excitation:radiator",),
    )

    with pytest.raises(ValueError, match=message):
        CoupledProductionBackend(bem_backend="cpu").create_system_session(request)


def test_compiler_infers_fractional_driver_symmetry_and_overrides_legacy_values(
    tmp_path: Path,
) -> None:
    system = _skram_fixture_system(tmp_path)
    component = replace(
        system.components[0],
        parameters={
            **system.components[0].parameters,
            "symmetry_role": "complete_representative",
            "surface_completion_factor": 1,
            "physical_driver_orbit_count": 2,
            "fractional_symmetry_axes": [],
        },
    )
    compiled = PhysicalSystemCompiler().compile(
        replace(system, components=(component,)),
        symmetry_mode="x",
    )
    request = SystemSolveRequest(
        compiled_system=compiled,
        frequencies_hz=(500.0,),
        excitation_port_ids=("excitation:skram-driver",),
        solver_options={"symmetry": "x"},
    )

    session = CoupledProductionBackend(bem_backend="cpu").create_system_session(request)

    assert compiled.components[0].parameters["fractional_symmetry_axes"] == ["x"]
    assert compiled.components[0].parameters["surface_completion_factor"] == 2
    assert compiled.components[0].parameters["physical_driver_orbit_count"] == 1
    assert session.request.solver_options["symmetry"] == "x"


def test_coupled_backend_rejects_ambiguous_transducer_symmetry_scaling() -> None:
    compiled = PhysicalSystemCompiler().compile(_electrodynamic_fixture_system())
    request = SystemSolveRequest(
        compiled_system=compiled,
        frequencies_hz=(500.0,),
        excitation_port_ids=("excitation:radiator",),
        solver_options={"symmetry": "x"},
    )

    with pytest.raises(ValueError, match="requires 2"):
        CoupledProductionBackend(bem_backend="cpu").create_system_session(request)


def test_coupled_backend_rejects_fractional_driver_motion_normal_to_cut_plane(
    tmp_path: Path,
) -> None:
    system = _skram_fixture_system(tmp_path)
    component = replace(
        system.components[0],
        parameters={
            **system.components[0].parameters,
            "motion_axis": [1.0, 0.0, 0.0],
        },
    )
    compiled = PhysicalSystemCompiler().compile(
        replace(system, components=(component,)),
        symmetry_mode="x",
    )
    request = SystemSolveRequest(
        compiled_system=compiled,
        frequencies_hz=(500.0,),
        excitation_port_ids=("excitation:skram-driver",),
        solver_options={"symmetry": "x"},
    )

    with pytest.raises(ValueError, match="motion_axis must lie"):
        CoupledProductionBackend(bem_backend="cpu").create_system_session(request)


def test_skram_multi_chamber_model_compiles_with_shared_fractional_driver(
    tmp_path: Path,
) -> None:
    system = _skram_fixture_system(tmp_path)

    compiled = PhysicalSystemCompiler().compile(system, symmetry_mode="x")
    request = SystemSolveRequest(
        compiled_system=compiled,
        frequencies_hz=(200.0,),
        excitation_port_ids=("excitation:skram-driver",),
        solver_options={"symmetry": "x"},
    )
    session = CoupledProductionBackend(bem_backend="cpu").create_system_session(request)

    assert len([region for region in compiled.regions if region.kind == AcousticRegionKind.BOUNDED_AIR]) == 2
    assert len(compiled.interfaces) == 2
    assert compiled.components[0].boundary_ids == (
        "boundary:front-radiator",
        "boundary:rear-radiator",
    )
    assert session.request.solver_options["symmetry"] == "x"


def test_simple_sealed_model_compiles_without_an_acoustic_interface() -> None:
    compiled = PhysicalSystemCompiler().compile(_simple_sealed_fixture_system())
    request = SystemSolveRequest(
        compiled_system=compiled,
        frequencies_hz=(200.0,),
        excitation_port_ids=("excitation:simple-sealed-driver",),
        solver_options={"static_condensation": True},
    )

    session = CoupledProductionBackend(bem_backend="cuda").create_system_session(request)

    assert compiled.interfaces == ()
    assert compiled.components[0].boundary_ids == (
        "boundary:simple-sealed-rear-diaphragm",
        "boundary:simple-sealed-front-diaphragm",
    )
    assert session.request.solver_options["static_condensation"] is True


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
            OutputRequest(id="output:bem-neumann", quantity="bem_boundary_neumann"),
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
    assert quantities["output:bem-neumann"].values.shape == (1, 2424)
    assert quantities["output:bem-neumann"].unit == "Pa/m"
    assert quantities["output:bem-neumann"].axes == ("excitation", "bem_face")
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
    compiled = PhysicalSystemCompiler().compile(_bidirectional_electrodynamic_fixture_system())
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
    electrical_residual = (
        abs(electrical_impedance * current + 7.0 * velocity - DEFAULT_TRANSDUCER_REFERENCE_VOLTAGE_V)
        / DEFAULT_TRANSDUCER_REFERENCE_VOLTAGE_V
    )

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
    os.environ.get("BLAB_RUN_COUPLED_REFERENCE") != "1",
    reason="Set BLAB_RUN_COUPLED_REFERENCE=1 to run the sealed zero-interface integration.",
)
def test_coupled_reference_backend_solves_sealed_zero_interface_fixture() -> None:
    compiled = PhysicalSystemCompiler().compile(_sealed_electrodynamic_fixture_system())
    request = SystemSolveRequest(
        compiled_system=compiled,
        frequencies_hz=(500.0,),
        excitation_port_ids=("excitation:radiator",),
        outputs=(
            OutputRequest(id="output:velocity", quantity="diaphragm_velocity"),
            OutputRequest(id="output:current", quantity="voice_coil_current"),
            OutputRequest(id="output:flux", quantity="interface_normal_derivative"),
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
    assert np.isfinite(quantities["output:velocity"].values[0, 0])
    assert np.isfinite(quantities["output:current"].values[0, 0])
    assert np.isfinite(quantities["output:field"].values[0, 0])
    assert quantities["output:flux"].values.shape == (1, 0)
    assert result.diagnostics["interface_count"] == 0
    assert result.diagnostics["interface_ids"] == []
    assert result.diagnostics["pressure_continuity_error"] is None
    assert result.diagnostics["flux_conservation_error"] is None
    assert result.diagnostics["interface_pressure_continuity_errors"] == []
    assert result.diagnostics["interface_flux_conservation_errors"] == []
    assert result.diagnostics["relative_residual"] < 1e-8
    assert result.diagnostics["all_bem_replay_error"] < 1e-8


@pytest.mark.skipif(
    os.environ.get("BLAB_RUN_COUPLED_CUDA") != "1",
    reason="Set BLAB_RUN_COUPLED_CUDA=1 to run electrodynamic CPU/CUDA parity.",
)
def test_coupled_electrodynamic_cuda_matches_cpu() -> None:
    compiled = PhysicalSystemCompiler().compile(_bidirectional_electrodynamic_fixture_system())
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
        relative_error = float(np.linalg.norm(cuda_quantities[quantity_id] - cpu_quantities[quantity_id])) / scale
        assert relative_error < 5e-3, (
            quantity_id,
            relative_error,
            cpu_quantities[quantity_id],
            cuda_quantities[quantity_id],
        )
    assert cpu_result.diagnostics["linear_backend"] == "cpu"
    assert cuda_result.diagnostics["linear_backend"] == "cuda"
    assert cpu_result.diagnostics["formulation"] == "monolithic"
    assert cuda_result.diagnostics["formulation"] == "fem_interface_condensed"
    assert cuda_result.diagnostics["static_condensation_requested"] is True
    assert cuda_result.diagnostics["static_condensation_active"] is True


@pytest.mark.skipif(
    os.environ.get("BLAB_RUN_COUPLED_CUDA") != "1",
    reason="Set BLAB_RUN_COUPLED_CUDA=1 to run sealed zero-interface CPU/CUDA parity.",
)
def test_coupled_sealed_zero_interface_cuda_matches_cpu() -> None:
    compiled = PhysicalSystemCompiler().compile(_sealed_electrodynamic_fixture_system())
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
        relative_error = float(np.linalg.norm(cuda_quantities[quantity_id] - cpu_quantities[quantity_id])) / scale
        assert relative_error < 5e-3, (quantity_id, relative_error)
    assert cpu_result.diagnostics["interface_count"] == 0
    assert cuda_result.diagnostics["interface_count"] == 0
    assert cuda_result.diagnostics["formulation"] == "fem_interface_condensed"
    assert cuda_result.diagnostics["static_condensation_active"] is True
    assert cuda_result.diagnostics["solved_system_order"] < cuda_result.diagnostics["full_system_order"]


@pytest.mark.skipif(
    os.environ.get("BLAB_RUN_COUPLED_CUDA") != "1",
    reason="Set BLAB_RUN_COUPLED_CUDA=1 to run the simple sealed condensed integration.",
)
def test_simple_sealed_cuda_condensed_solve() -> None:
    compiled = PhysicalSystemCompiler().compile(_simple_sealed_fixture_system())
    request = SystemSolveRequest(
        compiled_system=compiled,
        frequencies_hz=(200.0,),
        excitation_port_ids=("excitation:simple-sealed-driver",),
        outputs=(
            OutputRequest(id="output:velocity", quantity="diaphragm_velocity"),
            OutputRequest(id="output:current", quantity="voice_coil_current"),
            OutputRequest(
                id="output:field",
                quantity="exterior_pressure",
                options={"points_m": [[0.0, 0.0, 0.5]]},
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

    (result,) = tuple(
        CoupledProductionBackend(
            bem_backend="cuda",
            julia_executable=julia_executable,
            persistent_worker=True,
        )
        .create_system_session(request)
        .solve_stream()
    )

    quantities = {quantity.id: quantity for quantity in result.quantities}
    assert np.isfinite(quantities["output:velocity"].values[0, 0])
    assert np.isfinite(quantities["output:current"].values[0, 0])
    assert np.isfinite(quantities["output:field"].values[0, 0])
    assert result.diagnostics["interface_count"] == 0
    assert result.diagnostics["pressure_continuity_error"] is None
    assert result.diagnostics["flux_conservation_error"] is None
    assert result.diagnostics["formulation"] == "fem_interface_condensed"
    assert result.diagnostics["static_condensation_active"] is True
    assert result.diagnostics["solved_system_order"] < result.diagnostics["full_system_order"]


@pytest.mark.skipif(
    os.environ.get("BLAB_RUN_COUPLED_CUDA") != "1",
    reason="Set BLAB_RUN_COUPLED_CUDA=1 to run the condensed SKRAM integration.",
)
def test_skram_multi_chamber_cuda_condensed_solve(tmp_path: Path) -> None:
    compiled = PhysicalSystemCompiler().compile(
        _skram_fixture_system(tmp_path),
        symmetry_mode="x",
    )
    request = SystemSolveRequest(
        compiled_system=compiled,
        frequencies_hz=(200.0,),
        excitation_port_ids=("excitation:skram-driver",),
        outputs=(
            OutputRequest(id="output:velocity", quantity="diaphragm_velocity"),
            OutputRequest(id="output:current", quantity="voice_coil_current"),
        ),
        solver_options={
            "symmetry": "x",
            "quadrature_order": 1,
            "singular_order": 1,
            "validation_diagnostics": False,
            "static_condensation": True,
        },
    )
    julia_executable = os.environ.get("BLAB_JULIA_EXE", "julia")

    (result,) = tuple(
        CoupledProductionBackend(
            bem_backend="cuda",
            julia_executable=julia_executable,
            persistent_worker=True,
        )
        .create_system_session(request)
        .solve_stream()
    )

    quantities = {quantity.id: quantity for quantity in result.quantities}
    velocity = quantities["output:velocity"].values[0, 0]
    current = quantities["output:current"].values[0, 0]
    electrical_impedance = 6.0 - 1j * 2.0 * np.pi * 200.0 * 0.0005
    electrical_residual = (
        abs(electrical_impedance * current + 7.0 * velocity - DEFAULT_TRANSDUCER_REFERENCE_VOLTAGE_V)
        / DEFAULT_TRANSDUCER_REFERENCE_VOLTAGE_V
    )

    assert np.isfinite(velocity)
    assert np.isfinite(current)
    assert electrical_residual < 5e-4
    assert result.diagnostics["bounded_region_count"] == 2
    assert result.diagnostics["interface_count"] == 2
    assert result.diagnostics["symmetry"] == "x"
    assert result.diagnostics["formulation"] == "fem_interface_condensed"
    assert result.diagnostics["static_condensation_active"] is True
    assert result.diagnostics["solved_system_order"] < result.diagnostics["full_system_order"]


def test_editable_physical_system_round_trips_for_project_persistence() -> None:
    system = _electrodynamic_fixture_system()

    restored = physical_system_from_dict(physical_system_to_dict(system))

    assert restored == system
    assert restored.components[0].parameters["mmd_kg"] == pytest.approx(0.015)
    assert restored.components[0].parameters["motion_axis"] == [0.0, 0.0, 1.0]
    assert restored.excitation_ports[0].kind == ExcitationPortKind.VOLTAGE


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
            "motion_axis": [0.0, 0.0, 1.0],
            "motion_profile": "rigid_translation",
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
    exterior_boundary = next(boundary for boundary in system.boundaries if boundary.id == "boundary:exterior")
    moving_exterior = replace(exterior_boundary, kind=BoundaryKind.MOVING)
    component = replace(
        system.components[0],
        boundary_ids=("boundary:radiator", "boundary:exterior"),
    )
    return replace(
        system,
        boundaries=tuple(
            moving_exterior if boundary.id == moving_exterior.id else boundary for boundary in system.boundaries
        ),
        components=(component,),
    )


def _sealed_electrodynamic_fixture_system() -> PhysicalSystem:
    system = _bidirectional_electrodynamic_fixture_system()
    return replace(
        system,
        boundaries=tuple(
            replace(boundary, kind=BoundaryKind.RIGID) if boundary.kind == BoundaryKind.INTERFACE else boundary
            for boundary in system.boundaries
        ),
        interfaces=(),
    )


def _simple_sealed_fixture_system() -> PhysicalSystem:
    interior_mesh_id = "mesh:simple-sealed-interior"
    exterior_mesh_id = "mesh:simple-sealed-exterior"
    interior_region_id = "region:simple-sealed-interior"
    exterior_region_id = "region:simple-sealed-exterior"
    component = PhysicalComponent(
        id="component:simple-sealed-driver",
        name="Simple sealed driver",
        kind=ComponentKind.ELECTRODYNAMIC_TRANSDUCER,
        boundary_ids=(
            "boundary:simple-sealed-rear-diaphragm",
            "boundary:simple-sealed-front-diaphragm",
        ),
        parameters={
            "re_ohm": 6.0,
            "le_h": 0.0005,
            "bl_n_per_a": 7.0,
            "mmd_kg": 0.015,
            "cms_m_per_n": 0.0005,
            "rms_n_s_per_m": 1.0,
            "motion_axis": [0.0, 0.0, 1.0],
            "motion_profile": "rigid_translation",
        },
    )
    return PhysicalSystem(
        id="system:simple-sealed",
        name="Simple sealed enclosure",
        meshes=(
            MeshResource(
                id=interior_mesh_id,
                name="Simple sealed interior",
                file=str(SIMPLE_SEALED_FIXTURE_ROOT / "interior.msh"),
                purpose=MeshPurpose.FEM_VOLUME,
                scale_to_m=0.001,
            ),
            MeshResource(
                id=exterior_mesh_id,
                name="Simple sealed exterior",
                file=str(SIMPLE_SEALED_FIXTURE_ROOT / "Exterior.msh"),
                purpose=MeshPurpose.BEM_SURFACE,
                scale_to_m=0.001,
            ),
        ),
        regions=(
            AcousticRegion(
                id=interior_region_id,
                name="Sealed interior air",
                kind=AcousticRegionKind.BOUNDED_AIR,
                mesh_ids=(interior_mesh_id,),
                volume_groups=(
                    PhysicalGroupRef(
                        mesh_id=interior_mesh_id,
                        dimension=3,
                        name="interior",
                    ),
                ),
            ),
            AcousticRegion(
                id=exterior_region_id,
                name="Exterior air",
                kind=AcousticRegionKind.UNBOUNDED_AIR,
                mesh_ids=(exterior_mesh_id,),
            ),
        ),
        boundaries=(
            Boundary(
                id="boundary:simple-sealed-rear-diaphragm",
                name="Rear diaphragm",
                region_id=interior_region_id,
                group=PhysicalGroupRef(
                    mesh_id=interior_mesh_id,
                    dimension=2,
                    name="Radiator",
                ),
                kind=BoundaryKind.MOVING,
            ),
            Boundary(
                id="boundary:simple-sealed-interior-wall",
                name="Interior enclosure wall",
                region_id=interior_region_id,
                group=PhysicalGroupRef(
                    mesh_id=interior_mesh_id,
                    dimension=2,
                    name="interior_boundary",
                ),
                kind=BoundaryKind.RIGID,
            ),
            Boundary(
                id="boundary:simple-sealed-front-diaphragm",
                name="Front diaphragm",
                region_id=exterior_region_id,
                group=PhysicalGroupRef(
                    mesh_id=exterior_mesh_id,
                    dimension=2,
                    name="woofer",
                ),
                kind=BoundaryKind.MOVING,
            ),
            Boundary(
                id="boundary:simple-sealed-exterior-wall",
                name="Exterior enclosure wall",
                region_id=exterior_region_id,
                group=PhysicalGroupRef(
                    mesh_id=exterior_mesh_id,
                    dimension=2,
                    name="enclosure",
                ),
                kind=BoundaryKind.RIGID,
            ),
        ),
        interfaces=(),
        components=(component,),
        excitation_ports=(
            ExcitationPort(
                id="excitation:simple-sealed-driver",
                name="Simple sealed driver 2.83 V",
                component_id=component.id,
                kind=ExcitationPortKind.VOLTAGE,
            ),
        ),
    )


def _skram_fixture_system(tmp_path: Path) -> PhysicalSystem:
    front_file = SKRAM_FIXTURE_ROOT / "SkramFrontChamber.msh"
    rear_file = SKRAM_FIXTURE_ROOT / "SkramRearChamber.msh"
    exterior_file = SKRAM_FIXTURE_ROOT / "SkramExterior.msh"
    front_mesh = meshio.read(front_file)
    rear_mesh = meshio.read(rear_file)
    exterior_mesh = meshio.read(exterior_file)
    front_mesh.points = np.asarray(front_mesh.points, dtype=float) * 0.001
    rear_mesh.points = np.asarray(rear_mesh.points, dtype=float) * 0.001
    exterior_mesh.points = np.asarray(exterior_mesh.points, dtype=float) * 0.001
    conformed_exterior, _ = conform_bem_interface_to_fem(
        front_mesh,
        exterior_mesh,
        fem_interface_name="Interface",
        bem_interface_name="FrontChamberInterface",
        merge_tolerance=1e-8,
        symmetry_mode="x",
        protected_bem_interface_names=("RearChamberInterface",),
    )
    conformed_exterior, _ = conform_bem_interface_to_fem(
        rear_mesh,
        conformed_exterior,
        fem_interface_name="Interface",
        bem_interface_name="RearChamberInterface",
        merge_tolerance=1e-8,
        symmetry_mode="x",
        protected_bem_interface_names=("FrontChamberInterface",),
    )
    conformed_file = tmp_path / "skram_exterior_conformed.msh"
    meshio.write(conformed_file, conformed_exterior, file_format="gmsh22", binary=False)

    meshes = (
        MeshResource(
            id="mesh:skram-front",
            name="SKRAM front chamber",
            file=str(front_file),
            purpose=MeshPurpose.FEM_VOLUME,
            scale_to_m=0.001,
        ),
        MeshResource(
            id="mesh:skram-rear",
            name="SKRAM rear chamber",
            file=str(rear_file),
            purpose=MeshPurpose.FEM_VOLUME,
            scale_to_m=0.001,
        ),
        MeshResource(
            id="mesh:skram-exterior",
            name="SKRAM conformed exterior",
            file=str(conformed_file),
            purpose=MeshPurpose.BEM_SURFACE,
        ),
    )
    regions = (
        AcousticRegion(
            id="region:skram-front",
            name="Front chamber",
            kind=AcousticRegionKind.BOUNDED_AIR,
            mesh_ids=("mesh:skram-front",),
            volume_groups=(
                PhysicalGroupRef(
                    mesh_id="mesh:skram-front",
                    dimension=3,
                    name="FrontChamber",
                ),
            ),
        ),
        AcousticRegion(
            id="region:skram-rear",
            name="Rear chamber",
            kind=AcousticRegionKind.BOUNDED_AIR,
            mesh_ids=("mesh:skram-rear",),
            volume_groups=(
                PhysicalGroupRef(
                    mesh_id="mesh:skram-rear",
                    dimension=3,
                    name="RearChamber",
                ),
            ),
        ),
        AcousticRegion(
            id="region:skram-exterior",
            name="Exterior",
            kind=AcousticRegionKind.UNBOUNDED_AIR,
            mesh_ids=("mesh:skram-exterior",),
        ),
    )
    boundaries = (
        Boundary(
            id="boundary:front-radiator",
            name="Front diaphragm side",
            region_id="region:skram-front",
            group=PhysicalGroupRef(
                mesh_id="mesh:skram-front",
                dimension=2,
                name="Diaphragm",
            ),
            kind=BoundaryKind.MOVING,
        ),
        Boundary(
            id="boundary:front-port",
            name="Front chamber port",
            region_id="region:skram-front",
            group=PhysicalGroupRef(
                mesh_id="mesh:skram-front",
                dimension=2,
                name="Interface",
            ),
            kind=BoundaryKind.INTERFACE,
        ),
        Boundary(
            id="boundary:front-wall",
            name="Front chamber wall",
            region_id="region:skram-front",
            group=PhysicalGroupRef(
                mesh_id="mesh:skram-front",
                dimension=2,
                name="FrontChamber_boundary",
            ),
            kind=BoundaryKind.RIGID,
        ),
        Boundary(
            id="boundary:rear-radiator",
            name="Rear diaphragm side",
            region_id="region:skram-rear",
            group=PhysicalGroupRef(
                mesh_id="mesh:skram-rear",
                dimension=2,
                name="Diaphragm",
            ),
            kind=BoundaryKind.MOVING,
        ),
        Boundary(
            id="boundary:rear-port",
            name="Rear chamber port",
            region_id="region:skram-rear",
            group=PhysicalGroupRef(
                mesh_id="mesh:skram-rear",
                dimension=2,
                name="Interface",
            ),
            kind=BoundaryKind.INTERFACE,
        ),
        Boundary(
            id="boundary:rear-wall",
            name="Rear chamber wall",
            region_id="region:skram-rear",
            group=PhysicalGroupRef(
                mesh_id="mesh:skram-rear",
                dimension=2,
                name="RearChamber_boundary",
            ),
            kind=BoundaryKind.RIGID,
        ),
        Boundary(
            id="boundary:skram-exterior",
            name="Exterior enclosure",
            region_id="region:skram-exterior",
            group=PhysicalGroupRef(
                mesh_id="mesh:skram-exterior",
                dimension=2,
                name="Exterior",
            ),
            kind=BoundaryKind.RIGID,
        ),
        Boundary(
            id="boundary:front-exterior-interface",
            name="Exterior front port",
            region_id="region:skram-exterior",
            group=PhysicalGroupRef(
                mesh_id="mesh:skram-exterior",
                dimension=2,
                name="FrontChamberInterface",
            ),
            kind=BoundaryKind.INTERFACE,
        ),
        Boundary(
            id="boundary:rear-exterior-interface",
            name="Exterior rear port",
            region_id="region:skram-exterior",
            group=PhysicalGroupRef(
                mesh_id="mesh:skram-exterior",
                dimension=2,
                name="RearChamberInterface",
            ),
            kind=BoundaryKind.INTERFACE,
        ),
    )
    component = PhysicalComponent(
        id="component:skram-driver",
        name="SKRAM rigid driver",
        kind=ComponentKind.ELECTRODYNAMIC_TRANSDUCER,
        boundary_ids=(
            "boundary:front-radiator",
            "boundary:rear-radiator",
        ),
        parameters={
            "re_ohm": 6.0,
            "le_h": 0.0005,
            "bl_n_per_a": 7.0,
            "mmd_kg": 0.015,
            "cms_m_per_n": 0.0005,
            "rms_n_s_per_m": 1.0,
            "motion_axis": [0.0, 0.0, -1.0],
            "motion_profile": "rigid_translation",
            "symmetry_role": "fractional_driver",
            "surface_completion_factor": 2,
            "physical_driver_orbit_count": 1,
            "fractional_symmetry_axes": ["x"],
        },
    )
    return PhysicalSystem(
        id="system:skram",
        name="SKRAM two-chamber fixture",
        meshes=meshes,
        regions=regions,
        boundaries=boundaries,
        interfaces=(
            AcousticInterface(
                id="interface:skram-front",
                name="Front chamber to exterior",
                bounded_boundary_id="boundary:front-port",
                unbounded_boundary_id="boundary:front-exterior-interface",
            ),
            AcousticInterface(
                id="interface:skram-rear",
                name="Rear chamber to exterior",
                bounded_boundary_id="boundary:rear-port",
                unbounded_boundary_id="boundary:rear-exterior-interface",
            ),
        ),
        components=(component,),
        excitation_ports=(
            ExcitationPort(
                id="excitation:skram-driver",
                name="SKRAM 2.83 V",
                component_id=component.id,
                kind=ExcitationPortKind.VOLTAGE,
            ),
        ),
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
