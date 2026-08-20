from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from blab.headless import HeadlessSolveSpec, load_headless_project, prepare_headless_solve
from blab.physical_compiler import PhysicalModelCompileError, PhysicalSystemCompiler
from blab.physical_model import (
    AcousticRegionKind,
    BoundaryKind,
    PhysicalSolveKind,
    infer_physical_solve_kind,
)
from blab.solve_results import FEM_NODAL_PRESSURE_ID, FEM_VOLUME_DOMAIN_ID
from blab.solvers.coupled_backend import (
    DEFAULT_COUPLED_CPU_PROJECT,
    CoupledReferenceBackend,
    PhysicalSystemProductionBackend,
    validate_system_capabilities,
)
from blab.system_solve import prepare_system_ui_solve

FIXTURE = Path(__file__).parents[1] / "examples" / "compression_driver" / "compression_driver.blab.json"


def _prepared():
    project = load_headless_project(FIXTURE)
    return project, prepare_system_ui_solve(
        project.physical_system,
        freq_min_hz=1000.0,
        freq_max_hz=1000.0,
        freq_count=1,
        observation_distance_m=1.0,
        polar_angle_step_deg=10.0,
        component_channel_by_id=project.component_channel_by_id,
        backend_id="beat_cpu",
        symmetry_mode="off",
    )


def test_compression_driver_fixture_compiles_as_interior_fem() -> None:
    project, prepared = _prepared()
    compiled = PhysicalSystemCompiler().compile(project.physical_system)

    assert infer_physical_solve_kind(project.physical_system) == PhysicalSolveKind.INTERIOR_FEM
    assert prepared.solve_kind == PhysicalSolveKind.INTERIOR_FEM
    assert compiled.interfaces == ()
    assert len(compiled.regions) == 2
    assert (
        next(boundary for boundary in compiled.boundaries if boundary.id == "boundary:front-termination").kind
        == BoundaryKind.PLANE_WAVE_TUBE_TERMINATION
    )
    assert "boundary_motion_signs" not in compiled.components[0].parameters
    assert compiled.components[0].parameters["motion_axis"] == [0.0, 0.0, 1.0]


def test_interior_fem_request_omits_exterior_outputs_and_retains_volume_pressure() -> None:
    _project, prepared = _prepared()
    outputs = {output.id: output for output in prepared.request.outputs}
    domains = {domain.id: domain for domain in prepared.result_domains}

    validate_system_capabilities(prepared.request)
    assert "ui:exterior-pressure" not in outputs
    assert FEM_NODAL_PRESSURE_ID in outputs
    assert FEM_VOLUME_DOMAIN_ID in domains
    assert prepared.polar_angle_deg.size == 0
    assert prepared.horizontal_count == 0
    assert prepared.vertical_count == 0
    assert prepared.request.solver_options["static_condensation"] is False

    accelerated_backend = PhysicalSystemProductionBackend(bem_backend="cuda", persistent_worker=False)
    session = accelerated_backend.create_system_session(prepared.request)
    assert session.julia_project == DEFAULT_COUPLED_CPU_PROJECT.resolve()
    assert session.request.solver_options["bem_backend"] == "cpu"


def test_headless_compression_driver_request_prepares_without_exterior_observations() -> None:
    project = load_headless_project(FIXTURE)
    prepared = prepare_headless_solve(
        project,
        HeadlessSolveSpec(frequencies_hz=(1000.0,), include_project_observations=False),
        backend_id="beat_cpu",
    )

    assert prepared.solve_kind == PhysicalSolveKind.INTERIOR_FEM
    assert {output.quantity for output in prepared.request.outputs} >= {
        "fem_nodal_pressure",
        "diaphragm_velocity",
        "voice_coil_current",
    }


def test_plane_wave_termination_requires_parameter_free_bounded_boundary() -> None:
    project = load_headless_project(FIXTURE)
    system = project.physical_system
    termination = next(
        boundary for boundary in system.boundaries if boundary.kind == BoundaryKind.PLANE_WAVE_TUBE_TERMINATION
    )
    with pytest.raises(PhysicalModelCompileError, match="does not accept boundary parameters"):
        PhysicalSystemCompiler().compile(
            replace(
                system,
                boundaries=tuple(
                    replace(boundary, parameters={"impedance": 1.0}) if boundary.id == termination.id else boundary
                    for boundary in system.boundaries
                ),
            )
        )

    front_region = next(region for region in system.regions if region.id == termination.region_id)
    with pytest.raises(PhysicalModelCompileError, match="requires a bounded-air FEM region"):
        PhysicalSystemCompiler().compile(
            replace(
                system,
                regions=tuple(
                    replace(region, kind=AcousticRegionKind.UNBOUNDED_AIR) if region.id == front_region.id else region
                    for region in system.regions
                ),
            )
        )


@pytest.mark.skipif(
    os.environ.get("BLAB_RUN_INTERIOR_FEM") != "1",
    reason="Set BLAB_RUN_INTERIOR_FEM=1 to run the Julia compression-driver integration.",
)
def test_compression_driver_sparse_interior_fem_integration() -> None:
    _project, prepared = _prepared()
    request = prepared.request
    (result,) = tuple(
        CoupledReferenceBackend(
            julia_executable=os.environ.get("BLAB_JULIA_EXE", "julia"),
            persistent_worker=False,
        )
        .create_system_session(request)
        .solve_stream()
    )

    quantities = {quantity.quantity: quantity for quantity in result.quantities}
    pressure = quantities["fem_nodal_pressure"].values
    assert pressure.shape == (1, 1665)
    assert np.all(np.isfinite(pressure))
    assert np.linalg.norm(pressure) > 0.0
    assert np.isfinite(quantities["diaphragm_velocity"].values[0, 0])
    assert np.isfinite(quantities["voice_coil_current"].values[0, 0])
    assert result.diagnostics["formulation"] == "interior_fem_sparse"
    assert result.diagnostics["plane_wave_termination_boundary_ids"] == ["boundary:front-termination"]
    assert result.diagnostics["relative_residual"] < 1e-8
