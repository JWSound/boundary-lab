from pathlib import Path

import meshio
import numpy as np

from blab.config import ChannelConfig, MeshConfig, RadiatorConfig
from blab.live import FrequencyResult, LiveSolveDataset
from blab.ui.application_state import (
    OperationPhase,
    SolveCompletion,
    solve_invalidation_policy,
)
from blab.ui.mesh_assembly import MeshAssemblyService
from blab.ui.project_state import ImportedMeshState
from blab.ui.result_projection import ProjectionOptions, ResultProjectionService
from blab.ui.simulation_assembler import SimulationAssembler, SimulationParameters


def _write_triangle_mesh(path: Path, tag: int = 2) -> None:
    meshio.write(
        path,
        meshio.Mesh(
            points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            cells=[("triangle", np.array([[0, 1, 2]], dtype=np.int64))],
            cell_data={"gmsh:physical": [np.array([tag], dtype=np.int32)]},
            field_data={"driver": np.array([tag, 2], dtype=np.int32)},
        ),
        file_format="gmsh22",
        binary=False,
    )


def test_invalidation_policy_keeps_destructive_rapid_iteration_behavior() -> None:
    normal = solve_invalidation_policy("geometry_generation_started")
    project_load = solve_invalidation_policy("project_loaded")
    unknown = solve_invalidation_policy("future_change")

    assert normal.clear_solve_results is True
    assert normal.clear_comparison_history is False
    assert project_load.clear_solve_results is True
    assert project_load.clear_comparison_history is True
    assert unknown.clear_solve_results is True


def test_solve_completion_only_marks_full_frequency_sets_complete() -> None:
    partial = SolveCompletion(OperationPhase.COMPLETED, solved_count=4, expected_count=5, elapsed_s=1.0)
    complete = SolveCompletion(OperationPhase.COMPLETED, solved_count=5, expected_count=5, elapsed_s=1.0)

    assert partial.completed is False
    assert complete.completed is True


def test_simulation_assembler_builds_domain_request_without_widgets() -> None:
    prepared = SimulationAssembler().prepare(
        mesh_configs=(MeshConfig(name="speaker", file="speaker.msh", scale_factor=0.001),),
        radiators=(RadiatorConfig(name="driver", mesh="speaker", tag=2),),
        channels=(ChannelConfig(name="main"),),
        parameters=SimulationParameters(
            freq_min_hz=20000,
            freq_max_hz=200,
            freq_count=5,
            observation_distance_m=3.0,
            polar_angle_step_deg=5.0,
            use_burton_miller=False,
            gmres_tolerance=1e-4,
            normalized_channel_correction=True,
            horizontal_normalization_angle_deg=10.0,
            spherical_sampling_enabled=False,
            spherical_sampling_points=100,
            symmetry="off",
        ),
    )

    assert prepared.config.freq_min == 200.0
    assert prepared.config.freq_max == 20000.0
    assert prepared.config.distance == 3.0
    assert prepared.config.gmres_tolerance == 1e-4
    assert prepared.ordered_frequencies.size == 5


def test_mesh_assembly_prepares_preview_and_solver_contract(tmp_path: Path) -> None:
    mesh_path = tmp_path / "cabinet.msh"
    _write_triangle_mesh(mesh_path)
    service = MeshAssemblyService(tmp_path / "prepared")

    assembly = service.prepare(
        generated_mesh_configs=(),
        imported_meshes=(
            ImportedMeshState(
                name="cabinet",
                source_file=str(mesh_path),
                cleaned_file=str(mesh_path),
            ),
        ),
        radiators=(RadiatorConfig(name="cabinet:driver", mesh="cabinet", tag=2),),
        stitch_imported_meshes=False,
        stitch_tolerance_mm=2.0,
        symmetry="off",
    )

    assert assembly.mesh_configs[0].name == "cabinet"
    assert assembly.radiators[0].mesh == "cabinet"
    assert assembly.surface_tags == {"cabinet:driver": ("cabinet", 2)}


def test_result_projection_returns_typed_plot_models() -> None:
    dataset = LiveSolveDataset(
        np.array([-90.0, 0.0, 90.0], dtype=np.float32),
        radiator_names=np.array(["driver"]),
    )
    dataset.add(
        FrequencyResult(
            freq_hz=1000.0,
            horizontal_spl_norm_db=np.array([-6.0, 0.0, -6.0]),
            vertical_spl_norm_db=np.array([-8.0, 0.0, -8.0]),
            impedance=np.array([[1.0, 0.2]], dtype=np.float32),
        )
    )
    projection = ResultProjectionService().prepare(
        dataset,
        (ChannelConfig(name="main"),),
        ProjectionOptions(
            angle_samples=3,
            freq_samples=1,
            octave_smoothing=None,
            horizontal_reference_angle_deg=0.0,
            vertical_reference_angle_deg=0.0,
            spin_horizontal_reference_angle_deg=0.0,
            spin_vertical_reference_angle_deg=0.0,
            min_db=-30.0,
            max_db=0.0,
        ),
    )

    assert projection is not None
    assert projection.isobar.horizontal_db.shape == (3, 1)
    assert projection.impedance.radiator_names.tolist() == ["driver"]
    assert projection.response.freq_hz.tolist() == [1000.0]
