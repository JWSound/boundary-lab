import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QComboBox

from blab.physical_compiler import PhysicalSystemCompiler
from blab.physical_model import AcousticRegionKind, BoundaryKind
from blab.solvers.coupled_backend import CoupledProductionBackend
from blab.ui.dialogs import MeshDialogEntry
from blab.ui.mesh_assembly import MeshAssemblyService
from blab.ui.project_state import ImportedMeshState
from blab.ui.system_config import SystemConfigDialog, inspect_system_meshes
from blab.ui.system_solve import CoupledSolveWorker, prepare_coupled_ui_solve

_APP = QApplication.instance() or QApplication([])
FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"


def _fixture_mesh_entries() -> tuple[MeshDialogEntry, ...]:
    return (
        MeshDialogEntry(
            name="Interior",
            source_file=str(FIXTURE_ROOT / "femvolume.msh"),
            scale_factor=0.001,
        ),
        MeshDialogEntry(
            name="Exterior",
            source_file=str(FIXTURE_ROOT / "exterior_conforming.msh"),
            scale_factor=0.001,
        ),
    )


def _configured_fixture_dialog() -> SystemConfigDialog:
    dialog = SystemConfigDialog(inspect_system_meshes(_fixture_mesh_entries()), None, ("main",))
    dialog._add_default_region()
    dialog._refresh_boundaries()
    assignments = {
        ("Interior", "Radiator"): BoundaryKind.MOVING,
        ("Interior", "Volume_boundary"): BoundaryKind.RIGID,
        ("Interior", "Interface"): BoundaryKind.INTERFACE,
        ("Exterior", "ExteriorBox"): BoundaryKind.RIGID,
        ("Exterior", "Interface"): BoundaryKind.INTERFACE,
    }
    for row in range(dialog.boundaries_table.rowCount()):
        mesh_name = dialog.boundaries_table.item(row, 1).text()
        group_name = dialog.boundaries_table.item(row, 2).text()
        combo = dialog.boundaries_table.cellWidget(row, 3)
        assert isinstance(combo, QComboBox)
        combo.setCurrentIndex(combo.findData(assignments[(mesh_name, group_name)]))
    dialog._identify_interfaces()
    dialog._add_component()
    boundary_combo = dialog.components_table.cellWidget(0, 2)
    assert isinstance(boundary_combo, QComboBox)
    boundary_combo.setCurrentIndex(1)
    return dialog


def test_mesh_inventory_preserves_existing_scale_translation_and_volume_groups() -> None:
    entries = list(_fixture_mesh_entries())
    entries[0] = MeshDialogEntry(
        name=entries[0].name,
        source_file=entries[0].source_file,
        scale_factor=entries[0].scale_factor,
        translation_mm=(1.0, 2.0, 3.0),
    )
    fem, bem = inspect_system_meshes(tuple(entries))

    assert fem.has_tetrahedra
    assert fem.volume_groups == ("Volume",)
    assert set(fem.surface_groups) == {"Interface", "Radiator", "Volume_boundary"}
    assert fem.scale_to_m == 0.001
    assert fem.translation_m == (0.001, 0.002, 0.003)
    assert not bem.has_tetrahedra
    assert bem.volume_groups == ()


def test_legacy_surface_cleaner_skips_imported_tetrahedral_mesh() -> None:
    source = FIXTURE_ROOT / "femvolume.msh"
    state = ImportedMeshState(
        name="Interior",
        source_file=str(source),
        cleaned_file="should_not_be_used_for_a_volume_mesh.msh",
    )

    (preserved,) = MeshAssemblyService(".").clean_imported_meshes((state,))

    assert preserved.source_file == str(source)
    assert preserved.cleaned_file is None


def test_tabbed_system_editor_builds_compilable_coupled_fixture() -> None:
    dialog = _configured_fixture_dialog()

    system = dialog.physical_system()
    compiled = PhysicalSystemCompiler().compile(system)

    assert [region.kind for region in system.regions] == [
        AcousticRegionKind.UNBOUNDED_AIR,
        AcousticRegionKind.BOUNDED_AIR,
    ]
    assert len(system.interfaces) == 1
    assert len(system.components) == 1
    assert len(system.excitation_ports) == 1
    assert len(compiled.interfaces[0].topology.fem_face_indices) == 180
    assert "channel" not in system.components[0].parameters
    assert dialog.configuration().component_channel_by_id == {system.components[0].id: "main"}


def test_coupled_ui_request_uses_excitation_basis_and_polar_field_points() -> None:
    system = _configured_fixture_dialog().physical_system()

    prepared = prepare_coupled_ui_solve(
        system,
        freq_min_hz=500.0,
        freq_max_hz=1000.0,
        freq_count=3,
        observation_distance_m=2.0,
        polar_angle_step_deg=90.0,
    )

    assert prepared.polar_angle_deg.tolist() == [-180.0, -90.0, 0.0, 90.0, 180.0]
    assert prepared.horizontal_count == 5
    assert prepared.vertical_count == 5
    assert prepared.excitation_channel_names.tolist() == ["main"]
    assert prepared.request.excitation_port_ids == tuple(
        port.id for port in prepared.request.compiled_system.excitation_ports
    )
    assert prepared.request.solver_options["validation_diagnostics"] is False
    assert prepared.request.solver_options["cache_frequency_invariant"] is True
    assert "precision" not in prepared.request.solver_options
    assert "bem_backend" not in prepared.request.solver_options
    assert prepared.backend_id == "beat_cpu"
    points = np.asarray(prepared.request.outputs[0].options["points_m"])
    assert points.shape == (10, 3)


def test_excitation_rows_on_the_same_channel_are_combined_before_dsp() -> None:
    system = _configured_fixture_dialog().physical_system()
    prepared = prepare_coupled_ui_solve(
        system,
        freq_min_hz=500.0,
        freq_max_hz=500.0,
        freq_count=1,
        observation_distance_m=2.0,
        polar_angle_step_deg=90.0,
    )
    prepared = replace(
        prepared,
        excitation_channel_names=np.asarray(["main", "main"]),
    )
    worker = CoupledSolveWorker(prepared)
    horizontal = np.asarray([[1.0 + 0.0j], [2.0 + 0.0j]])
    vertical = np.asarray([[3.0 + 0.0j], [4.0 + 0.0j]])

    names, grouped_horizontal, grouped_vertical, sphere = worker._combine_channel_rows(
        horizontal,
        vertical,
        None,
    )

    assert names.tolist() == ["main"]
    assert grouped_horizontal.tolist() == [[3.0 + 0.0j]]
    assert grouped_vertical.tolist() == [[7.0 + 0.0j]]
    assert sphere is None


@pytest.mark.skipif(
    os.environ.get("BLAB_RUN_COUPLED_REFERENCE") != "1",
    reason="Set BLAB_RUN_COUPLED_REFERENCE=1 to run the Julia GUI-path integration.",
)
def test_coupled_ui_request_returns_live_plot_pressure_basis() -> None:
    system = _configured_fixture_dialog().physical_system()
    prepared = prepare_coupled_ui_solve(
        system,
        freq_min_hz=500.0,
        freq_max_hz=500.0,
        freq_count=1,
        observation_distance_m=2.0,
        polar_angle_step_deg=90.0,
    )
    backend = CoupledProductionBackend(
        bem_backend="cpu",
        julia_executable=os.environ.get("BLAB_JULIA_EXE", "julia"),
    )

    (system_result,) = tuple(backend.create_system_session(prepared.request).solve_stream())
    live_result = CoupledSolveWorker(prepared)._to_live_result(system_result)

    assert live_result.has_channel_basis
    assert live_result.horizontal_pressure.shape == (1, 5)
    assert live_result.vertical_pressure.shape == (1, 5)
    assert live_result.channel_names.tolist() == ["main"]
    assert system_result.quantities[0].values.dtype == np.complex64
    assert system_result.diagnostics["precision"] == "float32"
    assert system_result.diagnostics["bem_backend"] == "cpu"
    assert live_result.timings.assembly_s > 0.0
    assert live_result.timings.solve_s > 0.0
    assert live_result.timings.field_s > 0.0


def test_coupled_ui_request_routes_cuda_backend() -> None:
    system = _configured_fixture_dialog().physical_system()

    prepared = prepare_coupled_ui_solve(
        system,
        freq_min_hz=500.0,
        freq_max_hz=500.0,
        freq_count=1,
        observation_distance_m=2.0,
        polar_angle_step_deg=90.0,
        backend_id="beat_cuda",
    )

    assert prepared.backend_id == "beat_cuda"
    assert "precision" not in prepared.request.solver_options
    assert "bem_backend" not in prepared.request.solver_options


def test_coupled_ui_request_rejects_non_beat_backend() -> None:
    system = _configured_fixture_dialog().physical_system()

    with pytest.raises(ValueError, match="require BEAT Engine"):
        prepare_coupled_ui_solve(
            system,
            freq_min_hz=500.0,
            freq_max_hz=500.0,
            freq_count=1,
            observation_distance_m=2.0,
            polar_angle_step_deg=90.0,
            backend_id="local",
        )


@pytest.mark.skipif(
    os.environ.get("BLAB_RUN_COUPLED_CUDA") != "1",
    reason="Set BLAB_RUN_COUPLED_CUDA=1 to run coupled CPU/CUDA parity.",
)
def test_coupled_cuda_matches_cpu_exterior_pressure() -> None:
    system = _configured_fixture_dialog().physical_system()
    prepared = prepare_coupled_ui_solve(
        system,
        freq_min_hz=500.0,
        freq_max_hz=500.0,
        freq_count=1,
        observation_distance_m=2.0,
        polar_angle_step_deg=45.0,
        backend_id="beat_cpu",
    )
    julia_executable = os.environ.get("BLAB_JULIA_EXE", "julia")
    cpu_backend = CoupledProductionBackend(
        bem_backend="cpu",
        julia_executable=julia_executable,
    )
    cuda_backend = CoupledProductionBackend(
        bem_backend="cuda",
        julia_executable=julia_executable,
    )

    (cpu_result,) = tuple(cpu_backend.create_system_session(prepared.request).solve_stream())
    (cuda_result,) = tuple(cuda_backend.create_system_session(prepared.request).solve_stream())
    cpu_pressure = cpu_result.quantities[0].values
    cuda_pressure = cuda_result.quantities[0].values
    relative_l2 = np.linalg.norm(cuda_pressure - cpu_pressure) / np.linalg.norm(cpu_pressure)

    assert cpu_result.diagnostics["bem_backend"] == "cpu"
    assert cuda_result.diagnostics["bem_backend"] == "cuda"
    assert cpu_pressure.dtype == np.complex64
    assert cuda_pressure.dtype == np.complex64
    assert relative_l2 < 5e-3
