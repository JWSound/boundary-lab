import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import meshio
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QComboBox, QPushButton

import blab.ui.system_config as system_config_module
import blab.ui.system_solve as system_solve_module
from blab.acoustic_materials import miki_wall_impedance_parameters
from blab.ath import read_surface_physical_names
from blab.config import RadiatorConfig
from blab.interface_conform import InterfaceConformError, validate_conforming_interfaces
from blab.physical_compiler import PhysicalSystemCompiler
from blab.physical_model import (
    AcousticRegionKind,
    Boundary,
    BoundaryKind,
    ComponentKind,
    MeshPurpose,
    MeshResource,
    PhysicalGroupRef,
    PhysicalSolveKind,
    infer_physical_solve_kind,
)
from blab.solvers.coupled_backend import CoupledProductionBackend
from blab.ui.dialogs import MeshDialogEntry
from blab.ui.exterior_system import exterior_bem_inputs
from blab.ui.mesh_assembly import MeshAssemblyService
from blab.ui.physical_system_migration import seed_exterior_system
from blab.ui.project_state import ImportedMeshState
from blab.ui.system_config import (
    SystemConfigDialog,
    _ComponentDraft,
    _ComponentEditorDialog,
    _WallImpedanceDialog,
    infer_component_motion_axis,
    inspect_system_meshes,
    interface_bem_mesh_names_for_changes,
    rebuild_configured_interfaces,
)
from blab.ui.system_solve import CoupledSolveWorker, prepare_coupled_ui_solve

_APP = QApplication.instance() or QApplication([])
FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"


def _fixture_mesh_entries(
    bem_filename: str = "exterior_conforming.msh",
) -> tuple[MeshDialogEntry, ...]:
    return (
        MeshDialogEntry(
            name="Interior",
            source_file=str(FIXTURE_ROOT / "femvolume.msh"),
            scale_factor=0.001,
        ),
        MeshDialogEntry(
            name="Exterior",
            source_file=str(FIXTURE_ROOT / bem_filename),
            scale_factor=0.001,
        ),
    )


def _configured_fixture_dialog(
    *,
    bem_filename: str = "exterior_conforming.msh",
    interface_output_root: Path | None = None,
) -> SystemConfigDialog:
    dialog = SystemConfigDialog(
        inspect_system_meshes(_fixture_mesh_entries(bem_filename)),
        None,
        ("main",),
        interface_output_root=interface_output_root,
    )
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
    (moving_boundary,) = dialog._moving_boundaries()
    dialog._append_component_draft(
        name="Radiator 1",
        boundary_ids=(moving_boundary.id,),
        channel="main",
    )
    return dialog


def _planar_surface_mesh(
    *,
    group_name: str,
    z_m: float,
    reverse: bool,
) -> meshio.Mesh:
    triangle = [0, 2, 1] if reverse else [0, 1, 2]
    return meshio.Mesh(
        points=np.asarray(
            [
                [0.0, 0.0, z_m],
                [1.0, 0.0, z_m],
                [0.0, 1.0, z_m],
            ]
        ),
        cells=[("triangle", np.asarray([triangle], dtype=np.int32))],
        cell_data={
            "gmsh:physical": [np.asarray([1], dtype=np.int32)],
            "gmsh:geometrical": [np.asarray([1], dtype=np.int32)],
        },
        field_data={group_name: np.asarray([1, 2], dtype=np.int32)},
    )


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


def test_system_editor_uses_reduced_mesh_only_for_symmetry_analysis() -> None:
    full_path = FIXTURE_ROOT / "SAWMOD" / "SMfemvolume_full.msh"
    reduced_path = FIXTURE_ROOT / "SAWMOD" / "SMfemvolume_reduced_xy.msh"
    canonical = inspect_system_meshes(
        (MeshDialogEntry(name="SAWMOD", source_file=str(full_path), scale_factor=0.001),)
    )[0]
    analysis = inspect_system_meshes(
        (MeshDialogEntry(name="SAWMOD", source_file=str(reduced_path), scale_factor=0.001),)
    )[0]
    dialog = SystemConfigDialog(
        (canonical,),
        None,
        ("main",),
        symmetry_mode="xy",
        symmetry_analysis_meshes=(analysis,),
    )
    resource = MeshResource(
        id="mesh:sawmod",
        name="SAWMOD",
        file=str(full_path),
        purpose=MeshPurpose.FEM_VOLUME,
        scale_to_m=0.001,
    )

    resolved = dialog._symmetry_analysis_resources_by_id((resource,))

    assert resolved[resource.id].file == str(reduced_path)
    assert resource.file == str(full_path)


def test_boundary_assignments_default_to_rigid_without_unused_options() -> None:
    dialog = SystemConfigDialog(
        inspect_system_meshes(_fixture_mesh_entries()),
        None,
        ("main",),
    )

    assert dialog.boundaries_table.rowCount() > 0
    for row in range(dialog.boundaries_table.rowCount()):
        combo = dialog.boundaries_table.cellWidget(row, 3)
        assert isinstance(combo, QComboBox)
        assert [combo.itemText(index) for index in range(combo.count())] == [
            "Rigid",
            "Moving",
            "Interface",
        ]
        assert combo.currentData() == BoundaryKind.RIGID


def test_interfaces_tab_is_disabled_until_a_bounded_region_exists() -> None:
    dialog = SystemConfigDialog(
        inspect_system_meshes((_fixture_mesh_entries()[1],)),
        None,
        ("main",),
    )

    assert not dialog.tabs.isTabEnabled(dialog.tabs.indexOf(dialog.interfaces_tab))

    dialog = SystemConfigDialog(
        inspect_system_meshes(_fixture_mesh_entries()),
        None,
        ("main",),
    )
    dialog._add_default_region()

    assert dialog.tabs.isTabEnabled(dialog.tabs.indexOf(dialog.interfaces_tab))


def test_exterior_region_can_own_multiple_bem_mesh_resources() -> None:
    _fem, bem = inspect_system_meshes(_fixture_mesh_entries())
    second = replace(bem, name="Exterior B")
    dialog = SystemConfigDialog((bem, second), None, ("main",))

    system = dialog.physical_system()

    assert len(system.regions) == 1
    assert len(system.regions[0].mesh_ids) == 2
    assert infer_physical_solve_kind(system) == PhysicalSolveKind.EXTERIOR_BEM
    PhysicalSystemCompiler().compile(system)
    restored = SystemConfigDialog((bem, second), system, ("main",))
    assert len(restored.physical_system().regions[0].mesh_ids) == 2


def test_seeded_exterior_system_preserves_ath_style_velocity_offset() -> None:
    _fem, bem = inspect_system_meshes(_fixture_mesh_entries())
    tags = read_surface_physical_names(Path(bem.file))
    group_name, tag = next(iter(tags.items()))

    system, channels = seed_exterior_system(
        (bem,),
        (
            RadiatorConfig(
                name=f"{bem.name}:{group_name}",
                mesh=bem.name,
                tag=tag,
                channel="High",
                velocity_offset_db=-6.0,
            ),
        ),
    )
    inputs = exterior_bem_inputs(system, component_channel_by_id=channels)

    assert infer_physical_solve_kind(system) == PhysicalSolveKind.EXTERIOR_BEM
    assert len(system.components) == 1
    boundary_id = system.components[0].boundary_ids[0]
    assert system.components[0].parameters["boundary_motion_weights"][boundary_id] == pytest.approx(
        10.0 ** (-6.0 / 20.0)
    )
    assert inputs.radiators[0].channel == "High"
    assert inputs.radiators[0].velocity_offset_db == pytest.approx(-6.0)

    unsupported = replace(
        system,
        components=(
            replace(system.components[0], kind=ComponentKind.ELECTRODYNAMIC_TRANSDUCER),
        ),
    )
    with pytest.raises(ValueError, match="prescribed-velocity components only"):
        exterior_bem_inputs(unsupported, component_channel_by_id=channels)


def test_saved_unused_boundary_is_presented_and_collected_as_rigid() -> None:
    system = _configured_fixture_dialog().physical_system()
    rigid = next(boundary for boundary in system.boundaries if boundary.kind == BoundaryKind.RIGID)
    migrated = replace(
        system,
        boundaries=tuple(
            replace(boundary, kind=BoundaryKind.UNUSED) if boundary.id == rigid.id else boundary
            for boundary in system.boundaries
        ),
    )
    dialog = SystemConfigDialog(
        inspect_system_meshes(_fixture_mesh_entries()),
        migrated,
        ("main",),
    )

    restored = next(boundary for boundary in dialog._collect_boundaries() if boundary.id == rigid.id)
    assert restored.kind == BoundaryKind.RIGID


def test_mesh_inventory_prefers_persisted_derived_mesh_file() -> None:
    entries = list(_fixture_mesh_entries("exterior.msh"))
    entries[1] = replace(
        entries[1],
        cleaned_file=str(FIXTURE_ROOT / "exterior_conforming.msh"),
    )

    _fem, bem = inspect_system_meshes(tuple(entries))

    assert bem.source_file == str(FIXTURE_ROOT / "exterior.msh")
    assert bem.file == str(FIXTURE_ROOT / "exterior_conforming.msh")


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


def test_motion_axis_inference_does_not_cancel_opposed_surface_normals() -> None:
    resources = {
        "mesh:front": MeshResource(
            id="mesh:front",
            name="Front",
            file="unused-front.msh",
            purpose=MeshPurpose.FEM_VOLUME,
        ),
        "mesh:rear": MeshResource(
            id="mesh:rear",
            name="Rear",
            file="unused-rear.msh",
            purpose=MeshPurpose.FEM_VOLUME,
        ),
    }
    boundaries = (
        Boundary(
            id="boundary:front",
            name="Front",
            region_id="region:front",
            group=PhysicalGroupRef(mesh_id="mesh:front", dimension=2, name="Front"),
            kind=BoundaryKind.MOVING,
        ),
        Boundary(
            id="boundary:rear",
            name="Rear",
            region_id="region:rear",
            group=PhysicalGroupRef(mesh_id="mesh:rear", dimension=2, name="Rear"),
            kind=BoundaryKind.MOVING,
        ),
    )

    inferred = infer_component_motion_axis(
        boundaries,
        resources,
        mesh_cache={
            "mesh:front": _planar_surface_mesh(group_name="Front", z_m=0.0, reverse=False),
            "mesh:rear": _planar_surface_mesh(group_name="Rear", z_m=0.01, reverse=True),
        },
    )

    assert np.abs(inferred.axis) == pytest.approx((0.0, 0.0, 1.0))
    assert inferred.confidence == pytest.approx(1.0)
    assert inferred.mean_squared_alignment == pytest.approx(1.0)
    assert inferred.boundary_alignment == pytest.approx(1.0)
    assert inferred.triangle_count == 2


def test_motion_axis_inference_completes_a_curved_driver_across_its_symmetry_plane() -> None:
    resource = MeshResource(
        id="mesh:curved-fem",
        name="Curved FEM",
        file=str(FIXTURE_ROOT / "curvedinterfaceFEM.msh"),
        purpose=MeshPurpose.FEM_VOLUME,
        scale_to_m=0.001,
    )
    boundary = Boundary(
        id="boundary:mf",
        name="MF",
        region_id="region:interior",
        group=PhysicalGroupRef(mesh_id=resource.id, dimension=2, name="MF"),
        kind=BoundaryKind.MOVING,
    )

    reduced = infer_component_motion_axis((boundary,), {resource.id: resource})
    completed = infer_component_motion_axis(
        (boundary,),
        {resource.id: resource},
        fractional_symmetry_axes=("y",),
    )

    assert abs(reduced.axis[1]) > 0.2
    assert completed.axis == pytest.approx((0.913545, 0.0, -0.406737), abs=1e-6)
    assert completed.confidence > 0.8


def test_component_editor_infers_symmetry_and_completes_motion_axis() -> None:
    resource = MeshResource(
        id="mesh:curved-fem",
        name="Curved FEM",
        file=str(FIXTURE_ROOT / "curvedinterfaceFEM.msh"),
        purpose=MeshPurpose.FEM_VOLUME,
        scale_to_m=0.001,
    )
    boundary = Boundary(
        id="boundary:mf",
        name="MF",
        region_id="region:interior",
        group=PhysicalGroupRef(mesh_id=resource.id, dimension=2, name="MF"),
        kind=BoundaryKind.MOVING,
    )
    editor = _ComponentEditorDialog(
        _ComponentDraft(
            id="component:mid",
            name="Midrange",
            kind=ComponentKind.ELECTRODYNAMIC_TRANSDUCER,
            boundary_ids=(boundary.id,),
            channel="main",
            parameters={
                "re_ohm": 6.0,
                "le_h": 0.0005,
                "bl_n_per_a": 7.0,
                "mmd_kg": 0.015,
                "cms_m_per_n": 0.0005,
                "rms_n_s_per_m": 1.0,
                "motion_axis": [1.0, 0.0, 0.0],
                "symmetry_role": "complete_representative",
                "surface_completion_factor": 1,
                "physical_driver_orbit_count": 4,
                "fractional_symmetry_axes": [],
            },
            motion_axis_mode="automatic",
        ),
        boundaries=(boundary,),
        resources_by_id={resource.id: resource},
        region_names={"region:interior": "Interior"},
        channel_names=("main",),
        unavailable_boundary_ids=set(),
        symmetry_mode="xy",
        mesh_cache={},
    )
    updated = editor.component_draft()

    assert not hasattr(editor, "symmetry_combo")
    assert editor.symmetry_inference_label.text() == (
        "Moving surface(s) sliced along the y axis. "
        "Detected 2 distinct components in the fully mirrored system."
    )
    assert all(spin.decimals() == 3 for spin in editor.axis_spins)
    assert all(spin.singleStep() == pytest.approx(0.005) for spin in editor.axis_spins)
    assert np.abs(updated.parameters["motion_axis"]) == pytest.approx(
        (0.913545, 0.0, 0.406737),
        abs=1e-6,
    )
    assert updated.parameters["fractional_symmetry_axes"] == ["y"]
    assert updated.parameters["surface_completion_factor"] == 2
    assert updated.parameters["physical_driver_orbit_count"] == 2


def test_component_editor_persists_per_surface_velocity_weights() -> None:
    resource = MeshResource(
        id="mesh:surface",
        name="Surface",
        file="unused.msh",
        purpose=MeshPurpose.BEM_SURFACE,
    )
    boundary = Boundary(
        id="boundary:surround",
        name="Surround",
        region_id="region:exterior",
        group=PhysicalGroupRef(mesh_id=resource.id, dimension=2, name="Surround"),
        kind=BoundaryKind.MOVING,
    )
    editor = _ComponentEditorDialog(
        _ComponentDraft(
            id="component:tweeter",
            name="Tweeter",
            kind=ComponentKind.IDEAL_VELOCITY_SOURCE,
            boundary_ids=(boundary.id,),
            channel="main",
            parameters={"motion_profile": "uniform"},
        ),
        boundaries=(boundary,),
        resources_by_id={resource.id: resource},
        region_names={"region:exterior": "Exterior"},
        channel_names=("main",),
        unavailable_boundary_ids=set(),
        symmetry_mode="off",
        mesh_cache={},
    )

    editor.boundary_weight_spins[0].setValue(-12.0)
    updated = editor.component_draft()

    assert updated.parameters["boundary_motion_weights"][boundary.id] == pytest.approx(
        10.0 ** (-12.0 / 20.0)
    )


def test_component_editor_applies_automatic_axis_to_a_two_sided_transducer() -> None:
    resources = {
        "mesh:front": MeshResource(
            id="mesh:front",
            name="Front",
            file="unused-front.msh",
            purpose=MeshPurpose.FEM_VOLUME,
        ),
        "mesh:rear": MeshResource(
            id="mesh:rear",
            name="Rear",
            file="unused-rear.msh",
            purpose=MeshPurpose.FEM_VOLUME,
        ),
    }
    boundaries = (
        Boundary(
            id="boundary:front",
            name="Front",
            region_id="region:front",
            group=PhysicalGroupRef(mesh_id="mesh:front", dimension=2, name="Front"),
            kind=BoundaryKind.MOVING,
        ),
        Boundary(
            id="boundary:rear",
            name="Rear",
            region_id="region:rear",
            group=PhysicalGroupRef(mesh_id="mesh:rear", dimension=2, name="Rear"),
            kind=BoundaryKind.MOVING,
        ),
    )
    parameters = {
        "re_ohm": 6.0,
        "le_h": 0.0005,
        "bl_n_per_a": 7.0,
        "mmd_kg": 0.015,
        "cms_m_per_n": 0.0005,
        "rms_n_s_per_m": 1.0,
        "motion_axis": [1.0, 0.0, 0.0],
    }
    editor = _ComponentEditorDialog(
        _ComponentDraft(
            id="component:woofer",
            name="Woofer",
            kind=ComponentKind.ELECTRODYNAMIC_TRANSDUCER,
            boundary_ids=("boundary:front", "boundary:rear"),
            channel="main",
            parameters=parameters,
            motion_axis_mode="automatic",
        ),
        boundaries=boundaries,
        resources_by_id=resources,
        region_names={"region:front": "Front chamber", "region:rear": "Rear chamber"},
        channel_names=("main",),
        unavailable_boundary_ids=set(),
        symmetry_mode="off",
        mesh_cache={
            "mesh:front": _planar_surface_mesh(group_name="Front", z_m=0.0, reverse=False),
            "mesh:rear": _planar_surface_mesh(group_name="Rear", z_m=0.01, reverse=True),
        },
    )

    assert float(editor.parameter_edits["le_h"].text()) == pytest.approx(0.5)
    assert float(editor.parameter_edits["mmd_kg"].text()) == pytest.approx(15.0)
    assert float(editor.parameter_edits["cms_m_per_n"].text()) == pytest.approx(500.0)
    editor.parameter_edits["le_h"].setText("0.75")
    editor.parameter_edits["mmd_kg"].setText("18")
    editor.parameter_edits["cms_m_per_n"].setText("625")
    updated = editor.component_draft()

    assert updated.boundary_ids == ("boundary:front", "boundary:rear")
    assert np.abs(updated.parameters["motion_axis"]) == pytest.approx((0.0, 0.0, 1.0))
    assert updated.parameters["le_h"] == pytest.approx(0.00075)
    assert updated.parameters["mmd_kg"] == pytest.approx(0.018)
    assert updated.parameters["cms_m_per_n"] == pytest.approx(0.000625)
    assert updated.motion_axis_mode == "automatic"
    assert "High confidence" in editor.axis_confidence_label.text()


def test_components_tab_round_trips_multiple_moving_boundaries() -> None:
    dialog = _configured_fixture_dialog()
    for row in range(dialog.boundaries_table.rowCount()):
        mesh_name = dialog.boundaries_table.item(row, 1).text()
        group_name = dialog.boundaries_table.item(row, 2).text()
        if (mesh_name, group_name) != ("Interior", "Volume_boundary"):
            continue
        combo = dialog.boundaries_table.cellWidget(row, 3)
        assert isinstance(combo, QComboBox)
        combo.setCurrentIndex(combo.findData(BoundaryKind.MOVING))
    moving_ids = tuple(boundary.id for boundary in dialog._moving_boundaries())
    dialog._component_drafts[0].boundary_ids = moving_ids
    dialog._render_components_table()

    system = dialog.physical_system()
    restored = SystemConfigDialog(
        inspect_system_meshes(_fixture_mesh_entries()),
        system,
        ("main",),
        component_channel_by_id={system.components[0].id: "main"},
    )

    assert system.components[0].boundary_ids == moving_ids
    assert restored._component_drafts[0].boundary_ids == moving_ids
    assert "Radiator" in restored.components_table.item(0, 2).text()
    assert "Volume_boundary" in restored.components_table.item(0, 2).text()


def test_electrodynamic_component_collection_uses_voltage_port_and_preserves_auto_axis_mode() -> None:
    dialog = _configured_fixture_dialog()
    (moving_boundary,) = dialog._moving_boundaries()
    dialog._component_drafts.clear()
    dialog._append_component_draft(
        name="Woofer",
        boundary_ids=(moving_boundary.id,),
        channel="main",
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
            "symmetry_role": "complete_representative",
            "surface_completion_factor": 1,
            "physical_driver_orbit_count": 1,
            "fractional_symmetry_axes": [],
        },
        motion_axis_mode="automatic",
    )

    system = dialog.physical_system()

    assert system.components[0].kind == ComponentKind.ELECTRODYNAMIC_TRANSDUCER
    assert system.components[0].parameters["mmd_kg"] == pytest.approx(0.015)
    assert system.excitation_ports[0].kind.value == "voltage"
    assert system.metadata["component_editor"][system.components[0].id]["motion_axis_mode"] == "automatic"
    PhysicalSystemCompiler().compile(system)
    prepared = prepare_coupled_ui_solve(
        system,
        freq_min_hz=500.0,
        freq_max_hz=500.0,
        freq_count=1,
        observation_distance_m=2.0,
        polar_angle_step_deg=90.0,
    )
    session = CoupledProductionBackend(bem_backend="cpu", persistent_worker=False).create_system_session(
        prepared.request
    )
    assert session.request.solver_options["transducer_reference_voltage_v"] == pytest.approx(2.83)


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
    assert dialog.identify_interfaces_button.text() == "Build/Identify Interfaces"
    assert dialog.interfaces_table.item(0, 3).text() == "Ready"
    assert dialog.configuration().mesh_file_overrides_by_name == {}
    assert "channel" not in system.components[0].parameters
    assert dialog.configuration().component_channel_by_id == {system.components[0].id: "main"}


def test_identify_interfaces_reuses_each_transformed_mesh(monkeypatch) -> None:
    dialog = _configured_fixture_dialog()
    dialog._interface_mesh_cache.clear()
    transformed_resources = []
    original = system_config_module._transformed_mesh

    def tracked(resource):
        transformed_resources.append(resource.id)
        return original(resource)

    monkeypatch.setattr(system_config_module, "_transformed_mesh", tracked)

    dialog._identify_interfaces()

    assert sorted(transformed_resources) == ["mesh:exterior", "mesh:interior"]


def test_saved_regions_rebind_to_unique_compatible_renamed_meshes() -> None:
    system = _configured_fixture_dialog().physical_system()
    fem, bem = inspect_system_meshes(_fixture_mesh_entries())
    renamed_meshes = (
        replace(fem, name="Replacement Interior"),
        replace(bem, name="Replacement Exterior"),
    )

    dialog = SystemConfigDialog(renamed_meshes, system, ("main",))

    exterior_mesh = dialog.regions_table.cellWidget(0, 2)
    interior_mesh = dialog.regions_table.cellWidget(1, 2)
    interior_volume = dialog.regions_table.cellWidget(1, 3)
    assert isinstance(exterior_mesh, QComboBox)
    assert isinstance(interior_mesh, QComboBox)
    assert isinstance(interior_volume, QComboBox)
    assert exterior_mesh.currentData() == "Replacement Exterior"
    assert interior_mesh.currentData() == "Replacement Interior"
    assert interior_volume.currentData() == "Volume"
    assert dialog.interfaces_table.rowCount() == 1
    restored = dialog.physical_system()
    assert {mesh.id for mesh in restored.meshes} == {mesh.id for mesh in system.meshes}
    assert {mesh.name for mesh in restored.meshes} == {"Replacement Exterior", "Replacement Interior"}


def test_system_dialog_opens_with_an_incomplete_saved_region_selection() -> None:
    system = _configured_fixture_dialog().physical_system()
    fem, _bem = inspect_system_meshes(_fixture_mesh_entries())

    dialog = SystemConfigDialog((fem,), system, ("main",))

    exterior_mesh = dialog.regions_table.cellWidget(0, 2)
    assert isinstance(exterior_mesh, QComboBox)
    assert exterior_mesh.currentData() is None
    with pytest.raises(ValueError, match="Region 'Exterior Air' must select a mesh"):
        dialog._region_drafts()


def test_build_identify_interfaces_writes_and_uses_a_conformed_bem_asset(tmp_path: Path) -> None:
    dialog = _configured_fixture_dialog(
        bem_filename="exterior.msh",
        interface_output_root=tmp_path,
    )

    configuration = dialog.configuration()
    rebuilt_path = Path(configuration.mesh_file_overrides_by_name["Exterior"])

    assert rebuilt_path.is_file()
    assert rebuilt_path.parent == tmp_path
    exterior_resource = next(mesh for mesh in configuration.system.meshes if mesh.name == "Exterior")
    assert exterior_resource.file == str(rebuilt_path)
    assert dialog.interfaces_table.item(0, 3).text() == "Built"
    PhysicalSystemCompiler().compile(configuration.system)
    validate_conforming_interfaces(
        meshio.read(FIXTURE_ROOT / "femvolume.msh"),
        meshio.read(rebuilt_path),
    )
    with pytest.raises(InterfaceConformError):
        validate_conforming_interfaces(
            meshio.read(FIXTURE_ROOT / "femvolume.msh"),
            meshio.read(FIXTURE_ROOT / "exterior.msh"),
        )


def test_configured_interface_dependencies_identify_the_bem_mesh_to_rebuild(tmp_path: Path) -> None:
    system = _configured_fixture_dialog(
        bem_filename="exterior.msh",
        interface_output_root=tmp_path / "initial",
    ).physical_system()

    assert interface_bem_mesh_names_for_changes(system, {"Interior"}) == ("Exterior",)
    assert interface_bem_mesh_names_for_changes(system, {"Exterior"}) == ("Exterior",)
    assert interface_bem_mesh_names_for_changes(system, {"Unrelated"}) == ()


def test_known_interfaces_can_be_rebuilt_headlessly_after_a_mesh_change(tmp_path: Path) -> None:
    system = _configured_fixture_dialog(
        bem_filename="exterior.msh",
        interface_output_root=tmp_path / "initial",
    ).physical_system()
    available_meshes = inspect_system_meshes(_fixture_mesh_entries("exterior.msh"))

    result = rebuild_configured_interfaces(
        system,
        available_meshes,
        changed_mesh_names={"Interior"},
        interface_output_root=tmp_path / "refreshed",
    )

    rebuilt_path = Path(result.mesh_file_overrides_by_name["Exterior"])
    assert result.rebuilt_interface_ids == (system.interfaces[0].id,)
    assert rebuilt_path.is_file()
    rebuilt_resource = next(mesh for mesh in result.system.meshes if mesh.name == "Exterior")
    assert rebuilt_resource.file == str(rebuilt_path)
    validate_conforming_interfaces(
        meshio.read(FIXTURE_ROOT / "femvolume.msh"),
        meshio.read(rebuilt_path),
    )


def test_coupled_ui_request_uses_excitation_basis_and_polar_field_points() -> None:
    system = _configured_fixture_dialog().physical_system()
    system = replace(
        system,
        regions=tuple(
            replace(region, loss_model={"bulk_loss_factor": 0.01})
            if region.kind == AcousticRegionKind.BOUNDED_AIR
            else region
            for region in system.regions
        ),
    )

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
    assert prepared.request.solver_options["static_condensation"] is False
    assert prepared.request.solver_options["symmetry"] == "off"
    interior = next(
        region
        for region in prepared.request.compiled_system.regions
        if region.kind == AcousticRegionKind.BOUNDED_AIR
    )
    assert interior.loss_model["bulk_loss_factor"] == pytest.approx(0.01)
    assert "precision" not in prepared.request.solver_options
    assert "bem_backend" not in prepared.request.solver_options
    assert prepared.backend_id == "beat_cpu"
    points = np.asarray(prepared.request.outputs[0].options["points_m"])
    assert points.shape == (10, 3)


def test_system_dialog_edits_region_loss_and_rigid_wall_impedance() -> None:
    dialog = _configured_fixture_dialog()
    bounded_row = next(
        row
        for row in range(dialog.regions_table.rowCount())
        if dialog._region_kind(row) == AcousticRegionKind.BOUNDED_AIR
    )
    loss_combo = dialog.regions_table.cellWidget(bounded_row, 4)
    assert isinstance(loss_combo, QComboBox)
    assert loss_combo.isEnabled()
    assert [loss_combo.itemData(index) for index in range(loss_combo.count())] == [
        0.0,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
    ]
    loss_combo.setCurrentIndex(loss_combo.findData(0.02))
    unbounded_row = next(
        row
        for row in range(dialog.regions_table.rowCount())
        if dialog._region_kind(row) == AcousticRegionKind.UNBOUNDED_AIR
    )
    unbounded_loss_combo = dialog.regions_table.cellWidget(unbounded_row, 4)
    assert isinstance(unbounded_loss_combo, QComboBox)
    assert not unbounded_loss_combo.isEnabled()

    wall_row = next(
        row
        for row in range(dialog.boundaries_table.rowCount())
        if dialog.boundaries_table.item(row, 1).text() == "Interior"
        and dialog.boundaries_table.item(row, 2).text() == "Volume_boundary"
    )
    impedance_button = dialog.boundaries_table.cellWidget(wall_row, 4)
    assignment_combo = dialog.boundaries_table.cellWidget(wall_row, 3)
    assert isinstance(impedance_button, QPushButton)
    assert isinstance(assignment_combo, QComboBox)
    assert impedance_button.isEnabled()
    assignment_combo.blockSignals(True)
    assignment_combo.setCurrentIndex(assignment_combo.findData(BoundaryKind.MOVING))
    dialog._refresh_wall_impedance_button(impedance_button, assignment_combo, True)
    assert not impedance_button.isEnabled()
    assignment_combo.setCurrentIndex(assignment_combo.findData(BoundaryKind.RIGID))
    dialog._refresh_wall_impedance_button(impedance_button, assignment_combo, True)
    assignment_combo.blockSignals(False)
    assert impedance_button.isEnabled()
    impedance_button.setProperty("boundary_parameters", miki_wall_impedance_parameters())

    editor = _WallImpedanceDialog({})
    assert editor.thickness_spin.value() == pytest.approx(30.0)
    assert editor.flow_resistivity_spin.value() == pytest.approx(5000.0)

    system = dialog.physical_system()
    interior = next(region for region in system.regions if region.kind == AcousticRegionKind.BOUNDED_AIR)
    wall = next(boundary for boundary in system.boundaries if boundary.group.name == "Volume_boundary")
    assert interior.loss_model["bulk_loss_factor"] == pytest.approx(0.02)
    assert wall.parameters["wall_impedance"]["thickness_m"] == pytest.approx(0.03)
    assert wall.parameters["wall_impedance"]["flow_resistivity_pa_s_per_m2"] == pytest.approx(5000.0)


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


def test_coupled_worker_logs_backend_detail_without_emitting_visible_status(monkeypatch, caplog) -> None:
    system = _configured_fixture_dialog().physical_system()
    prepared = prepare_coupled_ui_solve(
        system,
        freq_min_hz=500.0,
        freq_max_hz=500.0,
        freq_count=1,
        observation_distance_m=2.0,
        polar_angle_step_deg=90.0,
    )

    class Session:
        def __init__(self, request):
            self.request = request

        def solve_stream(self, *, stop_requested=None):
            del stop_requested
            self.request.status_callback("assembling coupled backend detail")
            return iter(())

        def stop(self) -> None:
            pass

    class Backend:
        def __init__(self, **_kwargs):
            pass

        def create_system_session(self, request):
            request.status_callback("initializing coupled backend detail")
            return Session(request)

    monkeypatch.setattr(system_solve_module, "CoupledProductionBackend", Backend)
    worker = CoupledSolveWorker(prepared)
    statuses = []
    worker.status.connect(statuses.append)

    with caplog.at_level("INFO", logger="blab.ui.system_solve"):
        worker.run()

    assert statuses == []
    assert "initializing coupled backend detail" in caplog.text
    assert "assembling coupled backend detail" in caplog.text


@pytest.mark.skipif(
    os.environ.get("BLAB_RUN_COUPLED_REFERENCE") != "1",
    reason="Set BLAB_RUN_COUPLED_REFERENCE=1 to run the Julia GUI-path integration.",
)
def test_coupled_ui_request_returns_live_plot_pressure_basis() -> None:
    system = _configured_fixture_dialog().physical_system()
    system = replace(
        system,
        regions=tuple(
            replace(region, loss_model={"bulk_loss_factor": 0.01})
            if region.kind == AcousticRegionKind.BOUNDED_AIR
            else region
            for region in system.regions
        ),
    )
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
    interior_id = next(region.id for region in system.regions if region.kind == AcousticRegionKind.BOUNDED_AIR)
    assert system_result.diagnostics["fem_bulk_loss_factors_by_region"][interior_id] == pytest.approx(0.01)
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
    assert prepared.request.solver_options["static_condensation"] is True
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
    assert cpu_result.diagnostics["linear_backend"] == "cpu"
    assert cuda_result.diagnostics["linear_backend"] == "cuda"
    assert cpu_pressure.dtype == np.complex64
    assert cuda_pressure.dtype == np.complex64
    assert relative_l2 < 5e-3
