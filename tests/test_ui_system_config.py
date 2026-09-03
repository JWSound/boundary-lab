import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import meshio
import numpy as np
import pytest
from PySide6.QtCore import QLocale
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QComboBox, QPushButton

import blab.ui.system_config as system_config_module
import blab.ui.system_solve as system_solve_module
from blab.acoustic_materials import miki_wall_impedance_parameters
from blab.ath import read_surface_physical_names
from blab.config import RadiatorConfig
from blab.interface_conform import InterfaceConformError, validate_conforming_interfaces
from blab.observation_planes import ObservationPlaneType, new_observation_plane
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
from blab.solve_results import (
    BEM_BOUNDARY_DOMAIN_ID,
    BEM_BOUNDARY_NEUMANN_ID,
    BEM_BOUNDARY_PRESSURE_ID,
    FEM_NODAL_PRESSURE_ID,
    FEM_VOLUME_DOMAIN_ID,
    RADIATION_IMPEDANCE_ID,
    RADIATOR_DOMAIN_ID,
)
from blab.solvers.coupled_backend import CoupledProductionBackend
from blab.system_contract import QuantityResult, SystemFrequencyResult
from blab.ui.dialogs import MeshDialogEntry
from blab.ui.exterior_system import exterior_bem_inputs
from blab.ui.main_window.radiators import RadiatorsMixin
from blab.ui.mesh_assembly import MeshAssemblyService
from blab.ui.physical_system_migration import PhysicalSystemMigrationError, seed_exterior_system
from blab.ui.project_state import ImportedMeshState, new_project_document
from blab.ui.system_config import (
    SystemConfigDialog,
    _ComponentDraft,
    _ComponentEditorDialog,
    _SemiInductanceDialog,
    _WallImpedanceDialog,
    infer_component_motion_axis,
    inspect_system_mesh_variants,
    inspect_system_meshes,
    interface_bem_mesh_names_for_changes,
    rebuild_configured_interfaces,
)
from blab.ui.system_solve import CoupledSolveWorker, prepare_coupled_ui_solve

_APP = QApplication.instance() or QApplication([])
FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"


def test_identical_system_mesh_variants_are_only_inspected_once(monkeypatch) -> None:
    entries = _fixture_mesh_entries()
    inspected = (object(),)
    calls = []

    def inspect(value):
        calls.append(value)
        return inspected

    monkeypatch.setattr(system_config_module, "inspect_system_meshes", inspect)

    canonical, symmetry = inspect_system_mesh_variants(entries, entries)

    assert calls == [entries]
    assert canonical is inspected
    assert symmetry is inspected


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
    scale: float = 1.0,
) -> meshio.Mesh:
    triangle = [0, 2, 1] if reverse else [0, 1, 2]
    return meshio.Mesh(
        points=np.asarray(
            [
                [0.0, 0.0, z_m],
                [scale, 0.0, z_m],
                [0.0, scale, z_m],
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


def test_interior_only_editor_offers_tube_termination_without_enabling_interfaces() -> None:
    dialog = SystemConfigDialog(
        inspect_system_meshes((_fixture_mesh_entries()[0],)),
        None,
        ("main",),
    )
    dialog._refresh_boundaries()

    assert [dialog._region_kind(row) for row in range(dialog.regions_table.rowCount())] == [
        AcousticRegionKind.BOUNDED_AIR
    ]
    assert not dialog.tabs.isTabEnabled(dialog.tabs.indexOf(dialog.interfaces_tab))
    assert dialog.boundaries_table.rowCount() > 0
    for row in range(dialog.boundaries_table.rowCount()):
        combo = dialog.boundaries_table.cellWidget(row, 3)
        assert isinstance(combo, QComboBox)
        assert combo.findData(BoundaryKind.PLANE_WAVE_TUBE_TERMINATION) >= 0


def test_exterior_region_can_own_multiple_bem_mesh_resources() -> None:
    _fem, bem = inspect_system_meshes(_fixture_mesh_entries())
    second = replace(bem, name="Exterior B")
    dialog = SystemConfigDialog((bem, second), None, ("main",))

    system = dialog.physical_system()

    assert len(system.regions) == 1
    assert all(type(region.kind) is AcousticRegionKind for region in system.regions)
    assert all(type(boundary.kind) is BoundaryKind for boundary in system.boundaries)
    assert len(system.regions[0].mesh_ids) == 2
    assert infer_physical_solve_kind(system) == PhysicalSolveKind.EXTERIOR_BEM
    PhysicalSystemCompiler().compile(system)
    restored = SystemConfigDialog((bem, second), system, ("main",))
    assert len(restored.physical_system().regions[0].mesh_ids) == 2


def test_required_exterior_migration_reports_why_it_cannot_build_a_system() -> None:
    class MigrationHost(RadiatorsMixin):
        project = new_project_document()

        @staticmethod
        def _mesh_config_dialog_entries():
            return ()

    host = MigrationHost()

    assert host.ensure_seeded_exterior_system() is False
    with pytest.raises(PhysicalSystemMigrationError, match="No exterior surface mesh is available"):
        host.ensure_seeded_exterior_system(required=True)


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
        components=(replace(system.components[0], kind=ComponentKind.ELECTRODYNAMIC_TRANSDUCER),),
    )
    with pytest.raises(ValueError, match="prescribed-velocity components only"):
        exterior_bem_inputs(unsupported, component_channel_by_id=channels)


def test_seeded_exterior_system_groups_ath_driver_surfaces_into_one_component(tmp_path: Path) -> None:
    mesh_path = tmp_path / "ath_two_way.msh"
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [4.0, 1.0, 0.0],
            [6.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [6.0, 1.0, 0.0],
        ]
    )
    meshio.write(
        mesh_path,
        meshio.Mesh(
            points=points,
            cells=[("triangle", np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))],
            cell_data={"gmsh:physical": [np.array([2, 3, 4, 5], dtype=np.int32)]},
            field_data={
                "SD1D1001": np.array([2, 2], dtype=np.int32),
                "SD1D1002": np.array([3, 2], dtype=np.int32),
                "SD1D1003": np.array([4, 2], dtype=np.int32),
                "SD1D1004": np.array([5, 2], dtype=np.int32),
            },
        ),
        file_format="gmsh22",
        binary=False,
    )
    mesh = inspect_system_meshes((MeshDialogEntry(name="2way", source_file=str(mesh_path), scale_factor=0.001),))[0]
    radiators = tuple(
        RadiatorConfig(
            name=f"2way:{surface}",
            mesh="2way",
            tag=tag,
            drive_group=drive_group,
            drive_group_name=drive_name,
            channel=channel,
            velocity_offset_db=offset_db,
        )
        for surface, tag, drive_group, drive_name, channel, offset_db in (
            ("SD1D1001", 2, "ath:0", "horn_driver", "High", -12.042),
            ("SD1D1002", 3, "ath:0", "horn_driver", "High", -2.499),
            ("SD1D1003", 4, "ath:0", "horn_driver", "High", 0.0),
            ("SD1D1004", 5, "ath:2", "woofer_B", "Low", 0.0),
        )
    )

    system, channels = seed_exterior_system((mesh,), radiators)

    assert [(component.name, len(component.boundary_ids)) for component in system.components] == [
        ("horn_driver", 3),
        ("woofer_B", 1),
    ]
    tweeter = system.components[0]
    boundary_names = {boundary.id: boundary.name for boundary in system.boundaries}
    weights_by_surface = {
        boundary_names[boundary_id]: weight
        for boundary_id, weight in tweeter.parameters["boundary_motion_weights"].items()
    }
    assert weights_by_surface == pytest.approx(
        {
            "SD1D1001": 10.0 ** (-12.042 / 20.0),
            "SD1D1002": 10.0 ** (-2.499 / 20.0),
            "SD1D1003": 1.0,
        }
    )
    assert channels[tweeter.id] == "High"

    inputs = exterior_bem_inputs(system, component_channel_by_id=channels)
    assert [(radiator.tag, radiator.channel, radiator.velocity_offset_db) for radiator in inputs.radiators] == [
        (2, "High", pytest.approx(-12.042)),
        (3, "High", pytest.approx(-2.499)),
        (4, "High", pytest.approx(0.0)),
        (5, "Low", pytest.approx(0.0)),
    ]

    ungrouped, _channels = seed_exterior_system(
        (mesh,),
        tuple(replace(radiator, drive_group=None, drive_group_name=None) for radiator in radiators),
    )
    assert len(ungrouped.components) == 4


def test_exterior_system_ui_request_uses_canonical_bem_outputs() -> None:
    _fem, bem = inspect_system_meshes(_fixture_mesh_entries())
    tags = read_surface_physical_names(Path(bem.file))
    group_name, tag = next(iter(tags.items()))
    system, channels = seed_exterior_system(
        (bem,),
        (RadiatorConfig(name=f"{bem.name}:{group_name}", mesh=bem.name, tag=tag, channel="High"),),
    )
    exterior_plane = replace(
        new_observation_plane("Exterior Field"),
        plane_type=ObservationPlaneType.EXTERIOR,
    )

    prepared = system_solve_module.prepare_system_ui_solve(
        system,
        freq_min_hz=500.0,
        freq_max_hz=1000.0,
        freq_count=3,
        observation_distance_m=2.0,
        polar_angle_step_deg=90.0,
        component_channel_by_id=channels,
        backend_id="beat_cpu",
        observation_planes=(exterior_plane,),
    )

    outputs = {output.id: output for output in prepared.request.outputs}
    domains = {domain.id: domain for domain in prepared.result_domains}
    assert prepared.solve_kind == PhysicalSolveKind.EXTERIOR_BEM
    assert prepared.excitation_channel_names.tolist() == ["High"]
    assert prepared.request.solver_options["quadrature_order"] == 4
    assert prepared.request.solver_options["singular_order"] == 4
    assert prepared.request.solver_options["static_condensation"] is False
    assert outputs[RADIATION_IMPEDANCE_ID].quantity == "radiation_impedance"
    assert {BEM_BOUNDARY_PRESSURE_ID, BEM_BOUNDARY_NEUMANN_ID} <= outputs.keys()
    assert {RADIATOR_DOMAIN_ID, BEM_BOUNDARY_DOMAIN_ID} <= domains.keys()
    radiator_domain = domains[RADIATOR_DOMAIN_ID]
    assert radiator_domain.coordinates["effective_area_m2"].shape == (1,)
    assert radiator_domain.coordinates["effective_area_m2"][0] > 0.0
    assert np.isnan(radiator_domain.coordinates["relative_side_mismatch"][0])
    assert system_solve_module.supports_exterior_system_protocol(
        system,
        backend_id="beat_cpu",
        stitch_exterior_meshes=False,
    )
    assert not system_solve_module.supports_exterior_system_protocol(
        system,
        backend_id="beat_cpu",
        stitch_exterior_meshes=True,
    )
    assert not system_solve_module.supports_exterior_system_protocol(
        system,
        backend_id="bempp_cpu",
        stitch_exterior_meshes=False,
    )


def test_system_worker_projects_exterior_radiation_impedance_to_live_result() -> None:
    _fem, bem = inspect_system_meshes(_fixture_mesh_entries())
    tags = read_surface_physical_names(Path(bem.file))
    group_name, tag = next(iter(tags.items()))
    system, channels = seed_exterior_system(
        (bem,),
        (RadiatorConfig(name=f"{bem.name}:{group_name}", mesh=bem.name, tag=tag, channel="main"),),
    )
    prepared = system_solve_module.prepare_system_ui_solve(
        system,
        freq_min_hz=500.0,
        freq_max_hz=500.0,
        freq_count=1,
        observation_distance_m=2.0,
        polar_angle_step_deg=90.0,
        component_channel_by_id=channels,
        backend_id="beat_cpu",
    )
    point_count = prepared.horizontal_count + prepared.vertical_count
    result = SystemFrequencyResult(
        freq_hz=500.0,
        excitation_port_ids=prepared.request.excitation_port_ids,
        quantities=(
            QuantityResult(
                id="ui:exterior-pressure",
                quantity="exterior_pressure",
                unit="Pa",
                axes=("excitation", "observation"),
                values=np.ones((1, point_count), dtype=np.complex64),
            ),
            QuantityResult(
                id=RADIATION_IMPEDANCE_ID,
                quantity="radiation_impedance",
                unit="N*s/m",
                target_id=RADIATOR_DOMAIN_ID,
                axes=("radiator",),
                values=np.asarray([2.5 + 1.25j], dtype=np.complex64),
            ),
        ),
    )

    live = system_solve_module.SystemSolveWorker(prepared)._to_live_result(result)

    assert live.impedance.tolist() == [[2.5, -1.25]]


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

    assert type(updated.kind) is ComponentKind
    assert not hasattr(editor, "symmetry_combo")
    assert editor.symmetry_inference_label.text() == (
        "Moving surface(s) sliced along the y axis. Detected 2 distinct components in the fully mirrored system. "
        "Projected diaphragm area of 100.25 cm²."
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

    german_locale = QLocale("de_DE")
    re_edit = editor.parameter_edits["re_ohm"]
    re_edit.setLocale(german_locale)
    re_edit.selectAll()
    QTest.keyClicks(re_edit, "6,2")
    assert re_edit.hasAcceptableInput()
    assert editor.component_draft().parameters["re_ohm"] == pytest.approx(6.2)

    editor.axis_mode_combo.setCurrentIndex(editor.axis_mode_combo.findData("manual"))
    axis_spin = editor.axis_spins[0]
    axis_spin.setLocale(german_locale)
    axis_spin.lineEdit().selectAll()
    QTest.keyClicks(axis_spin.lineEdit(), "0,5")
    assert axis_spin.value() == pytest.approx(0.5)


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

    assert updated.parameters["boundary_motion_weights"][boundary.id] == pytest.approx(10.0 ** (-12.0 / 20.0))


def test_semi_inductance_dialog_converts_display_units_and_preserves_disabled_values() -> None:
    dialog = _SemiInductanceDialog(None)
    dialog.enabled_check.setChecked(True)
    for key, value in {
        "re_prime_ohm": "5,7",
        "leb_h": "0.12",
        "le_h": "1,2",
        "ke_semi_h": "0.04",
        "rss_ohm": "1000",
    }.items():
        dialog.parameter_edits[key].setLocale(QLocale("de_DE"))
        dialog.parameter_edits[key].setText(value)

    assert all(edit.hasAcceptableInput() for edit in dialog.parameter_edits.values())

    enabled = dialog.model_parameters()
    assert enabled is not None
    assert enabled.pop("enabled") is True
    assert enabled == pytest.approx(
        {
            "re_prime_ohm": 5.7,
            "leb_h": 0.00012,
            "le_h": 0.0012,
            "ke_semi_h": 0.04,
            "rss_ohm": 1000.0,
        }
    )

    dialog.enabled_check.setChecked(False)
    disabled = dialog.model_parameters()
    assert disabled is not None
    assert disabled["enabled"] is False
    assert disabled["le_h"] == pytest.approx(0.0012)


def test_semi_inductance_dialog_requires_a_complete_enabled_model() -> None:
    dialog = _SemiInductanceDialog(None)
    dialog.enabled_check.setChecked(True)
    dialog.parameter_edits["re_prime_ohm"].setText("5.7")

    with pytest.raises(ValueError, match="Leb is required"):
        dialog.model_parameters()


def test_component_editor_applies_automatic_axis_to_a_two_sided_transducer(monkeypatch) -> None:
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
        "semi_inductance": {
            "enabled": True,
            "re_prime_ohm": 6.2,
            "leb_h": 0.0001,
            "le_h": 0.001,
            "ke_semi_h": 0.04,
            "rss_ohm": 1000.0,
        },
        "lumped_sealed_rear_chamber": {
            "enabled": True,
            "volume_m3": 0.0125,
            "projected_area_m2": 0.5,
        },
    }
    projected_area_calls = 0
    original_projected_area = system_config_module.infer_projected_diaphragm_area

    def count_projected_area_calls(*args, **kwargs):
        nonlocal projected_area_calls
        projected_area_calls += 1
        return original_projected_area(*args, **kwargs)

    monkeypatch.setattr(system_config_module, "infer_projected_diaphragm_area", count_projected_area_calls)
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
            "mesh:rear": _planar_surface_mesh(
                group_name="Rear",
                z_m=0.01,
                reverse=True,
                scale=0.8,
            ),
        },
    )

    assert projected_area_calls == 1
    assert float(editor.parameter_edits["le_h"].text()) == pytest.approx(0.5)
    assert float(editor.parameter_edits["mmd_kg"].text()) == pytest.approx(15.0)
    assert float(editor.parameter_edits["cms_m_per_n"].text()) == pytest.approx(500.0)
    assert not editor.parameter_edits["le_h"].isEnabled()
    assert editor.semi_inductance_button.text() == "Semi-Inductance: On…"
    assert editor.rear_chamber_check.isChecked()
    assert editor.rear_chamber_volume_spin.isEnabled()
    assert editor.rear_chamber_volume_spin.value() == pytest.approx(12.5)
    debounce_spy = QSignalSpy(editor._projected_area_update_timer.timeout)
    editor.boundary_weight_spins[0].setValue(-0.1)
    QTest.qWait(300)
    assert debounce_spy.count() == 0
    editor.boundary_weight_spins[0].setValue(-0.2)
    QTest.qWait(300)
    assert debounce_spy.count() == 0
    QTest.qWait(250)
    assert debounce_spy.count() == 1
    assert editor._projected_area_update_timer.interval() == 500
    editor.boundary_weight_spins[0].setValue(0.0)
    editor._update_projected_area_readout()
    editor.rear_chamber_check.setChecked(False)
    assert not editor.rear_chamber_volume_spin.isEnabled()
    editor.rear_chamber_check.setChecked(True)
    assert "Projected diaphragm area of 4100.00 cm²" in editor.symmetry_inference_label.text()
    assert not editor.projected_area_warning_label.isHidden()
    assert "deviate by 36.0%" in editor.projected_area_warning_label.text()
    editor.parameter_edits["le_h"].setText("0.75")
    editor.parameter_edits["mmd_kg"].setText("18")
    editor.parameter_edits["cms_m_per_n"].setText("625")
    updated = editor.component_draft()

    assert updated.boundary_ids == ("boundary:front", "boundary:rear")
    assert np.abs(updated.parameters["motion_axis"]) == pytest.approx((0.0, 0.0, 1.0))
    assert updated.parameters["le_h"] == pytest.approx(0.00075)
    assert updated.parameters["mmd_kg"] == pytest.approx(0.018)
    assert updated.parameters["cms_m_per_n"] == pytest.approx(0.000625)
    assert updated.parameters["semi_inductance"] == parameters["semi_inductance"]
    assert updated.parameters["lumped_sealed_rear_chamber"] == {
        "enabled": True,
        "volume_m3": pytest.approx(0.0125),
        "projected_area_m2": pytest.approx(0.41),
    }
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
    assert [output.id for output in prepared.request.outputs] == [
        "ui:exterior-pressure",
        "mechanical:diaphragm-velocity",
        "electrical:voice-coil-current",
    ]
    assert prepared.result_domains[-1].id == "components:electrodynamic-transducers"
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


def test_build_identify_warns_when_ordered_seam_simplification_was_used(monkeypatch) -> None:
    dialog = _configured_fixture_dialog()
    original_match = dialog._match_interface_pair
    warnings = []

    def marked_match(*args, **kwargs):
        return replace(original_match(*args, **kwargs), seam_simplification_used=True)

    monkeypatch.setattr(dialog, "_match_interface_pair", marked_match)
    monkeypatch.setattr(
        system_config_module.QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )

    dialog._identify_interfaces()

    assert dialog.interfaces_table.item(0, 3).text() == "Built (inspect)"
    assert len(warnings) == 1
    assert warnings[0][0] == "Inspect Simplified Interface"
    assert "Visually inspect the conformed interface" in warnings[0][1]


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
        observation_planes=(new_observation_plane("Interior Slice"),),
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
    assert prepared.request.solver_options["static_condensation"] is True
    assert prepared.request.solver_options["symmetry"] == "off"

    compatibility_request = prepare_coupled_ui_solve(
        system,
        freq_min_hz=500.0,
        freq_max_hz=1000.0,
        freq_count=3,
        observation_distance_m=2.0,
        polar_angle_step_deg=90.0,
        observation_planes=(new_observation_plane("Interior Slice"),),
        backend_id="beat_cpu_condensed",
    )
    assert compatibility_request.request.solver_options["static_condensation"] is True
    assert compatibility_request.backend_id == "beat_cpu"
    interior = next(
        region for region in prepared.request.compiled_system.regions if region.kind == AcousticRegionKind.BOUNDED_AIR
    )
    assert interior.loss_model["bulk_loss_factor"] == pytest.approx(0.01)
    assert "precision" not in prepared.request.solver_options
    assert "bem_backend" not in prepared.request.solver_options
    assert prepared.backend_id == "beat_cpu"
    points = np.asarray(prepared.request.outputs[0].options["points_m"])
    assert points.shape == (10, 3)
    observation_domains = prepared.request.outputs[0].options["observation_domains"]
    assert [domain["id"] for domain in observation_domains] == [
        "observation:horizontal-polar",
        "observation:vertical-polar",
    ]
    assert {domain.id for domain in prepared.result_domains} == {
        "observation:horizontal-polar",
        "observation:vertical-polar",
        FEM_VOLUME_DOMAIN_ID,
    }
    fem_domain = next(domain for domain in prepared.result_domains if domain.id == FEM_VOLUME_DOMAIN_ID)
    assert fem_domain.coordinates["points_m"].shape == (842, 3)
    assert fem_domain.topology["tetrahedra"].shape == (2925, 4)
    assert fem_domain.metadata["node_offsets"] == [0]
    assert fem_domain.metadata["node_counts"] == [842]
    fem_output = next(output for output in prepared.request.outputs if output.id == FEM_NODAL_PRESSURE_ID)
    assert fem_output.quantity == "fem_nodal_pressure"
    assert fem_output.target_ids == (FEM_VOLUME_DOMAIN_ID,)

    raw_result = SystemFrequencyResult(
        freq_hz=500.0,
        excitation_port_ids=prepared.request.excitation_port_ids,
        quantities=(
            QuantityResult(
                id="ui:exterior-pressure",
                quantity="exterior_pressure",
                unit="Pa",
                axes=("excitation", "observation"),
                values=np.arange(10, dtype=np.float32).reshape(1, 10).astype(np.complex64),
            ),
        ),
    )
    worker = CoupledSolveWorker(prepared)
    canonical = worker._canonical_result(raw_result)
    canonical_by_id = {quantity.id: quantity for quantity in canonical.quantities}

    assert canonical_by_id["acoustic:pressure:horizontal-polar"].target_id == ("observation:horizontal-polar")
    assert canonical_by_id["acoustic:pressure:horizontal-polar"].values.tolist() == [[0.0, 1.0, 2.0, 3.0, 4.0]]
    assert canonical_by_id["acoustic:pressure:vertical-polar"].values.tolist() == [[5.0, 6.0, 7.0, 8.0, 9.0]]
    assert worker._to_live_result(raw_result).horizontal_pressure.shape == (1, 5)


def test_coupled_ui_request_retains_bem_traces_for_exterior_analysis() -> None:
    system = _configured_fixture_dialog().physical_system()
    exterior_plane = replace(
        new_observation_plane("Exterior Field"),
        plane_type=ObservationPlaneType.EXTERIOR,
    )

    prepared = prepare_coupled_ui_solve(
        system,
        freq_min_hz=500.0,
        freq_max_hz=1000.0,
        freq_count=3,
        observation_distance_m=2.0,
        polar_angle_step_deg=90.0,
        observation_planes=(exterior_plane,),
    )

    outputs = {output.id: output for output in prepared.request.outputs}
    domains = {domain.id: domain for domain in prepared.result_domains}
    assert outputs[BEM_BOUNDARY_PRESSURE_ID].quantity == "bem_boundary_pressure"
    assert outputs[BEM_BOUNDARY_PRESSURE_ID].target_ids == (BEM_BOUNDARY_DOMAIN_ID,)
    assert outputs[BEM_BOUNDARY_NEUMANN_ID].quantity == "bem_boundary_neumann"
    assert outputs[BEM_BOUNDARY_NEUMANN_ID].target_ids == (BEM_BOUNDARY_DOMAIN_ID,)
    assert FEM_NODAL_PRESSURE_ID not in outputs

    domain = domains[BEM_BOUNDARY_DOMAIN_ID]
    node_count = domain.coordinates["points_m"].shape[0]
    face_count = domain.topology["triangles"].shape[0]
    assert node_count == sum(domain.metadata["node_counts"])
    assert face_count == sum(domain.metadata["face_counts"])
    assert domain.metadata["pressure_space"] == "P1"
    assert domain.metadata["normal_derivative_space"] == "DP0"


def test_combined_plane_requests_both_fem_and_bem_spatial_fields() -> None:
    system = _configured_fixture_dialog().physical_system()
    combined_plane = replace(
        new_observation_plane("Combined Field"),
        plane_type=ObservationPlaneType.COMBINED,
    )

    prepared = prepare_coupled_ui_solve(
        system,
        freq_min_hz=500.0,
        freq_max_hz=1000.0,
        freq_count=3,
        observation_distance_m=2.0,
        polar_angle_step_deg=90.0,
        observation_planes=(combined_plane,),
    )

    output_ids = {output.id for output in prepared.request.outputs}
    domain_ids = {domain.id for domain in prepared.result_domains}
    assert {BEM_BOUNDARY_PRESSURE_ID, BEM_BOUNDARY_NEUMANN_ID, FEM_NODAL_PRESSURE_ID} <= output_ids
    assert {BEM_BOUNDARY_DOMAIN_ID, FEM_VOLUME_DOMAIN_ID} <= domain_ids


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

    monkeypatch.setattr(system_solve_module, "PhysicalSystemProductionBackend", Backend)
    worker = CoupledSolveWorker(prepared)
    statuses = []
    worker.status.connect(statuses.append)

    with caplog.at_level("INFO", logger="blab.ui.system_solve"):
        worker.run()

    assert statuses == []
    assert "initializing coupled backend detail" in caplog.text
    assert "assembling coupled backend detail" in caplog.text


def test_coupled_worker_selects_rocm_backend(monkeypatch) -> None:
    system = _configured_fixture_dialog().physical_system()
    prepared = prepare_coupled_ui_solve(
        system,
        freq_min_hz=500.0,
        freq_max_hz=500.0,
        freq_count=1,
        observation_distance_m=2.0,
        polar_angle_step_deg=90.0,
        backend_id="beat_rocm",
    )
    backend_options = []

    class Session:
        def solve_stream(self, *, stop_requested=None):
            del stop_requested
            return iter(())

        def stop(self) -> None:
            pass

    class Backend:
        def __init__(self, **kwargs):
            backend_options.append(kwargs)

        def create_system_session(self, request):
            del request
            return Session()

    monkeypatch.setattr(system_solve_module, "PhysicalSystemProductionBackend", Backend)

    CoupledSolveWorker(prepared).run()

    assert backend_options[0]["bem_backend"] == "rocm"


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


def test_coupled_ui_request_routes_rocm_backend_with_hybrid_condensation() -> None:
    system = _configured_fixture_dialog().physical_system()

    prepared = prepare_coupled_ui_solve(
        system,
        freq_min_hz=500.0,
        freq_max_hz=500.0,
        freq_count=1,
        observation_distance_m=2.0,
        polar_angle_step_deg=90.0,
        backend_id="beat_rocm",
    )

    assert prepared.backend_id == "beat_rocm"
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
