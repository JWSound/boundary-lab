import json
from dataclasses import replace
from pathlib import Path

import meshio
import numpy as np
import pytest

from blab.exterior_preparation import prepare_exterior_system
from blab.interface_conform import InterfaceConformError, conform_bem_interface_to_fem
from blab.physical_compiler import PhysicalSystemCompiler
from blab.physical_model import (
    AcousticInterface,
    AcousticRegion,
    AcousticRegionKind,
    Boundary,
    BoundaryKind,
    MeshPurpose,
    MeshResource,
    PhysicalGroupRef,
    PhysicalSystem,
)


@pytest.fixture
def cutout_system(tmp_path):
    """Two tetrahedral domains; the exterior cap is a separate stitch asset."""
    points = np.array([[0, 0, 0], [1000, 0, 0], [0, 1000, 0], [0, 0, 1000], [1000 / 3, 1000 / 3, 0]])
    fem = meshio.Mesh(
        np.array([[0.0, 0, 0], [1000, 0, 0], [0, 1000, 0], [0, 0, -1000]]),
        [("triangle", [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]), ("tetra", [[0, 2, 1, 3]])],
        cell_data={
            "gmsh:physical": [np.array([1, 2, 2, 2]), np.array([3])],
            "gmsh:geometrical": [np.array([1, 2, 2, 2]), np.array([3])],
        },
        field_data={"Interface": np.array([1, 2]), "Wall": np.array([2, 2]), "Air": np.array([3, 3])},
    )
    bem = meshio.Mesh(
        points,
        [("triangle", [[0, 4, 1], [1, 4, 2], [2, 4, 0], [0, 3, 2], [1, 2, 3]])],
        cell_data={"gmsh:physical": [np.array([1, 1, 1, 2, 2])], "gmsh:geometrical": [np.array([1, 1, 1, 2, 2])]},
        field_data={"Interface": np.array([1, 2]), "Wall": np.array([2, 2])},
    )
    cap = meshio.Mesh(
        points[:4],
        [("triangle", [[0, 1, 3]])],
        cell_data={"gmsh:physical": [np.array([1])], "gmsh:geometrical": [np.array([1])]},
        field_data={"Wall": np.array([1, 2])},
    )
    for name, mesh in (("fem", fem), ("bem", bem), ("cap", cap)):
        meshio.write(tmp_path / f"{name}.msh", mesh, file_format="gmsh22", binary=False)
    resources = tuple(
        MeshResource(
            name,
            name,
            str(tmp_path / f"{name}.msh"),
            MeshPurpose.FEM_VOLUME if name == "fem" else MeshPurpose.BEM_SURFACE,
            scale_to_m=0.001,
        )
        for name in ("fem", "bem", "cap")
    )
    boundaries = tuple(
        Boundary(id, id, region, PhysicalGroupRef(mesh, 2, group), kind)
        for id, region, mesh, group, kind in (
            ("fi", "inside", "fem", "Interface", BoundaryKind.INTERFACE),
            ("fw", "inside", "fem", "Wall", BoundaryKind.RIGID),
            ("bi", "outside", "bem", "Interface", BoundaryKind.INTERFACE),
            ("bw", "outside", "bem", "Wall", BoundaryKind.RIGID),
            ("cap", "outside", "cap", "Wall", BoundaryKind.RIGID),
        )
    )
    return PhysicalSystem(
        "test",
        "test",
        resources,
        (
            AcousticRegion(
                "inside", "inside", AcousticRegionKind.BOUNDED_AIR, ("fem",), (PhysicalGroupRef("fem", 3, "Air"),)
            ),
            AcousticRegion("outside", "outside", AcousticRegionKind.UNBOUNDED_AIR, ("bem", "cap")),
        ),
        boundaries,
        (AcousticInterface("pair", "pair", "fi", "bi"),),
    )


def test_stitch_precedes_conform_and_preserves_source_contract(cutout_system, tmp_path):
    system = cutout_system
    original_bytes = {m.id: Path(m.file).read_bytes() for m in system.meshes}
    with pytest.raises(InterfaceConformError, match="open boundary edges"):
        conform_bem_interface_to_fem(meshio.read(system.meshes[0].file), meshio.read(system.meshes[1].file))
    prepared = prepare_exterior_system(system, stitch_tolerance_mm=3, output_root=tmp_path / "derived")
    compiled = PhysicalSystemCompiler().compile(prepared)
    topology = compiled.interfaces[0].topology
    assert len(topology.fem_face_indices) == 1
    assert topology.max_coordinate_error == 0
    assert topology.bem_boundary_edges == 0
    assert prepared.interfaces == system.interfaces
    assert prepared.components == system.components
    assert prepared.excitation_ports == system.excitation_ports
    assert prepared.meshes[0] == system.meshes[0]
    assert [b.id for b in prepared.boundaries] == [b.id for b in system.boundaries]
    # Colliding source names/tags retain separate assignments.
    assert prepared.boundaries[-1].group != prepared.boundaries[-2].group
    assert {m.id: Path(m.file).read_bytes() for m in system.meshes} == original_bytes


def test_identify_uses_assembled_exterior(cutout_system, tmp_path):
    prepared = prepare_exterior_system(
        replace(cutout_system, interfaces=()),
        stitch_tolerance_mm=3,
        output_root=tmp_path / "derived",
        identify_interfaces=True,
    )
    assert [(p.bounded_boundary_id, p.unbounded_boundary_id) for p in prepared.interfaces] == [("fi", "bi")]
    PhysicalSystemCompiler().compile(prepared)


def test_unclosed_cutout_still_fails(cutout_system, tmp_path):
    region = replace(cutout_system.regions[1], mesh_ids=("bem",))
    system = replace(
        cutout_system,
        regions=(cutout_system.regions[0], region),
        boundaries=cutout_system.boundaries[:-1],
        meshes=cutout_system.meshes[:-1],
    )
    with pytest.raises(InterfaceConformError, match="open boundary edges"):
        prepare_exterior_system(system, stitch_tolerance_mm=3, output_root=tmp_path / "derived")


def test_volume_resource_cannot_be_stitched(cutout_system, tmp_path):
    region = replace(cutout_system.regions[1], mesh_ids=("bem", "fem"))
    with pytest.raises(ValueError, match="FEM volume"):
        prepare_exterior_system(
            replace(cutout_system, regions=(region,)), stitch_tolerance_mm=3, output_root=tmp_path / "derived"
        )


def test_headless_coupled_request_stitches_and_preserves_excitation(cutout_system, tmp_path, monkeypatch):
    from blab.headless import HeadlessProject, HeadlessSolveSpec, prepare_headless_solve
    from blab.physical_model import ComponentKind, ExcitationPort, ExcitationPortKind, PhysicalComponent
    from blab.ui.project_state import ProjectPreferencesState

    monkeypatch.chdir(tmp_path)
    system = replace(
        cutout_system,
        boundaries=tuple(replace(b, kind=BoundaryKind.MOVING) if b.id == "fw" else b for b in cutout_system.boundaries),
        components=(PhysicalComponent("source", "source", ComponentKind.IDEAL_VELOCITY_SOURCE, ("fw",)),),
        excitation_ports=(ExcitationPort("drive", "drive", "source", ExcitationPortKind.NORMAL_VELOCITY),),
    )
    project = HeadlessProject(
        path=tmp_path / "project.blab.json",
        payload={"stitch_exterior_meshes": True},
        physical_system=system,
        preferences=ProjectPreferencesState(freq_count=2, polar_angle_step_deg=90, spherical_sampling_enabled=False),
        symmetry="off",
        component_channel_by_id={"source": "input"},
    )
    prepared = prepare_headless_solve(project, HeadlessSolveSpec(), backend_id="beat_cpu")
    assert prepared.request.excitation_port_ids == ("drive",)
    assert prepared.excitation_channel_names.tolist() == ["input"]
    assert prepared.request.compiled_system.interfaces[0].topology.max_coordinate_error == 0
    assert project.physical_system is system


def test_editor_build_and_preview_share_preparation(cutout_system, tmp_path, monkeypatch, qapp):
    from blab.ui.dialogs import MeshDialogEntry
    from blab.ui.mesh_assembly import MeshAssemblyService
    from blab.ui.project_state import ImportedMeshState
    from blab.ui.system_config import SystemConfigDialog, inspect_system_meshes

    system = replace(cutout_system, interfaces=())
    meshes = inspect_system_meshes(
        tuple(MeshDialogEntry(name=m.name, source_file=m.file, scale_factor=0.001) for m in system.meshes)
    )
    dialog = SystemConfigDialog(
        meshes,
        system,
        ("main",),
        stitch_exterior_meshes=True,
        stitch_tolerance_mm=3,
        interface_output_root=tmp_path / "derived",
    )
    errors = []
    monkeypatch.setattr("blab.ui.system_config.QMessageBox.warning", lambda *args: errors.append(args[2]))
    dialog._identify_interfaces()
    assert errors == []
    configured = dialog.configuration()
    assert len(configured.system.interfaces) == 1
    assert configured.mesh_file_overrides_by_name == {}
    service = MeshAssemblyService(tmp_path / "preview")
    assembly = service.prepare(
        generated_mesh_configs=(),
        imported_meshes=tuple(
            ImportedMeshState(name=m.name, source_file=m.file, cleaned_file=m.file, scale_factor=0.001)
            for m in system.meshes
        ),
        radiators=(),
        stitch_imported_meshes=True,
        stitch_tolerance_mm=3,
        symmetry="off",
        physical_system=configured.system,
    )
    assert len(assembly.mesh_configs) == 2
    assert assembly.solver_surface_by_source[("fem", 1)] == ("fem", 1)
    assert assembly.solver_surface_by_source[("bem", 1)][0] == assembly.solver_surface_by_source[("cap", 1)][0]
    PhysicalSystemCompiler().compile(assembly.physical_system)
    dialog.close()


def test_editor_build_uses_symmetry_variant_without_replacing_authoring_asset(
    cutout_system, tmp_path, monkeypatch, qapp,
):
    from blab.ui.dialogs import MeshDialogEntry
    from blab.ui.system_config import SystemConfigDialog, inspect_system_meshes

    # Keep the fixture away from symmetry planes so its entire seam must stitch.
    for resource in cutout_system.meshes:
        mesh = meshio.read(resource.file)
        mesh.points[:, :2] += 2000
        meshio.write(resource.file, mesh, file_format="gmsh22", binary=False)
    reduced = cutout_system.meshes[-1]
    wrong_full = meshio.read(reduced.file)
    wrong_full.points[:, 0] += 1000
    full_path = tmp_path / "cap_full.msh"
    meshio.write(full_path, wrong_full, file_format="gmsh22", binary=False)
    system = replace(cutout_system, interfaces=(), meshes=(
        *cutout_system.meshes[:-1], replace(reduced, file=str(full_path)),
    ))
    canonical = inspect_system_meshes(tuple(
        MeshDialogEntry(name=m.name, source_file=m.file, scale_factor=.001) for m in system.meshes
    ))
    variants = tuple(replace(m, file=reduced.file) if m.name == reduced.name else m for m in canonical)
    dialog = SystemConfigDialog(
        canonical, system, ("main",), stitch_exterior_meshes=True, stitch_tolerance_mm=3,
        symmetry_mode="xy", symmetry_analysis_meshes=variants, interface_output_root=tmp_path / "derived",
    )
    errors = []
    monkeypatch.setattr("blab.ui.system_config.QMessageBox.warning", lambda *args: errors.append(args[2]))
    dialog._identify_interfaces()
    assert errors == []
    configured = dialog.configuration()
    assert len(configured.system.interfaces) == 1
    assert next(m.file for m in configured.system.meshes if m.name == reduced.name) == str(full_path)
    assert configured.mesh_file_overrides_by_name == {}
    dialog.close()


@pytest.mark.parametrize(("symmetry", "expected"), [("off", "full.msh"), ("xy", "reduced.msh")])
def test_headless_resolves_generated_variant_from_saved_canonical_resource(
    cutout_system, tmp_path, symmetry, expected,
):
    from blab.headless import load_headless_project
    from blab.physical_model import physical_system_to_dict

    path = tmp_path / "case.blab.json"
    path.write_text(json.dumps({
        "schema_version": 9, "symmetry": symmetry,
        "physical_system": physical_system_to_dict(cutout_system),
        "generator_documents": [{
            "id": "cap-generator", "name": "cap", "provider_id": "ath", "provider_schema_version": 1,
            "source": {"format": "ath_cfg", "text": ""}, "mesh_enabled": True,
            "mesh_scale_factor": .002, "mesh_translation_mm": [1, 2, 3],
            "artifact": {"output_dir": ".", "mesh_path": "reduced.msh",
                         "cleaned_mesh_path": "full.msh", "reduced_cleaned_mesh_path": "reduced.msh"},
        }],
    }), encoding="utf-8")
    project = load_headless_project(path)
    cap = next(m for m in project.physical_system.meshes if m.name == "cap")
    assert Path(cap.file) == tmp_path / expected
    assert cap.scale_to_m == .002
    assert cap.translation_m == (.001, .002, .003)
