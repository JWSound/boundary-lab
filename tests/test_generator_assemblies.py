"""A real conforming FEM/BEM/LEM fixture through the generator host boundary."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import meshio
import pytest

from blab.generators import GeneratedGeometry, GeneratedMesh, GenerationRequest, GenerationResponse, generated_mesh_id
from blab.generators.application import stage_generation
from blab.generators.base import complete_generation
from blab.generators.configuration import project_revision
from blab.generators.registry import restore_generator_document
from blab.generators.resources import active_assembly_system, generated_mesh_entries, without_assembly
from blab.headless import load_headless_project
from blab.mesh_data import MeshData
from blab.physical_compiler import PhysicalSystemCompiler
from blab.physical_model import physical_system_to_dict
from blab.project.io import PROJECT_SCHEMA_VERSION, read_project_file, write_project_file
from blab.project.model import (
    ProjectDocument,
    generator_document_to_payload,
    generator_documents_from_payload,
    new_generator_document,
)
from test_physical_compiler import _bidirectional_electrodynamic_fixture_system, _exterior_fixture_system


@pytest.fixture
def assembly_case(tmp_path):
    system = _bidirectional_electrodynamic_fixture_system()
    document = replace(new_generator_document("cabinet", provider_id="test.assembly"), id="design")
    project = ProjectDocument((document,), document.id)
    raw = physical_system_to_dict(system)
    ids = {
        item["id"]: f"design/{item['id']}"
        for key in ("regions", "boundaries", "components", "interfaces", "excitation_ports")
        for item in raw[key]
    }
    ids.update({mesh.id: generated_mesh_id(document.id, mesh.id.split(":")[-1]) for mesh in system.meshes})

    def remap(value):
        if isinstance(value, dict):
            return {ids.get(key, key): remap(item) for key, item in value.items() if item is not None}
        if isinstance(value, list):
            return [remap(item) for item in value]
        return ids.get(value, value) if isinstance(value, str) else value

    raw = remap(raw)
    patch = {
        target: {"upsert": raw[source]}
        for source, target in (
            ("regions", "regions"),
            ("boundaries", "surface_assignments"),
            ("components", "component_assignments"),
            ("interfaces", "interfaces"),
            ("excitation_ports", "excitation_ports"),
        )
    }
    geometry = GeneratedGeometry(
        "test.assembly",
        tmp_path,
        None,
        (),
        meshes=tuple(
            GeneratedMesh(mesh.id.split(":")[-1], mesh.purpose, mesh_path=Path(mesh.file)) for mesh in system.meshes
        ),
        provider_metadata={"generator_version": "test-1", "units": "mm"},
    )
    return project, geometry, patch


def stage(project, geometry, patch):
    document = project.generator_documents[0]
    request = GenerationRequest(
        document.provider_id,
        document.id,
        document.name,
        document.source,
        geometry.output_dir,
        "test",
        project_revision=project_revision(project),
    )
    return stage_generation(
        project, {}, complete_generation(request, GenerationResponse(request.request_id, geometry, patch))
    )


def test_assembly_creates_coupled_system_and_regenerates_without_losing_host_edits(assembly_case):
    from blab.system_solve import prepare_system_solve

    project, geometry, patch = assembly_case
    candidate = stage(project, geometry, patch)
    compiled = PhysicalSystemCompiler().compile(candidate.physical_system)
    assert len(compiled.interfaces) == 1
    assert {mesh.id for mesh in candidate.physical_system.meshes} == {"design/mesh/fem", "design/mesh/bem"}
    assert candidate.physical_system.components[0].kind == "electrodynamic_transducer"
    assert project.physical_system is None and project.generator_documents[0].artifact is None
    component = candidate.physical_system.components[0]
    candidate.physical_system = replace(candidate.physical_system, components=(replace(component, name="Host edit"),))
    candidate.channel_config_by_name = {"main": {"level_db": -6}}
    regenerated = stage(candidate, geometry, {})
    assert regenerated.physical_system == candidate.physical_system
    assert regenerated.channel_config_by_name == candidate.channel_config_by_name
    prepared = prepare_system_solve(
        candidate.physical_system,
        freq_min_hz=100,
        freq_max_hz=200,
        freq_count=2,
        observation_distance_m=2,
        polar_angle_step_deg=90,
    )
    assert prepared.solve_kind == "coupled_bem_fem"
    assert prepared.request.excitation_port_ids == ("design/excitation:radiator",)


@pytest.mark.parametrize("memory", [False, True])
def test_assembly_save_reopen_headless_and_preview_without_plugin(assembly_case, tmp_path, monkeypatch, memory):
    from blab.generators import registry
    from blab.ui.mesh_preparation import MeshPreparationSnapshot, prepare_preview

    project, geometry, patch = assembly_case
    meshes = []
    for mesh in geometry.meshes:
        if memory:
            meshes.append(replace(mesh, mesh_path=None, mesh_data=MeshData.from_meshio(meshio.read(mesh.mesh_path))))
        else:
            path = tmp_path / mesh.mesh_path.name
            path.write_bytes(mesh.mesh_path.read_bytes())
            meshes.append(replace(mesh, mesh_path=path))
    geometry = replace(geometry, meshes=tuple(meshes))
    project = stage(project, geometry, patch)
    payload = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "generator_documents": [generator_document_to_payload(project.generator_documents[0], absolute_paths=True)],
        "physical_system": physical_system_to_dict(project.physical_system),
    }
    path = write_project_file(tmp_path / "assembly.blab.json", payload)
    saved = json.loads(path.read_text())
    if not memory:
        assert all(
            not Path(mesh["mesh_path"]).is_absolute() for mesh in saved["generator_documents"][0]["artifact"]["meshes"]
        )
    monkeypatch.setattr(registry, "create_generator", lambda *_a, **_kw: pytest.fail("Restoration loaded plugin code"))
    loaded = read_project_file(path)
    document = generator_documents_from_payload(loaded["generator_documents"])[0]
    restored = restore_generator_document(document)
    assert [
        (mesh.id, mesh.purpose, mesh.mesh_path, mesh.mesh_data.digest if mesh.mesh_data else None)
        for mesh in restored.meshes
    ] == [
        (mesh.id, mesh.purpose, mesh.mesh_path, mesh.mesh_data.digest if mesh.mesh_data else None)
        for mesh in geometry.meshes
    ]
    assert restored.provider_metadata == geometry.provider_metadata
    headless = load_headless_project(path)
    assert len(PhysicalSystemCompiler().compile(headless.physical_system).interfaces) == 1
    snapshot = MeshPreparationSnapshot(
        (document,), {document.id: restored}, (), (), headless.physical_system, "off", False, 2
    )
    preview, _, options, warning = prepare_preview(snapshot, tmp_path)
    assert len(preview.mesh_configs) == 2 and not warning
    assert len(options["loaded_meshes"]) == 2


@pytest.mark.parametrize("fault", ["group", "purpose", "interface", "unowned", "missing_mesh", "stale", "symmetry"])
def test_assembly_rejection_is_atomic(assembly_case, fault):
    project, geometry, patch = assembly_case
    project = stage(project, geometry, patch)
    before = deepcopy(project)
    broken = {}
    if fault == "group":
        broken = {"surface_assignments": {"upsert": [{"id": "design/boundary:radiator", "group": {"name": "absent"}}]}}
    elif fault == "purpose":
        geometry = replace(geometry, meshes=(replace(geometry.meshes[0], purpose="bem_surface"), geometry.meshes[1]))
    elif fault == "interface":
        broken = {
            "interfaces": {
                "upsert": [{"id": "design/interface:port", "unbounded_boundary_id": "design/boundary:exterior"}]
            }
        }
    elif fault == "unowned":
        other = replace(project.physical_system.components[0], id="foreign", boundary_ids=())
        project.physical_system = replace(
            project.physical_system, components=(*project.physical_system.components, other)
        )
        before = deepcopy(project)
        broken = {"component_parameters": {"foreign": {"re_ohm": 7}}}
    elif fault == "missing_mesh":
        geometry = replace(geometry, meshes=(geometry.meshes[0],))
    elif fault == "symmetry":
        broken = {"symmetry_config": {"mode": "x"}}
    if fault == "stale":
        document = project.generator_documents[0]
        request = GenerationRequest(
            document.provider_id,
            document.id,
            document.name,
            document.source,
            geometry.output_dir,
            "test",
            project_revision="stale",
        )
        with pytest.raises(ValueError, match="changed"):
            stage_generation(
                project, {}, complete_generation(request, GenerationResponse(request.request_id, geometry))
            )
    else:
        with pytest.raises(ValueError):
            stage(project, geometry, broken)
    assert project == before


def test_assembly_enable_transform_and_removal(assembly_case):
    project, geometry, patch = assembly_case
    project = stage(project, geometry, patch)
    document = project.generator_documents[0]
    placed = replace(document, mesh_scale_factor=0.002, mesh_translation_mm=(10, 20, 30))
    entries = generated_mesh_entries(placed, geometry)
    assert len(entries) == 2
    assert all(entry.scale_factor == 0.002 and entry.translation_mm == (10, 20, 30) for entry in entries)
    assert not active_assembly_system(project.physical_system, (replace(document, mesh_enabled=False),)).meshes
    assert active_assembly_system(project.physical_system, (document,)) == project.physical_system
    assert not without_assembly(project.physical_system, document).components
    with pytest.raises(ValueError, match="symmetry Off"):
        generated_mesh_entries(document, geometry, "x")


def test_mesh_resource_contract_rejects_ambiguous_or_duplicate_resources(assembly_case):
    _, geometry, _ = assembly_case
    with pytest.raises(ValueError, match="unique"):
        replace(geometry, meshes=(geometry.meshes[0], geometry.meshes[0]))
    with pytest.raises(ValueError, match="legacy"):
        replace(geometry, mesh_path=Path("legacy.msh"))
    with pytest.raises(ValueError, match="exactly one"):
        GeneratedMesh("interior", "fem_volume")
    with pytest.raises(ValueError, match="lowercase"):
        replace(geometry.meshes[0], id="../foreign")


def test_assembly_attaches_to_shared_exterior_without_changing_unrelated_entities(assembly_case):
    project, geometry, patch = assembly_case
    original = _exterior_fixture_system()
    project.physical_system = original
    shared = original.regions[0]
    patch["regions"]["upsert"] = [region for region in patch["regions"]["upsert"] if region["kind"] != "unbounded_air"]
    for boundary in patch["surface_assignments"]["upsert"]:
        if boundary["region_id"] == "design/region:exterior":
            boundary["region_id"] = shared.id
    candidate = stage(project, geometry, patch)
    assert candidate.physical_system.components[0] == original.components[0]
    assert candidate.physical_system.meshes[0] == original.meshes[0]
    assert candidate.physical_system.regions[0] == replace(shared, mesh_ids=(*shared.mesh_ids, "design/mesh/bem"))
    regenerated = stage(candidate, geometry, {})
    assert regenerated.physical_system == candidate.physical_system
    detached = without_assembly(candidate.physical_system, candidate.generator_documents[0])
    assert detached.meshes == original.meshes
    assert detached.regions == original.regions
    assert detached.boundaries == original.boundaries
    assert detached.components == original.components


def test_retiring_resource_requires_and_accepts_explicit_dependency_updates(assembly_case):
    project, geometry, patch = assembly_case
    project = stage(project, geometry, patch)
    surface_only = replace(geometry, meshes=(geometry.meshes[1],))
    patch = {
        "regions": {"remove": ["design/region:interior"]},
        "surface_assignments": {
            "remove": ["design/boundary:radiator", "design/boundary:wall", "design/boundary:fem-interface"],
            "upsert": [{"id": "design/boundary:bem-interface", "kind": "rigid"}],
        },
        "interfaces": {"remove": ["design/interface:port"]},
        "component_assignments": {
            "upsert": [{"id": "design/component:radiator", "boundary_ids": ["design/boundary:exterior"]}]
        },
    }
    candidate = stage(project, surface_only, patch)
    assert [mesh.id for mesh in candidate.physical_system.meshes] == ["design/mesh/bem"]
    assert not candidate.physical_system.interfaces
    assert [mesh.id for mesh in candidate.generator_documents[0].artifact.meshes] == ["bem"]
