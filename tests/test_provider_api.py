"""Provider responses preserve host edits and fail without partial commits."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import meshio
import numpy as np
import pytest

from blab.generators.application import stage_generation
from blab.generators.base import (
    GeneratedGeometry,
    GenerationRequest,
    GenerationResponse,
    complete_generation,
)
from blab.generators.configuration import (
    apply_configuration_patch,
    configuration_snapshot,
    project_revision,
    validate_patch,
)
from blab.physical_model import (
    AcousticRegion,
    AcousticRegionKind,
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
)
from blab.project.model import ProjectDocument, ProjectPreferencesState, new_generator_document


@pytest.fixture
def project(tmp_path):
    document = replace(new_generator_document("horn", provider_id="test"), id="design")
    return ProjectDocument(
        generator_documents=(document,), active_generator_document_id="design",
        physical_system=PhysicalSystem(
            id="system", name="System",
            meshes=(MeshResource("mesh", "horn", str(tmp_path / "old.msh"), MeshPurpose.BEM_SURFACE),),
            regions=(AcousticRegion("air", "Air", AcousticRegionKind.UNBOUNDED_AIR, ("mesh",)),),
            boundaries=(Boundary("throat", "Throat", "air", PhysicalGroupRef("mesh", 2, "throat"), BoundaryKind.MOVING),),
            components=(PhysicalComponent(
                "driver", "Driver", ComponentKind.IDEAL_VELOCITY_SOURCE, ("throat",),
                {"motion_profile": "uniform", "custom": {"preserved": 7, "changed": 1}},
            ),),
            excitation_ports=(ExcitationPort("drive", "Drive", "driver", ExcitationPortKind.NORMAL_VELOCITY),),
        ),
        channel_config_by_name={"main": {"voltage_v": 2.0, "level_db": -3.0, "hpf": {"type": "none"}}},
        component_channel_by_id={"driver": "main"},
        project_preferences=ProjectPreferencesState(freq_min_hz=321, freq_max_hz=4321, freq_count=9),
    )


def apply(project, patch):
    return apply_configuration_patch(project, patch, document_id="design", mesh_ids={"mesh"})


def request(project, tmp_path):
    return GenerationRequest(
        provider_id="test", document_id="design", mesh_name="horn",
        source=project.generator_documents[0].source, run_root=tmp_path, case_name="test",
        project_revision=project_revision(project), configuration=configuration_snapshot(project),
    )


def geometry(tmp_path, *, group="throat"):
    path = tmp_path / "new.msh"
    meshio.write(path, meshio.Mesh(
        points=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
        cells=[("triangle", np.array([[0, 1, 2]]))],
        cell_data={"gmsh:physical": [np.array([2])], "gmsh:geometrical": [np.array([1])]},
        field_data={group: np.array([2, 2])},
    ), file_format="gmsh22", binary=False)
    return GeneratedGeometry("test", tmp_path, path, ())


def test_partial_updates_preserve_nested_parameters_and_host_frequency_settings(project):
    before = deepcopy(project)
    candidate = apply(project, {
        "component_parameters": {"driver": {"custom": {"changed": 2}}},
        "channel_config": {"upsert": [{"name": "main", "level_db": -6.0}]},
        "stitching_config": {"tolerance_mm": 0.25},
    })
    assert candidate.physical_system.components[0].parameters["custom"] == {"preserved": 7, "changed": 2}
    assert candidate.channel_config_by_name["main"] == {"voltage_v": 2.0, "level_db": -6.0, "hpf": {"type": "none"}}
    assert candidate.project_preferences == replace(before.project_preferences, stitch_tolerance_mm=0.25)
    assert project == before


def test_empty_sections_do_not_clear_configuration(project):
    assert apply(project, {}) == project
    assert apply(project, {"channel_config": {"upsert": [], "remove": []}}) == project
    assert apply(project, {"surface_assignments": {"upsert": [], "remove": []}}) == project
    candidate = apply(project, {"component_parameters": {"driver": {}}})
    assert candidate.physical_system.components == project.physical_system.components


def test_provider_channel_routing_survives_legacy_seed_guard(project):
    from blab.project.migration import AUTO_SEEDED_EXTERIOR_KEY

    project.physical_system = replace(project.physical_system, metadata={AUTO_SEEDED_EXTERIOR_KEY: True})
    candidate = apply(project, {
        "channel_config": {"upsert": [{"name": "other"}]},
        "component_channels": {"upsert": [{"id": "driver", "channel": "other"}]},
    })
    assert candidate.physical_system.metadata[AUTO_SEEDED_EXTERIOR_KEY] is False


def test_empty_operations_can_be_applied_before_first_geometry(project):
    project.physical_system = None
    candidate = apply(project, {"surface_assignments": {"upsert": [], "remove": []}})
    assert candidate == project


@pytest.mark.parametrize("patch", [
    {"freq_count": 2},
    {"channel_config": None},
    {"channel_config": {"upsert": [{"name": "main", "level_db": None}]}},
    {"channel_config": {"upsert": [{"name": "main", "level_db": float("nan")}]}},
    {"component_assignments": {"upsert": [{"id": "driver", "freq_count": 2}]}},
    {"component_assignments": {"upsert": [{"id": "driver"}], "remove": ["driver"]}},
    {"surface_assignments": {"upsert": [{"id": "throat", "group": {"typo": "throat"}}]}},
])
def test_malformed_patch_is_rejected(patch):
    with pytest.raises(ValueError):
        validate_patch(patch)


def test_removing_referenced_boundary_is_atomic(project):
    before = deepcopy(project)
    with pytest.raises(ValueError, match="missing boundaries"):
        apply(project, {
            "surface_assignments": {"remove": ["throat"]},
            "channel_config": {"upsert": [{"name": "main", "level_db": -9.0}]},
        })
    assert project == before


def test_explicit_remove_requires_removing_dependent_ports(project):
    with pytest.raises(ValueError, match="missing component"):
        apply(project, {"component_assignments": {"remove": ["driver"]}})
    candidate = apply(project, {
        "component_assignments": {"remove": ["driver"]},
        "excitation_ports": {"remove": ["drive"]},
    })
    assert candidate.physical_system.components == ()
    assert candidate.component_channel_by_id == {}


def test_cannot_modify_other_design_or_create_unnamespaced_entity(project):
    with pytest.raises(ValueError, match="unrelated"):
        apply_configuration_patch(project, {"component_parameters": {"driver": {"x": 1}}},
                                  document_id="other", mesh_ids=set())
    with pytest.raises(ValueError, match="must start"):
        apply(project, {"component_assignments": {"upsert": [{"id": "foreign"}]}})


def test_new_namespaced_component_can_be_routed(project):
    candidate = apply(project, {
        "component_assignments": {"upsert": [{
            "id": "design/new", "name": "New", "kind": "ideal_velocity_source", "boundary_ids": ["throat"],
        }]},
        "component_channels": {"upsert": [{"id": "design/new", "channel": "main"}]},
    })
    assert candidate.component_channel_by_id["design/new"] == "main"


def test_channel_removal_cannot_leave_dangling_routing(project):
    with pytest.raises(ValueError, match="missing channel"):
        apply(project, {"channel_config": {"remove": ["main"], "upsert": [{"name": "other"}]}})


@pytest.mark.parametrize("patch", [
    {"symmetry_config": {"mode": "bad"}},
    {"stitching_config": {"enabled": "false"}},
    {"stitching_config": {"tolerance_mm": -1}},
    {"channel_config": {"upsert": [{"name": "main", "polarity": 0}]}},
    {"channel_config": {"upsert": [{"name": "main", "hpf": {"type": "highpass"}}]}},
])
def test_invalid_settings_do_not_mutate_project(project, patch):
    before = deepcopy(project)
    with pytest.raises(ValueError):
        apply(project, patch)
    assert project == before


def test_request_detaches_provider_source_and_configuration(project, tmp_path):
    req = request(project, tmp_path)
    req.source["values"]["x"] = 5
    req.configuration["channel_config"]["main"]["level_db"] = 20
    assert project.generator_documents[0].source["values"] == {}
    assert project.channel_config_by_name["main"]["level_db"] == -3


def test_response_correlation_and_legacy_adapter(project, tmp_path):
    req, result = request(project, tmp_path), geometry(tmp_path)
    legacy = complete_generation(req, result)
    assert legacy.legacy_response
    assert legacy.configuration_patch == {}
    response = GenerationResponse(req.request_id, result)
    assert not complete_generation(req, response).legacy_response
    for invalid in (replace(response, request_id="wrong"), replace(response, schema_version=99),
                    replace(response, geometry=replace(result, provider_id="other"))):
        with pytest.raises(ValueError):
            complete_generation(req, invalid)


def test_stage_generation_updates_file_and_settings_together(project, tmp_path):
    req, result = request(project, tmp_path), geometry(tmp_path)
    completed = complete_generation(req, GenerationResponse(req.request_id, result, {
        "channel_config": {"upsert": [{"name": "main", "level_db": -12.0}]},
    }))
    before = deepcopy(project)
    candidate = stage_generation(project, {}, completed)
    assert candidate.generator_documents[0].artifact.mesh_path == str(result.mesh_path)
    assert candidate.physical_system.meshes[0].file == str(result.mesh_path)
    assert candidate.channel_config_by_name["main"]["level_db"] == -12
    assert project == before


def test_new_generation_with_omitted_configuration_checks_preserved_groups(project, tmp_path):
    req, result = request(project, tmp_path), geometry(tmp_path, group="different")
    before = deepcopy(project)
    completed = complete_generation(req, GenerationResponse(req.request_id, result))
    with pytest.raises(ValueError, match="no longer contains"):
        stage_generation(project, {}, completed)
    assert project == before


def test_stage_rejects_edited_or_removed_design(project, tmp_path):
    req, result = request(project, tmp_path), geometry(tmp_path)
    completed = complete_generation(req, GenerationResponse(req.request_id, result))
    project.channel_config_by_name["main"]["level_db"] = 4
    with pytest.raises(ValueError, match="inputs changed"):
        stage_generation(project, {}, completed)


def test_project_revision_ignores_tab_selection_but_tracks_artifacts(project):
    revision = project_revision(project)
    project.active_generator_document_id = None
    assert project_revision(project) == revision
    project.symmetry = "x"
    assert project_revision(project) != revision


def test_worker_emits_validated_response_and_rejects_bad_identity(project, tmp_path, monkeypatch):
    from blab.ui.generator_worker import GeneratorWorker

    req, result = request(project, tmp_path), geometry(tmp_path)

    class Backend:
        response_id = req.request_id

        def create_session(self, request):
            return self

        def generate(self, **kwargs):
            return GenerationResponse(self.response_id, result)

    backend = Backend()
    monkeypatch.setattr("blab.ui.generator_worker.create_generator", lambda *args, **kwargs: backend)
    worker = GeneratorWorker(req)
    received, errors, finished = [], [], []
    worker.generated.connect(received.append)
    worker.failed.connect(errors.append)
    worker.finished.connect(lambda: finished.append(True))
    worker.run()
    assert received[0].request.request_id == req.request_id
    backend.response_id = "wrong"
    worker.run()
    assert len(received) == 1
    assert "request ID" in errors[0]
    assert len(finished) == 2


def test_main_window_commits_provider_settings_and_preserves_them_on_regeneration(main_window, project, tmp_path):
    from blab.project.io import read_project_file, write_project_file

    main_window.project = project
    revision, configuration = main_window.generation_context()
    req = replace(request(project, tmp_path), project_revision=revision, configuration=configuration)
    result = geometry(tmp_path)
    completed = complete_generation(req, GenerationResponse(req.request_id, result, {
        "component_parameters": {"driver": {"custom": {"changed": 5}}},
        "channel_config": {"upsert": [{"name": "main", "level_db": -8.0}]},
        "stitching_config": {"tolerance_mm": 0.3},
    }))
    main_window.accept_generation(completed)
    assert main_window.project.channel_config_by_name["main"]["level_db"] == -8
    saved = write_project_file(tmp_path / "saved.blab.json", main_window.project_workflow.project_payload())
    restored = read_project_file(saved)
    assert restored["channel_config_by_name"]["main"]["level_db"] == -8
    assert restored["physical_system"]["components"][0]["parameters"]["custom"]["changed"] == 5
    assert main_window.preferences.stitch_tolerance_mm == 0.3
    assert main_window.generated_geometry_by_document_id["design"].mesh_path == result.mesh_path
    preserved = deepcopy(main_window.project.physical_system.components)
    revision, configuration = main_window.generation_context()
    req = replace(req, request_id="regenerate", project_revision=revision, configuration=configuration)
    main_window.accept_generation(complete_generation(req, GenerationResponse(req.request_id, result)))
    assert main_window.project.physical_system.components == preserved
    assert main_window.project.channel_config_by_name["main"]["level_db"] == -8


def test_main_window_rejects_bad_patch_without_replacing_artifact(main_window, project, tmp_path):
    main_window.project = project
    revision, configuration = main_window.generation_context()
    req = replace(request(project, tmp_path), project_revision=revision, configuration=configuration)
    completed = complete_generation(req, GenerationResponse(req.request_id, geometry(tmp_path), {
        "surface_assignments": {"remove": ["throat"]},
    }))
    with pytest.raises(ValueError):
        main_window.accept_generation(completed)
    assert main_window.project is project
    assert "design" not in main_window.generated_geometry_by_document_id


def test_registry_allows_explicit_registration_but_never_replaces_builtins(monkeypatch):
    from blab.generators import registry
    from blab.generators.base import GeneratorCapabilities

    monkeypatch.setattr(registry, "_BACKENDS", dict(registry._BACKENDS))
    provider = object()
    info = registry.GeneratorBackendInfo("test", "Test", GeneratorCapabilities(("json",)), factory=lambda: provider)
    registry.register_generator(info)
    assert registry.create_generator("test") is provider
    with pytest.raises(ValueError, match="already registered"):
        registry.register_generator(replace(info, provider_id="ath"))
