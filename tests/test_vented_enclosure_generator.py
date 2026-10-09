"""Exercise the shipped plugin through the public generator/physical contracts."""

import importlib
import json
import math
from dataclasses import replace
from pathlib import Path

import meshio
import numpy as np
import pytest

from blab.generators import GenerationRequest
from blab.generators.application import stage_generation
from blab.generators.base import GenerationCancelledError, complete_generation
from blab.generators.catalog import ProviderCatalog
from blab.generators.configuration import configuration_snapshot
from blab.headless import load_headless_project
from blab.physical_compiler import PhysicalSystemCompiler
from blab.physical_model import physical_system_to_dict
from blab.project.io import PROJECT_SCHEMA_VERSION, write_project_file
from blab.project.model import ProjectDocument, generator_document_to_payload, new_generator_document
from blab.system_solve import prepare_system_solve

ROOT = Path(__file__).resolve().parents[1] / "geometry_providers"
PROVIDER = "blab.vented_enclosure"


@pytest.fixture
def plugin():
    catalog = ProviderCatalog([ROOT], enabled=[PROVIDER])
    catalog.scan()
    backend = catalog.factory(PROVIDER, "backend")()
    module = importlib.import_module(type(backend).__module__)
    parameters = importlib.import_module(module.__package__ + ".parameters")
    return catalog, backend, parameters


def make_request(project, output):
    document = project.generator_documents[-1]
    return GenerationRequest(
        PROVIDER,
        document.id,
        document.name,
        document.source,
        output,
        "test",
        configuration=configuration_snapshot(project),
    )


def make_project(source):
    document = replace(new_generator_document("Enclosure", provider_id=PROVIDER, source=source), mesh_scale_factor=1)
    return ProjectDocument((document,), document.id)


@pytest.fixture
def generated(plugin, tmp_path):
    catalog, backend, _ = plugin
    project = make_project(catalog.source_defaults(PROVIDER))
    request = make_request(project, tmp_path)
    response = backend.create_session(request).generate()
    project = stage_generation(project, {}, complete_generation(request, response))
    return project, response


def test_default_meshes_compile_prepare_save_and_reopen(generated, tmp_path):
    project, response = generated
    assert response.geometry.meshes[0].mesh_path.read_text().splitlines()[1] == "4.1 0 8"
    system = project.physical_system
    compiled = PhysicalSystemCompiler().compile(system)
    assert len(compiled.interfaces) == 1
    assert len(system.regions) == 2
    assert system.components[0].parameters["mmd_kg"] == 0.035
    assert system.components[0].parameters["motion_axis"] == [0, 0, 1]
    prepared = prepare_system_solve(
        system, freq_min_hz=30, freq_max_hz=100, freq_count=3, observation_distance_m=2, polar_angle_step_deg=90
    )
    assert prepared.solve_kind == "coupled_bem_fem"
    assert len(prepared.request.excitation_port_ids) == 1
    info = response.geometry.provider_metadata
    assert 0.04 < info["air_volume_m3"] < 0.043
    assert info["interior"]["tetrahedra"] > 100
    path = write_project_file(
        tmp_path / "enclosure.blab.json",
        {
            "schema_version": PROJECT_SCHEMA_VERSION,
            "generator_documents": [generator_document_to_payload(project.generator_documents[0], absolute_paths=True)],
            "physical_system": physical_system_to_dict(system),
        },
    )
    assert not Path(json.loads(path.read_text())["physical_system"]["meshes"][0]["file"]).is_absolute()
    reopened = load_headless_project(path)
    assert len(PhysicalSystemCompiler().compile(reopened.physical_system).interfaces) == 1


def test_shared_mouth_and_driver_geometry_and_closed_exterior(generated):
    _, response = generated
    fem, bem = [meshio.read(mesh.mesh_path) for mesh in response.geometry.meshes]

    def faces(mesh, group):
        tag = mesh.field_data[group][0]
        tri = mesh.cells_dict["triangle"][mesh.cell_data_dict["gmsh:physical"]["triangle"] == tag]
        # The two ASCII writers use different final-digit formatting.
        return {tuple(sorted(tuple(p) for p in face)) for face in np.round(mesh.points[tri], 12)}

    assert faces(fem, "port_mouth") == faces(bem, "port_mouth")
    assert faces(fem, "driver") == faces(bem, "driver")
    tri = bem.cells_dict["triangle"]
    edges = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    assert np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2)
    a, b, c = bem.points[tri].transpose(1, 0, 2)
    assert np.einsum("ij,ij->i", a, np.cross(b, c)).sum() > 0


@pytest.mark.parametrize("nfr,lip", [(0, 0), (0, 0.008), (0.12, 0)])
def test_straight_and_unrounded_profiles_mesh(plugin, tmp_path, nfr, lip):
    catalog, backend, _ = plugin
    source = catalog.source_defaults(PROVIDER) | {"port_nfr": nfr, "port_roundover_m": lip}
    project = make_project(source)
    request = make_request(project, tmp_path)
    response = backend.create_session(request).generate()
    accepted = stage_generation(project, {}, complete_generation(request, response))
    assert len(PhysicalSystemCompiler().compile(accepted.physical_system).interfaces) == 1


def test_regeneration_preserves_driver_materials_routes_and_previous_files(plugin, generated, tmp_path):
    _, backend, _ = plugin
    project, previous = generated
    component = project.physical_system.components[0]
    project.physical_system = replace(
        project.physical_system, components=(replace(component, parameters=component.parameters | {"re_ohm": 7.3}),)
    )
    project.component_channel_by_id[component.id] = "custom"
    project.channel_config_by_name["custom"] = {"level_db": -4}
    old_bytes = previous.geometry.meshes[0].mesh_path.read_bytes()
    request = make_request(project, tmp_path)
    response = backend.create_session(request).generate()
    accepted = stage_generation(project, {}, complete_generation(request, response))
    assert accepted.physical_system.components[0].parameters["re_ohm"] == 7.3
    assert accepted.component_channel_by_id[component.id] == "custom"
    assert accepted.channel_config_by_name == project.channel_config_by_name
    assert previous.geometry.meshes[0].mesh_path.read_bytes() == old_bytes
    assert response.geometry.output_dir != previous.geometry.output_dir


def test_cancel_before_start_and_during_worker(plugin, tmp_path):
    catalog, backend, _ = plugin
    request = make_request(make_project(catalog.source_defaults(PROVIDER)), tmp_path)
    session = backend.create_session(request)
    session.stop()
    with pytest.raises(GenerationCancelledError):
        session.generate()
    session = backend.create_session(request)
    with pytest.raises(GenerationCancelledError):
        session.generate(stop_requested=lambda: session.process is not None)
    assert session.process is None
    assert not list(tmp_path.rglob("interior.msh"))


def test_second_enclosure_joins_shared_exterior_without_changing_material(plugin, generated, tmp_path):
    catalog, backend, _ = plugin
    project, _ = generated
    exterior = next(r for r in project.physical_system.regions if r.kind == "unbounded_air")
    project.physical_system = replace(
        project.physical_system,
        regions=tuple(
            replace(r, density_kg_per_m3=1.19) if r.id == exterior.id else r for r in project.physical_system.regions
        ),
    )
    document = replace(
        new_generator_document("Second enclosure", provider_id=PROVIDER, source=catalog.source_defaults(PROVIDER)),
        mesh_scale_factor=1,
        mesh_translation_mm=(500, 0, 0),
    )
    project.generator_documents += (document,)
    request = make_request(project, tmp_path)
    response = backend.create_session(request).generate()
    accepted = stage_generation(project, {}, complete_generation(request, response))
    shared = [r for r in accepted.physical_system.regions if r.kind == "unbounded_air"]
    assert len(shared) == 1 and len(shared[0].mesh_ids) == 2
    assert shared[0].density_kg_per_m3 == 1.19
    assert len(PhysicalSystemCompiler().compile(accepted.physical_system).interfaces) == 2


@pytest.mark.parametrize(
    "change,match",
    [
        ({"port_length_m": 0.8}, "rear wall"),
        ({"port_y_m": 0.1}, "overlap"),
        ({"cap_diameter_m": 0.2}, "Dust cap"),
        ({"width_m": float("nan")}, "finite|between"),
        ({"port_roundover_m": 0.05, "port_length_m": 0.1}, "quarter"),
        ({"detail_size_m": 0.04, "mesh_size_m": 0.02}, "element size"),
    ],
)
def test_invalid_geometry_rejected_before_launch(plugin, tmp_path, change, match):
    catalog, backend, _ = plugin
    source = catalog.source_defaults(PROVIDER) | change
    session = backend.create_session(make_request(make_project(source), tmp_path))
    with pytest.raises(ValueError, match=match):
        session.generate()
    assert session.process is None
    assert not list(tmp_path.iterdir())


def test_nfr_profile_matches_supplied_stv_workbook_geometry(plugin):
    catalog, _, parameters = plugin
    # Supplied Beta03: input+result C14/C16/C19/C20 (cm converted to m).
    length, diameter, nfr, radius = 0.48624864659810086, 0.06400963526512681, 0.13107188245354112, 1.854893046075131
    source = catalog.source_defaults(PROVIDER) | {
        "port_length_m": length,
        "port_area_m2": math.pi * (diameter / 2) ** 2,
        "port_nfr": nfr,
        "port_roundover_m": 0,
    }
    profile = parameters.port_profile(source)
    assert length / (2 * nfr) == pytest.approx(radius)
    assert 2 * profile["mouth"][1] == pytest.approx(0.09601445289769021)
