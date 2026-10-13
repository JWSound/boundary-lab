"""The bundled CAD import provider through the public generator contract."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import meshio
import numpy as np
import pytest

from blab.cad_import import face_ref, inspect_step
from blab.generators import GenerationRequest
from blab.generators.application import stage_generation
from blab.generators.base import GenerationCancelledError, complete_generation
from blab.generators.catalog import ProviderCatalog
from blab.generators.configuration import configuration_snapshot
from blab.headless import load_headless_project
from blab.physical_compiler import PhysicalSystemCompiler
from blab.physical_model import ExcitationPort, PhysicalComponent, physical_system_to_dict
from blab.project.io import PROJECT_SCHEMA_VERSION, write_project_file
from blab.project.model import (
    ProjectDocument,
    ProjectPreferencesState,
    generator_document_to_payload,
    new_generator_document,
)
from blab.system_solve import prepare_system_solve
from cad_fixtures import vented_box, waveguide

ROOT = Path(__file__).resolve().parents[1] / "geometry_providers"
PROVIDER = "blab.cad_import"
# Target element size 343 / 3430 / 5 = 20 mm.
FMAX = 3430


@pytest.fixture(scope="module")
def plugin():
    catalog = ProviderCatalog([ROOT], enabled=[PROVIDER])
    catalog.scan()
    return catalog, catalog.factory(PROVIDER, "backend")()


@pytest.fixture(scope="module")
def box_step(tmp_path_factory):
    return vented_box(tmp_path_factory.mktemp("box") / "box.step")


def make_project(catalog, step, /, *, symmetry="off", **changes):
    source = catalog.source_defaults(PROVIDER) | {"step_path": str(step)} | changes
    document = replace(new_generator_document("Speaker", provider_id=PROVIDER, source=source), mesh_scale_factor=1)
    return ProjectDocument(
        (document,), document.id, symmetry=symmetry, project_preferences=ProjectPreferencesState(freq_max_hz=FMAX)
    )


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


def generate(backend, project, output):
    request = make_request(project, output)
    response = backend.create_session(request).generate()
    return stage_generation(project, {}, complete_generation(request, response)), response


def test_vented_box_stages_regions_walls_and_interface_and_prepares_a_coupled_solve(plugin, box_step, tmp_path):
    catalog, backend = plugin
    project, _ = generate(backend, make_project(catalog, box_step), tmp_path)
    system = project.physical_system
    doc = project.generator_documents[0].id

    assert {r.kind: sorted(g.name for g in r.volume_groups) for r in system.regions} == {
        "bounded_air": ["Chamber", "Port"],
        "unbounded_air": [],
    }
    boundaries = {(b.group.mesh_id.removeprefix(f"{doc}/mesh/"), b.group.name): b.kind for b in system.boundaries}
    assert boundaries == {
        ("chamber", "Driver"): "rigid",  # STEP face name
        ("chamber", "walls"): "rigid",
        ("chamber", "Port opening"): "interface",
        ("exterior", "walls"): "rigid",
        ("exterior", "Port opening"): "interface",
    }
    assert len(PhysicalSystemCompiler().compile(system).interfaces) == 1
    prepared = prepare_system_solve(
        with_source(system, f"{doc}/mesh/chamber", "walls"),
        freq_min_hz=30,
        freq_max_hz=100,
        freq_count=3,
        observation_distance_m=2,
        polar_angle_step_deg=90,
    )
    assert prepared.solve_kind == "coupled_bem_fem"


def test_closed_shell_chosen_as_air_in_the_source_stages_a_bounded_region(plugin, tmp_path):
    catalog, backend = plugin
    step = waveguide(tmp_path / "wg.step", surface_model=True)
    project = make_project(catalog, step, closed_shells={"Closed shell 1": "air"})

    project, response = generate(backend, project, tmp_path)

    assert [m.id for m in response.geometry.meshes] == ["closed_shell_1", "exterior"]
    assert [r.kind for r in project.physical_system.regions] == ["bounded_air", "unbounded_air"]
    assert len(PhysicalSystemCompiler().compile(project.physical_system).interfaces) == 1


@pytest.mark.parametrize("symmetry", ["x", "xy"])
def test_reduced_model_is_accepted_under_matching_symmetry(plugin, tmp_path, symmetry):
    catalog, backend = plugin
    step = vented_box(tmp_path / "box.step", symmetry=symmetry)
    project, response = generate(backend, make_project(catalog, step, symmetry=symmetry), tmp_path)
    doc = project.generator_documents[0]

    assert response.geometry.reduced
    assert doc.artifact.reduced
    compiled = PhysicalSystemCompiler().compile(project.physical_system, symmetry_mode=symmetry)
    assert len(compiled.interfaces) == 1


def test_symmetry_mismatch_is_rejected_with_the_suggestion(plugin, tmp_path):
    catalog, backend = plugin
    step = vented_box(tmp_path / "box.step", symmetry="x")
    with pytest.raises(ValueError, match="Project symmetry Off does not match the geometry, which suggests symmetry X"):
        generate(backend, make_project(catalog, step), tmp_path / "runs")


def test_regeneration_keeps_mesh_ids_and_user_edits(plugin, box_step, tmp_path):
    catalog, backend = plugin
    project, first = generate(backend, make_project(catalog, box_step), tmp_path)
    doc = project.generator_documents[0].id
    system = with_source(project.physical_system, f"{doc}/mesh/chamber", "walls")
    walls = next(b for b in system.boundaries if b.kind == "moving")
    system = replace(
        system,
        boundaries=tuple(replace(b, name="Diaphragm") if b is walls else b for b in system.boundaries),
        regions=tuple(replace(r, density_kg_per_m3=1.19) for r in system.regions),
    )
    project.physical_system = system

    regenerated, second = generate(backend, project, tmp_path)

    assert [m.id for m in second.geometry.meshes] == [m.id for m in first.geometry.meshes]
    assert second.geometry.output_dir != first.geometry.output_dir
    kept = next(b for b in regenerated.physical_system.boundaries if b.id == walls.id)
    assert (kept.kind, kept.name) == ("moving", "Diaphragm")
    assert [c.boundary_ids for c in regenerated.physical_system.components] == [(walls.id,)]
    assert {r.density_kg_per_m3 for r in regenerated.physical_system.regions} == {1.19}


def stored_group(name, step, predicate, *, forward="+Z", up="+Y"):
    """A face group as the editor stores it: the faces of ``step`` matching ``predicate``."""
    faces = [f for f in inspect_step(step, forward=forward, up=up).faces if predicate(f)]
    return {"name": name, "faces": [asdict(face_ref(f, forward=forward, up=up)) for f in faces]}


def front_faces(face):
    """The cabinet front around the port hole and the chamber's front, both at z=0 / z=-20 mm."""
    return face.surface_type == "Plane" and abs(abs(face.normal[2]) - 1) < 1e-9 and face.centre[2] > -0.021


def test_new_face_group_arrives_as_a_rigid_boundary_and_user_edits_survive_regeneration(plugin, box_step, tmp_path):
    catalog, backend = plugin
    group = stored_group("Baffle", box_step, lambda f: f.bodies == ("Cabinet",) and front_faces(f))
    project, _ = generate(backend, make_project(catalog, box_step, face_groups=[group]), tmp_path)
    doc = project.generator_documents[0].id
    baffle = next(b for b in project.physical_system.boundaries if b.group.name == "Baffle")

    assert (baffle.id, baffle.kind, baffle.group.mesh_id) == (f"{doc}/exterior/Baffle", "rigid", f"{doc}/mesh/exterior")
    exterior = meshio.read(
        next(m.mesh_path for m in project.generator_documents[0].artifact.meshes if m.id == "exterior")
    )
    assert "Baffle" in exterior.field_data

    project.physical_system = with_source(project.physical_system, f"{doc}/mesh/chamber", "Driver")
    regenerated, _ = generate(backend, project, tmp_path)

    system = regenerated.physical_system
    driver = next(b for b in system.boundaries if b.group.name == "Driver")
    assert driver.kind == "moving"
    assert [c.boundary_ids for c in system.components] == [(driver.id,)]
    assert sum(b.group.name == "Baffle" for b in system.boundaries) == 1
    assert len(PhysicalSystemCompiler().compile(system).interfaces) == 1


def test_orientation_change_remeshes_reoriented_and_keeps_face_groups(plugin, tmp_path):
    catalog, backend = plugin
    step = vented_box(tmp_path / "box.step", cad_axes=("-Y", "+Z"))
    # Picked before the axes were fixed: the cabinet front lies on CAD Y=0.
    group = stored_group("Baffle", step, lambda f: f.bodies == ("Cabinet",) and abs(f.centre[1]) < 1e-6)
    project = make_project(catalog, step, face_groups=[group])
    project, first = generate(backend, project, tmp_path)
    project.generator_documents = (
        replace(
            project.generator_documents[0], source=project.generator_documents[0].source | {"forward": "-Y", "up": "+Z"}
        ),
    )

    _, second = generate(backend, project, tmp_path)

    def bounds(response):
        mesh = meshio.read(next(m.mesh_path for m in response.geometry.meshes if m.id == "exterior"))
        return mesh.points.min(axis=0), mesh.points.max(axis=0), mesh

    assert np.allclose(bounds(first)[:2], [(-0.06, 0, -0.06), (0.06, 0.21, 0.06)], atol=1e-6)
    low, high, mesh = bounds(second)
    assert np.allclose([low, high], [(-0.06, -0.06, -0.21), (0.06, 0.06, 0)], atol=1e-6)
    baffle = mesh.cells_dict["triangle"][
        mesh.cell_data_dict["gmsh:physical"]["triangle"] == mesh.field_data["Baffle"][0]
    ]
    assert np.allclose(mesh.points[baffle][..., 2], 0, atol=1e-9)


@pytest.mark.parametrize(
    "groups,match",
    [
        ({"Baffle": []}, "Face groups must be a list"),
        ([{"name": "A", "faces": []}, {"name": "A", "faces": []}], "Face group 'A' is defined twice"),
        ([{"faces": []}], "Invalid CAD import source"),
    ],
)
def test_invalid_face_groups_are_rejected_before_meshing(plugin, box_step, tmp_path, groups, match):
    catalog, backend = plugin
    session = backend.create_session(make_request(make_project(catalog, box_step, face_groups=groups), tmp_path))
    with pytest.raises(ValueError, match=match):
        session.generate()
    assert not list(tmp_path.iterdir())


def test_target_size_follows_project_sound_speed_and_maximum_frequency(plugin, box_step, tmp_path):
    catalog, backend = plugin
    project, first = generate(backend, make_project(catalog, box_step), tmp_path)
    system = project.physical_system
    project.physical_system = replace(
        system, regions=tuple(replace(r, sound_speed_m_per_s=686.0) for r in system.regions)
    )

    _, second = generate(backend, project, tmp_path)

    # 343 / 3430 / 5 = 20 mm, then 686 / 3430 / 5 = 40 mm.
    assert first.geometry.provider_metadata["target_size_m"] == pytest.approx(0.02)
    assert second.geometry.provider_metadata["target_size_m"] == pytest.approx(0.04)
    # The 120 mm cabinet faces hold only a few 40 mm elements, so edges grow less than 2x.
    assert median_edge(second) > 1.25 * median_edge(first)


def median_edge(response):
    """Median triangle edge of the exterior walls, in metres."""
    mesh = meshio.read(next(m.mesh_path for m in response.geometry.meshes if m.id == "exterior"))
    walls = mesh.cell_data_dict["gmsh:physical"]["triangle"] == mesh.field_data["walls"][0]
    triangles = mesh.points[mesh.cells_dict["triangle"][walls]]
    return np.median(np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=2))


def test_saved_project_reopens_with_meshes_without_the_plugin(plugin, box_step, tmp_path):
    catalog, backend = plugin
    project, _ = generate(backend, make_project(catalog, box_step), tmp_path)
    path = write_project_file(
        tmp_path / "speaker.blab.json",
        {
            "schema_version": PROJECT_SCHEMA_VERSION,
            "generator_documents": [generator_document_to_payload(project.generator_documents[0], absolute_paths=True)],
            "physical_system": physical_system_to_dict(project.physical_system),
        },
    )
    reopened = load_headless_project(path)
    assert len(PhysicalSystemCompiler().compile(reopened.physical_system).interfaces) == 1


def test_cancel_before_start_and_during_meshing(plugin, box_step, tmp_path):
    catalog, backend = plugin
    request = make_request(make_project(catalog, box_step), tmp_path)
    session = backend.create_session(request)
    session.stop()
    with pytest.raises(GenerationCancelledError):
        session.generate()
    session = backend.create_session(request)
    with pytest.raises(GenerationCancelledError):
        session.generate(stop_requested=lambda: session.process is not None)
    assert session.process is None
    assert not list(tmp_path.rglob("*.msh"))


@pytest.mark.parametrize(
    "change,match",
    [
        ({"step_path": ""}, "Choose a STEP file"),
        ({"step_path": "missing.step"}, "not found"),
        ({"elements_per_wavelength": 0}, "Elements per wavelength"),
        ({"forward": "+Y"}, "perpendicular"),
    ],
)
def test_invalid_source_is_rejected_before_meshing(plugin, box_step, tmp_path, change, match):
    catalog, backend = plugin
    session = backend.create_session(make_request(make_project(catalog, box_step, **change), tmp_path))
    with pytest.raises(ValueError, match=match):
        session.generate()
    assert not list(tmp_path.iterdir())


def with_source(system, mesh_id, group):
    """The user's System step: mark one boundary moving and drive it with an ideal velocity source."""
    boundary = next(b for b in system.boundaries if b.group.mesh_id == mesh_id and b.group.name == group)
    return replace(
        system,
        boundaries=tuple(replace(b, kind="moving") if b is boundary else b for b in system.boundaries),
        components=(PhysicalComponent("source", "Source", "ideal_velocity_source", (boundary.id,)),),
        excitation_ports=(ExcitationPort("port", "Port", "source", "normal_velocity"),),
    )
