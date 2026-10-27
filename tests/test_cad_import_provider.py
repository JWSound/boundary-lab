"""The bundled CAD import provider through the public generator contract."""

from __future__ import annotations

import sys
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import meshio
import numpy as np
import pytest

from blab.config import ChannelConfig
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
from cad_fixtures import bored_ball, multipart_port, single_solid, strip_names, vented_box, waveguide
from cad_plugin import CadGeometry, CadImportResult, CadInterface, CadMesh, face_ref, inspect_step, module

ZERO = module("pipeline").ZERO

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


@pytest.mark.parametrize("shape", ["box", "sphere"])
def test_a_standalone_solid_stages_one_bounded_region_and_rigid_walls(plugin, tmp_path, shape):
    catalog, backend = plugin
    step = single_solid(tmp_path / "air.step", shape)

    project, response = generate(backend, make_project(catalog, step), tmp_path / "out")

    system = project.physical_system
    (region,) = system.regions
    (boundary,) = system.boundaries
    assert region.kind == "bounded_air" and [g.name for g in region.volume_groups] == ["Air"]
    assert boundary.kind == "rigid" and boundary.group.name == "walls"
    assert boundary.group.mesh_id == region.volume_groups[0].mesh_id
    assert system.interfaces == ()
    assert len(response.geometry.meshes) == 1


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
        ("chamber", "Interface 1 Port"): "interface",
        ("exterior", "walls"): "rigid",
        ("exterior", "Interface 1 Port"): "interface",
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


def test_multipart_port_stages_and_compiles_every_mouth_interface(plugin, tmp_path):
    catalog, backend = plugin
    step = multipart_port(tmp_path / "multipart.step")
    project, response = generate(backend, make_project(catalog, step), tmp_path / "out")
    system = project.physical_system
    bounded = next(r for r in system.regions if r.kind == "bounded_air")
    assert {g.name for g in bounded.volume_groups} == {"Chamber", "Annulus", "Core"}
    assert len(system.interfaces) == 2
    assert len(PhysicalSystemCompiler().compile(system).interfaces) == 2
    assert not any("open edge" in warning for warning in response.geometry.provider_metadata["warnings"])


def test_closed_shell_guessed_air_stages_a_bounded_region(plugin, tmp_path):
    catalog, backend = plugin
    step = waveguide(tmp_path / "wg.step", surface_model=True)
    project = make_project(catalog, step)  # no choice: guessed air beside the open baffle

    project, response = generate(backend, project, tmp_path)

    assert [m.id for m in response.geometry.meshes] == ["closed_shell_1", "exterior"]
    assert [r.kind for r in project.physical_system.regions] == ["bounded_air", "unbounded_air"]
    assert len(PhysicalSystemCompiler().compile(project.physical_system).interfaces) == 1


def test_closed_shell_switched_to_exterior_retires_its_region_boundaries_and_interface(plugin, tmp_path):
    catalog, backend = plugin
    step = waveguide(tmp_path / "wg.step", surface_model=True)
    project, _ = generate(backend, make_project(catalog, step), tmp_path)
    document = project.generator_documents[0]
    changed = document.source | {"closed_shells": {"Closed shell 1": "exterior"}}
    project.generator_documents = (replace(document, source=changed),)

    regenerated, response = generate(backend, project, tmp_path)

    system = regenerated.physical_system
    assert [m.id for m in response.geometry.meshes] == ["exterior"]
    assert [r.kind for r in system.regions] == ["unbounded_air"]
    assert {b.group.mesh_id for b in system.boundaries} == {f"{document.id}/mesh/exterior"}
    assert system.interfaces == ()


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


def test_placement_from_the_source_puts_a_shifted_half_model_on_the_symmetry_plane(plugin, tmp_path):
    catalog, backend = plugin
    step = vented_box(tmp_path / "box.step", symmetry="x", shift_mm=(30, 0, 0))
    project = make_project(catalog, step, symmetry="x", rotation_deg=[0, 0, 0], translation_mm=[-30, 0, 0])

    project, _ = generate(backend, project, tmp_path)

    compiled = PhysicalSystemCompiler().compile(project.physical_system, symmetry_mode="x")
    assert len(compiled.interfaces) == 1
    with pytest.raises(ValueError, match="suggests symmetry Off"):
        generate(backend, make_project(catalog, step, symmetry="x"), tmp_path / "unplaced")


def bad_face_size(step):
    """A stored face size with a negative edge length, and the message rejecting it."""
    entry = {"face": asdict(face_ref(inspect_step(step).faces[0])), "size_m": -1}
    return entry, "Invalid face size: Size override edge length must be a positive number."


def test_an_invalid_stored_face_size_is_rejected_by_generate(plugin, box_step, tmp_path):
    catalog, backend = plugin
    entry, message = bad_face_size(box_step)
    project = make_project(catalog, box_step, face_sizes=[entry])

    with pytest.raises(ValueError) as error:
        generate(backend, project, tmp_path / "runs")

    assert str(error.value) == message


def test_symmetry_mismatch_is_rejected_with_the_suggestion(plugin, tmp_path):
    catalog, backend = plugin
    step = vented_box(tmp_path / "box.step", symmetry="x")
    with pytest.raises(ValueError, match="Project symmetry Off does not match the geometry, which suggests symmetry X"):
        generate(backend, make_project(catalog, step), tmp_path / "runs")


@pytest.mark.parametrize("cabinet_name", ["Cabinet", "Port"])
def test_regeneration_keeps_mesh_ids_and_user_edits(plugin, tmp_path, cabinet_name):
    catalog, backend = plugin
    step = vented_box(tmp_path / "box.step", cabinet_name=cabinet_name)
    project, first = generate(backend, make_project(catalog, step), tmp_path)
    doc = project.generator_documents[0].id
    system = with_source(project.physical_system, f"{doc}/mesh/chamber", "walls")
    walls = next(b for b in system.boundaries if b.kind == "moving")
    system = replace(
        system,
        boundaries=tuple(replace(b, name="Diaphragm") if b is walls else b for b in system.boundaries),
        regions=tuple(replace(r, density_kg_per_m3=1.19) for r in system.regions),
        components=tuple(
            replace(c, parameters={"boundary_motion_weights": {walls.id: 0.5}}) for c in system.components
        ),
    )
    project.physical_system = system
    project.channel_config_by_name = {
        "woofer": {k: v for k, v in asdict(ChannelConfig("woofer")).items() if k != "name"}
    }
    project.component_channel_by_id = {"source": "woofer"}

    regenerated, second = generate(backend, project, tmp_path)

    assert [m.id for m in second.geometry.meshes] == [m.id for m in first.geometry.meshes]
    assert second.geometry.output_dir != first.geometry.output_dir
    kept = next(b for b in regenerated.physical_system.boundaries if b.id == walls.id)
    assert (kept.kind, kept.name) == ("moving", "Diaphragm")
    (component,) = regenerated.physical_system.components
    assert component.boundary_ids == (walls.id,)
    assert component.parameters["boundary_motion_weights"] == {walls.id: 0.5}
    assert regenerated.component_channel_by_id == {"source": "woofer"}
    assert {r.density_kg_per_m3 for r in regenerated.physical_system.regions} == {1.19}
    assert regenerated.physical_system.interfaces == system.interfaces
    (interface,) = regenerated.physical_system.interfaces
    by_id = {b.id: b for b in regenerated.physical_system.boundaries}
    assert by_id[interface.bounded_boundary_id].group.mesh_id == f"{doc}/mesh/chamber"
    assert by_id[interface.unbounded_boundary_id].group.mesh_id == f"{doc}/mesh/exterior"
    assert len(PhysicalSystemCompiler().compile(regenerated.physical_system).interfaces) == 1


def stored_group(name, step, predicate):
    """A face group as the editor stores it: the faces of ``step`` matching ``predicate``."""
    faces = [f for f in inspect_step(step).faces if predicate(f)]
    return {"name": name, "faces": [asdict(face_ref(f)) for f in faces]}


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


def test_rotation_change_remeshes_reoriented_and_keeps_face_groups(plugin, tmp_path):
    catalog, backend = plugin
    step = vented_box(tmp_path / "box.step", cad_axes=("-Y", "+Z"))
    # Picked before the rotation was set: the cabinet front lies on CAD Y=0.
    group = stored_group("Baffle", step, lambda f: f.bodies == ("Cabinet",) and abs(f.centre[1]) < 1e-6)
    project = make_project(catalog, step, face_groups=[group])
    project, first = generate(backend, project, tmp_path)
    project.generator_documents = (
        replace(
            project.generator_documents[0],
            source=project.generator_documents[0].source | {"rotation_deg": [-90.0, 0.0, 0.0]},  # CAD -Y on axis
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


def chamber_sides(face):
    """The chamber's four sides (x or y = ±50 mm), whatever the chamber body is called."""
    return abs(face.normal[2]) < 1e-9 and max(abs(face.centre[0]), abs(face.centre[1])) == pytest.approx(0.05)


def chamber_back(face):
    return abs(face.normal[2]) > 0.999 and abs(face.area - 0.01) < 1e-9 and face.centre[2] < -0.1


def test_regeneration_after_a_small_cad_edit_keeps_unnamed_faces_by_fingerprint(plugin, tmp_path):
    catalog, backend = plugin
    step = strip_names(vented_box(tmp_path / "named.step"), tmp_path / "box.step")
    group = stored_group("Chamber sides", step, chamber_sides)
    assert len(group["faces"]) == 4
    project, first = generate(backend, make_project(catalog, step, face_groups=[group]), tmp_path / "runs")
    doc = project.generator_documents[0].id
    (volume,) = [m.id for m in first.geometry.meshes if m.purpose == "fem_volume"]
    project.physical_system = with_source(project.physical_system, f"{doc}/mesh/{volume}", "Chamber sides")

    # 1 mm deeper: the sides move 0.5 mm and grow 0.6 %.
    strip_names(vented_box(tmp_path / "named.step", depth=181), step)
    regenerated, response = generate(backend, project, tmp_path / "runs")

    sides = next(b for b in regenerated.physical_system.boundaries if b.group.name == "Chamber sides")
    assert sides.kind == "moving"
    chamber = meshio.read(next(m.mesh_path for m in response.geometry.meshes if m.id == volume))
    tagged = chamber.cell_data_dict["gmsh:physical"]["triangle"] == chamber.field_data["Chamber sides"][0]
    assert np.isclose(chamber.points[chamber.cells_dict["triangle"][tagged]][..., 2].min(), -0.201)


def test_regeneration_after_a_large_cad_edit_keeps_faces_by_step_name(plugin, tmp_path):
    catalog, backend = plugin
    step = vented_box(tmp_path / "box.step")
    group = stored_group("Diaphragm", step, lambda f: f.step_name == "Driver")
    project, _ = generate(backend, make_project(catalog, step, face_groups=[group]), tmp_path / "runs")
    doc = project.generator_documents[0].id
    project.physical_system = with_source(project.physical_system, f"{doc}/mesh/chamber", "Diaphragm")

    vented_box(step, depth=230)  # the named back moves 50 mm
    regenerated, _ = generate(backend, project, tmp_path / "runs")

    diaphragm = next(b for b in regenerated.physical_system.boundaries if b.group.name == "Diaphragm")
    assert diaphragm.kind == "moving"
    assert [c.boundary_ids for c in regenerated.physical_system.components] == [(diaphragm.id,)]


@pytest.mark.parametrize("with_walls", [False, True])
def test_deleting_a_face_group_removes_its_boundary_and_detaches_its_component(plugin, box_step, tmp_path, with_walls):
    catalog, backend = plugin
    group = stored_group("Diaphragm", box_step, lambda f: f.step_name == "Driver")
    project, _ = generate(backend, make_project(catalog, box_step, face_groups=[group]), tmp_path / "runs")
    doc = project.generator_documents[0].id
    system = with_source(project.physical_system, f"{doc}/mesh/chamber", "Diaphragm")
    walls = f"{doc}/chamber/walls"
    if with_walls:
        system = replace(
            system,
            boundaries=tuple(replace(b, kind="moving") if b.id == walls else b for b in system.boundaries),
            components=tuple(replace(c, boundary_ids=(*c.boundary_ids, walls)) for c in system.components),
        )
    project.physical_system = system
    project.component_channel_by_id = {"source": "main"}
    document = project.generator_documents[0]
    project.generator_documents = (replace(document, source=document.source | {"face_groups": []}),)

    regenerated, _ = generate(backend, project, tmp_path / "runs")

    system = regenerated.physical_system
    assert "Diaphragm" not in {b.group.name for b in system.boundaries}
    assert next(b for b in system.boundaries if b.group.name == "Driver").kind == "rigid"  # back to its STEP name
    if with_walls:
        assert [c.boundary_ids for c in system.components] == [(walls,)]
        assert [p.component_id for p in system.excitation_ports] == ["source"]
    else:
        assert (system.components, system.excitation_ports) == ((), ())
        assert regenerated.component_channel_by_id == {}
    assert len(PhysicalSystemCompiler().compile(system).interfaces) == 1


def chamber_front(face):
    """The chamber's front around the port, which stays at z=-20 mm whatever the depth."""
    return abs(face.normal[2]) > 0.999 and face.centre[2] == pytest.approx(-0.02) and face.area < 0.01


def test_groups_whose_stored_faces_are_gone_need_review_and_reject_generation(plugin, tmp_path):
    catalog, backend = plugin
    step = strip_names(vented_box(tmp_path / "named.step"), tmp_path / "box.step")
    groups = [
        stored_group("Back", step, chamber_back),
        stored_group("Both", step, lambda f: chamber_front(f) or chamber_back(f)),
    ]
    assert [len(g["faces"]) for g in groups] == [1, 2]

    # 50 mm deeper: the back moves 25 mm; the partly matched group is not trimmed silently.
    strip_names(vented_box(tmp_path / "named.step", depth=230), step)
    with pytest.raises(ValueError, match=r"Face groups need review .*: 'Back', 'Both'\. Re-pick or delete"):
        generate(backend, make_project(catalog, step, face_groups=groups), tmp_path / "runs")

    assert not list(tmp_path.rglob("*.msh"))


def test_rejected_regeneration_keeps_the_previous_meshes_and_system(plugin, tmp_path):
    catalog, backend = plugin
    step = strip_names(vented_box(tmp_path / "named.step"), tmp_path / "box.step")
    project, _ = generate(
        backend, make_project(catalog, step, face_groups=[stored_group("Back", step, chamber_back)]), tmp_path / "runs"
    )
    before = deepcopy(project)
    meshes = [m.mesh_path for m in project.generator_documents[0].artifact.meshes]

    strip_names(vented_box(tmp_path / "named.step", depth=230), step)
    with pytest.raises(ValueError, match="Face groups need review .*: 'Back'"):
        generate(backend, project, tmp_path / "runs")

    assert project == before
    assert all(Path(path).is_file() for path in meshes)


def test_unmeshable_face_rejects_regeneration_by_name_and_keeps_the_previous_meshes_and_system(plugin, tmp_path):
    catalog, backend = plugin
    step = vented_box(tmp_path / "box.step")
    project, _ = generate(backend, make_project(catalog, step), tmp_path / "runs")
    before = deepcopy(project)
    meshes = [m.mesh_path for m in project.generator_documents[0].artifact.meshes]

    bored_ball(step)
    with pytest.raises(ValueError, match=r"^Face \d+ \(Torus, Ball\) cannot be meshed: Impossible to mesh periodic"):
        generate(backend, project, tmp_path / "runs")

    assert project == before
    assert all(Path(path).is_file() for path in meshes)


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

    # New designs are Fine: 343 / 3430 / 6 = 16.7 mm, then 686 / 3430 / 6 = 33.3 mm.
    assert first.geometry.provider_metadata["target_size_m"] == pytest.approx(0.343 / 3430 * 1000 / 6)
    assert second.geometry.provider_metadata["target_size_m"] == pytest.approx(0.686 / 3430 * 1000 / 6)
    # The exterior is BEM, 5 per wavelength by default (20 mm, then 40 mm). The 120 mm cabinet faces hold only
    # a few such elements, so edges grow much less than 2x.
    assert median_edge(second) > 1.15 * median_edge(first)


def median_edge(response, mesh_id="exterior", group="walls", face=None):
    """Median triangle edge of a face group (or one face of it), in metres."""
    mesh = meshio.read(next(m.mesh_path for m in response.geometry.meshes if m.id == mesh_id))
    cells = mesh.cell_data_dict["gmsh:physical"]["triangle"] == mesh.field_data[group][0]
    if face is not None:
        cells &= mesh.cell_data_dict["gmsh:geometrical"]["triangle"] == face
    triangles = mesh.points[mesh.cells_dict["triangle"][cells]]
    return np.median(np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=2))


def test_the_bem_target_size_is_reported_and_recorded_beside_the_fem_one(plugin, box_step, tmp_path):
    catalog, backend = plugin
    request = make_request(
        make_project(catalog, box_step, elements_per_wavelength=5.0, bem_elements_per_wavelength=2.0), tmp_path
    )
    messages = []

    response = backend.create_session(request).generate(status_callback=messages.append)

    metadata = response.geometry.provider_metadata
    assert metadata["target_size_m"] == pytest.approx(0.02)
    assert metadata["bem_target_size_m"] == pytest.approx(0.05)
    assert "FEM 20.0 mm" in messages[0] and "BEM 50.0 mm" in messages[0]


def test_a_bem_fineness_given_as_text_is_converted_like_the_fem_one(plugin, box_step, tmp_path):
    catalog, backend = plugin
    project = make_project(catalog, box_step, elements_per_wavelength=5.0, bem_elements_per_wavelength="2")

    _, response = generate(backend, project, tmp_path)

    assert response.geometry.provider_metadata["bem_target_size_m"] == pytest.approx(0.05)


def test_existing_designs_keep_their_elements_per_wavelength_and_mm_overrides(plugin, box_step, tmp_path):
    catalog, backend = plugin
    legacy = dict(elements_per_wavelength=5.0, face_group_sizes_m={"Driver": 0.008}, body_sizes_m={"Port": 0.01})
    project = make_project(catalog, box_step, **legacy)
    for key in ("face_sizes", "volume_sizes"):  # saved before these existed
        del project.generator_documents[0].source[key]

    _, response = generate(backend, project, tmp_path)

    assert response.geometry.provider_metadata["target_size_m"] == pytest.approx(0.02)
    assert 0.0065 < median_edge(response, "chamber", "Driver") < 0.0095


def test_size_overrides_from_the_source_take_levels_and_single_faces(plugin, box_step, tmp_path):
    catalog, backend = plugin
    sides = [f for f in inspect_step(box_step).faces if f.bodies == ("Cabinet",) and abs(abs(f.normal[0]) - 1) < 1e-9]
    lost = asdict(face_ref(replace(sides[0], centre=(1.0, 1.0, 1.0))))
    project = make_project(
        catalog,
        box_step,
        face_group_sizes={"Driver": {"elements_per_wavelength": 12}},  # 343 / 3430 / 12 = 8.3 mm
        face_sizes=[{"face": asdict(face_ref(sides[0])), "size_m": 0.006}, {"face": lost, "size_m": 0.001}],
    )

    _, response = generate(backend, project, tmp_path)

    assert 0.0065 < median_edge(response, "chamber", "Driver") < 0.0095
    # Face ids are stable for the same file and placement; the lost face blocks nothing.
    assert median_edge(response, face=sides[0].id) < 0.008 < median_edge(response, face=sides[1].id)


def test_volume_size_from_the_source_sizes_the_interior_only(plugin, box_step, tmp_path):
    catalog, backend = plugin
    _, plain = generate(backend, make_project(catalog, box_step), tmp_path / "plain")
    project = make_project(catalog, box_step, volume_sizes={"Chamber": {"elements_per_wavelength": 2}})

    _, coarse = generate(backend, project, tmp_path / "coarse")

    assert median_edge(coarse, "chamber", "walls") == pytest.approx(median_edge(plain, "chamber", "walls"), rel=0.1)
    assert tet_count(coarse) < 0.85 * tet_count(plain)


def tet_count(response, mesh_id="chamber"):
    mesh = meshio.read(next(m.mesh_path for m in response.geometry.meshes if m.id == mesh_id))
    return len(mesh.cells_dict["tetra"])


@pytest.mark.parametrize("name", ["Nope", "Cabinet"])
def test_volume_size_for_no_air_body_is_rejected(plugin, box_step, tmp_path, name):
    catalog, backend = plugin
    project = make_project(catalog, box_step, volume_sizes={name: {"size_m": 0.01}})

    with pytest.raises(ValueError, match=f"Volume size for '{name}', which is no air body"):
        generate(backend, project, tmp_path)


def test_the_bem_fineness_from_the_source_sizes_the_exterior_and_without_it_the_fem_one_does(
    plugin, box_step, tmp_path
):
    catalog, backend = plugin
    project = make_project(catalog, box_step, elements_per_wavelength=10.0, bem_elements_per_wavelength=3.0)
    _, split = generate(backend, project, tmp_path / "split")
    project = make_project(catalog, box_step, elements_per_wavelength=10.0)
    del project.generator_documents[0].source["bem_elements_per_wavelength"]  # saved before it existed

    _, fem = generate(backend, project, tmp_path / "fem")

    assert 0.008 < median_edge(fem) < 0.011  # 343 / 3430 / 10 = 10 mm on the exterior too
    # The cabinet back lies beyond the growth from the finer port mouth: 33 mm asked for on a 120 mm face.
    (back,) = [f.id for f in inspect_step(box_step).faces if f.bodies == ("Cabinet",) and f.centre[2] < -0.209]
    assert median_edge(split, face=back) > 2 * median_edge(fem, face=back)
    chamber = median_edge(fem, "chamber", "walls")
    assert median_edge(split, "chamber", "walls") == pytest.approx(chamber, rel=0.1)


def test_new_designs_are_fem_fine_and_bem_medium():
    catalog = ProviderCatalog([ROOT], enabled=[PROVIDER])
    catalog.scan()
    source = catalog.source_defaults(PROVIDER)
    assert source["elements_per_wavelength"] == 6
    assert source["bem_elements_per_wavelength"] == 5
    assert (source["face_sizes"], source["volume_sizes"]) == ([], {})


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


@pytest.mark.timeout(30)
def test_cancel_stops_a_mesh_worker_that_never_returns(plugin, box_step, tmp_path, monkeypatch):
    catalog, backend = plugin
    jobs = sys.modules[type(backend).__module__.replace(".backend", ".jobs")]
    hang = tmp_path / "hang.py"
    hang.write_text("import time\ntime.sleep(600)\n", encoding="utf-8")
    monkeypatch.setattr(jobs, "WORKER", hang)
    session = backend.create_session(make_request(make_project(catalog, box_step), tmp_path / "out"))
    with pytest.raises(GenerationCancelledError):
        session.generate(stop_requested=lambda: session.process is not None)
    assert session.process is None


@pytest.mark.timeout(30)
def test_a_running_mesh_job_reports_its_elapsed_time(plugin, box_step, tmp_path, monkeypatch):
    catalog, backend = plugin
    jobs = sys.modules[type(backend).__module__.replace(".backend", ".jobs")]
    hang = tmp_path / "hang.py"
    hang.write_text("import time\ntime.sleep(600)\n", encoding="utf-8")
    monkeypatch.setattr(jobs, "WORKER", hang)
    session = backend.create_session(make_request(make_project(catalog, box_step), tmp_path / "out"))
    messages = []
    with pytest.raises(GenerationCancelledError):
        session.generate(
            status_callback=messages.append, stop_requested=lambda: any("elapsed 1 s" in m for m in messages)
        )
    assert "Meshing" in messages[0] and "elapsed" not in messages[0]
    assert "Meshing" in messages[-1] and "elapsed 1 s" in messages[-1]


@pytest.mark.parametrize(
    "change,match",
    [
        ({"step_path": ""}, "Choose a STEP file"),
        ({"step_path": "missing.step"}, "not found"),
        ({"elements_per_wavelength": 0}, "Elements per wavelength"),
        ({"bem_elements_per_wavelength": 0}, "BEM elements per wavelength"),
        ({"bem_elements_per_wavelength": "x"}, "BEM elements per wavelength"),
        ({"rotation_deg": [0, "x", 0]}, "Rotation must be three finite numbers"),
        ({"translation_mm": [0, 0]}, "Translation must be three finite numbers"),
    ],
)
def test_invalid_source_is_rejected_before_meshing(plugin, box_step, tmp_path, change, match):
    catalog, backend = plugin
    session = backend.create_session(make_request(make_project(catalog, box_step, **change), tmp_path))
    with pytest.raises(ValueError, match=match):
        session.generate()
    assert not list(tmp_path.iterdir())


def meshed(groups=("Driver", "Interface 1 Port", "walls")):
    """A hand-built meshing result: the vented box's chamber with ``groups``, and the exterior."""
    geometry = CadGeometry((), (), (), (CadInterface("Interface 1 Port", "chamber", (1,)),), (ZERO, ZERO), "off", ())
    meshes = (
        CadMesh("chamber", "fem_volume", Path("chamber.msh"), ("Chamber", "Port"), groups),
        CadMesh("exterior", "bem_surface", Path("exterior.msh"), (), ("Interface 1 Port", "walls")),
    )
    return CadImportResult(geometry, meshes, 0.02)


def patch_of(plugin, result, system=None):
    """The configuration patch of ``result`` for design ``doc`` in a project with ``system``."""
    _, backend = plugin
    request = SimpleNamespace(document_id="doc", mesh_name="Box", configuration={"physical_system": system})
    return sys.modules[type(backend).__module__].configuration_patch(request, result)


def test_first_generation_sends_regions_rigid_boundaries_and_the_interface(plugin):
    patch = patch_of(plugin, meshed())

    assert [(r["id"], r["kind"], r.get("name")) for r in patch["regions"]["upsert"]] == [
        ("doc/chamber", "bounded_air", "Box Chamber, Port"),
        ("doc/exterior", "unbounded_air", "Exterior air"),
    ]
    assert {b["id"]: b["kind"] for b in patch["surface_assignments"]["upsert"]} == {
        "doc/chamber/Driver": "rigid",
        "doc/chamber/Interface 1 Port": "interface",
        "doc/chamber/walls": "rigid",
        "doc/exterior/Interface 1 Port": "interface",
        "doc/exterior/walls": "rigid",
    }
    (interface,) = patch["interfaces"]["upsert"]
    assert (interface["bounded_boundary_id"], interface["unbounded_boundary_id"]) == (
        "doc/chamber/Interface 1 Port",
        "doc/exterior/Interface 1 Port",
    )
    assert "remove" not in patch["surface_assignments"]


def test_regeneration_resends_no_boundary_and_removes_a_gone_group_with_its_component(plugin):
    first = patch_of(plugin, meshed())
    boundaries = [
        b | {"kind": "moving"} if b["id"] == "doc/chamber/Driver" else b for b in first["surface_assignments"]["upsert"]
    ]
    system = {
        "regions": first["regions"]["upsert"],
        "boundaries": boundaries,
        "interfaces": first["interfaces"]["upsert"],
        "components": [{"id": "cone", "boundary_ids": ["doc/chamber/Driver"]}],
        "excitation_ports": [{"id": "port", "component_id": "cone"}],
    }

    patch = patch_of(plugin, meshed(groups=("Interface 1 Port", "walls")), system)

    assert patch["surface_assignments"] == {"upsert": [], "remove": ["doc/chamber/Driver"]}
    assert patch["component_assignments"] == {"upsert": [], "remove": ["cone"]}
    assert patch["excitation_ports"] == {"remove": ["port"]}
    assert (patch["interfaces"]["remove"], patch["regions"]["remove"]) == ([], [])


def test_regeneration_keeps_the_id_of_an_interface_the_user_already_built(plugin):
    first = patch_of(plugin, meshed())
    built = {**first["interfaces"]["upsert"][0], "id": "interface:interface-1-port"}
    system = {"boundaries": first["surface_assignments"]["upsert"], "interfaces": [built]}

    patch = patch_of(plugin, meshed(), system)

    assert [i["id"] for i in patch["interfaces"]["upsert"]] == ["interface:interface-1-port"]


def with_source(system, mesh_id, group):
    """The user's System step: mark one boundary moving and drive it with an ideal velocity source."""
    boundary = next(b for b in system.boundaries if b.group.mesh_id == mesh_id and b.group.name == group)
    return replace(
        system,
        boundaries=tuple(replace(b, kind="moving") if b is boundary else b for b in system.boundaries),
        components=(PhysicalComponent("source", "Source", "ideal_velocity_source", (boundary.id,)),),
        excitation_ports=(ExcitationPort("port", "Port", "source", "normal_velocity"),),
    )
