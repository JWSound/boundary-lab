"""Headless CAD import: STEP file in, conformal mesh assembly out."""

from __future__ import annotations

import json
import re
from dataclasses import replace

import gmsh
import meshio
import numpy as np
import pytest

from blab.cad_import import (
    CadClosedShell,
    CadFace,
    CadFaceRef,
    CadImportSettings,
    CadPreview,
    CadScan,
    face_ref,
    inspect_step,
    match_face_ref,
    mesh_step,
    open_step,
    place_preview,
    preview_from_json,
    preview_to_json,
)
from blab.cad_import import cli as cad_cli
from blab.interface_conform import validate_conforming_interfaces
from cad_fixtures import named_sides, strip_names, two_chambers, vented_box, waveguide

# Target element size 343 / 3430 / 5 = 20 mm.
SETTINGS = CadImportSettings(fmax_hz=3430.0, sound_speed_m_per_s=343.0)


def groups(path):
    """Physical group names by dimension, read back from a written mesh file."""
    mesh = meshio.read(path)
    return {dim: sorted(n for n, (_, d) in mesh.field_data.items() if d == dim) for dim in (2, 3)}


@pytest.fixture(scope="module")
def vented(tmp_path_factory):
    root = tmp_path_factory.mktemp("vented")
    return mesh_step(vented_box(root / "box.step"), SETTINGS, root / "out")


def test_vented_box_becomes_one_air_mesh_and_one_exterior_mesh(vented):
    meshes = {mesh.id: mesh for mesh in vented.meshes}

    assert {mesh_id: mesh.purpose for mesh_id, mesh in meshes.items()} == {
        "chamber": "fem_volume",
        "exterior": "bem_surface",
    }
    # The chamber's back face is named "Driver" in the STEP file.
    assert groups(meshes["chamber"].path) == {2: ["Driver", "Interface 1 Port", "walls"], 3: ["Chamber", "Port"]}
    assert groups(meshes["exterior"].path) == {2: ["Interface 1 Port", "walls"], 3: []}
    assert np.allclose(vented.geometry.bbox_m, [(-0.06, -0.06, -0.21), (0.06, 0.06, 0.0)], atol=1e-6)


def edge_lengths(mesh, cell_type="triangle", group=None):
    """Mesh edge lengths in metres, optionally of the cells of one physical group."""
    lengths = []
    for block, physical in zip(mesh.cells, mesh.cell_data["gmsh:physical"], strict=True):
        if block.type != cell_type:
            continue
        cells = block.data if group is None else block.data[physical == mesh.field_data[group][0]]
        corners = mesh.points[cells]
        for i in range(cells.shape[1]):
            for j in range(i + 1, cells.shape[1]):
                lengths.append(np.linalg.norm(corners[:, i] - corners[:, j], axis=1))
    return np.concatenate(lengths)


def read(result, mesh_id):
    return meshio.read(next(mesh.path for mesh in result.meshes if mesh.id == mesh_id))


def test_edges_follow_the_target_element_size(vented):
    assert vented.target_size_m == pytest.approx(0.02)
    walls = edge_lengths(read(vented, "exterior"), group="walls")
    tets = edge_lengths(read(vented, "chamber"), "tetra")

    assert 0.016 < np.median(walls) < 0.022
    # Gmsh's 3D Delaunay edges run slightly long against the requested size.
    assert 0.016 < np.median(tets) < 0.026
    assert np.quantile(tets, 0.95) < 0.034


def test_step_named_face_group_takes_its_own_size(vented, tmp_path):
    settings = replace(SETTINGS, face_group_sizes_m={"Driver": 0.008})
    result = mesh_step(vented_box(tmp_path / "box.step"), settings, tmp_path / "out")

    assert [m.surface_groups for m in result.meshes] == [
        ("Driver", "Interface 1 Port", "walls"),
        ("Interface 1 Port", "walls"),
    ]
    assert 0.0065 < np.median(edge_lengths(read(result, "chamber"), group="Driver")) < 0.0095
    assert 0.016 < np.median(edge_lengths(read(result, "exterior"), group="walls")) < 0.022


def test_face_group_spans_several_faces_and_overrides_a_step_name(vented, tmp_path):
    cabinet = [f.id for f in vented.geometry.faces if f.bodies == ("Cabinet",) and abs(f.centre[2]) > 1e-6]
    driver = next(f.id for f in vented.geometry.faces if f.step_name == "Driver")
    settings = replace(SETTINGS, face_groups={"Sides": tuple(cabinet[:2]), "Cone": (driver,)})
    result = mesh_step(vented_box(tmp_path / "box.step"), settings, tmp_path / "out")

    assert [m.surface_groups for m in result.meshes] == [
        ("Cone", "Interface 1 Port", "walls"),
        ("Interface 1 Port", "Sides", "walls"),
    ]
    sides = read(result, "exterior")
    assert (
        len(
            np.unique(
                sides.cell_data_dict["gmsh:geometrical"]["triangle"][
                    sides.cell_data_dict["gmsh:physical"]["triangle"] == sides.field_data["Sides"][0]
                ]
            )
        )
        == 2
    )


def test_placeholder_step_face_names_are_ignored(tmp_path):
    step = vented_box(tmp_path / "box.step", face_names=lambda centre: "NONE")

    geometry = inspect_step(step)
    assert all(face.step_name is None for face in geometry.faces)


def test_face_group_stored_in_cad_axes_survives_an_axis_mapping_change(tmp_path):
    step = vented_box(tmp_path / "box.step", cad_axes=("-Y", "+Z"))
    as_written = inspect_step(step)
    # Picked as written: the cabinet front with the port hole lies on CAD Y=0.
    front = next(f for f in as_written.faces if f.bodies == ("Cabinet",) and abs(f.centre[1]) < 1e-6)
    ref = face_ref(front, forward="+Z", up="+Y")

    settings = replace(SETTINGS, forward="-Y", up="+Z", face_groups={"Baffle": (ref,)})
    result = mesh_step(step, settings, tmp_path / "out")

    assert [m.surface_groups for m in result.meshes] == [
        ("Interface 1 Port", "walls"),
        ("Baffle", "Interface 1 Port", "walls"),
    ]
    bem = read(result, "exterior")
    baffle = bem.cells_dict["triangle"][bem.cell_data_dict["gmsh:physical"]["triangle"] == bem.field_data["Baffle"][0]]
    assert np.allclose(bem.points[baffle][..., 2], 0, atol=1e-9)  # now the +Z front


def test_faces_sharing_a_step_name_can_be_split_across_face_groups(tmp_path):
    step = vented_box(tmp_path / "box.step", face_names=named_sides)
    sides = [f for f in inspect_step(step).faces if f.step_name == "Side"]
    left, right = sorted((f for f in sides if abs(abs(f.normal[0]) - 1) < 1e-9), key=lambda f: f.centre[0])

    settings = replace(SETTINGS, face_groups={"Left": (face_ref(left),), "Right": (face_ref(right),)})
    result = mesh_step(step, settings, tmp_path / "out")

    assert len(sides) == 4
    assert [m.surface_groups for m in result.meshes] == [
        ("Driver", "Interface 1 Port", "walls"),
        ("Interface 1 Port", "Left", "Right", "Side", "walls"),
    ]
    bem = read(result, "exterior")
    physical, geometrical = (bem.cell_data_dict[k]["triangle"] for k in ("gmsh:physical", "gmsh:geometrical"))
    assert set(geometrical[physical == bem.field_data["Left"][0]]) == {left.id}
    assert len(set(geometrical[physical == bem.field_data["Side"][0]])) == 2


def test_face_group_whose_faces_no_longer_match_is_rejected(vented, tmp_path):
    lost = CadFaceRef(None, (1.0, 1.0, 1.0), 0.5, (0.0, 0.0, 1.0), "Plane")

    with pytest.raises(ValueError, match="Face groups need review .*: 'Lid'"):
        mesh_step(vented_box(tmp_path / "box.step"), replace(SETTINGS, face_groups={"Lid": (lost,)}), tmp_path / "out")


BBOX = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))  # 1 m diagonal


def plane(face_id, centre, normal=(0.0, 0.0, 1.0)):
    return CadFace(face_id, ("Body",), None, centre, 0.01, normal, "Plane")


def test_fingerprint_matches_a_slightly_moved_face():
    ref = CadFaceRef(None, (0.5, 0.0, 0.0), 0.01, (0.0, 0.0, 1.0), "Plane")

    assert match_face_ref(ref, [plane(1, (0.505, 0.0, 0.0)), plane(2, (0.2, 0.0, 0.0))], BBOX) == 1


def test_fingerprint_matching_several_identical_nearby_faces_is_ambiguous():
    ref = CadFaceRef(None, (0.5, 0.0, 0.0), 0.01, (0.0, 0.0, 1.0), "Plane")

    assert match_face_ref(ref, [plane(1, (0.5, 0.0, 0.0)), plane(2, (0.506, 0.0, 0.0))], BBOX) is None


def test_fingerprint_does_not_match_a_face_with_a_flipped_normal():
    ref = CadFaceRef(None, (0.5, 0.0, 0.0), 0.01, (0.0, 0.0, 1.0), "Plane")

    assert match_face_ref(ref, [plane(1, (0.5, 0.0, 0.0), normal=(0.0, 0.0, -1.0))], BBOX) is None


def test_body_override_coarsens_one_air_body_only(vented, tmp_path):
    settings = replace(SETTINGS, body_sizes_m={"Chamber": 0.04})
    result = mesh_step(vented_box(tmp_path / "box.step"), settings, tmp_path / "out")

    def chamber_median(result):
        return np.median(edge_lengths(read(result, "chamber"), "tetra", group="Chamber"))

    assert chamber_median(result) > 1.15 * chamber_median(vented)
    # The nearby cabinet keeps the target size.
    assert 0.016 < np.median(edge_lengths(read(result, "exterior"), group="walls")) < 0.022


@pytest.mark.parametrize("symmetry", ["x", "xy"])
def test_reduced_model_suggests_symmetry_and_meshes_conformally(tmp_path, symmetry):
    step = vented_box(tmp_path / "box.step", symmetry=symmetry)
    result = mesh_step(step, replace(SETTINGS, symmetry=symmetry), tmp_path / "out")
    fem, bem = read(result, "chamber"), read(result, "exterior")

    assert result.geometry.symmetry == symmetry
    assert result.geometry.warnings == ()
    assert result.geometry.bbox_m[0][0] == pytest.approx(0, abs=1e-6)
    validate_conforming_interfaces(
        fem, bem, fem_interface_name="Interface 1 Port", bem_interface_name="Interface 1 Port", symmetry_mode=symmetry
    )
    # Signed volume about the origin still sees the open half/quarter shell as outward.
    full = 0.12 * 0.12 * 0.21
    assert signed_volume(bem) == pytest.approx(full / {"x": 2, "xy": 4}[symmetry], rel=1e-3)


def test_symmetry_mismatch_rejects_generation(tmp_path):
    step = vented_box(tmp_path / "box.step", symmetry="x")

    with pytest.raises(ValueError, match="Project symmetry Off does not match the geometry, which suggests symmetry X"):
        mesh_step(step, replace(SETTINGS, symmetry="off"), tmp_path / "out")
    with pytest.raises(ValueError, match="Project symmetry XY .* suggests symmetry Off"):
        mesh_step(vented_box(tmp_path / "full.step"), replace(SETTINGS, symmetry="xy"), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_axis_mapping_turns_cad_forward_and_up_into_plus_z_and_plus_y(tmp_path):
    # Written with the front facing CAD -Y and CAD +Z up, as many CAD tools do.
    step = vented_box(tmp_path / "box.step", cad_axes=("-Y", "+Z"))

    assert np.allclose(inspect_step(step).bbox_m, [(-0.06, 0, -0.06), (0.06, 0.21, 0.06)], atol=1e-6)
    geometry = inspect_step(step, forward="-Y", up="+Z")
    assert np.allclose(geometry.bbox_m, [(-0.06, -0.06, -0.21), (0.06, 0.06, 0)], atol=1e-6)
    (opening,) = geometry.interfaces
    (mouth,) = [face for face in geometry.faces if face.id in opening.face_ids]
    assert mouth.surface_type == "Plane"
    assert np.allclose(mouth.centre, (0, 0, 0), atol=1e-9)
    assert np.allclose(np.abs(mouth.normal), (0, 0, 1))


def test_placement_rotates_about_x_y_z_then_translates_and_keeps_the_interface_conformal(tmp_path):
    step = vented_box(tmp_path / "box.step")
    driver = next(f for f in inspect_step(step).faces if f.step_name == "Driver")
    # Rotate X 90 takes the driver at (0, 0, -0.2) to (0, 0.2, 0); Z 90 then to (-0.2, 0, 0).
    settings = replace(SETTINGS, rotation_deg=(90.0, 0.0, 90.0), translation_m=(0.01, 0.02, 0.03))

    result = mesh_step(step, settings, tmp_path / "out")

    moved = next(f for f in result.geometry.faces if f.step_name == "Driver")
    assert np.allclose(driver.centre, (0, 0, -0.2), atol=1e-9)
    assert np.allclose(moved.centre, (-0.19, 0.02, 0.03), atol=1e-9)
    assert np.allclose(np.abs(moved.normal), (1, 0, 0), atol=1e-9)
    fem, bem = read(result, "chamber"), read(result, "exterior")
    mouth = bem.cells_dict["triangle"][
        bem.cell_data_dict["gmsh:physical"]["triangle"] == bem.field_data["Interface 1 Port"][0]
    ]
    assert np.allclose(bem.points[mouth].reshape(-1, 3)[:, 0], 0.01, atol=1e-9)  # the mouth plane z=0 is now x=0.01
    validate_conforming_interfaces(
        fem, bem, fem_interface_name="Interface 1 Port", bem_interface_name="Interface 1 Port"
    )
    assert signed_volume(bem) == pytest.approx(0.12 * 0.12 * 0.21, rel=1e-3)


def test_face_group_stored_before_a_placement_survives_it(tmp_path):
    step = vented_box(tmp_path / "box.step")
    back = next(f for f in inspect_step(step).faces if f.bodies == ("Cabinet",) and f.centre[2] < -0.2)
    placement = dict(rotation_deg=(0.0, 30.0, 0.0), translation_m=(0.0, 0.0, 0.05))
    settings = replace(SETTINGS, face_groups={"Back": (face_ref(back),)}, **placement)

    result = mesh_step(step, settings, tmp_path / "out")

    (placed,) = [f for f in result.geometry.faces if f.id in face_ids(result, "exterior", "Back")]
    assert not np.allclose(placed.centre, back.centre, atol=1e-3)
    ref = face_ref(placed, **placement)  # stored in CAD axes: as before the placement
    assert np.allclose(ref.centre, back.centre, atol=1e-9) and np.allclose(ref.normal, back.normal, atol=1e-9)


def face_ids(result, mesh_id, group):
    mesh = read(result, mesh_id)
    physical, geometrical = (mesh.cell_data_dict[k]["triangle"] for k in ("gmsh:physical", "gmsh:geometrical"))
    return set(geometrical[physical == mesh.field_data[group][0]].tolist())


def test_symmetry_is_detected_after_a_translation_puts_the_open_edges_on_x_0(tmp_path):
    step = vented_box(tmp_path / "box.step", symmetry="x", shift_mm=(30, 0, 0))
    assert inspect_step(step).symmetry == "off"

    result = mesh_step(step, replace(SETTINGS, symmetry="x", translation_m=(-0.03, 0.0, 0.0)), tmp_path / "out")

    assert result.geometry.symmetry == "x" and result.geometry.warnings == ()
    assert result.geometry.bbox_m[0][0] == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"rotation_deg": (0.0, float("nan"), 0.0)}, "Rotation must be three finite numbers"),
        ({"translation_m": (0.0, 0.0)}, "Translation must be three finite numbers"),
    ],
)
def test_placement_must_be_three_finite_numbers(changes, message):
    with pytest.raises(ValueError, match=message):
        replace(SETTINGS, **changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (lambda g: {"body_sizes_m": {"Nope": 0.01}}, "Size override for unknown body 'Nope'"),
        (lambda g: {"face_groups": {"Lid": (9999,)}}, "Face group 'Lid' refers to unknown face 9999"),
        (lambda g: {"face_groups": {"walls": ()}}, "Face group name 'walls' is reserved"),
        (
            lambda g: {"interface_names": {"Interface 1 Port": "walls"}},
            "Interface 'Interface 1 Port' cannot be named 'walls'",
        ),
        (
            lambda g: {"interface_names": {"Interface 1 Port": "Mouth"}, "face_groups": {"Mouth": (9999,)}},
            "Face group name 'Mouth' is reserved",
        ),
    ],
)
def test_invalid_face_groups_and_overrides_are_rejected(vented, tmp_path, changes, message):
    step = vented_box(tmp_path / "box.step")

    with pytest.raises(ValueError, match=message):
        mesh_step(step, replace(SETTINGS, **changes(vented.geometry)), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_interface_faces_stay_in_their_renamed_interface_face_group(vented, tmp_path):
    step = vented_box(tmp_path / "box.step")
    (interface,) = vented.geometry.interfaces
    settings = replace(
        SETTINGS,
        face_groups={"Mouth": interface.face_ids},  # grouped before the interface was detected
        interface_names={interface.name: "Port mouth", "Interface 2 Gone": "Ignored"},
    )

    result = mesh_step(step, settings, tmp_path / "out")

    assert [(i.name, i.face_ids) for i in result.geometry.interfaces] == [("Port mouth", interface.face_ids)]
    assert [m.surface_groups for m in result.meshes] == [("Driver", "Port mouth", "walls"), ("Port mouth", "walls")]


def test_missing_step_file_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="STEP file not found"):
        mesh_step(tmp_path / "missing.step", SETTINGS, tmp_path / "out")


def test_axes_must_be_perpendicular():
    with pytest.raises(ValueError, match=r"Forward axis \+Z and up axis -Z must be perpendicular"):
        replace(SETTINGS, up="-Z")


def test_faces_carry_step_names_and_fingerprints(vented):
    (driver,) = [face for face in vented.geometry.faces if face.step_name == "Driver"]

    assert driver.bodies == ("Chamber",)
    assert driver.surface_type == "Plane"
    assert driver.area == pytest.approx(0.1 * 0.1)
    assert np.allclose(driver.centre, (0, 0, -0.2))
    assert np.allclose(np.abs(driver.normal), (0, 0, 1))
    assert {body.name: body.kind for body in vented.geometry.bodies} == {
        "Chamber": "air",
        "Port": "air",
        "Cabinet": "exterior",
    }


def test_unnamed_bodies_get_stable_default_names(tmp_path):
    step = strip_names(vented_box(tmp_path / "named.step"), tmp_path / "unnamed.step")

    first = inspect_step(step)
    assert first == inspect_step(step)
    assert [(b.name, b.kind) for b in first.bodies] == [
        ("Air body 1", "air"),
        ("Air body 2", "air"),
        ("Exterior surface 1", "exterior"),
    ]
    assert [(r.mesh_id, r.bodies) for r in first.regions] == [
        ("air_body_1", ("Air body 1", "Air body 2")),
        ("exterior", ("Exterior surface 1",)),
    ]
    assert [i.name for i in first.interfaces] == ["Interface 1 Air body 2"]
    assert all(face.step_name is None for face in first.faces)


def test_exterior_covering_an_air_face_warns_instead_of_creating_an_interface(tmp_path):
    geometry = inspect_step(vented_box(tmp_path / "box.step", covered=True))

    assert geometry.interfaces == ()
    (warning,) = geometry.warnings
    assert "Exterior surface 'Cabinet' covers a face of air body 'Port'" in warning


def test_open_exterior_reports_the_edges_that_do_not_close(tmp_path):
    geometry = inspect_step(vented_box(tmp_path / "box.step", open_back=True))

    assert geometry.symmetry == "off"
    (warning,) = geometry.warnings
    assert warning.startswith("Exterior surface is not closed: 4 open edge(s) near ")
    assert "(-60.0, 0.0, -210.0) mm" in warning


def test_separate_air_bodies_get_separate_meshes_and_interfaces(tmp_path):
    result = mesh_step(two_chambers(tmp_path / "two.step"), SETTINGS, tmp_path / "out")

    assert [(r.mesh_id, r.kind, r.bodies) for r in result.geometry.regions] == [
        ("left", "bounded_air", ("Left", "Left port")),
        ("right", "bounded_air", ("Right", "Right port")),
        ("exterior", "unbounded_air", ("Cabinet",)),
    ]
    assert [(i.name, i.mesh_id) for i in result.geometry.interfaces] == [
        ("Interface 1 Left port", "left"),
        ("Interface 1 Right port", "right"),
    ]
    bem = read(result, "exterior")
    for interface in result.geometry.interfaces:
        validate_conforming_interfaces(
            read(result, interface.mesh_id), bem, fem_interface_name=interface.name, bem_interface_name=interface.name
        )


def boundary_edge_count(mesh, group):
    cells = mesh.cells_dict["triangle"][mesh.cell_data_dict["gmsh:physical"]["triangle"] == mesh.field_data[group][0]]
    edges = np.sort(np.vstack([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.sum(counts == 1))


def test_curvature_refines_curved_edges_beyond_the_target_size(tmp_path):
    # Target size 68.6 mm would put fewer than four elements around the 40 mm radius mouth.
    result = mesh_step(waveguide(tmp_path / "wg.step"), replace(SETTINGS, fmax_hz=1000.0), tmp_path / "out")

    assert [i.name for i in result.geometry.interfaces] == ["Interface 1 Throat"]
    assert boundary_edge_count(read(result, "throat"), "Interface 1 Throat") >= 12


@pytest.fixture(scope="module")
def surface_model(tmp_path_factory):
    """The waveguide written as SolidWorks does: the throat a closed shell, no solid."""
    return waveguide(tmp_path_factory.mktemp("surface_model") / "wg.step", surface_model=True)


def test_surface_model_fixture_holds_no_solid(surface_model):
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.importShapes(str(surface_model), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        assert gmsh.model.getEntities(3) == []
    finally:
        gmsh.finalize()


def test_closed_shell_without_a_choice_stays_exterior(surface_model):
    geometry = inspect_step(surface_model)

    (shell,) = geometry.closed_shells
    assert (shell.name, shell.role) == ("Closed shell 1", "exterior")
    faces = {f.id: f for f in geometry.faces}
    assert sorted(faces[f].surface_type for f in shell.face_ids) == ["Cone", "Plane", "Plane"]
    assert [(r.mesh_id, r.kind) for r in geometry.regions] == [("exterior", "unbounded_air")]
    assert geometry.interfaces == ()
    assert inspect_step(surface_model, closed_shells={"Closed shell 1": "exterior"}) == geometry


def test_closed_shell_chosen_as_air_becomes_a_bounded_region_with_an_interface(surface_model, tmp_path):
    settings = replace(SETTINGS, fmax_hz=1000.0, closed_shells={"Closed shell 1": "air"})
    result = mesh_step(surface_model, settings, tmp_path / "out")
    geometry = result.geometry

    body = next(b for b in geometry.bodies if b.name == "Closed shell 1")
    assert geometry.closed_shells == (CadClosedShell("Closed shell 1", "air", body.face_ids),)
    assert [(r.mesh_id, r.kind, r.bodies) for r in geometry.regions] == [
        ("closed_shell_1", "bounded_air", ("Closed shell 1",)),
        ("exterior", "unbounded_air", ("Exterior surface 1",)),
    ]
    assert [(i.name, i.mesh_id) for i in geometry.interfaces] == [("Interface 1 Closed shell 1", "closed_shell_1")]
    report = validate_conforming_interfaces(
        read(result, "closed_shell_1"),
        read(result, "exterior"),
        fem_interface_name="Interface 1 Closed shell 1",
        bem_interface_name="Interface 1 Closed shell 1",
    )
    assert report.interface_triangles > 0 and report.bem_boundary_edges == 0


@pytest.mark.parametrize(
    ("closed_shells", "message"),
    [
        ({"Closed shell 1": "solid"}, "Closed shell 'Closed shell 1' must be 'air' or 'exterior'"),
        ({"Closed shell 7": "air"}, "Unknown closed shell 'Closed shell 7'"),
    ],
)
def test_invalid_closed_shell_choices_are_rejected(surface_model, closed_shells, message):
    with pytest.raises(ValueError, match=message):
        inspect_step(surface_model, closed_shells=closed_shells)


def test_cli_sets_closed_shell_roles(surface_model, tmp_path, capsys):
    cad_cli.main(
        [str(surface_model), "--fmax", "1000", "--output", str(tmp_path), "--closed-shell", "Closed shell 1=air"],
        prog="blab cad-import",
    )

    output = capsys.readouterr().out
    assert "Closed shell: Closed shell 1 (air)" in output
    assert "Interface: Interface 1 Closed shell 1 (closed_shell_1 <-> exterior)" in output
    with pytest.raises(SystemExit):
        cad_cli.main([str(surface_model), "--fmax", "1000", "--output", str(tmp_path), "--closed-shell", "air"])
    assert "NAME=air or NAME=exterior" in capsys.readouterr().err


def test_cli_writes_the_same_meshes_as_the_pipeline(vented, tmp_path, capsys):
    step = vented_box(tmp_path / "box.step")

    cad_cli.main(
        [str(step), "--fmax", "3430", "--elements-per-wavelength", "5", "--output", str(tmp_path / "cli")],
        prog="blab cad-import",
    )

    for mesh in vented.meshes:
        assert (tmp_path / "cli" / mesh.path.name).read_bytes() == mesh.path.read_bytes()
    output = capsys.readouterr().out
    assert "Target element size: 20.0 mm" in output
    assert (
        "chamber.msh (fem_volume): volume groups Chamber, Port; surface groups Driver, Interface 1 Port, walls"
        in output
    )
    assert "Interface: Interface 1 Port (chamber <-> exterior)" in output
    assert "Suggested symmetry: Off" in output


def test_cli_places_the_geometry_with_rotate_and_translate(tmp_path, capsys):
    step = vented_box(tmp_path / "box.step", symmetry="x", shift_mm=(30, 0, 0))
    args = [str(step), "--fmax", "1000", "--output", str(tmp_path / "out")]

    cad_cli.main([*args, "--rotate", "180,0,0", "--translate=-30,0,0"], prog="blab cad-import")

    output = capsys.readouterr().out
    assert "Bounding box: 60.0 x 120.0 x 210.0 mm" in output
    assert "Suggested symmetry: X" in output  # the open edges at x=30 mm moved onto X=0
    with pytest.raises(SystemExit):
        cad_cli.main([*args, "--rotate", "1,2"])
    assert "--rotate takes X,Y,Z" in capsys.readouterr().err


def test_cli_reports_rejections_as_usage_errors(tmp_path, capsys):
    with pytest.raises(SystemExit):
        cad_cli.main([str(tmp_path / "missing.step"), "--fmax", "1000", "--output", str(tmp_path)])
    assert "STEP file not found" in capsys.readouterr().err


def signed_volume(mesh):
    a, b, c = mesh.points[mesh.cells_dict["triangle"]].transpose(1, 0, 2)
    return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6


def test_port_mouth_is_a_node_for_node_interface_on_a_closed_outward_exterior(vented):
    meshes = {mesh.id: meshio.read(mesh.path) for mesh in vented.meshes}

    assert [(i.name, i.mesh_id) for i in vented.geometry.interfaces] == [("Interface 1 Port", "chamber")]
    report = validate_conforming_interfaces(
        meshes["chamber"],
        meshes["exterior"],
        fem_interface_name="Interface 1 Port",
        bem_interface_name="Interface 1 Port",
    )
    assert report.interface_triangles > 0 and report.bem_boundary_edges == 0
    # Cabinet 120 x 120 x 210 mm, outward normals.
    assert signed_volume(meshes["exterior"]) == pytest.approx(0.12 * 0.12 * 0.21, rel=1e-3)
    assert vented.geometry.symmetry == "off"
    assert vented.geometry.warnings == ()


def test_closed_shell_without_a_choice_opens_as_a_quick_scan_only(surface_model):
    scan = open_step(surface_model)

    assert isinstance(scan, CadScan)
    (shell,) = scan.closed_shells
    assert (shell.name, shell.role) == ("Closed shell 1", "exterior")
    assert set(shell.face_ids) < set(scan.face_ids.tolist())  # the other faces are shown around it
    assert scan.triangles.max() < len(scan.points)
    preview = open_step(surface_model, closed_shells={"Closed shell 1": "air"})
    assert isinstance(preview, CadPreview)
    assert preview.geometry == inspect_step(surface_model, closed_shells={"Closed shell 1": "air"})


# Placements that keep the cabinet front, which the port mouth opens in, off X=0 and Y=0.
@pytest.mark.parametrize("axes", [("-Z", "+X"), ("+Z", "-Y")])
def test_placing_a_preview_reports_what_reading_it_with_the_placement_does(tmp_path, axes):
    step = vented_box(tmp_path / "box.step", open_back=True)
    forward, up = axes

    placed = place_preview(open_step(step), forward=forward, up=up)
    read = open_step(step, forward=forward, up=up)

    assert placed.axes == read.axes == axes
    # Placed: the faces' extent; read: OpenCASCADE's box, padded by its 1e-7 m tolerance.
    assert np.allclose(placed.geometry.bbox_m, read.geometry.bbox_m, atol=1e-6)
    assert np.allclose(placed.points.min(axis=0), read.points.min(axis=0), atol=1e-9)
    assert placed.geometry.symmetry == read.geometry.symmetry
    assert unsigned(placed.geometry.warnings) == unsigned(read.geometry.warnings)
    assert "Exterior surface is not closed" in placed.geometry.warnings[0]
    assert fingerprints(placed.geometry) == fingerprints(read.geometry)
    assert [(b.name, len(b.face_ids)) for b in placed.geometry.bodies] == [
        (b.name, len(b.face_ids)) for b in read.geometry.bodies
    ]
    assert placed.geometry.regions == read.geometry.regions


def test_placing_a_preview_with_rotation_and_translation_reports_what_reading_it_does(tmp_path):
    step = vented_box(tmp_path / "box.step", open_back=True)
    placement = dict(forward="-Z", up="+X", rotation_deg=(10.0, -20.0, 30.0), translation_m=(0.05, -0.02, 0.1))

    placed = place_preview(open_step(step, rotation_deg=(0.0, 0.0, 45.0)), **placement)
    read = open_step(step, **placement)

    assert (placed.axes, placed.rotation_deg, placed.translation_m) == (
        read.axes,
        read.rotation_deg,
        read.translation_m,
    )
    assert np.allclose(placed.points.min(axis=0), read.points.min(axis=0), atol=1e-9)
    assert np.allclose(placed.geometry.bbox_m, read.geometry.bbox_m, atol=1e-6)  # not a turned box around it
    assert placed.geometry.symmetry == read.geometry.symmetry == "off"
    assert unsigned(placed.geometry.warnings) == unsigned(read.geometry.warnings)
    assert fingerprints(placed.geometry) == fingerprints(read.geometry)


def fingerprints(geometry):
    """The faces by name, type, centre (mm) and normal, rounded: face ids differ between reads."""
    return sorted(
        (f.step_name or "", f.surface_type, *(np.round(np.array(f.centre) * 1000, 3) + 0), *(np.round(f.normal, 6) + 0))
        for f in geometry.faces
    )


def unsigned(warnings):
    """``warnings`` with -0.0 written as 0.0: rounding noise about zero."""
    return [re.sub(r"-(0\.0)\b", r"\1", w) for w in warnings]


def test_placement_moves_the_symmetry_planes_the_open_edges_suggest(tmp_path):
    preview = open_step(vented_box(tmp_path / "box.step", symmetry="x"))
    assert preview.geometry.symmetry == "x"

    turned = place_preview(preview, forward="+Z", up="+X")  # CAD X=0 becomes Y=0, no symmetry alone

    assert turned.geometry.symmetry == "off"
    assert turned.geometry.warnings[0].startswith("Exterior surface is not closed")
    back = place_preview(turned, forward="+Z", up="+Y")
    assert back.geometry.symmetry == "x" and back.geometry.warnings == ()
    with pytest.raises(ValueError, match="perpendicular"):
        place_preview(preview, forward="+Z", up="-Z")


def test_previews_survive_their_json_form(surface_model):
    for preview in (open_step(surface_model), open_step(surface_model, closed_shells={"Closed shell 1": "air"})):
        copy = preview_from_json(json.loads(json.dumps(preview_to_json(preview))))

        assert type(copy) is type(preview)
        for name, value in vars(preview).items():
            if isinstance(value, np.ndarray):
                assert np.array_equal(getattr(copy, name), value)
            else:
                assert getattr(copy, name) == value
