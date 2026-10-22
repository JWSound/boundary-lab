"""Headless CAD import: STEP file in, conformal mesh assembly out."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import gmsh
import meshio
import numpy as np
import pytest

from blab.cad_import import (
    LEVELS,
    CadClosedShell,
    CadFace,
    CadFaceGroups,
    CadFaceRef,
    CadImportSettings,
    CadPreview,
    CadSizeOverride,
    CadSizes,
    face_ref,
    inspect_step,
    match_face_ref,
    mesh_step,
    open_step,
    place_preview,
    preview_from_json,
    preview_to_json,
    result_from_json,
    result_to_json,
)
from blab.cad_import import cli as cad_cli
from blab.cad_import.sizing import legacy_sizes, migrated, read_face_sizes
from blab.interface_conform import validate_conforming_interfaces
from cad_fixtures import bored_ball, named_sides, strip_names, two_chambers, vented_box, waveguide

# Target element size 343 / 3430 / 5 = 20 mm.
SETTINGS = CadImportSettings(fmax_hz=3430.0, sound_speed_m_per_s=343.0, elements_per_wavelength=5.0)
# Rotation taking a model written with its front along CAD -Y and CAD +Z up onto +Z on axis and +Y vertical.
FRONT_ON_AXIS = (-90.0, 0.0, 0.0)


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


def test_new_settings_default_to_fine_six_elements_per_wavelength():
    assert CadImportSettings(fmax_hz=1000.0).elements_per_wavelength == 6


# An edge length, or a level: 343 / 3430 / 12 = 8.3 mm.
@pytest.mark.parametrize("size", [CadSizeOverride(size_m=0.008), CadSizeOverride(elements_per_wavelength=12)])
def test_step_named_face_group_takes_its_own_size(vented, tmp_path, size):
    settings = replace(SETTINGS, face_group_sizes={"Driver": size})
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


def test_face_group_stored_in_cad_axes_survives_a_rotation_change(tmp_path):
    step = vented_box(tmp_path / "box.step", cad_axes=("-Y", "+Z"))
    as_written = inspect_step(step)
    # Picked as written: the cabinet front with the port hole lies on CAD Y=0.
    front = next(f for f in as_written.faces if f.bodies == ("Cabinet",) and abs(f.centre[1]) < 1e-6)
    ref = face_ref(front)

    settings = replace(SETTINGS, rotation_deg=FRONT_ON_AXIS, face_groups={"Baffle": (ref,)})
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
    settings = replace(SETTINGS, body_sizes={"Chamber": CadSizeOverride(size_m=0.04)})
    result = mesh_step(vented_box(tmp_path / "box.step"), settings, tmp_path / "out")

    def chamber_median(result):
        return np.median(edge_lengths(read(result, "chamber"), "tetra", group="Chamber"))

    assert chamber_median(result) > 1.15 * chamber_median(vented)
    # The nearby cabinet keeps the target size.
    assert 0.016 < np.median(edge_lengths(read(result, "exterior"), group="walls")) < 0.022


def test_volume_size_coarsens_the_interior_only(vented, tmp_path):
    settings = replace(SETTINGS, volume_sizes={"Chamber": CadSizeOverride(elements_per_wavelength=2)})  # 50 mm
    result = mesh_step(vented_box(tmp_path / "box.step"), settings, tmp_path / "out")

    def chamber_edges(result, cell_type, group):
        return edge_lengths(read(result, "chamber"), cell_type, group=group)

    # The growth limit keeps the 100 mm wide interior well below 50 mm, but it holds fewer elements.
    assert len(chamber_edges(result, "tetra", "Chamber")) < 0.85 * len(chamber_edges(vented, "tetra", "Chamber"))
    # Its faces keep the target size.
    walls = np.median(chamber_edges(vented, "triangle", "walls"))
    assert np.median(chamber_edges(result, "triangle", "walls")) == pytest.approx(walls, rel=0.1)


def face_edges_and_tets(result):
    """``({(mesh id, face id): sorted triangle edge lengths}, {mesh id: tet count})`` of a meshing result."""
    faces, tets = {}, {}
    for mesh in result.meshes:
        data = meshio.read(mesh.path)
        triangles = data.cells_dict["triangle"]
        tags = data.cell_data_dict["gmsh:geometrical"]["triangle"]
        for face in np.unique(tags):
            corners = data.points[triangles[tags == face]]
            faces[mesh.id, int(face)] = np.sort(np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2).ravel())
        tets[mesh.id] = len(data.cells_dict.get("tetra", ()))
    return faces, tets


LEGACY = {
    "body_sizes_m": {"Chamber": 0.04},  # coarser: its faces shared with no other body only
    "face_group_sizes_m": {"Driver": 0.008},
    "body_sizes": {"Port": {"size_m": 0.03}, "Cabinet": {"size_m": 0.01}},
    "face_group_sizes": {"walls": {"elements_per_wavelength": 2}},
}


@pytest.mark.parametrize(
    ("covered", "settings", "legacy"),
    [
        (False, SETTINGS, LEGACY),
        # The covered port mouth is on the Port and the Cabinet: its 20 mm BEM target is finer than the Port's size.
        (
            True,
            replace(SETTINGS, elements_per_wavelength=2.0, bem_elements_per_wavelength=5.0),
            {"body_sizes": {"Port": {"size_m": 0.03}}},
        ),
    ],
)
def test_a_migrated_legacy_design_meshes_like_the_legacy_design(tmp_path, covered, settings, legacy):
    step = vented_box(tmp_path / "box.step", covered=covered)
    geometry = inspect_step(step)
    new = migrated(legacy, CadFaceGroups(geometry, {}), CadSizes(geometry, [], {}), settings.sizing)
    body_sizes, face_group_sizes = legacy_sizes(legacy)

    before = mesh_step(
        step, replace(settings, body_sizes=body_sizes, face_group_sizes=face_group_sizes), tmp_path / "legacy"
    )
    after = mesh_step(
        step,
        replace(settings, face_sizes=tuple(read_face_sizes(new["face_sizes"])), volume_sizes=new["volume_sizes"]),
        tmp_path / "migrated",
    )

    (faces_before, tets_before), (faces_after, tets_after) = face_edges_and_tets(before), face_edges_and_tets(after)
    assert faces_after.keys() == faces_before.keys()
    for face, edges in faces_before.items():
        assert faces_after[face].shape == edges.shape and np.allclose(faces_after[face], edges), face
    assert tets_after == tets_before


def face_edge_median(mesh, face_id):
    """Median edge length (m) of the triangles meshing CAD face ``face_id``."""
    cells = mesh.cells_dict["triangle"][mesh.cell_data_dict["gmsh:geometrical"]["triangle"] == face_id]
    corners = mesh.points[cells]
    return np.median(np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2))


def test_face_override_wins_over_its_face_group_and_one_no_longer_found_is_ignored(vented, tmp_path):
    step = vented_box(tmp_path / "box.step")
    sides = [f for f in vented.geometry.faces if f.bodies == ("Cabinet",) and abs(abs(f.normal[0]) - 1) < 1e-9]
    lost = CadFaceRef(None, (1.0, 1.0, 1.0), 0.5, (0.0, 0.0, 1.0), "Plane")
    settings = replace(
        SETTINGS,
        face_groups={"Sides": tuple(face_ref(f) for f in sides)},
        face_group_sizes={"Sides": CadSizeOverride(size_m=0.008)},
        face_sizes=((face_ref(sides[0]), CadSizeOverride(size_m=0.004)), (lost, CadSizeOverride(size_m=0.001))),
    )

    result = mesh_step(step, settings, tmp_path / "out")

    bem = read(result, "exterior")
    (own,) = [f.id for f in result.geometry.faces if np.allclose(f.centre, sides[0].centre)]
    (grouped,) = [f.id for f in result.geometry.faces if np.allclose(f.centre, sides[1].centre)]
    assert 0.003 < face_edge_median(bem, own) < 0.0055  # the face's own override, not its group's
    assert 0.0065 < face_edge_median(bem, grouped) < 0.0095  # the group's


def test_a_face_sized_bem_fine_meshes_at_twelve_elements_per_wavelength(vented, tmp_path):
    side = next(f for f in vented.geometry.faces if f.bodies == ("Cabinet",) and abs(abs(f.normal[0]) - 1) < 1e-9)
    bem_fine = CadSizeOverride(elements_per_wavelength=LEVELS["BEM Fine"])
    settings = replace(SETTINGS, face_sizes=((face_ref(side), bem_fine),))

    result = mesh_step(vented_box(tmp_path / "box.step"), settings, tmp_path / "out")

    (face,) = [f.id for f in result.geometry.faces if np.allclose(f.centre, side.centre)]
    assert 0.0065 < face_edge_median(read(result, "exterior"), face) < 0.0095  # 343 / 3430 / 12 = 8.3 mm


def test_exterior_faces_mesh_at_the_bem_size_air_bodies_at_the_fem_size_and_interfaces_at_the_finer(tmp_path):
    # 343 / 6860 = 50 mm per wavelength: FEM 5 gives 10 mm, BEM 2 gives 25 mm.
    settings = CadImportSettings(
        fmax_hz=6860.0, sound_speed_m_per_s=343.0, elements_per_wavelength=5.0, bem_elements_per_wavelength=2.0
    )

    result = mesh_step(vented_box(tmp_path / "box.step", port_radius=40), settings, tmp_path / "out")

    chamber, exterior = read(result, "chamber"), read(result, "exterior")
    (mouth,) = result.geometry.interfaces[0].face_ids  # an 80 mm disc
    assert 0.02 < np.median(edge_lengths(exterior, group="walls")) < 0.03
    assert 0.008 < np.median(edge_lengths(chamber, group="walls")) < 0.011
    assert 0.008 < np.median(edge_lengths(chamber, "tetra")) < 0.013
    assert 0.008 < face_edge_median(exterior, mouth) < 0.011
    assert result.target_size_m == pytest.approx(0.01)


def test_overrides_replace_the_faces_own_target_size_and_a_volume_one_sizes_the_interior_only(vented, tmp_path):
    side = next(f for f in vented.geometry.faces if f.bodies == ("Cabinet",) and abs(abs(f.normal[0]) - 1) < 1e-9)
    settings = replace(
        SETTINGS,
        bem_elements_per_wavelength=2.0,  # 50 mm on the exterior, 20 mm in the air bodies
        face_sizes=((face_ref(side), CadSizeOverride(size_m=0.008)),),
        volume_sizes={"Chamber": CadSizeOverride(elements_per_wavelength=2)},  # 50 mm
    )

    result = mesh_step(vented_box(tmp_path / "box.step"), settings, tmp_path / "out")

    (face,) = [f.id for f in result.geometry.faces if np.allclose(f.centre, side.centre)]
    assert 0.0065 < face_edge_median(read(result, "exterior"), face) < 0.0095
    chamber, before = read(result, "chamber"), read(vented, "chamber")
    assert len(edge_lengths(chamber, "tetra", "Chamber")) < 0.85 * len(edge_lengths(before, "tetra", "Chamber"))
    walls = np.median(edge_lengths(before, "triangle", "walls"))
    assert np.median(edge_lengths(chamber, "triangle", "walls")) == pytest.approx(walls, rel=0.1)


def test_without_a_bem_fineness_the_fem_one_sizes_the_exterior_too(vented, tmp_path):
    settings = replace(SETTINGS, bem_elements_per_wavelength=SETTINGS.elements_per_wavelength)

    result = mesh_step(vented_box(tmp_path / "box.step"), settings, tmp_path / "out")

    assert SETTINGS.bem_elements_per_wavelength is None
    for before, after in zip(vented.meshes, result.meshes, strict=True):
        assert after.path.read_bytes() == before.path.read_bytes()


@pytest.mark.parametrize(
    ("size", "message"),
    [
        (dict(), "Size override must be elements per wavelength or an edge length"),
        (dict(size_m=0.01, elements_per_wavelength=4), "Size override must be elements per wavelength or an edge"),
        (dict(size_m=-1.0), "Size override edge length must be a positive number"),
    ],
)
def test_size_override_is_one_positive_number(size, message):
    with pytest.raises(ValueError, match=message):
        CadSizeOverride(**size)


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


def test_rotation_turns_a_cad_front_and_up_onto_on_axis_and_vertical(tmp_path):
    # Written with the front facing CAD -Y and CAD +Z up, as many CAD tools do.
    step = vented_box(tmp_path / "box.step", cad_axes=("-Y", "+Z"))

    assert np.allclose(inspect_step(step).bbox_m, [(-0.06, 0, -0.06), (0.06, 0.21, 0.06)], atol=1e-6)
    geometry = inspect_step(step, rotation_deg=FRONT_ON_AXIS)
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
        (lambda g: {"body_sizes": {"Nope": {"size_m": 0.01}}}, "Size override for unknown body 'Nope'"),
        (lambda g: {"volume_sizes": {"Nope": {"size_m": 0.01}}}, "Volume size for 'Nope', which is no air body"),
        (lambda g: {"volume_sizes": {"Cabinet": {"size_m": 0.01}}}, "Volume size for 'Cabinet', which is no air body"),
        (lambda g: {"face_sizes": ((9999, {"size_m": 0.01}),)}, "Size override for unknown face 9999"),
        (lambda g: {"face_groups": {"Lid": (9999,)}}, "Face group 'Lid' refers to unknown face 9999"),
        (lambda g: {"face_groups": {"walls": ()}}, "Face group name 'walls' is reserved"),
        (
            lambda g: {"interface_names": {"Interface 1 Port": "walls"}},
            "Face group name 'walls' is taken",
        ),
        (
            lambda g: {"interface_names": {"Interface 1 Port": "Mouth"}, "face_groups": {"Mouth": (9999,)}},
            "Face group name 'Mouth' is taken",
        ),
    ],
)
def test_invalid_face_groups_and_overrides_are_rejected(vented, tmp_path, changes, message):
    step = vented_box(tmp_path / "box.step")

    with pytest.raises(ValueError, match=message):
        mesh_step(step, replace(SETTINGS, **changes(vented.geometry)), tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("name", ["walls", "Driver", "Lid"])
def test_generation_and_the_editor_reject_the_same_interface_names_alike(vented, tmp_path, name):
    (interface,) = vented.geometry.interfaces
    lid = next(f.id for f in vented.geometry.faces if f.bodies == ("Cabinet",))
    with pytest.raises(ValueError) as editor:
        CadFaceGroups(vented.geometry, {"Lid": [lid]}).renamed_interface(interface.name, name)
    settings = replace(SETTINGS, face_groups={"Lid": (lid,)}, interface_names={interface.name: name})

    with pytest.raises(ValueError) as generation:
        mesh_step(vented_box(tmp_path / "box.step"), settings, tmp_path / "out")

    assert str(generation.value) == str(editor.value) == f"Face group name '{name}' is taken."


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


def test_closed_shell_without_a_choice_is_guessed_air_beside_another_exterior_surface(surface_model):
    geometry = inspect_step(surface_model)

    (shell,) = geometry.closed_shells
    assert (shell.name, shell.role) == ("Closed shell 1", "air")
    assert [(r.mesh_id, r.kind) for r in geometry.regions] == [
        ("closed_shell_1", "bounded_air"),
        ("exterior", "unbounded_air"),
    ]
    assert inspect_step(surface_model, closed_shells={"Closed shell 1": "air"}) == geometry


def test_closed_shell_chosen_as_exterior_is_an_exterior_body_of_its_name(surface_model):
    geometry = inspect_step(surface_model, closed_shells={"Closed shell 1": "exterior"})

    (shell,) = geometry.closed_shells
    assert shell.role == "exterior"
    faces = {f.id: f for f in geometry.faces}
    assert sorted(faces[f].surface_type for f in shell.face_ids) == ["Cone", "Plane", "Plane"]
    assert [(b.name, b.kind, b.face_ids) for b in geometry.bodies if b.name == "Closed shell 1"] == [
        ("Closed shell 1", "exterior", shell.face_ids)
    ]
    assert [(r.mesh_id, r.kind) for r in geometry.regions] == [("exterior", "unbounded_air")]
    assert geometry.interfaces == ()


def test_with_only_closed_shells_the_largest_is_guessed_exterior_and_the_others_air(tmp_path):
    step = waveguide(tmp_path / "wg.step", surface_model=True, covered=True)

    geometry = inspect_step(step)

    assert [(s.name, s.role) for s in geometry.closed_shells] == [
        ("Closed shell 1", "air"),  # the throat
        ("Closed shell 2", "exterior"),  # the 200 mm baffle box
    ]
    assert {b.name: b.kind for b in geometry.bodies} == {"Closed shell 1": "air", "Closed shell 2": "exterior"}


MESH_IMPORT = Path(__file__).parent / "fixtures" / "cad_import" / "blab Mesh import.STEP"


@pytest.mark.skipif(not MESH_IMPORT.is_file(), reason="local-only fixture")
def test_surface_model_with_near_coincident_edges_reads_without_degenerate_faces():
    # Fragmented in metres, OpenCASCADE's fixed tolerance collapsed faces of this model to zero area, and finding
    # the normal of one never returned. A subprocess, so that hang fails the test instead of stalling the run.
    code = "import sys; from blab.cad_import import inspect_step; print(min(f.area for f in inspect_step(sys.argv[1]).faces))"
    run = subprocess.run([sys.executable, "-c", code, str(MESH_IMPORT)], capture_output=True, text=True, timeout=120)

    assert run.returncode == 0, run.stderr
    assert float(run.stdout) > 1e-9  # m², a 0.03 mm square


@pytest.mark.skipif(not MESH_IMPORT.is_file(), reason="local-only fixture")
def test_a_face_bounded_only_by_seams_still_has_curves_to_draw():
    preview = open_step(MESH_IMPORT)
    openings = preview.openings

    assert all(openings.face_curves[f] for f in openings.face_edges)
    assert set().union(*openings.face_curves.values()) <= set(openings.edges)  # sampled, so drawable
    assert preview_from_json(json.loads(json.dumps(preview_to_json(preview)))).openings.face_curves == openings.face_curves


@pytest.fixture(scope="module")
def bored_ball_step(tmp_path_factory):
    """A ball whose one torus face Gmsh cannot mesh, inside a cabinet."""
    return bored_ball(tmp_path_factory.mktemp("bored_ball") / "ball.step")


@pytest.fixture(scope="module")
def bored_balls_step(tmp_path_factory):
    """Two balls, each with one torus face Gmsh cannot mesh, inside a cabinet."""
    return bored_ball(tmp_path_factory.mktemp("bored_balls") / "balls.step", balls=2)


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_the_read_finds_every_unmeshable_face_not_just_the_first(bored_balls_step):
    preview = open_step(bored_balls_step)
    tori = [f.id for f in preview.geometry.faces if f.surface_type == "Torus"]

    assert len(tori) == 2
    assert [u.id for u in preview.geometry.unmeshable] == tori
    assert set(preview.face_ids.tolist()) == {f.id for f in preview.geometry.faces} - set(tori)


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_generate_names_every_unmeshable_face_in_face_order(bored_balls_step, tmp_path):
    a, b = [f.id for f in inspect_step(bored_balls_step).faces if f.surface_type == "Torus"]
    reason = "Impossible to mesh periodic surface {}; area reads as negative"

    with pytest.raises(ValueError) as error:
        mesh_step(bored_balls_step, SETTINGS, tmp_path / "out")

    assert str(error.value) == (
        f"Faces {a} (Torus, Ball 1), {b} (Torus, Ball 2) cannot be meshed: "
        f"face {a}: {reason.format(a)}; face {b}: {reason.format(b)}"
    )


def test_unmeshable_face_is_named_and_the_other_faces_are_still_tessellated(bored_ball_step):
    preview = open_step(bored_ball_step)
    geometry = preview.geometry

    (torus,) = [f for f in geometry.faces if f.surface_type == "Torus"]
    (unmeshable,) = geometry.unmeshable
    assert unmeshable.id == torus.id
    assert unmeshable.reason == f"Impossible to mesh periodic surface {torus.id}; area reads as negative"
    assert torus.area < 0  # legal in the faces
    # Its area reads negative, so no fill: the inside test is unreliable on this doubly covered surface.
    assert set(preview.face_ids.tolist()) == {f.id for f in geometry.faces} - {torus.id}
    assert inspect_step(bored_ball_step).unmeshable == geometry.unmeshable


def test_unmeshable_faces_follow_placement_and_survive_json(bored_ball_step, vented):
    preview = open_step(bored_ball_step)
    unmeshable = preview.geometry.unmeshable

    assert place_preview(preview, rotation_deg=(0.0, 0.0, 90.0)).geometry.unmeshable == unmeshable
    assert preview_from_json(json.loads(json.dumps(preview_to_json(preview)))).geometry.unmeshable == unmeshable
    result = replace(vented, geometry=preview.geometry)
    assert result_from_json(json.loads(json.dumps(result_to_json(result)))) == result


def test_generate_rejects_an_unmeshable_face_by_name_and_writes_no_meshes(bored_ball_step, tmp_path):
    (torus,) = [f for f in inspect_step(bored_ball_step).faces if f.surface_type == "Torus"]

    with pytest.raises(ValueError) as error:
        mesh_step(bored_ball_step, SETTINGS, tmp_path / "out")

    assert str(error.value) == (
        f"Face {torus.id} (Torus, Ball) cannot be meshed: "
        f"Impossible to mesh periodic surface {torus.id}; area reads as negative"
    )
    assert not list((tmp_path / "out").glob("*.msh"))


@pytest.mark.parametrize(
    "settings",
    [
        replace(SETTINGS, elements_per_wavelength=6.0, bem_elements_per_wavelength=5.0),
        replace(SETTINGS, elements_per_wavelength=6.0, body_sizes={"Ball": CadSizeOverride(size_m=0.04)}),
    ],
    ids=["default-finenesses", "body-override-fem-only"],
)
@pytest.mark.timeout(30)
def test_generate_rejects_an_unmeshable_face_quickly_when_face_sizes_differ(bored_ball_step, tmp_path, settings):
    with pytest.raises(ValueError, match=r"^Face \d+ \(Torus, Ball\) cannot be meshed: Impossible to mesh periodic"):
        mesh_step(bored_ball_step, settings, tmp_path / "out")

    assert not list((tmp_path / "out").glob("*.msh"))


def test_a_good_file_has_no_unmeshable_faces_and_every_face_is_tessellated(tmp_path):
    preview = open_step(vented_box(tmp_path / "box.step"))

    assert preview.geometry.unmeshable == ()
    assert set(preview.face_ids.tolist()) == {f.id for f in preview.geometry.faces}


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
    assert "FEM target element size: 20.0 mm" in output
    assert "BEM target element size: 20.0 mm" in output  # no BEM fineness: the FEM one
    assert (
        "chamber.msh (fem_volume): volume groups Chamber, Port; surface groups Driver, Interface 1 Port, walls"
        in output
    )
    assert "Interface: Interface 1 Port (chamber <-> exterior)" in output
    assert "Suggested symmetry: Off" in output


def test_cli_takes_a_bem_fineness_and_meshes_as_the_pipeline_does(tmp_path, capsys):
    step = vented_box(tmp_path / "box.step")
    settings = replace(SETTINGS, bem_elements_per_wavelength=2.0)
    expected = mesh_step(step, settings, tmp_path / "pipeline")
    args = ["--fmax", "3430", "--elements-per-wavelength", "5", "--bem-elements-per-wavelength", "2"]

    cad_cli.main([str(step), *args, "--output", str(tmp_path / "cli")], prog="blab cad-import")

    for mesh in expected.meshes:
        assert (tmp_path / "cli" / mesh.path.name).read_bytes() == mesh.path.read_bytes()
    output = capsys.readouterr().out
    assert "FEM target element size: 20.0 mm" in output
    assert "BEM target element size: 50.0 mm" in output


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


def test_cli_names_an_unmeshable_face(bored_ball_step, tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_:
        cad_cli.main([str(bored_ball_step), "--fmax", "1000", "--output", str(tmp_path)])

    assert exit_.value.code != 0
    err = capsys.readouterr().err
    assert "(Torus, Ball) cannot be meshed: Impossible to mesh periodic surface" in err
    assert "Traceback" not in err


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


def test_opening_a_file_with_a_closed_shell_reads_it_fully_with_the_guessed_kind(surface_model):
    preview = open_step(surface_model)

    assert isinstance(preview, CadPreview)
    assert preview.geometry == inspect_step(surface_model)
    assert preview.triangles.max() < len(preview.points)


def test_placing_a_preview_re_detects_the_interfaces(tmp_path):
    # A half model written 30 mm off X=0: only moved back onto X=0 does the half port mouth open into the cabinet.
    step = vented_box(tmp_path / "box.step", symmetry="x", shift_mm=(30, 0, 0))
    placement = dict(translation_m=(-0.03, 0.0, 0.0))
    preview = open_step(step)

    placed = place_preview(preview, **placement)
    read = open_step(step, **placement)

    assert interfaces(placed.geometry) == interfaces(read.geometry) == [("Interface 1 Port", "chamber", 1)]
    assert interfaces(preview.geometry) != interfaces(read.geometry)
    assert interfaces(place_preview(placed).geometry) == interfaces(preview.geometry)


def interfaces(geometry):
    return [(i.name, i.mesh_id, len(i.face_ids)) for i in geometry.interfaces]


# Placements that keep the cabinet front, which the port mouth opens in, off X=0 and Y=0.
@pytest.mark.parametrize("rotation", [(180.0, 0.0, 90.0), (0.0, 0.0, 180.0)])
def test_placing_a_preview_reports_what_reading_it_with_the_placement_does(tmp_path, rotation):
    step = vented_box(tmp_path / "box.step", open_back=True)

    placed = place_preview(open_step(step), rotation_deg=rotation)
    read = open_step(step, rotation_deg=rotation)

    assert placed.rotation_deg == read.rotation_deg == rotation
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
    placement = dict(rotation_deg=(190.0, -20.0, 120.0), translation_m=(0.05, -0.02, 0.1))

    placed = place_preview(open_step(step, rotation_deg=(0.0, 0.0, 45.0)), **placement)
    read = open_step(step, **placement)

    assert (placed.rotation_deg, placed.translation_m) == (read.rotation_deg, read.translation_m)
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

    turned = place_preview(preview, rotation_deg=(0.0, 0.0, 90.0))  # CAD X=0 becomes Y=0, no symmetry alone

    assert turned.geometry.symmetry == "off"
    assert turned.geometry.warnings[0].startswith("Exterior surface is not closed")
    back = place_preview(turned)
    assert back.geometry.symmetry == "x" and back.geometry.warnings == ()
    with pytest.raises(ValueError, match="Rotation must be three finite numbers"):
        place_preview(preview, rotation_deg=(0.0, 0.0))


def test_previews_survive_their_json_form(surface_model):
    preview = open_step(surface_model)
    copy = preview_from_json(json.loads(json.dumps(preview_to_json(preview))))

    for name, value in vars(preview).items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(getattr(copy, name), value)
        elif name != "openings":
            assert getattr(copy, name) == value
    # What finding the openings needs survives too: placing the copy finds the same.
    moved = dict(rotation_deg=(0.0, 0.0, 90.0))
    assert place_preview(copy, **moved).geometry == place_preview(preview, **moved).geometry


def test_meshing_results_survive_their_json_form(vented):
    assert result_from_json(json.loads(json.dumps(result_to_json(vented)))) == vented
