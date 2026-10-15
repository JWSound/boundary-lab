"""A CAD import design on a read geometry: face groups, walls and interface face groups, without Qt or meshing."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from blab.cad_import import CadFaceGroups, CadFaceRef, face_ref, inspect_step
from cad_fixtures import named_sides, vented_box


@pytest.fixture(scope="module")
def box(tmp_path_factory):
    return inspect_step(vented_box(tmp_path_factory.mktemp("box") / "box.step"))


def x_sides(geometry):
    """The cabinet's two X sides, left first."""
    sides = [f for f in geometry.faces if f.bodies == ("Cabinet",) and abs(abs(f.normal[0]) - 1) < 1e-9]
    return sorted(sides, key=lambda f: f.centre[0])


def rows(groups):
    return [(row.name, len(row.faces), row.kind) for row in groups.rows]


def test_walls_come_first_then_the_interfaces_then_the_step_named_groups(box):
    groups = CadFaceGroups(box, {})
    (interface,) = box.interfaces
    driver = next(f.id for f in box.faces if f.step_name == "Driver")

    assert rows(groups) == [
        ("walls", len(box.faces) - len(interface.face_ids) - 1, "walls"),
        ("Interface 1 Port", len(interface.face_ids), "interface"),
        ("Driver", 1, "group"),
    ]
    assert groups.group_of(driver).name == "Driver"
    assert groups.row("walls").faces == {f.id for f in box.faces} - {driver, *interface.face_ids}
    assert groups.needs_review == ()
    groups.check()


def test_stored_groups_come_before_step_named_ones_and_take_their_faces(box):
    left, right = x_sides(box)
    driver = next(f for f in box.faces if f.step_name == "Driver")

    groups = CadFaceGroups(box, {"Sides": [asdict(face_ref(left)), face_ref(right)], "Cone": [driver.id]})

    assert [row.name for row in groups.rows] == ["walls", "Interface 1 Port", "Sides", "Cone"]
    assert groups.row("Sides").faces == {left.id, right.id}
    assert groups.row("Driver") is None  # its only face is in Cone


def test_faces_sharing_a_step_name_split_between_a_stored_group_and_their_name(tmp_path):
    geometry = inspect_step(vented_box(tmp_path / "box.step", face_names=named_sides))
    left, _ = x_sides(geometry)

    groups = CadFaceGroups(geometry, {"Left": [face_ref(left)]})

    assert {row.name: len(row.faces) for row in groups.rows if row.kind == "group"} == {"Left": 1, "Side": 3, "Driver": 1}


def test_a_group_with_a_stored_face_not_found_needs_review_and_cannot_generate(box):
    left, _ = x_sides(box)
    lost = CadFaceRef(None, (1.0, 1.0, 1.0), 0.5, (0.0, 0.0, 1.0), "Plane")

    groups = CadFaceGroups(box, {"Lid": [lost], "Half": [face_ref(left), lost], "Fine": [face_ref(left)]})

    assert groups.needs_review == ("Lid", "Half")
    assert {row.name: len(row.faces) for row in groups.rows if row.kind == "group"} == {
        "Lid": 0,  # listed so it can be repaired or deleted
        "Half": 1,
        "Driver": 1,
    }
    with pytest.raises(ValueError, match=r"Face groups need review .*: 'Lid', 'Half'\. Re-pick or delete them"):
        groups.check()


def test_interface_faces_stay_in_their_renamed_interface_face_group(box):
    (interface,) = box.interfaces
    left, _ = x_sides(box)

    groups = CadFaceGroups(
        box,
        {"Mouth": list(interface.face_ids), "Mixed": [interface.face_ids[0], left.id]},  # grouped before detection
        interface_names={"Interface 1 Port": "Port mouth", "Interface 2 Gone": "Ignored"},
    )

    assert groups.interface_names == {"Interface 1 Port": "Port mouth"}
    assert groups.row("Port mouth").faces == set(interface.face_ids)
    assert groups.row("Mouth") is None  # holds interface faces only
    assert groups.row("Mixed").faces == {left.id}
    assert groups.held_by_interfaces == set(interface.face_ids)
    assert groups.group_of(interface.face_ids[0]).kind == "interface"
    groups.check()


def test_reserved_step_names_form_no_group(tmp_path):
    names = {-210: "walls", -200: "Driver"}
    geometry = inspect_step(vented_box(tmp_path / "walls.step", face_names=lambda c: names.get(round(c[2]))))

    groups = CadFaceGroups(geometry, {})

    assert [row.name for row in groups.rows if row.kind == "group"] == ["Driver"]


@pytest.mark.parametrize(
    ("face_groups", "message"),
    [
        ({"walls": []}, "Face group name 'walls' is reserved"),
        ({"Interface 1 Port": []}, "Face group name 'Interface 1 Port' is reserved"),
        ({"": []}, "Face group name '' is reserved"),
        ({"Lid": [9999]}, "Face group 'Lid' refers to unknown face 9999"),
        ({"A": ["left"], "B": ["left"]}, r"Face \d+ is in face groups 'A' and 'B'"),
    ],
)
def test_check_rejects_what_cannot_be_meshed(box, face_groups, message):
    left, _ = x_sides(box)
    face_groups = {name: [left.id if f == "left" else f for f in faces] for name, faces in face_groups.items()}

    with pytest.raises(ValueError, match=message):
        CadFaceGroups(box, face_groups).check()


def test_stored_faces_resolve_in_cad_axes_under_the_placement(tmp_path):
    step = vented_box(tmp_path / "box.step")
    left, _ = x_sides(inspect_step(step))
    placement = dict(rotation_deg=(0.0, 90.0, 0.0), translation_m=(0.025, 0.0, 0.0))
    placed = inspect_step(step, **placement)

    groups = CadFaceGroups(placed, {"Left": [face_ref(left)]}, **placement)

    (face,) = groups.row("Left").faces
    assert next(f for f in placed.faces if f.id == face).normal == pytest.approx((0.0, 0.0, 1.0), abs=1e-9)


def test_surface_groups_split_each_mesh_by_row_sorted_by_name(box):
    left, right = x_sides(box)
    groups = CadFaceGroups(box, {"Sides": [left.id, right.id]})
    driver = next(f.id for f in box.faces if f.step_name == "Driver")

    surface = groups.surface_groups({"exterior": [left.id, right.id, *box.interfaces[0].face_ids], "chamber": [driver]})

    assert {m: list(g) for m, g in surface.items()} == {
        "exterior": ["Interface 1 Port", "Sides"],
        "chamber": ["Driver"],
    }
