"""A CAD import design on a read geometry: face groups, walls and interface face groups, without Qt or meshing."""

from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from cad_fixtures import named_sides, vented_box
from cad_plugin import (
    CadDesign,
    CadFaceGroups,
    CadFaceRef,
    CadSizeOverride,
    CadSizing,
    design_settings,
    face_ref,
    inspect_step,
    module,
)

_design = module("design")
BEM_FINENESS, FEM_FINENESS = _design.BEM_FINENESS, _design.FEM_FINENESS


@pytest.fixture(scope="module")
def box(tmp_path_factory):
    return inspect_step(vented_box(tmp_path_factory.mktemp("box") / "box.step"))


def x_sides(geometry):
    """The cabinet's two X sides, left first."""
    sides = [f for f in geometry.faces if f.bodies == ("Cabinet",) and abs(abs(f.normal[0]) - 1) < 1e-9]
    return sorted(sides, key=lambda f: f.centre[0])


def row(groups, name):
    """The row named ``name``, else ``None``."""
    return next((row for row in groups.rows if row.name == name), None)


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
    assert row(groups, "walls").faces == {f.id for f in box.faces} - {driver, *interface.face_ids}
    assert groups.needs_review == ()
    groups.check()


def test_stored_groups_come_before_step_named_ones_and_take_their_faces(box):
    left, right = x_sides(box)
    driver = next(f for f in box.faces if f.step_name == "Driver")

    groups = CadFaceGroups(box, {"Sides": [asdict(face_ref(left)), face_ref(right)], "Cone": [driver.id]})

    assert [row.name for row in groups.rows] == ["walls", "Interface 1 Port", "Sides", "Cone"]
    assert row(groups, "Sides").faces == {left.id, right.id}
    assert row(groups, "Driver") is None  # its only face is in Cone


def test_faces_sharing_a_step_name_split_between_a_stored_group_and_their_name(tmp_path):
    geometry = inspect_step(vented_box(tmp_path / "box.step", face_names=named_sides))
    left, _ = x_sides(geometry)

    groups = CadFaceGroups(geometry, {"Left": [face_ref(left)]})

    assert {row.name: len(row.faces) for row in groups.rows if row.kind == "group"} == {
        "Left": 1,
        "Side": 3,
        "Driver": 1,
    }


def test_a_group_with_a_stored_face_not_found_needs_review_and_cannot_generate(box):
    left, _ = x_sides(box)
    lost = CadFaceRef(None, (1.0, 1.0, 1.0), 0.5, (0.0, 0.0, 1.0), "Plane")

    groups = CadFaceGroups(box, {"Lid": [lost], "Half": [face_ref(left), lost], "Fine": [face_ref(left)]})

    assert groups.needs_review == ("Lid", "Half")
    assert {row.name: len(row.faces) for row in groups.rows if row.kind == "group"} == {
        "Lid": 0,  # listed so it can be repaired or deleted
        "Half": 1,
        "Fine": 0,  # its face is Half's, which holds it first
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
    assert row(groups, "Port mouth").faces == set(interface.face_ids)
    assert row(groups, "Mouth").faces == set()  # holds interface faces only: listed, so it can be deleted
    assert row(groups, "Mixed").faces == {left.id}
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
        ({"Interface 1 Port": []}, "Face group name 'Interface 1 Port' is taken"),
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

    (face,) = row(groups, "Left").faces
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


# -- edits ---------------------------------------------------------------------------------------------------------


def stored(source):
    """``face_groups`` of a design source as the module takes them."""
    return {g["name"]: g["faces"] for g in source.get("face_groups", [])}


def edited(groups, edit, source=None):
    """The design after ``edit``: its source and its face groups."""
    source = (source or {"face_groups": [], "interface_names": {}}) | edit.changes
    return source, CadFaceGroups(groups.geometry, stored(source), interface_names=source["interface_names"])


def group_faces(groups):
    return {row.name: set(row.faces) for row in groups.rows if row.kind == "group"}


def test_moving_faces_into_a_new_then_an_existing_group_takes_them_out_of_any_other(box):
    left, right = x_sides(box)
    groups = CadFaceGroups(box, {})

    source, groups = edited(groups, groups.moved_to(" A ", {left.id, right.id}))
    assert group_faces(groups) == {"A": {left.id, right.id}, "Driver": {next(iter(row(groups, "Driver").faces))}}
    source, groups = edited(groups, groups.moved_to("B", {right.id}), source)

    assert [(g["name"], len(g["faces"])) for g in source["face_groups"]] == [("A", 1), ("B", 1)]
    assert source["face_groups"][1]["faces"] == [asdict(face_ref(right))]  # stored in CAD axes
    assert row(groups, "B").faces == {right.id}
    source, groups = edited(groups, groups.moved_to("B", {left.id}), source)
    assert [g["name"] for g in source["face_groups"]] == ["B"]  # A emptied, so gone


def test_moving_to_walls_takes_faces_out_but_step_named_and_interface_faces_stay(box):
    left, _ = x_sides(box)
    driver = next(f.id for f in box.faces if f.step_name == "Driver")
    mouth = box.interfaces[0].face_ids[0]
    groups = CadFaceGroups(box, {"Cone": [driver], "Side": [left.id]})

    edit = groups.moved_to_walls({driver, left.id, mouth})

    assert edit.changes == {"face_groups": []}
    assert edit.note == (
        f"Faces {driver} stay in the face groups their STEP names form. "
        f"Face {mouth} stays in interface 'Interface 1 Port'."
    )
    assert [g["name"] for g in groups.moved_to("walls", {left.id}).changes["face_groups"]] == ["Cone"]


@pytest.mark.parametrize(
    ("name", "faces", "message"),
    [
        ("", "left", "Select faces and enter a face group name first"),
        ("A", "", "Select faces and enter a face group name first"),
        ("Interface 1 Port", "left", "Face group name 'Interface 1 Port' is reserved"),
    ],
)
def test_moves_that_cannot_be_stored_are_rejected(box, name, faces, message):
    left, _ = x_sides(box)
    with pytest.raises(ValueError, match=message):
        CadFaceGroups(box, {}).moved_to(name, {left.id} if faces else set())


def test_interface_faces_are_skipped_with_a_note(box):
    mouth = box.interfaces[0].face_ids[0]
    left, _ = x_sides(box)
    groups = CadFaceGroups(box, {})

    only = groups.moved_to("Mouth", {mouth})
    mixed = groups.moved_to("Mouth", {mouth, left.id})

    assert only == ({}, f"Face {mouth} stays in interface 'Interface 1 Port'.")
    assert [len(g["faces"]) for g in mixed.changes["face_groups"]] == [1]


def test_unmatched_faces_are_kept_unless_the_group_is_re_picked(box):
    left, right = x_sides(box)
    lost = asdict(CadFaceRef(None, (1.0, 1.0, 1.0), 0.5, (0.0, 0.0, 1.0), "Plane"))
    groups = CadFaceGroups(box, {"Lid": [lost, asdict(face_ref(left))]})

    kept = groups.moved_to("Other", {left.id}).changes["face_groups"]
    repaired = groups.moved_to("Lid", {right.id}).changes["face_groups"]

    assert kept[0] == {"name": "Lid", "faces": [lost]}  # still needs review
    assert repaired == [{"name": "Lid", "faces": [asdict(face_ref(left)), asdict(face_ref(right))]}]


def test_deleting_a_picked_group_keeps_the_others(box):
    left, right = x_sides(box)
    groups = CadFaceGroups(box, {"A": [left.id], "B": [right.id]})

    assert groups.deleted("A").changes == {"face_groups": [{"name": "B", "faces": [right.id]}]}
    with pytest.raises(ValueError, match="'Driver' is no picked face group"):
        groups.deleted("Driver")


def test_picking_one_of_several_faces_sharing_a_step_name_groups_only_that_face(tmp_path):
    sided = inspect_step(vented_box(tmp_path / "sides.step", face_names=named_sides))
    left, right = x_sides(sided)
    groups = CadFaceGroups(sided, {})

    source, groups = edited(groups, groups.moved_to("Left", {left.id}))
    source, groups = edited(groups, groups.moved_to("Right", {right.id}), source)

    assert {g["name"]: len(g["faces"]) for g in source["face_groups"]} == {"Left": 1, "Right": 1}
    assert {name: len(faces) for name, faces in group_faces(groups).items()} == {
        "Left": 1,
        "Right": 1,
        "Side": 2,
        "Driver": 1,
    }


def test_a_group_whose_faces_are_gone_needs_review_and_re_picking_repairs_it(box):
    left, right = x_sides(box)
    lost = asdict(CadFaceRef(None, (1.0, 1.0, 1.0), 0.5, (0.0, 0.0, 1.0), "Plane"))
    groups = CadFaceGroups(box, {"Back": [lost]})
    assert groups.needs_review == ("Back",)

    source, groups = edited(groups, groups.moved_to("Back", {left.id}), {"face_groups": [], "interface_names": {}})

    assert groups.needs_review == ()
    assert [len(g["faces"]) for g in source["face_groups"]] == [1]
    assert group_faces(groups)["Back"] == {left.id}


def test_an_interface_is_renamed_unless_the_name_is_taken_and_an_empty_name_gives_back_the_detected_one(box):
    groups = CadFaceGroups(box, {"A": []}, interface_names={"Interface 9 Gone": "Old"})

    source = {"face_groups": [{"name": "A", "faces": []}], "interface_names": {"Interface 9 Gone": "Old"}}
    source, renamed = edited(groups, groups.renamed_interface("Interface 1 Port", " Mouth "), source)

    assert source["interface_names"] == {"Interface 9 Gone": "Old", "Interface 1 Port": "Mouth"}
    assert row(renamed, "Mouth").kind == "interface"
    for taken in ("walls", "Driver", "A"):
        with pytest.raises(ValueError, match=f"Face group name '{taken}' is taken"):
            renamed.renamed_interface("Mouth", taken)
    assert renamed.renamed_interface("Mouth", "").changes == {"interface_names": {"Interface 9 Gone": "Old"}}
    assert renamed.renamed_interface("Mouth", "Mouth").changes == {}


# -- design source -> pipeline settings, without the provider or Gmsh ---------------------------------------------------

CONFIGURATION = {"frequency": {"freq_max_hz": 1000.0}, "symmetry_config": {"mode": "off"}}
FACE = asdict(CadFaceRef("X", (0.0, 0.0, 0.0), 1.0, (0.0, 0.0, 1.0), "Plane"))


def source(**fields):
    return {
        "elements_per_wavelength": 6,
        "curvature_elements_per_circle": 12,
        "growth_rate": 1.3,
        "rotation_deg": [0, 0, 90],
        "translation_mm": [10, 0, -20],
    } | fields


def test_settings_come_from_the_source_with_the_project_frequency_and_symmetry():
    settings = design_settings(
        source(face_groups=[{"name": "A", "faces": [3, 4]}], interface_names={"Interface 1 Port": "Port"}),
        CONFIGURATION | {"symmetry_config": {"mode": "xy"}},
    )

    assert settings.fmax_hz == 1000.0
    assert settings.symmetry == "xy"
    assert settings.face_groups == {"A": (3, 4)}
    assert settings.interface_names == {"Interface 1 Port": "Port"}
    assert settings.rotation_deg == (0, 0, 90)
    assert settings.translation_m == (0.01, 0.0, -0.02)


def test_an_older_design_without_a_bem_fineness_uses_the_fem_one_for_both():
    settings = design_settings(source(), CONFIGURATION)

    assert settings.bem_elements_per_wavelength is None
    assert settings.sizing.bem_target_size_m == settings.sizing.target_size_m


def test_a_bem_fineness_is_read_as_a_number():
    assert design_settings(source(bem_elements_per_wavelength="5"), CONFIGURATION).bem_elements_per_wavelength == 5.0


def test_older_body_and_face_group_sizes_are_read_with_the_face_and_volume_sizes():
    settings = design_settings(
        source(
            body_sizes_m={"Cabinet": 0.01},
            face_group_sizes={"A": {"elements_per_wavelength": 12}},
            volume_sizes={"Cabinet": {"size_m": 0.02}},
            face_sizes=[{"face": FACE, "size_m": 0.005}],
        ),
        CONFIGURATION,
    )

    assert settings.body_sizes == {"Cabinet": CadSizeOverride(size_m=0.01)}
    assert settings.face_group_sizes == {"A": CadSizeOverride(elements_per_wavelength=12)}
    assert settings.volume_sizes == {"Cabinet": CadSizeOverride(size_m=0.02)}
    assert [size for _, size in settings.face_sizes] == [CadSizeOverride(size_m=0.005)]


@pytest.mark.parametrize(
    "fields,match",
    [
        ({"bem_elements_per_wavelength": "fine"}, "BEM elements per wavelength must be a positive number"),
        ({"face_groups": {"Baffle": []}}, "Face groups must be a list"),
        (
            {"face_groups": [{"name": "A", "faces": []}, {"name": "A", "faces": []}]},
            "Face group 'A' is defined twice",
        ),
        ({"face_groups": [{"faces": []}]}, "Invalid CAD import source"),
        ({"face_sizes": [{"face": FACE, "size_m": -1}]}, "Invalid face size"),
        ({"growth_rate": None}, "Invalid CAD import source"),
        ({"rotation_deg": None}, "Invalid CAD import source"),
    ],
)
def test_an_invalid_source_is_rejected(fields, match):
    with pytest.raises(ValueError, match=match):
        design_settings(source(**fields), CONFIGURATION)


# -- the editor's view of the design source: placement, fineness, sizing, size edits ------------------------------------

FINE = CadSizeOverride(elements_per_wavelength=6)


def test_the_placement_is_the_sources_with_the_translation_in_metres_and_defaults_to_none():
    design = CadDesign(source())

    assert design.rotation_deg == (0, 0, 90)
    assert design.translation_mm == (10, 0, -20)
    assert design.placement == {"rotation_deg": (0, 0, 90), "translation_m": (0.01, 0.0, -0.02)}
    assert CadDesign({"elements_per_wavelength": 6}).placement == {
        "rotation_deg": (0.0, 0.0, 0.0),
        "translation_m": (0.0, 0.0, 0.0),
    }


def test_the_bem_fineness_falls_back_to_the_fem_one_and_the_sizing_follows_both():
    design = CadDesign(source(elements_per_wavelength=4))

    assert design.number(FEM_FINENESS) == design.number(BEM_FINENESS) == 4.0
    assert design.number("growth_rate") == 1.3
    with pytest.raises(KeyError):
        CadDesign({"elements_per_wavelength": 4}).number("growth_rate")  # no fallback for other settings
    assert CadDesign(source(bem_elements_per_wavelength=12)).number(BEM_FINENESS) == 12.0
    sizing = CadDesign(source(bem_elements_per_wavelength=12)).sizing(3430.0, 343.0)
    assert sizing == CadSizing(3430.0, 343.0, 6.0, 12)


def test_the_design_names_its_stored_face_groups(box):
    design = CadDesign(source(face_groups=[{"name": "A", "faces": []}]), box)

    assert design.stored_groups == {"A"}
    assert "A" in [row.name for row in design.groups.rows]
    assert CadDesign(source(), None).groups is None


def test_sizes_are_set_on_faces_and_volumes_and_a_bad_size_or_stored_size_is_refused(box):
    chamber = [f.id for f in box.faces if "Chamber" in f.bodies]
    design = CadDesign(source(), box)

    changes = design.set_sizes(FINE, chamber[:1], ["Chamber"])
    assert changes["volume_sizes"] == {"Chamber": {"elements_per_wavelength": 6}}
    assert [e["elements_per_wavelength"] for e in changes["face_sizes"]] == [6]
    assert CadDesign(source() | changes, box).set_sizes(None, chamber[:1], []) == {
        "face_sizes": [],
        "volume_sizes": {"Chamber": {"elements_per_wavelength": 6}},
    }
    assert design.set_sizes({"size_m": 0.01}, chamber[:1])["face_sizes"][0]["size_m"] == 0.01
    with pytest.raises(ValueError, match="positive"):
        design.set_sizes({"size_m": -1}, chamber[:1])

    entry = {"face": asdict(face_ref(box.faces[0])), "size_m": -1}
    broken = CadDesign(source(face_sizes=[entry]), box)
    assert broken.size_error == "Invalid face size: Size override edge length must be a positive number."
    with pytest.raises(ValueError, match="Invalid face size"):
        broken.set_sizes(FINE, chamber[:1])
    assert broken.dropped() == ({}, "")
    assert broken.migrated(CadSizing(1000.0)) is None


def test_deleting_a_group_keeps_the_sizes_of_its_faces(box):
    left, _ = x_sides(box)
    design = CadDesign(source(), box)
    sized = source(**design.groups.moved_to("Sides", {left.id}).changes, **design.set_sizes(FINE, [left.id]))

    edit = CadDesign(sized, box).groups.deleted("Sides")

    assert edit.changes == {"face_groups": []}  # the sizes are other fields, stored as they were
    assert [e["elements_per_wavelength"] for e in sized["face_sizes"]] == [6]


def test_sizes_set_on_walls_and_interface_rows_are_written_to_their_faces(box):
    design = CadDesign(source(), box)
    walls, interface = (design.groups.rows[0], design.groups.rows[1])

    interface_changes = design.set_sizes(FINE, interface.faces)
    walls_changes = design.set_sizes(CadSizeOverride(elements_per_wavelength=2), walls.faces)

    assert len(interface_changes["face_sizes"]) == len(interface.faces)
    assert len(walls_changes["face_sizes"]) == len(walls.faces)
    assert set(interface_changes) == set(walls_changes) == {"face_sizes", "volume_sizes"}  # no face group sizes


def test_a_bem_level_stores_its_elements_per_wavelength_on_faces_and_volumes(box):
    side = x_sides(box)[0].id

    design = CadDesign(source(), box)

    faces = design.set_sizes(CadSizeOverride(elements_per_wavelength=0.3), [side])
    volumes = design.set_sizes(CadSizeOverride(elements_per_wavelength=12), [], ["Chamber"])

    assert [e["elements_per_wavelength"] for e in faces["face_sizes"]] == [0.3]
    assert volumes["volume_sizes"] == {"Chamber": {"elements_per_wavelength": 12}}


def test_the_shown_target_of_a_face_is_the_one_the_size_plan_meshes_it_with(box):
    size_plan = module("sizing").size_plan

    design = CadDesign(source(elements_per_wavelength=6, bem_elements_per_wavelength=2), box)
    settings = design_settings(design.source, CONFIGURATION)
    sizing = design.sizing(settings.fmax_hz, settings.sound_speed_m_per_s)
    named = {}
    for face in box.faces:
        named.setdefault(design.groups.group_of(face.id).name, []).append(face.id)
    plan = size_plan(settings, box, {"all": named})
    air = {f for b in box.bodies if b.kind == "air" for f in b.face_ids}
    exterior = {f for b in box.bodies if b.kind == "exterior" for f in b.face_ids}
    faces = [min(air - exterior), min(exterior - air)]

    shown = design.targets(sizing, faces, ["Chamber"])

    assert [sizing.edge_m(t) for t in shown[:2]] == [pytest.approx(plan.face_m[f]) for f in faces]
    assert sizing.edge_m(shown[2]) == pytest.approx(plan.volume_m["Chamber"])
    assert shown[0] != shown[1]


def test_lost_sizes_are_dropped_with_a_note(box):
    lost = {"face": asdict(face_ref(replace(box.faces[0], centre=(1.0, 1.0, 1.0)))), "size_m": 0.01}

    changes, note = CadDesign(source(face_sizes=[lost]), box).dropped()

    assert changes == {"face_sizes": [], "volume_sizes": {}}
    assert "1 face size override dropped" in note
    assert CadDesign(source(face_sizes=[]), box).dropped() == ({}, "")


def test_older_body_and_face_group_sizes_migrate_once_to_a_whole_new_source(box):
    old = source(elements_per_wavelength=5.0, body_sizes={"Port": {"size_m": 0.03}}, face_group_sizes={})
    sizing = CadSizing(3430.0, 343.0, 5.0)

    new = CadDesign(old, box).migrated(sizing)

    assert "body_sizes" not in new and new["volume_sizes"] == {"Port": {"size_m": 0.03}}
    assert CadDesign(new, box).migrated(sizing) is None
    assert CadDesign(source(body_sizes={}, face_group_sizes={}), box).migrated(sizing) is None
    assert CadDesign(old, None).migrated(sizing) is None
