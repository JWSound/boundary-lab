"""The size plan of a CAD import: edge lengths of faces and air-body volumes, validation and precedence, no meshing."""

from __future__ import annotations

from dataclasses import replace

import pytest

from cad_fixtures import vented_box
from cad_plugin import CadFaceGroups, CadImportSettings, CadSizeOverride, inspect_step, module

size_plan = module("sizing").size_plan

COARSE = CadSizeOverride(elements_per_wavelength=2)
MM_8, MM_30 = CadSizeOverride(size_m=0.008), CadSizeOverride(size_m=0.03)
# 343 / 3430 / 5 = 20 mm; BEM 2 per wavelength: 50 mm.
SETTINGS = CadImportSettings(3430.0, 343.0, 5.0, 2.0)


@pytest.fixture(scope="module")
def box(tmp_path_factory):
    return inspect_step(vented_box(tmp_path_factory.mktemp("box") / "box.step"))


@pytest.fixture(scope="module")
def groups(box):
    rows = CadFaceGroups(box, {})
    named = {}
    for face in box.faces:
        named.setdefault(rows.group_of(face.id).name, []).append(face.id)
    return {"all": named}


def face_of(box, kind):
    air = {f for b in box.bodies if b.kind == "air" for f in b.face_ids}
    exterior = {f for b in box.bodies if b.kind == "exterior" for f in b.face_ids}
    wanted = {"air": air - exterior, "exterior": exterior - air}[kind]
    return min(wanted)


def test_faces_and_volumes_default_to_their_target_element_sizes(box, groups):
    plan = size_plan(SETTINGS, box, groups)

    assert plan.face_m[face_of(box, "air")] == pytest.approx(0.02)
    assert plan.face_m[face_of(box, "exterior")] == pytest.approx(0.05)
    assert set(plan.face_m) == {f.id for f in box.faces}
    assert plan.volume_m == {b.name: pytest.approx(0.02) for b in box.bodies if b.kind == "air"}


def test_a_face_size_wins_over_a_face_group_which_wins_over_a_body(box, groups):
    face = face_of(box, "air")
    (group,) = [name for name, faces in groups["all"].items() if face in faces]
    (body,) = box.faces[[f.id for f in box.faces].index(face)].bodies
    settings = replace(SETTINGS, body_sizes={body: COARSE})

    assert size_plan(settings, box, groups).face_m[face] == pytest.approx(0.05)
    settings = replace(settings, face_group_sizes={group: MM_30})
    assert size_plan(settings, box, groups).face_m[face] == 0.03
    settings = replace(settings, face_sizes=((face, MM_8),))
    assert size_plan(settings, box, groups).face_m[face] == 0.008


def test_a_volume_size_wins_over_its_bodys_size_and_never_sizes_faces(box, groups):
    body = next(b for b in box.bodies if b.kind == "air")
    settings = replace(SETTINGS, body_sizes={body.name: MM_30}, volume_sizes={body.name: MM_8})

    plan = size_plan(settings, box, groups)

    assert plan.volume_m[body.name] == 0.008
    assert 0.008 not in plan.face_m.values()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"body_sizes": {"Nope": MM_8}}, "Size override for unknown body 'Nope'."),
        ({"face_group_sizes": {"Nope": MM_8}}, "Size override for unknown face group 'Nope'."),
        ({"face_sizes": ((9999, MM_8),)}, "Size override for unknown face 9999."),
        ({"volume_sizes": {"Nope": MM_8}}, "Volume size for 'Nope', which is no air body."),
    ],
)
def test_unknown_bodies_face_groups_faces_and_volumes_are_rejected(box, groups, change, message):
    with pytest.raises(ValueError) as error:
        size_plan(replace(SETTINGS, **change), box, groups)
    assert str(error.value) == message
