"""Mesh sizes of a CAD import design: texts, shared sizes, writes, drops and which size wins, without Qt or meshing."""

from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from cad_fixtures import vented_box
from cad_plugin import (
    FEM_LEVELS,
    LEVELS,
    MIXED,
    CadFaceGroups,
    CadImportSettings,
    CadSizeOverride,
    CadSizes,
    CadSizing,
    face_ref,
    inspect_step,
    module,
)

_sizing = module("sizing")
LEGACY_SIZES, legacy_sizes, migrated, read_face_sizes, size_plan = (
    _sizing.LEGACY_SIZES,
    _sizing.legacy_sizes,
    _sizing.migrated,
    _sizing.read_face_sizes,
    _sizing.size_plan,
)

COARSE, FINE = CadSizeOverride(elements_per_wavelength=2), CadSizeOverride(elements_per_wavelength=6)
MM_8, MM_30 = CadSizeOverride(size_m=0.008), CadSizeOverride(size_m=0.03)
# 343 / 3430 / 5 = 20 mm.
SIZING = CadSizing(3430.0, 343.0, 5.0)


@pytest.fixture(scope="module")
def box(tmp_path_factory):
    return inspect_step(vented_box(tmp_path_factory.mktemp("box") / "box.step"))


def test_edge_lengths_follow_sound_speed_maximum_frequency_and_fineness():
    assert SIZING.target_size_m == pytest.approx(0.02)
    assert SIZING.edge_m(None) == pytest.approx(0.02)
    assert SIZING.edge_m(COARSE) == pytest.approx(0.05)
    assert SIZING.edge_m(MM_8) == 0.008
    assert CadSizing(3430.0, 686.0, 5.0).target_size_m == pytest.approx(0.04)
    with pytest.raises(ValueError, match="maximum frequency is not known"):
        CadSizing(None).edge_m(COARSE)


def test_the_bem_target_element_size_follows_the_bem_fineness_else_the_fem_one():
    assert CadSizing(3430.0, 343.0, 5.0, 2.0).bem_target_size_m == pytest.approx(0.05)
    assert CadSizing(3430.0, 343.0, 5.0, 2.0).target_size_m == pytest.approx(0.02)
    assert SIZING.bem_target_size_m == pytest.approx(0.02)  # no BEM fineness: the FEM one


def face_kinds(geometry):
    """``{face: "air" | "exterior" | "interface"}``."""
    interface = {f for i in geometry.interfaces for f in i.face_ids}
    exterior = {b for b in geometry.bodies if b.kind == "exterior"}
    exterior = {f for b in exterior for f in b.face_ids}
    return {
        f.id: "interface" if f.id in interface else "exterior" if f.id in exterior else "air" for f in geometry.faces
    }


@pytest.mark.parametrize(("bem", "interface"), [(2.0, 5.0), (12.0, 12.0)])
def test_each_face_targets_its_method_and_an_interface_face_the_finer(box, bem, interface):
    sizing = CadSizing(3430.0, 343.0, 5.0, bem)

    targets = sizing.face_targets(box)

    kinds = face_kinds(box)
    assert set(kinds.values()) == {"air", "exterior", "interface"}
    n = {"air": 5.0, "exterior": bem, "interface": interface}
    expected = {kind: CadSizeOverride(elements_per_wavelength=v) for kind, v in n.items()}
    assert targets == {f: expected[kind] for f, kind in kinds.items()}


def test_a_body_without_a_size_counts_as_the_faces_own_target_size():
    sizing = CadSizing(3430.0, 343.0, 5.0, 2.0)  # FEM 20 mm, BEM 50 mm
    bem = CadSizeOverride(elements_per_wavelength=2.0)

    assert sizing.winning(None, None, [MM_30, None], bem) == MM_30  # finer than the face's 50 mm
    assert sizing.winning(None, None, [MM_30, None]) is None  # coarser than the FEM 20 mm


def test_texts_name_levels_custom_numbers_edge_lengths_global_and_mixed():
    assert SIZING.text(None) == "Global (BEM Medium)"  # N = 5 is BEM Medium's
    assert CadSizing(3430.0, 343.0, 3.0).text(None) == "Global (3 per wavelength)"
    assert CadSizing(3430.0).text(None) == "Global (FEM Fine)"
    assert SIZING.text(COARSE) == "FEM Coarse"
    assert SIZING.text(COARSE, edge=True) == "FEM Coarse (50 mm)"
    assert CadSizing(None).text(COARSE, edge=True) == "FEM Coarse"  # edge lengths once the frequency is known
    assert SIZING.text(MM_8, edge=True) == "8 mm"
    assert SIZING.text(MIXED) == "Mixed"


def test_six_levels_fine_to_coarse_each_named_by_its_elements_per_wavelength():
    assert LEVELS == {
        "FEM Fine": 6,
        "FEM Medium": 4,
        "FEM Coarse": 2,
        "BEM Fine": 12,
        "BEM Medium": 5,
        "BEM Coarse": 0.3,
    }
    assert list(LEVELS) == ["FEM Fine", "FEM Medium", "FEM Coarse", "BEM Fine", "BEM Medium", "BEM Coarse"]
    assert list(FEM_LEVELS) == ["FEM Fine", "FEM Medium", "FEM Coarse"]
    for name, n in LEVELS.items():
        assert SIZING.text(CadSizeOverride(elements_per_wavelength=n)) == name
    assert SIZING.text(CadSizeOverride(elements_per_wavelength=0.3), edge=True) == "BEM Coarse (333 mm)"


@pytest.mark.parametrize(
    ("size", "message"),
    [
        ({}, "must be elements per wavelength or an edge length"),
        ({"size_m": 0}, "edge length must be a positive number"),
        ({"elements_per_wavelength": 2, "size_m": 0.01}, "must be elements per wavelength or an edge length"),
    ],
)
def test_overrides_are_validated_and_stored_as_their_one_field(size, message):
    with pytest.raises(ValueError, match=message):
        CadSizeOverride(**size)
    assert MM_8.stored() == {"size_m": 0.008}


@pytest.mark.parametrize(
    ("face", "group", "bodies", "winner"),
    [
        (MM_8, COARSE, [MM_30], MM_8),  # the face's own
        (None, COARSE, [MM_8], COARSE),  # its face group's
        (None, None, [MM_30, COARSE], MM_30),  # the finest body's
        (None, None, [MM_8, None], MM_8),  # finer than the target size of the body without one
        (None, None, [MM_30, None], None),  # coarser: the target size
        (None, None, [None], None),
    ],
)
def test_the_most_specific_size_wins_on_a_face_else_the_finest_body(face, group, bodies, winner):
    assert SIZING.winning(face, group, bodies) == winner


def faces_of(geometry, body):
    return sorted(f.id for f in geometry.faces if body in f.bodies)


def sizes(geometry, source):
    return CadSizes(geometry, source.get("face_sizes", []), source.get("volume_sizes", {}))


def test_a_size_written_to_faces_and_a_volume_is_read_back_and_global_clears_it(box):
    chamber = faces_of(box, "Chamber")
    empty = sizes(box, {})

    written = empty.written(FINE, chamber, ["Chamber"])
    read = sizes(box, written)

    assert written["volume_sizes"] == {"Chamber": {"elements_per_wavelength": 6}}
    assert written["face_sizes"][0]["face"] == asdict(face_ref(next(f for f in box.faces if f.id == chamber[0])))
    assert read.faces == dict.fromkeys(chamber, FINE) and read.volumes == {"Chamber": FINE}
    assert read.shared(chamber, ["Chamber"]) == FINE
    assert read.shared(faces_of(box, "Cabinet")) is None  # all Global
    assert read.shared([*chamber, *faces_of(box, "Cabinet")]) == MIXED
    assert read.written(None, chamber[:1], ["Chamber"]) == {
        "face_sizes": written["face_sizes"][1:],
        "volume_sizes": {},
    }


@pytest.mark.parametrize(
    "entry", [{"size_m": 0.01}, {"face": {"step_name": None}, "size_m": 0.01}, {"face": "Driver", "size_m": 0.01}, "x"]
)
def test_stored_face_sizes_without_a_face_are_rejected(entry):
    with pytest.raises(ValueError, match="Invalid face size: a face and its size override expected"):
        read_face_sizes([entry])


@pytest.mark.parametrize(
    ("size", "message"),
    [
        ({"bogus": 1}, "a face and its size override expected"),
        ({"size_m": -1}, "Size override edge length must be a positive"),
    ],
)
def test_stored_face_sizes_with_a_bad_override_are_rejected(box, size, message):
    entry = {"face": asdict(face_ref(box.faces[0]))} | size
    for read in (lambda: read_face_sizes([entry]), lambda: CadSizes(box, [entry], {})):
        with pytest.raises(ValueError, match=f"Invalid face size: {message}"):
            read()


def test_stored_face_sizes_read_as_faces_and_overrides(box):
    ref = face_ref(box.faces[0])

    assert read_face_sizes([{"face": asdict(ref), "size_m": 0.008}]) == [(ref, MM_8)]


def test_writing_one_face_keeps_the_others_and_replaces_its_own(box):
    first, second = faces_of(box, "Cabinet")[:2]
    both = sizes(box, sizes(box, {}).written(COARSE, [first, second]))

    after = sizes(box, both.written(MM_8, [second]))

    assert after.faces == {first: COARSE, second: MM_8}


def test_sizes_of_faces_gone_and_volumes_of_bodies_no_longer_air_are_dropped_with_a_note(box):
    chamber = faces_of(box, "Chamber")
    source = sizes(box, {}).written(FINE, chamber[:1], ["Chamber"])
    lost = dict(source["face_sizes"][0], face=asdict(face_ref(replace(box.faces[0], centre=(1.0, 1.0, 1.0)))))
    source = {
        "face_sizes": [*source["face_sizes"], lost],
        "volume_sizes": source["volume_sizes"] | {"Cabinet": {"size_m": 0.01}},
    }

    changes, note = sizes(box, source).dropped()

    assert changes == {
        "face_sizes": source["face_sizes"][:1],
        "volume_sizes": {"Chamber": {"elements_per_wavelength": 6}},
    }
    assert (
        note
        == "1 face size override dropped: its face is no longer in the STEP file. Volume size of Cabinet dropped: no air body."
    )
    assert sizes(box, changes).dropped() == ({}, "")


# -- older designs ---------------------------------------------------------------------------------------------------


def settings_of(geometry, source, sizing=SIZING):
    """The pipeline settings of design ``source`` (also its older body and face-group sizes) at ``sizing``."""
    body_sizes, group_sizes = legacy_sizes(source)
    own = sizes(geometry, source)
    return CadImportSettings(
        sizing.fmax_hz,
        sizing.sound_speed_m_per_s,
        sizing.elements_per_wavelength,
        sizing.bem_elements_per_wavelength,
        body_sizes=body_sizes,
        face_group_sizes=group_sizes,
        face_sizes=tuple(own.faces.items()),
        volume_sizes=own.volumes,
    )


def named_faces(geometry):
    """The surface groups ``size_plan`` takes: every face under its face group row."""
    rows = CadFaceGroups(geometry, {})
    named = {}
    for f in geometry.faces:
        named.setdefault(rows.group_of(f.id).name, []).append(f.id)
    return {"all": named}


def test_body_face_group_and_legacy_sizes_become_face_and_volume_sizes_that_mesh_the_same(box):
    own = sizes(box, {}).written(MM_8, faces_of(box, "Cabinet")[:1], ["Port"])
    source = own | {
        "elements_per_wavelength": 5.0,  # 20 mm
        "body_sizes_m": {"Chamber": 0.04},  # coarser: its faces shared with no other body only
        "face_group_sizes_m": {"Driver": 0.008},
        "body_sizes": {"Port": {"size_m": 0.03}, "Cabinet": {"size_m": 0.01}},
        "face_group_sizes": {"walls": {"elements_per_wavelength": 2}},
    }

    new = migrated(source, CadFaceGroups(box, {}), sizes(box, source), SIZING)

    assert not set(LEGACY_SIZES) & new.keys()
    assert new["elements_per_wavelength"] == 5.0
    assert new["volume_sizes"] == {"Chamber": {"size_m": 0.04}, "Port": {"size_m": 0.008}}  # the volume's own stays
    assert new["face_sizes"][: len(own["face_sizes"])] == own["face_sizes"]
    assert size_plan(settings_of(box, new), box, named_faces(box)) == size_plan(
        settings_of(box, source), box, named_faces(box)
    )
    assert migrated(new, CadFaceGroups(box, {}), sizes(box, new), SIZING) is None


def test_migrating_compares_a_body_size_with_each_faces_own_target_size(tmp_path):
    # The cabinet front covers the port mouth: that face is on the Port and the Cabinet.
    covered = inspect_step(vented_box(tmp_path / "covered.step", covered=True))
    (shared,) = [f.id for f in covered.faces if set(f.bodies) == {"Port", "Cabinet"}]
    sizing = CadSizing(3430.0, 343.0, 2.0, 5.0)  # FEM 50 mm, BEM 20 mm: 20 mm on the shared face
    source = {"body_sizes": {"Port": {"size_m": 0.03}}}  # finer than FEM, coarser than the shared face's target

    new = migrated(source, CadFaceGroups(covered, {}), sizes(covered, source), sizing)

    groups = named_faces(covered)
    plan = size_plan(settings_of(covered, new, sizing), covered, groups)
    assert plan == size_plan(settings_of(covered, source, sizing), covered, groups)
    assert plan.face_m[shared] == pytest.approx(0.02)


def test_designs_without_body_or_face_group_sizes_are_not_migrated(box):
    source = {"face_sizes": [], "volume_sizes": {}, "body_sizes": {}, "face_group_sizes": {}}  # as earlier slices saved

    assert migrated(source, CadFaceGroups(box, {}), sizes(box, source), SIZING) is None
