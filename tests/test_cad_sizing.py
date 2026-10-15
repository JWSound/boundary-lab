"""Mesh sizes of a CAD import design: texts, shared sizes, writes, drops and which size wins, without Qt or meshing."""

from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from blab.cad_import import MIXED, CadSizeOverride, CadSizes, CadSizing, face_ref, inspect_step
from cad_fixtures import vented_box

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


def test_texts_name_levels_custom_numbers_edge_lengths_global_and_mixed():
    assert SIZING.text(None) == "Global (5 per wavelength)"
    assert CadSizing(3430.0).text(None) == "Global (Fine)"
    assert SIZING.text(COARSE) == "Coarse"
    assert SIZING.text(COARSE, edge=True) == "Coarse (50 mm)"
    assert CadSizing(None).text(COARSE, edge=True) == "Coarse"  # edge lengths once the frequency is known
    assert SIZING.text(MM_8, edge=True) == "8 mm"
    assert SIZING.text(MIXED) == "Mixed"


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
