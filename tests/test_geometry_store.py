"""GeometryStore derivations. Pure logic: no Qt, no MainWindow."""

from __future__ import annotations

from pathlib import Path

from blab.config import RadiatorConfig
from blab.generators.ath import ath_source
from blab.generators.base import GeneratedGeometry, GeneratorDocument
from blab.ui.main_window.geometry_store import GeometryStore, radiator_channel_assignments


def _document(doc_id: str, name: str, *, mesh_enabled: bool = True) -> GeneratorDocument:
    return GeneratorDocument(
        id=doc_id,
        name=name,
        provider_id="ath",
        provider_schema_version=1,
        source=ath_source(""),
        mesh_enabled=mesh_enabled,
    )


def _geometry(*radiators: RadiatorConfig) -> GeneratedGeometry:
    return GeneratedGeometry(
        provider_id="ath",
        output_dir=Path("runs"),
        mesh_path=Path("runs/case.msh"),
        source_path=Path("runs/case.cfg"),
        radiators=radiators,
    )


def test_an_empty_store_derives_nothing() -> None:
    store = GeometryStore()

    assert store.enabled_geometry(()) == ()
    assert store.all_radiators(()) == ()


def test_documents_without_generated_geometry_are_skipped() -> None:
    store = GeometryStore()
    document = _document("d1", "waveguide")

    assert store.enabled_geometry((document,)) == (), "no geometry generated yet"


def test_mesh_disabled_documents_are_excluded_even_with_geometry() -> None:
    document = _document("d1", "waveguide", mesh_enabled=False)
    store = GeometryStore(generated_by_document_id={"d1": _geometry()})

    assert store.enabled_geometry((document,)) == (), "a disabled mesh must not contribute"


def test_generated_radiators_are_rehomed_onto_their_mesh_name() -> None:
    document = _document("d1", "waveguide")
    store = GeometryStore(
        generated_by_document_id={"d1": _geometry(RadiatorConfig(name="throat", tag=2, mesh=None))},
    )

    radiators = store.all_radiators((document,))

    assert len(radiators) == 1
    assert radiators[0].mesh == "waveguide", "mesh must come from the owning document"
    assert radiators[0].name == "throat"


def test_imported_radiators_are_included_alongside_generated_ones() -> None:
    document = _document("d1", "waveguide")
    store = GeometryStore(
        generated_by_document_id={"d1": _geometry(RadiatorConfig(name="throat", tag=2))},
        imported_radiators=(RadiatorConfig(name="cone", tag=7, mesh="cabinet"),),
    )

    meshes = [radiator.mesh for radiator in store.all_radiators((document,))]

    assert meshes == ["waveguide", "cabinet"]


def test_applying_radiators_routes_each_back_to_its_generated_mesh() -> None:
    document = _document("d1", "waveguide")
    store = GeometryStore(generated_by_document_id={"d1": _geometry(RadiatorConfig(name="old", tag=1))})

    store.apply_radiators(
        (document,),
        (RadiatorConfig(name="throat", tag=2, mesh="waveguide", level_db=-6.0),),
    )

    updated = store.generated_by_document_id["d1"].radiators
    assert [r.name for r in updated] == ["throat"]
    assert updated[0].level_db == -6.0, "edits must survive the round trip"


def test_radiators_on_unknown_meshes_are_retained_as_imported() -> None:
    document = _document("d1", "waveguide")
    store = GeometryStore(generated_by_document_id={"d1": _geometry()})

    store.apply_radiators(
        (document,),
        (
            RadiatorConfig(name="throat", tag=2, mesh="waveguide"),
            RadiatorConfig(name="cone", tag=7, mesh="cabinet"),
        ),
    )

    assert [r.mesh for r in store.imported_radiators] == ["cabinet"]
    assert [r.name for r in store.generated_by_document_id["d1"].radiators] == ["throat"]


def test_applying_an_empty_set_clears_both_sides() -> None:
    document = _document("d1", "waveguide")
    store = GeometryStore(
        generated_by_document_id={"d1": _geometry(RadiatorConfig(name="throat", tag=2))},
        imported_radiators=(RadiatorConfig(name="cone", tag=7, mesh="cabinet"),),
    )

    store.apply_radiators((document,), ())

    assert store.imported_radiators == ()
    assert store.generated_by_document_id["d1"].radiators == ()


def test_clearing_imported_radiators_leaves_generated_geometry_alone() -> None:
    document = _document("d1", "waveguide")
    store = GeometryStore(
        generated_by_document_id={"d1": _geometry(RadiatorConfig(name="throat", tag=2))},
        imported_radiators=(RadiatorConfig(name="cone", tag=7, mesh="cabinet"),),
    )

    store.clear_imported_radiators()

    assert store.imported_radiators == ()
    assert store.all_radiators((document,))[0].mesh == "waveguide"


def test_radiators_on_deleted_channels_fall_back() -> None:
    fallback = "main"
    store = GeometryStore(
        generated_by_document_id={
            "d1": _geometry(
                RadiatorConfig(name="throat", tag=2, channel="tweeter"),
                RadiatorConfig(name="mouth", tag=3, channel="gone"),
            )
        }
    )

    store.reassign_channels({"tweeter", fallback}, fallback)

    channels = [r.channel for r in store.generated_by_document_id["d1"].radiators]
    assert channels == ["tweeter", fallback], "only the orphaned radiator moves"


def test_reassignment_is_a_no_op_when_every_channel_still_exists() -> None:
    store = GeometryStore(
        generated_by_document_id={"d1": _geometry(RadiatorConfig(name="throat", tag=2, channel="tweeter"))}
    )
    before = store.generated_by_document_id["d1"].radiators

    store.reassign_channels({"tweeter", "main"}, "main")

    assert store.generated_by_document_id["d1"].radiators == before


def test_channel_assignment_snapshot_detects_a_rehomed_radiator() -> None:
    before = radiator_channel_assignments((RadiatorConfig(name="throat", tag=2, mesh="wg", channel="tweeter"),))
    after = radiator_channel_assignments((RadiatorConfig(name="throat", tag=2, mesh="wg", channel="main"),))

    assert before != after


def test_channel_assignment_snapshot_ignores_unrelated_edits() -> None:
    before = radiator_channel_assignments((RadiatorConfig(name="throat", tag=2, mesh="wg", level_db=0.0),))
    after = radiator_channel_assignments((RadiatorConfig(name="throat", tag=2, mesh="wg", level_db=-6.0),))

    assert before == after, "a level change is not a channel reassignment"
