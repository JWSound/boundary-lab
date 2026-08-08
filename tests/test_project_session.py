"""ProjectSession and the unsaved-changes rule. Pure logic: no Qt, no MainWindow.

This rule decides whether the user is shown a modal "you have unsaved changes"
prompt on close, and whether closing can throw away their work. It had no
direct coverage while it lived on the window.
"""

from __future__ import annotations

from pathlib import Path

from blab.ui.main_window.project_session import ProjectSession, canonical_payload


def _payload(name: str = "waveguide", **extra) -> dict:
    return {
        "generator_documents": [{"id": "doc-1", "name": name}],
        "symmetry": "off",
        **extra,
    }


# -------------------------------------------------------------- canonical form


def test_key_order_does_not_count_as_an_edit() -> None:
    one = {"symmetry": "off", "generator_documents": []}
    other = {"generator_documents": [], "symmetry": "off"}

    assert canonical_payload(one) == canonical_payload(other)


def test_regenerating_geometry_does_not_dirty_an_untouched_project() -> None:
    """The artifact block records where geometry landed on disk, not user intent."""
    before = _payload()
    after = {"generator_documents": [{"id": "doc-1", "name": "waveguide", "artifact": {"mesh": "runs/x.msh"}}]}

    assert canonical_payload(after)["generator_documents"][0] == {"id": "doc-1", "name": "waveguide"}
    assert canonical_payload(before)["generator_documents"][0] == {"id": "doc-1", "name": "waveguide"}


def test_a_real_content_edit_survives_canonicalisation() -> None:
    assert canonical_payload(_payload("horn")) != canonical_payload(_payload("waveguide"))


def test_canonicalising_does_not_mutate_the_input() -> None:
    payload = _payload()
    payload["generator_documents"][0]["artifact"] = {"mesh": "runs/x.msh"}

    canonical_payload(payload)

    assert "artifact" in payload["generator_documents"][0], "the caller's payload must be left alone"


def test_documents_that_are_not_dicts_are_tolerated() -> None:
    """Payloads come off disk, so they cannot be assumed well-formed."""
    assert canonical_payload({"generator_documents": ["junk", None, 3]})["generator_documents"] == ["junk", None, 3]


# -------------------------------------------------------------- change tracking


def test_a_session_reports_no_changes_before_tracking_starts() -> None:
    """A half-built window must never claim the user has unsaved work."""
    session = ProjectSession()

    assert session.clean_payload is None
    assert session.has_unsaved_changes(_payload()) is False


def test_no_changes_immediately_after_marking_clean() -> None:
    session = ProjectSession()

    session.mark_clean(_payload())

    assert session.has_unsaved_changes(_payload()) is False


def test_an_edit_after_marking_clean_is_reported() -> None:
    session = ProjectSession()
    session.mark_clean(_payload("waveguide"))

    assert session.has_unsaved_changes(_payload("horn")) is True


def test_generating_geometry_after_saving_leaves_the_project_clean() -> None:
    """Regression: the whole point of dropping the artifact block."""
    session = ProjectSession()
    session.mark_clean(_payload())

    regenerated = _payload()
    regenerated["generator_documents"][0]["artifact"] = {"mesh": "runs/y.msh"}

    assert session.has_unsaved_changes(regenerated) is False


def test_saving_again_clears_the_dirty_flag() -> None:
    session = ProjectSession()
    session.mark_clean(_payload("waveguide"))
    assert session.has_unsaved_changes(_payload("horn")) is True

    session.mark_clean(_payload("horn"))

    assert session.has_unsaved_changes(_payload("horn")) is False


# -------------------------------------------------------------- file identity


def test_a_new_project_has_no_file_on_disk() -> None:
    session = ProjectSession()

    assert session.path is None
    assert session.is_saved_to_disk is False


def test_a_session_knows_once_it_has_a_file() -> None:
    session = ProjectSession()
    session.path = Path("projects/demo.blab.json")

    assert session.is_saved_to_disk is True


def test_replacing_the_project_restarts_change_tracking() -> None:
    """A freshly loaded project is not 'clean' until it is marked so."""
    session = ProjectSession()
    session.mark_clean(_payload())
    loaded = ProjectSession().document

    session.replace(loaded, path=Path("projects/demo.blab.json"))

    assert session.document is loaded
    assert session.path == Path("projects/demo.blab.json")
    assert session.clean_payload is None
    assert session.has_unsaved_changes(_payload()) is False
