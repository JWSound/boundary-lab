"""The open project: its document, its file, and whether it has been edited.

Third of the shared-state stores, after
:mod:`~blab.ui.main_window.geometry_store` and
:mod:`~blab.ui.main_window.solve_session`. Holding these three attributes here
rather than on the window makes the sharing explicit and, more usefully, makes
the unsaved-changes rule testable without a Qt application — it decides whether
the user gets a modal prompt on close, so it is worth being able to check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from blab.ui.project_state import ProjectDocument, new_project_document


def canonical_payload(payload: dict) -> dict:
    """A project payload reduced to what counts as user-visible content.

    Key order is normalised, and each design's ``artifact`` block is dropped:
    that records where generated geometry landed on disk, so regenerating a
    waveguide would otherwise mark an untouched project as edited.
    """
    canonical = json.loads(json.dumps(payload, sort_keys=True))
    for document in canonical.get("generator_documents", []):
        if isinstance(document, dict):
            document.pop("artifact", None)
    return canonical


@dataclass
class ProjectSession:
    """The currently open project document and its save state."""

    document: ProjectDocument = field(default_factory=new_project_document)

    #: Where the project was last saved, or None for one that has never been.
    path: Path | None = None

    #: Canonical payload as of the last save. None means change tracking has
    #: not started yet, and nothing is reported as unsaved.
    clean_payload: dict | None = None

    # -- derivations -------------------------------------------------------

    @property
    def is_saved_to_disk(self) -> bool:
        return self.path is not None

    def has_unsaved_changes(self, current_payload: dict) -> bool:
        """Whether the project differs from the last saved state.

        Returns False before the first :meth:`mark_clean`, so a window that is
        still being built never claims unsaved work.
        """
        if self.clean_payload is None:
            return False
        return canonical_payload(current_payload) != self.clean_payload

    # -- mutation ----------------------------------------------------------

    def mark_clean(self, current_payload: dict) -> None:
        """Record the current content as the saved baseline."""
        self.clean_payload = canonical_payload(current_payload)

    def replace(self, document: ProjectDocument, *, path: Path | None) -> None:
        """Swap in a freshly created or freshly loaded project."""
        self.document = document
        self.path = path
        self.clean_payload = None
