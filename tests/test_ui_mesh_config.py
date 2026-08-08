from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from blab.ui.dialogs import MeshConfigDialog, MeshDialogEntry

_APP = QApplication.instance() or QApplication([])


class _FileDialogsStub:
    def __init__(self, selected_path: Path):
        self.selected_path = selected_path
        self.calls = 0

    def open_file(self, *_args) -> Path:
        self.calls += 1
        return self.selected_path


def test_replace_mesh_preserves_row_identity_and_invalidates_cleaned_file(tmp_path: Path) -> None:
    replacement = tmp_path / "replacement.msh"
    file_dialogs = _FileDialogsStub(replacement)
    dialog = MeshConfigDialog(
        (
            MeshDialogEntry(
                name="Exterior",
                source_file=str(tmp_path / "original.msh"),
                cleaned_file=str(tmp_path / "original_clean.msh"),
                scale_factor=0.002,
                translation_mm=(1.0, 2.0, 3.0),
                enabled=False,
            ),
        ),
        file_dialog_service=file_dialogs,
    )

    dialog.table.selectRow(0)
    assert dialog.replace_button.isEnabled()
    dialog._replace_mesh()

    (mesh,) = dialog.meshes()
    assert mesh == MeshDialogEntry(
        name="Exterior",
        source_file=str(replacement),
        cleaned_file=None,
        scale_factor=0.002,
        translation_mm=(1.0, 2.0, 3.0),
        enabled=False,
    )
    assert dialog.replaced_mesh_names() == ("Exterior",)
    assert file_dialogs.calls == 1


def test_replace_mesh_is_disabled_for_generated_locked_rows(tmp_path: Path) -> None:
    file_dialogs = _FileDialogsStub(tmp_path / "replacement.msh")
    dialog = MeshConfigDialog(
        (
            MeshDialogEntry(
                name="Generated",
                source_file=str(tmp_path / "generated.msh"),
                locked=True,
            ),
        ),
        file_dialog_service=file_dialogs,
    )

    dialog.table.selectRow(0)
    assert not dialog.replace_button.isEnabled()
    dialog._replace_mesh()

    assert dialog.meshes()[0].source_file == str(tmp_path / "generated.msh")
    assert dialog.replaced_mesh_names() == ()
    assert file_dialogs.calls == 0
