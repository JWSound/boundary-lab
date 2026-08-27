from PySide6.QtWidgets import QApplication

from blab.ui.dialogs import MeshConfigDialog, MeshDialogEntry

_APP = QApplication.instance() or QApplication([])


def test_locked_generated_mesh_scale_is_not_editable() -> None:
    dialog = MeshConfigDialog(
        (
            MeshDialogEntry(name="ath", source_file="ath.msh", scale_factor=0.001, locked=True),
            MeshDialogEntry(name="cabinet", source_file="cabinet.msh", scale_factor=0.001),
        )
    )
    try:
        assert not dialog.scale_widgets[0].isEnabled()
        assert dialog.scale_widgets[1].isEnabled()
    finally:
        dialog.close()
