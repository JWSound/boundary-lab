from pathlib import Path

from blab.speaker_package import SpeakerPackageFidelity
from blab.ui.speaker_package_dialog import SpeakerPackageDialog


def test_speaker_package_dialog_exposes_solve_and_export_configuration(qapp, tmp_path: Path) -> None:
    dialog = SpeakerPackageDialog(default_name="Monitor A")
    try:
        assert dialog.solve_export_button.text() == "Solve and Export"
        assert dialog.fidelity_combo.count() == 3
        dialog.output_edit.setText(str(tmp_path / "monitor-a"))
        dialog.fidelity_combo.setCurrentIndex(2)

        config = dialog.config()

        assert config.name == "Monitor A"
        assert config.output_path == tmp_path / "monitor-a.blabsp"
        assert config.fidelity == SpeakerPackageFidelity.COUPLED
    finally:
        dialog.close()
        dialog.deleteLater()
