from pathlib import Path

from blab.speaker_package import SpeakerPackageCoupledRepresentation, SpeakerPackageFidelity
from blab.ui.speaker_package_dialog import SpeakerPackageDialog


def test_speaker_package_dialog_exposes_solve_and_export_configuration(qapp, tmp_path: Path) -> None:
    dialog = SpeakerPackageDialog(default_name="Monitor A")
    try:
        assert dialog.solve_export_button.text() == "Solve and Export"
        assert dialog.fidelity_combo.count() == 3
        assert not hasattr(dialog, "coupled_representation_combo")
        dialog.output_edit.setText(str(tmp_path / "monitor-a"))
        dialog.fidelity_combo.setCurrentIndex(2)

        config = dialog.config()

        assert config.name == "Monitor A"
        assert config.output_path == tmp_path / "monitor-a.blabsp"
        assert config.fidelity == SpeakerPackageFidelity.COUPLED
        assert config.coupled_representation == SpeakerPackageCoupledRepresentation.PARITY_ROM
        assert config.viewport_model_path is None
        obj = tmp_path / "cabinet.obj"
        obj.write_text("v 0 0 0\nv 120 0 0\nv 0 52.8 -80\nf 1 2 3\n")
        (tmp_path / "cabinet.mtl").write_text("newmtl wood\nKd 0.5 0.3 0.1\n")
        dialog.viewport_edit.setText(str(obj))
        dialog.viewport_units.setCurrentIndex(1)
        config = dialog.config()
        assert config.viewport_model_path == obj
        assert config.viewport_model_scale_to_m == 0.01
        assert "cabinet.mtl" in dialog.viewport_status.text()
        assert "1.2 × 0.528 × 0.8 m" in dialog.viewport_status.text()
    finally:
        dialog.close()
        dialog.deleteLater()
