"""Configuration dialog for solve-and-export speaker packages."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from blab.speaker_package import (
    SpeakerPackageConfig,
    SpeakerPackageCoupledRepresentation,
    SpeakerPackageFidelity,
)
from blab.ui.file_dialogs import FileDialogService


class SpeakerPackageDialog(QDialog):
    """Collect package settings before starting the background solve."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        default_name: str = "Speaker",
        file_dialogs: FileDialogService | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Speaker Package")
        self.setMinimumWidth(520)
        self.file_dialogs = file_dialogs or FileDialogService()

        self.name_edit = QLineEdit(default_name)
        self.fidelity_combo = QComboBox()
        self.fidelity_combo.addItem("Level 1 — Pattern superposition", int(SpeakerPackageFidelity.PATTERN))
        self.fidelity_combo.addItem(
            "Level 2 — Exterior BEM with fixed distributed sources",
            int(SpeakerPackageFidelity.FIXED_SOURCES),
        )
        self.fidelity_combo.addItem(
            "Level 3 — Dynamic interior with coupled exterior BEM",
            int(SpeakerPackageFidelity.COUPLED),
        )
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("speaker.blabsp")
        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(self._browse)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(browse_button)

        form = QFormLayout()
        form.addRow("Package name", self.name_edit)
        form.addRow("Fidelity", self.fidelity_combo)
        form.addRow("Output file", output_row)

        note = QLabel(
            "Boundary Lab will run a new solve with the complex spherical field and any boundary traces "
            "required by the selected fidelity. Level 3 stores a rank-32 parity Petrov–Galerkin ROM "
            "with one, two, or four sectors according to the project's symmetry. Exported packages use "
            "+Y as forward."
        )
        note.setWordWrap(True)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.solve_export_button = buttons.addButton("Solve and Export", QDialogButtonBox.AcceptRole)
        self.solve_export_button.clicked.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(buttons)

    def config(self) -> SpeakerPackageConfig:
        return SpeakerPackageConfig(
            output_path=Path(self.output_edit.text().strip()),
            name=self.name_edit.text().strip(),
            fidelity=SpeakerPackageFidelity(int(self.fidelity_combo.currentData())),
            coupled_representation=SpeakerPackageCoupledRepresentation.PARITY_ROM,
        ).normalized()

    @Slot()
    def _browse(self) -> None:
        suggested = self.output_edit.text().strip() or f"{_safe_filename(self.name_edit.text())}.blabsp"
        path = self.file_dialogs.save_file(
            self,
            "Export speaker package",
            "Boundary Lab speaker packages (*.blabsp);;All files (*)",
            suggested,
        )
        if path is not None:
            self.output_edit.setText(str(_with_package_suffix(path)))

    @Slot()
    def _accept_if_valid(self) -> None:
        if not self.output_edit.text().strip():
            QMessageBox.warning(self, "Output file required", "Choose an output .blabsp file.")
            return
        try:
            config = self.config()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid speaker package", str(exc))
            return
        self.output_edit.setText(str(config.output_path))
        self.accept()


def _safe_filename(value: str) -> str:
    name = "".join(char if char.isalnum() or char in "-_" else "_" for char in value.strip()).strip("_")
    return name or "speaker"


def _with_package_suffix(path: Path) -> Path:
    return path if path.suffix.lower() == ".blabsp" else path.with_suffix(".blabsp")


__all__ = ["SpeakerPackageDialog"]
