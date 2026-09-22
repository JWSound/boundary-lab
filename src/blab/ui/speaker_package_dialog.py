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
from blab.viewport_model import viewport_model_members


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

        self.viewport_edit = QLineEdit()
        self.viewport_edit.setPlaceholderText("Optional expanded OBJ, +Z forward")
        self.viewport_edit.setClearButtonEnabled(True)
        model_button = QPushButton("Browse…")
        model_button.clicked.connect(self._browse_viewport)
        model_row = QHBoxLayout()
        model_row.addWidget(self.viewport_edit, 1)
        model_row.addWidget(model_button)
        self.viewport_units = QComboBox()
        for label, scale in (("Meters", 1.0), ("Centimeters", 0.01), ("Millimeters", 0.001), ("Inches", 0.0254)):
            self.viewport_units.addItem(label, scale)
        self.viewport_status = QLabel("Uses the acoustic mesh when no OBJ is attached.")
        self.viewport_status.setWordWrap(True)
        self.viewport_edit.editingFinished.connect(self._update_viewport_status)
        self.viewport_units.currentIndexChanged.connect(self._update_viewport_status)

        form = QFormLayout()
        form.addRow("Package name", self.name_edit)
        form.addRow("Fidelity", self.fidelity_combo)
        form.addRow("Output file", output_row)
        form.addRow("Viewport model", model_row)
        form.addRow("OBJ units", self.viewport_units)
        form.addRow(self.viewport_status)

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
            viewport_model_path=Path(self.viewport_edit.text().strip()) if self.viewport_edit.text().strip() else None,
            viewport_model_scale_to_m=float(self.viewport_units.currentData()),
        ).normalized()

    @Slot()
    def _browse_viewport(self) -> None:
        path = self.file_dialogs.open_file(self, "Attach viewport model", "Wavefront OBJ (*.obj)")
        if path is not None:
            self.viewport_edit.setText(str(path))
            self._update_viewport_status()

    def _update_viewport_status(self, *_args) -> None:
        value = self.viewport_edit.text().strip()
        if not value:
            self.viewport_status.setText("Uses the acoustic mesh when no OBJ is attached.")
            return
        try:
            _, descriptor = viewport_model_members(Path(value), float(self.viewport_units.currentData()))
            materials = ", ".join(descriptor["materials_detected"]) or "None (default material)"
            sizes = [b - a for a, b in zip(descriptor["bounds_min_m"], descriptor["bounds_max_m"])]
            dimensions = f"Width × height × depth: {sizes[0]:.4g} × {sizes[2]:.4g} × {sizes[1]:.4g} m"
            self.viewport_status.setText(
                dimensions + "\nMaterials: " + materials + "\n" + "\n".join(descriptor["warnings"])
            )
        except ValueError as exc:
            self.viewport_status.setText(str(exc))

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
        self._update_viewport_status()
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
