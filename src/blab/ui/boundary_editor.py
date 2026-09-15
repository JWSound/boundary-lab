"""Focused physical-system boundary editor widgets."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from blab.acoustic_materials import (
    DEFAULT_WALL_LINING_FLOW_RESISTIVITY_PA_S_PER_M2,
    DEFAULT_WALL_LINING_THICKNESS_M,
    miki_wall_impedance_parameters,
    wall_impedance_parameters,
)


class _WallImpedanceDialog(QDialog):
    def __init__(self, parameters: dict | None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Wall Impedance")
        treatment = wall_impedance_parameters(parameters)

        self.enabled_check = QCheckBox("Enable porous wall lining")
        self.enabled_check.setChecked(treatment is not None)
        self.thickness_spin = QDoubleSpinBox()
        self.thickness_spin.setRange(0.1, 1000.0)
        self.thickness_spin.setDecimals(1)
        self.thickness_spin.setSingleStep(5.0)
        self.thickness_spin.setSuffix(" mm")
        self.thickness_spin.setValue(
            1000.0 * float(DEFAULT_WALL_LINING_THICKNESS_M if treatment is None else treatment["thickness_m"])
        )
        self.flow_resistivity_spin = QDoubleSpinBox()
        self.flow_resistivity_spin.setRange(1.0, 10_000_000.0)
        self.flow_resistivity_spin.setDecimals(0)
        self.flow_resistivity_spin.setSingleStep(500.0)
        self.flow_resistivity_spin.setGroupSeparatorShown(True)
        self.flow_resistivity_spin.setSuffix(" Pa·s/m²")
        self.flow_resistivity_spin.setValue(
            float(
                DEFAULT_WALL_LINING_FLOW_RESISTIVITY_PA_S_PER_M2
                if treatment is None
                else treatment["flow_resistivity_pa_s_per_m2"]
            )
        )
        self.enabled_check.toggled.connect(self.thickness_spin.setEnabled)
        self.enabled_check.toggled.connect(self.flow_resistivity_spin.setEnabled)
        self.thickness_spin.setEnabled(self.enabled_check.isChecked())
        self.flow_resistivity_spin.setEnabled(self.enabled_check.isChecked())

        note = QLabel(
            "Rigid-backed porous lining approximation. Generic loose polyfill defaults to 30 mm and 5,000 Pa·s/m²."
        )
        note.setWordWrap(True)
        form = QFormLayout()
        form.addRow("Lining thickness", self.thickness_spin)
        form.addRow("Airflow resistivity", self.flow_resistivity_spin)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(note)
        layout.addWidget(self.enabled_check)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def parameters(self) -> dict:
        if not self.enabled_check.isChecked():
            return {}
        return miki_wall_impedance_parameters(
            thickness_m=float(self.thickness_spin.value()) / 1000.0,
            flow_resistivity_pa_s_per_m2=float(self.flow_resistivity_spin.value()),
        )
