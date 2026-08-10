"""Properties editor for a project observation plane."""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from blab.observation_planes import (
    EVALUATION_POINT_WARNING_THRESHOLD,
    InteriorRenderingMode,
    ObservationPlane,
    ObservationPlaneDisplay,
    ObservationPlaneType,
)

PLANE_TYPE_LABELS = {
    ObservationPlaneType.INTERIOR: "Interior",
    ObservationPlaneType.EXTERIOR: "Exterior",
    ObservationPlaneType.COMBINED: "Combined",
}
DISPLAY_LABELS = {
    ObservationPlaneDisplay.SPL: "SPL",
    ObservationPlaneDisplay.PHASE: "Phase",
    ObservationPlaneDisplay.REAL_PRESSURE: "Real Pressure",
    ObservationPlaneDisplay.IMAGINARY_PRESSURE: "Imaginary Pressure",
    ObservationPlaneDisplay.NORMALIZED_SPL: "Normalized SPL",
}
INTERIOR_RENDERING_LABELS = {
    InteriorRenderingMode.SMOOTH_FIELD: "Smooth Field",
    InteriorRenderingMode.ELEMENT_FIELD: "Element Field",
}


class ObservationPlanePropertiesDialog(QDialog):
    """Edit authoring and future result-display properties for one plane."""

    def __init__(
        self,
        plane: ObservationPlane,
        parent: QWidget | None = None,
        *,
        solved_frequencies_hz: tuple[float, ...] = (),
        response_options: tuple[tuple[str, str], ...] = (),
    ) -> None:
        super().__init__(parent)
        self._plane = plane.validated()
        self._solved_frequencies_hz = tuple(float(value) for value in solved_frequencies_hz)
        self.setWindowTitle(f"{self._plane.name} Properties")
        self.setMinimumWidth(410)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit(self._plane.name)
        form.addRow("Name", self.name_edit)

        self.type_combo = QComboBox()
        for value, label in PLANE_TYPE_LABELS.items():
            self.type_combo.addItem(label, value.value)
        self._select_data(self.type_combo, self._plane.plane_type.value)
        form.addRow("Type", self.type_combo)

        size_row = QHBoxLayout()
        self.width_spin = _millimetre_spin(self._plane.width_m * 1000.0)
        self.height_spin = _millimetre_spin(self._plane.height_m * 1000.0)
        size_row.addWidget(self.width_spin)
        size_row.addWidget(QLabel("×"))
        size_row.addWidget(self.height_spin)
        form.addRow("Size X × Y (mm)", size_row)

        self.resolution_spin = _millimetre_spin(self._plane.resolution_m * 1000.0, minimum=0.01)
        form.addRow("Resolution (mm)", self.resolution_spin)

        self.point_count_label = QLabel()
        self.point_count_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("Evaluation points", self.point_count_label)

        self.display_combo = QComboBox()
        for value, label in DISPLAY_LABELS.items():
            self.display_combo.addItem(label, value.value)
        self._select_data(self.display_combo, self._plane.display.value)
        form.addRow("Display", self.display_combo)

        self.rendering_combo = QComboBox()
        for value, label in INTERIOR_RENDERING_LABELS.items():
            self.rendering_combo.addItem(label, value.value)
        self._select_data(self.rendering_combo, self._plane.interior_rendering.value)
        form.addRow("Interior rendering", self.rendering_combo)

        self.invert_clip_check = QCheckBox("Invert the clipped side of the FEM volume")
        self.invert_clip_check.setChecked(self._plane.invert_clip_side)
        form.addRow("Clipping", self.invert_clip_check)
        layout.addLayout(form)

        results_group = QGroupBox("Solved Result")
        results_form = QFormLayout(results_group)
        self.frequency_slider = QSlider(Qt.Horizontal)
        self.frequency_slider.setRange(0, max(len(self._solved_frequencies_hz) - 1, 0))
        self.frequency_slider.setEnabled(bool(self._solved_frequencies_hz))
        self.frequency_label = QLabel("No solved plane data available")
        frequency_row = QVBoxLayout()
        frequency_row.addWidget(self.frequency_slider)
        frequency_row.addWidget(self.frequency_label)
        results_form.addRow("Frequency", frequency_row)

        self.response_combo = QComboBox()
        options = response_options or (("system", "System Response"),)
        for response_id, label in options:
            self.response_combo.addItem(label, response_id)
        self._select_data(self.response_combo, self._plane.response_id)
        self.response_combo.setEnabled(bool(self._solved_frequencies_hz))
        results_form.addRow("Response", self.response_combo)

        self.animate_button = QPushButton("Animate Phase")
        self.animate_button.setCheckable(True)
        self.animate_button.setEnabled(False)
        self.animation_speed_slider = QSlider(Qt.Horizontal)
        self.animation_speed_slider.setRange(1, 40)
        self.animation_speed_slider.setValue(round(self._plane.animation_speed_hz * 10.0))
        self.animation_speed_slider.setEnabled(False)
        animation_row = QHBoxLayout()
        animation_row.addWidget(self.animate_button)
        animation_row.addWidget(QLabel("Speed"))
        animation_row.addWidget(self.animation_speed_slider, 1)
        results_form.addRow("Phase", animation_row)
        layout.addWidget(results_group)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.type_combo.currentIndexChanged.connect(self._refresh_dependent_controls)
        self.rendering_combo.currentIndexChanged.connect(self._refresh_dependent_controls)
        self.width_spin.valueChanged.connect(self._refresh_point_count)
        self.height_spin.valueChanged.connect(self._refresh_point_count)
        self.resolution_spin.valueChanged.connect(self._refresh_point_count)
        self.frequency_slider.valueChanged.connect(self._refresh_frequency_label)
        self._refresh_dependent_controls()
        self._refresh_point_count()
        self._refresh_frequency_label()

    @property
    def plane(self) -> ObservationPlane:
        return replace(
            self._plane,
            name=self.name_edit.text().strip() or self._plane.name,
            width_m=self.width_spin.value() / 1000.0,
            height_m=self.height_spin.value() / 1000.0,
            resolution_m=self.resolution_spin.value() / 1000.0,
            plane_type=ObservationPlaneType(self.type_combo.currentData()),
            display=ObservationPlaneDisplay(self.display_combo.currentData()),
            interior_rendering=InteriorRenderingMode(self.rendering_combo.currentData()),
            invert_clip_side=self.invert_clip_check.isChecked(),
            response_id=str(self.response_combo.currentData() or "system"),
            animation_speed_hz=self.animation_speed_slider.value() / 10.0,
        ).validated()

    def _refresh_dependent_controls(self) -> None:
        supports_interior = self.type_combo.currentData() in {
            ObservationPlaneType.INTERIOR.value,
            ObservationPlaneType.COMBINED.value,
        }
        self.rendering_combo.setEnabled(supports_interior)
        self.invert_clip_check.setEnabled(supports_interior)
        element_field = (
            supports_interior and self.rendering_combo.currentData() == InteriorRenderingMode.ELEMENT_FIELD.value
        )
        self.resolution_spin.setEnabled(not element_field)
        self._refresh_point_count()

    def _refresh_point_count(self) -> None:
        preview = replace(
            self._plane,
            width_m=self.width_spin.value() / 1000.0,
            height_m=self.height_spin.value() / 1000.0,
            resolution_m=self.resolution_spin.value() / 1000.0,
        ).validated()
        if not self.resolution_spin.isEnabled():
            self.point_count_label.setText("Not used by Element Field")
            self.point_count_label.setStyleSheet("")
            return
        count = preview.evaluation_point_count
        self.point_count_label.setText(f"{count:,}")
        if count > EVALUATION_POINT_WARNING_THRESHOLD:
            self.point_count_label.setText(
                f"{count:,} — warning: more than {EVALUATION_POINT_WARNING_THRESHOLD:,} evaluation points"
            )
            self.point_count_label.setStyleSheet("color: #e6a23c;")
        else:
            self.point_count_label.setStyleSheet("")

    def _refresh_frequency_label(self) -> None:
        if not self._solved_frequencies_hz:
            self.frequency_label.setText("No solved plane data available")
            return
        index = min(self.frequency_slider.value(), len(self._solved_frequencies_hz) - 1)
        self.frequency_label.setText(f"{self._solved_frequencies_hz[index]:g} Hz")

    @staticmethod
    def _select_data(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)


def _millimetre_spin(value: float, *, minimum: float = 0.1) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, 1_000_000.0)
    spin.setDecimals(2)
    spin.setSingleStep(1.0)
    spin.setSuffix(" mm")
    spin.setValue(float(value))
    return spin
