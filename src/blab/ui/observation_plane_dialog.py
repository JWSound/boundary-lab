"""Properties editor for a project observation plane."""

from __future__ import annotations

import time
from dataclasses import replace

from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
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
    ObservationPlaneDisplay.FLOW_ARROWS: "Acoustic Flow Arrows",
    ObservationPlaneDisplay.SPL: "SPL",
    ObservationPlaneDisplay.PHASE: "Phase",
    ObservationPlaneDisplay.REAL_PRESSURE: "Real Pressure",
    ObservationPlaneDisplay.IMAGINARY_PRESSURE: "Imaginary Pressure",
    ObservationPlaneDisplay.NORMALIZED_SPL: "Normalized SPL",
    ObservationPlaneDisplay.PARTICLE_VELOCITY: "Particle Velocity Magnitude",
}
INTERIOR_RENDERING_LABELS = {
    InteriorRenderingMode.SMOOTH_FIELD: "Smooth Field",
    InteriorRenderingMode.ELEMENT_FIELD: "Element Field",
}
FREQUENCY_SWEEP_INTERVAL_MS = 50
DEFAULT_FREQUENCY_SWEEP_SPEED = 5


class ObservationPlanePropertiesDialog(QDialog):
    """Edit authoring and future result-display properties for one plane."""

    previewChanged = Signal(object, bool)

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
        self._solved_frequencies_hz = tuple(sorted(float(value) for value in solved_frequencies_hz))
        self._preview_ready = False
        self.setWindowTitle(f"{self._plane.name} Properties")
        self.setMinimumWidth(410)

        layout = QVBoxLayout(self)
        form = self._form = QFormLayout()
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

        self.display_combo = QComboBox()
        for value, label in DISPLAY_LABELS.items():
            self.display_combo.addItem(label, value.value)
        self._select_data(self.display_combo, self._plane.display.value)
        form.addRow("Display", self.display_combo)

        self.arrow_density_slider = QSlider(Qt.Horizontal)
        self.arrow_density_slider.setRange(5, 60)
        self.arrow_density_slider.setValue(self._plane.arrow_density)
        self.arrow_density_slider.setToolTip(
            "Arrows across the longer side of the plane. Static: active intensity; animated: particle velocity."
        )
        form.addRow("Arrow density", self.arrow_density_slider)

        self._scale_rows = {}
        for prefix, label, unit, initial in (
            ("velocity", "Instantaneous velocity scale", " m/s", self._plane.velocity_scale_m_per_s),
            ("intensity", "Active intensity scale", " W/m^2", self._plane.intensity_scale_w_per_m2),
            ("particle_velocity", "Velocity color maximum", " m/s", self._plane.particle_velocity_limit_m_per_s),
        ):
            automatic = QCheckBox("Automatic")
            automatic.setChecked(initial is None)
            limit = QDoubleSpinBox()
            limit.setDecimals(2)
            limit.setRange(0.01, 1e9)
            limit.setSingleStep(0.01)
            limit.setSuffix(unit)
            limit.setValue(1.0 if initial is None else initial)
            limit.setToolTip(
                "Maximum for the velocity colour scale. Values above this limit are saturated."
                if prefix == "particle_velocity"
                else "Full-scale magnitude for arrow length and colour. Values above this limit are saturated."
            )
            setattr(self, f"{prefix}_scale_auto_check", automatic)
            setattr(self, f"{prefix}_scale_spin", limit)
            row = QHBoxLayout()
            row.addWidget(automatic)
            row.addWidget(limit, 1)
            form.addRow(label, row)
            self._scale_rows[prefix] = row

        self.pressure_range_auto_check = QCheckBox("Automatic")
        self.pressure_range_auto_check.setChecked(self._plane.pressure_color_limit_pa is None)
        self.pressure_limit_spin = QDoubleSpinBox()
        self.pressure_limit_spin.setRange(1.0, 1e9)
        self.pressure_limit_spin.setDecimals(0)
        self.pressure_limit_spin.setSingleStep(1.0)
        self.pressure_limit_spin.setSuffix(" Pa")
        self.pressure_limit_spin.setValue(
            10.0 if self._plane.pressure_color_limit_pa is None else self._plane.pressure_color_limit_pa
        )
        self.pressure_limit_spin.setToolTip(
            "Set the symmetric pressure color range from -limit to +limit. Values outside it are saturated."
        )
        pressure_range_row = self._pressure_range_row = QHBoxLayout()
        pressure_range_row.addWidget(self.pressure_range_auto_check)
        pressure_range_row.addWidget(QLabel("+/-"))
        pressure_range_row.addWidget(self.pressure_limit_spin, 1)
        form.addRow("Pressure color range", pressure_range_row)

        self.spl_range_auto_check = QCheckBox("Automatic")
        self.spl_range_auto_check.setToolTip("Use a 60 dB colour range below the current plane maximum.")
        self.spl_range_auto_check.setChecked(self._plane.spl_min_db is None)
        self.spl_min_spin = QDoubleSpinBox()
        self.spl_max_spin = QDoubleSpinBox()
        for spin in (self.spl_min_spin, self.spl_max_spin):
            spin.setRange(-1000, 1000)
            spin.setDecimals(2)
            spin.setSuffix(" dB")
        self.spl_min_spin.setValue(60 if self._plane.spl_min_db is None else self._plane.spl_min_db)
        self.spl_max_spin.setValue(120 if self._plane.spl_max_db is None else self._plane.spl_max_db)
        self.spl_min_spin.setMaximum(self.spl_max_spin.value() - 0.01)
        self.spl_max_spin.setMinimum(self.spl_min_spin.value() + 0.01)
        self.spl_min_spin.valueChanged.connect(lambda value: self.spl_max_spin.setMinimum(value + 0.01))
        self.spl_max_spin.valueChanged.connect(lambda value: self.spl_min_spin.setMaximum(value - 0.01))
        self._spl_range_row = QHBoxLayout()
        self._spl_range_row.addWidget(self.spl_range_auto_check)
        self._spl_range_row.addWidget(self.spl_min_spin)
        self._spl_range_row.addWidget(QLabel("to"))
        self._spl_range_row.addWidget(self.spl_max_spin)
        form.addRow("SPL color range", self._spl_range_row)
        self.normalized_range_spin = QDoubleSpinBox()
        self.normalized_range_spin.setDecimals(2)
        self.normalized_range_spin.setRange(0.01, 1000)
        self.normalized_range_spin.setSuffix(" dB")
        self.normalized_range_spin.setValue(self._plane.normalized_spl_range_db)
        self.normalized_range_spin.setToolTip("Colour range below the plane maximum: -range to 0 dB.")
        form.addRow("Normalized SPL range", self.normalized_range_spin)

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
        results_form = self._results_form = QFormLayout(results_group)
        self.frequency_slider = QSlider(Qt.Horizontal)
        self.frequency_slider.setRange(0, max(len(self._solved_frequencies_hz) - 1, 0))
        self.frequency_slider.setEnabled(bool(self._solved_frequencies_hz))
        if self._solved_frequencies_hz and self._plane.frequency_hz is not None:
            self.frequency_slider.setValue(
                min(
                    range(len(self._solved_frequencies_hz)),
                    key=lambda index: abs(self._solved_frequencies_hz[index] - self._plane.frequency_hz),
                )
            )
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

        self.animate_button = QPushButton("Animate Cycle")
        self.animate_button.setCheckable(True)
        self.animate_button.setToolTip(
            "Animate one acoustic cycle at the selected frequency. Speed controls playback, not the solved frequency."
        )
        self.animate_button.setEnabled(bool(self._solved_frequencies_hz))
        self.animation_speed_slider = QSlider(Qt.Horizontal)
        self.animation_speed_slider.setRange(1, 40)
        self.animation_speed_slider.setValue(round(self._plane.animation_speed_hz * 10.0))
        self.animation_speed_slider.setEnabled(False)
        animation_row = self._animation_row = QHBoxLayout()
        animation_row.addWidget(self.animate_button)
        self._animation_speed_label = QLabel("Speed")
        animation_row.addWidget(self._animation_speed_label)
        animation_row.addWidget(self.animation_speed_slider, 1)
        results_form.addRow("Cycle", animation_row)

        self.frequency_sweep_button = QPushButton("Sweep Frequency")
        self.frequency_sweep_button.setCheckable(True)
        self.frequency_sweep_button.setEnabled(len(self._solved_frequencies_hz) > 1)
        self.frequency_sweep_speed_slider = QSlider(Qt.Horizontal)
        self.frequency_sweep_speed_slider.setRange(1, 40)
        self.frequency_sweep_speed_slider.setValue(DEFAULT_FREQUENCY_SWEEP_SPEED)
        self.frequency_sweep_speed_slider.setEnabled(False)
        sweep_row = QHBoxLayout()
        sweep_row.addWidget(self.frequency_sweep_button)
        self._sweep_speed_label = QLabel("Steps/s")
        sweep_row.addWidget(self._sweep_speed_label)
        sweep_row.addWidget(self.frequency_sweep_speed_slider, 1)
        results_form.addRow("Frequency sweep", sweep_row)
        layout.addWidget(results_group)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.type_combo.currentIndexChanged.connect(self._refresh_dependent_controls)
        self.rendering_combo.currentIndexChanged.connect(self._refresh_dependent_controls)
        self.display_combo.currentIndexChanged.connect(self._refresh_dependent_controls)
        self.pressure_range_auto_check.toggled.connect(self._refresh_dependent_controls)
        self.velocity_scale_auto_check.toggled.connect(self._refresh_dependent_controls)
        self.intensity_scale_auto_check.toggled.connect(self._refresh_dependent_controls)
        self.particle_velocity_scale_auto_check.toggled.connect(self._refresh_dependent_controls)
        self.spl_range_auto_check.toggled.connect(self._refresh_dependent_controls)
        self.frequency_slider.valueChanged.connect(self._refresh_frequency_label)
        self._frequency_preview_timer = QTimer(self)
        self._frequency_preview_timer.setSingleShot(True)
        self._frequency_preview_timer.setInterval(16)
        self._frequency_preview_timer.timeout.connect(self._emit_preview)
        self.frequency_slider.valueChanged.connect(self._schedule_frequency_preview)
        self._frequency_sweep_timer = QTimer(self)
        self._frequency_sweep_timer.setInterval(FREQUENCY_SWEEP_INTERVAL_MS)
        self._frequency_sweep_timer.timeout.connect(self._advance_frequency_sweep)
        self._frequency_sweep_position = float(self.frequency_slider.value())
        self._frequency_sweep_direction = 1.0
        self._frequency_sweep_updated_at = time.monotonic()
        self.frequency_slider.sliderPressed.connect(self._stop_frequency_sweep)
        self.animate_button.toggled.connect(self._refresh_animation_controls)
        self.frequency_sweep_button.toggled.connect(self._refresh_frequency_sweep_controls)
        self.frequency_sweep_speed_slider.valueChanged.connect(self._reset_frequency_sweep_clock)
        preview_controls = (
            self.name_edit.textChanged,
            self.type_combo.currentIndexChanged,
            self.width_spin.valueChanged,
            self.height_spin.valueChanged,
            self.resolution_spin.valueChanged,
            self.display_combo.currentIndexChanged,
            self.arrow_density_slider.valueChanged,
            self.velocity_scale_auto_check.toggled,
            self.velocity_scale_spin.valueChanged,
            self.intensity_scale_auto_check.toggled,
            self.intensity_scale_spin.valueChanged,
            self.particle_velocity_scale_auto_check.toggled,
            self.particle_velocity_scale_spin.valueChanged,
            self.spl_range_auto_check.toggled,
            self.spl_min_spin.valueChanged,
            self.spl_max_spin.valueChanged,
            self.normalized_range_spin.valueChanged,
            self.pressure_range_auto_check.toggled,
            self.pressure_limit_spin.valueChanged,
            self.rendering_combo.currentIndexChanged,
            self.invert_clip_check.toggled,
            self.response_combo.currentIndexChanged,
            self.animate_button.toggled,
            self.animation_speed_slider.valueChanged,
        )
        for signal in preview_controls:
            signal.connect(self._emit_preview)
        self._refresh_dependent_controls()
        self._refresh_frequency_label()
        self._refresh_animation_controls()
        self._refresh_frequency_sweep_controls()
        self._preview_ready = True

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
            arrow_density=self.arrow_density_slider.value(),
            velocity_scale_m_per_s=(
                None if self.velocity_scale_auto_check.isChecked() else self.velocity_scale_spin.value()
            ),
            intensity_scale_w_per_m2=(
                None if self.intensity_scale_auto_check.isChecked() else self.intensity_scale_spin.value()
            ),
            particle_velocity_limit_m_per_s=(
                None
                if self.particle_velocity_scale_auto_check.isChecked()
                else self.particle_velocity_scale_spin.value()
            ),
            spl_min_db=None if self.spl_range_auto_check.isChecked() else self.spl_min_spin.value(),
            spl_max_db=None if self.spl_range_auto_check.isChecked() else self.spl_max_spin.value(),
            normalized_spl_range_db=self.normalized_range_spin.value(),
            interior_rendering=InteriorRenderingMode(self.rendering_combo.currentData()),
            invert_clip_side=self.invert_clip_check.isChecked(),
            response_id=str(self.response_combo.currentData() or "system"),
            frequency_hz=self._selected_frequency_hz(),
            animation_speed_hz=self.animation_speed_slider.value() / 10.0,
            pressure_color_limit_pa=(
                None if self.pressure_range_auto_check.isChecked() else self.pressure_limit_spin.value()
            ),
        ).validated()

    def update_geometry(self, plane: ObservationPlane) -> None:
        """Merge viewport-authored geometry without discarding pending properties."""

        plane = plane.validated()
        if plane.id != self._plane.id:
            return
        self._plane = replace(
            self._plane,
            center_m=plane.center_m,
            orientation_wxyz=plane.orientation_wxyz,
            width_m=plane.width_m,
            height_m=plane.height_m,
        )
        width_blocker = QSignalBlocker(self.width_spin)
        height_blocker = QSignalBlocker(self.height_spin)
        self.width_spin.setValue(plane.width_m * 1000.0)
        self.height_spin.setValue(plane.height_m * 1000.0)
        del width_blocker, height_blocker

    def set_solved_results(
        self,
        frequencies_hz: tuple[float, ...],
        response_options: tuple[tuple[str, str], ...],
    ) -> None:
        """Refresh result controls while this modeless editor remains open."""

        current_frequency = self._selected_frequency_hz()
        current_response = str(self.response_combo.currentData() or self._plane.response_id)
        self._solved_frequencies_hz = tuple(sorted(float(value) for value in frequencies_hz))

        frequency_blocker = QSignalBlocker(self.frequency_slider)
        self.frequency_slider.setRange(0, max(len(self._solved_frequencies_hz) - 1, 0))
        self.frequency_slider.setEnabled(bool(self._solved_frequencies_hz))
        if self._solved_frequencies_hz:
            target = self._plane.frequency_hz if current_frequency is None else current_frequency
            self.frequency_slider.setValue(
                0
                if target is None
                else min(
                    range(len(self._solved_frequencies_hz)),
                    key=lambda index: abs(self._solved_frequencies_hz[index] - target),
                )
            )
        del frequency_blocker

        response_blocker = QSignalBlocker(self.response_combo)
        self.response_combo.clear()
        for response_id, label in response_options or (("system", "System Response"),):
            self.response_combo.addItem(label, response_id)
        self._select_data(self.response_combo, current_response)
        self.response_combo.setEnabled(bool(self._solved_frequencies_hz))
        del response_blocker

        if not self._solved_frequencies_hz:
            animation_blocker = QSignalBlocker(self.animate_button)
            self.animate_button.setChecked(False)
            del animation_blocker
        self.animate_button.setEnabled(bool(self._solved_frequencies_hz))
        if len(self._solved_frequencies_hz) < 2:
            self._stop_frequency_sweep()
        self.frequency_sweep_button.setEnabled(len(self._solved_frequencies_hz) > 1)
        self._refresh_frequency_label()
        self._refresh_animation_controls()
        self._refresh_frequency_sweep_controls()

    def _refresh_dependent_controls(self) -> None:
        plane_type = self.type_combo.currentData()
        interior_only = plane_type == ObservationPlaneType.INTERIOR.value
        interior_displays = (ObservationPlaneDisplay.PARTICLE_VELOCITY, ObservationPlaneDisplay.FLOW_ARROWS)
        for display in interior_displays:
            item = self.display_combo.model().item(self.display_combo.findData(display.value))
            if item is not None:
                item.setEnabled(interior_only)
        if not interior_only and self.display_combo.currentData() in {d.value for d in interior_displays}:
            self._select_data(self.display_combo, ObservationPlaneDisplay.SPL.value)
        display = ObservationPlaneDisplay(self.display_combo.currentData())
        supports_cycle = display.supports_cycle_animation
        if not supports_cycle and self.animate_button.isChecked():
            self.animate_button.setChecked(False)
        animated = supports_cycle and self.animate_button.isChecked()
        self.animate_button.setEnabled(supports_cycle and bool(self._solved_frequencies_hz))
        self._results_form.setRowVisible(self._animation_row, supports_cycle)
        self.animation_speed_slider.setVisible(animated)
        self._animation_speed_label.setVisible(animated)
        flow = display == ObservationPlaneDisplay.FLOW_ARROWS
        self.arrow_density_slider.setEnabled(flow)
        self._form.setRowVisible(self.arrow_density_slider, flow)
        scale_visibility = {
            "velocity": flow and animated,
            "intensity": flow and not animated,
            "particle_velocity": display == ObservationPlaneDisplay.PARTICLE_VELOCITY,
        }
        for prefix, visible in scale_visibility.items():
            self._form.setRowVisible(self._scale_rows[prefix], visible)
            automatic = getattr(self, f"{prefix}_scale_auto_check")
            automatic.setEnabled(visible)
            getattr(self, f"{prefix}_scale_spin").setEnabled(visible and not automatic.isChecked())
        velocity_label = self._form.labelForField(self._scale_rows["particle_velocity"])
        velocity_label.setText("Instantaneous speed maximum" if animated else "Velocity magnitude maximum")
        supports_interior = plane_type in {ObservationPlaneType.INTERIOR.value, ObservationPlaneType.COMBINED.value}
        combined = plane_type == ObservationPlaneType.COMBINED.value
        if combined and self.rendering_combo.currentData() != InteriorRenderingMode.SMOOTH_FIELD.value:
            self._select_data(self.rendering_combo, InteriorRenderingMode.SMOOTH_FIELD.value)
        rendering = supports_interior and not combined and not flow
        self.rendering_combo.setEnabled(rendering)
        self._form.setRowVisible(self.rendering_combo, rendering)
        element_field = (
            supports_interior and self.rendering_combo.currentData() == InteriorRenderingMode.ELEMENT_FIELD.value
        )
        # Element fields clip the volume itself; their clipping side remains
        # meaningful even though they do not use a sampling resolution.
        clipping = supports_interior
        self.invert_clip_check.setEnabled(clipping)
        self._form.setRowVisible(self.invert_clip_check, clipping)
        resolution = not element_field and not flow
        self.resolution_spin.setEnabled(resolution)
        self._form.setRowVisible(self.resolution_spin, resolution)
        pressure_display = display in {
            ObservationPlaneDisplay.REAL_PRESSURE,
            ObservationPlaneDisplay.IMAGINARY_PRESSURE,
        }
        self._form.setRowVisible(self._pressure_range_row, pressure_display)
        self.pressure_range_auto_check.setEnabled(pressure_display)
        self.pressure_limit_spin.setEnabled(pressure_display and not self.pressure_range_auto_check.isChecked())
        spl_display = display == ObservationPlaneDisplay.SPL
        self._form.setRowVisible(self._spl_range_row, spl_display)
        self.spl_range_auto_check.setEnabled(spl_display)
        for spin in (self.spl_min_spin, self.spl_max_spin):
            spin.setEnabled(spl_display and not self.spl_range_auto_check.isChecked())
        self._form.setRowVisible(self.normalized_range_spin, display == ObservationPlaneDisplay.NORMALIZED_SPL)
        self.layout().activate()
        self.resize(self.width(), self.sizeHint().height())

    def _refresh_frequency_label(self) -> None:
        if not self._solved_frequencies_hz:
            self.frequency_label.setText("No solved plane data available")
            return
        index = min(self.frequency_slider.value(), len(self._solved_frequencies_hz) - 1)
        self.frequency_label.setText(f"{self._solved_frequencies_hz[index]:g} Hz")

    def _selected_frequency_hz(self) -> float | None:
        if not self._solved_frequencies_hz:
            return None
        index = min(self.frequency_slider.value(), len(self._solved_frequencies_hz) - 1)
        return self._solved_frequencies_hz[index]

    def _refresh_animation_controls(self) -> None:
        if self.animate_button.isChecked() and self.frequency_sweep_button.isChecked():
            self.frequency_sweep_button.setChecked(False)
        self.animation_speed_slider.setEnabled(bool(self._solved_frequencies_hz) and self.animate_button.isChecked())
        self._refresh_dependent_controls()

    def _refresh_frequency_sweep_controls(self) -> None:
        available = len(self._solved_frequencies_hz) > 1
        checked = available and self.frequency_sweep_button.isChecked()
        self.frequency_sweep_speed_slider.setEnabled(checked)
        self.frequency_sweep_speed_slider.setVisible(checked)
        self._sweep_speed_label.setVisible(checked)
        if not checked:
            self._frequency_sweep_timer.stop()
            return
        if self.animate_button.isChecked():
            self.animate_button.setChecked(False)
        self._frequency_sweep_position = float(self.frequency_slider.value())
        maximum = len(self._solved_frequencies_hz) - 1
        self._frequency_sweep_direction = -1.0 if self._frequency_sweep_position >= maximum else 1.0
        self._frequency_sweep_updated_at = time.monotonic()
        self._frequency_sweep_timer.start()

    def _advance_frequency_sweep(self) -> None:
        if not self.frequency_sweep_button.isChecked() or len(self._solved_frequencies_hz) < 2:
            self._stop_frequency_sweep()
            return
        now = time.monotonic()
        elapsed = max(now - self._frequency_sweep_updated_at, 0.0)
        self._frequency_sweep_updated_at = now
        maximum = float(len(self._solved_frequencies_hz) - 1)
        period = 2.0 * maximum
        phase = (
            self._frequency_sweep_position
            if self._frequency_sweep_direction > 0.0
            else period - self._frequency_sweep_position
        )
        phase = (phase + self.frequency_sweep_speed_slider.value() * elapsed) % period
        if phase < maximum:
            position = phase
            self._frequency_sweep_direction = 1.0
        else:
            position = period - phase
            self._frequency_sweep_direction = -1.0
        self._frequency_sweep_position = position
        self.frequency_slider.setValue(int(round(position)))

    def _reset_frequency_sweep_clock(self, *_args) -> None:
        self._frequency_sweep_updated_at = time.monotonic()

    def _stop_frequency_sweep(self) -> None:
        self._frequency_sweep_timer.stop()
        if self.frequency_sweep_button.isChecked():
            self.frequency_sweep_button.setChecked(False)

    def _emit_preview(self, *_args) -> None:
        if self._preview_ready:
            self._frequency_preview_timer.stop()
            self.setWindowTitle(f"{self.plane.name} Properties")
            self.previewChanged.emit(self.plane, self.animate_button.isChecked())

    def _schedule_frequency_preview(self, *_args) -> None:
        if self._preview_ready and not self._frequency_preview_timer.isActive():
            self._frequency_preview_timer.start()

    def accept(self) -> None:
        self._frequency_preview_timer.stop()
        self._frequency_sweep_timer.stop()
        super().accept()

    def reject(self) -> None:
        self._frequency_preview_timer.stop()
        self._frequency_sweep_timer.stop()
        super().reject()

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
