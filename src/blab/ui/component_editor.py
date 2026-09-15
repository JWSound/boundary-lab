"""Focused physical-system component editor widgets."""

from __future__ import annotations

from dataclasses import dataclass

import meshio
import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from blab.component_symmetry import (
    ComponentSymmetryInference,
    ComponentSymmetryInferenceError,
    ProjectedAreaGeometryCache,
    ProjectedDiaphragmAreaInference,
    infer_component_symmetry,
    infer_projected_diaphragm_area,
)
from blab.config import normalize_symmetry
from blab.physical_model import (
    Boundary,
    ComponentKind,
    MeshResource,
)
from blab.system_editing import (
    MotionAxisInference,
    infer_component_motion_axis,
)
from blab.ui.numeric_locale import FlexibleDoubleValidator, format_decimal_number, parse_decimal_number

PROJECTED_AREA_DEBOUNCE_MS = 500


@dataclass
class _ComponentDraft:
    id: str
    name: str
    kind: ComponentKind
    boundary_ids: tuple[str, ...]
    channel: str
    parameters: dict
    motion_axis_mode: str = "manual"
    axis_confidence: float | None = None


_BOUNDARY_MOTION_WEIGHTS_KEY = "boundary_motion_weights"


_SEMI_INDUCTANCE_KEY = "semi_inductance"


_LUMPED_SEALED_REAR_CHAMBER_KEY = "lumped_sealed_rear_chamber"


_TRANSDUCER_PARAMETER_FIELDS = (
    ("re_ohm", "Re", "Ω", 1.0),
    ("le_h", "Le", "mH", 1_000.0),
    ("bl_n_per_a", "Bl", "N/A", 1.0),
    ("mmd_kg", "Mmd", "g", 1_000.0),
    ("cms_m_per_n", "Cms", "µm/N", 1_000_000.0),
    ("rms_n_s_per_m", "Rms", "N·s/m", 1.0),
)


_SEMI_INDUCTANCE_PARAMETER_FIELDS = (
    ("re_prime_ohm", "Re′", "Ω", 1.0),
    ("leb_h", "Leb", "mH", 1_000.0),
    ("le_h", "Le", "mH", 1_000.0),
    ("ke_semi_h", "Ke", "sH", 1.0),
    ("rss_ohm", "Rss", "Ω", 1.0),
)


class _SemiInductanceDialog(QDialog):
    """Edit the optional Thorborg-Futtrup voice-coil impedance model."""

    def __init__(self, parameters: dict | None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Semi-Inductance")
        raw = parameters if isinstance(parameters, dict) else {}

        self.enabled_check = QCheckBox("Enable semi-inductance model")
        self.enabled_check.setChecked(raw.get("enabled") is True)
        self.parameter_edits: dict[str, QLineEdit] = {}
        parameter_validator = FlexibleDoubleValidator(self)
        form = QFormLayout()
        for key, label, unit, display_per_si in _SEMI_INDUCTANCE_PARAMETER_FIELDS:
            edit = QLineEdit()
            edit.setValidator(parameter_validator)
            value = raw.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                edit.setText(format_decimal_number(float(value) * display_per_si, edit.locale()))
            edit.setPlaceholderText(unit)
            self.parameter_edits[key] = edit
            form.addRow(f"{label} ({unit})", edit)

        note = QLabel(
            "Re′ and Leb are the series terms. Le, Ke, and Rss form the parallel "
            "bound-inductance, semi-inductance, and shunt-loss network."
        )
        note.setWordWrap(True)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.enabled_check)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(buttons)

        self.enabled_check.toggled.connect(self._refresh_enabled)
        self._refresh_enabled(self.enabled_check.isChecked())
        self.resize(390, 280)

    def model_parameters(self) -> dict | None:
        enabled = self.enabled_check.isChecked()
        values: dict[str, float | bool] = {"enabled": enabled}
        populated = False
        for key, label, _unit, display_per_si in _SEMI_INDUCTANCE_PARAMETER_FIELDS:
            text = self.parameter_edits[key].text().strip()
            if not text:
                if enabled:
                    raise ValueError(f"{label} is required when semi-inductance is enabled.")
                continue
            try:
                display_value = parse_decimal_number(text, self.parameter_edits[key].locale())
            except ValueError as exc:
                raise ValueError(f"{label} must be a finite number.") from exc
            if not np.isfinite(display_value):
                raise ValueError(f"{label} must be a finite number.")
            if display_value <= 0.0:
                raise ValueError(f"{label} must be greater than zero.")
            values[key] = display_value / display_per_si
            populated = True
        return values if enabled or populated else None

    def _accept(self) -> None:
        try:
            self._parameters = self.model_parameters()
        except ValueError as exc:
            QMessageBox.warning(self, "Semi-Inductance", str(exc))
            return
        self.accept()

    def _refresh_enabled(self, enabled: bool) -> None:
        for edit in self.parameter_edits.values():
            edit.setEnabled(enabled)


class _ComponentEditorDialog(QDialog):
    """Edit one component while keeping the Components table an overview."""

    def __init__(
        self,
        draft: _ComponentDraft,
        *,
        boundaries: tuple[Boundary, ...],
        resources_by_id: dict[str, MeshResource],
        region_names: dict[str, str],
        channel_names: tuple[str, ...],
        unavailable_boundary_ids: set[str],
        symmetry_mode: str,
        mesh_cache: dict[str, meshio.Mesh],
        projected_geometry_cache: ProjectedAreaGeometryCache | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Component")
        self._draft = draft
        self._boundaries_by_id = {boundary.id: boundary for boundary in boundaries}
        self._resources_by_id = resources_by_id
        self._mesh_cache = mesh_cache
        self._projected_geometry_cache = projected_geometry_cache or ProjectedAreaGeometryCache()
        self._axis_inference: MotionAxisInference | None = None
        self._automatic_axis: np.ndarray | None = None
        self._axis_inference_error: str | None = None
        self._symmetry_mode = normalize_symmetry(symmetry_mode)
        self._symmetry_inference: ComponentSymmetryInference | None = None
        self._symmetry_inference_error: str | None = None
        self._projected_area_inference: ProjectedDiaphragmAreaInference | None = None
        self._projected_area_error: str | None = None
        raw_semi_inductance = draft.parameters.get(_SEMI_INDUCTANCE_KEY)
        self._semi_inductance_parameters = dict(raw_semi_inductance) if isinstance(raw_semi_inductance, dict) else None
        raw_rear_chamber = draft.parameters.get(_LUMPED_SEALED_REAR_CHAMBER_KEY)
        self._rear_chamber_was_configured = isinstance(raw_rear_chamber, dict)
        rear_chamber = raw_rear_chamber if isinstance(raw_rear_chamber, dict) else {}

        self.name_edit = QLineEdit(draft.name)
        self.type_combo = QComboBox()
        self.type_combo.addItem("Prescribed Velocity", ComponentKind.IDEAL_VELOCITY_SOURCE)
        self.type_combo.addItem("Electrodynamic Transducer", ComponentKind.ELECTRODYNAMIC_TRANSDUCER)
        type_index = self.type_combo.findData(draft.kind)
        self.type_combo.setCurrentIndex(max(type_index, 0))
        self.channel_combo = QComboBox()
        for channel_name in channel_names:
            self.channel_combo.addItem(channel_name, channel_name)
        channel_index = self.channel_combo.findData(draft.channel)
        self.channel_combo.setCurrentIndex(max(channel_index, 0))

        identity_form = QFormLayout()
        identity_form.addRow("Name", self.name_edit)
        identity_form.addRow("Type", self.type_combo)
        identity_form.addRow("Channel", self.channel_combo)

        self.boundary_table = QTableWidget(len(boundaries), 4)
        self.boundary_table.setHorizontalHeaderLabels(["Use", "Surface", "Region", "Relative Velocity"])
        self.boundary_table.verticalHeader().setVisible(False)
        self.boundary_table.setAlternatingRowColors(True)
        self.boundary_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.boundary_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.boundary_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.boundary_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.boundary_weight_spins: list[QDoubleSpinBox] = []
        raw_weights = draft.parameters.get(_BOUNDARY_MOTION_WEIGHTS_KEY, {})
        weights = raw_weights if isinstance(raw_weights, dict) else {}
        self.boundary_table.blockSignals(True)
        for row, boundary in enumerate(boundaries):
            region_name = region_names.get(boundary.region_id, boundary.region_id)
            unavailable = boundary.id in unavailable_boundary_ids and boundary.id not in draft.boundary_ids
            use_item = QTableWidgetItem()
            use_item.setData(Qt.ItemDataRole.UserRole, boundary.id)
            flags = use_item.flags() | Qt.ItemFlag.ItemIsUserCheckable
            if unavailable:
                flags &= ~Qt.ItemFlag.ItemIsEnabled
            use_item.setFlags(flags)
            use_item.setCheckState(
                Qt.CheckState.Checked if boundary.id in draft.boundary_ids else Qt.CheckState.Unchecked
            )
            self.boundary_table.setItem(row, 0, use_item)
            name_item = QTableWidgetItem(boundary.name + (" (assigned to another component)" if unavailable else ""))
            region_item = QTableWidgetItem(region_name)
            for display_item in (name_item, region_item):
                display_item.setFlags(display_item.flags() & ~Qt.ItemIsEditable)
                if unavailable:
                    display_item.setFlags(display_item.flags() & ~Qt.ItemIsEnabled)
            self.boundary_table.setItem(row, 1, name_item)
            self.boundary_table.setItem(row, 2, region_item)
            try:
                weight = float(weights.get(boundary.id, 1.0))
            except (TypeError, ValueError):
                weight = 1.0
            weight_db = 20.0 * np.log10(max(weight, 1.0e-6))
            spin = QDoubleSpinBox()
            spin.setRange(-120.0, 20.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.5)
            spin.setSuffix(" dB")
            spin.setValue(float(weight_db))
            spin.setEnabled(not unavailable and boundary.id in draft.boundary_ids)
            self.boundary_table.setCellWidget(row, 3, spin)
            self.boundary_weight_spins.append(spin)
        self.boundary_table.blockSignals(False)

        surfaces_group = QGroupBox("Moving surfaces")
        surfaces_layout = QVBoxLayout(surfaces_group)
        surfaces_layout.addWidget(
            QLabel("Select every acoustic surface driven by this component, including front and rear sides.")
        )
        surfaces_layout.addWidget(self.boundary_table)

        self.axis_mode_combo = QComboBox()
        self.axis_mode_combo.addItem("Automatic from surface normals", "automatic")
        self.axis_mode_combo.addItem("Manual", "manual")
        mode_index = self.axis_mode_combo.findData(draft.motion_axis_mode)
        self.axis_mode_combo.setCurrentIndex(max(mode_index, 0))
        raw_axis = draft.parameters.get("motion_axis", (0.0, 0.0, 1.0))
        if not isinstance(raw_axis, (list, tuple)) or len(raw_axis) != 3:
            raw_axis = (0.0, 0.0, 1.0)
        self.axis_spins = []
        axis_row = QHBoxLayout()
        for label, value in zip(("X", "Y", "Z"), raw_axis):
            axis_row.addWidget(QLabel(label))
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(-1.0, 1.0)
            spin.setSingleStep(0.005)
            spin.setValue(float(value))
            self.axis_spins.append(spin)
            axis_row.addWidget(spin)
        self.flip_axis_button = QPushButton("Flip")
        axis_row.addWidget(self.flip_axis_button)
        self.axis_confidence_label = QLabel()
        self.axis_confidence_label.setWordWrap(True)

        axis_form = QFormLayout()
        axis_form.addRow("Motion axis", self.axis_mode_combo)
        axis_form.addRow("Direction", axis_row)
        axis_form.addRow("Inference", self.axis_confidence_label)

        self.parameter_edits: dict[str, QLineEdit] = {}
        parameter_validator = FlexibleDoubleValidator(self)
        self.semi_inductance_button = QPushButton()
        self.rear_chamber_check = QCheckBox("Lumped sealed rear chamber")
        self.rear_chamber_check.setChecked(rear_chamber.get("enabled") is True)
        self.rear_chamber_volume_spin = QDoubleSpinBox()
        self.rear_chamber_volume_spin.setRange(0.001, 1_000_000.0)
        self.rear_chamber_volume_spin.setDecimals(3)
        self.rear_chamber_volume_spin.setSingleStep(0.1)
        self.rear_chamber_volume_spin.setSuffix(" L")
        try:
            rear_volume_l = 1000.0 * float(rear_chamber.get("volume_m3", 0.001))
        except (TypeError, ValueError):
            rear_volume_l = 1.0
        self.rear_chamber_volume_spin.setValue(max(0.001, rear_volume_l))
        self.rear_chamber_volume_spin.setEnabled(self.rear_chamber_check.isChecked())
        self.rear_chamber_check.setToolTip("Add an ideal lumped compliance for an unmeshed sealed rear chamber.")
        self.rear_chamber_volume_spin.setToolTip("Net enclosed air volume in litres.")
        transducer_form = QFormLayout()
        for key, label, unit, display_per_si in _TRANSDUCER_PARAMETER_FIELDS:
            edit = QLineEdit()
            edit.setValidator(parameter_validator)
            if key in draft.parameters:
                edit.setText(format_decimal_number(float(draft.parameters[key]) * display_per_si, edit.locale()))
            edit.setPlaceholderText(unit)
            self.parameter_edits[key] = edit
            if key == "le_h":
                le_row = QHBoxLayout()
                le_row.addWidget(edit)
                le_row.addWidget(self.semi_inductance_button)
                transducer_form.addRow(f"{label} ({unit})", le_row)
            else:
                transducer_form.addRow(f"{label} ({unit})", edit)
        rear_chamber_row = QHBoxLayout()
        rear_chamber_row.addWidget(self.rear_chamber_check)
        rear_chamber_row.addWidget(self.rear_chamber_volume_spin)
        transducer_form.addRow("", rear_chamber_row)
        self.symmetry_inference_label = QLabel()
        self.symmetry_inference_label.setWordWrap(True)
        transducer_form.addRow("Symmetry", self.symmetry_inference_label)
        self.projected_area_warning_label = QLabel()
        self.projected_area_warning_label.setWordWrap(True)
        self.projected_area_warning_label.setStyleSheet("color: #d97706; font-weight: 600;")
        self.projected_area_warning_label.setVisible(False)
        transducer_form.addRow("", self.projected_area_warning_label)

        self.transducer_group = QGroupBox("Rigid-piston transducer")
        transducer_layout = QVBoxLayout(self.transducer_group)
        transducer_layout.addLayout(transducer_form)
        transducer_layout.addLayout(axis_form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(identity_form)
        layout.addWidget(surfaces_group)
        layout.addWidget(self.transducer_group)
        layout.addWidget(buttons)
        self.resize(680, 700)

        self.type_combo.currentIndexChanged.connect(self._refresh_type_controls)
        self.axis_mode_combo.currentIndexChanged.connect(self._refresh_axis_controls)
        self.boundary_table.itemChanged.connect(self._selected_boundaries_changed)
        self.flip_axis_button.clicked.connect(self._flip_axis)
        self.semi_inductance_button.clicked.connect(self._edit_semi_inductance)
        self.rear_chamber_check.toggled.connect(self.rear_chamber_volume_spin.setEnabled)
        self._projected_area_update_timer = QTimer(self)
        self._projected_area_update_timer.setSingleShot(True)
        self._projected_area_update_timer.setInterval(PROJECTED_AREA_DEBOUNCE_MS)
        self._projected_area_update_timer.timeout.connect(self._update_projected_area_readout)
        for spin in (*self.axis_spins, *self.boundary_weight_spins):
            spin.valueChanged.connect(self._schedule_projected_area_update)
        self._refresh_semi_inductance_controls()
        self._refresh_type_controls(update_geometry=False)
        self._refresh_axis_controls(update_geometry=False)
        if draft.kind == ComponentKind.ELECTRODYNAMIC_TRANSDUCER:
            symmetry_inference = self._infer_component_symmetry(update_projected_area=False)
            if symmetry_inference is not None:
                if draft.motion_axis_mode == "automatic":
                    self._infer_axis(symmetry_inference)
                else:
                    self._update_projected_area_readout(symmetry_inference)

    def selected_boundary_ids(self) -> tuple[str, ...]:
        return tuple(
            str(item.data(Qt.ItemDataRole.UserRole))
            for row in range(self.boundary_table.rowCount())
            if (item := self.boundary_table.item(row, 0)).checkState() == Qt.CheckState.Checked
        )

    def boundary_motion_weights(self) -> dict[str, float]:
        selected = set(self.selected_boundary_ids())
        return {
            str(self.boundary_table.item(row, 0).data(Qt.ItemDataRole.UserRole)): float(
                10.0 ** (self.boundary_weight_spins[row].value() / 20.0)
            )
            for row in range(self.boundary_table.rowCount())
            if str(self.boundary_table.item(row, 0).data(Qt.ItemDataRole.UserRole)) in selected
        }

    def component_draft(self) -> _ComponentDraft:
        self._projected_area_update_timer.stop()
        name = self.name_edit.text().strip()
        if not name:
            raise ValueError("The component must have a name.")
        boundary_ids = self.selected_boundary_ids()
        if not boundary_ids:
            raise ValueError(f"Component '{name}' must select at least one moving boundary.")
        kind = ComponentKind(self.type_combo.currentData())
        if kind == ComponentKind.ELECTRODYNAMIC_TRANSDUCER:
            symmetry_inference = self._infer_component_symmetry(update_projected_area=False)
            if symmetry_inference is None:
                raise ValueError(self._symmetry_inference_error or "Component symmetry could not be inferred.")
            parameters = {}
            for key, label, _unit, display_per_si in _TRANSDUCER_PARAMETER_FIELDS:
                text = self.parameter_edits[key].text().strip()
                try:
                    display_value = parse_decimal_number(text, self.parameter_edits[key].locale())
                except ValueError as exc:
                    raise ValueError(f"{label} must be a finite number.") from exc
                if not np.isfinite(display_value):
                    raise ValueError(f"{label} must be a finite number.")
                if key in {"le_h", "rms_n_s_per_m"}:
                    if display_value < 0.0:
                        raise ValueError(f"{label} must not be negative.")
                elif display_value <= 0.0:
                    raise ValueError(f"{label} must be greater than zero.")
                parameters[key] = display_value / display_per_si
            if self._semi_inductance_parameters is not None:
                parameters[_SEMI_INDUCTANCE_KEY] = dict(self._semi_inductance_parameters)
            mode = str(self.axis_mode_combo.currentData())
            if mode == "automatic":
                inference = self._infer_axis(symmetry_inference, update_projected_area=False)
                if inference is None:
                    raise ValueError(self._axis_inference_error or "The motion axis could not be inferred.")
                if inference.confidence < 0.2:
                    raise ValueError(
                        "Automatic motion-axis confidence is low. Check the selected surfaces or use a manual axis."
                    )
            axis = (
                np.asarray(self._automatic_axis, dtype=float)
                if mode == "automatic" and self._automatic_axis is not None
                else np.asarray([spin.value() for spin in self.axis_spins], dtype=float)
            )
            norm = float(np.linalg.norm(axis))
            if norm <= 0.0:
                raise ValueError("The motion axis must have nonzero length.")
            parameters["motion_axis"] = [float(value) for value in axis / norm]
            parameters["motion_profile"] = "rigid_translation"
            parameters.update(symmetry_inference.parameters())
            raw_signs = self._draft.parameters.get("boundary_motion_signs", {})
            if isinstance(raw_signs, dict):
                signs = {
                    str(boundary_id): float(sign)
                    for boundary_id, sign in raw_signs.items()
                    if str(boundary_id) in boundary_ids
                }
                if signs:
                    parameters["boundary_motion_signs"] = signs
            projected_area = self._update_projected_area_readout(
                symmetry_inference,
                axis=axis / norm,
            )
            rear_chamber_enabled = self.rear_chamber_check.isChecked()
            if projected_area is None and rear_chamber_enabled:
                raise ValueError(self._projected_area_error or "Projected diaphragm area could not be calculated.")
            rear_chamber_parameters: dict[str, float | bool] = {
                "enabled": rear_chamber_enabled,
                "volume_m3": float(self.rear_chamber_volume_spin.value()) / 1000.0,
            }
            if projected_area is not None:
                rear_chamber_parameters["projected_area_m2"] = projected_area.projected_area_m2
            if rear_chamber_enabled or self._rear_chamber_was_configured:
                parameters[_LUMPED_SEALED_REAR_CHAMBER_KEY] = rear_chamber_parameters
            confidence = None if self._axis_inference is None else self._axis_inference.confidence
        else:
            parameters = {"motion_profile": "uniform"}
            mode = "manual"
            confidence = None
        parameters[_BOUNDARY_MOTION_WEIGHTS_KEY] = self.boundary_motion_weights()
        return _ComponentDraft(
            id=self._draft.id,
            name=name,
            kind=kind,
            boundary_ids=boundary_ids,
            channel=str(self.channel_combo.currentData()),
            parameters=parameters,
            motion_axis_mode=mode,
            axis_confidence=confidence,
        )

    def _accept(self) -> None:
        try:
            self._draft = self.component_draft()
        except ValueError as exc:
            QMessageBox.warning(self, "Component", str(exc))
            return
        self.accept()

    def _refresh_type_controls(self, _index: int = -1, *, update_geometry: bool = True) -> None:
        electrodynamic = self.type_combo.currentData() == ComponentKind.ELECTRODYNAMIC_TRANSDUCER
        self.transducer_group.setVisible(electrodynamic)
        if not electrodynamic:
            self._projected_area_update_timer.stop()
        if electrodynamic and update_geometry:
            symmetry_inference = self._infer_component_symmetry(update_projected_area=False)
            if symmetry_inference is None:
                return
            if self.axis_mode_combo.currentData() == "automatic":
                self._infer_axis(symmetry_inference)
            else:
                self._update_projected_area_readout(symmetry_inference)

    def _edit_semi_inductance(self) -> None:
        dialog = _SemiInductanceDialog(self._semi_inductance_parameters, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._semi_inductance_parameters = dialog._parameters
        if (
            isinstance(self._semi_inductance_parameters, dict)
            and self._semi_inductance_parameters.get("enabled") is True
        ):
            if not self.parameter_edits["re_ohm"].text().strip():
                edit = self.parameter_edits["re_ohm"]
                edit.setText(
                    format_decimal_number(float(self._semi_inductance_parameters["re_prime_ohm"]), edit.locale())
                )
            if not self.parameter_edits["le_h"].text().strip():
                edit = self.parameter_edits["le_h"]
                edit.setText(
                    format_decimal_number(float(self._semi_inductance_parameters["le_h"]) * 1_000.0, edit.locale())
                )
        self._refresh_semi_inductance_controls()

    def _refresh_semi_inductance_controls(self) -> None:
        enabled = (
            isinstance(self._semi_inductance_parameters, dict)
            and self._semi_inductance_parameters.get("enabled") is True
        )
        self.semi_inductance_button.setText("Semi-Inductance: On…" if enabled else "Semi-Inductance…")
        self.parameter_edits["le_h"].setEnabled(not enabled)
        self.parameter_edits["le_h"].setToolTip(
            "The simple-model Le is retained as a fallback while semi-inductance is enabled."
            if enabled
            else "Simple voice-coil inductance."
        )

    def _refresh_axis_controls(self, _index: int = -1, *, update_geometry: bool = True) -> None:
        automatic = self.axis_mode_combo.currentData() == "automatic"
        for spin in self.axis_spins:
            spin.setEnabled(not automatic)
        if automatic:
            if update_geometry:
                self._infer_axis()
        elif not self.axis_confidence_label.text():
            self.axis_confidence_label.setText("Manual direction; it will be normalized when saved.")

    def _selected_boundaries_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == 0:
            self.boundary_weight_spins[item.row()].setEnabled(
                bool(item.flags() & Qt.ItemFlag.ItemIsEnabled) and item.checkState() == Qt.CheckState.Checked
            )
        symmetry_inference = self._infer_component_symmetry(update_projected_area=False)
        if symmetry_inference is None:
            return
        if self.axis_mode_combo.currentData() == "automatic":
            self._infer_axis(symmetry_inference)
        else:
            self._update_projected_area_readout(symmetry_inference)

    def _infer_component_symmetry(
        self,
        *,
        update_projected_area: bool = True,
    ) -> ComponentSymmetryInference | None:
        selected = tuple(
            self._boundaries_by_id[boundary_id]
            for boundary_id in self.selected_boundary_ids()
            if boundary_id in self._boundaries_by_id
        )
        try:
            inference = infer_component_symmetry(
                selected,
                self._resources_by_id,
                self._symmetry_mode,
                mesh_cache=self._mesh_cache,
            )
        except ComponentSymmetryInferenceError as exc:
            self._symmetry_inference = None
            self._symmetry_inference_error = str(exc)
            self.symmetry_inference_label.setText(str(exc))
            self._projected_area_inference = None
            self._projected_area_error = str(exc)
            self.projected_area_warning_label.setVisible(False)
            return None
        self._symmetry_inference = inference
        self._symmetry_inference_error = None
        if update_projected_area:
            self._update_projected_area_readout(inference)
        return inference

    def _schedule_projected_area_update(self, _value: float = 0.0) -> None:
        if self.type_combo.currentData() == ComponentKind.ELECTRODYNAMIC_TRANSDUCER:
            self._projected_area_update_timer.start()

    def _update_projected_area_readout(
        self,
        symmetry_inference: ComponentSymmetryInference | None = None,
        *,
        axis: np.ndarray | None = None,
    ) -> ProjectedDiaphragmAreaInference | None:
        if hasattr(self, "_projected_area_update_timer"):
            self._projected_area_update_timer.stop()
        inference = self._symmetry_inference if symmetry_inference is None else symmetry_inference
        if inference is None or not hasattr(self, "symmetry_inference_label"):
            return None
        selected = tuple(
            self._boundaries_by_id[boundary_id]
            for boundary_id in self.selected_boundary_ids()
            if boundary_id in self._boundaries_by_id
        )
        if axis is None:
            axis = (
                np.asarray(self._automatic_axis, dtype=float)
                if self.axis_mode_combo.currentData() == "automatic" and self._automatic_axis is not None
                else np.asarray([spin.value() for spin in self.axis_spins], dtype=float)
            )
        try:
            area = infer_projected_diaphragm_area(
                selected,
                self._resources_by_id,
                axis,
                inference.surface_completion_factor,
                boundary_motion_weights=self.boundary_motion_weights(),
                boundary_side_keys={boundary.id: boundary.region_id for boundary in selected},
                mesh_cache=self._mesh_cache,
                projected_geometry_cache=self._projected_geometry_cache,
            )
        except ComponentSymmetryInferenceError as exc:
            self._projected_area_inference = None
            self._projected_area_error = str(exc)
            self.symmetry_inference_label.setText(f"{inference.summary()} Projected diaphragm area unavailable: {exc}")
            self.projected_area_warning_label.setVisible(False)
            return None
        self._projected_area_inference = area
        self._projected_area_error = None
        self.symmetry_inference_label.setText(
            f"{inference.summary()} Projected diaphragm area of {area.projected_area_m2 * 10_000.0:.2f} cm²."
        )
        mismatch = area.relative_side_mismatch
        if mismatch is not None and mismatch > 0.10:
            self.projected_area_warning_label.setText(
                "Front/rear projected diaphragm areas deviate by "
                f"{mismatch:.1%} ({area.positive_side_area_m2 * 10_000.0:.2f} cm² versus "
                f"{area.negative_side_area_m2 * 10_000.0:.2f} cm²)."
            )
            self.projected_area_warning_label.setVisible(True)
        else:
            self.projected_area_warning_label.clear()
            self.projected_area_warning_label.setVisible(False)
        return area

    def _infer_axis(
        self,
        symmetry_inference: ComponentSymmetryInference | None = None,
        *,
        update_projected_area: bool = True,
    ) -> MotionAxisInference | None:
        selected = tuple(
            self._boundaries_by_id[boundary_id]
            for boundary_id in self.selected_boundary_ids()
            if boundary_id in self._boundaries_by_id
        )
        if symmetry_inference is None:
            symmetry_inference = self._infer_component_symmetry(update_projected_area=False)
        if symmetry_inference is None:
            self._axis_inference = None
            self._automatic_axis = None
            self._axis_inference_error = self._symmetry_inference_error
            self.axis_confidence_label.setText(
                self._symmetry_inference_error or "Component symmetry could not be inferred."
            )
            return None
        try:
            inference = infer_component_motion_axis(
                selected,
                self._resources_by_id,
                fractional_symmetry_axes=symmetry_inference.fractional_symmetry_axes,
                mesh_cache=self._mesh_cache,
            )
        except (ValueError, OSError) as exc:
            self._axis_inference = None
            self._automatic_axis = None
            self._axis_inference_error = str(exc)
            self.axis_confidence_label.setText(str(exc))
            return None
        inferred_axis = np.asarray(inference.axis, dtype=float)
        current_axis = np.asarray([spin.value() for spin in self.axis_spins], dtype=float)
        if float(np.linalg.norm(current_axis)) > 0.0 and float(np.dot(inferred_axis, current_axis)) < 0.0:
            inferred_axis *= -1.0
        signal_blockers = [QSignalBlocker(spin) for spin in self.axis_spins]
        try:
            for spin, value in zip(self.axis_spins, inferred_axis):
                spin.setValue(float(value))
        finally:
            del signal_blockers
        self._axis_inference = inference
        self._automatic_axis = inferred_axis.copy()
        self._axis_inference_error = None
        quality = "High" if inference.confidence >= 0.8 else "Moderate" if inference.confidence >= 0.2 else "Low"
        self.axis_confidence_label.setText(
            f"{quality} confidence ({inference.confidence:.0%}); "
            f"{inference.triangle_count} triangles, projected-normal alignment "
            f"{inference.mean_squared_alignment:.0%}."
        )
        if update_projected_area:
            self._update_projected_area_readout(symmetry_inference, axis=inferred_axis)
        return inference

    def _flip_axis(self) -> None:
        signal_blockers = [QSignalBlocker(spin) for spin in self.axis_spins]
        try:
            for spin in self.axis_spins:
                spin.setValue(-spin.value())
        finally:
            del signal_blockers
        if self._automatic_axis is not None:
            self._automatic_axis *= -1.0
        self._schedule_projected_area_update()
