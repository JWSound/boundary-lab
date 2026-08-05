"""Minimal tabbed editor for the coupled loudspeaker physical system."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import meshio
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
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
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from blab.config import normalize_symmetry
from blab.interface_conform import (
    InterfaceConformError,
    build_conforming_interface_map,
    conform_bem_interface_to_fem,
)
from blab.physical_model import (
    AcousticInterface,
    AcousticRegion,
    AcousticRegionKind,
    Boundary,
    BoundaryKind,
    ComponentKind,
    ExcitationPort,
    ExcitationPortKind,
    MeshPurpose,
    MeshResource,
    PhysicalComponent,
    PhysicalGroupRef,
    PhysicalSystem,
)
from blab.ui.dialogs import MeshDialogEntry


@dataclass(frozen=True)
class AvailableSystemMesh:
    """An enabled application mesh available to the physical-system editor."""

    name: str
    source_file: str
    file: str
    scale_to_m: float
    translation_m: tuple[float, float, float]
    surface_groups: tuple[str, ...]
    volume_groups: tuple[str, ...]
    has_tetrahedra: bool
    locked: bool = False


@dataclass(frozen=True)
class SystemConfigResult:
    system: PhysicalSystem
    component_channel_by_id: dict[str, str]
    mesh_file_overrides_by_name: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _InterfacePairMatch:
    boundary: Boundary
    conformed_bem_mesh: meshio.Mesh | None = None


@dataclass(frozen=True)
class MotionAxisInference:
    """Dominant unoriented rigid-translation axis inferred from surface normals."""

    axis: tuple[float, float, float]
    confidence: float
    mean_squared_alignment: float
    boundary_alignment: float
    area_m2: float
    triangle_count: int


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


_COMPONENT_UI_METADATA_KEY = "component_editor"
_TRANSDUCER_PARAMETER_FIELDS = (
    ("re_ohm", "Re", "Ω", 1.0),
    ("le_h", "Le", "mH", 1_000.0),
    ("bl_n_per_a", "Bl", "N/A", 1.0),
    ("mmd_kg", "Mmd", "g", 1_000.0),
    ("cms_m_per_n", "Cms", "µm/N", 1_000_000.0),
    ("rms_n_s_per_m", "Rms", "N·s/m", 1.0),
)


def infer_component_motion_axis(
    boundaries: tuple[Boundary, ...],
    resources_by_id: dict[str, MeshResource],
    *,
    fractional_symmetry_axes: tuple[str, ...] = (),
    mesh_cache: dict[str, meshio.Mesh] | None = None,
) -> MotionAxisInference:
    """Infer a common translation axis from the completed physical-driver surface."""

    if not boundaries:
        raise ValueError("Select at least one moving boundary before inferring its motion axis.")
    symmetry_axes = tuple(str(axis).strip().lower() for axis in fractional_symmetry_axes)
    if len(symmetry_axes) != len(set(symmetry_axes)) or any(
        axis not in {"x", "y"} for axis in symmetry_axes
    ):
        raise ValueError("Fractional symmetry axes must be unique axis names chosen from X and Y.")
    cache = {} if mesh_cache is None else mesh_cache
    tensors = []
    combined = np.zeros((3, 3), dtype=float)
    total_area = 0.0
    triangle_count = 0
    for boundary in boundaries:
        resource = resources_by_id.get(boundary.group.mesh_id)
        if resource is None:
            raise ValueError(
                f"Moving boundary '{boundary.name}' references unavailable mesh '{boundary.group.mesh_id}'."
            )
        cache_key = resource.id
        mesh = cache.get(cache_key)
        if mesh is None:
            mesh = _transformed_mesh(resource)
            cache[cache_key] = mesh
        tensor, area, count = _boundary_normal_tensor(mesh, boundary)
        tensors.append(tensor)
        combined += tensor
        total_area += area
        triangle_count += count
    if total_area <= 0.0 or triangle_count == 0:
        raise ValueError("The selected moving boundaries contain no non-degenerate triangles.")

    combined = _symmetry_completed_normal_tensor(combined, symmetry_axes)
    tensors = [_symmetry_completed_normal_tensor(tensor, symmetry_axes) for tensor in tensors]
    axis_tensor = _normal_tensor_in_symmetry_planes(combined, symmetry_axes)
    eigenvalues, eigenvectors = np.linalg.eigh(axis_tensor)
    axis = np.asarray(eigenvectors[:, -1], dtype=float)
    largest_component = int(np.argmax(np.abs(axis)))
    if axis[largest_component] < 0.0:
        axis *= -1.0
    trace = float(np.trace(combined))
    confidence = float(max(0.0, min(1.0, (eigenvalues[-1] - eigenvalues[-2]) / trace)))
    mean_squared_alignment = float(max(0.0, min(1.0, eigenvalues[-1] / trace)))

    boundary_axes = []
    for tensor in tensors:
        _values, vectors = np.linalg.eigh(_normal_tensor_in_symmetry_planes(tensor, symmetry_axes))
        boundary_axes.append(np.asarray(vectors[:, -1], dtype=float))
    boundary_alignment = min(
        (abs(float(np.dot(axis, boundary_axis))) for boundary_axis in boundary_axes),
        default=1.0,
    )
    confidence = min(confidence, boundary_alignment)
    return MotionAxisInference(
        axis=tuple(float(value) for value in axis),
        confidence=confidence,
        mean_squared_alignment=mean_squared_alignment,
        boundary_alignment=boundary_alignment,
        area_m2=total_area,
        triangle_count=triangle_count,
    )


def _symmetry_completed_normal_tensor(
    tensor: np.ndarray,
    symmetry_axes: tuple[str, ...],
) -> np.ndarray:
    completed = np.asarray(tensor, dtype=float).copy()
    axis_indices = {"x": 0, "y": 1}
    for axis in symmetry_axes:
        reflection = np.eye(3, dtype=float)
        reflection[axis_indices[axis], axis_indices[axis]] = -1.0
        completed += reflection @ completed @ reflection
    return completed


def _normal_tensor_in_symmetry_planes(
    tensor: np.ndarray,
    symmetry_axes: tuple[str, ...],
) -> np.ndarray:
    projected = np.asarray(tensor, dtype=float).copy()
    axis_indices = {"x": 0, "y": 1}
    for axis in symmetry_axes:
        axis_index = axis_indices[axis]
        projected[axis_index, :] = 0.0
        projected[:, axis_index] = 0.0
    return projected


def _boundary_normal_tensor(mesh: meshio.Mesh, boundary: Boundary) -> tuple[np.ndarray, float, int]:
    if boundary.group.name is not None:
        field = mesh.field_data.get(boundary.group.name)
        if field is None:
            raise ValueError(
                f"Mesh '{boundary.group.mesh_id}' does not contain surface group '{boundary.group.name}'."
            )
        tag, dimension = map(int, np.asarray(field).tolist())
        if dimension != 2:
            raise ValueError(f"Physical group '{boundary.group.name}' is not a surface group.")
    elif boundary.group.tag is not None:
        tag = int(boundary.group.tag)
    else:
        raise ValueError(f"Moving boundary '{boundary.name}' must identify a surface group by name or tag.")

    physical_blocks = mesh.cell_data.get("gmsh:physical")
    if physical_blocks is None:
        raise ValueError(f"Mesh '{boundary.group.mesh_id}' has no physical surface tags.")
    points = np.asarray(mesh.points, dtype=float)
    tensor = np.zeros((3, 3), dtype=float)
    total_area = 0.0
    count = 0
    for index, block in enumerate(mesh.cells):
        if block.type not in {"triangle", "triangle3"}:
            continue
        triangles = np.asarray(block.data, dtype=np.int64)
        physical = np.asarray(physical_blocks[index], dtype=np.int64)
        if len(physical) != len(triangles):
            raise ValueError(f"Mesh '{boundary.group.mesh_id}' has inconsistent triangle physical tags.")
        for triangle in triangles[physical == tag]:
            vertices = points[np.asarray(triangle[:3], dtype=np.int64)]
            area_vector = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
            magnitude = float(np.linalg.norm(area_vector))
            if magnitude <= 0.0:
                continue
            normal = area_vector / magnitude
            area = 0.5 * magnitude
            tensor += area * np.outer(normal, normal)
            total_area += area
            count += 1
    if count == 0:
        raise ValueError(f"Moving boundary '{boundary.name}' contains no non-degenerate triangles.")
    return tensor, total_area, count


def inspect_system_meshes(meshes: tuple[MeshDialogEntry, ...]) -> tuple[AvailableSystemMesh, ...]:
    """Read physical-group inventory without modifying imported mesh files."""

    inspected = []
    for entry in meshes:
        if not entry.enabled:
            continue
        source_path = Path(entry.source_file)
        effective_path = (
            Path(entry.cleaned_file)
            if entry.cleaned_file is not None and Path(entry.cleaned_file).is_file()
            else source_path
        )
        mesh = meshio.read(effective_path)
        surface_groups = []
        volume_groups = []
        for name, raw in mesh.field_data.items():
            _tag, dimension = map(int, np.asarray(raw).tolist())
            if dimension == 2:
                surface_groups.append(str(name))
            elif dimension == 3:
                volume_groups.append(str(name))
        inspected.append(
            AvailableSystemMesh(
                name=entry.name,
                source_file=str(source_path),
                file=str(effective_path),
                scale_to_m=float(entry.scale_factor),
                translation_m=tuple(float(value) / 1000.0 for value in entry.translation_mm),
                surface_groups=tuple(sorted(surface_groups)),
                volume_groups=tuple(sorted(volume_groups)),
                has_tetrahedra=any(block.type in {"tetra", "tetra4"} and len(block.data) for block in mesh.cells),
                locked=bool(entry.locked),
            )
        )
    return tuple(inspected)


def sync_physical_system_meshes(
    system: PhysicalSystem,
    meshes: tuple[AvailableSystemMesh, ...],
) -> PhysicalSystem:
    """Apply current application file/scale/translation settings by mesh name."""

    available_by_name = {mesh.name: mesh for mesh in meshes}
    resources = []
    for resource in system.meshes:
        available = available_by_name.get(resource.name)
        if available is None:
            resources.append(resource)
            continue
        resources.append(
            replace(
                resource,
                file=available.file,
                scale_to_m=available.scale_to_m,
                translation_m=available.translation_m,
            )
        )
    return replace(system, meshes=tuple(resources))


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
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Component")
        self._draft = draft
        self._boundaries_by_id = {boundary.id: boundary for boundary in boundaries}
        self._resources_by_id = resources_by_id
        self._mesh_cache = mesh_cache
        self._axis_inference: MotionAxisInference | None = None
        self._axis_inference_error: str | None = None

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

        self.boundary_list = QListWidget()
        self.boundary_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.boundary_list.blockSignals(True)
        for boundary in boundaries:
            region_name = region_names.get(boundary.region_id, boundary.region_id)
            label = f"{boundary.name} — {region_name}"
            unavailable = boundary.id in unavailable_boundary_ids and boundary.id not in draft.boundary_ids
            if unavailable:
                label += " (assigned to another component)"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, boundary.id)
            flags = item.flags() | Qt.ItemFlag.ItemIsUserCheckable
            if unavailable:
                flags &= ~Qt.ItemFlag.ItemIsEnabled
            item.setFlags(flags)
            item.setCheckState(
                Qt.CheckState.Checked if boundary.id in draft.boundary_ids else Qt.CheckState.Unchecked
            )
            self.boundary_list.addItem(item)
        self.boundary_list.blockSignals(False)

        surfaces_group = QGroupBox("Moving surfaces")
        surfaces_layout = QVBoxLayout(surfaces_group)
        surfaces_layout.addWidget(
            QLabel("Select every acoustic surface driven by this component, including front and rear sides.")
        )
        surfaces_layout.addWidget(self.boundary_list)

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
            spin.setDecimals(6)
            spin.setRange(-1.0, 1.0)
            spin.setSingleStep(0.05)
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
        transducer_form = QFormLayout()
        for key, label, unit, display_per_si in _TRANSDUCER_PARAMETER_FIELDS:
            edit = QLineEdit()
            if key in draft.parameters:
                edit.setText(f"{float(draft.parameters[key]) * display_per_si:.12g}")
            edit.setPlaceholderText(unit)
            self.parameter_edits[key] = edit
            transducer_form.addRow(f"{label} ({unit})", edit)
        self.symmetry_combo = QComboBox()
        for label, value in _component_symmetry_options(symmetry_mode):
            self.symmetry_combo.addItem(label, value)
        symmetry_value = _component_symmetry_value(draft.parameters)
        symmetry_index = next(
            (
                index
                for index in range(self.symmetry_combo.count())
                if self.symmetry_combo.itemData(index) == symmetry_value
            ),
            0,
        )
        self.symmetry_combo.setCurrentIndex(max(symmetry_index, 0))
        transducer_form.addRow("Symmetry representation", self.symmetry_combo)
        transducer_form.addRow("", QLabel("The voltage excitation uses the solver's 2.83 V reference signal."))

        self.transducer_group = QGroupBox("Rigid-piston transducer")
        transducer_layout = QVBoxLayout(self.transducer_group)
        transducer_layout.addLayout(transducer_form)
        transducer_layout.addLayout(axis_form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
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
        self.boundary_list.itemChanged.connect(self._selected_boundaries_changed)
        self.symmetry_combo.currentIndexChanged.connect(self._symmetry_representation_changed)
        self.flip_axis_button.clicked.connect(self._flip_axis)
        self._refresh_type_controls()
        self._refresh_axis_controls()
        if draft.kind == ComponentKind.ELECTRODYNAMIC_TRANSDUCER and draft.motion_axis_mode == "automatic":
            self._infer_axis()

    def selected_boundary_ids(self) -> tuple[str, ...]:
        return tuple(
            str(item.data(Qt.ItemDataRole.UserRole))
            for index in range(self.boundary_list.count())
            if (item := self.boundary_list.item(index)).checkState() == Qt.CheckState.Checked
        )

    def component_draft(self) -> _ComponentDraft:
        name = self.name_edit.text().strip()
        if not name:
            raise ValueError("The component must have a name.")
        boundary_ids = self.selected_boundary_ids()
        if not boundary_ids:
            raise ValueError(f"Component '{name}' must select at least one moving boundary.")
        kind = self.type_combo.currentData()
        if kind == ComponentKind.ELECTRODYNAMIC_TRANSDUCER:
            parameters = {}
            for key, label, _unit, display_per_si in _TRANSDUCER_PARAMETER_FIELDS:
                text = self.parameter_edits[key].text().strip()
                try:
                    display_value = float(text)
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
            mode = str(self.axis_mode_combo.currentData())
            if mode == "automatic":
                inference = self._infer_axis()
                if inference is None:
                    raise ValueError(self._axis_inference_error or "The motion axis could not be inferred.")
                if inference.confidence < 0.2:
                    raise ValueError(
                        "Automatic motion-axis confidence is low. Check the selected surfaces or use a manual axis."
                    )
            axis = np.asarray([spin.value() for spin in self.axis_spins], dtype=float)
            norm = float(np.linalg.norm(axis))
            if norm <= 0.0:
                raise ValueError("The motion axis must have nonzero length.")
            parameters["motion_axis"] = [float(value) for value in axis / norm]
            parameters["motion_profile"] = "rigid_translation"
            parameters.update(dict(self.symmetry_combo.currentData()))
            raw_signs = self._draft.parameters.get("boundary_motion_signs", {})
            if isinstance(raw_signs, dict):
                signs = {
                    str(boundary_id): float(sign)
                    for boundary_id, sign in raw_signs.items()
                    if str(boundary_id) in boundary_ids
                }
                if signs:
                    parameters["boundary_motion_signs"] = signs
            confidence = None if self._axis_inference is None else self._axis_inference.confidence
        else:
            parameters = {"motion_profile": "uniform"}
            mode = "manual"
            confidence = None
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

    def _refresh_type_controls(self, _index: int = -1) -> None:
        self.transducer_group.setVisible(
            self.type_combo.currentData() == ComponentKind.ELECTRODYNAMIC_TRANSDUCER
        )

    def _refresh_axis_controls(self, _index: int = -1) -> None:
        automatic = self.axis_mode_combo.currentData() == "automatic"
        for spin in self.axis_spins:
            spin.setEnabled(not automatic)
        if automatic:
            self._infer_axis()
        elif not self.axis_confidence_label.text():
            self.axis_confidence_label.setText("Manual direction; it will be normalized when saved.")

    def _selected_boundaries_changed(self, _item: QListWidgetItem) -> None:
        if self.axis_mode_combo.currentData() == "automatic":
            self._infer_axis()

    def _symmetry_representation_changed(self, _index: int = -1) -> None:
        if self.axis_mode_combo.currentData() == "automatic":
            self._infer_axis()

    def _infer_axis(self) -> MotionAxisInference | None:
        selected = tuple(
            self._boundaries_by_id[boundary_id]
            for boundary_id in self.selected_boundary_ids()
            if boundary_id in self._boundaries_by_id
        )
        symmetry_parameters = self.symmetry_combo.currentData()
        fractional_symmetry_axes = (
            tuple(str(axis) for axis in symmetry_parameters.get("fractional_symmetry_axes", ()))
            if isinstance(symmetry_parameters, dict)
            else ()
        )
        try:
            inference = infer_component_motion_axis(
                selected,
                self._resources_by_id,
                fractional_symmetry_axes=fractional_symmetry_axes,
                mesh_cache=self._mesh_cache,
            )
        except (ValueError, OSError) as exc:
            self._axis_inference = None
            self._axis_inference_error = str(exc)
            self.axis_confidence_label.setText(str(exc))
            return None
        inferred_axis = np.asarray(inference.axis, dtype=float)
        current_axis = np.asarray([spin.value() for spin in self.axis_spins], dtype=float)
        if float(np.linalg.norm(current_axis)) > 0.0 and float(np.dot(inferred_axis, current_axis)) < 0.0:
            inferred_axis *= -1.0
        for spin, value in zip(self.axis_spins, inferred_axis):
            spin.setValue(float(value))
        self._axis_inference = inference
        self._axis_inference_error = None
        quality = "High" if inference.confidence >= 0.8 else "Moderate" if inference.confidence >= 0.2 else "Low"
        self.axis_confidence_label.setText(
            f"{quality} confidence ({inference.confidence:.0%}); "
            f"{inference.triangle_count} triangles, projected-normal alignment "
            f"{inference.mean_squared_alignment:.0%}."
        )
        return inference

    def _flip_axis(self) -> None:
        for spin in self.axis_spins:
            spin.setValue(-spin.value())


def _component_symmetry_options(symmetry_mode: str) -> tuple[tuple[str, dict], ...]:
    if symmetry_mode == "x":
        return (
            (
                "Complete driver mirrored across X",
                {
                    "symmetry_role": "complete_representative",
                    "surface_completion_factor": 1,
                    "physical_driver_orbit_count": 2,
                    "fractional_symmetry_axes": [],
                },
            ),
            (
                "One driver cut by X",
                {
                    "symmetry_role": "fractional_driver",
                    "surface_completion_factor": 2,
                    "physical_driver_orbit_count": 1,
                    "fractional_symmetry_axes": ["x"],
                },
            ),
        )
    if symmetry_mode == "xy":
        return (
            (
                "Complete driver in a four-driver orbit",
                {
                    "symmetry_role": "complete_representative",
                    "surface_completion_factor": 1,
                    "physical_driver_orbit_count": 4,
                    "fractional_symmetry_axes": [],
                },
            ),
            (
                "Driver cut by X, mirrored across Y",
                {
                    "symmetry_role": "fractional_driver",
                    "surface_completion_factor": 2,
                    "physical_driver_orbit_count": 2,
                    "fractional_symmetry_axes": ["x"],
                },
            ),
            (
                "Driver cut by Y, mirrored across X",
                {
                    "symmetry_role": "fractional_driver",
                    "surface_completion_factor": 2,
                    "physical_driver_orbit_count": 2,
                    "fractional_symmetry_axes": ["y"],
                },
            ),
            (
                "One driver cut by X and Y",
                {
                    "symmetry_role": "fractional_driver",
                    "surface_completion_factor": 4,
                    "physical_driver_orbit_count": 1,
                    "fractional_symmetry_axes": ["x", "y"],
                },
            ),
        )
    return (
        (
            "Complete driver",
            {
                "symmetry_role": "complete_representative",
                "surface_completion_factor": 1,
                "physical_driver_orbit_count": 1,
                "fractional_symmetry_axes": [],
            },
        ),
    )


def _component_symmetry_value(parameters: dict) -> dict:
    completion_factor = int(parameters.get("surface_completion_factor", 1))
    return {
        "symmetry_role": str(
            parameters.get(
                "symmetry_role",
                "fractional_driver" if completion_factor > 1 else "complete_representative",
            )
        ),
        "surface_completion_factor": completion_factor,
        "physical_driver_orbit_count": int(parameters.get("physical_driver_orbit_count", 1)),
        "fractional_symmetry_axes": list(parameters.get("fractional_symmetry_axes", [])),
    }


class SystemConfigDialog(QDialog):
    """Edit regions, boundaries, inferred interfaces, and physical components."""

    systemApplied = Signal(object)

    def __init__(
        self,
        meshes: tuple[AvailableSystemMesh, ...],
        system: PhysicalSystem | None,
        channel_names: tuple[str, ...],
        component_channel_by_id: dict[str, str] | None = None,
        parent: QWidget | None = None,
        *,
        interface_output_root: str | Path | None = None,
        symmetry_mode: str = "off",
    ):
        super().__init__(parent)
        self.setWindowTitle("System")
        self._meshes = tuple(meshes)
        self._mesh_by_name = {mesh.name: mesh for mesh in meshes}
        self._initial_system = system
        self._channel_names = channel_names or ("main",)
        self._component_channel_by_id = dict(component_channel_by_id or {})
        self._collected_component_channels: dict[str, str] = {}
        self._mesh_file_overrides_by_name: dict[str, str] = {}
        self._interface_status_by_id: dict[str, str] = {}
        self._restored_resources_by_mesh_name: dict[str, MeshResource] = {}
        self._symmetry_mode = normalize_symmetry(symmetry_mode)
        self._interface_output_root = (
            Path.cwd() / "runs" / "imported_meshes" if interface_output_root is None else Path(interface_output_root)
        )
        self._interfaces = list(system.interfaces if system is not None else ())
        self._existing_regions = {region.id: region for region in (() if system is None else system.regions)}
        self._existing_boundaries = {
            (boundary.region_id, boundary.group.mesh_id, boundary.group.name): boundary
            for boundary in (() if system is None else system.boundaries)
        }
        self._existing_components = {
            component.id: component for component in (() if system is None else system.components)
        }
        self._component_drafts: list[_ComponentDraft] = []
        self._motion_axis_mesh_cache: dict[str, meshio.Mesh] = {}
        self._interface_mesh_cache: dict[tuple[str, float, tuple[float, float, float]], meshio.Mesh] = {}

        self.tabs = QTabWidget()
        self.regions_tab = QWidget()
        self.boundaries_tab = QWidget()
        self.interfaces_tab = QWidget()
        self.components_tab = QWidget()
        self.tabs.addTab(self.regions_tab, "Regions")
        self.tabs.addTab(self.boundaries_tab, "Boundaries")
        self.tabs.addTab(self.interfaces_tab, "Interfaces")
        self.tabs.addTab(self.components_tab, "Components")
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self._build_regions_tab()
        self._build_boundaries_tab()
        self._build_interfaces_tab()
        self._build_components_tab()
        self._load_regions()
        self._refresh_boundaries()
        self._load_interfaces()
        self._load_components()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)
        layout.addWidget(buttons)
        self.resize(980, 560)

    def _build_regions_tab(self) -> None:
        self.regions_table = QTableWidget(0, 4)
        self.regions_table.setHorizontalHeaderLabels(["Name", "Type", "Mesh", "Volume Group"])
        self.regions_table.verticalHeader().setVisible(False)
        self.regions_table.setAlternatingRowColors(True)
        self.regions_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 4):
            self.regions_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)

        add_button = QPushButton("Add Region")
        remove_button = QPushButton("Remove")
        add_button.clicked.connect(self._add_default_region)
        remove_button.clicked.connect(self._remove_selected_regions)
        row = QHBoxLayout()
        row.addWidget(add_button)
        row.addWidget(remove_button)
        row.addStretch(1)
        layout = QVBoxLayout(self.regions_tab)
        layout.addWidget(self.regions_table)
        layout.addLayout(row)

    def _build_boundaries_tab(self) -> None:
        self.boundaries_table = QTableWidget(0, 4)
        self.boundaries_table.setHorizontalHeaderLabels(["Region", "Mesh", "Surface Group", "Assignment"])
        self.boundaries_table.verticalHeader().setVisible(False)
        self.boundaries_table.setAlternatingRowColors(True)
        self.boundaries_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.boundaries_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.boundaries_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.boundaries_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        note = QLabel(
            "Classify every surface used by a region. Boundary Lab pairs conforming bounded and unbounded "
            "interface sides."
        )
        note.setWordWrap(True)
        layout = QVBoxLayout(self.boundaries_tab)
        layout.addWidget(note)
        layout.addWidget(self.boundaries_table)

    def _build_interfaces_tab(self) -> None:
        self.identify_interfaces_button = QPushButton("Build/Identify Interfaces")
        self.identify_interfaces_button.clicked.connect(self._identify_interfaces)
        self.interfaces_table = QTableWidget(0, 4)
        self.interfaces_table.setHorizontalHeaderLabels(["Name", "Bounded Interior", "Unbounded Exterior", "Status"])
        self.interfaces_table.verticalHeader().setVisible(False)
        self.interfaces_table.setAlternatingRowColors(True)
        for column in range(3):
            self.interfaces_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        self.interfaces_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        note = QLabel(
            "Boundary Lab identifies matching interface surfaces and, when necessary, rebuilds the imported "
            "BEM side to use the FEM interface nodes and faces."
        )
        note.setWordWrap(True)
        row = QHBoxLayout()
        row.addWidget(self.identify_interfaces_button)
        row.addStretch(1)
        layout = QVBoxLayout(self.interfaces_tab)
        layout.addWidget(note)
        layout.addLayout(row)
        layout.addWidget(self.interfaces_table)

    def _build_components_tab(self) -> None:
        self.components_table = QTableWidget(0, 4)
        self.components_table.setHorizontalHeaderLabels(["Name", "Type", "Moving Boundaries", "Channel"])
        self.components_table.verticalHeader().setVisible(False)
        self.components_table.setAlternatingRowColors(True)
        self.components_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.components_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.components_table.cellDoubleClicked.connect(lambda row, _column: self._edit_component(row))
        self.components_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.components_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.components_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.components_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        add_button = QPushButton("Add Component")
        edit_button = QPushButton("Edit")
        remove_button = QPushButton("Remove")
        add_button.clicked.connect(self._add_component)
        edit_button.clicked.connect(self._edit_selected_component)
        remove_button.clicked.connect(self._remove_selected_components)
        row = QHBoxLayout()
        row.addWidget(add_button)
        row.addWidget(edit_button)
        row.addWidget(remove_button)
        row.addStretch(1)
        note = QLabel(
            "A component may drive one or more moving surfaces. Electrodynamic components use one "
            "rigid-piston degree of freedom and a 2.83 V reference excitation."
        )
        note.setWordWrap(True)
        layout = QVBoxLayout(self.components_tab)
        layout.addWidget(note)
        layout.addWidget(self.components_table)
        layout.addLayout(row)

    def _load_regions(self) -> None:
        if self._initial_system is not None and self._initial_system.regions:
            resources = {mesh.id: mesh for mesh in self._initial_system.meshes}
            for region in self._initial_system.regions:
                resource = resources.get(region.mesh_ids[0]) if region.mesh_ids else None
                volume_name = region.volume_groups[0].name if region.volume_groups else None
                mesh_name = self._restored_region_mesh_name(region.kind, resource)
                if resource is not None and mesh_name is not None:
                    restored = self._restored_resources_by_mesh_name.get(mesh_name)
                    if restored is None or restored.id == resource.id:
                        self._restored_resources_by_mesh_name[mesh_name] = resource
                available = self._mesh_by_name.get(mesh_name)
                if (
                    region.kind == AcousticRegionKind.BOUNDED_AIR
                    and available is not None
                    and volume_name not in available.volume_groups
                    and len(available.volume_groups) == 1
                ):
                    volume_name = available.volume_groups[0]
                self._append_region(
                    name=region.name,
                    kind=region.kind,
                    mesh_name=mesh_name,
                    volume_group=volume_name,
                    region_id=region.id,
                )
            return
        exterior_mesh = next((mesh.name for mesh in self._meshes if not mesh.has_tetrahedra), None)
        self._append_region(
            name="Exterior Air",
            kind=AcousticRegionKind.UNBOUNDED_AIR,
            mesh_name=exterior_mesh,
            volume_group=None,
        )

    def _restored_region_mesh_name(
        self,
        kind: AcousticRegionKind,
        resource: MeshResource | None,
    ) -> str | None:
        if resource is not None and resource.name in self._mesh_by_name:
            return resource.name
        expects_tetrahedra = kind == AcousticRegionKind.BOUNDED_AIR
        compatible = [mesh.name for mesh in self._meshes if mesh.has_tetrahedra == expects_tetrahedra]
        return compatible[0] if len(compatible) == 1 else None

    def _append_region(
        self,
        *,
        name: str,
        kind: AcousticRegionKind,
        mesh_name: str | None,
        volume_group: str | None,
        region_id: str | None = None,
    ) -> None:
        row = self.regions_table.rowCount()
        self.regions_table.insertRow(row)
        name_edit = QLineEdit(name)
        name_edit.setProperty("region_id", region_id or "")
        self.regions_table.setCellWidget(row, 0, name_edit)

        type_combo = QComboBox()
        type_combo.addItem("Bounded Interior", AcousticRegionKind.BOUNDED_AIR)
        type_combo.addItem("Unbounded Exterior", AcousticRegionKind.UNBOUNDED_AIR)
        type_combo.setCurrentIndex(0 if kind == AcousticRegionKind.BOUNDED_AIR else 1)
        self.regions_table.setCellWidget(row, 1, type_combo)

        mesh_combo = QComboBox()
        mesh_combo.addItem("", None)
        for mesh in self._meshes:
            mesh_combo.addItem(mesh.name, mesh.name)
        index = mesh_combo.findData(mesh_name)
        mesh_combo.setCurrentIndex(max(index, 0))
        self.regions_table.setCellWidget(row, 2, mesh_combo)

        volume_combo = QComboBox()
        self.regions_table.setCellWidget(row, 3, volume_combo)
        type_combo.currentIndexChanged.connect(lambda _index, r=row: self._refresh_region_volume_combo(r))
        mesh_combo.currentIndexChanged.connect(lambda _index, r=row: self._refresh_region_volume_combo(r))
        self._refresh_region_volume_combo(row, selected=volume_group)

    def _add_default_region(self) -> None:
        volume_mesh = next((mesh for mesh in self._meshes if mesh.has_tetrahedra), None)
        index = 1 + sum(
            1
            for row in range(self.regions_table.rowCount())
            if self._region_kind(row) == AcousticRegionKind.BOUNDED_AIR
        )
        self._append_region(
            name=f"Interior Air {index}",
            kind=AcousticRegionKind.BOUNDED_AIR,
            mesh_name=None if volume_mesh is None else volume_mesh.name,
            volume_group=None if volume_mesh is None or not volume_mesh.volume_groups else volume_mesh.volume_groups[0],
        )

    def _remove_selected_regions(self) -> None:
        rows = sorted({index.row() for index in self.regions_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.regions_table.removeRow(row)
        self._interfaces.clear()
        self._load_interfaces()

    def _refresh_region_volume_combo(self, row: int, *, selected: str | None = None) -> None:
        if row >= self.regions_table.rowCount():
            return
        combo = self.regions_table.cellWidget(row, 3)
        mesh_combo = self.regions_table.cellWidget(row, 2)
        if not isinstance(combo, QComboBox) or not isinstance(mesh_combo, QComboBox):
            return
        current = selected if selected is not None else combo.currentData()
        combo.clear()
        combo.addItem("", None)
        bounded = self._region_kind(row) == AcousticRegionKind.BOUNDED_AIR
        mesh = self._mesh_by_name.get(mesh_combo.currentData())
        if bounded and mesh is not None:
            for name in mesh.volume_groups:
                combo.addItem(name, name)
        combo.setEnabled(bounded)
        index = combo.findData(current)
        combo.setCurrentIndex(max(index, 0))

    def _on_tab_changed(self, index: int) -> None:
        if self.tabs.widget(index) is self.boundaries_tab:
            self._refresh_boundaries()
        elif self.tabs.widget(index) is self.components_tab:
            self._refresh_components_boundary_choices()

    def _current_boundary_assignments(self) -> dict[tuple[str, str, str], tuple[str, BoundaryKind | None]]:
        assignments = {}
        for row in range(self.boundaries_table.rowCount()):
            item = self.boundaries_table.item(row, 0)
            mesh_item = self.boundaries_table.item(row, 1)
            group_item = self.boundaries_table.item(row, 2)
            combo = self.boundaries_table.cellWidget(row, 3)
            if item is None or mesh_item is None or group_item is None or not isinstance(combo, QComboBox):
                continue
            key = (
                str(item.data(Qt.ItemDataRole.UserRole)),
                str(mesh_item.data(Qt.ItemDataRole.UserRole)),
                str(group_item.data(Qt.ItemDataRole.UserRole)),
            )
            assignments[key] = (str(combo.property("boundary_id") or ""), combo.currentData())
        return assignments

    def _refresh_boundaries(self) -> None:
        current = self._current_boundary_assignments()
        self.boundaries_table.setRowCount(0)
        try:
            regions = self._region_drafts()
        except ValueError:
            return
        for region in regions:
            mesh = self._mesh_for_region_draft(region)
            if mesh is None:
                continue
            for group_name in mesh.surface_groups:
                key = (region["id"], region["mesh_id"], group_name)
                existing = self._existing_boundaries.get(key)
                boundary_id, selected = current.get(
                    key,
                    (
                        "" if existing is None else existing.id,
                        None if existing is None else existing.kind,
                    ),
                )
                self._append_boundary_row(region, mesh, group_name, boundary_id, selected)

    def _append_boundary_row(
        self,
        region: dict,
        mesh: AvailableSystemMesh,
        group_name: str,
        boundary_id: str,
        selected: BoundaryKind | None,
    ) -> None:
        row = self.boundaries_table.rowCount()
        self.boundaries_table.insertRow(row)
        region_item = QTableWidgetItem(region["name"])
        region_item.setData(Qt.ItemDataRole.UserRole, region["id"])
        region_item.setFlags(region_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        mesh_item = QTableWidgetItem(mesh.name)
        mesh_item.setData(Qt.ItemDataRole.UserRole, region["mesh_id"])
        mesh_item.setFlags(mesh_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        group_item = QTableWidgetItem(group_name)
        group_item.setData(Qt.ItemDataRole.UserRole, group_name)
        group_item.setFlags(group_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.boundaries_table.setItem(row, 0, region_item)
        self.boundaries_table.setItem(row, 1, mesh_item)
        self.boundaries_table.setItem(row, 2, group_item)
        combo = QComboBox()
        combo.addItem("Rigid", BoundaryKind.RIGID)
        combo.addItem("Moving", BoundaryKind.MOVING)
        combo.addItem("Interface", BoundaryKind.INTERFACE)
        combo.setProperty("boundary_id", boundary_id)
        normalized = BoundaryKind.RIGID if selected in {None, BoundaryKind.UNUSED} else selected
        index = combo.findData(normalized)
        combo.setCurrentIndex(max(index, 0))
        combo.currentIndexChanged.connect(self._invalidate_identified_interfaces)
        self.boundaries_table.setCellWidget(row, 3, combo)

    def _invalidate_identified_interfaces(self, _index: int) -> None:
        self._interfaces.clear()
        self._interface_status_by_id.clear()
        self._load_interfaces()

    def _collect_boundaries(self) -> tuple[Boundary, ...]:
        boundaries = []
        used_ids: set[str] = set()
        for row in range(self.boundaries_table.rowCount()):
            region_item = self.boundaries_table.item(row, 0)
            mesh_item = self.boundaries_table.item(row, 1)
            group_item = self.boundaries_table.item(row, 2)
            combo = self.boundaries_table.cellWidget(row, 3)
            if (
                region_item is None
                or mesh_item is None
                or group_item is None
                or not isinstance(combo, QComboBox)
                or combo.currentData() is None
            ):
                continue
            region_id = str(region_item.data(Qt.ItemDataRole.UserRole))
            mesh_id = str(mesh_item.data(Qt.ItemDataRole.UserRole))
            group_name = str(group_item.data(Qt.ItemDataRole.UserRole))
            boundary_id = str(combo.property("boundary_id") or "")
            if not boundary_id:
                boundary_id = _unique_id(f"boundary:{_slug(region_id)}:{_slug(group_name)}", used_ids)
                combo.setProperty("boundary_id", boundary_id)
            used_ids.add(boundary_id)
            boundaries.append(
                Boundary(
                    id=boundary_id,
                    name=group_name,
                    region_id=region_id,
                    group=PhysicalGroupRef(mesh_id=mesh_id, dimension=2, name=group_name),
                    kind=combo.currentData(),
                )
            )
        return tuple(boundaries)

    def _identify_interfaces(self) -> None:
        self.identify_interfaces_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            regions, resources = self._collect_regions_and_resources()
            boundaries = self._collect_boundaries()
            region_by_id = {region.id: region for region in regions}
            resource_by_id = {resource.id: resource for resource in resources}
            bounded = [
                boundary
                for boundary in boundaries
                if boundary.kind == BoundaryKind.INTERFACE
                and region_by_id[boundary.region_id].kind == AcousticRegionKind.BOUNDED_AIR
            ]
            unbounded = [
                boundary
                for boundary in boundaries
                if boundary.kind == BoundaryKind.INTERFACE
                and region_by_id[boundary.region_id].kind == AcousticRegionKind.UNBOUNDED_AIR
            ]
            interfaces = []
            used_ids: set[str] = set()
            available_unbounded = list(unbounded)
            for fem_boundary in bounded:
                matches: list[_InterfacePairMatch] = []
                last_error = None
                for bem_boundary in available_unbounded:
                    try:
                        matches.append(
                            self._match_interface_pair(
                                fem_boundary,
                                bem_boundary,
                                resource_by_id=resource_by_id,
                                protected_bem_interface_names=tuple(
                                    str(other.group.name)
                                    for other in unbounded
                                    if other.id != bem_boundary.id
                                    and other.group.name is not None
                                ),
                            )
                        )
                    except InterfaceConformError as exc:
                        last_error = exc
                        continue
                if len(matches) != 1:
                    detail = "" if last_error is None or matches else f" Last check: {last_error}"
                    raise ValueError(
                        f"Interface surface '{fem_boundary.group.name}' requires exactly one compatible "
                        f"unbounded interface side; found {len(matches)}.{detail}"
                    )
                match = matches[0]
                bem_boundary = match.boundary
                interface_status = "Ready"
                if match.conformed_bem_mesh is not None:
                    fem_resource = resource_by_id[fem_boundary.group.mesh_id]
                    bem_resource = resource_by_id[bem_boundary.group.mesh_id]
                    output_path = self._write_conformed_bem_mesh(
                        match.conformed_bem_mesh,
                        bem_resource,
                        fem_resource=fem_resource,
                        fem_interface_name=str(fem_boundary.group.name),
                        bem_interface_name=str(bem_boundary.group.name),
                    )
                    updated_bem_resource = replace(bem_resource, file=str(output_path))
                    resource_by_id[bem_resource.id] = updated_bem_resource
                    self._cache_interface_mesh(updated_bem_resource, match.conformed_bem_mesh)
                    self._set_available_mesh_file(bem_resource.name, output_path)
                    interface_status = "Built"
                available_unbounded.remove(bem_boundary)
                interface_name = (
                    str(fem_boundary.group.name)
                    if fem_boundary.group.name == bem_boundary.group.name
                    else f"{fem_boundary.group.name} / {bem_boundary.group.name}"
                )
                existing = next(
                    (
                        item
                        for item in self._interfaces
                        if item.bounded_boundary_id == fem_boundary.id and item.unbounded_boundary_id == bem_boundary.id
                    ),
                    None,
                )
                interface_id = (
                    existing.id if existing is not None else _unique_id(f"interface:{_slug(interface_name)}", used_ids)
                )
                used_ids.add(interface_id)
                interfaces.append(
                    AcousticInterface(
                        id=interface_id,
                        name=interface_name,
                        bounded_boundary_id=fem_boundary.id,
                        unbounded_boundary_id=bem_boundary.id,
                    )
                )
                self._interface_status_by_id[interface_id] = interface_status
            if not interfaces:
                raise ValueError("Mark matching bounded and unbounded surface groups as Interface first.")
            self._interfaces = interfaces
            self._load_interfaces()
        except (ValueError, OSError, InterfaceConformError) as exc:
            QMessageBox.warning(self, "Build/Identify Interfaces", str(exc))
        finally:
            QApplication.restoreOverrideCursor()
            self.identify_interfaces_button.setEnabled(True)

    def _match_interface_pair(
        self,
        fem_boundary: Boundary,
        bem_boundary: Boundary,
        *,
        resource_by_id: dict[str, MeshResource],
        protected_bem_interface_names: tuple[str, ...] = (),
    ) -> _InterfacePairMatch:
        try:
            self._check_interface_pair(
                fem_boundary,
                bem_boundary,
                resource_by_id=resource_by_id,
            )
            return _InterfacePairMatch(boundary=bem_boundary)
        except InterfaceConformError:
            fem_resource = resource_by_id[fem_boundary.group.mesh_id]
            bem_resource = resource_by_id[bem_boundary.group.mesh_id]
            available = self._mesh_by_name.get(bem_resource.name)
            if available is not None and available.locked:
                raise InterfaceConformError(
                    f"BEM mesh '{bem_resource.name}' is generated/locked. Interface rebuilding currently "
                    "requires an imported BEM mesh."
                ) from None
            fem_mesh = self._interface_mesh(fem_resource)
            bem_mesh = self._interface_mesh(bem_resource)
            conformed_mesh, _result = conform_bem_interface_to_fem(
                fem_mesh,
                bem_mesh,
                fem_interface_name=str(fem_boundary.group.name),
                bem_interface_name=str(bem_boundary.group.name),
                merge_tolerance=1e-8,
                symmetry_mode=self._symmetry_mode,
                protected_bem_interface_names=protected_bem_interface_names,
            )
            return _InterfacePairMatch(
                boundary=bem_boundary,
                conformed_bem_mesh=conformed_mesh,
            )

    def _write_conformed_bem_mesh(
        self,
        transformed_mesh: meshio.Mesh,
        resource: MeshResource,
        *,
        fem_resource: MeshResource,
        fem_interface_name: str,
        bem_interface_name: str,
    ) -> Path:
        available = self._mesh_by_name.get(resource.name)
        if available is None:
            raise InterfaceConformError(f"BEM mesh '{resource.name}' is not available in the System editor.")
        if available.locked:
            raise InterfaceConformError(
                f"BEM mesh '{resource.name}' is generated/locked. Interface rebuilding currently requires "
                "an imported BEM mesh."
            )
        output_path = self._conformed_mesh_path(
            available,
            fem_resource=fem_resource,
            fem_interface_name=fem_interface_name,
            bem_interface_name=bem_interface_name,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_mesh = _mesh_in_resource_coordinates(transformed_mesh, resource)
        meshio.write(output_path, output_mesh, file_format="gmsh22", binary=False)
        return output_path

    def _conformed_mesh_path(
        self,
        mesh: AvailableSystemMesh,
        *,
        fem_resource: MeshResource,
        fem_interface_name: str,
        bem_interface_name: str,
    ) -> Path:
        identity = "|".join(
            (
                str(Path(mesh.source_file).resolve()),
                repr(mesh.scale_to_m),
                repr(mesh.translation_m),
                str(Path(fem_resource.file).resolve()),
                repr(fem_resource.scale_to_m),
                repr(fem_resource.translation_m),
                fem_interface_name,
                bem_interface_name,
                self._symmetry_mode,
            )
        )
        digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:10]
        return self._interface_output_root / f"{_slug(mesh.name)}_{digest}_interface_conformed.msh"

    def _set_available_mesh_file(self, mesh_name: str, output_path: Path) -> None:
        updated = []
        for mesh in self._meshes:
            updated.append(replace(mesh, file=str(output_path)) if mesh.name == mesh_name else mesh)
        self._meshes = tuple(updated)
        self._mesh_by_name = {mesh.name: mesh for mesh in self._meshes}
        self._mesh_file_overrides_by_name[mesh_name] = str(output_path)

    def _check_interface_pair(
        self,
        fem_boundary: Boundary,
        bem_boundary: Boundary,
        *,
        resource_by_id: dict[str, MeshResource],
    ) -> None:
        fem_resource = resource_by_id[fem_boundary.group.mesh_id]
        bem_resource = resource_by_id[bem_boundary.group.mesh_id]
        fem_mesh = self._interface_mesh(fem_resource)
        bem_mesh = self._interface_mesh(bem_resource)
        build_conforming_interface_map(
            fem_mesh,
            bem_mesh,
            fem_interface_name=str(fem_boundary.group.name),
            bem_interface_name=str(bem_boundary.group.name),
            coordinate_tolerance=1e-8,
            require_closed_bem=True,
            symmetry_mode=self._symmetry_mode,
        )

    @staticmethod
    def _interface_mesh_cache_key(
        resource: MeshResource,
    ) -> tuple[str, float, tuple[float, float, float]]:
        return (
            str(Path(resource.file).resolve()),
            float(resource.scale_to_m),
            tuple(float(value) for value in resource.translation_m),
        )

    def _interface_mesh(self, resource: MeshResource) -> meshio.Mesh:
        key = self._interface_mesh_cache_key(resource)
        mesh = self._interface_mesh_cache.get(key)
        if mesh is None:
            mesh = _transformed_mesh(resource)
            self._interface_mesh_cache[key] = mesh
        return mesh

    def _cache_interface_mesh(self, resource: MeshResource, mesh: meshio.Mesh) -> None:
        self._interface_mesh_cache[self._interface_mesh_cache_key(resource)] = mesh

    def _load_interfaces(self, *, status: str | None = None) -> None:
        self.interfaces_table.setRowCount(0)
        boundaries = {boundary.id: boundary for boundary in self._collect_boundaries()}
        regions = self._region_names_by_id()
        valid = []
        for interface in self._interfaces:
            bounded = boundaries.get(interface.bounded_boundary_id)
            unbounded = boundaries.get(interface.unbounded_boundary_id)
            if bounded is None or unbounded is None:
                continue
            row = self.interfaces_table.rowCount()
            self.interfaces_table.insertRow(row)
            values = (
                interface.name,
                f"{regions.get(bounded.region_id, bounded.region_id)} / {bounded.group.name}",
                f"{regions.get(unbounded.region_id, unbounded.region_id)} / {unbounded.group.name}",
                status or self._interface_status_by_id.get(interface.id, "Configured"),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.interfaces_table.setItem(row, column, item)
            valid.append(interface)
        self._interfaces = valid

    def _region_names_by_id(self) -> dict[str, str]:
        names = {region.id: region.name for region in self._existing_regions.values()}
        for row in range(self.regions_table.rowCount()):
            name_edit = self.regions_table.cellWidget(row, 0)
            if not isinstance(name_edit, QLineEdit):
                continue
            region_id = str(name_edit.property("region_id") or "")
            name = name_edit.text().strip()
            if region_id and name:
                names[region_id] = name
        return names

    def _load_components(self) -> None:
        if self._initial_system is None:
            return
        raw_ui = self._initial_system.metadata.get(_COMPONENT_UI_METADATA_KEY, {})
        component_ui = raw_ui if isinstance(raw_ui, dict) else {}
        for component in self._initial_system.components:
            channel = self._component_channel_by_id.get(component.id, "main")
            raw_component_ui = component_ui.get(component.id, {})
            motion_axis_mode = (
                str(raw_component_ui.get("motion_axis_mode", "manual"))
                if isinstance(raw_component_ui, dict)
                else "manual"
            )
            if motion_axis_mode not in {"automatic", "manual"}:
                motion_axis_mode = "manual"
            self._component_drafts.append(
                _ComponentDraft(
                    id=component.id,
                    name=component.name,
                    kind=component.kind,
                    boundary_ids=tuple(component.boundary_ids),
                    channel=channel,
                    parameters=dict(component.parameters),
                    motion_axis_mode=motion_axis_mode,
                )
            )
        self._render_components_table()

    def _moving_boundaries(self) -> tuple[Boundary, ...]:
        return tuple(
            boundary
            for boundary in self._collect_boundaries()
            if boundary.kind == BoundaryKind.MOVING
        )

    def _refresh_components_boundary_choices(self) -> None:
        self._render_components_table()

    def _add_component(self) -> None:
        draft = _ComponentDraft(
            id="",
            name=f"Radiator {self.components_table.rowCount() + 1}",
            kind=ComponentKind.IDEAL_VELOCITY_SOURCE,
            boundary_ids=(),
            channel=self._channel_names[0],
            parameters={"motion_profile": "uniform"},
            motion_axis_mode="automatic",
        )
        self._open_component_editor(draft, row=None)

    def _append_component_draft(
        self,
        *,
        name: str,
        boundary_ids: tuple[str, ...],
        channel: str,
        kind: ComponentKind = ComponentKind.IDEAL_VELOCITY_SOURCE,
        parameters: dict | None = None,
        component_id: str | None = None,
        motion_axis_mode: str = "manual",
    ) -> None:
        self._component_drafts.append(
            _ComponentDraft(
                id=component_id or "",
                name=name,
                kind=kind,
                boundary_ids=tuple(boundary_ids),
                channel=channel,
                parameters=(
                    {"motion_profile": "uniform"}
                    if parameters is None and kind == ComponentKind.IDEAL_VELOCITY_SOURCE
                    else dict(parameters or {})
                ),
                motion_axis_mode=motion_axis_mode,
            )
        )
        self._render_components_table()

    def _render_components_table(self) -> None:
        current_row = self.components_table.currentRow()
        boundaries = {boundary.id: boundary for boundary in self._collect_boundaries()}
        region_names = self._region_names_by_id()
        self.components_table.setRowCount(0)
        for row, draft in enumerate(self._component_drafts):
            self.components_table.insertRow(row)
            boundary_labels = []
            for boundary_id in draft.boundary_ids:
                boundary = boundaries.get(boundary_id)
                if boundary is None:
                    boundary_labels.append(f"{boundary_id} (missing)")
                    continue
                region_name = region_names.get(boundary.region_id, boundary.region_id)
                boundary_labels.append(f"{boundary.name} — {region_name}")
            kind_label = (
                "Electrodynamic Transducer"
                if draft.kind == ComponentKind.ELECTRODYNAMIC_TRANSDUCER
                else "Prescribed Velocity"
            )
            values = (
                draft.name,
                kind_label,
                ", ".join(boundary_labels) if boundary_labels else "Not configured",
                draft.channel,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if column == 2:
                    item.setToolTip("\n".join(boundary_labels))
                self.components_table.setItem(row, column, item)
        if self._component_drafts and current_row >= 0:
            self.components_table.selectRow(min(current_row, len(self._component_drafts) - 1))

    def _edit_selected_component(self) -> None:
        row = self.components_table.currentRow()
        if row >= 0:
            self._edit_component(row)

    def _edit_component(self, row: int) -> None:
        if 0 <= row < len(self._component_drafts):
            self._open_component_editor(self._component_drafts[row], row=row)

    def _open_component_editor(self, draft: _ComponentDraft, *, row: int | None) -> None:
        self._refresh_boundaries()
        boundaries = self._moving_boundaries()
        try:
            _regions, resources = self._collect_regions_and_resources()
        except ValueError as exc:
            QMessageBox.warning(self, "Component", str(exc))
            return
        unavailable = {
            boundary_id
            for index, other in enumerate(self._component_drafts)
            if row is None or index != row
            for boundary_id in other.boundary_ids
        }
        editor = _ComponentEditorDialog(
            draft,
            boundaries=boundaries,
            resources_by_id={resource.id: resource for resource in resources},
            region_names=self._region_names_by_id(),
            channel_names=self._channel_names,
            unavailable_boundary_ids=unavailable,
            symmetry_mode=self._symmetry_mode,
            mesh_cache=self._motion_axis_mesh_cache,
            parent=self,
        )
        if editor.exec() != QDialog.DialogCode.Accepted:
            return
        updated = editor.component_draft()
        if row is None:
            self._component_drafts.append(updated)
        else:
            self._component_drafts[row] = updated
        self._render_components_table()
        self.components_table.selectRow(
            len(self._component_drafts) - 1 if row is None else row
        )

    def _remove_selected_components(self) -> None:
        rows = sorted({index.row() for index in self.components_table.selectedIndexes()}, reverse=True)
        for row in rows:
            if 0 <= row < len(self._component_drafts):
                del self._component_drafts[row]
        self._render_components_table()

    def _collect_components(
        self,
        boundaries: tuple[Boundary, ...] | None = None,
    ) -> tuple[tuple[PhysicalComponent, ...], tuple[ExcitationPort, ...], dict[str, str]]:
        components = []
        ports = []
        component_channels = {}
        used_ids: set[str] = set()
        used_boundaries: set[str] = set()
        moving_boundaries = {
            boundary.id: boundary
            for boundary in (self._collect_boundaries() if boundaries is None else boundaries)
            if boundary.kind == BoundaryKind.MOVING
        }
        for draft in self._component_drafts:
            name = draft.name.strip()
            if not name:
                raise ValueError("Each component must have a name.")
            if not draft.boundary_ids:
                raise ValueError(f"Component '{name}' must select at least one moving boundary.")
            for boundary_id in draft.boundary_ids:
                if boundary_id not in moving_boundaries:
                    raise ValueError(
                        f"Component '{name}' references a boundary that is missing or no longer moving: "
                        f"{boundary_id}."
                    )
                if boundary_id in used_boundaries:
                    raise ValueError("Each moving boundary can belong to only one component.")
                used_boundaries.add(boundary_id)
            component_id = draft.id
            if not component_id:
                component_id = _unique_id(f"component:{_slug(name)}", used_ids)
                draft.id = component_id
            elif component_id in used_ids:
                raise ValueError(f"Duplicate component id: {component_id}")
            used_ids.add(component_id)
            component = PhysicalComponent(
                id=component_id,
                name=name,
                kind=draft.kind,
                boundary_ids=tuple(draft.boundary_ids),
                parameters=dict(draft.parameters),
            )
            components.append(component)
            component_channels[component_id] = draft.channel
            existing_port = next(
                (
                    port
                    for port in (() if self._initial_system is None else self._initial_system.excitation_ports)
                    if port.component_id == component_id
                ),
                None,
            )
            port_kind = (
                ExcitationPortKind.VOLTAGE
                if draft.kind == ComponentKind.ELECTRODYNAMIC_TRANSDUCER
                else ExcitationPortKind.NORMAL_VELOCITY
            )
            default_port_name = (
                f"{name} voltage"
                if port_kind == ExcitationPortKind.VOLTAGE
                else f"{name} unit normal velocity"
            )
            ports.append(
                ExcitationPort(
                    id=existing_port.id if existing_port is not None else f"excitation:{_slug(component_id)}",
                    name=(
                        existing_port.name
                        if existing_port is not None and existing_port.kind == port_kind
                        else default_port_name
                    ),
                    component_id=component_id,
                    kind=port_kind,
                )
            )
        return tuple(components), tuple(ports), component_channels

    def _region_kind(self, row: int) -> AcousticRegionKind:
        combo = self.regions_table.cellWidget(row, 1)
        return combo.currentData() if isinstance(combo, QComboBox) else AcousticRegionKind.BOUNDED_AIR

    def _region_drafts(self) -> tuple[dict, ...]:
        drafts = []
        used_ids: set[str] = set()
        resource_ids = self._resource_ids_by_mesh_name()
        for row in range(self.regions_table.rowCount()):
            name_edit = self.regions_table.cellWidget(row, 0)
            mesh_combo = self.regions_table.cellWidget(row, 2)
            volume_combo = self.regions_table.cellWidget(row, 3)
            if not isinstance(name_edit, QLineEdit) or not isinstance(mesh_combo, QComboBox):
                continue
            name = name_edit.text().strip()
            if not name:
                raise ValueError("Each region must have a name.")
            mesh_name = mesh_combo.currentData()
            if mesh_name is None:
                raise ValueError(f"Region '{name}' must select a mesh.")
            region_id = str(name_edit.property("region_id") or "")
            if not region_id:
                region_id = _unique_id(f"region:{_slug(name)}", used_ids)
                name_edit.setProperty("region_id", region_id)
            if region_id in used_ids:
                raise ValueError(f"Duplicate region id: {region_id}")
            used_ids.add(region_id)
            kind = self._region_kind(row)
            volume_group = volume_combo.currentData() if isinstance(volume_combo, QComboBox) else None
            if kind == AcousticRegionKind.BOUNDED_AIR and volume_group is None:
                raise ValueError(f"Bounded region '{name}' must select a volume group.")
            drafts.append(
                {
                    "id": region_id,
                    "name": name,
                    "kind": kind,
                    "mesh_name": str(mesh_name),
                    "mesh_id": resource_ids[str(mesh_name)],
                    "volume_group": None if volume_group is None else str(volume_group),
                }
            )
        return tuple(drafts)

    def _resource_ids_by_mesh_name(self) -> dict[str, str]:
        existing = {
            mesh.name: mesh.id for mesh in (() if self._initial_system is None else self._initial_system.meshes)
        }
        existing.update(
            {mesh_name: resource.id for mesh_name, resource in self._restored_resources_by_mesh_name.items()}
        )
        resource_ids: dict[str, str] = {}
        used = set(existing.values())
        for mesh in self._meshes:
            resource_ids[mesh.name] = existing.get(mesh.name) or _unique_id(f"mesh:{_slug(mesh.name)}", used)
            used.add(resource_ids[mesh.name])
        return resource_ids

    def _mesh_for_region_draft(self, region: dict) -> AvailableSystemMesh | None:
        return self._mesh_by_name.get(region["mesh_name"])

    def _collect_regions_and_resources(self) -> tuple[tuple[AcousticRegion, ...], tuple[MeshResource, ...]]:
        drafts = list(self._region_drafts())
        initial_resources = {
            mesh.name: mesh for mesh in (() if self._initial_system is None else self._initial_system.meshes)
        }
        initial_resources.update(self._restored_resources_by_mesh_name)
        resource_by_name: dict[str, MeshResource] = {}
        resource_ids = self._resource_ids_by_mesh_name()
        for draft in drafts:
            mesh = self._mesh_by_name[draft["mesh_name"]]
            purpose = (
                MeshPurpose.FEM_VOLUME if draft["kind"] == AcousticRegionKind.BOUNDED_AIR else MeshPurpose.BEM_SURFACE
            )
            resource = resource_by_name.get(mesh.name)
            if resource is not None and resource.purpose != purpose:
                raise ValueError(f"Mesh '{mesh.name}' cannot be both a bounded FEM and unbounded BEM region.")
            if resource is None:
                existing = initial_resources.get(mesh.name)
                resource_id = existing.id if existing is not None else resource_ids[mesh.name]
                resource = MeshResource(
                    id=resource_id,
                    name=mesh.name,
                    file=mesh.file,
                    purpose=purpose,
                    scale_to_m=mesh.scale_to_m,
                    translation_m=mesh.translation_m,
                )
                resource_by_name[mesh.name] = resource
            draft["mesh_id"] = resource.id

        regions = []
        for draft in drafts:
            existing = self._existing_regions.get(draft["id"])
            regions.append(
                AcousticRegion(
                    id=draft["id"],
                    name=draft["name"],
                    kind=draft["kind"],
                    mesh_ids=(draft["mesh_id"],),
                    volume_groups=(
                        ()
                        if draft["kind"] == AcousticRegionKind.UNBOUNDED_AIR
                        else (
                            PhysicalGroupRef(
                                mesh_id=draft["mesh_id"],
                                dimension=3,
                                name=draft["volume_group"],
                            ),
                        )
                    ),
                    sound_speed_m_per_s=343.0 if existing is None else existing.sound_speed_m_per_s,
                    density_kg_per_m3=1.21 if existing is None else existing.density_kg_per_m3,
                    loss_model={} if existing is None else dict(existing.loss_model),
                )
            )
        return tuple(regions), tuple(resource_by_name.values())

    def physical_system(self) -> PhysicalSystem:
        self._refresh_boundaries()
        regions, resources = self._collect_regions_and_resources()
        boundaries = self._collect_boundaries()
        boundary_ids = {boundary.id for boundary in boundaries}
        interfaces = tuple(
            interface
            for interface in self._interfaces
            if interface.bounded_boundary_id in boundary_ids and interface.unbounded_boundary_id in boundary_ids
        )
        components, ports, component_channels = self._collect_components(boundaries)
        self._collected_component_channels = component_channels
        moving_ids = {boundary.id for boundary in boundaries if boundary.kind == BoundaryKind.MOVING}
        owned_ids = {boundary_id for component in components for boundary_id in component.boundary_ids}
        missing = moving_ids - owned_ids
        if missing:
            raise ValueError("Each moving boundary must be assigned to a component.")
        system_id = self._initial_system.id if self._initial_system is not None else "system:loudspeaker"
        system_name = self._initial_system.name if self._initial_system is not None else "Loudspeaker System"
        metadata = {} if self._initial_system is None else dict(self._initial_system.metadata)
        metadata[_COMPONENT_UI_METADATA_KEY] = {
            draft.id: {"motion_axis_mode": draft.motion_axis_mode}
            for draft in self._component_drafts
            if draft.id
        }
        return PhysicalSystem(
            id=system_id,
            name=system_name,
            meshes=resources,
            regions=regions,
            boundaries=boundaries,
            interfaces=interfaces,
            components=components,
            excitation_ports=ports,
            metadata=metadata,
        )

    def configuration(self) -> SystemConfigResult:
        system = self.physical_system()
        return SystemConfigResult(
            system=system,
            component_channel_by_id=dict(self._collected_component_channels),
            mesh_file_overrides_by_name=dict(self._mesh_file_overrides_by_name),
        )

    def apply(self) -> bool:
        try:
            configuration = self.configuration()
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "System", str(exc))
            return False
        self.systemApplied.emit(configuration)
        return True

    def _accept(self) -> None:
        if self.apply():
            self.accept()


def _transformed_mesh(resource: MeshResource) -> meshio.Mesh:
    mesh = meshio.read(Path(resource.file))
    points = np.asarray(mesh.points, dtype=float) * float(resource.scale_to_m)
    points += np.asarray(resource.translation_m, dtype=float)
    return meshio.Mesh(
        points=points,
        cells=mesh.cells,
        point_data=mesh.point_data,
        cell_data=mesh.cell_data,
        field_data=mesh.field_data,
        cell_sets=mesh.cell_sets,
    )


def _mesh_in_resource_coordinates(mesh: meshio.Mesh, resource: MeshResource) -> meshio.Mesh:
    scale = float(resource.scale_to_m)
    if scale <= 0.0:
        raise ValueError(f"Mesh '{resource.name}' scale must be greater than zero.")
    points = np.asarray(mesh.points, dtype=float) - np.asarray(resource.translation_m, dtype=float)
    points /= scale
    return meshio.Mesh(
        points=points,
        cells=mesh.cells,
        point_data=mesh.point_data,
        cell_data=mesh.cell_data,
        field_data=mesh.field_data,
        cell_sets=mesh.cell_sets,
    )


def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value)).strip("-").lower()
    return text or "item"


def _unique_id(base: str, used: set[str]) -> str:
    if base not in used:
        return base
    index = 2
    while f"{base}-{index}" in used:
        index += 1
    return f"{base}-{index}"


__all__ = [
    "AvailableSystemMesh",
    "MotionAxisInference",
    "SystemConfigResult",
    "SystemConfigDialog",
    "infer_component_motion_axis",
    "inspect_system_meshes",
    "sync_physical_system_meshes",
]
