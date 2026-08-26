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

from blab.acoustic_materials import (
    DEFAULT_WALL_LINING_FLOW_RESISTIVITY_PA_S_PER_M2,
    DEFAULT_WALL_LINING_THICKNESS_M,
    FEM_BULK_LOSS_FACTOR_OPTIONS,
    REGION_BULK_LOSS_FACTOR_KEY,
    miki_wall_impedance_parameters,
    region_bulk_loss_factor,
    wall_impedance_parameters,
)
from blab.component_symmetry import (
    SYMMETRY_PARAMETER_KEYS,
    ComponentSymmetryInference,
    ComponentSymmetryInferenceError,
    ProjectedDiaphragmAreaInference,
    infer_component_symmetry,
    infer_projected_diaphragm_area,
)
from blab.config import normalize_symmetry
from blab.fem_topology import selected_volume_surface_tags
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

INTERFACE_SEAM_SIMPLIFICATION_WARNING = (
    "Boundary Lab simplified a mismatched interface seam by collapsing redundant boundary edges. "
    "The result passed topology and element-quality checks, but local mesh quality may have changed. "
    "Visually inspect the conformed interface and its surrounding surface in the 3D viewport before solving."
)


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
    surface_groups_by_volume: tuple[tuple[str, tuple[str, ...]], ...] = ()
    locked: bool = False

    def surface_groups_for_volume(self, volume_group: str | None) -> tuple[str, ...]:
        if volume_group is None:
            return self.surface_groups
        return dict(self.surface_groups_by_volume).get(volume_group, ())


@dataclass(frozen=True)
class SystemConfigResult:
    system: PhysicalSystem
    component_channel_by_id: dict[str, str]
    mesh_file_overrides_by_name: dict[str, str] = field(default_factory=dict)
    stitch_exterior_meshes: bool = False


@dataclass(frozen=True)
class InterfaceRebuildResult:
    system: PhysicalSystem
    mesh_file_overrides_by_name: dict[str, str] = field(default_factory=dict)
    rebuilt_interface_ids: tuple[str, ...] = ()
    quality_warning_interface_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _InterfacePairMatch:
    boundary: Boundary
    conformed_bem_mesh: meshio.Mesh | None = None
    seam_simplification_used: bool = False


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


class _RegionMeshCombo(QComboBox):
    """Single-selection combo with an exterior-region multi-mesh chooser."""

    _CHOOSE_MULTIPLE = "__choose_multiple__"

    def __init__(self, mesh_names: tuple[str, ...], parent: QWidget | None = None):
        super().__init__(parent)
        self._mesh_names = mesh_names
        self._multiple_enabled = False
        self._selected_mesh_names: tuple[str, ...] = ()
        self.addItem("", None)
        for mesh_name in mesh_names:
            self.addItem(mesh_name, mesh_name)
        self.addItem("Select multiple meshes...", self._CHOOSE_MULTIPLE)
        self.currentIndexChanged.connect(self._current_changed)
        self.activated.connect(self._activated)

    def set_multiple_enabled(self, enabled: bool) -> None:
        self._multiple_enabled = bool(enabled)
        item = self.model().item(self.findData(self._CHOOSE_MULTIPLE))
        if item is not None:
            item.setEnabled(self._multiple_enabled)

    def selected_mesh_names(self) -> tuple[str, ...]:
        return self._selected_mesh_names

    def set_selected_mesh_names(self, mesh_names: tuple[str, ...]) -> None:
        selected = tuple(name for name in mesh_names if name in self._mesh_names)
        if len(selected) <= 1:
            self._selected_mesh_names = selected
            self.setCurrentIndex(max(self.findData(selected[0] if selected else None), 0))
            return
        self._set_summary_item(selected)

    def _activated(self, _index: int) -> None:
        if self.currentData() != self._CHOOSE_MULTIPLE or not self._multiple_enabled:
            return
        previous = self._selected_mesh_names
        dialog = QDialog(self)
        dialog.setWindowTitle("Exterior Region Meshes")
        mesh_list = QListWidget()
        for mesh_name in self._mesh_names:
            item = QListWidgetItem(mesh_name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if mesh_name in previous else Qt.CheckState.Unchecked)
            mesh_list.addItem(item)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout = QVBoxLayout(dialog)
        layout.addWidget(mesh_list)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self.set_selected_mesh_names(previous)
            return
        selected = tuple(
            mesh_list.item(index).text()
            for index in range(mesh_list.count())
            if mesh_list.item(index).checkState() == Qt.CheckState.Checked
        )
        self.set_selected_mesh_names(selected)

    def _set_summary_item(self, selected: tuple[str, ...]) -> None:
        summary_index = next(
            (index for index in range(self.count()) if isinstance(self.itemData(index), tuple)),
            -1,
        )
        label = ", ".join(selected)
        if summary_index < 0:
            summary_index = self.count() - 1
            self.insertItem(summary_index, label, selected)
        else:
            self.setItemText(summary_index, label)
            self.setItemData(summary_index, selected)
        self.setCurrentIndex(summary_index)
        self._selected_mesh_names = selected

    def _current_changed(self, _index: int) -> None:
        value = self.currentData()
        if value is None:
            self._selected_mesh_names = ()
        elif isinstance(value, tuple):
            self._selected_mesh_names = tuple(str(item) for item in value)
        elif isinstance(value, str) and value != self._CHOOSE_MULTIPLE:
            self._selected_mesh_names = (value,)


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
    if len(symmetry_axes) != len(set(symmetry_axes)) or any(axis not in {"x", "y"} for axis in symmetry_axes):
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
            raise ValueError(f"Mesh '{boundary.group.mesh_id}' does not contain surface group '{boundary.group.name}'.")
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
        surface_name_by_tag = {}
        volume_tag_by_name = {}
        for name, raw in mesh.field_data.items():
            tag, dimension = map(int, np.asarray(raw).tolist())
            if dimension == 2:
                surface_groups.append(str(name))
                surface_name_by_tag[tag] = str(name)
            elif dimension == 3:
                volume_groups.append(str(name))
                volume_tag_by_name[str(name)] = tag
        surface_groups_by_volume = ()
        if any(block.type in {"tetra", "tetra4"} and len(block.data) for block in mesh.cells):
            surface_groups_by_volume = tuple(
                (
                    volume_name,
                    tuple(
                        sorted(
                            surface_name_by_tag[tag]
                            for tag in selected_volume_surface_tags(mesh, (volume_tag,))
                            if tag in surface_name_by_tag
                        )
                    ),
                )
                for volume_name, volume_tag in sorted(volume_tag_by_name.items())
            )
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
                surface_groups_by_volume=surface_groups_by_volume,
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


def interface_bem_mesh_names_for_changes(
    system: PhysicalSystem | None,
    changed_mesh_names: set[str],
) -> tuple[str, ...]:
    """Return BEM resources whose configured interfaces depend on changed meshes."""

    if system is None or not system.interfaces or not changed_mesh_names:
        return ()
    resources = {resource.id: resource for resource in system.meshes}
    boundaries = {boundary.id: boundary for boundary in system.boundaries}
    affected = set()
    for interface in system.interfaces:
        bounded = boundaries.get(interface.bounded_boundary_id)
        unbounded = boundaries.get(interface.unbounded_boundary_id)
        if bounded is None or unbounded is None:
            continue
        fem_resource = resources.get(bounded.group.mesh_id)
        bem_resource = resources.get(unbounded.group.mesh_id)
        if fem_resource is None or bem_resource is None:
            continue
        if {fem_resource.name, bem_resource.name} & changed_mesh_names:
            if fem_resource.purpose != MeshPurpose.FEM_VOLUME or bem_resource.purpose != MeshPurpose.BEM_SURFACE:
                raise InterfaceConformError(
                    f"Configured interface '{interface.name}' does not connect a FEM volume to a BEM surface."
                )
            affected.add(bem_resource.name)
    return tuple(sorted(affected))


def rebuild_configured_interfaces(
    system: PhysicalSystem,
    meshes: tuple[AvailableSystemMesh, ...],
    *,
    changed_mesh_names: set[str],
    interface_output_root: str | Path,
    symmetry_mode: str = "off",
) -> InterfaceRebuildResult:
    """Validate and, when needed, rebuild known FEM-BEM interface pairs."""

    affected_bem_names = set(interface_bem_mesh_names_for_changes(system, changed_mesh_names))
    synced_system = sync_physical_system_meshes(system, meshes)
    if not affected_bem_names:
        return InterfaceRebuildResult(system=synced_system)

    available_by_name = {mesh.name: mesh for mesh in meshes}
    resources_by_id = {resource.id: resource for resource in synced_system.meshes}
    boundaries_by_id = {boundary.id: boundary for boundary in synced_system.boundaries}
    interfaces_by_bem_name: dict[str, list[tuple[AcousticInterface, Boundary, Boundary]]] = {}
    for interface in synced_system.interfaces:
        fem_boundary = boundaries_by_id.get(interface.bounded_boundary_id)
        bem_boundary = boundaries_by_id.get(interface.unbounded_boundary_id)
        if fem_boundary is None or bem_boundary is None:
            raise InterfaceConformError(f"Configured interface '{interface.name}' references a missing boundary.")
        bem_resource = resources_by_id.get(bem_boundary.group.mesh_id)
        if bem_resource is None:
            raise InterfaceConformError(f"Configured interface '{interface.name}' references a missing BEM mesh.")
        interfaces_by_bem_name.setdefault(bem_resource.name, []).append((interface, fem_boundary, bem_boundary))

    mesh_cache: dict[tuple[str, float, tuple[float, float, float]], meshio.Mesh] = {}

    def transformed(resource: MeshResource) -> meshio.Mesh:
        key = (
            str(Path(resource.file).resolve()),
            float(resource.scale_to_m),
            tuple(float(value) for value in resource.translation_m),
        )
        if key not in mesh_cache:
            mesh_cache[key] = _transformed_mesh(resource)
        return mesh_cache[key]

    overrides: dict[str, str] = {}
    rebuilt_interface_ids: list[str] = []
    quality_warning_interface_ids: list[str] = []
    normalized_symmetry = normalize_symmetry(symmetry_mode)
    output_root = Path(interface_output_root)
    for bem_name in sorted(affected_bem_names):
        pairs = interfaces_by_bem_name.get(bem_name, [])
        if not pairs:
            continue
        _first_interface, _first_fem_boundary, first_bem_boundary = pairs[0]
        bem_resource = resources_by_id[first_bem_boundary.group.mesh_id]
        available = available_by_name.get(bem_resource.name)
        if available is None:
            raise InterfaceConformError(f"BEM mesh '{bem_resource.name}' is not available for interface rebuilding.")
        if available.locked:
            raise InterfaceConformError(
                f"BEM mesh '{bem_resource.name}' is generated/locked. Interface rebuilding currently "
                "requires an imported BEM mesh."
            )
        if available.has_tetrahedra:
            raise InterfaceConformError(f"Interface rebuild target '{bem_resource.name}' contains FEM volume elements.")

        bem_mesh = transformed(bem_resource)
        rebuilt = False
        final_fem_resource = None
        final_fem_name = ""
        final_bem_name = ""
        protected_names = tuple(
            str(pair_bem.group.name)
            for _pair_interface, _pair_fem, pair_bem in pairs
            if pair_bem.group.name is not None
        )
        for interface, fem_boundary, bem_boundary in pairs:
            fem_resource = resources_by_id.get(fem_boundary.group.mesh_id)
            if fem_resource is None:
                raise InterfaceConformError(f"Configured interface '{interface.name}' references a missing FEM mesh.")
            fem_available = available_by_name.get(fem_resource.name)
            if fem_available is None or not fem_available.has_tetrahedra:
                raise InterfaceConformError(
                    f"Interface FEM mesh '{fem_resource.name}' is missing or contains no volume elements."
                )
            fem_name = str(fem_boundary.group.name)
            interface_bem_name = str(bem_boundary.group.name)
            fem_mesh = transformed(fem_resource)
            try:
                build_conforming_interface_map(
                    fem_mesh,
                    bem_mesh,
                    fem_interface_name=fem_name,
                    bem_interface_name=interface_bem_name,
                    coordinate_tolerance=float(interface.coordinate_tolerance_m),
                    require_closed_bem=True,
                    symmetry_mode=normalized_symmetry,
                )
            except InterfaceConformError:
                bem_mesh, _result = conform_bem_interface_to_fem(
                    fem_mesh,
                    bem_mesh,
                    fem_interface_name=fem_name,
                    bem_interface_name=interface_bem_name,
                    merge_tolerance=1e-8,
                    symmetry_mode=normalized_symmetry,
                    protected_bem_interface_names=tuple(name for name in protected_names if name != interface_bem_name),
                )
                rebuilt = True
                rebuilt_interface_ids.append(interface.id)
                if _result.seam_simplification_used:
                    quality_warning_interface_ids.append(interface.id)
            final_fem_resource = fem_resource
            final_fem_name = fem_name
            final_bem_name = interface_bem_name

        if not rebuilt or final_fem_resource is None:
            continue
        for interface, fem_boundary, bem_boundary in pairs:
            fem_resource = resources_by_id[fem_boundary.group.mesh_id]
            build_conforming_interface_map(
                transformed(fem_resource),
                bem_mesh,
                fem_interface_name=str(fem_boundary.group.name),
                bem_interface_name=str(bem_boundary.group.name),
                coordinate_tolerance=float(interface.coordinate_tolerance_m),
                require_closed_bem=True,
                symmetry_mode=normalized_symmetry,
            )
        output_path = _conformed_mesh_path(
            available,
            fem_resource=final_fem_resource,
            fem_interface_name=final_fem_name,
            bem_interface_name=final_bem_name,
            interface_output_root=output_root,
            symmetry_mode=normalized_symmetry,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        meshio.write(
            output_path,
            _mesh_in_resource_coordinates(bem_mesh, bem_resource),
            file_format="gmsh22",
            binary=False,
        )
        resources_by_id[bem_resource.id] = replace(bem_resource, file=str(output_path))
        overrides[bem_resource.name] = str(output_path)

    rebuilt_system = replace(
        synced_system,
        meshes=tuple(resources_by_id[resource.id] for resource in synced_system.meshes),
    )
    return InterfaceRebuildResult(
        system=rebuilt_system,
        mesh_file_overrides_by_name=overrides,
        rebuilt_interface_ids=tuple(rebuilt_interface_ids),
        quality_warning_interface_ids=tuple(quality_warning_interface_ids),
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
        form = QFormLayout()
        for key, label, unit, display_per_si in _SEMI_INDUCTANCE_PARAMETER_FIELDS:
            edit = QLineEdit()
            value = raw.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                edit.setText(f"{float(value) * display_per_si:.12g}")
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
                display_value = float(text)
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
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Component")
        self._draft = draft
        self._boundaries_by_id = {boundary.id: boundary for boundary in boundaries}
        self._resources_by_id = resources_by_id
        self._mesh_cache = mesh_cache
        self._axis_inference: MotionAxisInference | None = None
        self._automatic_axis: np.ndarray | None = None
        self._axis_inference_error: str | None = None
        self._symmetry_mode = normalize_symmetry(symmetry_mode)
        self._symmetry_inference: ComponentSymmetryInference | None = None
        self._symmetry_inference_error: str | None = None
        self._projected_area_inference: ProjectedDiaphragmAreaInference | None = None
        self._projected_area_error: str | None = None
        raw_semi_inductance = draft.parameters.get(_SEMI_INDUCTANCE_KEY)
        self._semi_inductance_parameters = (
            dict(raw_semi_inductance) if isinstance(raw_semi_inductance, dict) else None
        )
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
        self.rear_chamber_check.setToolTip(
            "Add an ideal lumped compliance for an unmeshed sealed rear chamber."
        )
        self.rear_chamber_volume_spin.setToolTip("Net enclosed air volume in litres.")
        transducer_form = QFormLayout()
        for key, label, unit, display_per_si in _TRANSDUCER_PARAMETER_FIELDS:
            edit = QLineEdit()
            if key in draft.parameters:
                edit.setText(f"{float(draft.parameters[key]) * display_per_si:.12g}")
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
        for spin in (*self.axis_spins, *self.boundary_weight_spins):
            spin.valueChanged.connect(lambda _value: self._update_projected_area_readout())
        self._refresh_semi_inductance_controls()
        self._refresh_type_controls()
        self._refresh_axis_controls()
        self._infer_component_symmetry()
        if draft.kind == ComponentKind.ELECTRODYNAMIC_TRANSDUCER and draft.motion_axis_mode == "automatic":
            self._infer_axis()

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
        name = self.name_edit.text().strip()
        if not name:
            raise ValueError("The component must have a name.")
        boundary_ids = self.selected_boundary_ids()
        if not boundary_ids:
            raise ValueError(f"Component '{name}' must select at least one moving boundary.")
        kind = ComponentKind(self.type_combo.currentData())
        if kind == ComponentKind.ELECTRODYNAMIC_TRANSDUCER:
            symmetry_inference = self._infer_component_symmetry()
            if symmetry_inference is None:
                raise ValueError(self._symmetry_inference_error or "Component symmetry could not be inferred.")
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
            if self._semi_inductance_parameters is not None:
                parameters[_SEMI_INDUCTANCE_KEY] = dict(self._semi_inductance_parameters)
            mode = str(self.axis_mode_combo.currentData())
            if mode == "automatic":
                inference = self._infer_axis()
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
                raise ValueError(
                    self._projected_area_error or "Projected diaphragm area could not be calculated."
                )
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

    def _refresh_type_controls(self, _index: int = -1) -> None:
        electrodynamic = self.type_combo.currentData() == ComponentKind.ELECTRODYNAMIC_TRANSDUCER
        self.transducer_group.setVisible(electrodynamic)
        if electrodynamic:
            self._infer_component_symmetry()

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
                self.parameter_edits["re_ohm"].setText(
                    f"{float(self._semi_inductance_parameters['re_prime_ohm']):.12g}"
                )
            if not self.parameter_edits["le_h"].text().strip():
                self.parameter_edits["le_h"].setText(
                    f"{float(self._semi_inductance_parameters['le_h']) * 1_000.0:.12g}"
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

    def _refresh_axis_controls(self, _index: int = -1) -> None:
        automatic = self.axis_mode_combo.currentData() == "automatic"
        for spin in self.axis_spins:
            spin.setEnabled(not automatic)
        if automatic:
            self._infer_axis()
        elif not self.axis_confidence_label.text():
            self.axis_confidence_label.setText("Manual direction; it will be normalized when saved.")

    def _selected_boundaries_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == 0:
            self.boundary_weight_spins[item.row()].setEnabled(
                bool(item.flags() & Qt.ItemFlag.ItemIsEnabled) and item.checkState() == Qt.CheckState.Checked
            )
        self._infer_component_symmetry()
        if self.axis_mode_combo.currentData() == "automatic":
            self._infer_axis()

    def _infer_component_symmetry(self) -> ComponentSymmetryInference | None:
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
        self._update_projected_area_readout(inference)
        return inference

    def _update_projected_area_readout(
        self,
        symmetry_inference: ComponentSymmetryInference | None = None,
        *,
        axis: np.ndarray | None = None,
    ) -> ProjectedDiaphragmAreaInference | None:
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
                mesh_cache=self._mesh_cache,
            )
        except ComponentSymmetryInferenceError as exc:
            self._projected_area_inference = None
            self._projected_area_error = str(exc)
            self.symmetry_inference_label.setText(
                f"{inference.summary()} Projected diaphragm area unavailable: {exc}"
            )
            self.projected_area_warning_label.setVisible(False)
            return None
        self._projected_area_inference = area
        self._projected_area_error = None
        self.symmetry_inference_label.setText(
            f"{inference.summary()} Projected diaphragm area of "
            f"{area.projected_area_m2 * 10_000.0:.2f} cm²."
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

    def _infer_axis(self) -> MotionAxisInference | None:
        selected = tuple(
            self._boundaries_by_id[boundary_id]
            for boundary_id in self.selected_boundary_ids()
            if boundary_id in self._boundaries_by_id
        )
        symmetry_inference = self._infer_component_symmetry()
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
        for spin, value in zip(self.axis_spins, inferred_axis):
            spin.setValue(float(value))
        self._axis_inference = inference
        self._automatic_axis = inferred_axis.copy()
        self._axis_inference_error = None
        quality = "High" if inference.confidence >= 0.8 else "Moderate" if inference.confidence >= 0.2 else "Low"
        self.axis_confidence_label.setText(
            f"{quality} confidence ({inference.confidence:.0%}); "
            f"{inference.triangle_count} triangles, projected-normal alignment "
            f"{inference.mean_squared_alignment:.0%}."
        )
        self._update_projected_area_readout(symmetry_inference, axis=inferred_axis)
        return inference

    def _flip_axis(self) -> None:
        for spin in self.axis_spins:
            spin.setValue(-spin.value())
        if self._automatic_axis is not None:
            self._automatic_axis *= -1.0


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
        stitch_exterior_meshes: bool = False,
        interface_output_root: str | Path | None = None,
        symmetry_mode: str = "off",
        symmetry_analysis_meshes: tuple[AvailableSystemMesh, ...] | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("System")
        self._meshes = tuple(meshes)
        self._mesh_by_name = {mesh.name: mesh for mesh in meshes}
        self._symmetry_analysis_mesh_by_name = {
            mesh.name: mesh for mesh in (meshes if symmetry_analysis_meshes is None else symmetry_analysis_meshes)
        }
        self._initial_system = system
        self._channel_names = channel_names or ("main",)
        self._component_channel_by_id = dict(component_channel_by_id or {})
        self._stitch_exterior_meshes = bool(stitch_exterior_meshes)
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
        self._existing_boundaries_by_mesh_group: dict[tuple[str, str | None], list[Boundary]] = {}
        for boundary in () if system is None else system.boundaries:
            self._existing_boundaries_by_mesh_group.setdefault(
                (boundary.group.mesh_id, boundary.group.name), []
            ).append(boundary)
        self._relocatable_boundary_ids: set[str] = set()
        if system is not None:
            regions_by_id = {region.id: region for region in system.regions}
            resources_by_id = {resource.id: resource for resource in system.meshes}
            for boundary in system.boundaries:
                region = regions_by_id.get(boundary.region_id)
                resource = resources_by_id.get(boundary.group.mesh_id)
                mesh = None if resource is None else self._mesh_by_name.get(resource.name)
                volume_group = next(
                    (
                        group.name
                        for group in (() if region is None else region.volume_groups)
                        if group.mesh_id == boundary.group.mesh_id
                    ),
                    None,
                )
                if (
                    region is not None
                    and region.kind == AcousticRegionKind.BOUNDED_AIR
                    and mesh is not None
                    and boundary.group.name not in mesh.surface_groups_for_volume(volume_group)
                ):
                    self._relocatable_boundary_ids.add(boundary.id)
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
        self._refresh_interfaces_tab_availability()

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
        self.regions_table = QTableWidget(0, 5)
        self.regions_table.setHorizontalHeaderLabels(["Name", "Type", "Mesh", "Volume Group", "FEM Bulk Loss Factor"])
        self.regions_table.verticalHeader().setVisible(False)
        self.regions_table.setAlternatingRowColors(True)
        self.regions_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 5):
            self.regions_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)

        add_button = QPushButton("Add Region")
        remove_button = QPushButton("Remove")
        add_button.clicked.connect(self._add_default_region)
        remove_button.clicked.connect(self._remove_selected_regions)
        row = QHBoxLayout()
        row.addWidget(add_button)
        row.addWidget(remove_button)
        self.stitch_exterior_meshes_check = QCheckBox("Stitch exterior region meshes")
        self.stitch_exterior_meshes_check.setChecked(self._stitch_exterior_meshes)
        self.stitch_exterior_meshes_check.setToolTip(
            "Join the exterior region's mesh parts into one conforming BEM solve mesh."
        )
        row.addSpacing(16)
        row.addWidget(self.stitch_exterior_meshes_check)
        row.addStretch(1)
        layout = QVBoxLayout(self.regions_tab)
        layout.addWidget(self.regions_table)
        layout.addLayout(row)

    def _build_boundaries_tab(self) -> None:
        self.boundaries_table = QTableWidget(0, 5)
        self.boundaries_table.setHorizontalHeaderLabels(
            ["Region", "Mesh", "Surface Group", "Assignment", "Wall Impedance"]
        )
        self.boundaries_table.verticalHeader().setVisible(False)
        self.boundaries_table.setAlternatingRowColors(True)
        self.boundaries_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.boundaries_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.boundaries_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.boundaries_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.boundaries_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        note = QLabel(
            "Classify every surface used by a region. Boundary Lab auto-detects interface pairs when assigned here."
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
            "Interfaces are automatically detected and built from the boundary assignments. "
            "Ensure that the interface surfaces are coplanar and have matching element density."
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
        self.components_table = QTableWidget(0, 5)
        self.components_table.setHorizontalHeaderLabels(["Name", "Type", "Moving Boundaries", "Symmetry", "Channel"])
        self.components_table.verticalHeader().setVisible(False)
        self.components_table.setAlternatingRowColors(True)
        self.components_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.components_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.components_table.cellDoubleClicked.connect(lambda row, _column: self._edit_component(row))
        self.components_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.components_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.components_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.components_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.components_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
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
            "A component may drive one or more moving surfaces. Electrodynamic transducers"
            " assume a rigid-piston model and a 2.83 V reference excitation."
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
                region_resources = tuple(
                    resource for mesh_id in region.mesh_ids if (resource := resources.get(mesh_id)) is not None
                )
                volume_name = region.volume_groups[0].name if region.volume_groups else None
                mesh_names = tuple(
                    mesh_name
                    for resource in region_resources
                    if (mesh_name := self._restored_region_mesh_name(region.kind, resource)) is not None
                )
                for resource, mesh_name in zip(region_resources, mesh_names, strict=False):
                    restored = self._restored_resources_by_mesh_name.get(mesh_name)
                    if restored is None or restored.id == resource.id:
                        self._restored_resources_by_mesh_name[mesh_name] = resource
                available = self._mesh_by_name.get(mesh_names[0] if mesh_names else None)
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
                    mesh_name=mesh_names,
                    volume_group=volume_name,
                    region_id=region.id,
                    bulk_loss_factor=region_bulk_loss_factor(region.loss_model),
                )
            return
        exterior_meshes = tuple(mesh.name for mesh in self._meshes if not mesh.has_tetrahedra)
        if not exterior_meshes:
            volume_mesh = next((mesh for mesh in self._meshes if mesh.has_tetrahedra), None)
            self._append_region(
                name="Interior Air 1",
                kind=AcousticRegionKind.BOUNDED_AIR,
                mesh_name=None if volume_mesh is None else volume_mesh.name,
                volume_group=(
                    None if volume_mesh is None or not volume_mesh.volume_groups else volume_mesh.volume_groups[0]
                ),
                bulk_loss_factor=0.0,
            )
            return
        self._append_region(
            name="Exterior Air",
            kind=AcousticRegionKind.UNBOUNDED_AIR,
            mesh_name=exterior_meshes,
            volume_group=None,
            bulk_loss_factor=0.0,
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
        mesh_name: str | tuple[str, ...] | None,
        volume_group: str | None,
        region_id: str | None = None,
        bulk_loss_factor: float = 0.0,
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

        mesh_combo = _RegionMeshCombo(tuple(mesh.name for mesh in self._meshes))
        mesh_combo.set_multiple_enabled(kind == AcousticRegionKind.UNBOUNDED_AIR)
        selected_meshes = mesh_name if isinstance(mesh_name, tuple) else (() if mesh_name is None else (mesh_name,))
        mesh_combo.set_selected_mesh_names(tuple(selected_meshes))
        self.regions_table.setCellWidget(row, 2, mesh_combo)

        volume_combo = QComboBox()
        self.regions_table.setCellWidget(row, 3, volume_combo)
        loss_combo = QComboBox()
        for loss_factor in FEM_BULK_LOSS_FACTOR_OPTIONS:
            loss_combo.addItem(f"{loss_factor:g}", loss_factor)
        loss_index = loss_combo.findData(float(bulk_loss_factor))
        if loss_index < 0:
            loss_combo.addItem(f"{float(bulk_loss_factor):g} (existing)", float(bulk_loss_factor))
            loss_index = loss_combo.count() - 1
        loss_combo.setCurrentIndex(loss_index)
        loss_combo.setToolTip(
            "Homogeneous FEM bulk loss for this bounded region; approximate isolated-mode Q is 1/loss factor."
        )
        loss_combo.setEnabled(kind == AcousticRegionKind.BOUNDED_AIR)
        self.regions_table.setCellWidget(row, 4, loss_combo)
        type_combo.currentIndexChanged.connect(lambda _index, r=row: self._refresh_region_volume_combo(r))
        type_combo.currentIndexChanged.connect(self._refresh_interfaces_tab_availability)
        type_combo.currentIndexChanged.connect(
            lambda _index, combo=loss_combo, r=row: combo.setEnabled(
                self._region_kind(r) == AcousticRegionKind.BOUNDED_AIR
            )
        )
        type_combo.currentIndexChanged.connect(
            lambda _index, combo=mesh_combo, r=row: combo.set_multiple_enabled(
                self._region_kind(r) == AcousticRegionKind.UNBOUNDED_AIR
            )
        )
        mesh_combo.currentIndexChanged.connect(lambda _index, r=row: self._refresh_region_volume_combo(r))
        self._refresh_region_volume_combo(row, selected=volume_group)
        self._refresh_interfaces_tab_availability()

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
            bulk_loss_factor=0.0,
        )

    def _remove_selected_regions(self) -> None:
        rows = sorted({index.row() for index in self.regions_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.regions_table.removeRow(row)
        self._interfaces.clear()
        self._load_interfaces()
        self._refresh_interfaces_tab_availability()

    def _refresh_interfaces_tab_availability(self, _index: int = -1) -> None:
        has_bounded_region = any(
            self._region_kind(row) == AcousticRegionKind.BOUNDED_AIR for row in range(self.regions_table.rowCount())
        )
        has_unbounded_region = any(
            self._region_kind(row) == AcousticRegionKind.UNBOUNDED_AIR for row in range(self.regions_table.rowCount())
        )
        interfaces_available = has_bounded_region and has_unbounded_region
        tab_index = self.tabs.indexOf(self.interfaces_tab)
        self.tabs.setTabEnabled(tab_index, interfaces_available)
        self.identify_interfaces_button.setEnabled(interfaces_available)

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
        selected_meshes = mesh_combo.selected_mesh_names() if isinstance(mesh_combo, _RegionMeshCombo) else ()
        mesh = self._mesh_by_name.get(selected_meshes[0] if selected_meshes else None)
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

    def _current_boundary_assignments(self) -> dict[tuple[str, str, str], tuple[str, BoundaryKind | None, dict]]:
        assignments = {}
        for row in range(self.boundaries_table.rowCount()):
            item = self.boundaries_table.item(row, 0)
            mesh_item = self.boundaries_table.item(row, 1)
            group_item = self.boundaries_table.item(row, 2)
            combo = self.boundaries_table.cellWidget(row, 3)
            impedance_button = self.boundaries_table.cellWidget(row, 4)
            if item is None or mesh_item is None or group_item is None or not isinstance(combo, QComboBox):
                continue
            key = (
                str(item.data(Qt.ItemDataRole.UserRole)),
                str(mesh_item.data(Qt.ItemDataRole.UserRole)),
                str(group_item.data(Qt.ItemDataRole.UserRole)),
            )
            parameters = (
                dict(impedance_button.property("boundary_parameters") or {})
                if isinstance(impedance_button, QPushButton)
                else {}
            )
            assignments[key] = (str(combo.property("boundary_id") or ""), combo.currentData(), parameters)
        return assignments

    def _refresh_boundaries(self) -> None:
        current = self._current_boundary_assignments()
        self.boundaries_table.setRowCount(0)
        try:
            regions = self._region_drafts()
        except ValueError:
            return
        used_boundary_ids: set[str] = set()
        for region in regions:
            for mesh_name, mesh_id in zip(region["mesh_names"], region["mesh_ids"], strict=True):
                mesh = self._mesh_by_name.get(mesh_name)
                if mesh is None:
                    continue
                mesh_region = dict(region, mesh_id=mesh_id)
                group_names = (
                    mesh.surface_groups_for_volume(region["volume_group"])
                    if region["kind"] == AcousticRegionKind.BOUNDED_AIR
                    else mesh.surface_groups
                )
                for group_name in group_names:
                    key = (region["id"], mesh_id, group_name)
                    existing = self._existing_boundaries.get(key)
                    if existing is None:
                        candidates = [
                            boundary
                            for boundary in self._existing_boundaries_by_mesh_group.get((mesh_id, group_name), ())
                            if boundary.id not in used_boundary_ids and boundary.id in self._relocatable_boundary_ids
                        ]
                        if len(candidates) == 1:
                            existing = candidates[0]
                    boundary_id, selected, parameters = current.get(
                        key,
                        (
                            "" if existing is None else existing.id,
                            None if existing is None else existing.kind,
                            {} if existing is None else dict(existing.parameters),
                        ),
                    )
                    if boundary_id:
                        used_boundary_ids.add(boundary_id)
                    self._append_boundary_row(mesh_region, mesh, group_name, boundary_id, selected, parameters)

    def _append_boundary_row(
        self,
        region: dict,
        mesh: AvailableSystemMesh,
        group_name: str,
        boundary_id: str,
        selected: BoundaryKind | None,
        parameters: dict,
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
        if region["kind"] == AcousticRegionKind.BOUNDED_AIR:
            combo.addItem("Plane-wave tube termination", BoundaryKind.PLANE_WAVE_TUBE_TERMINATION)
        combo.setProperty("boundary_id", boundary_id)
        normalized = BoundaryKind.RIGID if selected in {None, BoundaryKind.UNUSED} else selected
        index = combo.findData(normalized)
        combo.setCurrentIndex(max(index, 0))
        combo.currentIndexChanged.connect(self._invalidate_identified_interfaces)
        self.boundaries_table.setCellWidget(row, 3, combo)
        impedance_button = QPushButton()
        impedance_button.setProperty("boundary_parameters", dict(parameters))
        impedance_button.clicked.connect(
            lambda _checked=False, button=impedance_button, assignment=combo, bounded=(region["kind"] == AcousticRegionKind.BOUNDED_AIR): (
                self._edit_wall_impedance(button, assignment, bounded)
            )
        )
        combo.currentIndexChanged.connect(
            lambda _index, button=impedance_button, assignment=combo, bounded=(region["kind"] == AcousticRegionKind.BOUNDED_AIR): (
                self._refresh_wall_impedance_button(button, assignment, bounded)
            )
        )
        self.boundaries_table.setCellWidget(row, 4, impedance_button)
        self._refresh_wall_impedance_button(
            impedance_button,
            combo,
            region["kind"] == AcousticRegionKind.BOUNDED_AIR,
        )

    @staticmethod
    def _refresh_wall_impedance_button(button: QPushButton, assignment: QComboBox, bounded: bool) -> None:
        treatment = wall_impedance_parameters(dict(button.property("boundary_parameters") or {}))
        button.setText(
            "None"
            if treatment is None
            else f"{1000.0 * float(treatment['thickness_m']):g} mm / "
            f"{float(treatment['flow_resistivity_pa_s_per_m2']):,.0f} Pa·s/m²"
        )
        button.setEnabled(bounded and assignment.currentData() == BoundaryKind.RIGID)

    def _edit_wall_impedance(self, button: QPushButton, assignment: QComboBox, bounded: bool) -> None:
        dialog = _WallImpedanceDialog(dict(button.property("boundary_parameters") or {}), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        button.setProperty("boundary_parameters", dialog.parameters())
        self._refresh_wall_impedance_button(button, assignment, bounded)

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
            impedance_button = self.boundaries_table.cellWidget(row, 4)
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
                    kind=BoundaryKind(combo.currentData()),
                    parameters=(
                        dict(impedance_button.property("boundary_parameters") or {})
                        if isinstance(impedance_button, QPushButton)
                        and combo.currentData() == BoundaryKind.RIGID
                        and impedance_button.isEnabled()
                        else {}
                    ),
                )
            )
        return tuple(boundaries)

    def _identify_interfaces(self) -> None:
        self.identify_interfaces_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        quality_warning_interfaces: list[str] = []
        identify_succeeded = False
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
                                    if other.id != bem_boundary.id and other.group.name is not None
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
                if match.seam_simplification_used:
                    quality_warning_interfaces.append(interface_name)
                    interface_status = "Built (inspect)"
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
            identify_succeeded = True
        except (ValueError, OSError, InterfaceConformError) as exc:
            QMessageBox.warning(self, "Build/Identify Interfaces", str(exc))
        finally:
            QApplication.restoreOverrideCursor()
            self.identify_interfaces_button.setEnabled(True)
        if identify_succeeded and quality_warning_interfaces:
            names = ", ".join(quality_warning_interfaces)
            QMessageBox.warning(
                self,
                "Inspect Simplified Interface",
                f"{INTERFACE_SEAM_SIMPLIFICATION_WARNING}\n\nInterface: {names}",
            )

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
            conformed_mesh, result = conform_bem_interface_to_fem(
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
                seam_simplification_used=result.seam_simplification_used,
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
        return _conformed_mesh_path(
            mesh,
            fem_resource=fem_resource,
            fem_interface_name=fem_interface_name,
            bem_interface_name=bem_interface_name,
            interface_output_root=self._interface_output_root,
            symmetry_mode=self._symmetry_mode,
        )

    def _set_available_mesh_file(self, mesh_name: str, output_path: Path) -> None:
        updated = []
        for mesh in self._meshes:
            updated.append(replace(mesh, file=str(output_path)) if mesh.name == mesh_name else mesh)
        self._meshes = tuple(updated)
        self._mesh_by_name = {mesh.name: mesh for mesh in self._meshes}
        analysis_mesh = self._symmetry_analysis_mesh_by_name.get(mesh_name)
        if analysis_mesh is not None:
            self._symmetry_analysis_mesh_by_name[mesh_name] = replace(
                analysis_mesh,
                file=str(output_path),
            )
        self._motion_axis_mesh_cache.clear()
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
        return tuple(boundary for boundary in self._collect_boundaries() if boundary.kind == BoundaryKind.MOVING)

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
        try:
            _regions, resources = self._collect_regions_and_resources()
            analysis_resources = self._symmetry_analysis_resources_by_id(resources)
        except ValueError:
            analysis_resources = {}
        self.components_table.setRowCount(0)
        for row, draft in enumerate(self._component_drafts):
            self.components_table.insertRow(row)
            boundary_labels = []
            raw_weights = draft.parameters.get(_BOUNDARY_MOTION_WEIGHTS_KEY, {})
            weights = raw_weights if isinstance(raw_weights, dict) else {}
            for boundary_id in draft.boundary_ids:
                boundary = boundaries.get(boundary_id)
                if boundary is None:
                    boundary_labels.append(f"{boundary_id} (missing)")
                    continue
                region_name = region_names.get(boundary.region_id, boundary.region_id)
                try:
                    weight = float(weights.get(boundary_id, 1.0))
                except (TypeError, ValueError):
                    weight = 1.0
                offset_db = 20.0 * np.log10(max(weight, 1.0e-6))
                boundary_labels.append(f"{boundary.name} — {region_name} ({offset_db:g} dB)")
            kind_label = (
                "Electrodynamic Transducer"
                if draft.kind == ComponentKind.ELECTRODYNAMIC_TRANSDUCER
                else "Prescribed Velocity"
            )
            symmetry_summary = "Handled by acoustic symmetry"
            if draft.kind == ComponentKind.ELECTRODYNAMIC_TRANSDUCER:
                selected = tuple(
                    boundaries[boundary_id] for boundary_id in draft.boundary_ids if boundary_id in boundaries
                )
                try:
                    inference = infer_component_symmetry(
                        selected,
                        analysis_resources,
                        self._symmetry_mode,
                        mesh_cache=self._motion_axis_mesh_cache,
                    )
                    symmetry_summary = inference.summary()
                except ComponentSymmetryInferenceError as exc:
                    symmetry_summary = f"Could not infer: {exc}"
            values = (
                draft.name,
                kind_label,
                ", ".join(boundary_labels) if boundary_labels else "Not configured",
                symmetry_summary,
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
            resources_by_id=self._symmetry_analysis_resources_by_id(resources),
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
        self.components_table.selectRow(len(self._component_drafts) - 1 if row is None else row)

    def _symmetry_analysis_resources_by_id(
        self,
        resources: tuple[MeshResource, ...],
    ) -> dict[str, MeshResource]:
        resolved = {}
        for resource in resources:
            analysis_mesh = self._symmetry_analysis_mesh_by_name.get(resource.name)
            if analysis_mesh is None:
                resolved[resource.id] = resource
                continue
            resolved[resource.id] = replace(
                resource,
                file=analysis_mesh.file,
                scale_to_m=analysis_mesh.scale_to_m,
                translation_m=analysis_mesh.translation_m,
            )
        return resolved

    def _refresh_component_symmetry_parameters(
        self,
        boundaries: tuple[Boundary, ...],
        resources: tuple[MeshResource, ...],
    ) -> None:
        boundaries_by_id = {boundary.id: boundary for boundary in boundaries}
        analysis_resources = self._symmetry_analysis_resources_by_id(resources)
        for draft in self._component_drafts:
            if draft.kind != ComponentKind.ELECTRODYNAMIC_TRANSDUCER:
                continue
            selected = tuple(
                boundaries_by_id[boundary_id] for boundary_id in draft.boundary_ids if boundary_id in boundaries_by_id
            )
            inference = infer_component_symmetry(
                selected,
                analysis_resources,
                self._symmetry_mode,
                mesh_cache=self._motion_axis_mesh_cache,
            )
            parameters = {key: value for key, value in draft.parameters.items() if key not in SYMMETRY_PARAMETER_KEYS}
            parameters.update(inference.parameters())
            draft.parameters = parameters

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
                        f"Component '{name}' references a boundary that is missing or no longer moving: {boundary_id}."
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
                f"{name} voltage" if port_kind == ExcitationPortKind.VOLTAGE else f"{name} unit normal velocity"
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
        return (
            AcousticRegionKind(combo.currentData())
            if isinstance(combo, QComboBox)
            else AcousticRegionKind.BOUNDED_AIR
        )

    def _region_drafts(self) -> tuple[dict, ...]:
        drafts = []
        used_ids: set[str] = set()
        resource_ids = self._resource_ids_by_mesh_name()
        for row in range(self.regions_table.rowCount()):
            name_edit = self.regions_table.cellWidget(row, 0)
            mesh_combo = self.regions_table.cellWidget(row, 2)
            volume_combo = self.regions_table.cellWidget(row, 3)
            loss_combo = self.regions_table.cellWidget(row, 4)
            if not isinstance(name_edit, QLineEdit) or not isinstance(mesh_combo, _RegionMeshCombo):
                continue
            name = name_edit.text().strip()
            if not name:
                raise ValueError("Each region must have a name.")
            mesh_names = mesh_combo.selected_mesh_names()
            if not mesh_names:
                raise ValueError(f"Region '{name}' must select a mesh.")
            region_id = str(name_edit.property("region_id") or "")
            if not region_id:
                region_id = _unique_id(f"region:{_slug(name)}", used_ids)
                name_edit.setProperty("region_id", region_id)
            if region_id in used_ids:
                raise ValueError(f"Duplicate region id: {region_id}")
            used_ids.add(region_id)
            kind = self._region_kind(row)
            if kind == AcousticRegionKind.BOUNDED_AIR and len(mesh_names) != 1:
                raise ValueError(f"Bounded region '{name}' must select exactly one FEM volume mesh.")
            volume_group = volume_combo.currentData() if isinstance(volume_combo, QComboBox) else None
            if kind == AcousticRegionKind.BOUNDED_AIR and volume_group is None:
                raise ValueError(f"Bounded region '{name}' must select a volume group.")
            drafts.append(
                {
                    "id": region_id,
                    "name": name,
                    "kind": kind,
                    "mesh_names": mesh_names,
                    "mesh_ids": tuple(resource_ids[mesh_name] for mesh_name in mesh_names),
                    "volume_group": None if volume_group is None else str(volume_group),
                    "bulk_loss_factor": (
                        float(loss_combo.currentData())
                        if kind == AcousticRegionKind.BOUNDED_AIR and isinstance(loss_combo, QComboBox)
                        else 0.0
                    ),
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

    def _meshes_for_region_draft(self, region: dict) -> tuple[AvailableSystemMesh, ...]:
        return tuple(
            mesh for mesh_name in region["mesh_names"] if (mesh := self._mesh_by_name.get(mesh_name)) is not None
        )

    def _collect_regions_and_resources(self) -> tuple[tuple[AcousticRegion, ...], tuple[MeshResource, ...]]:
        drafts = list(self._region_drafts())
        initial_resources = {
            mesh.name: mesh for mesh in (() if self._initial_system is None else self._initial_system.meshes)
        }
        initial_resources.update(self._restored_resources_by_mesh_name)
        resource_by_name: dict[str, MeshResource] = {}
        resource_ids = self._resource_ids_by_mesh_name()
        for draft in drafts:
            purpose = (
                MeshPurpose.FEM_VOLUME if draft["kind"] == AcousticRegionKind.BOUNDED_AIR else MeshPurpose.BEM_SURFACE
            )
            resolved_ids = []
            for mesh_name in draft["mesh_names"]:
                mesh = self._mesh_by_name[mesh_name]
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
                resolved_ids.append(resource.id)
            draft["mesh_ids"] = tuple(resolved_ids)

        regions = []
        for draft in drafts:
            existing = self._existing_regions.get(draft["id"])
            regions.append(
                AcousticRegion(
                    id=draft["id"],
                    name=draft["name"],
                    kind=draft["kind"],
                    mesh_ids=draft["mesh_ids"],
                    volume_groups=(
                        ()
                        if draft["kind"] == AcousticRegionKind.UNBOUNDED_AIR
                        else (
                            PhysicalGroupRef(
                                mesh_id=draft["mesh_ids"][0],
                                dimension=3,
                                name=draft["volume_group"],
                            ),
                        )
                    ),
                    sound_speed_m_per_s=343.0 if existing is None else existing.sound_speed_m_per_s,
                    density_kg_per_m3=1.21 if existing is None else existing.density_kg_per_m3,
                    loss_model=(
                        {}
                        if draft["kind"] == AcousticRegionKind.UNBOUNDED_AIR
                        else {REGION_BULK_LOSS_FACTOR_KEY: draft["bulk_loss_factor"]}
                    ),
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
        self._refresh_component_symmetry_parameters(boundaries, resources)
        components, ports, component_channels = self._collect_components(boundaries)
        if not any(region.kind == AcousticRegionKind.BOUNDED_AIR for region in regions):
            unsupported = [
                component.name for component in components if component.kind != ComponentKind.IDEAL_VELOCITY_SOURCE
            ]
            if unsupported:
                raise ValueError(
                    "Exterior-only systems currently support prescribed-velocity components only: "
                    + ", ".join(unsupported)
                )
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
            draft.id: {"motion_axis_mode": draft.motion_axis_mode} for draft in self._component_drafts if draft.id
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
            stitch_exterior_meshes=bool(self.stitch_exterior_meshes_check.isChecked()),
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


def _conformed_mesh_path(
    mesh: AvailableSystemMesh,
    *,
    fem_resource: MeshResource,
    fem_interface_name: str,
    bem_interface_name: str,
    interface_output_root: Path,
    symmetry_mode: str,
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
            normalize_symmetry(symmetry_mode),
        )
    )
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:10]
    return interface_output_root / f"{_slug(mesh.name)}_{digest}_interface_conformed.msh"


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
    "INTERFACE_SEAM_SIMPLIFICATION_WARNING",
    "InterfaceRebuildResult",
    "MotionAxisInference",
    "SystemConfigResult",
    "SystemConfigDialog",
    "infer_component_motion_axis",
    "interface_bem_mesh_names_for_changes",
    "inspect_system_meshes",
    "rebuild_configured_interfaces",
    "sync_physical_system_meshes",
]
