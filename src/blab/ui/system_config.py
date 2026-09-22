"""Minimal tabbed editor for the coupled loudspeaker physical system."""

from __future__ import annotations

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
    FEM_BULK_LOSS_FACTOR_OPTIONS,
    REGION_BULK_LOSS_FACTOR_KEY,
    region_bulk_loss_factor,
    wall_impedance_parameters,
)
from blab.component_symmetry import (
    SYMMETRY_PARAMETER_KEYS,
    ComponentSymmetryInference,
    ProjectedAreaGeometryCache,
    infer_component_symmetry,
)
from blab.config import normalize_symmetry
from blab.exterior_preparation import prepare_exterior_system
from blab.interface_conform import (
    APPLICATION_INTERFACE_GEOMETRY_TOLERANCE_M,
    InterfaceConformError,
    build_conforming_interface_map,
    conform_bem_interface_to_fem,
)
from blab.mesh_inventory import AvailableSystemMesh, inspect_system_mesh_variants, inspect_system_meshes
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
from blab.system_editing import (
    InterfaceRebuildResult,
    MotionAxisInference,
    _conformed_mesh_path,
    _mesh_in_resource_coordinates,
    _slug,
    _transformed_mesh,
    infer_component_motion_axis,
    interface_bem_mesh_names_for_changes,
    rebuild_configured_interfaces,
    sync_physical_system_meshes,
)
from blab.ui.boundary_editor import _WallImpedanceDialog
from blab.ui.component_editor import (
    _BOUNDARY_MOTION_WEIGHTS_KEY,
    _ComponentDraft,
    _ComponentEditorDialog,
)

INTERFACE_SEAM_SIMPLIFICATION_WARNING = (
    "Boundary Lab simplified a mismatched interface seam by collapsing redundant boundary edges. "
    "The result passed topology and element-quality checks, but local mesh quality may have changed. "
    "Visually inspect the conformed interface and its surrounding surface in the 3D viewport before solving."
)


@dataclass(frozen=True)
class SystemConfigResult:
    system: PhysicalSystem
    component_channel_by_id: dict[str, str]
    mesh_file_overrides_by_name: dict[str, str] = field(default_factory=dict)
    stitch_exterior_meshes: bool = False


@dataclass(frozen=True)
class _InterfacePairMatch:
    boundary: Boundary
    conformed_bem_mesh: meshio.Mesh | None = None
    seam_simplification_used: bool = False


_COMPONENT_UI_METADATA_KEY = "component_editor"


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
        stitch_tolerance_mm: float = 2.0,
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
        self._stitch_tolerance_mm = stitch_tolerance_mm
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
        self._projected_area_geometry_cache = ProjectedAreaGeometryCache()
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
        note = QLabel("A component may drive one or more moving surfaces.")
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
            if self.stitch_exterior_meshes_check.isChecked():
                # Authoring keeps canonical assets, but assembly must use the
                # same symmetry variants as preview and solve preparation.
                assembly_resources = tuple(self._symmetry_analysis_resources_by_id(resources).values())
                prepared = prepare_exterior_system(
                    PhysicalSystem(
                        id="interface-build",
                        name="Interface build",
                        meshes=assembly_resources,
                        regions=regions,
                        boundaries=boundaries,
                        interfaces=tuple(self._interfaces),
                    ),
                    stitch_tolerance_mm=self._stitch_tolerance_mm,
                    symmetry_mode=self._symmetry_mode,
                    output_root=self._interface_output_root,
                    identify_interfaces=True,
                )
                if not prepared.interfaces:
                    raise ValueError("Mark matching bounded and unbounded surface groups as Interface first.")
                self._interfaces = list(prepared.interfaces)
                self._interface_status_by_id.update({pair.id: "Ready" for pair in prepared.interfaces})
                self._load_interfaces()
                if any(entry["quality_warning_interface_ids"] for entry in prepared.metadata["exterior_preparation"]):
                    QMessageBox.warning(self, "Inspect Simplified Interface", INTERFACE_SEAM_SIMPLIFICATION_WARNING)
                return
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
                geometry_tolerance=APPLICATION_INTERFACE_GEOMETRY_TOLERANCE_M,
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
            resource.mesh_data.digest if resource.mesh_data is not None else str(Path(resource.file).resolve()),
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
                symmetry_summary = self._stored_component_symmetry_summary(draft)
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

    def _stored_component_symmetry_summary(self, draft: _ComponentDraft) -> str:
        try:
            fractional_axes = tuple(str(axis) for axis in draft.parameters["fractional_symmetry_axes"])
            completion_factor = int(draft.parameters["surface_completion_factor"])
            orbit_count = int(draft.parameters["physical_driver_orbit_count"])
        except (KeyError, TypeError, ValueError):
            return "Inferred when the component is edited or the system is applied"
        return ComponentSymmetryInference(
            symmetry_mode=self._symmetry_mode,
            fractional_symmetry_axes=fractional_axes,
            surface_completion_factor=completion_factor,
            physical_driver_orbit_count=orbit_count,
            surface_patch_count=0,
            perimeter_edge_count=0,
            plane_edge_counts=(),
        ).summary()

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
            projected_geometry_cache=self._projected_area_geometry_cache,
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
                mesh_data=analysis_mesh.mesh_data,
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
            AcousticRegionKind(combo.currentData()) if isinstance(combo, QComboBox) else AcousticRegionKind.BOUNDED_AIR
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
                        mesh_data=mesh.mesh_data,
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
    "inspect_system_mesh_variants",
    "inspect_system_meshes",
    "rebuild_configured_interfaces",
    "sync_physical_system_meshes",
]
