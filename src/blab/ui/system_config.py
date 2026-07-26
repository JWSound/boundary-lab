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
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

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


class SystemConfigDialog(QDialog):
    """Edit regions, boundaries, inferred interfaces, and ideal components."""

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
        self._interface_output_root = (
            Path.cwd() / "runs" / "imported_meshes"
            if interface_output_root is None
            else Path(interface_output_root)
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
        self.components_table.setHorizontalHeaderLabels(["Name", "Type", "Moving Boundary", "Channel"])
        self.components_table.verticalHeader().setVisible(False)
        self.components_table.setAlternatingRowColors(True)
        self.components_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.components_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.components_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.components_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        add_button = QPushButton("Add Component")
        remove_button = QPushButton("Remove")
        add_button.clicked.connect(self._add_component)
        remove_button.clicked.connect(self._remove_selected_components)
        row = QHBoxLayout()
        row.addWidget(add_button)
        row.addWidget(remove_button)
        row.addStretch(1)
        note = QLabel(
            "The coupled solver currently supports prescribed-velocity components. "
            "A unit normal-velocity excitation is created automatically."
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
                self._append_region(
                    name=region.name,
                    kind=region.kind,
                    mesh_name=None if resource is None else resource.name,
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
        combo.addItem("Unassigned", None)
        combo.addItem("Rigid", BoundaryKind.RIGID)
        combo.addItem("Moving", BoundaryKind.MOVING)
        combo.addItem("Interface", BoundaryKind.INTERFACE)
        combo.addItem("Unused", BoundaryKind.UNUSED)
        combo.setProperty("boundary_id", boundary_id)
        index = combo.findData(selected)
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
                    resource_by_id[bem_resource.id] = replace(bem_resource, file=str(output_path))
                    self._set_available_mesh_file(bem_resource.name, output_path)
                    interface_status = "Built"
                self._check_interface_pair(
                    fem_boundary,
                    bem_boundary,
                    resource_by_id=resource_by_id,
                )
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
                        if item.bounded_boundary_id == fem_boundary.id
                        and item.unbounded_boundary_id == bem_boundary.id
                    ),
                    None,
                )
                interface_id = (
                    existing.id
                    if existing is not None
                    else _unique_id(f"interface:{_slug(interface_name)}", used_ids)
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
            fem_mesh = _transformed_mesh(fem_resource)
            bem_mesh = _transformed_mesh(bem_resource)
            conformed_mesh, _result = conform_bem_interface_to_fem(
                fem_mesh,
                bem_mesh,
                fem_interface_name=str(fem_boundary.group.name),
                bem_interface_name=str(bem_boundary.group.name),
                merge_tolerance=1e-8,
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
        fem_mesh = _transformed_mesh(fem_resource)
        bem_mesh = _transformed_mesh(bem_resource)
        build_conforming_interface_map(
            fem_mesh,
            bem_mesh,
            fem_interface_name=str(fem_boundary.group.name),
            bem_interface_name=str(bem_boundary.group.name),
            coordinate_tolerance=1e-8,
            require_closed_bem=True,
        )

    def _load_interfaces(self, *, status: str | None = None) -> None:
        self.interfaces_table.setRowCount(0)
        boundaries = {boundary.id: boundary for boundary in self._collect_boundaries()}
        regions = {region["id"]: region["name"] for region in self._region_drafts()}
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

    def _load_components(self) -> None:
        if self._initial_system is None:
            return
        boundaries = {boundary.id: boundary for boundary in self._initial_system.boundaries}
        for component in self._initial_system.components:
            boundary_id = component.boundary_ids[0] if component.boundary_ids else None
            boundary = boundaries.get(boundary_id)
            channel = self._component_channel_by_id.get(component.id, "main")
            self._append_component_row(
                name=component.name,
                boundary_id=None if boundary is None else boundary.id,
                channel=channel,
                component_id=component.id,
            )

    def _moving_boundary_options(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (boundary.id, f"{boundary.name} ({boundary.region_id})")
            for boundary in self._collect_boundaries()
            if boundary.kind == BoundaryKind.MOVING
        )

    def _refresh_components_boundary_choices(self) -> None:
        options = self._moving_boundary_options()
        for row in range(self.components_table.rowCount()):
            combo = self.components_table.cellWidget(row, 2)
            if not isinstance(combo, QComboBox):
                continue
            current = combo.currentData()
            combo.clear()
            combo.addItem("", None)
            for boundary_id, label in options:
                combo.addItem(label, boundary_id)
            index = combo.findData(current)
            combo.setCurrentIndex(max(index, 0))

    def _add_component(self) -> None:
        self._append_component_row(
            name=f"Radiator {self.components_table.rowCount() + 1}",
            boundary_id=None,
            channel=self._channel_names[0],
        )

    def _append_component_row(
        self,
        *,
        name: str,
        boundary_id: str | None,
        channel: str,
        component_id: str | None = None,
    ) -> None:
        row = self.components_table.rowCount()
        self.components_table.insertRow(row)
        name_edit = QLineEdit(name)
        name_edit.setProperty("component_id", component_id or "")
        self.components_table.setCellWidget(row, 0, name_edit)
        type_combo = QComboBox()
        type_combo.addItem("Prescribed Velocity", ComponentKind.IDEAL_VELOCITY_SOURCE)
        self.components_table.setCellWidget(row, 1, type_combo)
        boundary_combo = QComboBox()
        boundary_combo.addItem("", None)
        for candidate_id, label in self._moving_boundary_options():
            boundary_combo.addItem(label, candidate_id)
        index = boundary_combo.findData(boundary_id)
        boundary_combo.setCurrentIndex(max(index, 0))
        self.components_table.setCellWidget(row, 2, boundary_combo)
        channel_combo = QComboBox()
        for channel_name in self._channel_names:
            channel_combo.addItem(channel_name, channel_name)
        index = channel_combo.findData(channel)
        channel_combo.setCurrentIndex(max(index, 0))
        self.components_table.setCellWidget(row, 3, channel_combo)

    def _remove_selected_components(self) -> None:
        rows = sorted({index.row() for index in self.components_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.components_table.removeRow(row)

    def _collect_components(
        self,
    ) -> tuple[tuple[PhysicalComponent, ...], tuple[ExcitationPort, ...], dict[str, str]]:
        components = []
        ports = []
        component_channels = {}
        used_ids: set[str] = set()
        used_boundaries: set[str] = set()
        for row in range(self.components_table.rowCount()):
            name_edit = self.components_table.cellWidget(row, 0)
            boundary_combo = self.components_table.cellWidget(row, 2)
            channel_combo = self.components_table.cellWidget(row, 3)
            if (
                not isinstance(name_edit, QLineEdit)
                or not isinstance(boundary_combo, QComboBox)
                or not isinstance(channel_combo, QComboBox)
            ):
                continue
            name = name_edit.text().strip()
            boundary_id = boundary_combo.currentData()
            if not name:
                raise ValueError("Each component must have a name.")
            if boundary_id is None:
                raise ValueError(f"Component '{name}' must select a moving boundary.")
            boundary_id = str(boundary_id)
            if boundary_id in used_boundaries:
                raise ValueError("Each moving boundary can belong to only one component.")
            used_boundaries.add(boundary_id)
            component_id = str(name_edit.property("component_id") or "")
            if not component_id:
                component_id = _unique_id(f"component:{_slug(name)}", used_ids)
                name_edit.setProperty("component_id", component_id)
            used_ids.add(component_id)
            component = PhysicalComponent(
                id=component_id,
                name=name,
                kind=ComponentKind.IDEAL_VELOCITY_SOURCE,
                boundary_ids=(boundary_id,),
                parameters={
                    "motion_profile": "uniform",
                },
            )
            components.append(component)
            component_channels[component_id] = str(channel_combo.currentData())
            existing_port = next(
                (
                    port
                    for port in (() if self._initial_system is None else self._initial_system.excitation_ports)
                    if port.component_id == component_id
                ),
                None,
            )
            ports.append(
                ExcitationPort(
                    id=existing_port.id if existing_port is not None else f"excitation:{_slug(component_id)}",
                    name=(
                        existing_port.name
                        if existing_port is not None
                        else f"{name} unit normal velocity"
                    ),
                    component_id=component_id,
                    kind=ExcitationPortKind.NORMAL_VELOCITY,
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
        resource_by_name: dict[str, MeshResource] = {}
        resource_ids = self._resource_ids_by_mesh_name()
        for draft in drafts:
            mesh = self._mesh_by_name[draft["mesh_name"]]
            purpose = (
                MeshPurpose.FEM_VOLUME
                if draft["kind"] == AcousticRegionKind.BOUNDED_AIR
                else MeshPurpose.BEM_SURFACE
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
        components, ports, component_channels = self._collect_components()
        self._collected_component_channels = component_channels
        moving_ids = {boundary.id for boundary in boundaries if boundary.kind == BoundaryKind.MOVING}
        owned_ids = {boundary_id for component in components for boundary_id in component.boundary_ids}
        missing = moving_ids - owned_ids
        if missing:
            raise ValueError("Each moving boundary must be assigned to a component.")
        system_id = self._initial_system.id if self._initial_system is not None else "system:loudspeaker"
        system_name = self._initial_system.name if self._initial_system is not None else "Loudspeaker System"
        metadata = {} if self._initial_system is None else dict(self._initial_system.metadata)
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
    "SystemConfigResult",
    "SystemConfigDialog",
    "inspect_system_meshes",
    "sync_physical_system_meshes",
]
