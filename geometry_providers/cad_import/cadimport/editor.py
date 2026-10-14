"""3D view with face picking, a body/face visibility tree, face groups, axes and sizing; the host's Generate meshes."""

import json
import math
import shutil
import tempfile
import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from blab.cad_import import AXES, WALLS, CadFace, CadFaceRef, face_ref, match_face_ref

from .backend import run_worker
from .parameters import DEFAULT_SOUND_SPEED_M_PER_S

SYMMETRY_LABELS = {"off": "Off", "x": "X", "xy": "XY"}
FACE_ROLE = Qt.ItemDataRole.UserRole
SIZING = (
    ("elements_per_wavelength", "Elements per wavelength", 1.0, 50.0, 1, ""),
    ("curvature_elements_per_circle", "Curvature elements per circle", 3.0, 100.0, 0, ""),
    ("growth_rate", "Growth rate", 1.05, 3.0, 2, "×"),
)
FACE_COLOUR, SELECTED_COLOUR = "#c8ccd2", "#e8a33d"


class _Relay(QObject):
    loaded = Signal(object, object, object)  # key, preview or None, error or None


def _actor_key(actor):
    return actor.GetAddressAsString("")


class Viewport(QWidget):
    """Visible faces of the preview tessellation in a VTK plotter; a click picks a face."""

    def __init__(self, parent=None, *, picked=lambda face_id, add: None):
        super().__init__(parent)
        from pyvistaqt import QtInteractor

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter)
        self.setMinimumHeight(220)
        self.picked = picked
        self._cell_faces = {}  # actor -> face id per cell
        self._press = None
        interactor = self.plotter.iren.interactor
        interactor.AddObserver("LeftButtonPressEvent", self._on_press, 10.0)
        interactor.AddObserver("LeftButtonReleaseEvent", self._on_release, 10.0)

    def show_faces(self, preview, visible, selected=frozenset(), *, reset_camera=False):
        import pyvista as pv

        self.plotter.clear()
        self._cell_faces = {}
        if preview is not None:
            for faces, colour in (
                (set(visible) - set(selected), FACE_COLOUR),
                (set(visible) & set(selected), SELECTED_COLOUR),
            ):
                mask = np.isin(preview["face_ids"], list(faces))
                if mask.any():
                    triangles = preview["triangles"][mask]
                    cells = np.hstack([np.full((len(triangles), 1), 3), triangles]).ravel()
                    actor = self.plotter.add_mesh(pv.PolyData(preview["points"], cells), color=colour, show_edges=True)
                    self._cell_faces[_actor_key(actor)] = preview["face_ids"][mask]
            if reset_camera:
                self.plotter.reset_camera()
        self.plotter.render()

    def _on_press(self, interactor, _event):
        self._press = interactor.GetEventPosition()

    def _on_release(self, interactor, _event):
        x, y = interactor.GetEventPosition()
        if self._press is None or abs(x - self._press[0]) + abs(y - self._press[1]) > 3:
            return  # a drag rotates the view
        from vtkmodules.vtkRenderingCore import vtkCellPicker

        picker = vtkCellPicker()
        picker.SetTolerance(0.0005)
        face = None
        if picker.Pick(x, y, 0, self.plotter.renderer):
            faces = self._cell_faces.get(_actor_key(picker.GetActor()))
            if faces is not None and 0 <= picker.GetCellId() < len(faces):
                face = int(faces[picker.GetCellId()])
        self.picked(face, bool(interactor.GetControlKey() or interactor.GetShiftKey()))

    def close(self):
        self.plotter.close()


class CadImportEditor:
    def __init__(self, parent, host):
        self.host = host
        self.source = {}
        self.revision = ""
        self.geometry = None
        self.loading = False
        self._preview = None
        self._key = None
        self._hidden = set()
        self._face_items = {}
        self._body_items = {}
        self._faces = None  # face id -> CadFace of the loaded preview, for matching stored face groups
        self._syncing = False
        self._fmax_hz = None
        self._disposed = False
        self._cancel_preview = threading.Event()
        self._relay = _Relay()
        self._relay.loaded.connect(self._loaded)

        self.widget = QWidget(parent)
        layout = QVBoxLayout(self.widget)
        row = QHBoxLayout()
        self.step_path = QLineEdit()
        self.step_path.setPlaceholderText("STEP file")
        self.step_path.setAccessibleName("STEP file")
        self.step_path.editingFinished.connect(lambda: self.set_step_path(self.step_path.text().strip()))
        self.browse_button = QPushButton("Open STEP…")
        self.browse_button.clicked.connect(self._browse)
        row.addWidget(self.step_path, 1)
        row.addWidget(self.browse_button)
        layout.addLayout(row)
        self.info = QLabel("Open a STEP file: solids become air bodies, surface bodies the exterior.")
        self.info.setWordWrap(True)
        layout.addWidget(self.info)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.viewport = Viewport(picked=self._viewport_picked)
        splitter.addWidget(self.viewport)
        tree_panel = QWidget()
        tree_layout = QVBoxLayout(tree_panel)
        tree_layout.setContentsMargins(0, 0, 0, 0)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Body / face", "Size (mm)"])
        self.tree.headerItem().setToolTip(1, "Element size override for an air or exterior surface body.")
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self.tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tree.setAccessibleName("Bodies and faces")
        self.tree.itemChanged.connect(self._item_changed)
        self.tree.itemDoubleClicked.connect(self._tree_double_clicked)
        self.tree.itemSelectionChanged.connect(lambda: self._render_timer.start(0))
        tree_layout.addWidget(self.tree)
        buttons = QHBoxLayout()
        self.show_all_button = QPushButton("Show all")
        self.show_all_button.clicked.connect(self.show_all)
        self.clear_button = QPushButton("Clear")
        self.clear_button.setToolTip("Show everything and clear the selection.")
        self.clear_button.clicked.connect(self.clear)
        buttons.addWidget(self.show_all_button)
        buttons.addWidget(self.clear_button)
        buttons.addStretch()
        tree_layout.addLayout(buttons)
        splitter.addWidget(tree_panel)
        layout.addWidget(splitter, 1)

        self.groups_box = QGroupBox("Face groups")
        self.groups_box.setToolTip("Pick faces in the view or the tree, then name them as a face group.")
        groups_layout = QVBoxLayout(self.groups_box)
        name_row = QHBoxLayout()
        self.group_name = QLineEdit()
        self.group_name.setPlaceholderText("Face group name")
        self.group_name.setAccessibleName("Face group name")
        self.group_button = QPushButton("Group selection")
        self.group_button.setToolTip("Put the selected faces into the named face group (new or existing).")
        self.group_button.clicked.connect(lambda: self.group_selection(self.group_name.text()))
        self.delete_group_button = QPushButton("Delete group")
        self.delete_group_button.clicked.connect(self._delete_selected_group)
        name_row.addWidget(self.group_name, 1)
        name_row.addWidget(self.group_button)
        name_row.addWidget(self.delete_group_button)
        groups_layout.addLayout(name_row)
        self.groups_table = QTableWidget(0, 3)
        self.groups_table.setHorizontalHeaderLabels(["Face group", "Faces", "Size (mm)"])
        self.groups_table.verticalHeader().hide()
        self.groups_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.groups_table.setAccessibleName("Face groups")
        self.groups_table.itemChanged.connect(self._group_item_changed)
        self.groups_table.itemSelectionChanged.connect(self._group_selected)
        groups_layout.addWidget(self.groups_table)
        layout.addWidget(self.groups_box)

        self.closed_shell_box = QGroupBox("Closed shells")
        self.closed_shell_box.setToolTip("Closed shells of surfaces can be air volumes or exterior surfaces.")
        self.closed_shell_form = QFormLayout(self.closed_shell_box)
        self.closed_shell_box.hide()
        self.closed_shell_combos = {}
        layout.addWidget(self.closed_shell_box)

        form = QFormLayout()
        self.forward_combo, self.up_combo = QComboBox(), QComboBox()
        for key, combo, label in (("forward", self.forward_combo, "Forward (+Z)"), ("up", self.up_combo, "Up (+Y)")):
            for axis in AXES:
                combo.addItem(f"CAD {axis}", axis)
            combo.setAccessibleName(label)
            combo.setToolTip(f"The CAD axis that becomes Boundary Lab {label[-3:-1]}.")
            combo.currentIndexChanged.connect(lambda _, k=key, c=combo: self.edited({k: c.currentData()}))
            form.addRow(label, combo)
        self.spins = {}
        for key, label, low, high, decimals, suffix in SIZING:
            spin = QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setDecimals(decimals)
            spin.setSingleStep(10**-decimals)
            spin.setSuffix(suffix)
            spin.setKeyboardTracking(False)
            spin.setAccessibleName(label)
            spin.valueChanged.connect(lambda value, k=key: self.edited({k: value}))
            form.addRow(label, spin)
            self.spins[key] = spin
        self.target_label = QLabel()
        self.target_label.setWordWrap(True)
        form.addRow(self.target_label)
        layout.addLayout(form)

        self.status = QLabel("Use Generate to create the meshes.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self._render_timer = QTimer(self.widget)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render)
        self.host.services.subscribe(self.event)

    # -- source -------------------------------------------------------------------------------

    def apply_source(self, source, revision):
        self.source, self.revision = dict(source), revision
        self.step_path.setText(str(source.get("step_path", "")))
        for key, spin in self.spins.items():
            blocked = spin.blockSignals(True)
            spin.setValue(float(source[key]))
            spin.blockSignals(blocked)
        for key, combo in (("forward", self.forward_combo), ("up", self.up_combo)):
            blocked = combo.blockSignals(True)
            combo.setCurrentIndex(combo.findData(source.get(key, "+Z" if key == "forward" else "+Y")))
            combo.blockSignals(blocked)
        self._refresh_context()
        key = (
            str(source.get("step_path") or ""),
            source.get("forward", "+Z"),
            source.get("up", "+Y"),
            json.dumps(source.get("closed_shells", {}), sort_keys=True),
        )
        if key != self._key:
            self._load(key)
        else:
            self._show_overrides()

    def edited(self, changes):
        try:
            snapshot = self.host.update_source(self.source | changes, expected_revision=self.revision)
        except (ValueError, RuntimeError) as exc:
            self.status.setText(str(exc))
            return
        self.apply_source(snapshot.source, snapshot.revision)
        self.status.setText("Settings changed. Use Generate to update the meshes.")

    def set_step_path(self, path):
        if path != self.source.get("step_path"):
            # Closed shell choices belong to a file; another file is asked about afresh.
            self.edited({"step_path": path, "closed_shells": {}})

    def _browse(self):
        start = str(Path(self.source.get("step_path") or Path.home()).parent)
        path, _ = QFileDialog.getOpenFileName(self.widget, "Open STEP file", start, "STEP files (*.step *.stp)")
        if path:
            self.set_step_path(path)

    # -- preview ------------------------------------------------------------------------------

    def _load(self, key):
        self._key = key
        self._cancel_preview.set()
        self._cancel_preview = cancel = threading.Event()
        self._show_preview(None)
        path, forward, up, closed_shells = key
        if not path:
            self.loading = False
            return
        if not Path(path).is_file():
            self.loading = False
            self.info.setText(f"STEP file not found: {path}")
            return
        self.loading = True
        self.info.setText(f"Reading {Path(path).name}…")
        relay = self._relay

        def work():
            output = Path(tempfile.mkdtemp(prefix="blab_cad_preview_"))
            try:
                job = {
                    "mode": "preview",
                    "step_path": path,
                    "forward": forward,
                    "up": up,
                    "closed_shells": json.loads(closed_shells),
                }
                run_worker(job, output, cancelled=cancel.is_set)
                preview = json.loads((output / "preview.json").read_text(encoding="utf-8"))
            except Exception as exc:
                if not cancel.is_set():
                    relay.loaded.emit(key, None, str(exc))
            else:
                if not cancel.is_set():
                    relay.loaded.emit(key, preview, None)
            finally:
                shutil.rmtree(output, ignore_errors=True)

        threading.Thread(target=work, name="cad-import-preview", daemon=True).start()

    def _loaded(self, key, preview, error):
        if self._disposed or key != self._key:
            return
        self.loading = False
        if error is not None:
            self.info.setText(error)
            return
        preview["points"] = np.asarray(preview["points"], dtype=float).reshape(-1, 3)
        preview["triangles"] = np.asarray(preview["triangles"], dtype=np.int64).reshape(-1, 3)
        preview["face_ids"] = np.asarray(preview["face_ids"], dtype=np.int64)
        self._show_preview(preview)
        stored = self.source.get("closed_shells", {})
        answers = {
            shell["name"]: self.ask_closed_shell(shell["name"])
            for shell in self.geometry["closed_shells"]
            if shell["name"] not in stored
        }
        if answers:
            self.edited({"closed_shells": stored | answers})

    def ask_closed_shell(self, name):
        """Ask whether closed shell ``name`` is an air volume or an exterior surface."""
        box = QMessageBox(self.widget)
        box.setWindowTitle("Closed shell")
        box.setText(
            f"'{name}' is a closed shell of surfaces. Is it an air volume (meshed inside, like a solid) "
            "or an exterior surface?"
        )
        air = box.addButton("Air volume", QMessageBox.ButtonRole.AcceptRole)
        exterior = box.addButton("Exterior surface", QMessageBox.ButtonRole.RejectRole)
        box.setEscapeButton(exterior)
        box.exec()
        return "air" if box.clickedButton() is air else "exterior"

    def _show_closed_shells(self):
        if self.geometry is None:
            # Hide only: a role change reloads the preview from inside its combo's signal.
            self.closed_shell_box.hide()
            return
        while self.closed_shell_form.rowCount():
            self.closed_shell_form.removeRow(0)
        self.closed_shell_combos = {}
        shells = self.geometry["closed_shells"]
        for shell in shells:
            combo = QComboBox()
            combo.addItem("Air volume", "air")
            combo.addItem("Exterior surface", "exterior")
            combo.setCurrentIndex(combo.findData(shell["role"]))
            combo.setAccessibleName(shell["name"])
            combo.currentIndexChanged.connect(lambda _, n=shell["name"], c=combo: self._set_role(n, c.currentData()))
            self.closed_shell_form.addRow(shell["name"], combo)
            self.closed_shell_combos[shell["name"]] = combo
        self.closed_shell_box.setVisible(bool(shells))

    def _set_role(self, name, role):
        self.edited({"closed_shells": self.source.get("closed_shells", {}) | {name: role}})

    def _show_preview(self, preview):
        self._preview = preview
        self.geometry = None if preview is None else preview["geometry"]
        self._faces = None
        self._hidden.clear()
        self._face_items = {}
        self._body_items = {}
        self._syncing = True
        self.tree.clear()
        if self.geometry is not None:
            faces = {face["id"]: face for face in self.geometry["faces"]}
            self._faces = {face_id: CadFace(**face) for face_id, face in faces.items()}
            for body in self.geometry["bodies"]:
                item = QTreeWidgetItem([body["name"]])
                item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsAutoTristate
                    | Qt.ItemFlag.ItemIsEditable
                )
                item.setCheckState(0, Qt.CheckState.Checked)
                item.setToolTip(0, "Air body" if body["kind"] == "air" else "Exterior surface body")
                item.setToolTip(1, "Double-click to override the element size (mm); empty uses the target size.")
                self.tree.addTopLevelItem(item)
                self._body_items[body["name"]] = item
                for face_id in body["face_ids"]:
                    face = faces[face_id]
                    child = QTreeWidgetItem([face["step_name"] or f"Face {face_id}"])
                    child.setFlags(
                        Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable
                    )
                    child.setData(0, FACE_ROLE, face_id)
                    child.setCheckState(0, Qt.CheckState.Checked)
                    item.addChild(child)
                    self._face_items.setdefault(face_id, []).append(child)
            low, high = self.geometry["bbox_m"]
            size = " x ".join(f"{(b - a) * 1000:.1f}" for a, b in zip(low, high, strict=True))
            lines = [f"Size {size} mm · Symmetry {SYMMETRY_LABELS[self.geometry['symmetry']]} suggested"]
            lines += [f"Warning: {warning}" for warning in self.geometry["warnings"]]
            self.info.setText("\n".join(lines))
        self._syncing = False
        self._show_overrides()
        self._show_closed_shells()
        self.viewport.show_faces(preview, self.visible_faces(), reset_camera=True)

    # -- visibility and selection -------------------------------------------------------------

    def visible_faces(self):
        return set(self._face_items) - self._hidden

    def selected_faces(self):
        return {item.data(0, FACE_ROLE) for item in self.tree.selectedItems()} - {None}

    def select_faces(self, face_ids):
        self.tree.clearSelection()
        for face_id in face_ids:
            for item in self._face_items.get(face_id, ()):
                item.setSelected(True)

    def pick_face(self, face_id, *, add=False):
        """Select ``face_id`` (as a click in the view does); ``add`` toggles it in the selection."""
        selected = self.selected_faces()
        if not add:
            selected = set()
        if face_id is not None:
            selected ^= {face_id}
        self.select_faces(selected)

    def _viewport_picked(self, face_id, add):
        self.pick_face(face_id, add=add)

    def _item_changed(self, item, column):
        if self._syncing:
            return
        if column == 1 and item.parent() is None:
            self._size_typed(lambda size: self.set_body_size(item.text(0), size), item.text(1))
            return
        face_id = item.data(0, FACE_ROLE)
        if face_id is None:
            return  # Bodies only propagate their state to their faces.
        visible = item.checkState(0) == Qt.CheckState.Checked
        (self._hidden.discard if visible else self._hidden.add)(face_id)
        self._syncing = True
        for other in self._face_items[face_id]:
            if other is not item:
                other.setCheckState(0, item.checkState(0))
        self._syncing = False
        self._render_timer.start(0)

    def _tree_double_clicked(self, item, column):
        if column == 1 and item.parent() is None:
            self.tree.editItem(item, 1)

    def show_all(self):
        self._syncing = True
        for items in self._face_items.values():
            for item in items:
                item.setCheckState(0, Qt.CheckState.Checked)
        self._syncing = False
        self._hidden.clear()
        self._render()

    def clear(self):
        """Show everything and clear the selection."""
        self.show_all()
        self.tree.clearSelection()

    def _render(self):
        self.viewport.show_faces(self._preview, self.visible_faces(), self.selected_faces())

    # -- face groups --------------------------------------------------------------------------

    def _axes(self):
        """The Forward/Up axes the loaded faces were read with."""
        _, forward, up, _ = self._key
        return {"forward": forward, "up": up}

    def _ref(self, face_id):
        return asdict(face_ref(self._faces[face_id], **self._axes()))

    def _members(self, group):
        """Face ids of a stored group in the loaded geometry, and its refs that match nothing."""
        ids, unmatched = set(), []
        for ref in group["faces"]:
            found = match_face_ref(CadFaceRef(**ref), self._faces.values(), self.geometry["bbox_m"], **self._axes())
            if found is None:
                unmatched.append(ref)
            else:
                ids.add(found)
        return ids, unmatched

    def _interface_of(self):
        return {f: i["name"] for i in self.geometry["interfaces"] for f in i["face_ids"]}

    def face_groups(self, stored=None):
        """``{name: face ids}``: ``stored`` (default: the source's) face groups, then those named in the STEP file."""
        groups = {}
        if self._faces is None:
            return groups
        for group in self.source.get("face_groups", []) if stored is None else stored:
            groups[group["name"]] = self._members(group)[0]
        interface_of = self._interface_of()
        reserved = {WALLS, *interface_of.values()}
        taken = set().union(*groups.values(), interface_of)
        for face in self._faces.values():
            if face.step_name and face.step_name not in reserved and face.id not in taken:
                groups.setdefault(face.step_name, set()).add(face.id)
        return groups

    def group_selection(self, name):
        """Put the selected faces into face group ``name``, out of any other group."""
        name = name.strip()
        selected = self.selected_faces()
        if not name or not selected or self._faces is None:
            self.status.setText("Select faces and enter a face group name first.")
            return
        interface_of = self._interface_of()
        if name in {WALLS, *interface_of.values()}:
            self.status.setText(f"Face group name '{name}' is reserved.")
            return
        if selected & interface_of.keys():
            face = min(selected & interface_of.keys())
            self.status.setText(f"Face group '{name}' contains face {face} of interface '{interface_of[face]}'.")
            return
        stored = list(self.source.get("face_groups", []))
        if name not in [g["name"] for g in stored]:
            stored.append({"name": name, "faces": []})
        groups = []
        for group in stored:
            ids, unmatched = self._members(group)
            ids = (ids | selected) if group["name"] == name else (ids - selected)
            if ids or unmatched:
                groups.append({"name": group["name"], "faces": unmatched + [self._ref(f) for f in sorted(ids)]})
        self._edit_groups(groups)

    def delete_face_group(self, name):
        self._edit_groups([g for g in self.source.get("face_groups", []) if g["name"] != name])

    def _edit_groups(self, groups):
        """Store ``groups``; size overrides of face groups no longer listed are dropped."""
        listed = self.face_groups(groups).keys() | {g["name"] for g in groups}
        sizes = {k: v for k, v in self.source.get("face_group_sizes_m", {}).items() if k in listed}
        self.edited({"face_groups": groups, "face_group_sizes_m": sizes})

    def _delete_selected_group(self):
        rows = {item.row() for item in self.groups_table.selectedItems()}
        for row in sorted(rows):
            self.delete_face_group(self.groups_table.item(row, 0).text())

    def set_face_group_size(self, name, size_m):
        """Override the element size of face group ``name`` (metres); ``None`` clears it."""
        self.edited({"face_group_sizes_m": _with_size(self.source.get("face_group_sizes_m", {}), name, size_m)})

    def set_body_size(self, name, size_m):
        """Override the element size of body ``name`` (metres); ``None`` clears it."""
        self.edited({"body_sizes_m": _with_size(self.source.get("body_sizes_m", {}), name, size_m)})

    def _size_typed(self, apply, text):
        text = text.strip()
        try:
            size = None if not text else float(text) / 1000
            if size is not None and not (math.isfinite(size) and size > 0):
                raise ValueError
        except ValueError:
            self.status.setText(f"Size override must be a positive number of millimetres, not '{text}'.")
            self._show_overrides()
            return
        apply(size)

    def _group_item_changed(self, item):
        if not self._syncing and item.column() == 2:
            name = self.groups_table.item(item.row(), 0).text()
            self._size_typed(lambda size: self.set_face_group_size(name, size), item.text())

    def _group_selected(self):
        if self._syncing:
            return
        groups = self.face_groups()
        rows = {item.row() for item in self.groups_table.selectedItems()}
        self.select_faces(set().union(*(groups[self.groups_table.item(r, 0).text()] for r in rows)))

    def _show_groups(self):
        self._syncing = True
        stored = {g["name"] for g in self.source.get("face_groups", [])}
        sizes = self.source.get("face_group_sizes_m", {})
        groups = self.face_groups()
        self.groups_table.setRowCount(len(groups))
        for row, (name, faces) in enumerate(groups.items()):
            cells = [QTableWidgetItem(name), QTableWidgetItem(str(len(faces))), QTableWidgetItem(_mm(sizes.get(name)))]
            for cell in cells[:2]:
                cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
            cells[0].setToolTip("Picked face group" if name in stored else "Named in the STEP file")
            cells[2].setToolTip("Element size override in mm; empty uses the body or target size.")
            for column, cell in enumerate(cells):
                self.groups_table.setItem(row, column, cell)
        self._syncing = False

    def _show_overrides(self):
        """Body sizes in the tree, face groups and their sizes in the table."""
        self._syncing = True
        sizes = self.source.get("body_sizes_m", {})
        for name, item in self._body_items.items():
            item.setText(1, _mm(sizes.get(name)))
        self._syncing = False
        self._show_groups()

    # -- sizing and meshing -------------------------------------------------------------------

    def _refresh_context(self):
        self._show_target_size()
        future = self.host.services.context()

        def done(result):
            if not self._disposed and not result.cancelled() and result.exception() is None:
                self._fmax_hz = result.result().freq_max_hz
                self._show_target_size()

        future.add_done_callback(done)

    def _show_target_size(self):
        if self._fmax_hz is None:
            self.target_label.setText("Target element size: waiting for the project frequency range.")
            return
        n = float(self.source.get("elements_per_wavelength", 5.0))
        size = DEFAULT_SOUND_SPEED_M_PER_S / self._fmax_hz / n
        self.target_label.setText(
            f"Target element size {size * 1000:.1f} mm = c / {self._fmax_hz:g} Hz / {n:g} "
            f"(c = {DEFAULT_SOUND_SPEED_M_PER_S:g} m/s unless the exterior region sets another)"
        )

    def event(self, event):
        if event.kind == "geometry_accepted":
            self.status.setText("Meshes accepted.")
        elif event.job is not None:
            self.status.setText(event.job.message)

    def set_operation_state(self, state):
        widgets = (
            self.step_path,
            self.browse_button,
            self.forward_combo,
            self.up_combo,
            self.groups_box,
            *self.spins.values(),
            self.closed_shell_box,
        )
        for widget in widgets:
            widget.setEnabled(not state.active)

    def dispose(self):
        self._disposed = True
        self._cancel_preview.set()
        self._render_timer.stop()
        self.viewport.close()


def _mm(size_m):
    return "" if size_m is None else f"{size_m * 1000:g}"


def _with_size(sizes, name, size_m):
    sizes = {k: v for k, v in sizes.items() if k != name}
    return sizes if size_m is None else sizes | {name: size_m}


def create_editor(parent, document_host):
    return CadImportEditor(parent, document_host)
