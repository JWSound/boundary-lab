"""3D view with a body/face visibility tree and sizing settings; the host's Generate meshes."""

import json
import shutil
import tempfile
import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .backend import run_worker
from .parameters import DEFAULT_SOUND_SPEED_M_PER_S

SYMMETRY_LABELS = {"off": "Off", "x": "X", "xy": "XY"}
FACE_ROLE = Qt.ItemDataRole.UserRole
SIZING = (
    ("elements_per_wavelength", "Elements per wavelength", 1.0, 50.0, 1, ""),
    ("curvature_elements_per_circle", "Curvature elements per circle", 3.0, 100.0, 0, ""),
    ("growth_rate", "Growth rate", 1.05, 3.0, 2, "×"),
)


class _Relay(QObject):
    loaded = Signal(object, object, object)  # key, preview or None, error or None


class Viewport(QWidget):
    """Visible faces of the preview tessellation in a VTK plotter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from pyvistaqt import QtInteractor

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter)
        self.setMinimumHeight(220)

    def show_faces(self, preview, visible, *, reset_camera=False):
        import pyvista as pv

        self.plotter.clear()
        if preview is not None:
            triangles = preview["triangles"][np.isin(preview["face_ids"], list(visible))]
            if len(triangles):
                cells = np.hstack([np.full((len(triangles), 1), 3), triangles]).ravel()
                self.plotter.add_mesh(pv.PolyData(preview["points"], cells), color="#c8ccd2", show_edges=True)
            if reset_camera:
                self.plotter.reset_camera()
        self.plotter.render()

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
        self.viewport = Viewport()
        splitter.addWidget(self.viewport)
        tree_panel = QWidget()
        tree_layout = QVBoxLayout(tree_panel)
        tree_layout.setContentsMargins(0, 0, 0, 0)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self.tree.setAccessibleName("Bodies and faces")
        self.tree.itemChanged.connect(self._item_changed)
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

        form = QFormLayout()
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
        self._refresh_context()
        key = (str(source.get("step_path") or ""), source.get("forward", "+Z"), source.get("up", "+Y"))
        if key != self._key:
            self._load(key)

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
            self.edited({"step_path": path})

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
        path, forward, up = key
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
                job = {"mode": "preview", "step_path": path, "forward": forward, "up": up}
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

    def _show_preview(self, preview):
        self._preview = preview
        self.geometry = None if preview is None else preview["geometry"]
        self._hidden.clear()
        self._face_items = {}
        self._syncing = True
        self.tree.clear()
        if self.geometry is not None:
            faces = {face["id"]: face for face in self.geometry["faces"]}
            for body in self.geometry["bodies"]:
                item = QTreeWidgetItem([body["name"]])
                item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate
                )
                item.setCheckState(0, Qt.CheckState.Checked)
                item.setToolTip(0, "Air body" if body["kind"] == "air" else "Exterior surface body")
                self.tree.addTopLevelItem(item)
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
        self.viewport.show_faces(preview, self.visible_faces(), reset_camera=True)

    # -- visibility ---------------------------------------------------------------------------

    def visible_faces(self):
        return set(self._face_items) - self._hidden

    def _item_changed(self, item, _column):
        face_id = item.data(0, FACE_ROLE)
        if self._syncing or face_id is None:
            return  # Bodies only propagate their state to their faces.
        visible = item.checkState(0) == Qt.CheckState.Checked
        (self._hidden.discard if visible else self._hidden.add)(face_id)
        self._syncing = True
        for other in self._face_items[face_id]:
            if other is not item:
                other.setCheckState(0, item.checkState(0))
        self._syncing = False
        self._render_timer.start(0)

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
        self.viewport.show_faces(self._preview, self.visible_faces())

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
        for widget in (self.step_path, self.browse_button, *self.spins.values()):
            widget.setEnabled(not state.active)

    def dispose(self):
        self._disposed = True
        self._cancel_preview.set()
        self._render_timer.stop()
        self.viewport.close()


def create_editor(parent, document_host):
    return CadImportEditor(parent, document_host)
