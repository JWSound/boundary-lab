"""3D view with face picking, a body/face visibility tree, face groups, placement and sizing.

The host's Generate meshes.
"""

import json
import math
import shutil
import tempfile
import threading
import zlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
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

from blab.cad_import import (
    WALLS,
    CadFace,
    CadFaceRef,
    CadScan,
    face_ref,
    match_face_ref,
    place_preview,
    preview_from_json,
)
from blab.ui.file_dialogs import FileDialogService

from .backend import run_worker
from .parameters import DEFAULT_SOUND_SPEED_M_PER_S

SYMMETRY_LABELS = {"off": "Off", "x": "X", "xy": "XY"}
FACE_ROLE = Qt.ItemDataRole.UserRole
SIZING = (
    ("elements_per_wavelength", "Elements per wavelength", 1.0, 50.0, 1, ""),
    ("curvature_elements_per_circle", "Curvature elements per circle", 3.0, 100.0, 0, ""),
    ("growth_rate", "Growth rate", 1.05, 3.0, 2, "×"),
)
FACE_COLOUR, SELECTED_COLOUR, OUTLINE_COLOUR = "#c8ccd2", "#e8a33d", "#2b2f36"
HIGHLIGHT_COLOUR = "#e0218a"  # a closed shell being asked about, a right-clicked face
SEE_THROUGH = 0.15  # opacity of the faces around highlighted ones
AXIS_COLOURS = ("#d62728", "#2ca02c", "#1f77b4")  # X, Y, Z
AXIS_LABELS = ("X · Horizontal", "Y · Vertical", "Z · On Axis")  # as the balloon and mesh views name them
PLANE_OPACITY = 0.15  # the symmetry planes X=0 and Y=0
GROUP_COLOURS = (
    "#4e79a7", "#59a14f", "#e15759", "#76b7b2", "#b07aa1",
    "#9c755f", "#edc948", "#ff9da7", "#2f6f4f", "#8c6bb1",
)  # fmt: skip


class _Relay(QObject):
    loaded = Signal(object, object, object)  # key, preview or None, error or None


def _actor_key(actor):
    return actor.GetAddressAsString("")


def outline_edges(triangles, face_ids):
    """Point index pairs of the triangle edges between faces or on the border: the face outlines."""
    edges = np.sort(triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    faces = np.repeat(face_ids, 3)
    order = np.lexsort((faces, edges[:, 1], edges[:, 0]))
    edges, faces = edges[order], faces[order]
    first = np.ones(len(edges), dtype=bool)
    first[1:] = (edges[1:] != edges[:-1]).any(axis=1)
    starts = np.flatnonzero(first)
    counts = np.diff(np.append(starts, len(edges)))
    # An inner edge of one face is used twice by it; any other edge outlines a face.
    inner = (counts == 2) & (faces[starts] == faces[starts + counts - 1])
    return edges[starts[~inner]]


class Viewport(QWidget):
    """Visible faces of the preview tessellation in a VTK plotter; a click picks a face, a right-click asks a menu.

    Clicks come from the plotter widget's Qt mouse events: the camera interaction holds VTK's mouse focus
    between press and release, so VTK observers never see the release.
    """

    def __init__(self, parent=None, *, picked=lambda face_id, add: None, menu=lambda face_id, global_pos: None):
        super().__init__(parent)
        from pyvistaqt import QtInteractor
        from vtkmodules.vtkRenderingCore import vtkMapper

        # Faces back off from the outlines on them, which would otherwise z-fight (a VTK-wide setting).
        vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter)
        self.setMinimumHeight(220)
        self.picked = picked
        self.menu = menu
        self._cell_faces = {}  # actor -> face id per cell
        self._press = None  # button and position of the last press
        self.plotter.installEventFilter(self)

    def show_faces(
        self, preview, visible, selected=frozenset(), colours=None, *, highlight=None, guides=None, reset_camera=False
    ):
        """Draw the ``visible`` faces smooth with their outlines, in their ``colours`` (face id -> colour, default
        plain), ``selected`` on top; ``highlight`` (face ids) stands out, all other faces see-through. ``guides``
        (a bounding box, m) draws the axes and symmetry planes around it."""
        import pyvista as pv

        self.plotter.clear()
        self._cell_faces = {}
        if preview is not None:
            by_look = {}
            for face in visible:
                colour = SELECTED_COLOUR if face in selected else (colours or {}).get(face, FACE_COLOUR)
                opacity = 1.0 if highlight is None or face in highlight else SEE_THROUGH
                if highlight is not None and face in highlight:
                    colour = HIGHLIGHT_COLOUR
                by_look.setdefault((colour, opacity), []).append(face)
            for (colour, opacity), faces in by_look.items():
                mask = np.isin(preview["face_ids"], faces)
                if mask.any():
                    triangles = preview["triangles"][mask]
                    cells = np.hstack([np.full((len(triangles), 1), 3), triangles]).ravel()
                    # pyvista would reset (zoom out) the camera on each redraw: it never counts it as set.
                    actor = self.plotter.add_mesh(
                        pv.PolyData(preview["points"], cells),
                        color=colour,
                        opacity=opacity,
                        smooth_shading=True,
                        split_sharp_edges=True,
                        reset_camera=False,
                    )
                    self._cell_faces[_actor_key(actor)] = preview["face_ids"][mask]
            outlined = {}
            for face in visible:
                outlined.setdefault(1.0 if highlight is None or face in highlight else SEE_THROUGH, []).append(face)
            for opacity, faces in outlined.items():
                mask = np.isin(preview["face_ids"], faces)
                edges = outline_edges(preview["triangles"][mask], preview["face_ids"][mask])
                if len(edges):
                    lines = np.hstack([np.full((len(edges), 1), 2), edges]).ravel()
                    self.plotter.add_mesh(
                        pv.PolyData(preview["points"], lines=lines),
                        color=OUTLINE_COLOUR,
                        opacity=opacity,
                        line_width=1.5,
                        pickable=False,
                        reset_camera=False,
                    )
            if guides is not None:
                self._show_guides(guides)
            if reset_camera:
                self.plotter.reset_camera()
        self.plotter.render()

    def _show_guides(self, bbox_m):
        """Labelled axes through the origin and the see-through symmetry planes X=0 and Y=0, all reaching
        past the model and the origin by a fifth of their span (at least a tenth of the model's size)."""
        import pyvista as pv

        low, high = np.minimum(bbox_m[0], 0), np.maximum(bbox_m[1], 0)
        margin = np.maximum(0.2 * (high - low), 0.1 * (float(np.max(np.subtract(bbox_m[1], bbox_m[0]))) or 1.0))
        low, high = low - margin, high + margin
        for axis, name in enumerate("XYZ"):
            start, tip = np.zeros(3), np.zeros(3)
            start[axis], tip[axis] = low[axis], high[axis]
            common = dict(color=AXIS_COLOURS[axis], pickable=False, reset_camera=False)
            self.plotter.add_mesh(pv.Line(start, tip), line_width=3, name=f"axis {name}", **common)
            self.plotter.add_point_labels(
                [tip],
                [AXIS_LABELS[axis]],
                text_color=AXIS_COLOURS[axis],
                shape=None,
                show_points=False,
                always_visible=True,
                name=f"label {name}",
                reset_camera=False,
            )
            if axis == 2:
                continue  # Z=0 is no symmetry plane
            i, j = (axis + 1) % 3, (axis + 2) % 3
            corners = np.zeros((4, 3))
            corners[:, i] = low[i], high[i], high[i], low[i]
            corners[:, j] = low[j], low[j], high[j], high[j]
            self.plotter.add_mesh(
                pv.PolyData(corners, [4, 0, 1, 2, 3]),
                opacity=PLANE_OPACITY,
                name=f"plane {name}=0",
                **common,
            )

    def frame(self, preview, face_ids):
        """Fit the view to ``face_ids``; returns the camera to give back to :meth:`restore_camera`."""
        camera = self.plotter.camera_position
        points = preview["points"][np.unique(preview["triangles"][np.isin(preview["face_ids"], list(face_ids))])]
        if len(points):
            low, high = points.min(axis=0), points.max(axis=0)
            self.plotter.reset_camera(bounds=[v for pair in zip(low, high, strict=True) for v in pair])
        return camera

    def restore_camera(self, camera):
        self.plotter.camera_position = camera
        self.plotter.render()

    def eventFilter(self, watched, event):
        kind = event.type()
        if kind == QEvent.Type.MouseButtonPress:
            self._press = event.button(), event.position()
        elif kind == QEvent.Type.MouseButtonRelease and self._press is not None:
            button, press = self._press
            self._press = None
            moved = event.position() - press
            if button == event.button() and abs(moved.x()) + abs(moved.y()) <= 3:  # else a drag moves the camera
                self._clicked(event)
        return False  # the camera interaction sees every event

    def _clicked(self, event):
        ratio = self.plotter.devicePixelRatioF()
        position = event.position()
        x = round(position.x() * ratio)
        y = round(self.plotter.height() * ratio) - round(position.y() * ratio) - 1
        face = self.face_at(x, y)
        if event.button() == Qt.MouseButton.LeftButton:
            add = event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
            self.picked(face, bool(add))
        elif event.button() == Qt.MouseButton.RightButton and face is not None:
            # After the release: the camera interaction gets it before the menu blocks.
            global_pos = event.globalPosition().toPoint()
            QTimer.singleShot(0, lambda: self.menu(face, global_pos))

    def face_at(self, x, y):
        """The face under VTK display position ``x``, ``y`` (pixels, y up), or ``None``."""
        from vtkmodules.vtkRenderingCore import vtkCellPicker

        picker = vtkCellPicker()
        picker.SetTolerance(0.0005)
        if picker.Pick(x, y, 0, self.plotter.renderer):
            faces = self._cell_faces.get(_actor_key(picker.GetActor()))
            if faces is not None and 0 <= picker.GetCellId() < len(faces):
                return int(faces[picker.GetCellId()])
        return None

    def close(self):
        self.plotter.close()


class CadImportEditor:
    def __init__(self, parent, host):
        self.host = host
        self.source = {}
        self.revision = ""
        self.geometry = None
        self.loading = False
        self._read = None  # the CadPreview last read; placement changes only move it
        self._preview = None  # what the viewport shows: the placed preview's geometry and triangles
        self._key = None
        self._placement = None  # of the shown preview, as keyword arguments of the pipeline functions
        self._hidden = set()
        self._highlight = None  # face ids of the closed shell being asked about
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
        self.viewport = Viewport(picked=self._viewport_picked, menu=self._show_face_menu)
        splitter.addWidget(self.viewport)
        tree_panel = QWidget()
        tree_layout = QVBoxLayout(tree_panel)
        tree_layout.setContentsMargins(0, 0, 0, 0)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        tree_layout.addWidget(self.hint)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Body / face", "Size (mm)"])
        self.tree.headerItem().setToolTip(1, "Element size override for an air or exterior surface body.")
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self.tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tree.setAccessibleName("Bodies and faces")
        self.tree.itemChanged.connect(self._item_changed)
        self.tree.itemDoubleClicked.connect(self._tree_double_clicked)
        self.tree.itemSelectionChanged.connect(self._selection_changed)
        tree_layout.addWidget(self.tree)
        buttons = QHBoxLayout()
        self.show_all_button = QPushButton("Show all")
        self.show_all_button.clicked.connect(self.show_all)
        self.clear_button = QPushButton("Clear")
        self.clear_button.setToolTip("Show everything and clear the selection.")
        self.clear_button.clicked.connect(self.clear)
        self.guides_check = QCheckBox("Show axes and planes")
        self.guides_check.setToolTip("The Boundary Lab axes and the symmetry planes X=0 and Y=0.")
        self.guides_check.setChecked(True)
        self.guides_check.toggled.connect(lambda _: self._render())
        buttons.addWidget(self.show_all_button)
        buttons.addWidget(self.clear_button)
        buttons.addStretch()
        buttons.addWidget(self.guides_check)
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
        self.group_name.textChanged.connect(self._group_name_changed)
        self.group_button = QPushButton("New group")
        self.group_button.setToolTip("Put the selected faces into the named face group (new or existing).")
        self.group_button.clicked.connect(lambda: self.group_selection(self.group_name.text()))
        self.remove_button = QPushButton("Remove from group")
        self.remove_button.setToolTip("Move the selected faces back to walls.")
        self.remove_button.clicked.connect(lambda: self.remove_from_group(self.selected_faces()))
        self.delete_group_button = QPushButton("Delete group")
        self.delete_group_button.clicked.connect(self._delete_selected_group)
        name_row.addWidget(self.group_name, 1)
        name_row.addWidget(self.group_button)
        name_row.addWidget(self.remove_button)
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
        self.placement_box = QGroupBox("Placement")
        self.placement_box.setToolTip(
            "Rotate about the Boundary Lab X (horizontal), Y (vertical) and Z (on axis) axes, in that order, "
            "then translate."
        )
        placement_form = QFormLayout(self.placement_box)
        self.rotation_spins, self.translation_spins = [], []
        for label, key, spins, limit, suffix in (
            ("Rotate", "rotation_deg", self.rotation_spins, 360.0, "°"),
            ("Translate", "translation_mm", self.translation_spins, 100000.0, " mm"),
        ):
            row = QHBoxLayout()
            for axis in "XYZ":
                spin = QDoubleSpinBox()
                spin.setRange(-limit, limit)
                spin.setDecimals(3)
                spin.setSuffix(suffix)
                spin.setKeyboardTracking(False)
                spin.setAccessibleName(f"{label} {axis}")
                spin.valueChanged.connect(lambda _, k=key, s=spins: self.edited({k: [x.value() for x in s]}))
                row.addWidget(QLabel(axis))
                row.addWidget(spin, 1)
                spins.append(spin)
            placement_form.addRow(label, row)
        form.addRow(self.placement_box)
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
        self._selection_changed()
        self.host.services.subscribe(self.event)
        self._stamp = None  # (mtime, size) of the STEP file when last read
        QGuiApplication.instance().applicationStateChanged.connect(self._application_state_changed)

    # -- source -------------------------------------------------------------------------------

    def apply_source(self, source, revision):
        self.source, self.revision = dict(source), revision
        self.step_path.setText(str(source.get("step_path", "")))
        for key, spin in self.spins.items():
            blocked = spin.blockSignals(True)
            spin.setValue(float(source[key]))
            spin.blockSignals(blocked)
        for key, spins in (("rotation_deg", self.rotation_spins), ("translation_mm", self.translation_spins)):
            for spin, value in zip(spins, source.get(key, (0.0, 0.0, 0.0)), strict=True):
                blocked = spin.blockSignals(True)
                spin.setValue(float(value))
                spin.blockSignals(blocked)
        self._refresh_context()
        key = (str(source.get("step_path") or ""), json.dumps(source.get("closed_shells", {}), sort_keys=True))
        if key != self._key:
            self._load(key)
        elif self._placement_of(source) != self._placement:
            self._place()
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
        path = FileDialogService().open_file(self.widget, "Open STEP file", "STEP files (*.step *.stp)")
        if path is not None:
            self.set_step_path(str(path))

    # -- preview ------------------------------------------------------------------------------

    def _load(self, key):
        self._key = key
        self._cancel_preview.set()
        self._cancel_preview = cancel = threading.Event()
        self._read = None
        self._placement = self._placement_of(self.source)
        self._show_preview(None)
        path, closed_shells = key
        placement = self._placement
        # Taken before reading: an edit while the file loads changes it, so the next focus asks.
        self._stamp = _file_stamp(path) if path else None
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
                    **placement,
                    "closed_shells": json.loads(closed_shells),
                }
                run_worker(job, output, cancelled=cancel.is_set)
                preview = preview_from_json(json.loads((output / "preview.json").read_text(encoding="utf-8")))
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
        if isinstance(preview, CadScan):
            # Closed shells without a choice: ask now, then the answers read the file fully, once.
            try:
                preview = place_preview(preview, **self._placement)
            except ValueError:
                pass  # shown as read; the full read reports the placement
            view = _view(preview)
            stored = self.source.get("closed_shells", {})
            answers = {
                shell.name: self._ask_highlighted(view, shell)
                for shell in preview.closed_shells
                if shell.name not in stored
            }
            self.edited({"closed_shells": stored | answers})
            return
        self._read = preview
        self._place()

    @staticmethod
    def _placement_of(source):
        return dict(
            rotation_deg=tuple(source.get("rotation_deg", (0.0, 0.0, 0.0))),
            translation_m=tuple(v / 1000 for v in source.get("translation_mm", (0.0, 0.0, 0.0))),
        )

    def _place(self):
        """Show the read preview with the source's placement, moved without reading the STEP file again."""
        self._placement = self._placement_of(self.source)
        if self._read is None:
            if not self.loading and self._key is not None and self._key[0]:
                self._load(self._key)  # nothing read yet, e.g. the last read's placement was invalid
            return
        try:
            placed = place_preview(self._read, **self._placement)
        except ValueError as exc:
            self._show_preview(None)
            self.info.setText(str(exc))
            return
        self._show_preview(_view(placed) | {"geometry": asdict(placed.geometry)})

    def _application_state_changed(self, state):
        """Back in the application with a changed STEP file: offer to re-mesh, never regenerate unasked."""
        if state != Qt.ApplicationState.ApplicationActive or self._disposed or self.loading or self._key is None:
            return
        path = self._key[0]
        stamp = _file_stamp(path)
        if stamp is None or self._stamp is None or stamp == self._stamp:
            return
        self._stamp = stamp  # asked once per change
        if not self.ask_remesh(path):
            return
        self._load(self._key)
        try:
            self.host.generate(expected_revision=self.revision)
        except (ValueError, RuntimeError) as exc:
            self.status.setText(str(exc))

    def ask_remesh(self, path):
        """Ask whether to re-mesh after the STEP file ``path`` changed on disk."""
        answer = QMessageBox.question(
            self.widget, "STEP file changed", f"{Path(path).name} changed on disk. Re-mesh it now?"
        )
        return answer == QMessageBox.StandardButton.Yes

    def _ask_highlighted(self, view, shell):
        """Ask about closed ``shell`` of the scan ``view`` with its faces highlighted and framed; the view
        comes back after."""
        faces = set(view["face_ids"].tolist())
        self._highlight = set(shell.face_ids)
        self.viewport.show_faces(view, faces, highlight=self._highlight)
        camera = self.viewport.frame(view, self._highlight)
        try:
            return self.ask_closed_shell(shell.name)
        finally:
            self._highlight = None
            self.viewport.restore_camera(camera)
            self.viewport.show_faces(view, faces)

    def highlighted_faces(self):
        """The faces drawn highlighted (the rest see-through), empty when none are."""
        return set(self._highlight or ())

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
        self._render(reset_camera=True)
        if self.geometry is not None:
            self._drop_interface_faces()

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

    def _selection_changed(self):
        count = len(self.selected_faces())
        self.hint.setText(
            "Click a face to select it, Ctrl or Shift+click to add or remove one, right-click to show its group and act on it. "
            f"{count} face{'' if count == 1 else 's'} selected."
        )
        self._render_timer.start(0)

    def _show_face_menu(self, face_id, global_pos):
        if self._faces is None:
            return
        self.mark_face(face_id)
        menu = self.face_menu(face_id)
        try:
            menu.exec(global_pos)
        finally:
            menu.deleteLater()
            self._highlight = None
            self._render()

    def mark_face(self, face_id):
        """Highlight ``face_id``, name it and its face group in the status line and point at the group's row."""
        name, _ = self.group_of(face_id)
        self._highlight = {face_id}
        self._render()
        self.status.setText(f"Face {face_id} · Group '{name}'")
        row = next(r for r, (n, _, _) in enumerate(self.rows()) if n == name)
        self._syncing = True  # the row only points: the face selection stays
        self.groups_table.selectRow(row)
        self._syncing = False

    def face_menu(self, face_id):
        """The right-click menu of ``face_id``: an interface face renames its interface; any other face's menu
        acts on the selection, or on the face when nothing is selected."""
        menu = QMenu(self.widget)
        if self._faces is None:
            return menu
        interface = self._interface_of().get(face_id)
        if interface is not None:
            menu.addAction("Rename interface…", lambda: self._rename_interface_asked(interface))
            return menu
        faces = self.selected_faces() or {face_id}
        for name, _, kind in self.rows():
            if kind == "group":
                menu.addAction(f'Add to "{name}"', lambda n=name: self.add_to_group(n, faces))
        menu.addAction("New group…", lambda: self._add_to_new_group(faces))
        menu.addAction("Move to walls", lambda: self.remove_from_group(faces))
        return menu

    def _rename_interface_asked(self, name):
        new_name = self.ask_interface_name(name)
        if new_name is not None:
            self.rename_interface(name, new_name)

    def ask_interface_name(self, name):
        """Ask the new name of interface face group ``name``; ``None`` when cancelled."""
        new_name, ok = QInputDialog.getText(
            self.widget, "Rename interface", "Interface face group name (empty: as detected):", text=name
        )
        return new_name if ok else None

    def rename_interface(self, name, new_name):
        """Rename interface face group ``name``; an empty ``new_name`` gives back the detected name."""
        shown = self._interface_names()
        detected = next(d for d, s in shown.items() if s == name)
        new_name = new_name.strip() or detected
        if new_name == name:
            return
        if new_name in {WALLS, *shown.values(), *self.face_groups()}:
            self.status.setText(f"Face group name '{new_name}' is taken.")
            return
        names = {d: s for d, s in self.source.get("interface_names", {}).items() if d != detected}
        if new_name != detected:
            names[detected] = new_name
        sizes = dict(self.source.get("face_group_sizes_m", {}))
        if name in sizes:
            sizes[new_name] = sizes.pop(name)
        self.edited({"interface_names": names, "face_group_sizes_m": sizes})

    def _add_to_new_group(self, face_ids):
        name = self.ask_group_name()
        if name is not None:
            self.add_to_group(name, face_ids)

    def ask_group_name(self):
        """Ask the name of a new face group; ``None`` when cancelled."""
        name, ok = QInputDialog.getText(self.widget, "New face group", "Face group name:")
        return name if ok else None

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

    def _render(self, *, reset_camera=False):
        colours = self.group_colours()
        face_colours = {face: colours[name] for name, faces, _ in self.rows() for face in faces}
        self.viewport.show_faces(
            self._preview,
            self.visible_faces(),
            self.selected_faces(),
            face_colours,
            highlight=self._highlight,
            guides=self.geometry["bbox_m"] if self.geometry is not None and self.guides_check.isChecked() else None,
            reset_camera=reset_camera,
        )

    def group_colours(self):
        """``{row name: colour}``: walls the plain face colour, the others each their own, stable by name."""
        colours, free = {WALLS: FACE_COLOUR}, list(GROUP_COLOURS)
        for name in sorted(name for name, _, kind in self.rows() if kind != "walls"):
            start = zlib.crc32(name.encode()) % len(GROUP_COLOURS)
            colour = next((c for c in GROUP_COLOURS[start:] + GROUP_COLOURS[:start] if c in free), None)
            if colour is None:  # more groups than colours: share
                colour = GROUP_COLOURS[start]
            else:
                free.remove(colour)
            colours[name] = colour
        return colours

    # -- face groups --------------------------------------------------------------------------

    def _ref(self, face_id):
        return asdict(face_ref(self._faces[face_id], **self._placement))

    def _members(self, group):
        """Face ids of a stored group in the loaded geometry, and its refs that match nothing."""
        ids, unmatched = set(), []
        for ref in group["faces"]:
            found = match_face_ref(CadFaceRef(**ref), self._faces.values(), self.geometry["bbox_m"], **self._placement)
            if found is None:
                unmatched.append(ref)
            else:
                ids.add(found)
        return ids, unmatched

    def _interface_names(self):
        """``{detected name: shown name}`` of the interfaces."""
        names = self.source.get("interface_names", {})
        return {i["name"]: names.get(i["name"], i["name"]) for i in self.geometry["interfaces"]}

    def _interface_of(self):
        shown = self._interface_names()
        return {f: shown[i["name"]] for i in self.geometry["interfaces"] for f in i["face_ids"]}

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

    def groups_needing_review(self):
        """Stored face groups with faces the loaded STEP file no longer has; Generate rejects them."""
        if self._faces is None:
            return []
        return [g["name"] for g in self.source.get("face_groups", []) if self._members(g)[1]]

    def rows(self):
        """``[(name, face ids, kind)]`` as listed: walls, the face groups, then the interface groups."""
        if self._faces is None:
            return []
        groups = self.face_groups()
        shown = self._interface_names()
        interfaces = {shown[i["name"]]: set(i["face_ids"]) for i in self.geometry["interfaces"]}
        walls = set(self._faces) - set().union(*groups.values(), *interfaces.values())
        return (
            [(WALLS, walls, "walls")]
            + [(name, faces, "group") for name, faces in groups.items()]
            + [(name, faces, "interface") for name, faces in interfaces.items()]
        )

    def group_selection(self, name):
        """Put the selected faces into face group ``name``."""
        self.add_to_group(name, self.selected_faces())

    def add_to_group(self, name, face_ids):
        """Put ``face_ids`` into face group ``name`` (new or existing), out of any other; walls removes them.
        Interface faces stay in their interface's face group."""
        name, face_ids = name.strip(), set(face_ids)
        if not name or not face_ids or self._faces is None:
            self.status.setText("Select faces and enter a face group name first.")
            return
        if name == WALLS:
            self.remove_from_group(face_ids)
            return
        interface_of = self._interface_of()
        if name in interface_of.values():
            self.status.setText(f"Face group name '{name}' is reserved.")
            return
        skipped = face_ids & interface_of.keys()
        face_ids -= skipped
        if face_ids:
            stored = list(self.source.get("face_groups", []))
            if name not in [g["name"] for g in stored]:
                stored.append({"name": name, "faces": []})
            # Re-picking a group that needs review repairs it: its missing faces are dropped.
            self._edit_groups(self._regrouped(stored, face_ids, name))
        if skipped:
            self.status.setText(_in_interfaces(skipped, interface_of))

    def _drop_interface_faces(self):
        """Take the faces that are now interface faces out of the stored face groups, saying so."""
        stored = self.source.get("face_groups", [])
        interface_of = self._interface_of()
        moved = {f for group in stored for f in self._members(group)[0]} & interface_of.keys()
        if moved:
            self._edit_groups(self._regrouped(stored, moved))
            self.status.setText(_in_interfaces(moved, interface_of))

    def remove_from_group(self, face_ids):
        """Move ``face_ids`` out of their face groups, back to walls."""
        face_ids = set(face_ids)
        if not face_ids or self._faces is None:
            self.status.setText("Select faces first.")
            return
        self._edit_groups(self._regrouped(self.source.get("face_groups", []), face_ids))
        named = sorted(f for f in face_ids if self.group_of(f)[1] == "group")
        if named:
            self.status.setText(f"Faces {', '.join(map(str, named))} stay in the face groups their STEP names form.")

    def _regrouped(self, stored, face_ids, name=None):
        """``stored`` groups with ``face_ids`` moved into group ``name`` (``None``: into none)."""
        groups = []
        for group in stored:
            ids, unmatched = self._members(group)
            if group["name"] == name:
                ids, unmatched = ids | face_ids, []
            else:
                ids -= face_ids
            if ids or unmatched:
                groups.append({"name": group["name"], "faces": unmatched + [self._ref(f) for f in sorted(ids)]})
        return groups

    def group_of(self, face_id):
        """``(name, kind)`` of the row holding ``face_id``."""
        return next((name, kind) for name, faces, kind in self.rows() if face_id in faces)

    def set_current_group(self, name):
        """Make ``name`` the group the selection is added to."""
        self.group_name.setText(name)

    def _group_name_changed(self, text):
        names = {name for name, _, kind in self.rows() if kind != "interface"}
        text = text.strip()
        self.group_button.setText(f'Add to "{text}"' if text in names else "New group")

    def delete_face_group(self, name):
        self._edit_groups([g for g in self.source.get("face_groups", []) if g["name"] != name])

    def _edit_groups(self, groups):
        """Store ``groups``; size overrides of face groups no longer listed are dropped."""
        listed = self.face_groups(groups).keys() | {g["name"] for g in groups} | {WALLS, *self._interface_of().values()}
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
        """A clicked row selects its faces; a single row other than an interface becomes the current group."""
        if self._syncing:
            return
        rows = self.rows()
        selected = sorted({item.row() for item in self.groups_table.selectedItems()})
        self.select_faces(set().union(*(rows[r][1] for r in selected)))
        if len(selected) == 1 and rows[selected[0]][2] != "interface":
            self.set_current_group(rows[selected[0]][0])

    def _show_groups(self):
        self._syncing = True
        stored = {g["name"] for g in self.source.get("face_groups", [])}
        sizes = self.source.get("face_group_sizes_m", {})
        rows = self.rows()
        review = set(self.groups_needing_review())
        colours = self.group_colours()
        tips = {
            "walls": "Faces in no other face group",
            "interface": "Interface face group: sizable; right-click one of its faces to rename it",
        }
        self.groups_table.setRowCount(len(rows))
        for row, (name, faces, kind) in enumerate(rows):
            count = f"{len(faces)} (needs review)" if name in review else str(len(faces))
            cells = [QTableWidgetItem(name), QTableWidgetItem(count), QTableWidgetItem(_mm(sizes.get(name)))]
            for cell in cells[:2]:
                cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
            cells[0].setToolTip(tips.get(kind) or ("Picked face group" if name in stored else "Named in the STEP file"))
            cells[0].setData(Qt.ItemDataRole.DecorationRole, QColor(colours[name]))
            if name in review:
                cells[1].setToolTip("Some faces of this group are not in the STEP file: re-pick or delete the group.")
            cells[2].setToolTip("Element size override in mm; empty uses the body or target size.")
            for column, cell in enumerate(cells):
                self.groups_table.setItem(row, column, cell)
        self._syncing = False
        self._group_name_changed(self.group_name.text())
        self._render_timer.start(0)  # group colours

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
            self.placement_box,
            self.groups_box,
            *self.spins.values(),
            self.closed_shell_box,
        )
        for widget in widgets:
            widget.setEnabled(not state.active)

    def dispose(self):
        self._disposed = True
        QGuiApplication.instance().applicationStateChanged.disconnect(self._application_state_changed)
        self._cancel_preview.set()
        self._render_timer.stop()
        self.viewport.close()


def _view(preview):
    """What the viewport draws of a scan or preview."""
    return {"points": preview.points, "triangles": preview.triangles, "face_ids": preview.face_ids}


def _file_stamp(path):
    try:
        stat = Path(path).stat()
    except (OSError, ValueError):
        return None
    return stat.st_mtime_ns, stat.st_size


def _in_interfaces(face_ids, interface_of):
    return "; ".join(f"Face {f} stays in interface '{interface_of[f]}'" for f in sorted(face_ids)) + "."


def _mm(size_m):
    return "" if size_m is None else f"{size_m * 1000:g}"


def _with_size(sizes, name, size_m):
    sizes = {k: v for k, v in sizes.items() if k != name}
    return sizes if size_m is None else sizes | {name: size_m}


def create_editor(parent, document_host):
    return CadImportEditor(parent, document_host)
