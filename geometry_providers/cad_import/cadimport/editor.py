"""3D view with face picking, a body/face visibility tree, face groups, placement and sizing.

The host's Generate meshes.
"""

import json
import shutil
import tempfile
import threading
import zlib
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QPalette
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
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from blab.cad_import import (
    BEM_LEVELS,
    FEM_LEVELS,
    LEVELS,
    MIXED,
    WALLS,
    CadDesign,
    CadFaceGroups,
    CadSizeOverride,
    place_preview,
)
from blab.cad_import.design import BEM_FINENESS, FEM_FINENESS, source_placement
from blab.cad_import.sizing import shared
from blab.generators.configuration import DEFAULT_SOUND_SPEED_M_PER_S
from blab.ui.file_dialogs import FileDialogService

from .jobs import preview_in_worker

SYMMETRY_LABELS = {"off": "Off", "x": "X", "xy": "XY"}
FACE_ROLE = Qt.ItemDataRole.UserRole
ROW_ROLE = Qt.ItemDataRole.UserRole + 1  # ("body" | "volume" | "faces", body name) of the rows above the faces
CUSTOM = "custom"  # the Mesh size choice that asks an edge length
SIZE_CHOICES = (*LEVELS.items(), ("Custom…", CUSTOM))  # after Global, which clears
SIZING = (
    ("curvature_elements_per_circle", "Curvature elements per circle", 3.0, 100.0, 0, ""),
    ("growth_rate", "Growth rate", 1.05, 3.0, 2, "×"),
)
FACE_COLOUR, OUTLINE_COLOUR = "#c8ccd2", "#2b2f36"
# The selection is orange; the face under the mouse pale orange, or dark orange when selected.
SELECTED_COLOUR, HOVER_COLOUR, HOVER_SELECTED_COLOUR = "#e8a33d", "#f6cf98", "#b8620f"
CONTEXT_COLOUR, CONTEXT_OPACITY = "#a0a4aa", 0.25  # the faces around a shown one: see-through grey
UNMESHABLE_COLOUR, UNMESHABLE_OPACITY = "#e53935", 0.5  # the fill triangles of faces Gmsh cannot mesh
AXIS_COLOURS = ("#d62728", "#2ca02c", "#1f77b4")  # X, Y, Z
AXIS_LABELS = ("X · Horizontal", "Y · Vertical", "Z · On Axis")  # as the balloon and mesh views name them
PLANE_COLOUR, PLANE_OPACITY = "#98df8a", 0.15  # the symmetry planes X=0 and Y=0, a lighter green than the Y axis
WARNING_COLOURS = ("#c62828", "#ff6b6b")  # on a light, a dark background
GROUP_COLOURS = (
    "#4e79a7", "#59a14f", "#e15759", "#76b7b2", "#b07aa1",
    "#9c755f", "#edc948", "#ff9da7", "#2f6f4f", "#8c6bb1",
)  # fmt: skip
MAX_ROWS = 8  # rows a list or table sized to its rows shows before it scrolls


class _Relay(QObject):
    loaded = Signal(object, object, object)  # key, preview or None, error or None


class _WarningLabel(QLabel):
    """Red text: dark red on a light background, light red on a dark one, following theme changes."""

    def __init__(self):
        super().__init__()
        self.setWordWrap(True)
        self._recolour()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:
            self._recolour()

    def _recolour(self):
        palette = self.palette()
        red = QColor(WARNING_COLOURS[palette.color(QPalette.ColorRole.Window).lightness() < 128])
        if palette.color(QPalette.ColorRole.WindowText) != red:  # else setting it changes the palette again
            palette.setColor(QPalette.ColorRole.WindowText, red)
            self.setPalette(palette)


def _capped(hint, rows, row_height):
    """A view's ``hint`` for all its ``rows`` capped at MAX_ROWS rows."""
    hint.setHeight(hint.height() - max(0, rows - MAX_ROWS) * row_height)
    return hint


class _UnmeshableList(QListWidget):
    def viewportSizeHint(self):
        return _capped(super().viewportSizeHint(), self.count(), self.sizeHintForRow(0))


class _GroupsTable(QTableWidget):
    def viewportSizeHint(self):
        return _capped(super().viewportSizeHint(), self.rowCount(), self.rowHeight(0))


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
    """Visible faces of the preview tessellation in a VTK plotter; a click picks a face, a right-click asks a menu,
    and the face under the mouse is reported as it changes.

    Clicks come from the plotter widget's Qt mouse events: the camera interaction holds VTK's mouse focus
    between press and release, so VTK observers never see the release.
    """

    def __init__(
        self,
        parent=None,
        *,
        picked=lambda face_id, add: None,
        menu=lambda face_id, global_pos: None,
        hovered=lambda face_id: None,
    ):
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
        self.hovered = hovered
        self._cell_faces = {}  # actor -> face id per cell
        self._press = None  # button and position of the last press
        self._hover = None  # the face under the mouse
        self.plotter.setMouseTracking(True)
        self.plotter.installEventFilter(self)

    def show_faces(
        self,
        preview,
        visible,
        selected=frozenset(),
        colours=None,
        *,
        context=frozenset(),
        unmeshable=frozenset(),
        outlines=(),
        hovered=None,
        guides=None,
        reset_camera=False,
    ):
        """Draw the ``visible`` faces smooth with their outlines, in their ``colours`` (face id -> colour, default
        plain), ``selected`` in orange and ``hovered`` (a face id) pale or, when selected, dark orange. ``context``
        faces are see-through, grey unless selected or hovered, and stay pickable. ``unmeshable`` faces are always
        see-through red. ``outlines`` (sampled edges, m, of selected unmeshable faces) are drawn in orange.
        ``guides`` (a bounding box, m) draws the axes and symmetry planes around it."""
        import pyvista as pv

        self.plotter.clear()
        self._cell_faces = {}
        if preview is not None:
            by_look = {}
            for face in visible:
                opacity = CONTEXT_OPACITY if face in context else 1.0
                if face in unmeshable:
                    colour, opacity = UNMESHABLE_COLOUR, UNMESHABLE_OPACITY
                elif face == hovered:
                    colour = HOVER_SELECTED_COLOUR if face in selected else HOVER_COLOUR
                elif face in selected:
                    colour = SELECTED_COLOUR
                elif face in context:
                    colour = CONTEXT_COLOUR
                else:
                    colour = (colours or {}).get(face, FACE_COLOUR)
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
            mask = np.isin(preview["face_ids"], list(visible))
            edges = outline_edges(preview["triangles"][mask], preview["face_ids"][mask])
            if len(edges):
                lines = np.hstack([np.full((len(edges), 1), 2), edges]).ravel()
                self.plotter.add_mesh(
                    pv.PolyData(preview["points"], lines=lines),
                    color=OUTLINE_COLOUR,
                    line_width=1.5,
                    pickable=False,
                    reset_camera=False,
                )
            for edges in outlines:  # (edge, point, xyz): a polyline per edge
                n, k = edges.shape[:2]
                lines = np.hstack([np.full((n, 1), k), np.arange(n * k).reshape(n, k)]).ravel()
                self.plotter.add_mesh(
                    pv.PolyData(edges.reshape(-1, 3), lines=lines),
                    color=SELECTED_COLOUR,
                    line_width=4,
                    pickable=False,
                    reset_camera=False,
                )
            if guides is not None:
                self._show_guides(guides)
            if reset_camera:
                self.plotter.view_xy()  # along -Z: Z towards the viewer, Y up, X to the right
        self.plotter.render()

    def frame(self, points):
        """Point the camera at ``points`` (m) so that they fill the view."""
        points = np.reshape(points, (-1, 3))
        self.plotter.reset_camera(bounds=np.column_stack([points.min(axis=0), points.max(axis=0)]).ravel())
        self.plotter.render()

    def _show_guides(self, bbox_m):
        """Labelled axes through the origin and the see-through symmetry planes X=0 and Y=0. The planes reach past
        the model and the origin by a fifth of their span (at least a tenth of the model's size), the axes by at
        least half their span, so the labels at their tips sit clear of the model."""
        import pyvista as pv

        low, high = np.minimum(bbox_m[0], 0), np.maximum(bbox_m[1], 0)
        margin = np.maximum(0.2 * (high - low), 0.1 * (float(np.max(np.subtract(bbox_m[1], bbox_m[0]))) or 1.0))
        axis_margin = np.maximum(margin, 0.5 * (high - low))
        for axis, name in enumerate("XYZ"):
            start, tip = np.zeros(3), np.zeros(3)
            start[axis], tip[axis] = low[axis] - axis_margin[axis], high[axis] + axis_margin[axis]
            self.plotter.add_mesh(
                pv.Line(start, tip),
                color=AXIS_COLOURS[axis],
                line_width=3,
                name=f"axis {name}",
                pickable=False,
                reset_camera=False,
            )
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
            corners[:, i] = low[i] - margin[i], high[i] + margin[i], high[i] + margin[i], low[i] - margin[i]
            corners[:, j] = low[j] - margin[j], low[j] - margin[j], high[j] + margin[j], high[j] + margin[j]
            self.plotter.add_mesh(
                pv.PolyData(corners, [4, 0, 1, 2, 3]),
                color=PLANE_COLOUR,
                opacity=PLANE_OPACITY,
                name=f"plane {name}=0",
                pickable=False,
                reset_camera=False,
            )

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
        elif kind == QEvent.Type.MouseMove and event.buttons() == Qt.MouseButton.NoButton:
            self._hover_at(self.face_at(*self._display(event.position())))
        elif kind == QEvent.Type.Leave:
            self._hover_at(None)
        return False  # the camera interaction sees every event

    def _hover_at(self, face):
        if face != self._hover:
            self._hover = face
            self.hovered(face)

    def _display(self, position):
        """VTK display coordinates (pixels, y up) of widget ``position``."""
        ratio = self.plotter.devicePixelRatioF()
        return round(position.x() * ratio), round(self.plotter.height() * ratio) - round(position.y() * ratio) - 1

    def _clicked(self, event):
        face = self.face_at(*self._display(event.position()))
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
        self._context = set()  # the faces greyed around a shown unmeshable face
        self._hover = None  # the face under the mouse
        self._face_items = {}
        self._body_items = {}
        self._geometry = None  # the CadGeometry shown
        self._design = CadDesign({})  # the source on that geometry
        self._groups = self._sizes = None  # its CadFaceGroups and CadSizes
        self._size_error = ""  # why the source's sizes cannot be read
        self._interface_note = ""  # last shown: which stored faces stay in their interface face groups
        self._syncing = False
        self._fmax_hz = None
        self._sound_speed = DEFAULT_SOUND_SPEED_M_PER_S  # the project's, once its context arrives
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

        # The view on top of the splitter; below it what explains and changes what it shows, then the tree.
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.viewport = Viewport(picked=self._viewport_picked, menu=self._show_face_menu, hovered=self._hovered)
        splitter.addWidget(self.viewport)
        tree_panel = QWidget()
        tree_layout = QVBoxLayout(tree_panel)
        tree_layout.setContentsMargins(0, 0, 0, 0)
        self.warnings = _WarningLabel()
        self.warnings.setAccessibleName("Warnings")
        self.warnings.hide()
        tree_layout.addWidget(self.warnings)
        self.unmeshable_box = QGroupBox("Unmeshable faces")
        self.unmeshable_box.setToolTip("Faces Gmsh cannot mesh: fix them in the CAD tool, then read the file again.")
        unmeshable_layout = QVBoxLayout(self.unmeshable_box)
        self.unmeshable_list = _UnmeshableList()
        self.unmeshable_list.setSizeAdjustPolicy(QAbstractItemView.SizeAdjustPolicy.AdjustToContents)
        # Its rows, up to MAX_ROWS; the tree gets the rest.
        self.unmeshable_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.unmeshable_list.setAccessibleName("Unmeshable faces")
        self.unmeshable_list.setToolTip("Click a face to show it in context, select it and frame it.")
        self.unmeshable_list.itemClicked.connect(lambda item: self.show_unmeshable(item.data(FACE_ROLE)))
        unmeshable_layout.addWidget(self.unmeshable_list)
        self.unmeshable_box.hide()
        tree_layout.addWidget(self.unmeshable_box)
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
        tree_layout.addWidget(self.placement_box)
        self.tree_box = QGroupBox("Bodies and faces")
        tree_box_layout = QVBoxLayout(self.tree_box)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        tree_box_layout.addWidget(self.hint)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Body / face", "Mesh size", "Kind"])
        self.tree.headerItem().setToolTip(
            1,
            "Mesh size of a face or volume; Global follows the mesh fineness. A body or Faces row sets it on its "
            "members and shows their shared size or Mixed.",
        )
        self.tree.headerItem().setToolTip(2, "Air body or exterior surface; a closed shell of surfaces can be either.")
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.kind_combos = {}  # closed shell name -> its Kind dropdown
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self.tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tree.setAccessibleName("Bodies and faces")
        self.tree.itemChanged.connect(self._item_changed)
        self.tree.itemClicked.connect(self._tree_clicked)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(
            lambda pos: self.show_row_menu(self.tree.itemAt(pos), self.tree.viewport().mapToGlobal(pos))
        )
        self._size_rows = []  # (Mesh size dropdown, face ids, air-body volumes) of each tree row
        self.tree.itemSelectionChanged.connect(self._selection_changed)
        tree_box_layout.addWidget(self.tree)
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
        tree_box_layout.addLayout(buttons)
        tree_layout.addWidget(self.tree_box)
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
        self.groups_table = _GroupsTable(0, 3)
        self.groups_table.setHorizontalHeaderLabels(["Face group", "Faces", "Mesh size"])
        self.groups_table.verticalHeader().hide()
        groups_header = self.groups_table.horizontalHeader()
        groups_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        groups_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        groups_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.groups_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.groups_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.groups_table.setSizeAdjustPolicy(QAbstractItemView.SizeAdjustPolicy.AdjustToContents)
        # Its rows, up to MAX_ROWS.
        self.groups_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.groups_table.setAccessibleName("Face groups")
        self.groups_table.itemSelectionChanged.connect(self._group_selected)
        self.groups_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.groups_table.customContextMenuRequested.connect(self._group_menu_requested)
        groups_layout.addWidget(self.groups_table)
        layout.addWidget(self.groups_box)

        self.sizing_box = QGroupBox("Mesh sizes")
        form = QFormLayout(self.sizing_box)
        self._custom_fineness = set()  # source keys whose Custom was chosen: it stays while the number is a level's
        self.spins = {}
        self._fineness_combos = {}  # source key -> (dropdown, its levels)
        self.fineness = self._fineness_row(form, "FEM", FEM_FINENESS, FEM_LEVELS, (1.0, 50.0))
        self.bem_fineness = self._fineness_row(form, "BEM", BEM_FINENESS, BEM_LEVELS, (0.1, 50.0))
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
        layout.addWidget(self.sizing_box)

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
        self._resolve()
        self.step_path.setText(str(source.get("step_path", "")))
        for key, spin in self.spins.items():
            blocked = spin.blockSignals(True)
            spin.setValue(self._design.number(key))
            spin.blockSignals(blocked)
        for spins, values in (
            (self.rotation_spins, self._design.rotation_deg),
            (self.translation_spins, self._design.translation_mm),
        ):
            for spin, value in zip(spins, values, strict=True):
                blocked = spin.blockSignals(True)
                spin.setValue(float(value))
                spin.blockSignals(blocked)
        self._show_fineness()
        self._refresh_context()
        key = (str(source.get("step_path") or ""), json.dumps(source.get("closed_shells", {}), sort_keys=True))
        if key != self._key:
            self._load(key)
        elif source_placement(source) != self._placement:
            self._place()
        else:
            self._show_overrides()
            self._migrate_sizes()

    def _fineness_row(self, form, method, key, levels, limits):
        """A ``method`` (FEM or BEM) mesh fineness dropdown of ``levels`` and Custom with a number field, a row of
        ``form`` writing source ``key``."""
        combo = QComboBox()
        for name, n in levels.items():
            combo.addItem(name, n)
        combo.addItem("Custom", None)
        combo.setAccessibleName(f"{method} mesh fineness")
        combo.activated.connect(lambda index: self._fineness_chosen(key, combo.itemData(index)))
        custom = QDoubleSpinBox()
        custom.setRange(*limits)
        custom.setDecimals(1)
        custom.setSuffix(" per wavelength")
        custom.setKeyboardTracking(False)
        custom.setAccessibleName(f"{method} elements per wavelength")
        custom.valueChanged.connect(lambda value: self.edited({key: value}))
        row = QHBoxLayout()
        row.addWidget(combo)
        row.addWidget(custom, 1)
        form.addRow(f"{method} mesh fineness", row)
        self.spins[key] = custom
        self._fineness_combos[key] = (combo, levels)
        return combo

    def _show_fineness(self):
        """The level of each source elements per wavelength, else (or while chosen) Custom with its number."""
        for key, (combo, levels) in self._fineness_combos.items():
            n = self._design.number(key)
            custom = key in self._custom_fineness or n not in levels.values()
            combo.setCurrentIndex(combo.count() - 1 if custom else combo.findData(n))
            self.spins[key].setHidden(not custom)

    def _fineness_chosen(self, key, n):
        self._custom_fineness.discard(key)
        if n is None:
            self._custom_fineness.add(key)
            self._show_fineness()
        else:
            self.edited({key: float(n)})

    def edited(self, changes, drop=()):
        """Store ``changes`` in the source, without the keys in ``drop``."""
        source = {k: v for k, v in self.source.items() if k not in drop} | changes
        try:
            snapshot = self.host.update_source(source, expected_revision=self.revision)
        except (ValueError, RuntimeError) as exc:
            self.status.setText(str(exc))
            return
        self.apply_source(snapshot.source, snapshot.revision)
        self.status.setText("Settings changed. Use Generate to update the meshes.")

    def set_step_path(self, path):
        if path != self.source.get("step_path"):
            # Closed shell choices belong to a file; another file starts from the guesses.
            self.edited({"step_path": path, "closed_shells": {}})

    def _browse(self):
        path = FileDialogService().open_file(self.widget, "Open STEP file", "STEP files (*.step *.stp)")
        if path is not None:
            self.set_step_path(str(path))

    # -- preview ------------------------------------------------------------------------------

    def _load(self, key):
        """Read ``key``'s STEP file; the shown preview, tree and controls stay until it arrives, and with the
        error when reading fails, so the change that caused it can be undone."""
        self._key = key
        self._cancel_preview.set()
        self._cancel_preview = cancel = threading.Event()
        path, closed_shells = key
        placement = source_placement(self.source)
        # Taken before reading: an edit while the file loads changes it, so the next focus asks.
        self._stamp = _file_stamp(path) if path else None
        if not path:
            self.loading = False
            self._read = None
            self._geometry = None
            self._show_preview(None)
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
                preview = preview_in_worker(
                    path, output, **placement, closed_shells=json.loads(closed_shells), cancelled=cancel.is_set
                )
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
        self._read = preview
        self._place(reset_camera=True)

    def _place(self, *, reset_camera=False):
        """Show the read preview with the source's placement, moved without reading the STEP file again."""
        if self._read is None:
            if not self.loading and self._key is not None and self._key[0]:
                self._load(self._key)  # nothing read yet, e.g. the last read failed
            return
        placement = source_placement(self.source)
        try:
            placed = place_preview(self._read, **placement)
        except ValueError as exc:
            self.info.setText(str(exc))
            return
        self._placement = placement
        self._geometry = placed.geometry
        self._show_preview(_view(placed) | {"geometry": asdict(placed.geometry)}, reset_camera=reset_camera)

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

    def _set_role(self, name, role):
        self.edited({"closed_shells": self.source.get("closed_shells", {}) | {name: role}})

    def _show_preview(self, preview, *, reset_camera=False):
        self._preview = preview
        self.geometry = None if preview is None else preview["geometry"]
        self._resolve()
        self._hidden.clear()
        self._context.clear()
        self._hover = None
        self._face_items = {}
        self._body_items = {}
        self._size_rows = []
        self.kind_combos = {}
        self._syncing = True
        self.tree.clear()
        self.unmeshable_list.clear()
        unmeshable = {} if self.geometry is None else {u["id"]: u["reason"] for u in self.geometry["unmeshable"]}
        self.unmeshable_box.setHidden(not unmeshable)
        if self.geometry is not None:
            faces = {face["id"]: face for face in self.geometry["faces"]}
            for face_id, reason in unmeshable.items():
                face = faces[face_id]
                centre = ", ".join(str(round(v * 1000)) for v in face["centre"])
                item = QListWidgetItem(
                    f"{_face_label(face)} · {face['surface_type']} · "
                    f"{' and '.join(face['bodies'])} · near ({centre}) mm · {reason}"
                )
                item.setData(FACE_ROLE, face_id)
                self.unmeshable_list.addItem(item)
            warning_icon = self.widget.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)
            roles = {shell["name"]: shell["role"] for shell in self.geometry["closed_shells"]}
            checkable = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable
            for body in self.geometry["bodies"]:
                name, air = body["name"], body["kind"] == "air"
                body_faces, volumes = set(body["face_ids"]), (name,) if air else ()
                item = QTreeWidgetItem([name])
                item.setFlags(checkable)  # not tristate: it follows its Faces row, the Volume row has no checkbox
                item.setCheckState(0, Qt.CheckState.Checked)
                item.setData(0, ROW_ROLE, ("body", name))
                item.setToolTip(0, "Air body" if air else "Exterior surface body")
                self.tree.addTopLevelItem(item)
                self._body_items[name] = item
                if name in roles:
                    self._add_kind_combo(item, name, roles[name])
                else:
                    item.setText(2, "Air body" if air else "Exterior surface")
                rows = [(item, body_faces, volumes)]
                if air:
                    volume = QTreeWidgetItem(["Volume"])
                    volume.setFlags(Qt.ItemFlag.ItemIsEnabled)  # not drawn: no checkbox
                    volume.setData(0, ROW_ROLE, ("volume", name))
                    volume.setToolTip(0, "The interior of the air body")
                    item.addChild(volume)
                    rows.append((volume, set(), volumes))
                faces_item = QTreeWidgetItem(["Faces"])
                faces_item.setFlags(checkable | Qt.ItemFlag.ItemIsAutoTristate)
                faces_item.setCheckState(0, Qt.CheckState.Checked)
                faces_item.setData(0, ROW_ROLE, ("faces", name))
                item.addChild(faces_item)
                rows.append((faces_item, body_faces, ()))
                for face_id in body["face_ids"]:
                    face = faces[face_id]
                    child = QTreeWidgetItem([_face_label(face)])
                    child.setFlags(checkable | Qt.ItemFlag.ItemIsSelectable)
                    child.setData(0, FACE_ROLE, face_id)
                    child.setCheckState(0, Qt.CheckState.Checked)
                    if face_id in unmeshable:
                        child.setIcon(0, warning_icon)
                        child.setToolTip(0, f"Cannot be meshed: {unmeshable[face_id]}")
                    faces_item.addChild(child)
                    self._face_items.setdefault(face_id, []).append(child)
                    rows.append((child, {face_id}, ()))
                for row, row_faces, row_volumes in rows:
                    combo = self._size_combo(row_faces, row_volumes)
                    self.tree.setItemWidget(row, 1, combo)
                    self._size_rows.append((combo, row_faces, row_volumes))
            low, high = self.geometry["bbox_m"]
            size = " x ".join(f"{(b - a) * 1000:.1f}" for a, b in zip(low, high, strict=True))
            self.info.setText(f"Size {size} mm · Symmetry {SYMMETRY_LABELS[self.geometry['symmetry']]} suggested")
            warnings = [f"Warning: {warning}" for warning in self.geometry["warnings"]]
            if unmeshable:
                count = len(unmeshable)
                warnings.append(f"{count} face{'' if count == 1 else 's'} cannot be meshed; see the list.")
            self.warnings.setText("\n".join(warnings))
        self.warnings.setHidden(self.geometry is None or not self.warnings.text())
        self._syncing = False
        self._show_overrides()
        self._render(reset_camera=reset_camera)
        if self.geometry is not None:
            notes = [self._size_error, self._drop_lost_sizes()]
            self._migrate_sizes()
            interface_note = self._groups.interface_note(self._groups.held_by_interfaces)
            if interface_note != self._interface_note:  # once per change, not on every read
                self._interface_note = interface_note
                notes.append(interface_note)
            if any(notes):
                self.status.setText(" ".join(note for note in notes if note))

    def _add_kind_combo(self, item, name, role):
        """The Kind dropdown of closed shell ``name``'s body row; a change re-reads the file."""
        combo = QComboBox()
        combo.addItem("Air body", "air")
        combo.addItem("Exterior surface", "exterior")
        combo.setCurrentIndex(combo.findData(role))
        combo.setAccessibleName(f"Kind of {name}")
        combo.setToolTip("A closed shell of surfaces: guessed until you choose.")
        combo.currentIndexChanged.connect(lambda _: self._set_role(name, combo.currentData()))
        self.tree.setItemWidget(item, 2, combo)
        self.kind_combos[name] = combo

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

    def _hovered(self, face_id):
        self._hover = face_id
        self._render_timer.start(0)

    def hovered_face(self):
        """The face under the mouse in the view, or ``None``."""
        return self._hover

    def _show_face_menu(self, face_id, global_pos):
        if self._groups is None:
            return
        if face_id not in self.selected_faces():
            self.select_faces({face_id})  # the menu never acts on faces not pointed at
        self.mark_face(face_id)
        self._render()
        _exec(self.face_menu(face_id), global_pos)

    def show_row_menu(self, item, global_pos):
        """The menu of a tree row: the face menu on a face or Faces row's faces, Mesh size on a Volume row; a body
        row has none."""
        if item is None or self._groups is None:
            return
        face_id = item.data(0, FACE_ROLE)
        kind, name = item.data(0, ROW_ROLE) or ("face", None)
        if face_id is not None:
            self._show_menu_on({face_id}, self._groups.interface_of.get(face_id), global_pos)
        elif kind == "faces":
            self._show_menu_on(set(self._body(name)["face_ids"]), None, global_pos)
        elif kind == "volume":
            menu = QMenu(self.widget)
            self._add_size_menu(menu, partial(self.set_volume_size, name))
            _exec(menu, global_pos)

    def show_group_menu(self, row, global_pos):
        """The face menu on the faces of face-group list ``row``."""
        name, faces, kind = self.rows()[row]
        if faces:
            self._show_menu_on(set(faces), name if kind == "interface" else None, global_pos)

    def _group_menu_requested(self, pos):
        row = self.groups_table.rowAt(pos.y())
        if row >= 0:
            self.show_group_menu(row, self.groups_table.viewport().mapToGlobal(pos))

    def _show_menu_on(self, face_ids, interface, global_pos):
        """The face menu on the selection, which ``face_ids`` become unless they are in it."""
        if not face_ids <= self.selected_faces():
            self.select_faces(face_ids)
        self._render()
        _exec(self._face_menu(self.selected_faces(), interface), global_pos)

    def mark_face(self, face_id):
        """Name ``face_id`` and its face group in the status line and point at the group's row."""
        name = self._groups.group_of(face_id).name
        self.status.setText(f"Face {face_id} · Group '{name}'")
        row = next(r for r, (n, _, _) in enumerate(self.rows()) if n == name)
        self._syncing = True  # the row only points: the face selection stays
        self.groups_table.selectRow(row)
        self._syncing = False

    def face_menu(self, face_id):
        """The right-click menu of ``face_id``, acting on the selection when the face is in it, else on the face.
        On an interface face it renames the interface and sizes faces."""
        if self._groups is None:
            return QMenu(self.widget)
        selected = self.selected_faces()
        return self._face_menu(selected if face_id in selected else {face_id}, self._groups.interface_of.get(face_id))

    def _face_menu(self, face_ids, interface):
        """The menu acting on ``face_ids``: moves between groups, leaving out those that would do nothing, or, for
        ``interface``, its renaming; and Mesh size."""
        menu = QMenu(self.widget)
        if interface is not None:
            menu.addAction("Rename interface…", lambda: self._rename_interface_asked(interface))
        elif movable := face_ids - self._groups.interface_of.keys():  # interface faces stay in their interface
            rows = self.rows()
            held = {name for name, faces, _ in rows if faces & movable}
            for name, _, kind in rows:
                if kind == "group" and held != {name}:
                    menu.addAction(f'Move to "{name}"', lambda n=name: self.add_to_group(n, face_ids))
            menu.addAction("New group…", lambda: self._add_to_new_group(face_ids))
            if held != {WALLS}:
                menu.addAction("Move to walls", lambda: self.remove_from_group(face_ids))
        self._add_size_menu(menu, partial(self.set_face_size, face_ids))
        return menu

    def _add_size_menu(self, menu, apply):
        """A Mesh size submenu of ``menu`` that ``apply``-s the chosen override (``None``: Global)."""
        sizes = menu.addMenu("Mesh size")
        for label, choice in ((self._sizing().text(None), None), *SIZE_CHOICES):
            sizes.addAction(label, lambda c=choice: self._size_chosen(c, apply))

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
        self._edit_groups(CadFaceGroups.renamed_interface, name, new_name)

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
        row = item.data(0, ROW_ROLE)
        if row is not None:  # a body row sets its Faces row, which sets its faces, and shows its state
            body = self._body_items[row[1]]
            faces_item = body.child(body.childCount() - 1)
            if row[0] == "faces":
                self._show_body_check(faces_item)
            elif row[0] == "body" and item.checkState(0) != Qt.CheckState.PartiallyChecked:
                faces_item.setCheckState(0, item.checkState(0))
            return
        visible = item.checkState(0) == Qt.CheckState.Checked
        face_id = item.data(0, FACE_ROLE)
        (self._hidden.discard if visible else self._hidden.add)(face_id)
        self._syncing = True
        for other in self._face_items[face_id]:
            if other is not item:
                other.setCheckState(0, item.checkState(0))
                self._show_body_check(other.parent())
        self._syncing = False
        self._render_timer.start(0)

    def _show_body_check(self, faces_item):
        syncing, self._syncing = self._syncing, True
        faces_item.parent().setCheckState(0, faces_item.checkState(0))
        self._syncing = syncing

    def _tree_clicked(self, item, column):
        """A body, Volume or Faces row selects all the body's faces (those around its volume)."""
        row = item.data(0, ROW_ROLE)
        if row is not None and column == 0:
            self.select_faces(self._body(row[1])["face_ids"])

    def _body(self, name):
        return next(body for body in self.geometry["bodies"] if body["name"] == name)

    def show_all(self):
        """Check every tree row, show every face and end the grey context."""
        self._syncing = True
        for items in self._face_items.values():
            for item in items:
                item.setCheckState(0, Qt.CheckState.Checked)
        for body in self._body_items.values():
            self._show_body_check(body.child(body.childCount() - 1))
        self._syncing = False
        self._hidden.clear()
        self._context.clear()
        self._render()

    def show_unmeshable(self, face_id):
        """Grey the visible faces outside the bodies containing unmeshable ``face_id``, select and mark it, and
        frame it."""
        bodies = set().union(*(body["face_ids"] for body in self.geometry["bodies"] if face_id in body["face_ids"]))
        self._context = self.visible_faces() - bodies
        self.select_faces({face_id})
        self.mark_face(face_id)
        self._render()
        preview = self._preview
        if len(preview["edges"][face_id]):
            self.viewport.frame(preview["edges"][face_id])
        elif (preview["face_ids"] == face_id).any():  # its fill; a face with neither leaves the camera
            self.viewport.frame(preview["points"][preview["triangles"][preview["face_ids"] == face_id]])

    def clear(self):
        """Show everything, end the grey context and clear the selection."""
        self.show_all()
        self.tree.clearSelection()

    def _render(self, *, reset_camera=False):
        colours = self.group_colours()
        face_colours = {face: colours[name] for name, faces, _ in self.rows() for face in faces}
        unmeshable = {u["id"] for u in self.geometry["unmeshable"]} if self.geometry is not None else set()
        outlined = self.visible_faces() & self.selected_faces() & unmeshable  # red or without triangles
        self.viewport.show_faces(
            self._preview,
            self.visible_faces(),
            self.selected_faces(),
            face_colours,
            context=self._context,
            unmeshable=unmeshable,
            outlines=[self._preview["edges"][face] for face in outlined if len(self._preview["edges"][face])],
            hovered=self._hover,
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

    def _resolve(self):
        """Resolve the source's face groups and sizes on the shown geometry, once per change of either."""
        self._design = CadDesign(self.source, self._geometry, placement=self._placement)
        self._groups, self._sizes, self._size_error = self._design.groups, self._design.sizes, self._design.size_error
        if self._size_error:  # as Generate rejects it; shown Global, never written over
            self.status.setText(self._size_error)

    def groups_needing_review(self):
        """Stored face groups with faces the loaded STEP file no longer has; Generate rejects them."""
        return [] if self._groups is None else list(self._groups.needs_review)

    def rows(self):
        """``[(name, face ids, kind)]`` as listed: walls, the interface groups, then the face groups."""
        return [] if self._groups is None else self._groups.rows

    def group_selection(self, name):
        """Put the selected faces into face group ``name``."""
        self.add_to_group(name, self.selected_faces())

    def add_to_group(self, name, face_ids):
        """Put ``face_ids`` into face group ``name`` (new or existing), out of any other; walls removes them.
        Interface faces stay in their interface's face group."""
        self._edit_groups(CadFaceGroups.moved_to, name, face_ids)

    def remove_from_group(self, face_ids):
        """Move ``face_ids`` out of their face groups, back to walls."""
        self._edit_groups(CadFaceGroups.moved_to_walls, face_ids)

    def set_current_group(self, name):
        """Make ``name`` the group the selection is added to."""
        self.group_name.setText(name)

    def _group_name_changed(self, text):
        names = {name for name, _, kind in self.rows() if kind != "interface"}
        text = text.strip()
        self.group_button.setText(f'Add to "{text}"' if text in names else "New group")

    def delete_face_group(self, name):
        self._edit_groups(CadFaceGroups.deleted, name)

    def _edit_groups(self, operation, *args):
        """Store what the face-group ``operation`` (a :class:`CadFaceGroups` edit) changes, with its note, or show
        why it cannot; sizes stay on the faces."""
        if self._groups is None:
            self.status.setText("Select faces first.")
            return
        try:
            edit = operation(self._groups, *args)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        if edit.changes:
            self.edited(edit.changes)
        if edit.note:
            self.status.setText(edit.note)

    def _delete_selected_group(self):
        rows = sorted({item.row() for item in self.groups_table.selectedItems()})
        for name in [self.groups_table.item(row, 0).text() for row in rows]:  # before a deletion redraws the list
            self.delete_face_group(name)

    # -- mesh sizes ---------------------------------------------------------------------------
    # Only faces and air-body volumes carry a size: {"elements_per_wavelength": N} (a level) or {"size_m": x};
    # None is Global (the mesh fineness). A body, Faces or face-group row writes to its members.

    def set_face_size(self, face_ids, size):
        """Set the size of each of ``face_ids``, stored like face group faces."""
        self._set_sizes(size, face_ids)

    def set_volume_size(self, name, size):
        """Set the size of air body ``name``'s volume (its interior)."""
        self._set_sizes(size, volumes=(name,))

    def set_face_group_size(self, name, size):
        """Set the size of the faces of face-group list row ``name``."""
        self._set_sizes(size, next(faces for n, faces, _ in self.rows() if n == name))

    def _set_sizes(self, size, face_ids=(), volumes=()):
        """Write ``size`` (an override or its stored fields; ``None``: Global) to ``face_ids`` and ``volumes``."""
        if self._sizes is None:
            return
        if self._size_error:
            self.status.setText(self._size_error)
            return
        try:
            changes = self._design.set_sizes(size, face_ids, volumes)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.edited(changes)

    def _drop_lost_sizes(self):
        """Drop the face sizes whose faces the loaded STEP file no longer has, and volume sizes of bodies that are no
        air bodies; the note saying so, else empty."""
        if self._size_error:
            return ""
        changes, note = self._design.dropped()
        if changes:
            self.edited(changes)
        return note

    def _migrate_sizes(self):
        """Convert an older design's body and face group sizes once its geometry and the maximum frequency, which
        compares levels with edge lengths, are known."""
        source = self._design.migrated(self._sizing())
        if source is not None:
            self.edited(source, drop=self.source)  # the whole new source
            self.status.setText("Body and face group sizes are now set on their faces and volumes.")

    def ask_size_mm(self):
        """Ask a custom edge length in mm; ``None`` when cancelled."""
        size, ok = QInputDialog.getDouble(self.widget, "Custom mesh size", "Edge length (mm):", 10.0, 0.001, 1e5, 3)
        return size if ok else None

    def _size_chosen(self, choice, apply):
        """``apply`` the size of a Mesh size ``choice``: ``None`` (Global), a level's N or Custom…, which asks the
        edge length; ``False`` when that is cancelled."""
        if choice == CUSTOM:
            size_mm = self.ask_size_mm()
            if size_mm is None:
                return False
            apply(CadSizeOverride(size_m=size_mm / 1000))
        else:
            apply(None if choice is None else CadSizeOverride(elements_per_wavelength=choice))
        return True

    def _size_combo(self, face_ids, volumes):
        """A Mesh size dropdown writing to ``face_ids`` and air-body ``volumes``."""
        combo = QComboBox()
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)  # Global is named once shown
        combo.setAccessibleName("Mesh size")
        combo.addItem("", None)  # Global, named when shown
        for label, choice in SIZE_CHOICES:
            combo.addItem(label, choice)
        apply = partial(self._set_sizes, face_ids=face_ids, volumes=volumes)

        def chosen(index):
            choice = combo.itemData(index)

            def apply_choice():
                if not self._size_chosen(choice, apply):
                    self._show_overrides()  # Custom… cancelled: back to the stored size

            # Once the dropdown has closed: the change rewrites the dropdowns, and Custom… asks first.
            QTimer.singleShot(0, apply_choice)

        combo.activated.connect(chosen)
        return combo

    def _show_size(self, combo, faces, volumes=()):
        """Show the size ``faces`` and air-body ``volumes`` share (``None``: Global, or :data:`MIXED`) in ``combo``;
        one no entry names gets an entry of its own, shown but never chosen. Global names the mesh fineness that
        sizes them, if one does."""
        size = self._sizes.shared(faces, volumes)
        blocked = combo.blockSignals(True)
        fixed = 1 + len(SIZE_CHOICES)
        while combo.count() > fixed:
            combo.removeItem(fixed)
        sizing = self._sizing()
        targets = self._design.targets(sizing, faces, volumes)
        combo.setItemText(0, sizing.global_text(targets))
        levels = list(LEVELS.values())
        if size is None:
            index = 0
        elif size != MIXED and size.elements_per_wavelength in levels:
            index = 1 + levels.index(size.elements_per_wavelength)
        else:
            combo.addItem(sizing.text(size))
            combo.model().item(fixed).setEnabled(False)
            index = fixed
        combo.setCurrentIndex(index)
        if size is None:
            level = shared(targets)
            combo.setToolTip(
                "Its members follow different mesh finenesses."
                if level == MIXED
                else f"The mesh fineness: {sizing.text(level, edge=True)}"
            )
        else:
            combo.setToolTip("Its members have different sizes." if size == MIXED else sizing.text(size, edge=True))
        combo.blockSignals(blocked)

    def _sizing(self):
        """The design's sizing at the project's maximum frequency and sound speed (once they are known)."""
        return self._design.sizing(self._fmax_hz, self._sound_speed)

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
        stored = self._design.stored_groups
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
            cells = [QTableWidgetItem(name), QTableWidgetItem(count)]
            for cell in cells:
                cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
            cells[0].setToolTip(tips.get(kind) or ("Picked face group" if name in stored else "Named in the STEP file"))
            cells[0].setData(Qt.ItemDataRole.DecorationRole, QColor(colours[name]))
            if name in review:
                cells[1].setToolTip("Some faces of this group are not in the STEP file: re-pick or delete the group.")
            for column, cell in enumerate(cells):
                self.groups_table.setItem(row, column, cell)
            combo = self._size_combo(faces, ())
            self._show_size(combo, faces)
            self.groups_table.setCellWidget(row, 2, combo)
        self._syncing = False
        self._group_name_changed(self.group_name.text())
        self._render_timer.start(0)  # group colours

    def _show_overrides(self):
        """Each row's size in its Mesh size dropdown: a face's or volume's own or Global, a body's, Faces' or face
        group's the one its members share or Mixed."""
        for combo, row_faces, row_volumes in self._size_rows:  # rows exist only with a geometry, and so sizes
            self._show_size(combo, row_faces, row_volumes)
        self._show_groups()

    # -- sizing and meshing -------------------------------------------------------------------

    def _refresh_context(self):
        self._show_target_size()
        future = self.host.services.context()

        def done(result):
            if not self._disposed and not result.cancelled() and result.exception() is None:
                context = result.result()
                self._fmax_hz, self._sound_speed = context.freq_max_hz, context.sound_speed_m_per_s
                self._show_target_size()
                self._show_overrides()  # levels' edge lengths in the tooltips
                self._migrate_sizes()

        future.add_done_callback(done)

    def _show_target_size(self):
        if self._fmax_hz is None:
            self.target_label.setText("Target element size: waiting for the project frequency range.")
            return
        sizing = self._sizing()
        bem = sizing.bem_fineness.elements_per_wavelength
        self.target_label.setText(
            f"Target element size FEM {sizing.target_size_m * 1000:.1f} mm = c / {self._fmax_hz:g} Hz / "
            f"{sizing.elements_per_wavelength:g}, BEM {sizing.bem_target_size_m * 1000:.1f} mm = c / "
            f"{self._fmax_hz:g} Hz / {bem:g} (c = {self._sound_speed:g} m/s)"
        )
        held = FEM_LEVELS["FEM Fine"]  # the elements per wavelength a level's frequency limit holds
        for key, (combo, levels) in self._fineness_combos.items():
            for index, level in enumerate(levels.values()):
                tip = (
                    f"{level:g} elements per wavelength at {self._fmax_hz:g} Hz "
                    f"({sizing.edge_m(CadSizeOverride(elements_per_wavelength=level)) * 1000:.3g} mm)"
                )
                if key == FEM_FINENESS:  # FEM: the frequency up to which the level still holds Fine
                    tip += f": {held} per wavelength up to {self._fmax_hz * level / held:.0f} Hz"
                combo.setItemData(index, tip, Qt.ItemDataRole.ToolTipRole)

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
            *self.kind_combos.values(),
            self.fineness,
            self.bem_fineness,
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
    """What the viewport draws of a scan or preview; ``edges`` are each face's sampled boundary curves (m)."""
    openings = preview.openings
    points = dict(zip(openings.edges, openings.edge_points, strict=True))
    edges = {
        face: np.array([points[c] for c in curves]).reshape(-1, 9, 3) for face, curves in openings.face_curves.items()
    }
    return {"points": preview.points, "triangles": preview.triangles, "face_ids": preview.face_ids, "edges": edges}


def _face_label(face):
    """A face's STEP name, else its number."""
    return face["step_name"] or f"Face {face['id']}"


def _file_stamp(path):
    try:
        stat = Path(path).stat()
    except (OSError, ValueError):
        return None
    return stat.st_mtime_ns, stat.st_size


def _exec(menu, global_pos):
    try:
        menu.exec(global_pos)
    finally:
        menu.deleteLater()


def create_editor(parent, document_host):
    return CadImportEditor(parent, document_host)
