"""PyVista preview widget for generated and imported meshes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import meshio
import numpy as np
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from blab.ath import read_surface_physical_names
from blab.config import MeshConfig
from blab.generators.base import GeneratedGeometry
from blab.preview_hierarchy import PreviewHierarchy, build_preview_hierarchy
from blab.ui.observation_plane_viewport import ObservationPlaneViewport
from blab.ui.theme import themed_content_background

try:  # pragma: no cover - optional visual dependency
    import pyvista as pv
    import vtk
    from pyvistaqt import QtInteractor
except ImportError:  # pragma: no cover
    pv = None
    vtk = None
    QtInteractor = None


AXIS_LINE_WIDTH = 1.5
AXIS_COLORS = ("#e25d5d", "#5da8e2", "#f2d15f")
AXIS_LABELS = ("Horizontal", "Vertical", "On Axis")
PREVIEW_HOME_CAMERA_DIRECTION = np.array([-1.0, 1.0, 1.0], dtype=float) / np.sqrt(3.0)
PREVIEW_HOME_VIEW_UP = np.array([0.0, 1.0, 0.0], dtype=float)
PREVIEW_HOME_ZOOM = 1.2
RIGID_COLOR = "#cfcfcf"
RIGID_MIRROR_COLOR = "#a9a9a9"
RIGID_EDGE_COLOR = "#555555"
RIGID_MIRROR_EDGE_COLOR = "#4a4a4a"
DRIVEN_COLOR = "#3292bf"
DRIVEN_MIRROR_COLOR = "#236787"
DRIVEN_EDGE_COLOR = "#20343c"
DRIVEN_MIRROR_EDGE_COLOR = "#1b2c33"
INTERFACE_COLOR = "#1cad0c"
INTERFACE_MIRROR_COLOR = "#116b07"
INTERFACE_EDGE_COLOR = "#155b0d"
INTERFACE_MIRROR_EDGE_COLOR = "#0b3506"
TOPOLOGY_ISSUE_COLOR = "#ff2020"
TOPOLOGY_ISSUE_LINE_WIDTH = 6.0
PREVIEW_REGION_INTERIOR = "interior"
PREVIEW_REGION_EXTERIOR = "exterior"
BODY_TREE_OVERLAY_WIDTH = 240
BODY_TREE_OVERLAY_MAX_HEIGHT = 420
BODY_TREE_OVERLAY_MARGIN = 5
BODY_TREE_PROJECT_NODE_ID = "project"

_TREE_SURFACE_KEYS_ROLE = int(Qt.ItemDataRole.UserRole)
_TREE_NODE_ID_ROLE = _TREE_SURFACE_KEYS_ROLE + 1


@dataclass
class _PreviewActorRecord:
    actor: object
    mesh_name: str
    surface_key: tuple[str, int | None] | None
    mesh_region: str | None
    diagnostic: bool = False


class MeshPreview(QWidget):
    newObservationPlaneRequested = Signal(object)
    observationPlaneChanged = Signal(object)
    observationPlanePropertiesRequested = Signal(str)
    observationPlaneDeleteRequested = Signal(str)
    observationPlaneExteriorFieldRequested = Signal(object)

    def __init__(self):
        super().__init__()
        self._hover_picker = None
        self._hover_observer = None
        self._actor_surface_labels: dict[str, str] = {}
        self._actor_records: list[_PreviewActorRecord] = []
        self._topology_issue_actors: list[object] = []
        self._surface_visibility: dict[tuple[str, int | None], bool] = {}
        self._hierarchy: PreviewHierarchy | None = None
        self._tree_updating = False
        self._observation_clip_active = False
        self._observation_editor = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if QtInteractor is None:
            self.viewer = None
            label = QLabel("Install pyvista and pyvistaqt to enable mesh preview.")
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
            return

        scene = QWidget(self)
        scene_layout = QGridLayout(scene)
        scene_layout.setContentsMargins(0, 0, 0, 0)

        self.viewer = QtInteractor(scene)
        scene_layout.addWidget(self.viewer, 0, 0)

        self.body_tree_overlay = QFrame(scene)
        self.body_tree_overlay.setObjectName("mesh_body_tree_overlay")
        self.body_tree_overlay.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.body_tree_overlay.setFixedWidth(BODY_TREE_OVERLAY_WIDTH)
        overlay_layout = QVBoxLayout(self.body_tree_overlay)
        overlay_layout.setContentsMargins(
            BODY_TREE_OVERLAY_MARGIN,
            BODY_TREE_OVERLAY_MARGIN,
            BODY_TREE_OVERLAY_MARGIN,
            BODY_TREE_OVERLAY_MARGIN,
        )
        self.body_tree = QTreeWidget()
        self.body_tree.setObjectName("mesh_body_tree")
        self.body_tree.setHeaderHidden(True)
        self.body_tree.setIndentation(15)
        self.body_tree.setUniformRowHeights(True)
        self.body_tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.body_tree.setAlternatingRowColors(False)
        self.body_tree.setStyleSheet(
            """
            QTreeWidget#mesh_body_tree {
                background: transparent;
                border: none;
                color: #f0f2f5;
                outline: none;
            }
            QTreeWidget#mesh_body_tree::item {
                background: transparent;
                min-height: 22px;
            }
            QTreeWidget#mesh_body_tree::item:hover {
                background: rgba(255, 255, 255, 24);
            }
            QTreeWidget#mesh_body_tree::item:selected {
                background: rgba(92, 132, 181, 145);
                color: white;
            }
            """
        )
        self.body_tree.viewport().setStyleSheet("background: transparent;")
        self.body_tree.itemChanged.connect(self._on_body_tree_item_changed)
        self.body_tree.itemExpanded.connect(self._resize_body_tree_overlay)
        self.body_tree.itemCollapsed.connect(self._resize_body_tree_overlay)
        overlay_layout.addWidget(self.body_tree)
        self.body_tree_overlay.setStyleSheet(
            """
            QFrame#mesh_body_tree_overlay {
                background-color: rgba(38, 44, 55, 120);
                border: 1px solid rgba(170, 180, 195, 90);
                border-radius: 2px;
            }
            """
        )
        scene_layout.addWidget(
            self.body_tree_overlay,
            0,
            0,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )
        self.body_tree_overlay.raise_()
        self._rebuild_body_tree()

        viewport = QWidget()
        viewport_layout = QVBoxLayout(viewport)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        viewport_layout.addWidget(scene)
        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        self.hover_label = QLabel("")
        self.hover_label.setMinimumHeight(22)
        self.hover_label.setMinimumWidth(0)
        self.hover_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.total_elements_label = QLabel("")
        self.total_elements_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.total_elements_label.setMinimumHeight(22)
        self.total_elements_label.setMinimumWidth(0)
        self.total_elements_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        status_row.addWidget(self.hover_label, 1)
        status_row.addWidget(self.total_elements_label)
        viewport_layout.addLayout(status_row)

        layout.addWidget(viewport)
        self._refresh_viewer_theme()
        self._observation_editor = ObservationPlaneViewport(self.viewer, vtk, self)
        self._observation_editor.newPlaneRequested.connect(self.newObservationPlaneRequested.emit)
        self._observation_editor.planeChanged.connect(self.observationPlaneChanged.emit)
        self._observation_editor.propertiesRequested.connect(self.observationPlanePropertiesRequested.emit)
        self._observation_editor.deleteRequested.connect(self.observationPlaneDeleteRequested.emit)
        self._observation_editor.clipStateChanged.connect(self._set_observation_clip_active)
        self._observation_editor.exteriorFieldRequested.connect(self.observationPlaneExteriorFieldRequested.emit)
        self._install_hover_picker()

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:
            self._refresh_viewer_theme()

    def _refresh_viewer_theme(self) -> None:
        viewer = getattr(self, "viewer", None)
        if viewer is not None:
            viewer.set_background(themed_content_background(self.palette()))
        observation_editor = getattr(self, "_observation_editor", None)
        if observation_editor is not None:
            observation_editor.refresh_theme()

    def clear(self) -> None:
        self._actor_records = []
        self._topology_issue_actors = []
        self._hierarchy = None
        self._surface_visibility = {}
        body_tree = getattr(self, "body_tree", None)
        if body_tree is not None:
            self._rebuild_body_tree()
        if self.viewer is None:
            return
        self.viewer.clear()
        self._actor_surface_labels = {}
        self.hover_label.setText("")
        self._set_total_element_count(0)
        self._restore_observation_plane_scene(np.empty((0, 3)))

    def set_observation_planes(self, planes, *, selected_id: str | None = None) -> None:
        if self._observation_editor is not None:
            self._observation_editor.set_planes(tuple(planes), selected_id=selected_id)

    def set_observation_plane_results(self, results) -> None:
        if self._observation_editor is not None:
            self._observation_editor.set_field_results(results)

    def set_observation_plane_field_preferences(self, *, cache_size_mb: object, translation_target_fps: object) -> None:
        if self._observation_editor is not None:
            self._observation_editor.set_field_preferences(
                cache_size_mb=cache_size_mb,
                translation_target_fps=translation_target_fps,
            )

    def set_observation_plane_animation(self, plane_id: str | None, enabled: bool) -> None:
        if self._observation_editor is not None:
            self._observation_editor.set_animation(plane_id, enabled)

    def set_observation_plane_active(self, plane_id: str | None) -> None:
        if self._observation_editor is not None:
            self._observation_editor.set_active_plane(plane_id)

    def set_observation_plane_exterior_field(self, key, pressure) -> None:
        if self._observation_editor is not None:
            self._observation_editor.set_exterior_field_result(key, pressure)

    def set_observation_plane_exterior_field_failed(self, key, message: str) -> None:
        if self._observation_editor is not None:
            self._observation_editor.set_exterior_field_failed(key, message)

    def discard_observation_plane_exterior_field_request(self, key) -> None:
        if self._observation_editor is not None:
            self._observation_editor.discard_exterior_field_request(key)

    def _set_observation_clip_active(self, active: bool) -> None:
        self._observation_clip_active = bool(active)
        self._apply_actor_visibility(render=False)

    def _restore_observation_plane_scene(self, points: np.ndarray) -> None:
        if self._observation_editor is None:
            return
        self._observation_editor.set_scene_bounds(points)
        self._observation_editor.scene_cleared()

    def load_generated_geometry(self, result: GeneratedGeometry) -> None:
        self.load_mesh_configs(
            (MeshConfig(name="waveguide", file=str(result.solver_mesh_path), scale_factor=0.001),),
            driven_surfaces={("waveguide", radiator.tag) for radiator in result.radiators},
            surface_tags_by_mesh={"waveguide": read_surface_physical_names(result.solver_mesh_path)},
        )

    def load_msh(
        self,
        msh_path: Path,
        driven_tags: set[int] | None = None,
        surface_tags: dict[str, int] | None = None,
    ) -> None:
        if self.viewer is None:
            return
        camera_position = self._camera_position()
        mesh = meshio.read(msh_path)
        triangles = _extract_triangles_for_preview(mesh)
        physical_tags = _extract_triangle_physical_tags_for_preview(mesh)
        self.viewer.clear()
        self._actor_surface_labels = {}
        self._actor_records = []
        self._topology_issue_actors = []
        self._hierarchy = None
        self._surface_visibility = {}
        self._rebuild_body_tree()
        self.hover_label.setText("")
        display_points = np.asarray(mesh.points, dtype=float)
        self._restore_observation_plane_scene(display_points)
        self._set_total_element_count(
            int(triangles.shape[0]),
            dimensions_mm=_dimensions_lwh_mm(display_points),
        )

        if physical_tags is None:
            actor = self.viewer.add_mesh(
                _triangles_to_polydata(mesh.points, triangles),
                color="#cfcfcf",
                show_edges=True,
                edge_color="#555555",
                smooth_shading=False,
            )
            self._actor_surface_labels[_vtk_actor_address(actor)] = _surface_hover_label(
                None,
                "untagged",
                None,
                int(triangles.shape[0]),
            )
            self._add_orientation_guides(display_points)
            self._restore_camera_or_reset(camera_position)
            return

        names_by_tag = {tag: name for name, tag in (surface_tags or {}).items()}
        for tag in sorted(np.unique(physical_tags)):
            tag_mask = physical_tags == tag
            tag_triangles = triangles[tag_mask]
            if not tag_triangles.size:
                continue

            is_driven = int(tag) in (driven_tags or set())
            actor = self.viewer.add_mesh(
                _triangles_to_polydata(mesh.points, tag_triangles),
                color=DRIVEN_COLOR if is_driven else RIGID_COLOR,
                show_edges=True,
                edge_color=DRIVEN_EDGE_COLOR if is_driven else RIGID_EDGE_COLOR,
                smooth_shading=False,
            )
            surface_name = names_by_tag.get(int(tag), f"Tag {int(tag)}")
            self._actor_surface_labels[_vtk_actor_address(actor)] = _surface_hover_label(
                None,
                surface_name,
                int(tag),
                int(tag_triangles.shape[0]),
            )

        self._add_orientation_guides(display_points)
        self._restore_camera_or_reset(camera_position)

    def load_mesh_configs(
        self,
        meshes: tuple[MeshConfig, ...],
        *,
        driven_surfaces: set[tuple[str, int]] | None = None,
        surface_tags_by_mesh: dict[str, dict[str, int]] | None = None,
        interface_surfaces: set[tuple[str, int]] | None = None,
        mesh_regions: dict[str, str] | None = None,
        symmetry: str = "off",
        topology_report=None,
        hierarchy: PreviewHierarchy | None = None,
    ) -> None:
        if self.viewer is None:
            return
        camera_position = self._camera_position()
        self.viewer.clear()
        self._actor_surface_labels = {}
        self._actor_records = []
        self._topology_issue_actors = []
        self.hover_label.setText("")
        total_elements = 0
        preview_points = []
        mirrored = str(symmetry or "off").strip().lower() != "off"

        for mesh_cfg in meshes:
            mesh_elements, mesh_points = self._add_msh_mesh(
                mesh_cfg,
                driven_surfaces=driven_surfaces or set(),
                interface_surfaces=interface_surfaces or set(),
                surface_tags=(surface_tags_by_mesh or {}).get(mesh_cfg.name, {}),
                mesh_region=(mesh_regions or {}).get(mesh_cfg.name),
                symmetry=symmetry,
            )
            total_elements += mesh_elements
            preview_points.append(mesh_points)

        display_points = np.vstack(preview_points) if preview_points else np.empty((0, 3))
        self._restore_observation_plane_scene(display_points)
        self._set_total_element_count(
            total_elements,
            mirrored=mirrored,
            dimensions_mm=_dimensions_lwh_mm(display_points),
        )
        if preview_points:
            self._add_orientation_guides(display_points)
        if hierarchy is None:
            identity_map = {
                record.surface_key: record.surface_key
                for record in self._actor_records
                if record.surface_key is not None
            }
            hierarchy = build_preview_hierarchy(
                None,
                source_mesh_configs=meshes,
                source_surface_tags_by_mesh=surface_tags_by_mesh or {},
                solver_surface_by_source=identity_map,
            )
        self.set_hierarchy(hierarchy, render=False)
        self.set_topology_report(topology_report, render=False)
        self._apply_actor_visibility(render=False)
        self._restore_camera_or_reset(camera_position)

    def set_topology_report(self, report, *, render: bool = True) -> None:
        """Replace the red invalid-edge overlay without rebuilding the mesh scene."""

        if self.viewer is None:
            return
        old_actor_ids = {id(actor) for actor in self._topology_issue_actors}
        self._actor_records = [
            record for record in self._actor_records if id(record.actor) not in old_actor_ids
        ]
        for actor in self._topology_issue_actors:
            self.viewer.remove_actor(actor, render=False)
        self._topology_issue_actors = []

        if report is not None:
            for mesh in report.meshes:
                source_segments = mesh.problem_edge_segments_m
                if not len(source_segments):
                    continue
                segments = _line_segments_with_symmetry_images(
                    source_segments,
                    report.symmetry,
                )
                actor = self.viewer.add_mesh(
                    _line_segments_to_polydata(segments),
                    color=TOPOLOGY_ISSUE_COLOR,
                    line_width=TOPOLOGY_ISSUE_LINE_WIDTH,
                    render_lines_as_tubes=True,
                    lighting=False,
                    pickable=False,
                )
                self._topology_issue_actors.append(actor)
                self._register_mesh_actor(
                    actor,
                    PREVIEW_REGION_EXTERIOR,
                    mesh_name=mesh.mesh_name,
                    surface_tag=None,
                    diagnostic=True,
                )
        self._apply_actor_visibility(render=render)

    def set_hierarchy(self, hierarchy: PreviewHierarchy, *, render: bool = True) -> None:
        """Populate the body tree while preserving visibility for stable surface IDs."""

        old_visibility = self._surface_visibility
        self._hierarchy = hierarchy
        self._surface_visibility = {
            surface_key: old_visibility.get(surface_key, True) for surface_key in hierarchy.surface_keys
        }
        self._rebuild_body_tree()
        self._apply_actor_visibility(render=render)

    def _rebuild_body_tree(self) -> None:
        if not hasattr(self, "body_tree"):
            return
        had_hierarchy = any(
            str(item.data(0, _TREE_NODE_ID_ROLE)) != BODY_TREE_PROJECT_NODE_ID
            for item in _tree_items(self.body_tree)
        )
        expanded_ids = set()
        for item in _tree_items(self.body_tree):
            if item.isExpanded():
                expanded_ids.add(str(item.data(0, _TREE_NODE_ID_ROLE)))

        self._tree_updating = True
        try:
            self.body_tree.clear()
            project_keys = () if self._hierarchy is None else self._hierarchy.surface_keys
            project_item = _body_tree_item(
                "Project",
                BODY_TREE_PROJECT_NODE_ID,
                project_keys,
                self._surface_visibility,
            )
            project_font = project_item.font(0)
            project_font.setBold(True)
            project_item.setFont(0, project_font)
            self.body_tree.addTopLevelItem(project_item)
            if self._hierarchy is None:
                return
            for region in self._hierarchy.regions:
                region_keys = tuple(
                    dict.fromkeys(
                        key
                        for mesh in region.meshes
                        for boundary in mesh.boundaries
                        for key in boundary.surface_keys
                    )
                )
                region_item = _body_tree_item(region.name, region.id, region_keys, self._surface_visibility)
                project_item.addChild(region_item)
                for mesh in region.meshes:
                    mesh_keys = tuple(
                        dict.fromkeys(key for boundary in mesh.boundaries for key in boundary.surface_keys)
                    )
                    mesh_item = _body_tree_item(mesh.name, mesh.id, mesh_keys, self._surface_visibility)
                    region_item.addChild(mesh_item)
                    for boundary in mesh.boundaries:
                        mesh_item.addChild(
                            _body_tree_item(
                                boundary.name,
                                boundary.id,
                                boundary.surface_keys,
                                self._surface_visibility,
                            )
                        )
            if had_hierarchy:
                for item in _tree_items(self.body_tree):
                    item.setExpanded(str(item.data(0, _TREE_NODE_ID_ROLE)) in expanded_ids)
            else:
                project_item.setExpanded(False)
                for child_index in range(project_item.childCount()):
                    project_item.child(child_index).setExpanded(True)
        finally:
            self._tree_updating = False
            self._resize_body_tree_overlay()

    def _resize_body_tree_overlay(self, _item: QTreeWidgetItem | None = None) -> None:
        if not hasattr(self, "body_tree_overlay"):
            return
        row_height = max(22, self.body_tree.fontMetrics().height() + 7)
        tree_height = min(
            BODY_TREE_OVERLAY_MAX_HEIGHT - (2 * BODY_TREE_OVERLAY_MARGIN),
            max(1, _visible_tree_row_count(self.body_tree)) * row_height + 2,
        )
        self.body_tree.setFixedHeight(tree_height)
        self.body_tree_overlay.setFixedHeight(tree_height + (2 * BODY_TREE_OVERLAY_MARGIN))
        self.body_tree_overlay.raise_()

    def _on_body_tree_item_changed(self, item: QTreeWidgetItem, _column: int) -> None:
        if self._tree_updating:
            return
        state = item.checkState(0)
        if state == Qt.CheckState.PartiallyChecked:
            return
        surface_keys = tuple(item.data(0, _TREE_SURFACE_KEYS_ROLE) or ())
        visible = state == Qt.CheckState.Checked
        for surface_key in surface_keys:
            self._surface_visibility[tuple(surface_key)] = visible

        self._tree_updating = True
        try:
            for tree_item in _tree_items(self.body_tree):
                keys = tuple(tree_item.data(0, _TREE_SURFACE_KEYS_ROLE) or ())
                tree_item.setCheckState(0, _visibility_check_state(keys, self._surface_visibility))
        finally:
            self._tree_updating = False
        self._apply_actor_visibility()

    def _add_msh_mesh(
        self,
        mesh_cfg: MeshConfig,
        *,
        driven_surfaces: set[tuple[str, int]],
        interface_surfaces: set[tuple[str, int]],
        surface_tags: dict[str, int],
        mesh_region: str | None,
        symmetry: str,
    ) -> tuple[int, np.ndarray]:
        mesh = meshio.read(mesh_cfg.file)
        points = np.asarray(mesh.points, dtype=float)
        scale_factor = 0.001 if mesh_cfg.scale_factor is None else float(mesh_cfg.scale_factor)
        points = points * scale_factor + np.asarray(mesh_cfg.translation_m, dtype=float)
        triangles = _extract_triangles_for_preview(mesh)
        physical_tags = _extract_triangle_physical_tags_for_preview(mesh)
        mirrored_images = _mirrored_triangle_images_for_preview(points, triangles, symmetry)
        base_count = int(triangles.shape[0])

        if physical_tags is None:
            actor = self.viewer.add_mesh(
                _triangles_to_polydata(points, triangles),
                color=RIGID_COLOR,
                show_edges=True,
                edge_color=RIGID_EDGE_COLOR,
                smooth_shading=False,
            )
            self._actor_surface_labels[_vtk_actor_address(actor)] = _surface_hover_label(
                mesh_cfg.name,
                "untagged",
                None,
                int(triangles.shape[0]),
            )
            self._register_mesh_actor(
                actor,
                mesh_region,
                mesh_name=mesh_cfg.name,
                surface_tag=None,
            )
            for mirror_label, mirror_points, mirror_triangles, _source_indices in mirrored_images:
                if not mirror_triangles.size:
                    continue
                actor = self.viewer.add_mesh(
                    _triangles_to_polydata(mirror_points, mirror_triangles),
                    color=RIGID_MIRROR_COLOR,
                    show_edges=True,
                    edge_color=RIGID_MIRROR_EDGE_COLOR,
                    smooth_shading=False,
                )
                self._actor_surface_labels[_vtk_actor_address(actor)] = _surface_hover_label(
                    mesh_cfg.name,
                    f"untagged ({mirror_label} image)",
                    None,
                    int(mirror_triangles.shape[0]),
                )
                self._register_mesh_actor(
                    actor,
                    mesh_region,
                    mesh_name=mesh_cfg.name,
                    surface_tag=None,
                )
            return base_count, _preview_points_with_images(points, mirrored_images)

        names_by_tag = {tag: name for name, tag in surface_tags.items()}
        for tag in sorted(np.unique(physical_tags)):
            tag_mask = physical_tags == tag
            tag_triangles = triangles[tag_mask]
            if not tag_triangles.size:
                continue

            is_driven = (mesh_cfg.name, int(tag)) in driven_surfaces
            is_interface = (mesh_cfg.name, int(tag)) in interface_surfaces
            color, edge_color = _surface_preview_colors(
                is_driven=is_driven,
                is_interface=is_interface,
                mirrored=False,
            )
            actor = self.viewer.add_mesh(
                _triangles_to_polydata(points, tag_triangles),
                color=color,
                show_edges=True,
                edge_color=edge_color,
                smooth_shading=False,
            )
            surface_name = names_by_tag.get(int(tag), f"Tag {int(tag)}")
            self._actor_surface_labels[_vtk_actor_address(actor)] = _surface_hover_label(
                mesh_cfg.name,
                surface_name,
                int(tag),
                int(tag_triangles.shape[0]),
            )
            self._register_mesh_actor(
                actor,
                mesh_region,
                mesh_name=mesh_cfg.name,
                surface_tag=int(tag),
            )
            for mirror_label, mirror_points, mirror_triangles, source_indices in mirrored_images:
                mirror_tag_triangles = mirror_triangles[physical_tags[source_indices] == tag]
                if not mirror_tag_triangles.size:
                    continue
                mirror_color, mirror_edge_color = _surface_preview_colors(
                    is_driven=is_driven,
                    is_interface=is_interface,
                    mirrored=True,
                )
                actor = self.viewer.add_mesh(
                    _triangles_to_polydata(mirror_points, mirror_tag_triangles),
                    color=mirror_color,
                    show_edges=True,
                    edge_color=mirror_edge_color,
                    smooth_shading=False,
                )
                self._actor_surface_labels[_vtk_actor_address(actor)] = _surface_hover_label(
                    mesh_cfg.name,
                    f"{surface_name} ({mirror_label} image)",
                    int(tag),
                    int(mirror_tag_triangles.shape[0]),
                )
                self._register_mesh_actor(
                    actor,
                    mesh_region,
                    mesh_name=mesh_cfg.name,
                    surface_tag=int(tag),
                )
        return base_count, _preview_points_with_images(points, mirrored_images)

    def _register_mesh_actor(
        self,
        actor: object,
        mesh_region: str | None,
        *,
        mesh_name: str,
        surface_tag: int | None,
        diagnostic: bool = False,
    ) -> None:
        self._actor_records.append(
            _PreviewActorRecord(
                actor=actor,
                mesh_name=mesh_name,
                surface_key=None if diagnostic else (mesh_name, surface_tag),
                mesh_region=mesh_region,
                diagnostic=diagnostic,
            )
        )

    def _apply_actor_visibility(self, *, render: bool = True) -> None:
        if self.viewer is None:
            return
        visible_meshes = {
            mesh_name
            for (mesh_name, _tag), visible in self._surface_visibility.items()
            if visible
        }
        for record in self._actor_records:
            if record.diagnostic:
                visible = record.mesh_name in visible_meshes or not self._surface_visibility
            else:
                visible = self._surface_visibility.get(record.surface_key, True)
            if self._observation_clip_active and record.mesh_region == PREVIEW_REGION_INTERIOR:
                visible = False
            record.actor.SetVisibility(visible)
        if render:
            self.viewer.render()

    def _set_total_element_count(
        self,
        count: int,
        *,
        mirrored: bool = False,
        dimensions_mm: tuple[int, int, int] | None = None,
    ) -> None:
        self.total_elements_label.setText(_mesh_stats_label(count, mirrored=mirrored, dimensions_mm=dimensions_mm))

    def _camera_position(self):
        if self.viewer is None:
            return None
        if not self._actor_surface_labels:
            return None
        try:
            return self.viewer.camera_position
        except Exception:
            return None

    def _restore_camera_or_reset(self, camera_position) -> None:
        if self.viewer is None:
            return
        if camera_position is None:
            self._reset_camera_to_home()
            return
        try:
            self.viewer.camera_position = camera_position
        except Exception:
            self._reset_camera_to_home()

    def _reset_camera_to_home(self) -> None:
        if self.viewer is None:
            return
        self.viewer.reset_camera()
        try:
            camera = self.viewer.camera
            focal_point = np.asarray(camera.focal_point, dtype=float)
            distance = float(camera.distance)
            if not np.isfinite(distance) or distance <= 0.0:
                distance = 1.0
            position = focal_point + PREVIEW_HOME_CAMERA_DIRECTION * distance
            self.viewer.camera_position = (
                tuple(float(value) for value in position),
                tuple(float(value) for value in focal_point),
                tuple(float(value) for value in PREVIEW_HOME_VIEW_UP),
            )
            camera.zoom(PREVIEW_HOME_ZOOM)
            self.viewer.reset_camera_clipping_range()
        except Exception:
            self.viewer.reset_camera()

    def _add_orientation_guides(self, points: np.ndarray) -> None:
        if self.viewer is None or pv is None:
            return

        length = _preview_axis_length(points)
        axis_specs = (
            ((-length, 0.0, 0.0), (length, 0.0, 0.0), AXIS_COLORS[0]),
            ((0.0, -length, 0.0), (0.0, length, 0.0), AXIS_COLORS[1]),
            ((0.0, 0.0, -length), (0.0, 0.0, length), AXIS_COLORS[2]),
        )
        for start, end, color in axis_specs:
            self.viewer.add_mesh(
                pv.Line(start, end),
                color=color,
                line_width=AXIS_LINE_WIDTH,
                render_lines_as_tubes=True,
                pickable=False,
            )

        self.viewer.add_point_labels(
            _preview_axis_label_points(length),
            list(AXIS_LABELS),
            font_size=14,
            text_color="white",
            point_color="white",
            point_size=0,
            shape_opacity=0.35,
            always_visible=True,
        )

    def _install_hover_picker(self) -> None:
        if self.viewer is None or vtk is None:
            return

        self._hover_picker = vtk.vtkCellPicker()
        self._hover_picker.SetTolerance(0.0005)
        interactor = _preview_interactor(self.viewer)
        if interactor is None:
            return

        if hasattr(interactor, "add_observer"):
            self._hover_observer = interactor.add_observer("MouseMoveEvent", self._on_mouse_move)
        elif hasattr(interactor, "AddObserver"):
            self._hover_observer = interactor.AddObserver("MouseMoveEvent", self._on_mouse_move)

    def _on_mouse_move(self, *args) -> None:
        if self.viewer is None or self._hover_picker is None:
            return

        interactor = args[0] if args and hasattr(args[0], "GetEventPosition") else _preview_interactor(self.viewer)
        renderer = getattr(self.viewer, "renderer", None)
        if interactor is None or renderer is None:
            return

        x_pos, y_pos = interactor.GetEventPosition()
        if not self._hover_picker.Pick(x_pos, y_pos, 0, renderer):
            self.hover_label.setText("")
            return

        actor = self._hover_picker.GetActor()
        label = self._actor_surface_labels.get(_vtk_actor_address(actor))
        self.hover_label.setText(f"{label}" if label else "")


def _preview_interactor(viewer):
    interactor = getattr(viewer, "interactor", None)
    if interactor is not None and hasattr(interactor, "GetEventPosition"):
        return interactor

    plotter_interactor = getattr(viewer, "iren", None)
    if plotter_interactor is not None:
        raw_interactor = getattr(plotter_interactor, "interactor", None)
        if raw_interactor is not None and hasattr(raw_interactor, "GetEventPosition"):
            return raw_interactor
        if hasattr(plotter_interactor, "GetEventPosition"):
            return plotter_interactor

    return None


def _vtk_actor_address(actor) -> str:
    if actor is None:
        return ""
    if hasattr(actor, "GetAddressAsString"):
        return actor.GetAddressAsString("")
    return str(id(actor))


def _surface_preview_colors(
    *,
    is_driven: bool,
    is_interface: bool,
    mirrored: bool,
) -> tuple[str, str]:
    if is_interface:
        if mirrored:
            return INTERFACE_MIRROR_COLOR, INTERFACE_MIRROR_EDGE_COLOR
        return INTERFACE_COLOR, INTERFACE_EDGE_COLOR
    if is_driven:
        if mirrored:
            return DRIVEN_MIRROR_COLOR, DRIVEN_MIRROR_EDGE_COLOR
        return DRIVEN_COLOR, DRIVEN_EDGE_COLOR
    if mirrored:
        return RIGID_MIRROR_COLOR, RIGID_MIRROR_EDGE_COLOR
    return RIGID_COLOR, RIGID_EDGE_COLOR


def _body_tree_item(
    name: str,
    node_id: str,
    surface_keys: tuple[tuple[str, int | None], ...],
    visibility: dict[tuple[str, int | None], bool],
) -> QTreeWidgetItem:
    item = QTreeWidgetItem([name])
    item.setData(0, _TREE_SURFACE_KEYS_ROLE, surface_keys)
    item.setData(0, _TREE_NODE_ID_ROLE, node_id)
    if surface_keys:
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
    item.setCheckState(0, _visibility_check_state(surface_keys, visibility))
    return item


def _visibility_check_state(
    surface_keys: tuple[tuple[str, int | None], ...],
    visibility: dict[tuple[str, int | None], bool],
) -> Qt.CheckState:
    values = [visibility.get(tuple(surface_key), True) for surface_key in surface_keys]
    if not values or all(values):
        return Qt.CheckState.Checked
    if not any(values):
        return Qt.CheckState.Unchecked
    return Qt.CheckState.PartiallyChecked


def _tree_items(tree: QTreeWidget) -> tuple[QTreeWidgetItem, ...]:
    items = []

    def append_item(item: QTreeWidgetItem) -> None:
        items.append(item)
        for child_index in range(item.childCount()):
            append_item(item.child(child_index))

    for root_index in range(tree.topLevelItemCount()):
        append_item(tree.topLevelItem(root_index))
    return tuple(items)


def _visible_tree_row_count(tree: QTreeWidget) -> int:
    def count_item(item: QTreeWidgetItem) -> int:
        count = 1
        if item.isExpanded():
            for child_index in range(item.childCount()):
                count += count_item(item.child(child_index))
        return count

    return sum(count_item(tree.topLevelItem(index)) for index in range(tree.topLevelItemCount()))


def _surface_hover_label(mesh_name: str | None, surface_name: str, tag: int | None, element_count: int) -> str:
    parts = []
    if mesh_name:
        parts.append(f"Mesh: {mesh_name}")
    parts.append(f"Surface: {surface_name}")
    parts.append("Tag: untagged" if tag is None else f"Tag: {tag}")
    parts.append(f"Elements: {element_count:,}")
    return " | ".join(parts)


def _mesh_stats_label(
    count: int,
    *,
    mirrored: bool = False,
    dimensions_mm: tuple[int, int, int] | None = None,
) -> str:
    if not count:
        return ""

    element_text = f"Total elements: {count:,}"
    if mirrored:
        element_text = f"{element_text} (Mirrored)"
    if dimensions_mm is None:
        return element_text

    length_mm, width_mm, height_mm = dimensions_mm
    return f"{element_text} | {length_mm}mm x {width_mm}mm x {height_mm}mm (LWH)"


def _dimensions_lwh_mm(points: np.ndarray) -> tuple[int, int, int]:
    if points.size == 0:
        return (0, 0, 0)

    finite_points = np.asarray(points, dtype=float)
    finite_points = finite_points[np.all(np.isfinite(finite_points), axis=1)]
    if finite_points.size == 0:
        return (0, 0, 0)

    min_bounds = np.nanmin(finite_points, axis=0)
    max_bounds = np.nanmax(finite_points, axis=0)
    extents_mm = np.maximum(max_bounds - min_bounds, 0.0) * 1000.0
    width_mm = int(round(float(extents_mm[0])))
    height_mm = int(round(float(extents_mm[1])))
    length_mm = int(round(float(extents_mm[2])))
    return (length_mm, width_mm, height_mm)


def _preview_axis_length(points: np.ndarray) -> float:
    if points.size == 0:
        return 1.0
    finite_points = np.asarray(points, dtype=float)
    finite_points = finite_points[np.all(np.isfinite(finite_points), axis=1)]
    if finite_points.size == 0:
        return 1.0
    min_bounds = np.nanmin(finite_points, axis=0)
    max_bounds = np.nanmax(finite_points, axis=0)
    extent = float(np.linalg.norm(max_bounds - min_bounds))
    radius = float(np.nanmax(np.linalg.norm(finite_points, axis=1)))
    return max(extent * 0.56, radius * 1.12, 1.0)


def _preview_axis_label_points(length: float) -> np.ndarray:
    return np.array(
        [
            [length * 1.06, 0.0, 0.0],
            [0.0, length * 1.06, 0.0],
            [0.0, 0.0, length * 1.16],
        ],
        dtype=float,
    )


def _preview_points_with_images(
    points: np.ndarray, mirrored_images: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]
) -> np.ndarray:
    image_points = [
        image_points for _label, image_points, image_triangles, _indices in mirrored_images if image_triangles.size
    ]
    if not image_points:
        return points
    return np.vstack((points, *image_points))


def _mirrored_triangle_images_for_preview(
    points: np.ndarray,
    triangles: np.ndarray,
    symmetry: str,
    *,
    tolerance: float = 1e-9,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    transforms = _symmetry_preview_transforms(symmetry)
    if not transforms or triangles.size == 0:
        return []

    seen = {_triangle_geometry_key(points, triangle, tolerance) for triangle in np.asarray(triangles, dtype=np.int64)}
    images: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for label, signs in transforms:
        mirror_points = np.asarray(points, dtype=float) * np.asarray(signs, dtype=float)
        odd_reflections = sum(1 for sign in signs if sign < 0) % 2 == 1
        oriented_triangles = triangles[:, [0, 2, 1]] if odd_reflections else triangles.copy()
        kept_triangles = []
        source_indices = []
        for source_index, triangle in enumerate(oriented_triangles):
            key = _triangle_geometry_key(mirror_points, triangle, tolerance)
            if key in seen:
                continue
            seen.add(key)
            kept_triangles.append(triangle)
            source_indices.append(source_index)
        if kept_triangles:
            images.append(
                (
                    label,
                    mirror_points,
                    np.asarray(kept_triangles, dtype=np.int64),
                    np.asarray(source_indices, dtype=np.int64),
                )
            )
    return images


def _symmetry_preview_transforms(symmetry: str) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    mode = str(symmetry or "off").strip().lower()
    if mode == "x":
        return (("X", (-1.0, 1.0, 1.0)),)
    if mode == "xy":
        return (
            ("X", (-1.0, 1.0, 1.0)),
            ("Y", (1.0, -1.0, 1.0)),
            ("XY", (-1.0, -1.0, 1.0)),
        )
    return ()


def _triangle_geometry_key(
    points: np.ndarray, triangle: np.ndarray, tolerance: float
) -> tuple[tuple[int, int, int], ...]:
    scale = 1.0 / max(float(tolerance), 1e-12)
    coords = np.rint(np.asarray(points, dtype=float)[triangle] * scale).astype(np.int64)
    return tuple(sorted(tuple(int(value) for value in coord) for coord in coords))


def _line_segments_with_symmetry_images(
    segments: np.ndarray,
    symmetry: str,
    *,
    tolerance: float = 1e-12,
) -> np.ndarray:
    source = np.asarray(segments, dtype=float).reshape((-1, 2, 3))
    transforms = (("base", (1.0, 1.0, 1.0)), *_symmetry_preview_transforms(symmetry))
    unique_segments = []
    seen = set()
    scale = 1.0 / max(float(tolerance), 1e-15)
    for _label, signs in transforms:
        transformed = source * np.asarray(signs, dtype=float)
        for segment in transformed:
            quantized = np.rint(segment * scale).astype(np.int64)
            key = tuple(sorted(tuple(int(value) for value in point) for point in quantized))
            if key in seen:
                continue
            seen.add(key)
            unique_segments.append(segment)
    if not unique_segments:
        return np.empty((0, 2, 3), dtype=float)
    return np.asarray(unique_segments, dtype=float)


def _line_segments_to_polydata(segments: np.ndarray):
    values = np.asarray(segments, dtype=float).reshape((-1, 2, 3))
    points = values.reshape((-1, 3))
    line_indices = np.arange(len(points), dtype=np.int64).reshape((-1, 2))
    lines = np.column_stack((np.full(len(line_indices), 2, dtype=np.int64), line_indices)).ravel()
    polydata = pv.PolyData(points)
    polydata.lines = lines
    return polydata


def _triangles_to_polydata(points: np.ndarray, triangles: np.ndarray):
    faces = np.column_stack(
        [
            np.full(triangles.shape[0], 3, dtype=np.int64),
            triangles.astype(np.int64, copy=False),
        ]
    ).ravel()
    return pv.PolyData(points, faces)


def _extract_triangles_for_preview(mesh: meshio.Mesh) -> np.ndarray:
    if "triangle" in mesh.cells_dict:
        return np.asarray(mesh.cells_dict["triangle"], dtype=np.int64)
    if "triangle3" in mesh.cells_dict:
        return np.asarray(mesh.cells_dict["triangle3"], dtype=np.int64)
    raise ValueError("No triangle surface cells found in mesh.")


def _extract_triangle_physical_tags_for_preview(mesh: meshio.Mesh) -> np.ndarray | None:
    tri_key = "triangle" if "triangle" in mesh.cells_dict else "triangle3" if "triangle3" in mesh.cells_dict else None
    if tri_key is None:
        return None

    for data_name, by_cell_type in mesh.cell_data_dict.items():
        if data_name == "gmsh:physical" and tri_key in by_cell_type:
            return np.asarray(by_cell_type[tri_key])
    return None
