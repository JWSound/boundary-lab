"""PyVista mesh preview widget for Ath-generated and imported meshes."""

from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from blab.ath import AthRunResult, read_surface_physical_names
from blab.config import MeshConfig

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


class MeshPreview(QWidget):
    def __init__(self):
        super().__init__()
        self._hover_picker = None
        self._hover_observer = None
        self._actor_surface_labels: dict[str, str] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if QtInteractor is None:
            self.viewer = None
            label = QLabel("Install pyvista and pyvistaqt to enable mesh preview.")
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
            return

        self.viewer = QtInteractor(self)
        layout.addWidget(self.viewer)
        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        self.hover_label = QLabel("")
        self.hover_label.setMinimumHeight(22)
        self.total_elements_label = QLabel("")
        self.total_elements_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.total_elements_label.setMinimumHeight(22)
        status_row.addWidget(self.hover_label, 1)
        status_row.addWidget(self.total_elements_label)
        layout.addLayout(status_row)
        self.viewer.set_background("white")
        self._install_hover_picker()

    def clear(self) -> None:
        if self.viewer is None:
            return
        self.viewer.clear()
        self._actor_surface_labels = {}
        self.hover_label.setText("")
        self._set_total_element_count(0)

    def load_ath_result(self, result: AthRunResult) -> None:
        self.load_mesh_configs(
            (
                MeshConfig(name="ath", file=str(result.solver_msh_path), scale_factor=0.001),
            ),
            driven_surfaces={("ath", radiator.tag) for radiator in result.radiators},
            surface_tags_by_mesh={"ath": read_surface_physical_names(result.solver_msh_path)},
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
        self.hover_label.setText("")
        self._set_total_element_count(int(triangles.shape[0]))

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
            self._add_orientation_guides(np.asarray(mesh.points, dtype=float))
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
                color="#395865" if is_driven else "#cfcfcf",
                show_edges=True,
                edge_color="#20343c" if is_driven else "#555555",
                smooth_shading=False,
            )
            surface_name = names_by_tag.get(int(tag), f"Tag {int(tag)}")
            self._actor_surface_labels[_vtk_actor_address(actor)] = _surface_hover_label(
                None,
                surface_name,
                int(tag),
                int(tag_triangles.shape[0]),
            )

        self._add_orientation_guides(np.asarray(mesh.points, dtype=float))
        self._restore_camera_or_reset(camera_position)

    def load_mesh_configs(
        self,
        meshes: tuple[MeshConfig, ...],
        *,
        driven_surfaces: set[tuple[str, int]] | None = None,
        surface_tags_by_mesh: dict[str, dict[str, int]] | None = None,
    ) -> None:
        if self.viewer is None:
            return
        camera_position = self._camera_position()
        self.viewer.clear()
        self._actor_surface_labels = {}
        self.hover_label.setText("")
        total_elements = 0
        preview_points = []

        for mesh_cfg in meshes:
            mesh_elements, mesh_points = self._add_msh_mesh(
                mesh_cfg,
                driven_surfaces=driven_surfaces or set(),
                surface_tags=(surface_tags_by_mesh or {}).get(mesh_cfg.name, {}),
            )
            total_elements += mesh_elements
            preview_points.append(mesh_points)

        self._set_total_element_count(total_elements)
        if preview_points:
            self._add_orientation_guides(np.vstack(preview_points))
        self._restore_camera_or_reset(camera_position)

    def _add_msh_mesh(
        self,
        mesh_cfg: MeshConfig,
        *,
        driven_surfaces: set[tuple[str, int]],
        surface_tags: dict[str, int],
    ) -> tuple[int, np.ndarray]:
        mesh = meshio.read(mesh_cfg.file)
        points = np.asarray(mesh.points, dtype=float)
        scale_factor = 0.001 if mesh_cfg.scale_factor is None else float(mesh_cfg.scale_factor)
        points = points * scale_factor + np.asarray(mesh_cfg.translation_m, dtype=float)
        triangles = _extract_triangles_for_preview(mesh)
        physical_tags = _extract_triangle_physical_tags_for_preview(mesh)

        if physical_tags is None:
            actor = self.viewer.add_mesh(
                _triangles_to_polydata(points, triangles),
                color="#cfcfcf",
                show_edges=True,
                edge_color="#555555",
                smooth_shading=False,
            )
            self._actor_surface_labels[_vtk_actor_address(actor)] = _surface_hover_label(
                mesh_cfg.name,
                "untagged",
                None,
                int(triangles.shape[0]),
            )
            return int(triangles.shape[0]), points

        names_by_tag = {tag: name for name, tag in surface_tags.items()}
        for tag in sorted(np.unique(physical_tags)):
            tag_mask = physical_tags == tag
            tag_triangles = triangles[tag_mask]
            if not tag_triangles.size:
                continue

            is_driven = (mesh_cfg.name, int(tag)) in driven_surfaces
            actor = self.viewer.add_mesh(
                _triangles_to_polydata(points, tag_triangles),
                color="#395865" if is_driven else "#cfcfcf",
                show_edges=True,
                edge_color="#20343c" if is_driven else "#555555",
                smooth_shading=False,
            )
            surface_name = names_by_tag.get(int(tag), f"Tag {int(tag)}")
            self._actor_surface_labels[_vtk_actor_address(actor)] = _surface_hover_label(
                mesh_cfg.name,
                surface_name,
                int(tag),
                int(tag_triangles.shape[0]),
            )
        return int(triangles.shape[0]), points

    def _set_total_element_count(self, count: int) -> None:
        self.total_elements_label.setText(f"Total elements: {count:,}" if count else "")

    def _camera_position(self):
        if self.viewer is None:
            return None
        try:
            return self.viewer.camera_position
        except Exception:
            return None

    def _restore_camera_or_reset(self, camera_position) -> None:
        if self.viewer is None:
            return
        if camera_position is None:
            self.viewer.reset_camera()
            return
        try:
            self.viewer.camera_position = camera_position
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


def _surface_hover_label(mesh_name: str | None, surface_name: str, tag: int | None, element_count: int) -> str:
    parts = []
    if mesh_name:
        parts.append(f"Mesh: {mesh_name}")
    parts.append(f"Surface: {surface_name}")
    parts.append("Tag: untagged" if tag is None else f"Tag: {tag}")
    parts.append(f"Elements: {element_count:,}")
    return " | ".join(parts)


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
