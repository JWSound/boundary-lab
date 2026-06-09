"""PyVista mesh preview widget for Ath-generated and imported meshes."""

from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

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
RIGID_COLOR = "#cfcfcf"
RIGID_MIRROR_COLOR = "#a9a9a9"
RIGID_EDGE_COLOR = "#555555"
RIGID_MIRROR_EDGE_COLOR = "#4a4a4a"
DRIVEN_COLOR = "#395865"
DRIVEN_MIRROR_COLOR = "#2f4751"
DRIVEN_EDGE_COLOR = "#20343c"
DRIVEN_MIRROR_EDGE_COLOR = "#1b2c33"


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
        self.hover_label.setMinimumWidth(0)
        self.hover_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.total_elements_label = QLabel("")
        self.total_elements_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.total_elements_label.setMinimumHeight(22)
        self.total_elements_label.setMinimumWidth(0)
        self.total_elements_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
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
        display_points = np.asarray(mesh.points, dtype=float)
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

        self._add_orientation_guides(display_points)
        self._restore_camera_or_reset(camera_position)

    def load_mesh_configs(
        self,
        meshes: tuple[MeshConfig, ...],
        *,
        driven_surfaces: set[tuple[str, int]] | None = None,
        surface_tags_by_mesh: dict[str, dict[str, int]] | None = None,
        symmetry: str = "off",
    ) -> None:
        if self.viewer is None:
            return
        camera_position = self._camera_position()
        self.viewer.clear()
        self._actor_surface_labels = {}
        self.hover_label.setText("")
        total_elements = 0
        preview_points = []
        mirrored = str(symmetry or "off").strip().lower() != "off"

        for mesh_cfg in meshes:
            mesh_elements, mesh_points = self._add_msh_mesh(
                mesh_cfg,
                driven_surfaces=driven_surfaces or set(),
                surface_tags=(surface_tags_by_mesh or {}).get(mesh_cfg.name, {}),
                symmetry=symmetry,
            )
            total_elements += mesh_elements
            preview_points.append(mesh_points)

        display_points = np.vstack(preview_points) if preview_points else np.empty((0, 3))
        self._set_total_element_count(
            total_elements,
            mirrored=mirrored,
            dimensions_mm=_dimensions_lwh_mm(display_points),
        )
        if preview_points:
            self._add_orientation_guides(display_points)
        self._restore_camera_or_reset(camera_position)

    def _add_msh_mesh(
        self,
        mesh_cfg: MeshConfig,
        *,
        driven_surfaces: set[tuple[str, int]],
        surface_tags: dict[str, int],
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
            return base_count, _preview_points_with_images(points, mirrored_images)

        names_by_tag = {tag: name for name, tag in surface_tags.items()}
        for tag in sorted(np.unique(physical_tags)):
            tag_mask = physical_tags == tag
            tag_triangles = triangles[tag_mask]
            if not tag_triangles.size:
                continue

            is_driven = (mesh_cfg.name, int(tag)) in driven_surfaces
            actor = self.viewer.add_mesh(
                _triangles_to_polydata(points, tag_triangles),
                color=DRIVEN_COLOR if is_driven else RIGID_COLOR,
                show_edges=True,
                edge_color=DRIVEN_EDGE_COLOR if is_driven else RIGID_EDGE_COLOR,
                smooth_shading=False,
            )
            surface_name = names_by_tag.get(int(tag), f"Tag {int(tag)}")
            self._actor_surface_labels[_vtk_actor_address(actor)] = _surface_hover_label(
                mesh_cfg.name,
                surface_name,
                int(tag),
                int(tag_triangles.shape[0]),
            )
            for mirror_label, mirror_points, mirror_triangles, source_indices in mirrored_images:
                mirror_tag_triangles = mirror_triangles[physical_tags[source_indices] == tag]
                if not mirror_tag_triangles.size:
                    continue
                actor = self.viewer.add_mesh(
                    _triangles_to_polydata(mirror_points, mirror_tag_triangles),
                    color=DRIVEN_MIRROR_COLOR if is_driven else RIGID_MIRROR_COLOR,
                    show_edges=True,
                    edge_color=DRIVEN_MIRROR_EDGE_COLOR if is_driven else RIGID_MIRROR_EDGE_COLOR,
                    smooth_shading=False,
                )
                self._actor_surface_labels[_vtk_actor_address(actor)] = _surface_hover_label(
                    mesh_cfg.name,
                    f"{surface_name} ({mirror_label} image)",
                    int(tag),
                    int(mirror_tag_triangles.shape[0]),
                )
        return base_count, _preview_points_with_images(points, mirrored_images)

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


def _preview_points_with_images(points: np.ndarray, mirrored_images: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    image_points = [image_points for _label, image_points, image_triangles, _indices in mirrored_images if image_triangles.size]
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

    seen = {
        _triangle_geometry_key(points, triangle, tolerance)
        for triangle in np.asarray(triangles, dtype=np.int64)
    }
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


def _triangle_geometry_key(points: np.ndarray, triangle: np.ndarray, tolerance: float) -> tuple[tuple[int, int, int], ...]:
    scale = 1.0 / max(float(tolerance), 1e-12)
    coords = np.rint(np.asarray(points, dtype=float)[triangle] * scale).astype(np.int64)
    return tuple(sorted(tuple(int(value) for value in coord) for coord in coords))


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
