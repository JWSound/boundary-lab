"""Interactive PyVista authoring layer for observation planes."""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass, replace

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import QMenu

from blab.observation_planes import (
    InteriorRenderingMode,
    ObservationPlane,
    ObservationPlaneType,
    rotate_observation_plane,
)
from blab.solvers.coupled_field import BemFieldEvaluationRequest
from blab.ui.exterior_field_service import ExteriorFieldTask
from blab.ui.observation_plane_results import (
    ObservationFieldResults,
    project_field_scalars,
)

AXIS_COLORS = ("#ef5350", "#66bb6a", "#42a5f5")
SELECTED_EDGE_COLOR = "#ffd54f"
UNSELECTED_EDGE_COLOR = "#80a4b8"
PLANE_COLOR = "#48a9d6"
HELP_ACTOR_NAME = "observation-plane:help"
ANGLE_ACTOR_NAME = "observation-plane:rotation-angle"
FIELD_MESSAGE_ACTOR_NAME = "observation-plane:field-message"
FIELD_SCALAR_NAME = "observation-plane-field"
ROTATION_SNAP_DEG = 5.0


@dataclass
class _DragState:
    original: ObservationPlane
    mode: str
    control: int
    start_axis_parameter: float | None = None
    start_rotation_vector: np.ndarray | None = None
    fixed_corner: np.ndarray | None = None
    start_display_xy: np.ndarray | None = None
    fallback_scale: float = 0.0
    fallback_screen_direction: np.ndarray | None = None


@dataclass
class _ActiveFieldState:
    plane_id: str
    mesh: object
    pressure: np.ndarray
    association: str
    actor: object | None = None


@dataclass(frozen=True)
class _SmoothFieldGeometry:
    mesh: object
    source_node_ids: np.ndarray
    interpolation_weights: np.ndarray
    clipped: object


@dataclass(frozen=True)
class _ElementFieldGeometry:
    mesh: object
    source_tetrahedron_ids: np.ndarray


class ObservationPlaneViewport(QObject):
    """Own plane actors and translate Qt input into pure plane-model updates."""

    newPlaneRequested = Signal(object)
    planeChanged = Signal(object)
    propertiesRequested = Signal(str)
    deleteRequested = Signal(str)
    clipStateChanged = Signal(bool)
    exteriorFieldRequested = Signal(object)

    def __init__(self, viewer, vtk_module, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self.vtk = vtk_module
        self._planes: tuple[ObservationPlane, ...] = ()
        self._selected_id: str | None = None
        self._active_id: str | None = None
        self._mode: str | None = None
        self._drag: _DragState | None = None
        self._plane_actor_ids: dict[str, str] = {}
        self._tool_actor_controls: dict[str, tuple[str, int]] = {}
        self._actor_names: set[str] = set()
        self._tool_actors: list[object] = []
        self._scene_bounds: tuple[float, float, float, float, float, float] | None = None
        self._rotation_angle_deg: float | None = None
        self._field_results: ObservationFieldResults | None = None
        self._field_results_signature: tuple[object, ...] | None = None
        self._field_message: str | None = None
        self._field_cache: dict[tuple[str, str], tuple[tuple[object, ...], object]] = {}
        self._field_generation = 0
        self._scalar_bar_titles: set[str] = set()
        self._clip_active = False
        self._animation_plane_id: str | None = None
        self._animation_phase_deg = 0.0
        self._animation_updated_at = time.monotonic()
        self._active_field: _ActiveFieldState | None = None
        self._exterior_results: OrderedDict[tuple[object, ...], np.ndarray] = OrderedDict()
        self._exterior_pending: set[tuple[object, ...]] = set()
        self._exterior_failures: dict[tuple[object, ...], str] = {}
        self._animation_timer = QTimer(self)
        self._animation_timer.setInterval(50)
        self._animation_timer.timeout.connect(self._advance_animation)
        self._picker = vtk_module.vtkCellPicker()
        self._picker.SetTolerance(0.001)
        self._foreground_renderer = self._create_foreground_renderer()
        viewer.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        viewer.installEventFilter(self)
        if hasattr(viewer, "clear_events_for_key"):
            viewer.clear_events_for_key("r")
        if hasattr(viewer, "add_key_event"):
            viewer.add_key_event("r", self._activate_rotate_mode)

    @property
    def selected_id(self) -> str | None:
        return self._selected_id

    def set_scene_bounds(self, points: np.ndarray) -> None:
        finite = np.asarray(points, dtype=float)
        finite = finite[np.all(np.isfinite(finite), axis=1)] if finite.size else finite
        if not finite.size:
            self._scene_bounds = None
            return
        minimum = np.min(finite, axis=0)
        maximum = np.max(finite, axis=0)
        self._scene_bounds = (
            float(minimum[0]),
            float(maximum[0]),
            float(minimum[1]),
            float(maximum[1]),
            float(minimum[2]),
            float(maximum[2]),
        )

    def scene_cleared(self) -> None:
        self._remove_actors()
        self._plane_actor_ids.clear()
        self._tool_actor_controls.clear()
        self._field_message = None
        self._render()

    def set_planes(self, planes: tuple[ObservationPlane, ...], *, selected_id: str | None = None) -> None:
        validated = tuple(plane.validated() for plane in planes)
        previous_planes = self._planes
        previous_selected_id = self._selected_id
        self._planes = validated
        available_ids = {plane.id for plane in self._planes}
        if self._active_id not in available_ids:
            self._active_id = None
        for cache_key in tuple(self._field_cache):
            if cache_key[1] not in available_ids:
                self._field_cache.pop(cache_key, None)
        if selected_id is not None:
            self._selected_id = selected_id if selected_id in available_ids else None
        elif self._selected_id not in available_ids:
            self._selected_id = None
        if self._selected_id is None:
            self._mode = None
        result_update = _result_selection_update(
            previous_planes,
            self._planes,
            previous_selected_id,
            self._selected_id,
            self._active_id,
        )
        if result_update == "state_only":
            return
        if result_update == "field" and self._update_active_field(self._field_plane()):
            return
        self._render()

    def set_active_plane(self, plane_id: str | None) -> None:
        available_ids = {plane.id for plane in self._planes}
        updated = plane_id if plane_id in available_ids else None
        if updated == self._active_id:
            return
        previous_field_id = self._field_plane_id()
        self._active_id = updated
        if self._field_plane_id() != previous_field_id:
            self._render()

    def set_field_results(self, results: ObservationFieldResults | None) -> None:
        signature = _observation_field_results_signature(results)
        if signature == self._field_results_signature:
            self._field_results = results
            return
        self._field_results = results
        self._field_results_signature = signature
        self._field_generation += 1
        self._field_cache.clear()
        self._exterior_results.clear()
        self._exterior_pending.clear()
        self._exterior_failures.clear()
        if results is None:
            self.set_animation(None, False)
        self._render()

    def set_exterior_field_result(self, key: tuple[object, ...], pressure: np.ndarray) -> None:
        self._exterior_pending.discard(key)
        self._exterior_failures.pop(key, None)
        self._exterior_results[key] = np.asarray(pressure, dtype=np.complex64)
        self._exterior_results.move_to_end(key)
        while len(self._exterior_results) > 16:
            self._exterior_results.popitem(last=False)
        self._render()

    def set_exterior_field_failed(self, key: tuple[object, ...], message: str) -> None:
        self._exterior_pending.discard(key)
        self._exterior_failures[key] = str(message)
        self._render()

    def discard_exterior_field_request(self, key: tuple[object, ...]) -> None:
        self._exterior_pending.discard(key)

    def set_animation(self, plane_id: str | None, enabled: bool) -> None:
        if not enabled or plane_id is None or self._field_results is None:
            if self._animation_plane_id is None and not self._animation_timer.isActive():
                return
            self._animation_plane_id = None
            self._animation_timer.stop()
            self._animation_phase_deg = 0.0
            self._render()
            return
        if self._animation_plane_id == str(plane_id) and self._animation_timer.isActive():
            return
        self._animation_plane_id = str(plane_id)
        self._animation_phase_deg = 0.0
        self._animation_updated_at = time.monotonic()
        if not self._animation_timer.isActive():
            self._animation_timer.start()
        self._render()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt override
        if watched is not self.viewer:
            return False
        event_type = event.type()
        if event_type == QEvent.Type.ContextMenu:
            self._show_context_menu(event)
            return True
        if event_type == QEvent.Type.KeyPress:
            return self._on_key_press(event)
        if event_type == QEvent.Type.MouseButtonDblClick and event.button() == Qt.MouseButton.LeftButton:
            plane_id = self._picked_plane_id(event.position())
            if plane_id is None:
                return False
            self._select(plane_id)
            self.propertiesRequested.emit(plane_id)
            return True
        if event_type == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            return self._on_left_press(event)
        if event_type == QEvent.Type.MouseMove and self._drag is not None:
            self._on_drag(event)
            return True
        if event_type == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
            if self._drag is None:
                return False
            self._drag = None
            self._rotation_angle_deg = None
            self._render()
            plane = self._selected_plane()
            if plane is not None:
                self.planeChanged.emit(plane)
            return True
        return False

    def _on_key_press(self, event) -> bool:
        key = event.key()
        if key == Qt.Key.Key_R:
            self._activate_rotate_mode()
            return True
        if self._selected_id is None:
            return False
        if key == Qt.Key.Key_M:
            self._set_mode("move")
            return True
        if key == Qt.Key.Key_S:
            self._set_mode("size")
            return True
        if key in {Qt.Key.Key_Delete, Qt.Key.Key_Backspace}:
            self.deleteRequested.emit(self._selected_id)
            return True
        if key == Qt.Key.Key_Escape:
            if self._mode is not None:
                self._set_mode(None)
            else:
                self._select(None)
            return True
        return False

    def _activate_rotate_mode(self) -> None:
        if self._selected_id is None:
            return
        camera_position = _camera_position_snapshot(self.viewer)
        self._set_mode("rotate")
        if camera_position is not None:
            QTimer.singleShot(0, lambda: self._restore_camera_position(camera_position))

    def _restore_camera_position(self, camera_position) -> None:
        try:
            self.viewer.camera_position = camera_position
            self.viewer.render()
        except Exception:
            pass

    def _on_left_press(self, event) -> bool:
        actor = self._pick_actor(event.position())
        address = _vtk_actor_address(actor)
        tool = self._tool_actor_controls.get(address)
        if tool is not None:
            self._begin_drag(tool, event.position())
            return self._drag is not None
        plane_id = self._plane_actor_ids.get(address)
        if plane_id is not None:
            self._select(plane_id)
            return True
        if self._selected_id is not None:
            self._select(None)
        return False

    def _begin_drag(self, tool: tuple[str, int], position) -> None:
        plane = self._selected_plane()
        if plane is None:
            return
        mode, control = tool
        if mode == "move":
            axis = plane.local_axes[control]
            parameter = self._axis_parameter(position, np.asarray(plane.center_m), axis)
            self._drag = _DragState(
                plane,
                mode,
                control,
                start_axis_parameter=parameter,
                start_display_xy=_position_array(position),
                fallback_scale=self._world_per_display_pixel(np.asarray(plane.center_m)),
            )
        elif mode == "rotate":
            axis = plane.local_axes[control]
            intersection = self._ray_plane_intersection(position, np.asarray(plane.center_m), axis)
            vector = None if intersection is None else intersection - np.asarray(plane.center_m)
            if vector is not None and np.linalg.norm(vector) <= 1e-10:
                vector = None
            self._drag = _DragState(
                plane,
                mode,
                control,
                start_rotation_vector=vector,
                start_display_xy=_position_array(position),
                fallback_screen_direction=self._rotation_screen_direction(
                    np.asarray(plane.center_m), axis, self._tool_scale(plane)
                ),
            )
        elif mode == "size":
            opposite_index = (control + 2) % 4
            self._drag = _DragState(
                plane,
                mode,
                control,
                fixed_corner=plane.corners_m[opposite_index].copy(),
            )

    def _on_drag(self, event) -> None:
        drag = self._drag
        if drag is None:
            return
        if drag.mode == "move":
            self._drag_move(event.position(), drag)
        elif drag.mode == "rotate":
            self._drag_rotate(event.position(), drag, alt_pressed=bool(event.modifiers() & Qt.AltModifier))
        elif drag.mode == "size":
            self._drag_size(event.position(), drag)

    def _drag_move(self, position, drag: _DragState) -> None:
        axis = drag.original.local_axes[drag.control]
        parameter = self._axis_parameter(position, np.asarray(drag.original.center_m), axis)
        if parameter is not None and drag.start_axis_parameter is not None:
            distance = parameter - drag.start_axis_parameter
        elif drag.start_display_xy is not None:
            display_delta = _position_array(position) - drag.start_display_xy
            distance = -float(display_delta[1]) * drag.fallback_scale
        else:
            return
        center = np.asarray(drag.original.center_m) + axis * distance
        self._replace_selected(replace(drag.original, center_m=tuple(float(value) for value in center)))

    def _drag_rotate(self, position, drag: _DragState, *, alt_pressed: bool) -> None:
        axis = drag.original.local_axes[drag.control]
        center = np.asarray(drag.original.center_m)
        intersection = self._ray_plane_intersection(position, center, axis)
        if intersection is not None and drag.start_rotation_vector is not None:
            start = drag.start_rotation_vector.copy()
            current = intersection - center
            start -= axis * float(np.dot(start, axis))
            current -= axis * float(np.dot(current, axis))
            if np.linalg.norm(start) <= 1e-10 or np.linalg.norm(current) <= 1e-10:
                return
            start /= np.linalg.norm(start)
            current /= np.linalg.norm(current)
            angle = math.degrees(
                math.atan2(float(np.dot(axis, np.cross(start, current))), float(np.dot(start, current)))
            )
        elif drag.start_display_xy is not None and drag.fallback_screen_direction is not None:
            display_delta = _position_array(position) - drag.start_display_xy
            angle = float(np.dot(display_delta, drag.fallback_screen_direction)) * 0.5
        else:
            return
        if not alt_pressed:
            angle = round(angle / ROTATION_SNAP_DEG) * ROTATION_SNAP_DEG
        self._rotation_angle_deg = angle
        self._replace_selected(rotate_observation_plane(drag.original, drag.control, angle))

    def _drag_size(self, position, drag: _DragState) -> None:
        if drag.fixed_corner is None:
            return
        plane = drag.original
        _axis_u, _axis_v, normal = plane.local_axes
        point = self._ray_plane_intersection(position, np.asarray(plane.center_m), normal)
        if point is None:
            point = self._display_to_world_at_depth(position, np.asarray(plane.center_m))
            if point is None:
                return
            point -= normal * float(np.dot(point - np.asarray(plane.center_m), normal))
        axis_u, axis_v, _normal = plane.local_axes
        signs = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
        sign_u, sign_v = signs[drag.control]
        delta = point - drag.fixed_corner
        width = max(sign_u * float(np.dot(delta, axis_u)), 1e-4)
        height = max(sign_v * float(np.dot(delta, axis_v)), 1e-4)
        center = drag.fixed_corner + sign_u * axis_u * width * 0.5 + sign_v * axis_v * height * 0.5
        self._replace_selected(
            replace(
                plane,
                center_m=tuple(float(value) for value in center),
                width_m=width,
                height_m=height,
            )
        )

    def _show_context_menu(self, event) -> None:
        picked_id = self._picked_plane_id(event.pos())
        if picked_id is not None:
            self._select(picked_id)
        menu = QMenu(self.viewer)
        new_action = menu.addAction("New Observation Plane")
        properties_action = None
        delete_action = None
        if picked_id is not None:
            menu.addSeparator()
            properties_action = menu.addAction("Properties")
            delete_action = menu.addAction("Delete Observation Plane")
        selected = menu.exec(event.globalPos())
        if selected is new_action:
            self.newPlaneRequested.emit(self._new_plane_placement(event.pos()))
        elif properties_action is not None and selected is properties_action:
            self.propertiesRequested.emit(picked_id)
        elif delete_action is not None and selected is delete_action:
            self.deleteRequested.emit(picked_id)

    def _new_plane_placement(self, position) -> dict[str, object]:
        self._picker.Pick(*self._vtk_xy(position), 0, self.viewer.renderer)
        if self._picker.GetActor() is not None:
            center = np.asarray(self._picker.GetPickPosition(), dtype=float)
        else:
            center = np.asarray(self.viewer.camera.focal_point, dtype=float)

        size = self._default_size()
        return {
            "center_m": tuple(float(value) for value in center),
            "orientation_wxyz": (1.0, 0.0, 0.0, 0.0),
            "width_m": size,
            "height_m": size,
        }

    def _default_size(self) -> float:
        if self._scene_bounds is None:
            return 0.25
        bounds = self._scene_bounds
        extent = np.asarray((bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]))
        return max(float(np.linalg.norm(extent)) * 0.35, 0.01)

    def _select(self, plane_id: str | None) -> None:
        if plane_id == self._selected_id:
            return
        self._selected_id = plane_id
        self._mode = None
        self._drag = None
        self._rotation_angle_deg = None
        self._render()

    def _set_mode(self, mode: str | None) -> None:
        self._mode = mode
        self._drag = None
        self._rotation_angle_deg = None
        self._render()

    def _replace_selected(self, updated: ObservationPlane) -> None:
        updated = updated.validated()
        self._planes = tuple(updated if plane.id == updated.id else plane for plane in self._planes)
        for cache_key in (("smooth", updated.id), ("element", updated.id)):
            self._field_cache.pop(cache_key, None)
        self._render()

    def _selected_plane(self) -> ObservationPlane | None:
        return next((plane for plane in self._planes if plane.id == self._selected_id), None)

    def _field_plane_id(self) -> str | None:
        return self._active_id or self._selected_id

    def _field_plane(self) -> ObservationPlane | None:
        field_id = self._field_plane_id()
        return next((plane for plane in self._planes if plane.id == field_id), None)

    def _render(self) -> None:
        self._remove_actors()
        self._active_field = None
        self._plane_actor_ids.clear()
        self._tool_actor_controls.clear()
        self._field_message = None
        if not self._planes:
            self._set_clip_active(False)
            self.viewer.render()
            return
        field_plane_id = self._field_plane_id()
        for plane in self._planes:
            selected = plane.id == self._selected_id
            field_visible = (
                plane.id == field_plane_id
                and self._field_results is not None
                and (
                    (
                        plane.plane_type in {ObservationPlaneType.INTERIOR, ObservationPlaneType.COMBINED}
                        and self._field_results.interior is not None
                    )
                    or (
                        plane.plane_type == ObservationPlaneType.EXTERIOR
                        and self._field_results.exterior is not None
                    )
                )
            )
            mesh = _plane_polydata(plane)
            name = f"observation-plane:{plane.id}"
            actor = self.viewer.add_mesh(
                mesh,
                name=name,
                color=PLANE_COLOR,
                opacity=1.0 if field_visible else 0.32 if selected else 0.18,
                style="wireframe" if field_visible else "surface",
                show_edges=True,
                edge_color=SELECTED_EDGE_COLOR if selected else UNSELECTED_EDGE_COLOR,
                line_width=3.0 if selected else 1.5,
                pickable=True,
                render=False,
            )
            self._actor_names.add(name)
            self._plane_actor_ids[_vtk_actor_address(actor)] = plane.id
        selected = self._selected_plane()
        field_plane = self._field_plane()
        clip_active = False
        if field_plane is not None:
            if field_plane.plane_type == ObservationPlaneType.EXTERIOR:
                self._add_exterior_field(field_plane)
            else:
                clip_active = self._add_interior_field(field_plane)
        if selected is not None:
            self._add_help_text(selected)
            if self._mode == "move":
                self._add_move_gizmo(selected)
            elif self._mode == "rotate":
                self._add_rotation_gizmo(selected)
                if self._rotation_angle_deg is not None:
                    self._add_rotation_angle_text(self._rotation_angle_deg)
            elif self._mode == "size":
                self._add_size_handles(selected)
        if self._field_message:
            self.viewer.add_text(
                self._field_message,
                position="lower_left",
                font_size=9,
                name=FIELD_MESSAGE_ACTOR_NAME,
                render=False,
            )
            self._actor_names.add(FIELD_MESSAGE_ACTOR_NAME)
        self._set_clip_active(clip_active)
        self.viewer.render()

    def _set_clip_active(self, active: bool) -> None:
        active = bool(active)
        if active == self._clip_active:
            return
        self._clip_active = active
        self.clipStateChanged.emit(active)

    def _add_interior_field(self, plane: ObservationPlane) -> bool:
        results = None if self._field_results is None else self._field_results.interior
        if (
            results is None
            or (self._drag is not None and self._drag.original.id == plane.id)
            or plane.plane_type not in {ObservationPlaneType.INTERIOR, ObservationPlaneType.COMBINED}
        ):
            return False
        try:
            mesh, complex_values, association, clipped = self._field_mesh_and_pressure(plane)
            if clipped is not None:
                clip_name = f"observation-plane:field:clip:{plane.id}"
                self.viewer.add_mesh(
                    clipped,
                    name=clip_name,
                    color="#91a4ad",
                    opacity=0.12,
                    show_edges=True,
                    edge_color="#63747c",
                    line_width=0.5,
                    pickable=False,
                    render=False,
                )
                self._actor_names.add(clip_name)
            self._add_colored_field_actor(
                mesh,
                complex_values,
                plane,
                association=association,
                name=f"observation-plane:field:{plane.interior_rendering.value}:{plane.id}",
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            self._field_message = f"Interior field unavailable: {exc}"
            return False
        return True

    def _field_mesh_and_pressure(
        self,
        plane: ObservationPlane,
    ) -> tuple[object, np.ndarray, str, object | None]:
        results = None if self._field_results is None else self._field_results.interior
        if results is None:
            raise ValueError("No interior field results are available.")
        pressure = results.pressure(plane.frequency_hz, plane.response_id)
        if plane.interior_rendering == InteriorRenderingMode.ELEMENT_FIELD:
            geometry = self._element_field_mesh(
                plane,
                results.points_m,
                results.tetrahedra,
                results.symmetry,
            )
            source_pressure = np.mean(pressure[np.asarray(results.tetrahedra, dtype=np.int64)], axis=1)
            values = source_pressure[geometry.source_tetrahedron_ids]
            return geometry.mesh, values, "cell", None
        geometry = self._smooth_field_mesh(
            plane,
            results.points_m,
            results.tetrahedra,
            results.symmetry,
        )
        values = np.sum(
            pressure[geometry.source_node_ids] * geometry.interpolation_weights,
            axis=1,
        )
        return geometry.mesh, values, "point", geometry.clipped

    def _add_exterior_field(self, plane: ObservationPlane) -> bool:
        results = None if self._field_results is None else self._field_results.exterior
        if results is None and self._field_results is not None and plane.plane_type == ObservationPlaneType.EXTERIOR:
            self._field_message = "Exterior field data was not retained; solve again with this plane configured."
            return False
        if (
            results is None
            or (self._drag is not None and self._drag.original.id == plane.id)
            or plane.plane_type != ObservationPlaneType.EXTERIOR
        ):
            return False
        try:
            frequency, boundary_pressure, boundary_normal = results.boundary_response(
                plane.frequency_hz,
                plane.response_id,
            )
            mesh = self._exterior_plane_mesh(plane)
            key = _exterior_field_key(self._field_generation, results.run_id, plane, frequency)
            pressure = self._exterior_results.get(key)
            if pressure is None:
                failure = self._exterior_failures.get(key)
                if failure is not None:
                    self._field_message = f"Exterior field unavailable: {failure}"
                    return False
                if key not in self._exterior_pending:
                    self._exterior_pending.add(key)
                    self.exteriorFieldRequested.emit(
                        ExteriorFieldTask(
                            key=key,
                            backend_id=results.backend_id,
                            request=BemFieldEvaluationRequest(
                                points_m=np.asarray(mesh.points),
                                boundary_points_m=results.traces.points_m,
                                boundary_triangles=results.traces.triangles,
                                boundary_pressure=boundary_pressure,
                                boundary_normal_derivative=boundary_normal,
                                frequency_hz=frequency,
                                sound_speed_m_per_s=results.sound_speed_m_per_s,
                                symmetry=results.traces.symmetry,
                            ),
                        )
                    )
                self._field_message = "Evaluating exterior field..."
                return False
            if pressure.shape != (mesh.n_points,):
                raise ValueError("Evaluated exterior pressure does not align with the plane sample grid.")
            boundary_mask = self._exterior_boundary_mask(plane, mesh, results)
            if np.any(boundary_mask):
                pressure = pressure.copy()
                pressure[boundary_mask] = np.nan + 1j * np.nan
            self._add_colored_field_actor(
                mesh,
                pressure,
                plane,
                association="point",
                name=f"observation-plane:field:exterior:{plane.id}",
            )
            return True
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            self._field_message = f"Exterior field unavailable: {exc}"
            return False

    def _exterior_plane_mesh(self, plane: ObservationPlane):
        signature = _field_geometry_signature(self._field_generation, plane, "off", smooth=True)
        key = ("exterior", plane.id)
        cached = self._field_cache.get(key)
        if cached is not None and cached[0] == signature:
            return cached[1]
        pv = __import__("pyvista")
        mesh = _sample_plane_polydata(pv, plane)
        self._field_cache[key] = (signature, mesh)
        return mesh

    def _exterior_boundary_mask(self, plane: ObservationPlane, plane_mesh, results) -> np.ndarray:
        signature = (
            _field_geometry_signature(self._field_generation, plane, results.traces.symmetry, smooth=True),
            results.run_id,
        )
        key = ("exterior-mask", plane.id)
        cached = self._field_cache.get(key)
        if cached is not None and cached[0] == signature:
            return cached[1]
        pv = __import__("pyvista")
        boundary_points, boundary_triangles = _expanded_boundary_geometry(
            results.traces.points_m,
            results.traces.triangles,
            results.traces.symmetry,
        )
        faces = np.column_stack(
            (np.full(boundary_triangles.shape[0], 3, dtype=np.int64), boundary_triangles)
        ).reshape(-1)
        boundary_mesh = pv.PolyData(boundary_points, faces)
        _cell_ids, closest_points = boundary_mesh.find_closest_cell(
            np.asarray(plane_mesh.points),
            return_closest_point=True,
        )
        distances = np.linalg.norm(np.asarray(plane_mesh.points) - np.asarray(closest_points), axis=1)
        mask = distances <= max(1.0e-7, plane.resolution_m * 0.25)
        self._field_cache[key] = (signature, mask)
        return mask

    def _smooth_field_mesh(
        self,
        plane: ObservationPlane,
        points: np.ndarray,
        tetrahedra: np.ndarray,
        symmetry: str,
    ) -> _SmoothFieldGeometry:
        signature = _field_geometry_signature(self._field_generation, plane, symmetry, smooth=True)
        key = ("smooth", plane.id)
        cached = self._field_cache.get(key)
        if cached is not None and cached[0] == signature:
            return cached[1]
        pv = __import__("pyvista")
        expanded_points, expanded_tetrahedra, source_node_ids = _expanded_fem_geometry(
            points,
            tetrahedra,
            symmetry,
        )
        volume = _fem_unstructured_grid(pv, expanded_points, expanded_tetrahedra)
        plane_mesh = _sample_plane_polydata(pv, plane)
        volume.cell_data["_observation_plane_source_tetrahedron"] = np.arange(
            expanded_tetrahedra.shape[0],
            dtype=np.int64,
        )
        sampled = plane_mesh.sample(volume)
        valid = np.asarray(sampled.point_data["vtkValidPointMask"], dtype=bool)
        faces = np.asarray(sampled.faces).reshape(-1, 5)[:, 1:]
        valid_cells = np.all(valid[faces], axis=1)
        if not np.any(valid_cells):
            raise ValueError("The observation plane does not intersect the bounded FEM region.")
        sampled = sampled.extract_cells(valid_cells)
        sampled_cell_ids = np.asarray(
            sampled.point_data["_observation_plane_source_tetrahedron"],
            dtype=np.int64,
        )
        for internal_name in (
            "_observation_plane_source_tetrahedron",
            "vtkValidPointMask",
            "vtkGhostType",
        ):
            if internal_name in sampled.point_data:
                del sampled.point_data[internal_name]
        expanded_node_ids = expanded_tetrahedra[sampled_cell_ids]
        interpolation_weights = _tetrahedral_interpolation_weights(
            np.asarray(sampled.points),
            expanded_points[expanded_node_ids],
        )
        sampled_source_node_ids = source_node_ids[expanded_node_ids]
        del volume.cell_data["_observation_plane_source_tetrahedron"]
        _axis_u, _axis_v, normal = plane.local_axes
        clipped = volume.clip(
            normal=normal,
            origin=np.asarray(plane.center_m),
            invert=plane.invert_clip_side,
        )
        result = _SmoothFieldGeometry(
            mesh=sampled,
            source_node_ids=sampled_source_node_ids,
            interpolation_weights=interpolation_weights,
            clipped=clipped,
        )
        self._field_cache[key] = (signature, result)
        return result

    def _element_field_mesh(
        self,
        plane: ObservationPlane,
        points: np.ndarray,
        tetrahedra: np.ndarray,
        symmetry: str,
    ) -> _ElementFieldGeometry:
        signature = _field_geometry_signature(self._field_generation, plane, symmetry, smooth=False)
        key = ("element", plane.id)
        cached = self._field_cache.get(key)
        if cached is not None and cached[0] == signature:
            return cached[1]
        pv = __import__("pyvista")
        expanded_points, expanded_tetrahedra, _source_node_ids = _expanded_fem_geometry(
            points,
            tetrahedra,
            symmetry,
        )
        volume = _fem_unstructured_grid(pv, expanded_points, expanded_tetrahedra)
        volume.cell_data["_observation_plane_source_tetrahedron"] = np.arange(
            expanded_tetrahedra.shape[0],
            dtype=np.int64,
        )
        _axis_u, _axis_v, normal = plane.local_axes
        clipped = volume.clip(
            normal=normal,
            origin=np.asarray(plane.center_m),
            invert=plane.invert_clip_side,
        )
        clipped_source_ids = np.asarray(
            clipped.cell_data["_observation_plane_source_tetrahedron"],
            dtype=np.int64,
        )
        del clipped.cell_data["_observation_plane_source_tetrahedron"]
        result = _ElementFieldGeometry(
            mesh=clipped,
            source_tetrahedron_ids=clipped_source_ids % np.asarray(tetrahedra).shape[0],
        )
        self._field_cache[key] = (signature, result)
        return result

    def _add_colored_field_actor(
        self,
        mesh,
        pressure: np.ndarray,
        plane: ObservationPlane,
        *,
        association: str,
        name: str,
    ) -> None:
        phase = self._animation_phase_deg if self._animation_plane_id == plane.id else None
        projection = project_field_scalars(pressure, plane.display, animation_phase_deg=phase)
        if association == "cell":
            mesh.cell_data[FIELD_SCALAR_NAME] = projection.values
        else:
            mesh.point_data[FIELD_SCALAR_NAME] = projection.values
        actor = self.viewer.add_mesh(
            mesh,
            name=name,
            scalars=FIELD_SCALAR_NAME,
            preference=association,
            cmap=projection.cmap,
            clim=projection.clim,
            opacity=0.92,
            show_edges=association == "cell",
            edge_color="#36464d",
            line_width=0.4,
            lighting=False,
            pickable=False,
            show_scalar_bar=True,
            scalar_bar_args={"title": projection.title, "vertical": True},
            render=False,
        )
        self._active_field = _ActiveFieldState(
            plane_id=plane.id,
            mesh=mesh,
            pressure=np.asarray(pressure),
            association=association,
            actor=actor,
        )
        self._scalar_bar_titles.add(projection.title)
        self._actor_names.add(name)

    def _update_active_field(self, plane: ObservationPlane | None) -> bool:
        """Refresh frequency-dependent scalars without replacing scene actors."""

        state = self._active_field
        if plane is None or state is None or state.plane_id != plane.id:
            return False
        try:
            mesh, pressure, association, _clipped = self._field_mesh_and_pressure(plane)
        except (KeyError, RuntimeError, TypeError, ValueError):
            return False
        if mesh is not state.mesh or association != state.association:
            return False
        phase = self._animation_phase_deg if self._animation_plane_id == plane.id else None
        projection = project_field_scalars(pressure, plane.display, animation_phase_deg=phase)
        if not _update_mesh_scalars(mesh, association, projection.values):
            return False
        state.pressure = np.asarray(pressure)
        mapper = state.actor.GetMapper() if state.actor is not None and hasattr(state.actor, "GetMapper") else None
        if mapper is not None:
            mapper.SetScalarRange(*projection.clim)
        self.viewer.render()
        return True

    def _advance_animation(self) -> None:
        plane = next(
            (candidate for candidate in self._planes if candidate.id == self._animation_plane_id),
            None,
        )
        if plane is None or self._animation_plane_id != plane.id or self._field_results is None:
            self.set_animation(None, False)
            return
        now = time.monotonic()
        elapsed = max(now - self._animation_updated_at, 0.0)
        self._animation_updated_at = now
        self._animation_phase_deg = (self._animation_phase_deg + 360.0 * plane.animation_speed_hz * elapsed) % 360.0
        if not self._update_animation_frame(plane):
            exterior_pending = plane.plane_type == ObservationPlaneType.EXTERIOR and any(
                len(key) > 2 and key[2] == plane.id for key in self._exterior_pending
            )
            if exterior_pending:
                return
            self.set_animation(None, False)

    def _update_animation_frame(self, plane: ObservationPlane) -> bool:
        """Update only the animated scalar array while preserving scene actors."""

        state = self._active_field
        if state is None or state.plane_id != plane.id:
            return False
        phase_rad = math.radians(self._animation_phase_deg)
        values = np.real(state.pressure) * math.cos(phase_rad) + np.imag(state.pressure) * math.sin(phase_rad)
        if not _update_mesh_scalars(state.mesh, state.association, values):
            return False
        self.viewer.render()
        return True

    def _add_help_text(self, plane: ObservationPlane) -> None:
        mode = f"\nMode: {self._mode.title()}" if self._mode else ""
        self.viewer.add_text(
            f"{plane.name}\nM - Move    R - Rotate    S - Size{mode}",
            position="upper_left",
            font_size=10,
            name=HELP_ACTOR_NAME,
            render=False,
        )
        self._actor_names.add(HELP_ACTOR_NAME)

    def _add_rotation_angle_text(self, angle_deg: float) -> None:
        self.viewer.add_text(
            _relative_rotation_text(angle_deg),
            position="upper_edge",
            font_size=12,
            name=ANGLE_ACTOR_NAME,
            render=False,
        )
        self._actor_names.add(ANGLE_ACTOR_NAME)

    def _add_move_gizmo(self, plane: ObservationPlane) -> None:
        center = np.asarray(plane.center_m)
        scale = self._tool_scale(plane)
        for axis_index, (axis, color) in enumerate(zip(plane.local_axes, AXIS_COLORS, strict=True)):
            mesh = __import__("pyvista").Arrow(
                start=center,
                direction=axis,
                scale=scale,
                tip_radius=0.12,
                shaft_radius=0.035,
            )
            self._add_tool_actor(mesh, "move", axis_index, color=color)

    def _add_rotation_gizmo(self, plane: ObservationPlane) -> None:
        pv = __import__("pyvista")
        center = np.asarray(plane.center_m)
        radius = self._tool_scale(plane) * 0.8
        axes = plane.local_axes
        for axis_index, (axis, color) in enumerate(zip(axes, AXIS_COLORS, strict=True)):
            polar = axes[(axis_index + 1) % 3] * radius
            ring = pv.CircularArcFromNormal(center=center, normal=axis, polar=polar, angle=360.0, resolution=120)
            self._add_tool_actor(
                ring,
                "rotate",
                axis_index,
                color=color,
                line_width=5.0,
                render_lines_as_tubes=True,
            )

    def _add_size_handles(self, plane: ObservationPlane) -> None:
        pv = __import__("pyvista")
        radius = self._tool_scale(plane) * 0.075
        for corner_index, corner in enumerate(plane.corners_m):
            handle = pv.Sphere(radius=radius, center=corner, theta_resolution=16, phi_resolution=12)
            self._add_tool_actor(handle, "size", corner_index, color=SELECTED_EDGE_COLOR)

    def _add_tool_actor(self, mesh, mode: str, control: int, **options) -> None:
        name = f"observation-plane:tool:{mode}:{control}"
        actor = self.viewer.add_mesh(mesh, name=name, pickable=True, render=False, **options)
        if self._foreground_renderer is not None:
            self.viewer.renderer.RemoveActor(actor)
            self._foreground_renderer.AddActor(actor)
            self._tool_actors.append(actor)
        self._actor_names.add(name)
        self._tool_actor_controls[_vtk_actor_address(actor)] = (mode, control)

    def _tool_scale(self, plane: ObservationPlane) -> float:
        return max(min(plane.width_m, plane.height_m) * 0.4, self._default_size() * 0.15, 0.005)

    def _remove_actors(self) -> None:
        for title in tuple(self._scalar_bar_titles):
            try:
                self.viewer.remove_scalar_bar(title=title, render=False)
            except (AttributeError, KeyError, TypeError):
                pass
        self._scalar_bar_titles.clear()
        if self._foreground_renderer is not None:
            for actor in self._tool_actors:
                self._foreground_renderer.RemoveActor(actor)
        self._tool_actors.clear()
        for name in tuple(self._actor_names):
            try:
                self.viewer.remove_actor(name, render=False)
            except Exception:
                pass
        self._actor_names.clear()

    def _picked_plane_id(self, position) -> str | None:
        return self._plane_actor_ids.get(_vtk_actor_address(self._pick_actor(position)))

    def _pick_actor(self, position):
        if self._foreground_renderer is not None:
            self._picker.Pick(*self._vtk_xy(position), 0, self._foreground_renderer)
            actor = self._picker.GetActor()
            if actor is not None:
                return actor
        self._picker.Pick(*self._vtk_xy(position), 0, self.viewer.renderer)
        return self._picker.GetActor()

    def _create_foreground_renderer(self):
        render_window = getattr(self.viewer, "render_window", None)
        if render_window is None:
            render_window = getattr(self.viewer, "ren_win", None)
        renderer = getattr(self.viewer, "renderer", None)
        if render_window is None or renderer is None:
            return None
        foreground = self.vtk.vtkRenderer()
        foreground.SetLayer(1)
        foreground.SetInteractive(False)
        foreground.SetPreserveColorBuffer(True)
        foreground.SetPreserveDepthBuffer(False)
        foreground.SetActiveCamera(renderer.GetActiveCamera())
        render_window.SetNumberOfLayers(max(int(render_window.GetNumberOfLayers()), 2))
        render_window.AddRenderer(foreground)
        return foreground

    def _vtk_xy(self, position) -> tuple[int, int]:
        ratio = float(self.viewer.devicePixelRatioF())
        x_pos = int(round(float(position.x()) * ratio))
        y_pos = int(round((float(self.viewer.height()) - float(position.y()) - 1.0) * ratio))
        return x_pos, y_pos

    def _display_ray(self, position) -> tuple[np.ndarray, np.ndarray] | None:
        renderer = self.viewer.renderer
        x_pos, y_pos = self._vtk_xy(position)
        points = []
        for depth in (0.0, 1.0):
            renderer.SetDisplayPoint(x_pos, y_pos, depth)
            renderer.DisplayToWorld()
            homogeneous = np.asarray(renderer.GetWorldPoint(), dtype=float)
            if abs(homogeneous[3]) <= 1e-12:
                return None
            points.append(homogeneous[:3] / homogeneous[3])
        direction = points[1] - points[0]
        norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            return None
        return points[0], direction / norm

    def _axis_parameter(self, position, origin: np.ndarray, axis: np.ndarray) -> float | None:
        ray = self._display_ray(position)
        if ray is None:
            return None
        ray_origin, ray_direction = ray
        axis = _normalized(axis, fallback=(1.0, 0.0, 0.0))
        dot = float(np.dot(axis, ray_direction))
        denominator = 1.0 - dot * dot
        if denominator <= 1e-6:
            return None
        offset = origin - ray_origin
        axis_offset = float(np.dot(axis, offset))
        ray_offset = float(np.dot(ray_direction, offset))
        return (dot * ray_offset - axis_offset) / denominator

    def _ray_plane_intersection(
        self,
        position,
        plane_origin: np.ndarray,
        plane_normal: np.ndarray,
    ) -> np.ndarray | None:
        ray = self._display_ray(position)
        if ray is None:
            return None
        ray_origin, ray_direction = ray
        denominator = float(np.dot(ray_direction, plane_normal))
        if abs(denominator) <= 1e-7:
            return None
        distance = float(np.dot(plane_origin - ray_origin, plane_normal)) / denominator
        return ray_origin + ray_direction * distance

    def _world_per_display_pixel(self, world_point: np.ndarray) -> float:
        renderer = self.viewer.renderer
        renderer.SetWorldPoint(*world_point, 1.0)
        renderer.WorldToDisplay()
        display = np.asarray(renderer.GetDisplayPoint(), dtype=float)
        renderer.SetDisplayPoint(display[0] + 100.0, display[1], display[2])
        renderer.DisplayToWorld()
        shifted = np.asarray(renderer.GetWorldPoint(), dtype=float)
        if abs(shifted[3]) <= 1e-12:
            return max(self._default_size() / 300.0, 1e-6)
        shifted = shifted[:3] / shifted[3]
        scale = float(np.linalg.norm(shifted - world_point)) / 100.0
        return scale if np.isfinite(scale) and scale > 0.0 else max(self._default_size() / 300.0, 1e-6)

    def _display_to_world_at_depth(self, position, reference_world: np.ndarray) -> np.ndarray | None:
        renderer = self.viewer.renderer
        renderer.SetWorldPoint(*reference_world, 1.0)
        renderer.WorldToDisplay()
        depth = float(renderer.GetDisplayPoint()[2])
        x_pos, y_pos = self._vtk_xy(position)
        renderer.SetDisplayPoint(x_pos, y_pos, depth)
        renderer.DisplayToWorld()
        homogeneous = np.asarray(renderer.GetWorldPoint(), dtype=float)
        if abs(homogeneous[3]) <= 1e-12:
            return None
        return homogeneous[:3] / homogeneous[3]

    def _rotation_screen_direction(self, center: np.ndarray, axis: np.ndarray, scale: float) -> np.ndarray:
        renderer = self.viewer.renderer
        display_points = []
        for point in (center, center + axis * scale):
            renderer.SetWorldPoint(*point, 1.0)
            renderer.WorldToDisplay()
            display_points.append(np.asarray(renderer.GetDisplayPoint()[:2], dtype=float))
        projected_axis = display_points[1] - display_points[0]
        projected_axis[1] *= -1.0  # VTK display Y is opposite Qt widget Y.
        norm = float(np.linalg.norm(projected_axis))
        if norm <= 1e-6:
            return np.asarray((1.0, 0.0))
        projected_axis /= norm
        return np.asarray((-projected_axis[1], projected_axis[0]))


def _plane_polydata(plane: ObservationPlane):
    pv = __import__("pyvista")
    corners = plane.corners_m
    faces = np.asarray((4, 0, 1, 2, 3), dtype=np.int64)
    return pv.PolyData(corners, faces)


def _normalized(values: np.ndarray, *, fallback: tuple[float, float, float]) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 1e-12:
        return np.asarray(fallback, dtype=float)
    return np.asarray(values, dtype=float) / norm


def _position_array(position) -> np.ndarray:
    return np.asarray((float(position.x()), float(position.y())), dtype=float)


def _relative_rotation_text(angle_deg: float) -> str:
    return f"Relative rotation: {float(angle_deg):+.1f}°"


def _result_selection_update(
    previous: tuple[ObservationPlane, ...],
    current: tuple[ObservationPlane, ...],
    previous_selected_id: str | None,
    selected_id: str | None,
    active_id: str | None,
) -> str | None:
    """Classify a plane update that can preserve the existing viewport actors."""

    if previous_selected_id != selected_id or len(previous) != len(current):
        return None
    field_plane_id = active_id or selected_id
    if field_plane_id is None:
        return "state_only" if previous == current else None
    field_changed = False
    state_changed = False
    for old, new in zip(previous, current, strict=True):
        if old.id != new.id:
            return None
        if old.id != field_plane_id:
            if old != new:
                return None
            continue
        comparable = replace(
            new,
            frequency_hz=old.frequency_hz,
            response_id=old.response_id,
            animation_speed_hz=old.animation_speed_hz,
        )
        if comparable != old:
            return None
        field_changed = old.frequency_hz != new.frequency_hz or old.response_id != new.response_id
        state_changed = old.animation_speed_hz != new.animation_speed_hz
    if field_changed:
        return "field"
    if state_changed or previous == current:
        return "state_only"
    return None


def _field_geometry_signature(
    generation: int,
    plane: ObservationPlane,
    symmetry: str,
    *,
    smooth: bool,
) -> tuple[object, ...]:
    return (
        generation,
        str(symmetry),
        plane.center_m,
        plane.orientation_wxyz,
        plane.width_m,
        plane.height_m,
        plane.resolution_m if smooth else None,
        plane.invert_clip_side,
    )


def _observation_field_results_signature(results: ObservationFieldResults | None) -> tuple[object, ...] | None:
    if results is None:
        return None

    def field_key(field) -> tuple[object, ...] | None:
        if field is None:
            return None
        return (
            field.run_id,
            field.channel_names_by_excitation,
            tuple(repr(config) for config in field.channel_configs),
            field.flat_target_enabled,
            field.flat_target_reference_angle_deg,
        )

    return field_key(results.interior), field_key(results.exterior)


def _exterior_field_key(
    generation: int,
    run_id: str,
    plane: ObservationPlane,
    frequency_hz: float,
) -> tuple[object, ...]:
    return (
        generation,
        run_id,
        plane.id,
        plane.center_m,
        plane.orientation_wxyz,
        plane.width_m,
        plane.height_m,
        plane.resolution_m,
        float(frequency_hz),
        plane.response_id,
    )


def _expanded_fem_geometry(
    points: np.ndarray,
    tetrahedra: np.ndarray,
    symmetry: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int64)
    signs = [(1.0, 1.0, 1.0)]
    if symmetry in {"x", "xy"}:
        signs.append((-1.0, 1.0, 1.0))
    if symmetry == "xy":
        signs.extend(((1.0, -1.0, 1.0), (-1.0, -1.0, 1.0)))
    point_blocks = []
    tetrahedron_blocks = []
    for image_index, image_signs in enumerate(signs):
        point_blocks.append(points * np.asarray(image_signs, dtype=float))
        tetrahedron_blocks.append(tetrahedra + image_index * points.shape[0])
    return (
        np.vstack(point_blocks),
        np.vstack(tetrahedron_blocks),
        np.tile(np.arange(points.shape[0], dtype=np.int64), len(signs)),
    )


def _expanded_boundary_geometry(
    points: np.ndarray,
    triangles: np.ndarray,
    symmetry: str,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    signs = [(1.0, 1.0, 1.0)]
    if symmetry in {"x", "xy"}:
        signs.append((-1.0, 1.0, 1.0))
    if symmetry == "xy":
        signs.extend(((1.0, -1.0, 1.0), (-1.0, -1.0, 1.0)))
    return (
        np.vstack([points * np.asarray(image_signs, dtype=float) for image_signs in signs]),
        np.vstack([triangles + index * points.shape[0] for index, _signs in enumerate(signs)]),
    )


def _expanded_fem_field(
    points: np.ndarray,
    tetrahedra: np.ndarray,
    pressure: np.ndarray,
    symmetry: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expanded_points, expanded_tetrahedra, source_node_ids = _expanded_fem_geometry(
        points,
        tetrahedra,
        symmetry,
    )
    return expanded_points, expanded_tetrahedra, np.asarray(pressure)[source_node_ids]


def _fem_unstructured_grid(
    pv,
    points: np.ndarray,
    tetrahedra: np.ndarray,
    pressure: np.ndarray | None = None,
):
    cells = np.column_stack((np.full(tetrahedra.shape[0], 4, dtype=np.int64), tetrahedra)).reshape(-1)
    cell_types = np.full(tetrahedra.shape[0], int(pv.CellType.TETRA), dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    if pressure is not None:
        grid.point_data["pressure_real"] = np.real(pressure)
        grid.point_data["pressure_imag"] = np.imag(pressure)
    return grid


def _sample_plane_polydata(pv, plane: ObservationPlane):
    x_count, y_count = plane.sample_shape
    axis_u, axis_v, _normal = plane.local_axes
    u_values = np.linspace(-plane.width_m * 0.5, plane.width_m * 0.5, x_count)
    v_values = np.linspace(-plane.height_m * 0.5, plane.height_m * 0.5, y_count)
    center = np.asarray(plane.center_m, dtype=float)
    u_grid, v_grid = np.meshgrid(u_values, v_values)
    points = center + u_grid.reshape(-1, 1) * axis_u + v_grid.reshape(-1, 1) * axis_v
    lower_left = (
        np.arange(y_count - 1, dtype=np.int64)[:, np.newaxis] * x_count + np.arange(x_count - 1, dtype=np.int64)
    ).reshape(-1)
    faces = np.column_stack(
        (
            np.full(lower_left.size, 4, dtype=np.int64),
            lower_left,
            lower_left + 1,
            lower_left + x_count + 1,
            lower_left + x_count,
        )
    ).reshape(-1)
    return pv.PolyData(points, faces)


def _tetrahedral_interpolation_weights(sample_points: np.ndarray, tetrahedron_points: np.ndarray) -> np.ndarray:
    origins = tetrahedron_points[:, 0]
    matrices = np.stack(
        (
            tetrahedron_points[:, 1] - origins,
            tetrahedron_points[:, 2] - origins,
            tetrahedron_points[:, 3] - origins,
        ),
        axis=2,
    )
    coefficients = np.linalg.solve(matrices, (sample_points - origins)[..., np.newaxis])[..., 0]
    return np.column_stack((1.0 - np.sum(coefficients, axis=1), coefficients))


def _update_mesh_scalars(mesh, association: str, values: np.ndarray) -> bool:
    attributes = mesh.cell_data if association == "cell" else mesh.point_data
    current = attributes.get(FIELD_SCALAR_NAME)
    if current is None or np.shape(current) != np.shape(values):
        return False
    np.copyto(current, values, casting="unsafe")
    vtk_array = getattr(current, "VTKObject", None)
    if vtk_array is not None:
        vtk_array.Modified()
    if hasattr(mesh, "Modified"):
        mesh.Modified()
    return True


def _camera_position_snapshot(viewer):
    try:
        position = viewer.camera_position
        return tuple(tuple(float(value) for value in point) for point in position)
    except Exception:
        return None


def _vtk_actor_address(actor) -> str:
    if actor is None:
        return ""
    if hasattr(actor, "GetAddressAsString"):
        return actor.GetAddressAsString("")
    return str(id(actor))
