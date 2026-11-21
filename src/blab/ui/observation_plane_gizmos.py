"""Classic observation-plane gizmo geometry and screen-space rotation radar."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyvista as pv
import vtk

from blab.observation_planes import ObservationPlane

# Distinct radial bands leave room for unsnapped rotation between the marks.
ROTATION_SNAP_TOLERANCE_PX = 10.0
ROTATION_RADIUS_PX = 168.0
ROTATION_GRIP_SCALE = 1.25

SNAP_BANDS = ((45, 0.38, 0.61), (10, 0.73, 0.86), (5, 1.02, 1.12))


def radar_ticks(center, basis, radius):
    """Return screen segments and (angle, increment) metadata in matching order."""
    segments, metadata = [], []
    for step, inner, outer in SNAP_BANDS:
        for angle in range(0, 360, step):
            radians = np.deg2rad(angle)
            direction = np.asarray(basis) @ np.array([np.cos(radians), np.sin(radians)])
            segments.append([center + radius * inner * direction, center + radius * outer * direction])
            metadata.append((angle, step))
    return np.asarray(segments), metadata


def hit_tick(point, segments, tolerance=ROTATION_SNAP_TOLERANCE_PX):
    """Hit only a visible tick segment, not its radial band or infinite line."""
    start, end = segments[:, 0], segments[:, 1]
    delta = end - start
    length_squared = np.einsum("ij,ij->i", delta, delta)
    fraction = np.clip(np.einsum("ij,ij->i", point - start, delta) / np.maximum(length_squared, 1e-12), 0, 1)
    closest = start + fraction[:, None] * delta
    distances = np.linalg.norm(closest - point, axis=1)
    index = int(np.argmin(distances))
    return index if distances[index] <= tolerance else None


def rotation_readout(angle: float) -> str:
    return f"{angle:+.1f}\N{DEGREE SIGN}"


def unwrap_angle(previous, wrapped):
    """Choose the equivalent angle nearest the previous unwrapped angle."""
    return previous + (wrapped - previous + 180.0) % 360.0 - 180.0


@dataclass
class RadarDrag:
    original: ObservationPlane
    axis: int
    polar: np.ndarray
    tangent: np.ndarray
    face_on: bool
    grab_offset: np.ndarray = field(default_factory=lambda: np.zeros(2))
    free_angle: float = 0.0
    angle: float = 0.0
    tick: int | None = None
    step: int | None = None


class RotationRadar:
    """2D actors keep widths, hit targets and dial radius stable under zoom."""

    def __init__(self, viewer, renderer):
        self.viewer = viewer
        self.renderer = renderer
        self.drag: RadarDrag | None = None
        self.radius = ROTATION_RADIUS_PX
        self.snap_tolerance = ROTATION_SNAP_TOLERANCE_PX
        self.colors = ("#d95b52", "#86b568", "#729ddd")
        self._draw_signature = None
        self._dark = True
        self._actors = {}
        self._labels = {}
        self._grips = {}
        self.hovered_axis: int | None = None

    def _center(self, plane):
        renderer = self.viewer.renderer
        renderer.SetWorldPoint(*plane.center_m, 1.0)
        renderer.WorldToDisplay()
        return np.asarray(renderer.GetDisplayPoint()[:2]) / self.viewer.devicePixelRatioF()

    def _basis(self, polar, tangent):
        camera = self.viewer.camera
        forward = np.asarray(camera.GetDirectionOfProjection())
        right = np.cross(forward, camera.GetViewUp())
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        return np.array([[np.dot(right, polar), np.dot(right, tangent)], [np.dot(up, polar), np.dot(up, tangent)]])

    def _frame(self, plane, axis):
        axes = plane.local_axes
        polar = axes[(axis + 1) % 3]
        tangent = np.cross(axes[axis], polar)
        basis = self._basis(polar, tangent)
        face_on = abs(np.linalg.det(basis)) < 0.22
        if face_on:
            # Use another cardinal spoke if the usual one points into the screen.
            # The circular preview shares its zero endpoint with the idle ellipse.
            polar = basis[:, 0] if np.linalg.norm(basis[:, 0]) >= 0.7 else -basis[:, 1]
            tangent = np.array([-polar[1], polar[0]])
            if np.linalg.det(basis) < 0:
                tangent = -tangent
        return RadarDrag(plane, axis, polar, tangent, face_on)

    def begin(self, plane, axis, point):
        self.hovered_axis = None
        self.drag = self._frame(plane, axis)
        center, basis, _, _ = self.geometry()
        self.drag.grab_offset = center + self.radius * basis[:, 0] - point

    def geometry(self, drag=None):
        if drag is None:
            drag = self.drag
        basis = (
            np.column_stack((drag.polar[:2], drag.tangent[:2]))
            if drag.face_on
            else self._basis(drag.polar, drag.tangent)
        )
        center = self._center(drag.original)
        segments, metadata = radar_ticks(center, basis, self.radius)
        return center, basis, segments, metadata

    def angle_at(self, point):
        center, basis, segments, metadata = self.geometry()
        grip_point = point + self.drag.grab_offset
        local = np.linalg.solve(basis, grip_point - center)
        if np.linalg.norm(local) < 4.0:
            self.drag.tick = None
            self.drag.step = None
            self.drag.angle = self.drag.free_angle
            return self.drag.angle
        raw = np.rad2deg(np.arctan2(local[1], local[0]))
        drag = self.drag
        drag.free_angle = unwrap_angle(drag.free_angle, raw)
        # Keep the press at zero even if an arrowhead overlaps a neighboring tick.
        at_start = np.linalg.norm(grip_point - (center + self.radius * basis[:, 0])) < 1
        drag.tick = None if at_start else hit_tick(point, segments, self.snap_tolerance)
        drag.step = None
        drag.angle = drag.free_angle
        if drag.tick is not None:
            angle, drag.step = metadata[drag.tick]
            drag.angle = unwrap_angle(drag.free_angle, angle)
        return drag.angle

    def pick(self, point):
        hits = []
        for axis, (center, tangent) in self._grips.items():
            delta = (point - center) / ROTATION_GRIP_SCALE
            nearest = np.clip(delta @ tangent, -20, 20) * tangent
            if np.linalg.norm(delta - nearest) <= 8:
                hits.append((np.linalg.norm(delta), axis))
        return min(hits)[1] if hits else None

    def _grip(self, axis, center, tangent, color, *, on_dial=False):
        tangent = tangent / np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        self._grips[axis] = (center, tangent)
        # The square and both tangential arrowheads share one logical-pixel frame.
        vertices = np.array(
            [
                [-6, -6],
                [6, -6],
                [6, 6],
                [-6, 6],
                [-9, -6],
                [-22, 0],
                [-9, 6],
                [9, -6],
                [22, 0],
                [9, 6],
            ]
        )
        points = center + ROTATION_GRIP_SCALE * (vertices @ np.stack((tangent, normal)))
        name = "dial-grip" if on_dial else f"grip-{axis}"
        self._mesh(name, points, [4, 0, 1, 2, 3, 3, 4, 5, 6, 3, 7, 8, 9], color, fill=True)

    def _mesh(self, name, points, cells, color, *, width=1.0, fill=False, opacity=1.0):
        if name not in self._actors:
            mapper = vtk.vtkPolyDataMapper2D()
            actor = vtk.vtkActor2D()
            actor.SetMapper(mapper)
            actor.PickableOff()
            self.renderer.AddViewProp(actor)
            self._actors[name] = actor
        actor = self._actors[name]
        ratio = self.viewer.devicePixelRatioF()
        xyz = np.column_stack((np.asarray(points) * ratio, np.zeros(len(points))))
        mesh = pv.PolyData(xyz, faces=cells) if fill else pv.PolyData(xyz, lines=cells)
        actor.GetMapper().SetInputData(mesh)
        actor.GetProperty().SetColor(pv.Color(color).float_rgb)
        actor.GetProperty().SetLineWidth(width * ratio)
        actor.GetProperty().SetOpacity(opacity)
        actor.VisibilityOn()

    def _lines(self, name, segments, color, width=1.0):
        points = np.asarray(segments).reshape(-1, 2)
        cells = np.column_stack((np.full(len(segments), 2), np.arange(len(segments) * 2).reshape(-1, 2))).ravel()
        self._mesh(name, points, cells, color, width=width)

    def _path(self, name, points, color, width=1.0):
        self._lines(name, np.stack((points[:-1], points[1:]), axis=1), color, width)

    def _text(self, name, text, point, color, size=12):
        if name not in self._labels:
            actor = vtk.vtkTextActor()
            actor.PickableOff()
            self.renderer.AddViewProp(actor)
            self._labels[name] = actor
        actor = self._labels[name]
        ratio = self.viewer.devicePixelRatioF()
        actor.SetInput(text)
        actor.SetDisplayPosition(*[int(v * ratio) for v in point])
        prop = actor.GetTextProperty()
        prop.SetFontFamilyToArial()
        prop.SetFontSize(round(size * ratio))
        prop.SetColor(pv.Color(color).float_rgb)
        prop.SetBackgroundColor(pv.Color("#293134" if self._dark else "#ffffff").float_rgb)
        prop.SetBackgroundOpacity(0.9)
        actor.VisibilityOn()

    def draw(self, plane, *, active, dragging, radius=ROTATION_RADIUS_PX, colors, dark=True):
        drag = self.drag
        signature = (
            plane,
            active,
            dragging,
            self.hovered_axis,
            radius,
            colors,
            dark,
            self.viewer.renderer.GetActiveCamera().GetMTime(),
            tuple(self.viewer.render_window.GetSize()),
            self.viewer.devicePixelRatioF(),
            None if drag is None else (id(drag), drag.angle, drag.tick),
        )
        if signature == self._draw_signature:
            return
        self._draw_signature = signature
        for actor in (*self._actors.values(), *self._labels.values()):
            actor.VisibilityOff()
        self._grips.clear()
        self.radius = radius
        self._dark = dark
        self.colors = colors
        if not dragging:
            self.drag = None
        if not active or plane is None:
            self.hovered_axis = None
            return
        center = self._center(plane)
        angles = np.linspace(0, 2 * np.pi, 181)
        circle = np.column_stack((np.cos(angles), np.sin(angles)))
        drag = self.drag
        if drag is None and self.hovered_axis is not None:
            drag = self._frame(plane, self.hovered_axis)
        if drag is None:
            axes = plane.local_axes
            grips = []
            for axis, color in enumerate(colors):
                polar = axes[(axis + 1) % 3]
                tangent = np.cross(axes[axis], polar)
                basis = self._basis(polar, tangent)
                points = center + radius * (circle @ basis.T)
                self._path(f"outline-{axis}", points, "#101719", 5)
                self._path(f"ring-{axis}", points, color, 2)
                frame = self._frame(plane, axis)
                spoke = frame.polar if frame.face_on else basis[:, 0]
                direction = frame.tangent if frame.face_on else basis[:, 1]
                endpoint = center + radius * spoke
                self._lines(f"spoke-{axis}", [[center, endpoint]], color, 1.5)
                grips.append((axis, endpoint, direction, color))
            # Add handles last so intersecting rings cannot cover the click targets.
            for grip in grips:
                self._grip(*grip)
            return
        center, basis, segments, metadata = self.geometry(drag)
        color = colors[drag.axis]
        sweep = np.deg2rad(np.linspace(0, np.clip(drag.angle, -360, 360), max(2, int(min(abs(drag.angle), 360)) + 1)))
        sector = np.vstack((center, center + radius * np.column_stack((np.cos(sweep), np.sin(sweep))) @ basis.T))
        self._mesh("sector", sector, np.r_[len(sector), np.arange(len(sector))], color, fill=True, opacity=0.16)
        self._path("dial-shadow", center + radius * (circle @ basis.T), "#101719", 4)
        self._path("dial", center + radius * (circle @ basis.T), color, 2)
        self._lines("tick-shadow", segments, "#101719" if dark else "#ffffff", 3)
        self._lines("ticks", segments, ("#dedbd2" if dark else "#404850"), 1)
        self._lines("quarter-turns", segments[:8:2], ("#ffffff" if dark else "#20272d"), 3)
        if drag.tick is not None:
            self._lines("active-tick", segments[drag.tick : drag.tick + 1], color, 4)
        theta = np.deg2rad(drag.angle)
        endpoint = center + radius * basis @ np.array([np.cos(theta), np.sin(theta)])
        zero = center + radius * basis[:, 0]
        self._lines("zero", [[center, zero]], "#899399", 1)
        self._lines("needle-shadow", [[center, endpoint]], "#101719", 4)
        self._lines("needle", [[center, endpoint]], color, 2)
        self._grip(drag.axis, endpoint, basis @ np.array([-np.sin(theta), np.cos(theta)]), color, on_dial=True)
        self._text(
            "angle",
            f"{'XYZ'[drag.axis]}  {rotation_readout(drag.angle)}",
            endpoint + np.array([0, 16]),
            ("#e8e8e8" if dark else "#20272d"),
            13,
        )
        self._labels["angle"].GetTextProperty().SetJustificationToCentered()

    def close(self):
        for actor in (*self._actors.values(), *self._labels.values()):
            self.renderer.RemoveViewProp(actor)
        self._actors.clear()
        self._labels.clear()
        self._draw_signature = None


def colored_mesh(mesh, color):
    mesh.cell_data["tool_rgb"] = np.tile(np.asarray(pv.Color(color).int_rgb, dtype=np.uint8), (mesh.n_cells, 1))
    return mesh


def square_grip(face_color, center_color):
    """Two-sided square with raised rim and inset center; local XY footprint."""
    points = []
    faces = []
    colors = []
    for side in (1, -1):
        offset = len(points)
        for extent, depth in ((1.0, 0.0), (0.76, 0.20), (0.40, 0.20), (0.32, 0.06)):
            points.extend((x * extent, y * extent, side * depth) for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1)))
        for ring in range(3):
            for edge in range(4):
                nxt = (edge + 1) % 4
                faces.extend(
                    (
                        4,
                        offset + ring * 4 + edge,
                        offset + ring * 4 + nxt,
                        offset + (ring + 1) * 4 + nxt,
                        offset + (ring + 1) * 4 + edge,
                    )
                )
                shade = face_color if ring == 1 else ("#edf0f2" if (edge in (1, 2)) == (ring == 0) else "#555e65")
                colors.append(pv.Color(shade).int_rgb)
        faces.extend((4, offset + 12, offset + 13, offset + 14, offset + 15))
        colors.append(pv.Color(center_color).int_rgb)
    mesh = pv.PolyData(np.asarray(points, dtype=float), faces)
    mesh.cell_data["tool_rgb"] = np.asarray(colors, dtype=np.uint8)
    return mesh
