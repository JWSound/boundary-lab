"""Classic observation-plane gizmo geometry and screen-space rotation radar."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyvista as pv
import vtk

from blab.observation_planes import ObservationPlane

# Distinct radial bands leave room for unsnapped rotation between the marks.
ROTATION_SNAP_TOLERANCE_PX = 10.0
ROTATION_RADIUS_PX = 168.0

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
        self._ring_points = []

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

    def begin(self, plane, axis, point):
        axes = plane.local_axes
        polar = axes[(axis + 1) % 3]
        tangent = np.cross(axes[axis], polar)
        basis = self._basis(polar, tangent)
        face_on = abs(np.linalg.det(basis)) < 0.22
        coordinates = point - self._center(plane)
        if not face_on:
            coordinates = np.linalg.solve(basis, coordinates)
        theta = np.arctan2(coordinates[1], coordinates[0])
        if face_on:
            # Store a screen-space starting angle in the polar vector.
            polar = np.array([np.cos(theta), np.sin(theta), 0.0])
            tangent = np.array([-np.sin(theta), np.cos(theta), 0.0])
        else:
            polar, tangent = (
                np.cos(theta) * polar + np.sin(theta) * tangent,
                -np.sin(theta) * polar + np.cos(theta) * tangent,
            )
        self.drag = RadarDrag(plane, axis, polar, tangent, face_on)

    def geometry(self):
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
        local = np.linalg.solve(basis, point - center)
        if np.linalg.norm(local) < 4.0:
            self.drag.tick = None
            self.drag.step = None
            self.drag.angle = self.drag.free_angle
            return self.drag.angle
        raw = np.rad2deg(np.arctan2(local[1], local[0]))
        drag = self.drag
        drag.free_angle = unwrap_angle(drag.free_angle, raw)
        drag.tick = hit_tick(point, segments, self.snap_tolerance)
        drag.step = None
        drag.angle = drag.free_angle
        if drag.tick is not None:
            angle, drag.step = metadata[drag.tick]
            drag.angle = unwrap_angle(drag.free_angle, angle)
        return drag.angle

    def pick(self, point):
        best = None
        distance = 7.0
        for axis, points in enumerate(self._ring_points):
            segments = np.stack((points[:-1], points[1:]), axis=1)
            index = hit_tick(point, segments, distance)
            if index is not None:
                a, b = segments[index]
                delta = b - a
                t = np.clip(np.dot(point - a, delta) / max(np.dot(delta, delta), 1e-12), 0, 1)
                distance = np.linalg.norm(point - (a + t * delta))
                best = axis
        return best

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
        self._ring_points = []
        self.radius = radius
        self._dark = dark
        self.colors = colors
        if not dragging:
            self.drag = None
        if not active or plane is None:
            return
        center = self._center(plane)
        angles = np.linspace(0, 2 * np.pi, 181)
        circle = np.column_stack((np.cos(angles), np.sin(angles)))
        if self.drag is None:
            axes = plane.local_axes
            for axis, color in enumerate(colors):
                polar = axes[(axis + 1) % 3]
                tangent = np.cross(axes[axis], polar)
                basis = self._basis(polar, tangent)
                points = center + radius * (circle @ basis.T)
                self._ring_points.append(points)
                self._path(f"outline-{axis}", points, "#101719", 5)
                self._path(f"ring-{axis}", points, color, 2)
            return
        center, basis, segments, metadata = self.geometry()
        drag = self.drag
        color = colors[drag.axis]
        sweep = np.deg2rad(np.linspace(0, np.clip(drag.angle, -360, 360), max(2, int(min(abs(drag.angle), 360)) + 1)))
        sector = np.vstack((center, center + radius * np.column_stack((np.cos(sweep), np.sin(sweep))) @ basis.T))
        self._mesh("sector", sector, np.r_[len(sector), np.arange(len(sector))], color, fill=True, opacity=0.16)
        self._path("dial-shadow", center + radius * (circle @ basis.T), "#101719", 4)
        self._path("dial", center + radius * (circle @ basis.T), ("#c9cdd1" if dark else "#404850"), 1.5)
        self._lines("tick-shadow", segments, "#101719" if dark else "#ffffff", 3)
        self._lines("ticks", segments, ("#dedbd2" if dark else "#404850"), 1)
        if drag.tick is not None:
            self._lines("active-tick", segments[drag.tick : drag.tick + 1], color, 4)
        theta = np.deg2rad(drag.angle)
        endpoint = center + radius * basis @ np.array([np.cos(theta), np.sin(theta)])
        zero = center + radius * basis[:, 0]
        self._lines("zero", [[center, zero]], "#899399", 1)
        self._lines("needle-shadow", [[center, endpoint]], "#101719", 4)
        self._lines("needle", [[center, endpoint]], color, 2)
        self._text(
            "angle",
            rotation_readout(drag.angle),
            center + np.array([0, radius * 1.2 + 12]),
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
