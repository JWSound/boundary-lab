"""Qt/PyVista balloon plot viewer."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QSize, QTimer, Qt, Slot
from PySide6.QtGui import QColor, QFontMetrics, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QSplitter,
    QSpinBox,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from blab.balloon import BalloonPrepConfig, prepare_balloon_data
from blab.postprocess import _fractional_octave_smooth, _interpolate_isobar_heatmap
from blab.ui.plots import LIVE_ISOBAR_ANGLE_SAMPLES, LIVE_ISOBAR_FREQ_SAMPLES, IsobarCanvas


SPL_SCALAR_NAME = "Normalized SPL (dB)"
HORIZONTAL_ANGLE_SCALAR_NAME = "Horizontal Angle (deg)"
VERTICAL_ANGLE_SCALAR_NAME = "Vertical Angle (deg)"
CONTOUR_STEP_DB = 6.0
GUIDE_LINE_WIDTH = 3
CONTOUR_COLOR = "#f4f0e6"
LEGEND_TICKS_DB = (0.0, -6.0, -12.0, -18.0, -24.0, -30.0)
PROTRACTOR_ANGLES_DEG = (30.0, 60.0, 90.0, 120.0, 150.0)
PROTRACTOR_COLOR = "#d8dee9"
PROTRACTOR_AXIS_COLOR = "#ffffff"


class BalloonPlotWindow(QDialog):
    def __init__(
        self,
        raw_balloon_data: dict[str, np.ndarray],
        *,
        min_db: float,
        max_db: float,
        polar_smoothing: int | float | None = 24,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Balloon Plot")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        self.resize(1100, 760)

        try:
            import pyvista as pv
            import vtk
            from pyvistaqt import QtInteractor
        except ImportError as exc:
            raise RuntimeError("Install the GUI extras with pyvista and pyvistaqt to use the balloon plot viewer.") from exc

        self._pv = pv
        self._vtk = vtk
        self._raw_balloon_data = raw_balloon_data
        self._prepared = None
        self._min_db = float(min_db)
        self._max_db = float(max_db)
        self._polar_smoothing = polar_smoothing
        self._mesh_actor = None
        self._balloon_mesh = None
        self._protractor_actors = []
        self._protractor_radius = 1.0
        self._hover_picker = None
        self._hover_observer = None

        self.plotter = QtInteractor(self)
        self.plotter.set_background("#111316")
        self._install_hover_picker()

        self.frequency_combo = QComboBox()
        self.frequency_combo.setEnabled(False)
        self.frequency_combo.currentIndexChanged.connect(self._on_frequency_changed)

        self.protractor_angle_slider = QSlider(Qt.Horizontal)
        self.protractor_angle_slider.setRange(-180, 180)
        self.protractor_angle_slider.setSingleStep(1)
        self.protractor_angle_slider.setPageStep(15)
        self.protractor_angle_slider.setValue(0)
        self.protractor_angle_slider.valueChanged.connect(self._on_protractor_angle_changed)
        self.protractor_angle_slider.sliderReleased.connect(self._render_isobar_slice_if_enabled)

        self.protractor_angle_spin = QSpinBox()
        self.protractor_angle_spin.setRange(-180, 180)
        self.protractor_angle_spin.setSuffix(" deg")
        self.protractor_angle_spin.setValue(0)
        self.protractor_angle_spin.valueChanged.connect(self._on_protractor_angle_changed)

        self.protractor_toggle = QCheckBox("Radar Slicer")
        self.protractor_toggle.setChecked(True)
        self.protractor_toggle.toggled.connect(self._on_protractor_toggled)

        self.slice_plot = IsobarCanvas("Isobar Angle Slice", left_margin=0.17, right_margin=0.92)
        self.slice_plot.setMinimumSize(330, 286)
        self.slice_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.radar_plot = SliceRadarCanvas(self._min_db, self._max_db)
        self.radar_plot.setMinimumSize(220, 286)
        self.radar_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.loading_label = QLabel("Rendering Balloon...")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setStyleSheet(
            "QLabel {"
            "color: white;"
            "background: rgba(17, 19, 22, 190);"
            "font-size: 22px;"
            "font-weight: 600;"
            "}"
        )

        viewport_stack = QStackedLayout()
        viewport_stack.setStackingMode(QStackedLayout.StackAll)
        viewport_stack.addWidget(self.plotter.interactor)
        viewport_stack.addWidget(self.loading_label)
        viewport = QWidget()
        viewport.setLayout(viewport_stack)

        self.hover_label = QLabel("")
        self.hover_label.setMinimumHeight(24)
        self.hover_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.hover_label.setStyleSheet(
            "QLabel {"
            "background: #111316;"
            "color: #e8e8e8;"
            "padding-left: 8px;"
            "padding-right: 8px;"
            "}"
        )

        viewport_container = QWidget()
        viewport_layout = QVBoxLayout(viewport_container)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        viewport_layout.setSpacing(0)
        viewport_layout.addWidget(viewport, stretch=1)
        viewport_layout.addWidget(self.hover_label)

        side_panel = QWidget()
        side_panel.setStyleSheet("QWidget { background: #1f1f1f; color: white; }")
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(12, 12, 12, 12)
        form = QFormLayout()
        form.addRow("Frequency", self.frequency_combo)
        form.addRow("", self.protractor_toggle)
        form.addRow("Slice Angle", self.protractor_angle_slider)
        form.addRow("", self.protractor_angle_spin)
        side_layout.addLayout(form)
        side_layout.addSpacing(14)
        legend_radar_layout = QHBoxLayout()
        legend_radar_layout.setContentsMargins(0, 0, 0, 0)
        legend_radar_layout.setSpacing(4)
        legend_radar_layout.addWidget(ColorLegend(self._min_db, self._max_db, side_panel))
        legend_radar_layout.addWidget(self.radar_plot, stretch=1)
        side_layout.addLayout(legend_radar_layout)
        side_layout.addWidget(self.slice_plot, stretch=1)
        side_layout.addStretch(1)
        side_panel.setMinimumWidth(430)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(viewport_container)
        splitter.addWidget(side_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([700, 460])

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        QTimer.singleShot(0, self._prepare_and_render_initial)

    @Slot()
    def _prepare_and_render_initial(self) -> None:
        self.loading_label.show()
        self._prepared = prepare_balloon_data(
            self._raw_balloon_data,
            BalloonPrepConfig(min_db=self._min_db, max_db=self._max_db),
        )
        self._min_db = float(self._prepared["min_db"])
        self._max_db = float(self._prepared["max_db"])

        self.frequency_combo.blockSignals(True)
        self.frequency_combo.clear()
        for freq in self._prepared["freq_hz"]:
            self.frequency_combo.addItem(_format_frequency(float(freq)), float(freq))
        self.frequency_combo.blockSignals(False)
        self.frequency_combo.setEnabled(True)
        self._set_protractor_controls_enabled(self.protractor_toggle.isChecked())

        self._render_frequency(0, reset_camera=True)
        self._render_isobar_slice_if_enabled()
        self.loading_label.hide()

    @Slot(int)
    def _on_frequency_changed(self, index: int) -> None:
        self._render_frequency(index, reset_camera=False)
        self._render_isobar_slice_if_enabled()

    def _render_frequency(self, index: int, *, reset_camera: bool) -> None:
        if index < 0 or self._prepared is None:
            return

        x = self._prepared["balloon_x"][index]
        y = self._prepared["balloon_y"][index]
        z = self._prepared["balloon_z"][index]
        spl = self._prepared["balloon_surface_spl"][index]
        mesh = self._pv.StructuredGrid(x, y, z)
        mesh[SPL_SCALAR_NAME] = spl.ravel(order="F")
        mesh[HORIZONTAL_ANGLE_SCALAR_NAME], mesh[VERTICAL_ANGLE_SCALAR_NAME] = _balloon_angle_arrays(
            self._prepared["theta_grid_rad"],
            self._prepared["phi_grid_rad"],
        )

        self.hover_label.setText("")
        self.plotter.clear()
        self._balloon_mesh = mesh
        self._mesh_actor = self.plotter.add_mesh(
            mesh,
            scalars=SPL_SCALAR_NAME,
            cmap="turbo",
            clim=(self._min_db, self._max_db),
            smooth_shading=True,
            show_scalar_bar=False,
        )
        self._add_spl_contours(mesh)
        self._add_orientation_guides(mesh)
        if self.protractor_toggle.isChecked():
            self._add_protractor(mesh)
        self.plotter.add_axes()
        self.plotter.enable_anti_aliasing()
        if reset_camera:
            self.plotter.reset_camera()
            self.plotter.camera_position = "iso"
        self.plotter.render()

    @Slot(int)
    def _on_protractor_angle_changed(self, angle_deg: int) -> None:
        sender = self.sender()
        angle = int(angle_deg)
        if sender is not self.protractor_angle_slider:
            self.protractor_angle_slider.blockSignals(True)
            self.protractor_angle_slider.setValue(angle)
            self.protractor_angle_slider.blockSignals(False)
        if sender is not self.protractor_angle_spin:
            self.protractor_angle_spin.blockSignals(True)
            self.protractor_angle_spin.setValue(angle)
            self.protractor_angle_spin.blockSignals(False)

        if self._balloon_mesh is None:
            return
        self._set_protractor_angle(angle)
        self.plotter.render()
        if sender is self.protractor_angle_spin or (
            sender is self.protractor_angle_slider and not self.protractor_angle_slider.isSliderDown()
        ):
            self._render_isobar_slice_if_enabled()

    @Slot(bool)
    def _on_protractor_toggled(self, enabled: bool) -> None:
        self._set_protractor_controls_enabled(enabled)
        self.radar_plot.setVisible(enabled)
        self.slice_plot.setVisible(enabled)
        if enabled and self._balloon_mesh is not None:
            self._add_protractor(self._balloon_mesh)
            self._render_isobar_slice()
        elif not enabled:
            self._remove_protractor()
        self.plotter.render()

    @Slot()
    def _render_isobar_slice(self) -> None:
        if self._prepared is None:
            return
        freqs_hz, angles_deg, values_db = _balloon_isobar_slice(
            self._prepared,
            float(self.protractor_angle_slider.value()),
            clip_min_db=self._min_db,
            angle_samples=LIVE_ISOBAR_ANGLE_SAMPLES,
            freq_samples=LIVE_ISOBAR_FREQ_SAMPLES,
            octave_smoothing=self._polar_smoothing,
        )
        self.slice_plot.update_plot(
            freqs_hz,
            angles_deg,
            values_db,
            self._min_db,
            self._max_db,
        )
        self._render_radar_slice()

    @Slot()
    def _render_isobar_slice_if_enabled(self) -> None:
        if self.protractor_toggle.isChecked():
            self._render_isobar_slice()

    def _render_radar_slice(self) -> None:
        if self._prepared is None:
            return
        frequency_index = self.frequency_combo.currentIndex()
        if frequency_index < 0:
            return
        angles_deg, values_db = _balloon_radar_slice(
            self._prepared,
            frequency_index,
            float(self.protractor_angle_slider.value()),
        )
        self.radar_plot.update_plot(angles_deg, values_db)

    def _set_protractor_controls_enabled(self, enabled: bool) -> None:
        self.protractor_angle_slider.setEnabled(enabled)
        self.protractor_angle_spin.setEnabled(enabled)

    def _install_hover_picker(self) -> None:
        self._hover_picker = self._vtk.vtkCellPicker()
        self._hover_picker.SetTolerance(0.0005)
        interactor = _plotter_interactor(self.plotter)
        if interactor is None:
            return

        if hasattr(interactor, "add_observer"):
            self._hover_observer = interactor.add_observer("MouseMoveEvent", self._on_mouse_move)
        elif hasattr(interactor, "AddObserver"):
            self._hover_observer = interactor.AddObserver("MouseMoveEvent", self._on_mouse_move)

    def _on_mouse_move(self, *args) -> None:
        if self._hover_picker is None or self._balloon_mesh is None or self._mesh_actor is None:
            self.hover_label.setText("")
            return

        interactor = args[0] if args and hasattr(args[0], "GetEventPosition") else _plotter_interactor(self.plotter)
        renderer = getattr(self.plotter, "renderer", None)
        if interactor is None or renderer is None:
            self.hover_label.setText("")
            return

        x_pos, y_pos = interactor.GetEventPosition()
        if not self._hover_picker.Pick(x_pos, y_pos, 0, renderer):
            self.hover_label.setText("")
            return

        if _vtk_actor_address(self._hover_picker.GetActor()) != _vtk_actor_address(self._mesh_actor):
            self.hover_label.setText("")
            return

        point_id = _picked_point_id(self._hover_picker, self._balloon_mesh)
        if point_id is None:
            self.hover_label.setText("")
            return

        self.hover_label.setText(_balloon_hover_text(self._balloon_mesh, point_id))

    def _add_spl_contours(self, mesh) -> None:
        levels = _contour_levels(self._min_db, self._max_db, CONTOUR_STEP_DB)
        if not levels:
            return

        contour_mesh = _offset_mesh_points(mesh, offset=max(_mesh_extent(mesh) * 0.002, 0.03))
        contours = contour_mesh.contour(isosurfaces=levels, scalars=SPL_SCALAR_NAME)
        if contours.n_points == 0:
            return

        tube_radius = max(_mesh_extent(mesh) * 0.0015, 0.02)
        self.plotter.add_mesh(
            contours.tube(radius=tube_radius),
            color=CONTOUR_COLOR,
            smooth_shading=True,
            show_scalar_bar=False,
        )

    def _add_orientation_guides(self, mesh) -> None:
        length = max(_mesh_extent(mesh) * 1.12, 1.0)
        pv = self._pv

        guide_specs = (
            ((-length, 0.0, 0.0), (length, 0.0, 0.0), "#e25d5d"),
            ((0.0, -length, 0.0), (0.0, length, 0.0), "#5da8e2"),
            ((0.0, 0.0, -length), (0.0, 0.0, length), "#f2d15f"),
        )
        for start, end, color in guide_specs:
            self.plotter.add_mesh(
                pv.Line(start, end),
                color=color,
                line_width=GUIDE_LINE_WIDTH,
                render_lines_as_tubes=True,
            )

        arrow_tip = np.array([0.0, 0.0, length], dtype=float)
        self.plotter.add_arrows(
            arrow_tip[np.newaxis, :],
            np.array([[0.0, 0.0, 1.0]], dtype=float),
            mag=length * 0.12,
            color="#f2d15f",
        )
        label_points = np.array(
            [
                [length * 1.06, 0.0, 0.0],
                [0.0, length * 1.06, 0.0],
                [0.0, 0.0, length * 1.16],
            ],
            dtype=float,
        )
        self.plotter.add_point_labels(
            label_points,
            ["Horizontal", "Vertical", "On Axis"],
            font_size=14,
            text_color="white",
            point_color="white",
            point_size=0,
            shape_opacity=0.35,
            always_visible=True,
        )

    def _add_protractor(self, mesh) -> None:
        self._remove_protractor()
        db_radius = max(_mesh_extent(mesh), 1.0)
        radius = db_radius * 1.04
        self._protractor_radius = radius
        tube_radius = max(radius * 0.002, 0.018)
        angle_deg = float(self.protractor_angle_slider.value())

        for points, color, width_scale in _protractor_line_specs(radius, db_radius, 0.0):
            line = self._pv.Spline(points, len(points)) if len(points) > 2 else self._pv.Line(points[0], points[-1])
            actor = self.plotter.add_mesh(
                line.tube(radius=tube_radius * width_scale),
                color=color,
                smooth_shading=True,
                opacity=0.78,
                show_scalar_bar=False,
            )
            self._protractor_actors.append(actor)
        self._set_protractor_angle(angle_deg)

    def _set_protractor_angle(self, angle_deg: float) -> None:
        for actor in self._protractor_actors:
            if hasattr(actor, "SetOrientation"):
                actor.SetOrientation(0.0, 0.0, float(angle_deg))

    def _remove_protractor(self) -> None:
        for actor in self._protractor_actors:
            try:
                self.plotter.remove_actor(actor, render=False)
            except Exception:
                pass
        self._protractor_actors = []

    def closeEvent(self, event) -> None:
        self.plotter.close()
        super().closeEvent(event)


def _format_frequency(freq_hz: float) -> str:
    if freq_hz >= 1000.0:
        return f"{_format_decimal(freq_hz / 1000.0)} kHz"
    return f"{_format_decimal(freq_hz)} Hz"


def _format_decimal(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def _balloon_angle_arrays(theta_rad: np.ndarray, phi_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta_rad, dtype=float)
    phi = np.asarray(phi_rad, dtype=float)
    direction_x = np.sin(theta) * np.cos(phi)
    direction_y = np.sin(theta) * np.sin(phi)
    direction_z = np.cos(theta)
    direction_x[np.isclose(direction_x, 0.0)] = 0.0
    direction_y[np.isclose(direction_y, 0.0)] = 0.0
    direction_z[np.isclose(direction_z, 0.0)] = 0.0
    horizontal = _normalize_signed_angle(np.rad2deg(np.arctan2(direction_x, direction_z)))
    vertical = _normalize_signed_angle(np.rad2deg(np.arctan2(direction_y, direction_z)))
    return horizontal.ravel(order="F").astype(np.float32), vertical.ravel(order="F").astype(np.float32)


def _normalize_signed_angle(angle_deg: np.ndarray) -> np.ndarray:
    angle = (np.asarray(angle_deg, dtype=float) + 180.0) % 360.0 - 180.0
    angle[np.isclose(angle, -180.0)] = 180.0
    angle[np.isclose(angle, 0.0)] = 0.0
    return angle


def _balloon_hover_text(mesh, point_id: int) -> str:
    horizontal = float(mesh[HORIZONTAL_ANGLE_SCALAR_NAME][point_id])
    vertical = float(mesh[VERTICAL_ANGLE_SCALAR_NAME][point_id])
    spl = float(mesh[SPL_SCALAR_NAME][point_id])
    return " | ".join(
        (
            f"Horizontal Angle: {horizontal:+.1f} deg",
            f"Vertical Angle: {vertical:+.1f} deg",
            f"Normalized SPL: {spl:.1f} dB",
        )
    )


def _balloon_isobar_slice(
    prepared: dict[str, np.ndarray],
    azimuth_deg: float,
    *,
    clip_min_db: float | None = None,
    angle_samples: int | None = None,
    freq_samples: int | None = None,
    octave_smoothing: int | float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freqs_hz, angles_deg, values_db = _balloon_raw_slice(prepared, azimuth_deg)
    values_db = _fractional_octave_smooth(values_db.astype(float, copy=False), freqs_hz.astype(float), octave_smoothing)
    if clip_min_db is not None:
        values_db = np.maximum(values_db, float(clip_min_db))
    angles_deg, freqs_hz, values_db = _interpolate_isobar_heatmap(
        angles_deg.astype(float, copy=False),
        freqs_hz.astype(float, copy=False),
        values_db,
        angle_samples,
        freq_samples,
        float(clip_min_db) if clip_min_db is not None else float(np.nanmin(values_db)),
    )
    return (
        freqs_hz.astype(np.float32, copy=False),
        angles_deg.astype(np.float32, copy=False),
        values_db.astype(np.float32, copy=False),
    )


def _balloon_radar_slice(
    prepared: dict[str, np.ndarray],
    frequency_index: int,
    azimuth_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    freqs_hz, angles_deg, values_db = _balloon_raw_slice(prepared, azimuth_deg)
    del freqs_hz
    index = int(np.clip(frequency_index, 0, values_db.shape[1] - 1))
    return angles_deg, values_db[:, index].astype(np.float32, copy=False)


def _balloon_raw_slice(
    prepared: dict[str, np.ndarray],
    azimuth_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freqs_hz = np.asarray(prepared["freq_hz"], dtype=np.float32)
    spl = np.asarray(prepared["balloon_surface_spl"], dtype=np.float32)
    theta_grid = np.asarray(prepared["theta_grid_rad"], dtype=float)
    phi_grid = np.asarray(prepared["phi_grid_rad"], dtype=float)

    if spl.ndim != 3 or theta_grid.ndim != 2 or phi_grid.ndim != 2:
        raise ValueError("Prepared balloon data must contain 3D SPL and 2D angle grids.")
    if spl.shape[1:] != theta_grid.shape or theta_grid.shape != phi_grid.shape:
        raise ValueError("Prepared balloon SPL and angle grids must have matching surface shapes.")

    theta_values = theta_grid[:, 0]
    phi_values = phi_grid[0, :]
    theta_deg = np.rad2deg(theta_values).astype(np.float32)
    azimuth_rad = np.deg2rad(float(azimuth_deg) % 360.0)
    opposite_azimuth_rad = (azimuth_rad + np.pi) % (2.0 * np.pi)
    positive_phi_index = _nearest_periodic_angle_index(phi_values, azimuth_rad)
    negative_phi_index = _nearest_periodic_angle_index(phi_values, opposite_azimuth_rad)

    positive_values = spl[:, :, positive_phi_index]
    negative_values = spl[:, 1:, negative_phi_index][:, ::-1]
    angles_deg = np.concatenate((-theta_deg[1:][::-1], theta_deg)).astype(np.float32, copy=False)
    values_db = np.concatenate((negative_values, positive_values), axis=1).T.astype(np.float32, copy=False)
    return freqs_hz, angles_deg, values_db


def _nearest_periodic_angle_index(angles_rad: np.ndarray, target_rad: float) -> int:
    wrapped_angles = np.mod(np.asarray(angles_rad, dtype=float), 2.0 * np.pi)
    target = float(target_rad) % (2.0 * np.pi)
    delta = np.abs((wrapped_angles - target + np.pi) % (2.0 * np.pi) - np.pi)
    return int(np.argmin(delta))


def _protractor_line_specs(
    outer_radius: float,
    db_radius: float,
    azimuth_deg: float,
) -> list[tuple[np.ndarray, str, float]]:
    outer_radius = max(float(outer_radius), 1e-6)
    db_radius = max(float(db_radius), 1e-6)
    u_axis, z_axis = _protractor_basis(azimuth_deg)
    specs: list[tuple[np.ndarray, str, float]] = []

    for ring_radius in _protractor_ring_radii(db_radius):
        specs.append((_circle_points_in_plane(ring_radius, u_axis, z_axis), PROTRACTOR_COLOR, 0.75))

    specs.append((_circle_points_in_plane(outer_radius, u_axis, z_axis), PROTRACTOR_AXIS_COLOR, 1.1))
    specs.append((np.vstack((-outer_radius * z_axis, outer_radius * z_axis)), PROTRACTOR_AXIS_COLOR, 1.15))
    specs.append((np.vstack((-outer_radius * u_axis, outer_radius * u_axis)), PROTRACTOR_COLOR, 0.9))

    for angle_deg in PROTRACTOR_ANGLES_DEG:
        for sign in (1.0, -1.0):
            angle_rad = np.deg2rad(angle_deg)
            direction = np.cos(angle_rad) * z_axis + sign * np.sin(angle_rad) * u_axis
            specs.append((np.vstack((np.zeros(3), outer_radius * direction)), PROTRACTOR_COLOR, 0.9))

    return specs


def _protractor_basis(azimuth_deg: float) -> tuple[np.ndarray, np.ndarray]:
    azimuth = np.deg2rad(float(azimuth_deg))
    u_axis = np.array([np.cos(azimuth), np.sin(azimuth), 0.0], dtype=float)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    return u_axis, z_axis


def _protractor_ring_radii(radius: float, step_db: float = CONTOUR_STEP_DB) -> list[float]:
    if radius <= 0.0 or step_db <= 0.0:
        return []
    rings = np.arange(step_db, radius + step_db * 0.25, step_db, dtype=float)
    return [float(ring) for ring in rings if ring < radius]


def _circle_points_in_plane(
    radius: float,
    u_axis: np.ndarray,
    z_axis: np.ndarray,
    samples: int = 145,
) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, int(samples), dtype=float)
    return radius * (
        np.cos(angles)[:, np.newaxis] * u_axis[np.newaxis, :]
        + np.sin(angles)[:, np.newaxis] * z_axis[np.newaxis, :]
    )


def _picked_point_id(picker, mesh) -> int | None:
    if hasattr(picker, "GetPointId"):
        point_id = int(picker.GetPointId())
        if 0 <= point_id < mesh.n_points:
            return point_id

    if not hasattr(picker, "GetPickPosition"):
        return None

    pick_position = np.asarray(picker.GetPickPosition(), dtype=float)
    points = np.asarray(mesh.points, dtype=float)
    if points.size == 0:
        return None

    distances = np.linalg.norm(points - pick_position[np.newaxis, :], axis=1)
    point_id = int(np.argmin(distances))
    return point_id if np.isfinite(distances[point_id]) else None


def _plotter_interactor(plotter):
    interactor = getattr(plotter, "interactor", None)
    if interactor is not None and hasattr(interactor, "GetEventPosition"):
        return interactor

    plotter_interactor = getattr(plotter, "iren", None)
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


class SliceRadarCanvas(FigureCanvas):
    def __init__(self, min_db: float, max_db: float):
        self.figure = Figure(figsize=(2.75, 2.75), dpi=100)
        self.axes = self.figure.add_subplot(111, projection="polar")
        super().__init__(self.figure)
        self._min_db = float(min_db)
        self._max_db = float(max_db)
        self._draw_empty()

    def sizeHint(self) -> QSize:
        return QSize(235, 286)

    def _draw_empty(self) -> None:
        self.axes.clear()
        self._apply_axes()
        self.draw_idle()

    def update_plot(self, angles_deg: np.ndarray, values_db: np.ndarray) -> None:
        self.axes.clear()
        self._apply_axes()

        angles = np.asarray(angles_deg, dtype=float)
        values = np.asarray(values_db, dtype=float)
        if angles.size >= 2 and values.size == angles.size:
            theta = np.deg2rad(np.concatenate([angles, angles[:1]]))
            radius = np.maximum(np.concatenate([values, values[:1]]) - self._min_db, 0.0)
            self.axes.plot(theta, radius, color="#ff4f7a", linewidth=1.8)

        self.draw_idle()

    def _apply_axes(self) -> None:
        radius_max = max(self._max_db - self._min_db, 1.0)
        self.figure.patch.set_facecolor("#1f1f1f")
        self.axes.set_facecolor("#1f1f1f")
        self.figure.subplots_adjust(left=0.04, right=0.88, top=0.94, bottom=0.06)
        self.axes.set_theta_zero_location("N")
        self.axes.set_theta_direction(-1)
        self.axes.set_ylim(0.0, radius_max)
        theta_ticks = np.arange(0, 360, 30)
        theta_labels = [str(angle) if angle in (30, 60, 90) else "" for angle in theta_ticks]
        self.axes.set_thetagrids(theta_ticks, labels=theta_labels)
        ring_radii = _protractor_ring_radii(radius_max)
        self.axes.set_yticks(ring_radii)
        ring_labels = [f"{int(round(self._min_db + radius))}" for radius in ring_radii]
        self.axes.set_yticklabels(ring_labels)
        self.axes.set_rlabel_position(102)
        self.axes.grid(color="#6f757c", linewidth=0.8, alpha=0.75)
        self.axes.spines["polar"].set_color("#9aa3ad")
        self.axes.spines["polar"].set_linewidth(0.8)
        self.axes.tick_params(axis="x", length=0, pad=1, colors="#d8dee9", labelsize=8)
        self.axes.tick_params(axis="y", length=0, pad=1, colors="#d8dee9", labelsize=8)


class ColorLegend(QWidget):
    def __init__(self, min_db: float, max_db: float, parent: QWidget | None = None):
        super().__init__(parent)
        self._min_db = float(min_db)
        self._max_db = float(max_db)
        self.setMinimumSize(170, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def sizeHint(self) -> QSize:
        return QSize(170, 320)

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#1f1f1f"))

        painter.setPen(QPen(QColor("white")))
        painter.drawText(0, 0, self.width(), 24, Qt.AlignLeft | Qt.AlignVCenter, SPL_SCALAR_NAME)

        bar_left = 22
        bar_top = 40
        bar_width = 34
        bar_height = self.height() - 58
        gradient = QLinearGradient(0, bar_top, 0, bar_top + bar_height)
        for stop in np.linspace(0.0, 1.0, 48):
            scalar_fraction = 1.0 - stop
            gradient.setColorAt(float(stop), _turbo_color(scalar_fraction))
        painter.fillRect(bar_left, bar_top, bar_width, bar_height, gradient)
        painter.setPen(QPen(QColor("#cfcfcf")))
        painter.drawRect(bar_left, bar_top, bar_width, bar_height)

        painter.setPen(QPen(QColor("white")))
        metrics = QFontMetrics(painter.font())
        for value in LEGEND_TICKS_DB:
            if value < self._min_db or value > self._max_db:
                continue
            y = bar_top + _legend_fraction(value, self._min_db, self._max_db) * bar_height
            painter.drawLine(bar_left + bar_width, int(round(y)), bar_left + bar_width + 7, int(round(y)))
            label = f"{int(value)}"
            painter.drawText(
                bar_left + bar_width + 12,
                int(round(y + metrics.ascent() / 2 - 2)),
                label,
            )
        painter.end()


def _contour_levels(min_db: float, max_db: float, step_db: float) -> list[float]:
    first = np.ceil(float(min_db) / step_db) * step_db
    levels = np.arange(first, float(max_db) + 0.5 * step_db, step_db, dtype=float)
    return [float(level) for level in levels if min_db < level < max_db]


def _legend_fraction(value_db: float, min_db: float, max_db: float) -> float:
    if np.isclose(max_db, min_db):
        return 0.0
    return float((max_db - value_db) / (max_db - min_db))


def _turbo_color(fraction: float) -> QColor:
    try:
        from matplotlib import colormaps

        r, g, b, _ = colormaps["turbo"](float(np.clip(fraction, 0.0, 1.0)))
        return QColor.fromRgbF(float(r), float(g), float(b))
    except Exception:
        fallback = (
            (0.0, QColor("#30123b")),
            (0.2, QColor("#4664f0")),
            (0.4, QColor("#1ae4b6")),
            (0.6, QColor("#a4fc3c")),
            (0.8, QColor("#ff9b20")),
            (1.0, QColor("#b40426")),
        )
        fraction = float(np.clip(fraction, 0.0, 1.0))
        for index, (stop, color) in enumerate(fallback[1:], start=1):
            prev_stop, prev_color = fallback[index - 1]
            if fraction <= stop:
                span = max(stop - prev_stop, 1e-9)
                local = (fraction - prev_stop) / span
                return QColor(
                    round(prev_color.red() + (color.red() - prev_color.red()) * local),
                    round(prev_color.green() + (color.green() - prev_color.green()) * local),
                    round(prev_color.blue() + (color.blue() - prev_color.blue()) * local),
                )
        return fallback[-1][1]


def _mesh_extent(mesh) -> float:
    points = np.asarray(mesh.points)
    if points.size == 0:
        return 1.0
    return float(np.nanmax(np.linalg.norm(points, axis=1)))


def _offset_mesh_points(mesh, offset: float):
    contour_mesh = mesh.copy(deep=True)
    points = np.asarray(contour_mesh.points)
    radii = np.linalg.norm(points, axis=1)
    mask = radii > 1e-9
    points[mask] += (points[mask] / radii[mask, np.newaxis]) * float(offset)
    contour_mesh.points = points
    return contour_mesh
