"""Qt/PyVista balloon plot viewer."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from PySide6.QtCore import QEvent, QSize, QSettings, Qt, QTimer, Slot
from PySide6.QtGui import QAction, QColor, QFontMetrics, QIcon, QLinearGradient, QPainter, QPalette, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QFileDialog,
    QFrame,
    QGridLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import minimize_scalar

from blab.balloon import BalloonPrepConfig, prepare_balloon_data
from blab.exporting import export_balloon_data, export_plot_png
from blab.plotting import VisualizerConfig
from blab.ui.settings import SETTINGS_APP, SETTINGS_ORG
from blab.postprocess import _fractional_octave_smooth, _interpolate_isobar_heatmap
from blab.ui.main_window_widgets import DockTitleBar
from blab.ui.plots import (
    FINAL_ISOBAR_ANGLE_SAMPLES,
    FINAL_ISOBAR_FREQ_SAMPLES,
    FINAL_ISOBAR_SHADING,
    LIVE_ISOBAR_ANGLE_SAMPLES,
    LIVE_ISOBAR_FREQ_SAMPLES,
    LIVE_ISOBAR_SHADING,
    IsobarCanvas,
    apply_audio_frequency_axis,
    apply_compact_plot_text,
    clear_plot_axes,
)
from blab.ui.theme import APP_ROOT

SPL_SCALAR_NAME = "Normalized SPL (dB)"
HORIZONTAL_ANGLE_SCALAR_NAME = "Horizontal Angle (deg)"
VERTICAL_ANGLE_SCALAR_NAME = "Vertical Angle (deg)"
CONTOUR_STEP_DB = 6.0
GUIDE_LINE_WIDTH = 3
CONTOUR_COLOR = "#ffffff"
LEGEND_TICKS_DB = (0.0, -6.0, -12.0, -18.0, -24.0, -30.0)
PROTRACTOR_ANGLES_DEG = (30.0, 60.0, 90.0, 120.0, 150.0)
PROTRACTOR_COLOR = "#d8dee9"
PROTRACTOR_AXIS_COLOR = "#ffffff"
WAVEFRONT_LEVEL_DB = -6.0
WAVEFRONT_RAY_COUNT = 145
WAVEFRONT_RAY_SAMPLES = 181
WAVEFRONT_MAX_FRONT_ANGLE_DEG = 89.0
SAVE_DARK_ICON = APP_ROOT / "assets" / "save_dark.ico"
SAVE_LIGHT_ICON = APP_ROOT / "assets" / "save_light.ico"
HIRES_RENDER_DARK_ICON = APP_ROOT / "assets" / "hiresrender_dark.ico"
HIRES_RENDER_LIGHT_ICON = APP_ROOT / "assets" / "hiresrender_light.ico"


class BalloonPlotWindow(QMainWindow):
    def __init__(
        self,
        raw_balloon_data: dict[str, np.ndarray],
        *,
        min_db: float,
        max_db: float,
        polar_smoothing: int | float | None = 24,
        raw_balloon_data_provider: Callable[[], dict[str, np.ndarray] | None] | None = None,
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
            raise RuntimeError(
                "Install the GUI extras with pyvista and pyvistaqt to use the balloon plot viewer."
            ) from exc

        self._pv = pv
        self._vtk = vtk
        self._raw_balloon_data = raw_balloon_data
        self._raw_balloon_data_provider = raw_balloon_data_provider
        self._raw_balloon_signature = _balloon_raw_signature(raw_balloon_data)
        self._prepared = None
        self._min_db = float(min_db)
        self._max_db = float(max_db)
        self._polar_smoothing = polar_smoothing
        self._mesh_actor = None
        self._balloon_mesh = None
        self._contour_actors = []
        self._guide_actors_added = False
        self._protractor_actors = []
        self._protractor_radius = 1.0
        self._hover_picker = None
        self._hover_observer = None
        self._slice_plot_high_res = False
        self._wavefront_shape_summary_cache = None
        self.settings = QSettings(SETTINGS_ORG, SETTINGS_APP)

        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        export_action = QAction("Export Balloon Data", self)
        export_action.triggered.connect(self._export_balloon_data)
        file_menu.addAction(export_action)

        view_menu = menu_bar.addMenu("View")

        self.plotter = QtInteractor(self)
        self.plotter.set_background("#111316")
        self._install_hover_picker()

        self.frequency_slider = QSlider(Qt.Horizontal)
        self.frequency_slider.setEnabled(False)
        self.frequency_slider.setRange(0, 0)
        self.frequency_slider.setSingleStep(1)
        self.frequency_slider.setPageStep(1)
        self.frequency_slider.valueChanged.connect(self._on_frequency_changed)

        self.frequency_label = QLabel("")
        self.frequency_label.setMinimumWidth(86)
        self.frequency_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

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

        self.slice_plot = IsobarCanvas("Isobar Angle Slice", left_margin=0.17, right_margin=0.92, show_colorbar=False)
        self.slice_plot.setMinimumSize(330, 286)
        self.slice_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.hires_slice_action = QAction("Render High Resolution", self)
        self.hires_slice_action.setToolTip("Render high resolution plot")
        self.hires_slice_action.setEnabled(False)
        self.hires_slice_action.triggered.connect(self._render_high_resolution_isobar_slice)

        self.save_slice_action = QAction("Save Plot Image", self)
        self.save_slice_action.setToolTip("Save plot image")
        self.save_slice_action.setEnabled(False)
        self.save_slice_action.triggered.connect(self._save_isobar_slice_image)

        self.save_wavefront_shape_action = QAction("Save Plot Image", self)
        self.save_wavefront_shape_action.setToolTip("Save plot image")
        self.save_wavefront_shape_action.setEnabled(False)
        self.save_wavefront_shape_action.triggered.connect(self._save_wavefront_shape_image)
        self._refresh_plot_button_icons()

        self.radar_plot = SliceRadarCanvas(self._min_db, self._max_db)
        self.radar_plot.setMinimumSize(220, 286)
        self.radar_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.wavefront_shape_plot = WavefrontShapeCanvas()
        self.wavefront_shape_plot.setMinimumSize(330, 245)
        self.wavefront_shape_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.viewport_stack = QStackedLayout()
        self.viewport_stack.setStackingMode(QStackedLayout.StackAll)
        self.viewport_stack.addWidget(self.plotter.interactor)
        viewport = QWidget()
        viewport.setLayout(self.viewport_stack)

        self.spl_legend = ColorLegend(self._min_db, self._max_db, orientation="horizontal")
        self.spl_legend.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        controls_bar = QFrame()
        controls_layout = QGridLayout(controls_bar)
        controls_layout.addWidget(QLabel("Frequency"), 0, 0)
        controls_layout.addWidget(self.frequency_slider, 0, 1)
        controls_layout.addWidget(self.frequency_label, 0, 2)
        controls_layout.addWidget(self.protractor_toggle, 0, 3)
        controls_layout.addWidget(QLabel("Slice Angle"), 1, 0)
        controls_layout.addWidget(self.protractor_angle_slider, 1, 1)
        controls_layout.addWidget(self.protractor_angle_spin, 1, 2)
        controls_layout.addWidget(self.spl_legend, 0, 4, 2, 1)
        controls_layout.setColumnStretch(1, 1)
        controls_layout.setColumnStretch(4, 1)

        self.hover_label = QLabel("")
        self.hover_label.setMinimumHeight(24)
        self.hover_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.hover_label.setStyleSheet(
            "QLabel {background: #2d2d30;color: #e8e8e8;padding-left: 8px;padding-right: 8px;}"
        )

        self.workspace = QMainWindow()
        self.workspace.setDockOptions(
            QMainWindow.AllowNestedDocks | QMainWindow.AllowTabbedDocks | QMainWindow.AnimatedDocks
        )
        workspace_placeholder = QWidget()
        workspace_placeholder.setMaximumSize(QSize(0, 0))
        workspace_placeholder.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.workspace.setCentralWidget(workspace_placeholder)

        self.balloon_dock = self._make_dock("3D Balloon Plot", viewport, object_name="balloon_3d")
        self.radar_dock = self._make_dock("Radar Slicer Plot", self.radar_plot, object_name="radar_slicer")
        self.wavefront_shape_dock = self._make_dock(
            "Forward Beam Shape",
            self.wavefront_shape_plot,
            object_name="forward_beam_shape",
            tool_actions=(self.save_wavefront_shape_action,),
        )
        self.isobar_dock = self._make_dock(
            "Isobar Angle Slice",
            self.slice_plot,
            object_name="isobar_angle_slice",
            tool_actions=(self.hires_slice_action, self.save_slice_action),
        )
        self.workspace.addDockWidget(Qt.LeftDockWidgetArea, self.balloon_dock)
        self.workspace.addDockWidget(Qt.RightDockWidgetArea, self.radar_dock)
        self.workspace.addDockWidget(Qt.RightDockWidgetArea, self.wavefront_shape_dock)
        self.workspace.addDockWidget(Qt.RightDockWidgetArea, self.isobar_dock)
        self.wavefront_shape_dock.visibilityChanged.connect(self._on_wavefront_shape_visibility_changed)
        self.workspace.splitDockWidget(self.balloon_dock, self.radar_dock, Qt.Horizontal)
        self.workspace.splitDockWidget(self.radar_dock, self.wavefront_shape_dock, Qt.Vertical)
        self.workspace.splitDockWidget(self.wavefront_shape_dock, self.isobar_dock, Qt.Vertical)
        self.workspace.resizeDocks(
            [self.balloon_dock, self.radar_dock, self.wavefront_shape_dock, self.isobar_dock],
            [605, 345, 330, 380],
            Qt.Horizontal,
        )
        view_menu.addAction(self.balloon_dock.toggleViewAction())
        view_menu.addAction(self.radar_dock.toggleViewAction())
        view_menu.addAction(self.wavefront_shape_dock.toggleViewAction())
        view_menu.addAction(self.isobar_dock.toggleViewAction())
        self.wavefront_shape_dock.hide()

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self.workspace, stretch=1)
        layout.addWidget(controls_bar)
        layout.addWidget(self.hover_label)
        self.setCentralWidget(central)

        self._restore_window_state()
        QTimer.singleShot(0, self._prepare_and_render_initial)

    def _make_dock(
        self,
        title: str,
        widget: QWidget,
        *,
        object_name: str,
        tool_actions: tuple[QAction, ...] = (),
    ) -> QDockWidget:
        dock = QDockWidget(title, self.workspace)
        dock.setObjectName(object_name)
        dock.setWidget(widget)
        dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable
        )
        dock.setTitleBarWidget(DockTitleBar(title, dock, tool_actions=tool_actions))
        return dock

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:
            self._refresh_plot_button_icons()
        elif event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            self.refresh_from_latest_results()

    @Slot()
    def refresh_from_latest_results(self) -> None:
        if self._raw_balloon_data_provider is None:
            return
        raw_balloon_data = self._raw_balloon_data_provider()
        if raw_balloon_data is None:
            return
        signature = _balloon_raw_signature(raw_balloon_data)
        if signature == self._raw_balloon_signature:
            return
        self._raw_balloon_data = raw_balloon_data
        self._raw_balloon_signature = signature
        self._prepared = None
        self._wavefront_shape_summary_cache = None
        self._prepare_and_render(preserve_frequency=True)

    @Slot()
    def _prepare_and_render_initial(self) -> None:
        self._prepare_and_render(preserve_frequency=False)

    def _prepare_and_render(self, *, preserve_frequency: bool) -> None:
        self._prepared = prepare_balloon_data(
            self._raw_balloon_data,
            BalloonPrepConfig(min_db=self._min_db, max_db=self._max_db),
        )
        self._min_db = float(self._prepared["min_db"])
        self._max_db = float(self._prepared["max_db"])
        self.spl_legend.set_range(self._min_db, self._max_db)

        frequency_count = int(self._prepared["freq_hz"].size)
        frequency_index = self._current_frequency_index() if preserve_frequency else 0
        frequency_index = int(np.clip(frequency_index, 0, max(frequency_count - 1, 0)))
        self.frequency_slider.blockSignals(True)
        self.frequency_slider.setRange(0, max(frequency_count - 1, 0))
        self.frequency_slider.setPageStep(max(frequency_count // 12, 1))
        self.frequency_slider.setValue(frequency_index)
        self.frequency_slider.blockSignals(False)
        self.frequency_slider.setEnabled(frequency_count > 1)
        self._update_frequency_label(frequency_index)
        self._set_protractor_controls_enabled(self.protractor_toggle.isChecked())
        self.hires_slice_action.setEnabled(frequency_count > 0)
        self.save_slice_action.setEnabled(frequency_count > 0)
        self.save_wavefront_shape_action.setEnabled(frequency_count > 0)

        if not self.wavefront_shape_dock.isHidden():
            self._render_wavefront_shape_plot()
        self._render_frequency(frequency_index, reset_camera=True)
        self._render_isobar_slice_if_enabled()

    @Slot()
    def _export_balloon_data(self) -> None:
        output_text = QFileDialog.getExistingDirectory(
            self,
            "Export balloon data",
            str(Path.cwd()),
        )
        if not output_text:
            return

        try:
            prepared = self._prepared_balloon_data()
            result = export_balloon_data(prepared, output_text)
            QMessageBox.information(
                self,
                "Balloon data exported",
                (
                    f"Exported {result.frequency_count} frequencies, "
                    f"{result.point_count} points, and {result.quad_count} quads to:\n"
                    f"{result.output_dir}"
                ),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export balloon data failed", str(exc))

    def _refresh_plot_button_icons(self) -> None:
        palette = self.palette()
        window_color = palette.color(QPalette.Window)
        light_theme = window_color.lightness() >= 128
        self.hires_slice_action.setIcon(QIcon(str(HIRES_RENDER_LIGHT_ICON if light_theme else HIRES_RENDER_DARK_ICON)))
        self.save_slice_action.setIcon(QIcon(str(SAVE_LIGHT_ICON if light_theme else SAVE_DARK_ICON)))
        self.save_wavefront_shape_action.setIcon(QIcon(str(SAVE_LIGHT_ICON if light_theme else SAVE_DARK_ICON)))

    @Slot()
    def _render_high_resolution_isobar_slice(self) -> None:
        self._render_isobar_slice(final_resolution=True)

    @Slot()
    def _save_isobar_slice_image(self) -> None:
        if self._prepared is None:
            QMessageBox.warning(self, "No plot data", "Run a solve before saving a plot image.")
            return

        path_text, _ = QFileDialog.getSaveFileName(
            self,
            "Save plot image",
            str(Path.cwd() / "balloon_isobar_slice.png"),
            "PNG images (*.png);;All files (*)",
        )
        if not path_text:
            return

        output_path = Path(path_text)
        if output_path.suffix == "":
            output_path = output_path.with_suffix(".png")
        try:
            self._render_isobar_slice(final_resolution=True)
            export_plot_png(self.slice_plot.figure, output_path, dpi=VisualizerConfig.figure_dpi)
        except Exception as exc:
            QMessageBox.critical(self, "Save plot image failed", str(exc))

    @Slot()
    def _save_wavefront_shape_image(self) -> None:
        if self._prepared is None:
            QMessageBox.warning(self, "No plot data", "Run a solve before saving a plot image.")
            return

        path_text, _ = QFileDialog.getSaveFileName(
            self,
            "Save plot image",
            str(Path.cwd() / "forward_beam_shape.png"),
            "PNG images (*.png);;All files (*)",
        )
        if not path_text:
            return

        output_path = Path(path_text)
        if output_path.suffix == "":
            output_path = output_path.with_suffix(".png")
        try:
            self._render_wavefront_shape_plot()
            export_plot_png(self.wavefront_shape_plot.figure, output_path, dpi=VisualizerConfig.figure_dpi)
        except Exception as exc:
            QMessageBox.critical(self, "Save plot image failed", str(exc))

    def _prepared_balloon_data(self) -> dict[str, np.ndarray]:
        if self._prepared is None:
            self._prepared = prepare_balloon_data(
                self._raw_balloon_data,
                BalloonPrepConfig(min_db=self._min_db, max_db=self._max_db),
            )
        return self._prepared

    @Slot(int)
    def _on_frequency_changed(self, index: int) -> None:
        self._render_frequency(index, reset_camera=False)
        if self.protractor_toggle.isChecked():
            self._render_radar_slice()
        self._update_frequency_label(index)
        self._update_wavefront_shape_frequency_cursor(index)

    def _current_frequency_index(self) -> int:
        return int(self.frequency_slider.value())

    def _update_frequency_label(self, index: int) -> None:
        if self._prepared is None or self._prepared["freq_hz"].size == 0:
            self.frequency_label.setText("")
            return
        safe_index = int(np.clip(index, 0, self._prepared["freq_hz"].size - 1))
        self.frequency_label.setText(_format_frequency(float(self._prepared["freq_hz"][safe_index])))

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

        if self._mesh_actor is not None:
            try:
                self.plotter.remove_actor(self._mesh_actor, render=False)
            except Exception:
                pass
        self._balloon_mesh = mesh
        self._mesh_actor = self.plotter.add_mesh(
            mesh,
            scalars=SPL_SCALAR_NAME,
            cmap="turbo",
            clim=(self._min_db, self._max_db),
            smooth_shading=True,
            show_scalar_bar=False,
            ambient=0.45,
            diffuse=0.9,
            specular=0.18,
            specular_power=24,
        )

        self._refresh_spl_contours()
        if not self._guide_actors_added:
            self._add_orientation_guides(self._balloon_mesh)
            self._guide_actors_added = True
        if self.protractor_toggle.isChecked() and not self._protractor_actors:
            self._add_protractor(self._balloon_mesh)
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
        if enabled and self._balloon_mesh is not None:
            if not self._protractor_actors:
                self._add_protractor(self._balloon_mesh)
            self._render_isobar_slice()
        elif not enabled:
            self._remove_protractor()
        self.plotter.render()

    @Slot()
    def _render_isobar_slice(self, *, final_resolution: bool = False) -> None:
        if self._prepared is None:
            return
        freqs_hz, angles_deg, values_db = _balloon_isobar_slice(
            self._prepared,
            float(self.protractor_angle_slider.value()),
            clip_min_db=self._min_db,
            angle_samples=FINAL_ISOBAR_ANGLE_SAMPLES if final_resolution else LIVE_ISOBAR_ANGLE_SAMPLES,
            freq_samples=FINAL_ISOBAR_FREQ_SAMPLES if final_resolution else LIVE_ISOBAR_FREQ_SAMPLES,
            octave_smoothing=self._polar_smoothing,
        )
        self.slice_plot.update_plot(
            freqs_hz,
            angles_deg,
            values_db,
            self._min_db,
            self._max_db,
            shading=FINAL_ISOBAR_SHADING if final_resolution else LIVE_ISOBAR_SHADING,
        )
        self._slice_plot_high_res = bool(final_resolution)
        self._render_radar_slice()

    @Slot()
    def _render_isobar_slice_if_enabled(self) -> None:
        if self.protractor_toggle.isChecked():
            self._render_isobar_slice()

    def _render_radar_slice(self) -> None:
        if self._prepared is None:
            return
        frequency_index = self._current_frequency_index()
        if frequency_index < 0:
            return
        angles_deg, values_db = _balloon_radar_slice(
            self._prepared,
            frequency_index,
            float(self.protractor_angle_slider.value()),
        )
        self.radar_plot.update_plot(angles_deg, values_db)

    @Slot(bool)
    def _on_wavefront_shape_visibility_changed(self, visible: bool) -> None:
        if visible:
            self._render_wavefront_shape_plot()

    def _render_wavefront_shape_plot(self) -> None:
        if self._prepared is None:
            return
        if self._wavefront_shape_summary_cache is None:
            self._wavefront_shape_summary_cache = _wavefront_shape_summary(
                self._prepared,
                raw_balloon_data=self._raw_balloon_data,
            )
        self.wavefront_shape_plot.update_plot(self._wavefront_shape_summary_cache)
        self._update_wavefront_shape_frequency_cursor(self._current_frequency_index())

    def _update_wavefront_shape_frequency_cursor(self, index: int) -> None:
        if self._prepared is None or self._prepared["freq_hz"].size == 0:
            self.wavefront_shape_plot.set_frequency_cursor(None)
            return
        safe_index = int(np.clip(index, 0, self._prepared["freq_hz"].size - 1))
        self.wavefront_shape_plot.set_frequency_cursor(float(self._prepared["freq_hz"][safe_index]))

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
        actor = self.plotter.add_mesh(
            contours.tube(radius=tube_radius),
            color=CONTOUR_COLOR,
            smooth_shading=True,
            lighting=False,
            show_scalar_bar=False,
        )
        self._contour_actors.append(actor)

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

    def _refresh_spl_contours(self) -> None:
        self._remove_spl_contours()
        if self._balloon_mesh is not None:
            self._add_spl_contours(self._balloon_mesh)

    def _remove_spl_contours(self) -> None:
        for actor in self._contour_actors:
            try:
                self.plotter.remove_actor(actor, render=False)
            except Exception:
                pass
        self._contour_actors = []

    def _restore_window_state(self) -> None:
        geometry = self.settings.value("balloon_window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

        dock_state = self.settings.value("balloon_window/dock_state")
        if dock_state is not None:
            self.workspace.restoreState(dock_state)

    def _save_window_state(self) -> None:
        self.settings.setValue("balloon_window/geometry", self.saveGeometry())
        self.settings.setValue("balloon_window/dock_state", self.workspace.saveState())
        self.settings.sync()

    def closeEvent(self, event) -> None:
        self._save_window_state()
        self.plotter.close()
        super().closeEvent(event)



def _balloon_raw_signature(raw_balloon_data: dict[str, np.ndarray]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for key in ("freq_hz", "r_distance_m", "theta_polar_rad", "phi_azimuth_rad", "spl_norm"):
        array = np.ascontiguousarray(np.asarray(raw_balloon_data.get(key), dtype=np.float32))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.view(np.uint8))
    return digest.hexdigest()

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
        np.cos(angles)[:, np.newaxis] * u_axis[np.newaxis, :] + np.sin(angles)[:, np.newaxis] * z_axis[np.newaxis, :]
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


def _wavefront_shape_summary(
    prepared: dict[str, np.ndarray],
    *,
    raw_balloon_data: dict[str, np.ndarray] | None = None,
    level_db: float = WAVEFRONT_LEVEL_DB,
) -> dict[str, np.ndarray]:
    freqs_hz = np.asarray(prepared["freq_hz"], dtype=np.float32)
    spl = np.asarray(prepared["balloon_surface_spl"], dtype=np.float32)
    theta_grid = np.asarray(prepared["theta_grid_rad"], dtype=float)
    phi_grid = np.asarray(prepared["phi_grid_rad"], dtype=float)

    shape = (freqs_hz.size,)
    result = {
        "freq_hz": freqs_hz,
        "shape_exponent": np.full(shape, np.nan, dtype=np.float32),
        "fit_residual_percent": np.full(shape, np.nan, dtype=np.float32),
        "horizontal_beamwidth_deg": np.full(shape, np.nan, dtype=np.float32),
        "vertical_beamwidth_deg": np.full(shape, np.nan, dtype=np.float32),
        "aspect_ratio": np.full(shape, np.nan, dtype=np.float32),
        "directivity_index_db": _spherical_directivity_index_db(prepared, raw_balloon_data),
        "valid": np.zeros(shape, dtype=bool),
    }
    if spl.ndim != 3 or spl.shape[0] != freqs_hz.size or spl.shape[1:] != theta_grid.shape:
        return result

    horizontal_deg, vertical_deg, front_mask = _front_angle_meshes(theta_grid, phi_grid)
    for index in range(freqs_hz.size):
        fit = _fit_wavefront_shape_for_frequency(
            spl[index],
            horizontal_deg,
            vertical_deg,
            front_mask,
            level_db=float(level_db),
        )
        if fit is None:
            continue
        result["shape_exponent"][index] = fit["shape_exponent"]
        result["fit_residual_percent"][index] = fit["fit_residual_percent"]
        result["horizontal_beamwidth_deg"][index] = fit["horizontal_beamwidth_deg"]
        result["vertical_beamwidth_deg"][index] = fit["vertical_beamwidth_deg"]
        result["aspect_ratio"][index] = fit["aspect_ratio"]
        result["valid"][index] = True
    return result


def _spherical_directivity_index_db(
    prepared: dict[str, np.ndarray],
    raw_balloon_data: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    freqs_hz = np.asarray(prepared["freq_hz"], dtype=np.float32)
    if raw_balloon_data is not None:
        raw_di = _spherical_directivity_index_from_raw(raw_balloon_data, freqs_hz)
        if raw_di is not None:
            return raw_di
    return _spherical_directivity_index_from_prepared(prepared, freqs_hz)


def _spherical_directivity_index_from_raw(
    raw_balloon_data: dict[str, np.ndarray],
    prepared_freqs_hz: np.ndarray,
) -> np.ndarray | None:
    try:
        raw_freqs_hz = np.asarray(raw_balloon_data["freq_hz"], dtype=np.float32)
        spl = np.asarray(raw_balloon_data["spl_norm"], dtype=float)
    except KeyError:
        return None

    if spl.ndim != 2 or spl.shape[0] != prepared_freqs_hz.size:
        return None
    if raw_freqs_hz.shape != prepared_freqs_hz.shape or not np.allclose(raw_freqs_hz, prepared_freqs_hz):
        return None
    return _directivity_index_from_energy_mean(np.mean(_db_to_energy(spl), axis=1))


def _spherical_directivity_index_from_prepared(
    prepared: dict[str, np.ndarray],
    freqs_hz: np.ndarray,
) -> np.ndarray:
    spl = np.asarray(prepared.get("balloon_surface_spl"), dtype=float)
    theta_grid = np.asarray(prepared.get("theta_grid_rad"), dtype=float)
    phi_grid = np.asarray(prepared.get("phi_grid_rad"), dtype=float)
    output = np.full(freqs_hz.shape, np.nan, dtype=np.float32)
    if spl.ndim != 3 or spl.shape[0] != freqs_hz.size or spl.shape[1:] != theta_grid.shape:
        return output

    if phi_grid.ndim == 2 and phi_grid.shape[1] >= 2 and np.isclose(phi_grid[0, -1] - phi_grid[0, 0], 2.0 * np.pi):
        spl = spl[:, :, :-1]
        theta_grid = theta_grid[:, :-1]

    weights = np.sin(theta_grid)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return output

    energy_mean = np.sum(_db_to_energy(spl) * weights[np.newaxis, :, :], axis=(1, 2)) / weight_sum
    return _directivity_index_from_energy_mean(energy_mean)


def _directivity_index_from_energy_mean(energy_mean: np.ndarray) -> np.ndarray:
    return (-10.0 * np.log10(np.maximum(np.asarray(energy_mean, dtype=float), np.finfo(float).tiny))).astype(
        np.float32,
        copy=False,
    )


def _db_to_energy(db: np.ndarray) -> np.ndarray:
    return np.power(10.0, np.asarray(db, dtype=float) / 10.0)


def _fit_wavefront_shape_for_frequency(
    spl_db: np.ndarray,
    horizontal_deg: np.ndarray,
    vertical_deg: np.ndarray,
    front_mask: np.ndarray,
    *,
    level_db: float,
) -> dict[str, float] | None:
    interpolator = _front_tangent_interpolator(horizontal_deg, vertical_deg, spl_db, front_mask)
    if interpolator is None:
        return None

    ray_angles = np.linspace(0.0, 2.0 * np.pi, WAVEFRONT_RAY_COUNT, endpoint=False, dtype=float)
    contour_angles: list[float] = []
    contour_radii: list[float] = []
    for angle in ray_angles:
        radius = _level_crossing_for_ray(interpolator, angle, level_db)
        if radius is None:
            continue
        contour_angles.append(float(angle))
        contour_radii.append(float(radius))

    if len(contour_radii) < max(24, WAVEFRONT_RAY_COUNT // 4):
        return None

    angles = np.asarray(contour_angles, dtype=float)
    radii = np.asarray(contour_radii, dtype=float)
    horizontal_extent = _axis_tangent_extent_from_rays(interpolator, 0.0, np.pi, level_db)
    vertical_extent = _axis_tangent_extent_from_rays(interpolator, 0.5 * np.pi, 1.5 * np.pi, level_db)

    if horizontal_extent is None or vertical_extent is None:
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        horizontal_extent = _signed_axis_extent(x)
        vertical_extent = _signed_axis_extent(y)
    if horizontal_extent is None or vertical_extent is None:
        return None

    horizontal_positive, horizontal_negative = horizontal_extent
    vertical_positive, vertical_negative = vertical_extent
    a = 0.5 * float(horizontal_positive + horizontal_negative)
    b = 0.5 * float(vertical_positive + vertical_negative)
    if not np.isfinite(a) or not np.isfinite(b) or a <= 1e-6 or b <= 1e-6:
        return None

    def objective(exponent: float) -> float:
        model = _superellipse_radius(angles, a, b, exponent)
        return float(np.nanmean((radii - model) ** 2))

    optimum = minimize_scalar(objective, bounds=(0.75, 8.0), method="bounded", options={"xatol": 1e-3})
    if not optimum.success or not np.isfinite(optimum.x):
        return None

    exponent = float(optimum.x)
    model = _superellipse_radius(angles, a, b, exponent)
    residual = float(np.sqrt(np.nanmean((radii - model) ** 2)))
    residual_percent = 100.0 * residual / max(float(np.nanmean([a, b])), 1e-6)
    horizontal_beamwidth = _tangent_extent_to_beamwidth_deg(horizontal_positive, horizontal_negative)
    vertical_beamwidth = _tangent_extent_to_beamwidth_deg(vertical_positive, vertical_negative)
    return {
        "shape_exponent": exponent,
        "fit_residual_percent": residual_percent,
        "horizontal_beamwidth_deg": float(horizontal_beamwidth),
        "vertical_beamwidth_deg": float(vertical_beamwidth),
        "aspect_ratio": float(horizontal_beamwidth / max(vertical_beamwidth, 1e-6)),
    }


def _front_angle_meshes(theta_rad: np.ndarray, phi_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.asarray(theta_rad, dtype=float)
    phi = np.asarray(phi_rad, dtype=float)
    direction_x = np.sin(theta) * np.cos(phi)
    direction_y = np.sin(theta) * np.sin(phi)
    direction_z = np.cos(theta)
    horizontal = _normalize_signed_angle(np.rad2deg(np.arctan2(direction_x, direction_z)))
    vertical = _normalize_signed_angle(np.rad2deg(np.arctan2(direction_y, direction_z)))
    front_mask = (
        (direction_z > 0.0)
        & np.isfinite(horizontal)
        & np.isfinite(vertical)
        & (np.abs(horizontal) <= WAVEFRONT_MAX_FRONT_ANGLE_DEG)
        & (np.abs(vertical) <= WAVEFRONT_MAX_FRONT_ANGLE_DEG)
    )
    return horizontal, vertical, front_mask


def _front_tangent_interpolator(
    horizontal_deg: np.ndarray,
    vertical_deg: np.ndarray,
    spl_db: np.ndarray,
    front_mask: np.ndarray,
):
    values = np.asarray(spl_db, dtype=float)
    horizontal_tangent = np.tan(np.deg2rad(horizontal_deg))
    vertical_tangent = np.tan(np.deg2rad(vertical_deg))
    mask = front_mask & np.isfinite(values) & np.isfinite(horizontal_tangent) & np.isfinite(vertical_tangent)
    if np.count_nonzero(mask) < 4:
        return None

    points = np.column_stack((horizontal_tangent[mask].ravel(), vertical_tangent[mask].ravel()))
    point_values = values[mask].ravel()
    unique_points, unique_indices = np.unique(np.round(points, decimals=6), axis=0, return_index=True)
    if unique_points.shape[0] < 4:
        return None
    try:
        return LinearNDInterpolator(unique_points, point_values[unique_indices], fill_value=np.nan)
    except Exception:
        return None


def _axis_tangent_extent_from_rays(
    interpolator,
    positive_angle_rad: float,
    negative_angle_rad: float,
    level_db: float,
) -> tuple[float, float] | None:
    positive = _level_crossing_for_ray(interpolator, positive_angle_rad, level_db)
    negative = _level_crossing_for_ray(interpolator, negative_angle_rad, level_db)
    if positive is None or negative is None:
        return None
    return float(positive), float(negative)


def _signed_axis_extent(values: np.ndarray) -> tuple[float, float] | None:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    positive = float(np.nanmax(finite))
    negative = float(-np.nanmin(finite))
    if not np.isfinite(positive) or not np.isfinite(negative) or positive <= 0.0 or negative <= 0.0:
        return None
    return positive, negative


def _tangent_extent_to_beamwidth_deg(positive: float, negative: float) -> float:
    return float(np.rad2deg(np.arctan(max(float(positive), 0.0))) + np.rad2deg(np.arctan(max(float(negative), 0.0))))


def _level_crossing_for_ray(interpolator, angle_rad: float, level_db: float) -> float | None:
    limit = _front_ray_limit(angle_rad)
    angular_radii = np.linspace(0.0, np.arctan(limit), WAVEFRONT_RAY_SAMPLES, dtype=float)
    radii = np.tan(angular_radii)
    coords = np.column_stack((radii * np.cos(angle_rad), radii * np.sin(angle_rad)))
    values = np.asarray(interpolator(coords), dtype=float)
    finite = np.isfinite(values)
    if np.count_nonzero(finite) < 2:
        return None

    last_radius: float | None = None
    last_value: float | None = None
    for radius, value, is_finite in zip(radii, values, finite):
        if not is_finite:
            continue
        radius = float(radius)
        value = float(value)
        if last_radius is not None and last_value is not None:
            if last_value >= level_db and value <= level_db:
                span = value - last_value
                if np.isclose(span, 0.0):
                    return radius
                fraction = (level_db - last_value) / span
                return float(last_radius + np.clip(fraction, 0.0, 1.0) * (radius - last_radius))
        last_radius = radius
        last_value = value
    return None


def _front_ray_limit(angle_rad: float) -> float:
    tangent_limit = float(np.tan(np.deg2rad(WAVEFRONT_MAX_FRONT_ANGLE_DEG)))
    cosine = abs(float(np.cos(angle_rad)))
    sine = abs(float(np.sin(angle_rad)))
    limits = []
    if cosine > 1e-6:
        limits.append(tangent_limit / cosine)
    if sine > 1e-6:
        limits.append(tangent_limit / sine)
    return float(min(limits) if limits else tangent_limit)


def _superellipse_radius(angle_rad: np.ndarray, horizontal_radius: float, vertical_radius: float, exponent: float) -> np.ndarray:
    p = max(float(exponent), 1e-6)
    a = max(float(horizontal_radius), 1e-6)
    b = max(float(vertical_radius), 1e-6)
    denom = (np.abs(np.cos(angle_rad)) / a) ** p + (np.abs(np.sin(angle_rad)) / b) ** p
    return np.power(np.maximum(denom, 1e-12), -1.0 / p)


def _rounded_square_marker() -> MplPath:
    radius = 0.42
    width = 1.65
    height = 1.65
    left = -width / 2.0
    right = width / 2.0
    bottom = -height / 2.0
    top = height / 2.0
    vertices = [
        (left + radius, bottom),
        (right - radius, bottom),
        (right, bottom),
        (right, bottom + radius),
        (right, top - radius),
        (right, top),
        (right - radius, top),
        (left + radius, top),
        (left, top),
        (left, top - radius),
        (left, bottom + radius),
        (left, bottom),
        (left + radius, bottom),
        (left + radius, bottom),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.CLOSEPOLY,
    ]
    return MplPath(vertices, codes)


class WavefrontShapeCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(4.1, 2.45), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.di_axes = self.axes.twinx()
        super().__init__(self.figure)
        self._colorbar = None
        self._colorbar_axes = None
        self._frequency_cursor_line = None
        self._current_frequency_hz: float | None = None
        self._cmap = LinearSegmentedColormap.from_list(
            "wavefront_shape_residual",
            ["#33d17a", "#f6d32d", "#e01b24"],
        )
        self._draw_empty()

    def sizeHint(self) -> QSize:
        return QSize(330, 245)

    def _draw_empty(self) -> None:
        clear_plot_axes(self.axes)
        clear_plot_axes(self.di_axes)
        self._frequency_cursor_line = None
        self._remove_colorbar()
        self._configure_axes()
        self.draw_idle()

    def update_plot(self, summary: dict[str, np.ndarray]) -> None:
        clear_plot_axes(self.axes)
        clear_plot_axes(self.di_axes)
        self._frequency_cursor_line = None
        self._remove_colorbar()
        self._configure_axes()

        freqs = np.asarray(summary.get("freq_hz", []), dtype=float)
        exponents = np.asarray(summary.get("shape_exponent", []), dtype=float)
        residuals = np.asarray(summary.get("fit_residual_percent", []), dtype=float)
        directivity_index = np.asarray(summary.get("directivity_index_db", []), dtype=float)
        shape_valid = np.asarray(summary.get("valid", np.zeros(freqs.shape, dtype=bool)), dtype=bool)
        shape_valid &= np.isfinite(freqs) & np.isfinite(exponents) & np.isfinite(residuals) & (freqs > 0.0)
        di_valid = (
            freqs.shape == directivity_index.shape
            and np.isfinite(freqs)
            & np.isfinite(directivity_index)
            & (freqs > 0.0)
        )
        if not np.any(shape_valid) and not np.any(di_valid):
            self.axes.text(
                0.5,
                0.5,
                "No usable forward beam data",
                transform=self.axes.transAxes,
                ha="center",
                va="center",
                color="#d8dee9",
                fontsize=9,
            )
            self._draw_frequency_cursor()
            self.draw_idle()
            return

        if np.any(shape_valid):
            self._plot_shape_exponent(freqs[shape_valid], exponents[shape_valid], residuals[shape_valid])
        if np.any(di_valid):
            self._plot_directivity_index(freqs[di_valid], directivity_index[di_valid])

        self.axes.set_ylim(0.75, 8.5)
        self.di_axes.set_ylim(-5.0, 50.0)
        self._draw_frequency_cursor()
        self.draw_idle()

    def set_frequency_cursor(self, freq_hz: float | None) -> None:
        if freq_hz is None or not np.isfinite(freq_hz) or freq_hz <= 0.0:
            self._current_frequency_hz = None
            self._remove_frequency_cursor()
            self.draw_idle()
            return
        self._current_frequency_hz = float(freq_hz)
        self._draw_frequency_cursor()
        self.draw_idle()

    def _draw_frequency_cursor(self) -> None:
        freq_hz = self._current_frequency_hz
        if freq_hz is None or not np.isfinite(freq_hz) or freq_hz <= 0.0:
            self._remove_frequency_cursor()
            return
        if self._frequency_cursor_line is None:
            self._frequency_cursor_line = self.axes.axvline(
                freq_hz,
                color="#4aa3ff",
                linewidth=1.2,
                linestyle="-",
                alpha=0.92,
                zorder=5,
            )
        else:
            self._frequency_cursor_line.set_xdata([freq_hz, freq_hz])

    def _remove_frequency_cursor(self) -> None:
        if self._frequency_cursor_line is None:
            return
        try:
            self._frequency_cursor_line.remove()
        except ValueError:
            pass
        self._frequency_cursor_line = None

    def _plot_shape_exponent(self, freqs: np.ndarray, exponents: np.ndarray, residuals: np.ndarray) -> None:
        order = np.argsort(freqs)
        freqs = freqs[order]
        exponents = exponents[order]
        residuals = residuals[order]
        norm = Normalize(vmin=0.0, vmax=15.0, clip=True)
        if freqs.size >= 2:
            points = np.column_stack((freqs, exponents))[:, np.newaxis, :]
            segments = np.concatenate((points[:-1], points[1:]), axis=1)
            segment_residuals = 0.5 * (residuals[:-1] + residuals[1:])
            line = LineCollection(segments, cmap=self._cmap, norm=norm, linewidths=2.1)
            line.set_array(segment_residuals)
            self.axes.add_collection(line)
        scatter = self.axes.scatter(
            freqs,
            exponents,
            c=residuals,
            cmap=self._cmap,
            norm=norm,
            s=24,
            linewidths=0.45,
            edgecolors="#101214",
            zorder=3,
        )
        self._colorbar_axes = self.figure.add_axes([0.89, 0.22, 0.025, 0.68])
        self._colorbar = self.figure.colorbar(scatter, cax=self._colorbar_axes)
        self._colorbar.set_label("Fit residual (%)", fontsize=8)
        self._colorbar.ax.tick_params(labelsize=8)
        self._style_colorbar()

    def _plot_directivity_index(self, freqs: np.ndarray, directivity_index: np.ndarray) -> None:
        order = np.argsort(freqs)
        self.di_axes.plot(
            freqs[order],
            directivity_index[order],
            color="#c678dd",
            linewidth=1.35,
            linestyle="--",
        )

    def _configure_axes(self) -> None:
        self.figure.patch.set_facecolor("#ffffff")
        self.figure.subplots_adjust(left=0.18, right=0.74, top=0.9, bottom=0.22)
        self.axes.set_facecolor("#ffffff")
        self.axes.set_title("Forward Beam Shape", pad=1)
        self.axes.set_xlabel("Frequency (Hz)")
        self.axes.set_ylabel("Superellipse p")
        self.di_axes.set_ylabel("Spherical DI (dB)", labelpad=2)
        self.di_axes.yaxis.set_label_position("right")
        self.di_axes.yaxis.tick_right()
        self.di_axes.yaxis.set_label_coords(1.17, 0.5)
        apply_audio_frequency_axis(self.axes)
        apply_audio_frequency_axis(self.di_axes)
        self.axes.set_ylim(0.75, 8.5)
        self.axes.set_yticks([1.0, 2.0, 4.0, 6.0, 8.0])
        self.di_axes.set_ylim(-5.0, 50.0)
        self.axes.grid(which="major", color="#808080", linewidth=0.8, alpha=0.6)
        self.axes.tick_params(axis="both", colors="#1d1d1d")
        self.di_axes.tick_params(axis="y", colors="#1d1d1d")
        for axes in (self.axes, self.di_axes):
            for spine in axes.spines.values():
                spine.set_color("#9aa3ad")
        self.axes.title.set_color("#1d1d1d")
        self.axes.xaxis.label.set_color("#1d1d1d")
        self.axes.yaxis.label.set_color("#1d1d1d")
        self.di_axes.yaxis.label.set_color("#1d1d1d")
        apply_compact_plot_text(self.axes)
        apply_compact_plot_text(self.di_axes)
        self._draw_shape_reference_markers()

    def _draw_shape_reference_markers(self) -> None:
        marker_specs = (
            (1.0, "D"),
            (2.0, "o"),
            (4.0, _rounded_square_marker()),
            (8.0, "s"),
        )
        transform = self.axes.get_yaxis_transform()
        marker_color = self._theme_marker_color()
        for value, marker in marker_specs:
            self.axes.scatter(
                [-0.12],
                [value],
                marker=marker,
                s=42,
                transform=transform,
                facecolors="none",
                edgecolors=marker_color,
                linewidths=1.50,
                clip_on=False,
                zorder=4,
            )

    def _theme_marker_color(self) -> str:
        return "#101214" if self.palette().color(QPalette.Window).lightness() >= 128 else "#f2f2f2"

    def _style_colorbar(self) -> None:
        if self._colorbar is None:
            return
        text_color = "#1d1d1d"
        spine_color = "#9aa3ad"
        self._colorbar.ax.yaxis.label.set_color(text_color)
        self._colorbar.ax.tick_params(colors=text_color)
        self._colorbar.outline.set_edgecolor(spine_color)
        for spine in self._colorbar.ax.spines.values():
            spine.set_color(spine_color)

    def _remove_colorbar(self) -> None:
        if self._colorbar is not None:
            try:
                self._colorbar.remove()
            except (AttributeError, KeyError, ValueError):
                pass
        elif self._colorbar_axes is not None:
            try:
                self._colorbar_axes.remove()
            except (AttributeError, KeyError, ValueError):
                pass
        self._colorbar = None
        self._colorbar_axes = None


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
    def __init__(
        self,
        min_db: float,
        max_db: float,
        parent: QWidget | None = None,
        *,
        orientation: str = "vertical",
    ):
        super().__init__(parent)
        self._min_db = float(min_db)
        self._max_db = float(max_db)
        self._orientation = str(orientation).lower()
        if self._orientation == "horizontal":
            self.setMinimumSize(330, 62)
            self.setMaximumHeight(68)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        else:
            self.setMinimumSize(170, 320)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_range(self, min_db: float, max_db: float) -> None:
        self._min_db = float(min_db)
        self._max_db = float(max_db)
        self.update()

    def sizeHint(self) -> QSize:
        if self._orientation == "horizontal":
            return QSize(380, 62)
        return QSize(170, 320)

    def paintEvent(self, event) -> None:
        del event
        if self._orientation == "horizontal":
            self._paint_horizontal()
            return
        self._paint_vertical()

    def _paint_horizontal(self) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(31, 31, 31, 180))

        painter.setPen(QPen(QColor("white")))
        painter.drawText(12, 4, self.width() - 24, 18, Qt.AlignLeft | Qt.AlignVCenter, SPL_SCALAR_NAME)

        label_edge_pad = 22
        bar_left = label_edge_pad
        bar_top = 28
        bar_width = max(self.width() - 2 * label_edge_pad, 40)
        bar_height = 18
        gradient = QLinearGradient(bar_left, 0, bar_left + bar_width, 0)
        for stop in np.linspace(0.0, 1.0, 48):
            gradient.setColorAt(float(stop), _turbo_color(float(stop)))
        painter.fillRect(bar_left, bar_top, bar_width, bar_height, gradient)
        painter.setPen(QPen(QColor("#cfcfcf")))
        painter.drawRect(bar_left, bar_top, bar_width, bar_height)

        painter.setPen(QPen(QColor("white")))
        metrics = QFontMetrics(painter.font())
        for value in LEGEND_TICKS_DB:
            if value < self._min_db or value > self._max_db:
                continue
            x = bar_left + _legend_horizontal_fraction(value, self._min_db, self._max_db) * bar_width
            x_int = int(round(x))
            painter.drawLine(x_int, bar_top + bar_height, x_int, bar_top + bar_height + 6)
            label = f"{int(value)}"
            label_width = metrics.horizontalAdvance(label)
            label_x = int(round(np.clip(x - label_width / 2, 2, max(self.width() - label_width - 2, 2))))
            painter.drawText(
                label_x,
                bar_top + bar_height + metrics.ascent() + 7,
                label,
            )
        painter.end()

    def _paint_vertical(self) -> None:
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


def _legend_horizontal_fraction(value_db: float, min_db: float, max_db: float) -> float:
    if np.isclose(max_db, min_db):
        return 0.0
    return float((value_db - min_db) / (max_db - min_db))


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

