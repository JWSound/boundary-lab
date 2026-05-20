"""Qt/PyVista balloon plot viewer."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QSize, QTimer, Qt, Slot
from PySide6.QtGui import QColor, QFontMetrics, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from blab.balloon import BalloonPrepConfig, prepare_balloon_data


SPL_SCALAR_NAME = "Normalized SPL (dB)"
CONTOUR_STEP_DB = 6.0
GUIDE_LINE_WIDTH = 3
CONTOUR_COLOR = "#f4f0e6"
LEGEND_TICKS_DB = (0.0, -6.0, -12.0, -18.0, -24.0, -30.0)


class BalloonPlotWindow(QDialog):
    def __init__(
        self,
        raw_balloon_data: dict[str, np.ndarray],
        *,
        min_db: float,
        max_db: float,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Balloon Plot")
        self.resize(1100, 760)

        try:
            import pyvista as pv
            from pyvistaqt import QtInteractor
        except ImportError as exc:
            raise RuntimeError("Install the GUI extras with pyvista and pyvistaqt to use the balloon plot viewer.") from exc

        self._pv = pv
        self._raw_balloon_data = raw_balloon_data
        self._prepared = None
        self._min_db = float(min_db)
        self._max_db = float(max_db)
        self._mesh_actor = None

        self.plotter = QtInteractor(self)
        self.plotter.set_background("#111316")

        self.frequency_combo = QComboBox()
        self.frequency_combo.setEnabled(False)
        self.frequency_combo.currentIndexChanged.connect(self._on_frequency_changed)

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

        side_panel = QWidget()
        side_panel.setStyleSheet("QWidget { background: #1f1f1f; color: white; }")
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(12, 12, 12, 12)
        form = QFormLayout()
        form.addRow("Frequency", self.frequency_combo)
        side_layout.addLayout(form)
        side_layout.addSpacing(14)
        side_layout.addWidget(ColorLegend(self._min_db, self._max_db, side_panel))
        side_layout.addStretch(1)
        side_panel.setFixedWidth(220)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(viewport, stretch=1)
        layout.addWidget(side_panel)

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

        self._render_frequency(0, reset_camera=True)
        self.loading_label.hide()

    @Slot(int)
    def _on_frequency_changed(self, index: int) -> None:
        self._render_frequency(index, reset_camera=False)

    def _render_frequency(self, index: int, *, reset_camera: bool) -> None:
        if index < 0 or self._prepared is None:
            return

        x = self._prepared["balloon_x"][index]
        y = self._prepared["balloon_y"][index]
        z = self._prepared["balloon_z"][index]
        spl = self._prepared["balloon_surface_spl"][index]
        mesh = self._pv.StructuredGrid(x, y, z)
        mesh[SPL_SCALAR_NAME] = spl.ravel(order="F")

        self.plotter.clear()
        self.plotter.add_mesh(
            mesh,
            scalars=SPL_SCALAR_NAME,
            cmap="turbo",
            clim=(self._min_db, self._max_db),
            smooth_shading=True,
            show_scalar_bar=False,
        )
        self._add_spl_contours(mesh)
        self._add_orientation_guides(mesh)
        self.plotter.add_axes()
        self.plotter.enable_anti_aliasing()
        if reset_camera:
            self.plotter.reset_camera()
            self.plotter.camera_position = "iso"
        self.plotter.render()

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

    def closeEvent(self, event) -> None:
        self.plotter.close()
        super().closeEvent(event)


def _format_frequency(freq_hz: float) -> str:
    if freq_hz >= 1000.0:
        return f"{_format_decimal(freq_hz / 1000.0)} kHz"
    return f"{_format_decimal(freq_hz)} Hz"


def _format_decimal(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


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
    return [float(level) for level in levels if min_db < level <= max_db]


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
