"""Sliders, exact numeric inputs and an inexpensive geometry sketch."""

import math

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .parameters import FIELDS, port_profile, validate


class Sketch(QWidget):
    def __init__(self):
        super().__init__()
        self.source = None
        self.setMinimumHeight(170)
        self.setMaximumHeight(210)
        self.setAccessibleName("Enclosure front and port section preview")

    def paintEvent(self, event):
        if self.source is None:
            return
        p = self.source
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(self.palette().text().color(), 1.5))
        scale = min((self.width() * 0.45 - 20) / p["width_m"], (self.height() - 40) / p["height_m"])
        cx, cy = self.width() * 0.23, self.height() / 2
        w, h = p["width_m"] * scale, p["height_m"] * scale
        painter.drawRect(int(cx - w / 2), int(cy - h / 2), int(w), int(h))
        for diameter, y in (
            (p["driver_diameter_m"], p["driver_y_m"]),
            (2 * port_profile(p)["mouth"][1], p["port_y_m"]),
        ):
            r = diameter / 2 * scale
            painter.drawEllipse(QPointF(cx, cy - y * scale), r, r)
        painter.drawText(10, self.height() - 5, "Front (+Z)")

        profile = port_profile(p)
        length = p["port_length_m"]
        samples = []
        # Sample the same exact circular arcs used by Gmsh.
        for start, end, centre in (
            (profile["throat"], profile["join"], profile["flare_centre"]),
            (profile["join"], profile["mouth"], profile["lip_centre"]),
        ):
            if start == end:
                continue
            if centre is None or p["port_roundover_m"] == 0 and start == profile["join"]:
                samples.extend((start, end))
                continue
            radius = math.hypot(start[0] - centre[0], start[1] - centre[1])
            a = math.atan2(start[1] - centre[1], start[0] - centre[0])
            b = math.atan2(end[1] - centre[1], end[0] - centre[0])
            delta = (b - a + math.pi) % (2 * math.pi) - math.pi
            for i in range(17):
                angle = a + delta * i / 16
                samples.append((centre[0] + radius * math.cos(angle), centre[1] + radius * math.sin(angle)))
        samples = [(-length - z, r) for z, r in reversed(samples)] + samples
        scale = min((self.width() * 0.48 - 20) / length, (self.height() - 50) / (2 * profile["mouth"][1]))
        x0 = self.width() * 0.52
        for sign in (-1, 1):
            path = QPainterPath()
            for i, (z, r) in enumerate(samples):
                point = QPointF(x0 + (z + length) * scale, cy + sign * r * scale)
                path.moveTo(point) if i == 0 else path.lineTo(point)
            painter.drawPath(path)
        painter.drawText(int(x0), self.height() - 5, "Port section → exit")


class EnclosureEditor:
    def __init__(self, parent, host):
        self.host = host
        self.source = {}
        self.revision = ""
        self.widget = QWidget(parent)
        layout = QVBoxLayout(self.widget)
        self.sketch = Sketch()
        layout.addWidget(self.sketch)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self.spins, self.sliders, self.connections = {}, {}, {}
        for title, fields in FIELDS.items():
            page = QWidget()
            form = QVBoxLayout(page)
            for key, label, unit, factor, low, high, default, decimals in fields:
                text = QLabel(label)
                form.addWidget(text)
                row = QHBoxLayout()
                slider = QSlider(Qt.Horizontal)
                slider.setRange(0, 1000)
                slider.setAccessibleName(label)
                spin = QDoubleSpinBox()
                spin.setDecimals(decimals)
                spin.setRange(low, high)
                spin.setSuffix(" " + unit if unit else "")
                spin.setSingleStep(10**-decimals if not unit else 1)
                spin.setValue(default)
                spin.setKeyboardTracking(False)
                spin.setAccessibleName(label)
                text.setBuddy(spin)
                slider.valueChanged.connect(
                    lambda value, s=spin, lo=low, hi=high: s.setValue(lo + (hi - lo) * value / 1000)
                )

                def slot(value, k=key, scale=factor):
                    self.edited(k, value / scale)

                self.connections[key] = slot
                spin.valueChanged.connect(slot)
                row.addWidget(slider, 1)
                row.addWidget(spin)
                form.addLayout(row)
                self.spins[key], self.sliders[key] = spin, slider
            if title == "Transducer geometry":
                note = QLabel(
                    "The surround translates with the cone; the dust cap is a shallow spherical dome. "
                    "After Generate, set Re, Le, Bl, dry Mmd, Cms and Rms in System → Components. "
                    "The initial values are illustrative and are preserved when regenerating."
                )
                note.setWordWrap(True)
                form.addWidget(note)
            if title == "Port":
                note = QLabel(
                    "NFR = length / (2 × flare radius). Zero gives a straight throat. "
                    "Both lips have tangent roundovers. This controls geometry; solve to evaluate tuning."
                )
                note.setWordWrap(True)
                form.addWidget(note)
            form.addStretch()
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.tabs.addTab(scroll, title.replace(" geometry", ""))
        self.status = QLabel("Use Generate to create the meshes. Symmetry must be Off.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.host.services.subscribe(self.event)

    def apply_source(self, source, revision):
        self.source, self.revision = dict(source), revision
        for fields in FIELDS.values():
            for key, _, _, factor, low, high, _, _ in fields:
                spin, slider = self.spins[key], self.sliders[key]
                spin.blockSignals(True)
                slider.blockSignals(True)
                spin.setValue(float(source[key]) * factor)
                slider.setValue(round(1000 * (spin.value() - low) / (high - low)))
                spin.blockSignals(False)
                slider.blockSignals(False)
        self.refresh()

    def edited(self, key, value):
        source = self.source | {key: value}
        try:
            snapshot = self.host.update_source(source, expected_revision=self.revision)
        except (ValueError, RuntimeError) as exc:
            self.status.setText(str(exc))
            return
        self.apply_source(snapshot.source, snapshot.revision)
        self.status.setText("Geometry changed. Use Generate to update the meshes.")

    def refresh(self):
        try:
            p = validate(self.source)
        except ValueError as exc:
            self.summary.setText(str(exc))
            self.sketch.source = None
        else:
            mouth = 2 * port_profile(p)["mouth"][1] * 1000
            throat = 2 * math.sqrt(p["port_area_m2"] / math.pi) * 1000
            self.summary.setText(
                f"Port throat Ø{throat:.1f} mm · mouth Ø{mouth:.1f} mm\n"
                "Sketch only. Generate creates the FEM cavity and exterior BEM mesh."
            )
            self.sketch.source = p
        self.sketch.update()

    def event(self, event):
        if event.kind == "geometry_accepted":
            self.status.setText("Meshes accepted. Review the driver parameters in System → Components before solving.")
        elif event.job is not None:
            self.status.setText(event.job.message)

    def set_operation_state(self, state):
        self.tabs.setEnabled(not state.active)

    def dispose(self):
        for key, spin in self.spins.items():
            spin.valueChanged.disconnect(self.connections[key])


def create_editor(parent, document_host):
    return EnclosureEditor(parent, document_host)
