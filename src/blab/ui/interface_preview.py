"""Focused interface inspection using the mesh preview's geometry primitives."""

from PySide6.QtWidgets import QCheckBox, QDialog, QHBoxLayout, QLabel, QVBoxLayout

from blab.interface_geometry import InterfaceGeometry
from blab.ui.mesh_preview import QtInteractor, _line_segments_to_polydata, _triangles_to_polydata


class InterfacePreviewDialog(QDialog):
    def __init__(self, geometry: InterfaceGeometry, labels: tuple[str, str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interface pair — source surfaces")
        self.resize(850, 650)
        layout = QVBoxLayout(self)
        note = QLabel(
            "Original positions and scale. Red marks the largest sampled perimeter mismatch. "
            "Toggle either side to inspect coincident surfaces; drag to rotate and scroll to zoom."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        sources = QLabel(f"FEM: {labels[0]}\nBEM: {labels[1]}")
        sources.setWordWrap(True)
        layout.addWidget(sources)
        self.viewer = None
        if QtInteractor is None:
            layout.addWidget(QLabel("Install pyvista and pyvistaqt to enable the 3D preview."))
            return
        self.viewer = QtInteractor(self)
        controls = QHBoxLayout()
        layout.addLayout(controls)
        layout.addWidget(self.viewer)
        self.viewer.set_background("#252525")
        for name, patch, color, label in (
            ("FEM (blue)", geometry.fem, "#36b7ff", labels[0]),
            ("BEM (orange)", geometry.bem, "#ffac42", labels[1]),
        ):
            actors = [
                self.viewer.add_mesh(
                    _triangles_to_polydata(patch.points, patch.triangles),
                    color=color,
                    opacity=0.45,
                    show_edges=True,
                    edge_color=color,
                    lighting=False,
                    render=False,
                )
            ]
            if len(patch.perimeter_segments):
                actors.append(
                    self.viewer.add_mesh(
                        _line_segments_to_polydata(patch.perimeter_segments),
                        color=color,
                        line_width=4,
                        render_lines_as_tubes=True,
                        lighting=False,
                        render=False,
                    )
                )
            toggle = QCheckBox(name)
            toggle.setToolTip(label)
            toggle.setChecked(True)
            toggle.toggled.connect(lambda checked, items=actors: self._set_visible(items, checked))
            controls.addWidget(toggle)
        controls.addStretch()
        witness = geometry.discrepancy_segment
        if witness is not None and geometry.sampled_perimeter_deviation_m > 0:
            self.viewer.add_mesh(
                _line_segments_to_polydata(witness[None, ...]),
                color="#ff2020",
                line_width=6,
                render_lines_as_tubes=True,
                lighting=False,
                render=False,
            )
            self.viewer.add_points(
                witness,
                color="#ff2020",
                point_size=12,
                render_points_as_spheres=True,
                render=False,
            )
        self.viewer.add_axes()
        self.viewer.view_isometric()
        self.viewer.reset_camera()

    def _set_visible(self, actors, visible):
        for actor in actors:
            actor.SetVisibility(visible)
        self.viewer.render()

    def done(self, result):
        if self.viewer is not None:
            self.viewer.close()
        super().done(result)
