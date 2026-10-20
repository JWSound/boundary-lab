from types import SimpleNamespace

import numpy as np
from PySide6.QtWidgets import QCheckBox, QWidget

from blab.ui import interface_preview as preview_module


def test_pair_preview_preserves_positions_and_toggles_each_side(qapp, monkeypatch):
    class Actor:
        visible = True

        def SetVisibility(self, value):
            self.visible = value

    class Viewer(QWidget):
        def __init__(self, parent):
            super().__init__(parent)
            self.meshes = []
            self.points = []
            self.camera_reset = False
            self.closed = False

        def add_mesh(self, data, **options):
            actor = Actor()
            self.meshes.append((data, options, actor))
            return actor

        def add_points(self, points, **options):
            self.points.append(points)

        def set_background(self, _color):
            pass

        def add_axes(self):
            pass

        def view_isometric(self):
            pass

        def reset_camera(self):
            self.camera_reset = True

        def render(self):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setattr(preview_module, "QtInteractor", Viewer)
    monkeypatch.setattr(preview_module, "_triangles_to_polydata", lambda points, faces: points)
    monkeypatch.setattr(preview_module, "_line_segments_to_polydata", lambda segments: segments)
    points = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    fem = SimpleNamespace(points=points, triangles=np.array([[0, 1, 2]]), perimeter_segments=points[[[0, 1]]])
    bem = SimpleNamespace(
        points=points + [0, 0, 0.01],
        triangles=fem.triangles,
        perimeter_segments=fem.perimeter_segments + [0, 0, 0.01],
    )
    witness = np.array([points[0], bem.points[0]])
    report = SimpleNamespace(fem=fem, bem=bem, discrepancy_segment=witness, sampled_perimeter_deviation_m=0.01)
    dialog = preview_module.InterfacePreviewDialog(report, ("Interior / mouth", "Exterior / opening"))
    viewer = dialog.viewer
    assert viewer.camera_reset
    assert len(viewer.meshes) == 5
    np.testing.assert_array_equal(viewer.meshes[0][0], fem.points)
    np.testing.assert_array_equal(viewer.meshes[2][0], bem.points)
    np.testing.assert_array_equal(viewer.points[0], witness)
    assert viewer.meshes[0][1]["color"] != viewer.meshes[2][1]["color"]
    toggles = dialog.findChildren(QCheckBox)
    toggles[0].setChecked(False)
    assert not viewer.meshes[0][2].visible
    assert not viewer.meshes[1][2].visible
    assert viewer.meshes[2][2].visible
    assert viewer.meshes[4][2].visible
    dialog.reject()
    assert viewer.closed


def test_pair_preview_handles_missing_visual_dependencies(qapp, monkeypatch):
    monkeypatch.setattr(preview_module, "QtInteractor", None)
    dialog = preview_module.InterfacePreviewDialog(None, ("FEM", "BEM"))
    assert dialog.viewer is None
    dialog.reject()
