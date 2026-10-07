from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import vtk

from blab.config import MeshConfig
from blab.physical_model import ComponentKind
from blab.preview_motion import motion_arrow_geometry, preview_motion_assignments
from blab.ui.mesh_preview import MeshPreview
from test_prescribed_source_motion import axial, source_system


def test_assignments_use_saved_axes_and_resolved_surface_tags():
    system = source_system(axial((0, 0, -20)))
    # The assembly may change both the mesh name and its physical tag.
    system = replace(system, meshes=(replace(system.meshes[0], name="Assembled"),))
    result = preview_motion_assignments(system, {"Assembled": {"Drive": 71}})
    assert set(result) == {("Assembled", 71)}
    assert result[("Assembled", 71)].axis == (0, 0, -1)
    component = replace(
        system.components[0],
        kind=ComponentKind.ELECTRODYNAMIC_TRANSDUCER,
        parameters={**axial(), "boundary_motion_signs": {"Drive": -1}},
    )
    result = preview_motion_assignments(replace(system, components=(component,)), {"Assembled": {"Drive": 71}})
    assert result[("Assembled", 71)].axis == (0, 0, -1)
    normal = preview_motion_assignments(source_system({}), {"Mesh": {"Drive": 1}})
    assert normal[("Mesh", 1)].axis is None
    invalid = preview_motion_assignments(source_system(axial((0, 0, 0))), {"Mesh": {"Drive": 1}})
    assert invalid == {}


def test_sparse_arrows_are_parallel_and_inward_tips_remain_above_surface():
    system = source_system(axial(), flat=True)
    mesh = system.meshes[0].mesh_data
    triangles = mesh.cells[0][1][:4]
    origins, vectors = motion_arrow_geometry(mesh.points, triangles, (0, 0, -1))
    assert len(origins) == 4
    np.testing.assert_allclose(vectors[:, :2], 0)
    assert np.all(vectors[:, 2] < 0)
    assert np.all((origins + vectors)[:, 2] > 0)
    duplicate_triangles = np.tile(triangles, (5000, 1))
    sparse, _ = motion_arrow_geometry(mesh.points, duplicate_triangles, (0, 0, 1))
    assert len(sparse) <= 32
    empty, _ = motion_arrow_geometry(np.zeros((3, 3)), [[0, 1, 2]], (0, 0, 1))
    assert empty.shape == (0, 3)


class PreviewHarness:
    """Exercise real scene assembly and glyphs without a native render window."""

    load_mesh_configs = MeshPreview.load_mesh_configs
    _add_msh_mesh = MeshPreview._add_msh_mesh
    _register_mesh_actor = MeshPreview._register_mesh_actor
    _queue_motion_arrows = MeshPreview._queue_motion_arrows
    _build_motion_arrows = MeshPreview._build_motion_arrows
    _apply_actor_visibility = MeshPreview._apply_actor_visibility
    set_motion_directions_visible = MeshPreview.set_motion_directions_visible
    clear = MeshPreview.clear

    def __init__(self):
        self.geometry = []
        self._motion_directions_visible = False
        self._surface_visibility = {}
        self._observation_clip_active = False
        self.viewer = SimpleNamespace(add_mesh=self.add_mesh, clear=self.geometry.clear, render=lambda: None)
        self.hover_label = SimpleNamespace(setText=lambda _: None)
        self.motion_legend = SimpleNamespace(setVisible=lambda _: None)
        self._camera_position = lambda: "unchanged"
        self._restore_camera_or_reset = lambda value: None
        self._restore_observation_plane_scene = lambda _: None
        self._set_total_element_count = lambda *args, **kwargs: None
        self._add_orientation_guides = lambda _: None
        self.set_hierarchy = lambda *args, **kwargs: None
        self.set_topology_report = lambda *args, **kwargs: None

    def add_mesh(self, geometry, **options):
        self.geometry.append((geometry, options))
        return vtk.vtkActor()


def test_overlay_visibility_symmetry_and_reload():
    system = source_system(axial((1, 0, 1)))
    resource = system.meshes[0]
    mesh = resource.mesh_data.to_meshio()
    # Shift off the symmetry plane so that mirrored geometry is distinct.
    mesh.points[:, 0] += 0.3
    assignments = preview_motion_assignments(system, {"Mesh": {"Drive": 1}})
    preview = PreviewHarness()
    options = dict(
        loaded_meshes={"Mesh": mesh},
        driven_surfaces={("Mesh", 1)},
        motion_assignments=assignments,
        symmetry="x",
        mesh_regions={"Mesh": "interior"},
    )
    configs = (MeshConfig(name="Mesh", file="", scale_factor=1),)
    preview.load_mesh_configs(configs, **options)
    assert not any(record.motion_arrow for record in preview._actor_records)
    base, reflected = preview._motion_arrow_batches
    np.testing.assert_allclose(reflected[1], base[1] * [-1, 1, 1])
    preview.set_motion_directions_visible(True)
    arrows = [record.actor for record in preview._actor_records if record.motion_arrow]
    assert len(arrows) == 2
    assert all(actor.GetVisibility() and not actor.GetUseBounds() for actor in arrows)
    assert all(options["pickable"] is False for _, options in preview.geometry if "pickable" in options)
    preview._surface_visibility[("Mesh", 1)] = False
    preview._apply_actor_visibility()
    assert not any(actor.GetVisibility() for actor in arrows)
    preview._surface_visibility[("Mesh", 1)] = True
    preview._observation_clip_active = True
    preview._apply_actor_visibility()
    assert not any(actor.GetVisibility() for actor in arrows)
    preview._observation_clip_active = False
    preview.set_motion_directions_visible(False)
    assert not any(actor.GetVisibility() for actor in arrows)
    count = len(preview.geometry)
    preview.set_motion_directions_visible(True)
    assert len(preview.geometry) == count
    preview.load_mesh_configs(configs, **options)
    assert len(preview.geometry) == count
    assert len([record for record in preview._actor_records if record.motion_arrow]) == 2
    preview.clear()
    assert not preview._motion_arrow_batches
    assert not preview._actor_records
    assert preview._motion_directions_visible


def test_view_menu_toggle_controls_preview(main_window):
    action = main_window.motion_directions_action
    assert action.isCheckable()
    assert not action.isChecked()
    action.setChecked(True)
    assert main_window.preview.motion_directions_visible
    action.setChecked(False)
    assert not main_window.preview.motion_directions_visible
