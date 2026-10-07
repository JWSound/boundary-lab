from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import vtk

from blab.config import MeshConfig
from blab.physical_model import ComponentKind
from blab.preview_motion import motion_arrow_anchor, motion_arrow_geometry, preview_motion_assignments
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


def test_one_arrow_is_winding_independent_and_points_in_assigned_direction():
    mesh = source_system(axial(), flat=True).meshes[0].mesh_data
    triangles = mesh.cells[0][1][:4]
    anchor = motion_arrow_anchor(mesh.points, triangles)
    reversed_anchor = motion_arrow_anchor(mesh.points, triangles[:, ::-1])
    for axis in ((0, 0, 1), (0, 0, -1)):
        origins, vectors = motion_arrow_geometry(anchor, axis, (0, 0, 1))
        other_origins, other_vectors = motion_arrow_geometry(reversed_anchor, axis, (0, 0, 1))
        assert origins.shape == (1, 3)
        np.testing.assert_allclose(origins, other_origins)
        np.testing.assert_allclose(vectors, other_vectors)
        assert origins[0, 2] > 0 and (origins + vectors)[0, 2] > 0
        assert np.sign(vectors[0, 2]) == axis[2]
    assert motion_arrow_anchor(np.zeros((3, 3)), [[0, 1, 2]]) is None


class PreviewHarness:
    """Exercise real scene assembly and glyphs without a native render window."""

    load_mesh_configs = MeshPreview.load_mesh_configs
    _add_msh_mesh = MeshPreview._add_msh_mesh
    _register_mesh_actor = MeshPreview._register_mesh_actor
    _queue_motion_arrows = MeshPreview._queue_motion_arrows
    _build_motion_arrows = MeshPreview._build_motion_arrows
    _update_motion_arrows = MeshPreview._update_motion_arrows
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
        self._camera_position = lambda: "unchanged"
        self._restore_camera_or_reset = lambda value: None
        self._restore_observation_plane_scene = lambda _: None
        self._set_total_element_count = lambda *args, **kwargs: None
        self._add_orientation_guides = lambda _: None
        self.set_hierarchy = lambda *args, **kwargs: None
        self.set_topology_report = lambda *args, **kwargs: None

    def add_mesh(self, geometry, **options):
        self.geometry.append((geometry, options))
        actor = vtk.vtkActor()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(geometry)
        actor.SetMapper(mapper)
        return actor


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
    np.testing.assert_allclose(reflected[1], np.asarray(base[1]) * [-1, 1, 1])
    preview.set_motion_directions_visible(True)
    arrows = [record.actor for record in preview._actor_records if record.motion_arrow]
    assert len(arrows) == 1
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
    assert len([record for record in preview._actor_records if record.motion_arrow]) == 1
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


def test_legacy_exterior_driven_surface_gets_one_arrow():
    mesh = source_system({}).meshes[0].mesh_data.to_meshio()
    assignments = preview_motion_assignments(None, {"Waveguide": {"SD1D1001": 1}}, driven_surfaces={("Waveguide", 1)})
    assert assignments[("Waveguide", 1)].axis is None
    preview = PreviewHarness()
    preview.load_mesh_configs(
        (MeshConfig(name="Waveguide", file="", scale_factor=1),),
        loaded_meshes={"Waveguide": mesh},
        motion_assignments=assignments,
        driven_surfaces={("Waveguide", 1)},
        symmetry="xy",
        mesh_regions={"Waveguide": "exterior"},
    )
    preview.set_motion_directions_visible(True)
    assert len(preview._motion_component_actors) == 1
    assert next(iter(preview._motion_component_actors.values())).actor.GetVisibility()


def test_component_arrow_moves_to_another_visible_region_and_tracks_camera():
    import pyvista as pv

    mesh = source_system(axial(), flat=True).meshes[0].mesh_data.to_meshio()
    motion = preview_motion_assignments(source_system(axial()), {"Mesh": {"Drive": 1}})[("Mesh", 1)]
    preview = PreviewHarness()
    preview.viewer.camera = pv.Camera()
    preview.viewer.camera.position = (0, 0, 1)
    preview.load_mesh_configs(
        (MeshConfig(name="FEM", file="", scale_factor=1), MeshConfig(name="BEM", file="", scale_factor=1)),
        loaded_meshes={"FEM": mesh, "BEM": mesh},
        motion_assignments={("FEM", 1): motion, ("BEM", 1): motion},
        mesh_regions={"FEM": "interior", "BEM": "exterior"},
    )
    preview.set_motion_directions_visible(True)
    assert len(preview._motion_component_actors) == 1
    record = next(iter(preview._motion_component_actors.values()))
    assert record.mesh_name == "FEM"
    preview._observation_clip_active = True
    preview._apply_actor_visibility()
    assert record.mesh_name == "BEM" and record.actor.GetVisibility()
    before = record.actor.GetMapper().GetInput().GetBounds()
    preview.viewer.camera.position = (0, 0, -1)
    after = record.actor.GetMapper().GetInput().GetBounds()
    assert after[5] < before[4]
    assert len(preview._motion_component_actors) == 1
