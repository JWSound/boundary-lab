from dataclasses import replace

import meshio
import numpy as np
import pytest

from blab.physical_model import (
    AcousticRegion,
    AcousticRegionKind,
    MeshPurpose,
    MeshResource,
    PhysicalGroupRef,
    PhysicalSystem,
)
from blab.pressure_unknowns import pressure_unknown_counts


@pytest.mark.parametrize(("cell_type", "width"), [("tetra", 4), ("tetra10", 10)])
def test_only_active_assigned_pressure_nodes_are_counted(cell_type, width):
    surface = MeshResource("s", "surface", "", MeshPurpose.BEM_SURFACE)
    volume = MeshResource("v", "volume", "", MeshPurpose.FEM_VOLUME)
    spare = replace(surface, id="spare", name="unassigned")
    disabled = replace(surface, id="d", name="disabled")
    outside = AcousticRegion("e", "outside", AcousticRegionKind.UNBOUNDED_AIR, ("s", "d"))
    group = PhysicalGroupRef("v", 3, name="chamber")
    inside = AcousticRegion("i", "inside", AcousticRegionKind.BOUNDED_AIR, ("v",), (group,))
    system = PhysicalSystem("p", "system", (surface, volume, spare, disabled), (outside, inside), ())
    shell = meshio.Mesh(np.zeros((4, 3)), [("triangle", [[0, 1, 2], [1, 2, 3]])])
    mesh = meshio.Mesh(
        np.zeros((width * 3, 3)),
        [(cell_type, [np.arange(width), np.arange(1, width + 1), np.arange(width * 2, width * 3)])],
        cell_data={"gmsh:physical": [[3, 3, 4]]},
        field_data={"chamber": [3, 3]},
    )
    meshes = {"volume": mesh, "surface": shell, "unassigned": shell, "disabled": shell}
    active = {"volume", "surface", "unassigned"}
    assert pressure_unknown_counts(system, meshes, active) == (4, width + 1)
    assert pressure_unknown_counts(system, meshes, active, exterior_ready=False) == (None, width + 1)
    assert pressure_unknown_counts(system, meshes, active - {"volume"}) == (4, 0)
    assert pressure_unknown_counts(replace(system, regions=(inside,)), meshes, active) == (0, width + 1)
    broken = replace(inside, volume_groups=(replace(group, name="missing"),))
    assert pressure_unknown_counts(replace(system, regions=(outside, broken)), meshes, active) == (4, None)


def test_stitched_pressure_nodes_use_prepared_connectivity_and_source_membership():
    resource = MeshResource("joined", "stitched", "", MeshPurpose.BEM_SURFACE)
    region = AcousticRegion("outside", "outside", AcousticRegionKind.UNBOUNDED_AIR, ("joined",))
    preparation = {
        "region_id": "outside",
        "sources": [{"id": "a", "name": "left"}, {"id": "b", "name": "right"}],
        "surface_map": [
            {"mesh_id": "a", "assembled_name": "left-face"},
            {"mesh_id": "b", "assembled_name": "right-face"},
        ],
    }
    system = PhysicalSystem("p", "system", (resource,), (region,), (), metadata={"exterior_preparation": [preparation]})
    mesh = meshio.Mesh(
        np.zeros((4, 3)),
        [("triangle", [[0, 1, 2], [1, 2, 3]])],
        field_data={"left-face": [1, 2], "right-face": [2, 2]},
        cell_data={"gmsh:physical": [[1, 2]]},
    )
    assert pressure_unknown_counts(system, {"stitched": mesh}, {"left", "right"}) == (4, 0)
    assert pressure_unknown_counts(system, {"stitched": mesh}, {"left"}) == (3, 0)
    assert pressure_unknown_counts(system, {}, set()) == (0, 0)
