from blab.config import MeshConfig
from blab.physical_model import (
    AcousticRegion,
    AcousticRegionKind,
    Boundary,
    BoundaryKind,
    MeshPurpose,
    MeshResource,
    PhysicalGroupRef,
    PhysicalSystem,
)
from blab.preview_hierarchy import build_preview_hierarchy


def test_physical_system_builds_region_mesh_boundary_hierarchy_with_solver_actor_keys() -> None:
    system = PhysicalSystem(
        id="system",
        name="System",
        meshes=(MeshResource("bem", "Cabinet", "cabinet.msh", MeshPurpose.BEM_SURFACE),),
        regions=(AcousticRegion("outside", "Exterior", AcousticRegionKind.UNBOUNDED_AIR, ("bem",)),),
        boundaries=(
            Boundary(
                "woofer",
                "Woofer",
                "outside",
                PhysicalGroupRef("bem", 2, name="Woofer Surface"),
                BoundaryKind.MOVING,
            ),
            Boundary(
                "cabinet",
                "Cabinet Walls",
                "outside",
                PhysicalGroupRef("bem", 2, tag=2),
                BoundaryKind.RIGID,
            ),
        ),
    )

    hierarchy = build_preview_hierarchy(
        system,
        source_mesh_configs=(MeshConfig("Cabinet", "cabinet.msh"),),
        source_surface_tags_by_mesh={"Cabinet": {"Woofer Surface": 1, "Walls": 2}},
        solver_surface_by_source={
            ("Cabinet", 1): ("stitched", 7),
            ("Cabinet", 2): ("stitched", 8),
        },
    )

    assert [region.name for region in hierarchy.regions] == ["Exterior"]
    mesh = hierarchy.regions[0].meshes[0]
    assert mesh.name == "Cabinet"
    assert [(node.name, node.surface_keys) for node in mesh.boundaries] == [
        ("Woofer", (("stitched", 7),)),
        ("Cabinet Walls", (("stitched", 8),)),
    ]


def test_unconfigured_preview_uses_synthetic_exterior_region_and_physical_groups() -> None:
    hierarchy = build_preview_hierarchy(
        None,
        source_mesh_configs=(MeshConfig("Waveguide", "waveguide.msh"),),
        source_surface_tags_by_mesh={"Waveguide": {"Throat": 2, "Walls": 1}},
        solver_surface_by_source={
            ("Waveguide", 1): ("Waveguide", 1),
            ("Waveguide", 2): ("Waveguide", 2),
        },
    )

    assert hierarchy.regions[0].name == "Exterior"
    mesh = hierarchy.regions[0].meshes[0]
    assert [boundary.name for boundary in mesh.boundaries] == ["Walls", "Throat"]
    assert hierarchy.surface_keys == (("Waveguide", 1), ("Waveguide", 2))

