from dataclasses import replace

import meshio
import numpy as np
import pytest

from blab.mesh_data import MeshData
from blab.mesh_quality import inspect_near_coincident_vertices, repair_near_coincident_vertices
from blab.physical_model import AcousticRegion, AcousticRegionKind, MeshPurpose, MeshResource, PhysicalSystem
from blab.project.model import ImportedMeshState
from blab.ui.mesh_assembly import MeshAssemblyService


def sliver_system(*, scale=1.0, tiny_group=False):
    # A closed tetrahedron with one edge split almost at its endpoint.
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1e-7, 0, 0]]) / scale
    triangles = [[0, 2, 4], [4, 2, 1], [0, 4, 3], [4, 1, 3], [0, 3, 2], [1, 2, 3]]
    tags = np.array([2 if tiny_group else 1, 1, 1, 1, 1, 1])
    names = {"wall": [1, 2], **({"tiny radiator": [2, 2]} if tiny_group else {})}
    mesh = meshio.Mesh(points, [("triangle", triangles)], cell_data={"gmsh:physical": [tags]}, field_data=names)
    resource = MeshResource(
        "mesh:test", "test", "", MeshPurpose.BEM_SURFACE, scale, mesh_data=MeshData.from_meshio(mesh)
    )
    system = PhysicalSystem(
        "system:test",
        "Test",
        (resource,),
        (AcousticRegion("exterior", "Exterior", AcousticRegionKind.UNBOUNDED_AIR, (resource.id,)),),
        (),
    )
    return system


@pytest.mark.parametrize("scale", [1.0, 0.001])
def test_scale_aware_detection_and_cleanup_preserve_closed_surface(scale):
    system = sliver_system(scale=scale)
    (issue,) = inspect_near_coincident_vertices(system)
    assert issue.vertex_count == 2
    assert issue.minimum_distance_m == pytest.approx(1e-7)
    assert issue.merge_tolerance_m == pytest.approx(8 * np.finfo(np.float32).eps)
    cleaned = repair_near_coincident_vertices(system.meshes[0], issue, symmetry="off")
    assert len(cleaned.points) == 4
    assert len(cleaned.cells[0].data) == 4
    repaired = replace(system.meshes[0], mesh_data=MeshData.from_meshio(cleaned))
    assert not inspect_near_coincident_vertices(replace(system, meshes=(repaired,)))


def test_fem_meshes_are_not_offered_surface_cleanup():
    system = sliver_system()
    system = replace(system, meshes=(replace(system.meshes[0], purpose=MeshPurpose.FEM_VOLUME),))
    assert not inspect_near_coincident_vertices(system)


def test_cleanup_rejects_removing_a_physical_surface():
    system = sliver_system(tiny_group=True)
    (issue,) = inspect_near_coincident_vertices(system)
    with pytest.raises(ValueError, match="preserve every physical surface"):
        repair_near_coincident_vertices(system.meshes[0], issue, symmetry="off")


def test_separate_resources_are_not_compared_or_merged():
    system = sliver_system()
    (issue,) = inspect_near_coincident_vertices(system)
    cleaned = repair_near_coincident_vertices(system.meshes[0], issue, symmetry="off")
    first = replace(system.meshes[0], mesh_data=MeshData.from_meshio(cleaned))
    second = replace(first, id="mesh:second", name="second")
    system = replace(
        system, meshes=(first, second), regions=(replace(system.regions[0], mesh_ids=(first.id, second.id)),)
    )
    assert not inspect_near_coincident_vertices(system)


@pytest.mark.parametrize("cache_aliases_source", [False, True])
def test_repair_reuses_cleanup_cache_and_preserves_source(tmp_path, cache_aliases_source):
    system = sliver_system()
    source = tmp_path / "source.msh"
    cached = source if cache_aliases_source else tmp_path / "existing_clean.msh"
    meshio.write(source, system.meshes[0].mesh_data.to_meshio(), file_format="gmsh22", binary=False)
    if cached != source:
        cached.write_bytes(source.read_bytes())
    original = source.read_bytes()
    resource = replace(system.meshes[0], file=str(cached), mesh_data=None)
    system = replace(system, meshes=(resource,))
    states = (ImportedMeshState(name="test", source_file=str(source), cleaned_file=str(cached), scale_factor=1.0),)
    service = MeshAssemblyService(tmp_path / "cache")
    repaired, (state,) = service.repair_near_coincident_meshes(
        system, states, inspect_near_coincident_vertices(system), symmetry="off"
    )
    assert source.read_bytes() == original
    assert repaired.meshes[0].file == state.cleaned_file
    assert state.cleaned_file != str(source)
    if not cache_aliases_source:
        assert state.cleaned_file == str(cached)
    assert not inspect_near_coincident_vertices(repaired)
    # A normal cleanup/preview must not replace the accepted repair with the source.
    assert service.clean_imported_meshes((state,)) == (state,)
    assert not inspect_near_coincident_vertices(repaired)


def test_failed_repair_does_not_overwrite_cleanup_cache(tmp_path):
    system = sliver_system(tiny_group=True)
    cached = tmp_path / "clean.msh"
    meshio.write(cached, system.meshes[0].mesh_data.to_meshio(), file_format="gmsh22", binary=False)
    original = cached.read_bytes()
    system = replace(system, meshes=(replace(system.meshes[0], file=str(cached), mesh_data=None),))
    state = ImportedMeshState(name="test", source_file=str(tmp_path / "source.msh"), cleaned_file=str(cached))
    with pytest.raises(ValueError, match="preserve every physical surface"):
        MeshAssemblyService(tmp_path).repair_near_coincident_meshes(
            system, (state,), inspect_near_coincident_vertices(system), symmetry="off"
        )
    assert cached.read_bytes() == original


def test_generated_mesh_repair_stays_in_memory(tmp_path):
    system = sliver_system()
    repaired, imports = MeshAssemblyService(tmp_path).repair_near_coincident_meshes(
        system, (), inspect_near_coincident_vertices(system), symmetry="off"
    )
    assert imports == ()
    assert not list(tmp_path.iterdir())
    assert not inspect_near_coincident_vertices(repaired)
    assert inspect_near_coincident_vertices(system)


@pytest.mark.parametrize("value", [complex(float("nan"), 0), complex(0, float("inf"))])
def test_backend_rejects_nonfinite_output_before_publishing_a_frequency(monkeypatch, value):
    from types import SimpleNamespace

    import blab.solvers.coupled_backend as backend
    from blab.phasor import SOLVER_PHASOR_CONVENTION
    from blab.system_contract import QuantityResult, SystemFrequencyResult

    result = SystemFrequencyResult(1000.0, (QuantityResult("pressure", "pressure", "Pa", np.array([value])),))
    monkeypatch.setattr(backend, "system_frequency_result_from_dict", lambda _raw: result)
    session = SimpleNamespace(request=SimpleNamespace(solver_options={}))
    with pytest.raises(RuntimeError, match="non-finite results.*1000 Hz"):
        backend.CoupledSession._parse_result(session, {"diagnostics": {"phasor_convention": SOLVER_PHASOR_CONVENTION}})
