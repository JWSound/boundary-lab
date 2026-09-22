"""Memory transport must preserve geometry, configuration and complex results."""

import json
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import meshio
import numpy as np
import pytest

from blab.exterior_preparation import prepare_exterior_system
from blab.generators.base import GeneratedGeometry, GeneratorDocument
from blab.generators.registry import restore_generator_document
from blab.mesh_data import MeshData
from blab.physical_compiler import PhysicalSystemCompiler
from blab.physical_model import (
    AcousticRegion,
    AcousticRegionKind,
    Boundary,
    BoundaryKind,
    ComponentKind,
    ExcitationPort,
    ExcitationPortKind,
    MeshPurpose,
    MeshResource,
    PhysicalComponent,
    PhysicalGroupRef,
    PhysicalSystem,
    physical_system_from_dict,
    physical_system_to_dict,
)
from blab.project.model import generator_document_to_payload, generator_documents_from_payload
from blab.solvers.coupled_backend import CoupledProductionBackend
from blab.system_contract import OutputRequest, SystemSolveRequest, system_solve_request_to_dict
from blab.ui.mesh_preparation import MeshPreparationSnapshot, prepare_preview


def tetra_surface():
    return MeshData(
        points=[[0.0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
        cells=[("triangle", [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])],
        physical_tags=[[2, 2, 2, 2]],
        physical_names={"radiator": [2, 2]},
    )


def memory_system():
    return PhysicalSystem(
        id="system",
        name="Memory tetrahedron",
        meshes=(MeshResource("mesh", "tetra", "", MeshPurpose.BEM_SURFACE, mesh_data=tetra_surface()),),
        regions=(AcousticRegion("air", "Air", AcousticRegionKind.UNBOUNDED_AIR, ("mesh",)),),
        boundaries=(
            Boundary("surface", "Radiator", "air", PhysicalGroupRef("mesh", 2, "radiator"), BoundaryKind.MOVING),
        ),
        components=(
            PhysicalComponent(
                "source",
                "Source",
                ComponentKind.IDEAL_VELOCITY_SOURCE,
                ("surface",),
                parameters={"motion_profile": "uniform"},
            ),
        ),
        excitation_ports=(ExcitationPort("drive", "Drive", "source", ExcitationPortKind.NORMAL_VELOCITY),),
    )


def test_snapshot_is_immutable_and_wire_roundtrip_is_lossless():
    data = tetra_surface()
    with pytest.raises(ValueError):
        data.points.setflags(write=True)
    with pytest.raises(ValueError):
        data.cells[0][1][0, 0] = 3
    view = data.points
    view.shape = (12,)
    assert data.points.shape == (4, 3)
    detached = data.to_meshio()
    detached.points[:] = 7
    assert data.points[0, 0] == 0
    assert deepcopy(data) is data
    restored = MeshData.from_payload(json.loads(json.dumps(data.to_payload())))
    assert restored.digest == data.digest
    np.testing.assert_array_equal(restored.points, data.points)
    np.testing.assert_array_equal(restored.cells[0][1], data.cells[0][1])


@pytest.mark.parametrize("damage", ["version", "size", "type", "tag", "nan"])
def test_malformed_wire_mesh_is_rejected(damage):
    import base64
    import struct

    payload = tetra_surface().to_payload()
    if damage == "version":
        payload["schema_version"] = True
    elif damage == "size":
        payload["points"]["shape"][0] += 1
    elif damage == "type":
        payload["cells"][0]["type"] = []
    elif damage == "tag":
        payload["physical_names"] = {}
    else:
        values = bytearray(base64.b64decode(payload["points"]["data"]))
        values[:8] = struct.pack("<d", float("nan"))
        payload["points"]["data"] = base64.b64encode(values).decode()
    with pytest.raises(ValueError):
        MeshData.from_payload(payload)


def test_prepare_compile_preview_and_restore_without_mesh_io(monkeypatch, tmp_path):
    system = memory_system()
    geometry = GeneratedGeometry("custom", tmp_path, None, (), mesh_data=system.meshes[0].mesh_data)
    document = GeneratorDocument(
        "design", "tetra", "custom", 1, {}, mesh_scale_factor=1.0, artifact=geometry.to_reference()
    )
    restored_document = generator_documents_from_payload([generator_document_to_payload(document)])[0]
    restored = restore_generator_document(restored_document)
    assert restored.mesh_data.digest == geometry.mesh_data.digest
    restored_system = physical_system_from_dict(json.loads(json.dumps(physical_system_to_dict(system))))

    def forbidden(*args, **kwargs):
        raise AssertionError("Mesh-file I/O on memory path")

    monkeypatch.setattr(meshio, "read", forbidden)
    monkeypatch.setattr(meshio, "write", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    prepared = prepare_exterior_system(restored_system, stitch_tolerance_mm=0.1, output_root=tmp_path / "unused")
    compiled = PhysicalSystemCompiler().compile(prepared)
    wire = system_solve_request_to_dict(SystemSolveRequest(compiled, (100.0,), ("drive",)))
    assert wire["compiled_system"]["meshes"][0]["file"] == ""
    assert wire["compiled_system"]["meshes"][0]["mesh_data"]["schema_version"] == 1
    snapshot = MeshPreparationSnapshot((document,), {document.id: geometry}, (), (), system, "off", False, 0.1)
    prepare_preview(snapshot, tmp_path / "unused")


def test_memory_session_submits_inline_without_tempfiles(monkeypatch):
    import blab.solvers.coupled_backend as backend

    submitted = []

    class Worker:
        worker_info = {}

        def submit(self, payload, **kwargs):
            submitted.append(payload)
            yield {"type": "completed"}

    monkeypatch.setattr(backend, "get_beat_engine_worker", lambda **kwargs: Worker())
    monkeypatch.setattr(backend.tempfile, "TemporaryDirectory", lambda **kwargs: pytest.fail("Temporary file used"))
    request = SystemSolveRequest(PhysicalSystemCompiler().compile(memory_system()), (100.0,), ("drive",))
    assert list(CoupledProductionBackend(bem_backend="cpu").create_system_session(request).solve_stream()) == []
    assert submitted[0]["compiled_system"]["meshes"][0]["mesh_data"]
    assert "cancel_path" not in submitted[0]


@pytest.mark.skipif(os.environ.get("BLAB_TEST_MEMORY_SOLVE") != "1", reason="Opt-in real Julia transport parity")
@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_real_file_and_inline_solve_preserve_complex_results(tmp_path, backend):
    from blab.solvers.beat_engine_backend import shutdown_beat_engine_workers
    from blab.system_contract import system_frequency_result_to_dict

    system = memory_system()
    file = tmp_path / "tetra.msh"
    meshio.write(file, system.meshes[0].mesh_data.to_meshio(), file_format="gmsh22", binary=False)
    file_system = replace(system, meshes=(replace(system.meshes[0], file=str(file), mesh_data=None),))
    results = []
    try:
        for source in (file_system, system):
            compiled = PhysicalSystemCompiler().compile(source)
            request = SystemSolveRequest(
                compiled,
                (100.0, 300.0),
                ("drive",),
                outputs=(
                    OutputRequest("pressure", "bem_boundary_pressure"),
                    OutputRequest("neumann", "bem_boundary_neumann"),
                    OutputRequest("probe", "exterior_pressure", options={"points_m": [[1.0, 1.0, 1.0]]}),
                ),
            )
            session = CoupledProductionBackend(bem_backend=backend).create_system_session(request)
            results.append(list(session.solve_stream()))
        assert len(results[0]) == len(results[1]) == 2
        for old, new in zip(*results, strict=True):
            old_wire = system_frequency_result_to_dict(old)
            new_wire = system_frequency_result_to_dict(new)
            assert old_wire["quantities"] == new_wire["quantities"]
            assert old_wire["quantities"]
            assert old.excitation_port_ids == new.excitation_port_ids
    finally:
        shutdown_beat_engine_workers()


def test_memory_project_save_and_headless_reload(tmp_path):
    from blab.headless import load_headless_project
    from blab.project.io import build_project_payload, write_project_file

    system = memory_system()
    geometry = GeneratedGeometry("custom", tmp_path, None, (), mesh_data=system.meshes[0].mesh_data)
    document = GeneratorDocument(
        "design", "tetra", "custom", 1, {}, mesh_scale_factor=1.0, artifact=geometry.to_reference()
    )
    payload = build_project_payload(
        generator_documents=[generator_document_to_payload(document)],
        active_generator_document_id="design",
        imported_meshes=[],
        source_config_by_name={},
        physical_system=physical_system_to_dict(system),
    )
    path = write_project_file(tmp_path / "memory.blab.json", payload)
    loaded = load_headless_project(path)
    resource = loaded.physical_system.meshes[0]
    assert resource.file == ""
    assert resource.mesh_data.digest == system.meshes[0].mesh_data.digest
    assert resource.scale_to_m == 1.0
    PhysicalSystemCompiler().compile(loaded.physical_system)


def test_memory_cancel_retires_worker_without_marker_file(monkeypatch):
    import blab.solvers.coupled_backend as backend

    events = []

    class Worker:
        worker_info = {}

        def submit(self, payload, **kwargs):
            yield {"type": "status", "message": "solving"}
            yield {"type": "completed"}

        def terminate(self):
            events.append("terminated")

    monkeypatch.setattr(backend, "get_beat_engine_worker", lambda **kwargs: Worker())
    monkeypatch.setattr(Path, "touch", lambda *args, **kwargs: pytest.fail("Cancellation wrote a file"))
    request = SystemSolveRequest(PhysicalSystemCompiler().compile(memory_system()), (100.0,), ("drive",))
    session = CoupledProductionBackend(bem_backend="cpu").create_system_session(request)
    assert list(session.solve_stream(stop_requested=lambda: True)) == []
    assert events == ["terminated"]


def test_geometry_rejects_mixed_sources(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        GeneratedGeometry("custom", tmp_path, tmp_path / "mesh.msh", (), mesh_data=tetra_surface())
    with pytest.raises(ValueError, match="primary"):
        GeneratedGeometry("custom", tmp_path, tmp_path / "mesh.msh", (), reduced_mesh_data=tetra_surface())
