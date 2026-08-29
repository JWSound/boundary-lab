from pathlib import Path

import meshio
import numpy as np

import blab.solvers.exterior_compatibility as compatibility_module
from blab.config import ChannelConfig, MeshConfig, RadiatorConfig, SimulationConfig
from blab.solvers.base import FrequencyResult, SolveMetadata
from blab.solvers.exterior_compatibility import ExteriorCompatibilitySession
from blab.system_solve import prepare_system_ui_solve, with_exterior_compatibility
from blab.ui.physical_system_migration import seed_exterior_system_from_solver_inputs
from blab.ui.system_solve import SystemSolveWorker


def _compatibility_fixture(tmp_path: Path):
    mesh_path = tmp_path / "two_source.msh"
    mesh = meshio.Mesh(
        points=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
            ]
        ),
        cells=[("triangle", np.asarray([[0, 1, 2], [3, 4, 5]]))],
        cell_data={
            "gmsh:physical": [np.asarray([2, 3])],
            "gmsh:geometrical": [np.asarray([2, 3])],
        },
        field_data={"Source A": np.asarray([2, 2]), "Source B": np.asarray([3, 2])},
    )
    meshio.write(mesh_path, mesh, file_format="gmsh22", binary=False)
    mesh_config = MeshConfig("exterior", str(mesh_path), scale_factor=1.0)
    radiators = (
        RadiatorConfig("exterior:Source A", 2, mesh="exterior", channel="main"),
        RadiatorConfig("exterior:Source B", 3, mesh="exterior", channel="main"),
    )
    system, component_channels = seed_exterior_system_from_solver_inputs((mesh_config,), radiators)
    prepared = prepare_system_ui_solve(
        system,
        freq_min_hz=100.0,
        freq_max_hz=100.0,
        freq_count=1,
        observation_distance_m=2.0,
        polar_angle_step_deg=180.0,
        component_channel_by_id=component_channels,
        backend_id="local",
        allow_exterior_compatibility=True,
    )
    config = SimulationConfig(
        mesh_file=str(mesh_path),
        meshes=(mesh_config,),
        radiators=radiators,
        channels=(ChannelConfig("main"),),
        freq_min=100.0,
        freq_max=100.0,
        freq_count=1,
        distance=2.0,
        step_size=180.0,
    )
    return with_exterior_compatibility(
        prepared,
        config=config,
        server_url="http://solver.example.test",
        server_access_token="secret",
    )


def test_compatibility_plan_maps_a_channel_group_to_one_physical_port(tmp_path: Path) -> None:
    prepared = _compatibility_fixture(tmp_path)

    assert prepared.backend_id == "local"
    assert prepared.compatibility is not None
    assert len(prepared.request.compiled_system.excitation_ports) == 2
    assert len(prepared.request.excitation_port_ids) == 1
    assert prepared.excitation_channel_names.tolist() == ["main"]
    assert prepared.compatibility.excitation_port_id_by_channel == (
        ("main", prepared.request.excitation_port_ids[0]),
    )
    assert "secret" not in repr(prepared.compatibility)
    assert prepared.request.solver_options["compatibility_excitation_basis"] == "channel_group"


def test_compatibility_session_returns_canonical_and_live_results(monkeypatch, tmp_path: Path) -> None:
    prepared = _compatibility_fixture(tmp_path)
    live = FrequencyResult(
        freq_hz=100.0,
        horizontal_spl_norm_db=np.zeros(3, dtype=np.float32),
        vertical_spl_norm_db=np.zeros(3, dtype=np.float32),
        impedance=np.asarray([[1.0, 2.0]], dtype=np.float32),
        channel_names=np.asarray(["main"]),
        horizontal_pressure=np.ones((1, 3), dtype=np.complex64),
        vertical_pressure=np.ones((1, 3), dtype=np.complex64),
    )
    captured = {}

    class Session:
        metadata = SolveMetadata(
            polar_angle_deg=np.asarray([-180.0, 0.0, 180.0]),
            radiator_names=np.asarray(["Source A", "Source B"]),
            sphere_metadata=None,
        )

        @staticmethod
        def solve_stream(*, stop_requested=None):
            del stop_requested
            yield live

        @staticmethod
        def stop():
            pass

    class Backend:
        @staticmethod
        def create_session(request):
            captured["request"] = request
            return Session()

    def create_backend(backend_id, **kwargs):
        captured["backend_id"] = backend_id
        captured["kwargs"] = kwargs
        return Backend()

    monkeypatch.setattr(compatibility_module, "create_backend", create_backend)
    session = ExteriorCompatibilitySession(prepared.request, prepared.compatibility)

    (result,) = tuple(session.solve_stream())

    assert captured["backend_id"] == "local"
    assert captured["kwargs"]["server_access_token"] == "secret"
    assert tuple(captured["request"].frequencies_hz) == (100.0,)
    assert result.live is live
    assert result.canonical.excitation_port_ids == prepared.request.excitation_port_ids
    assert result.canonical.quantities[0].axes == ("excitation", "observation")

    canonical_results = []
    live_results = []
    initialized = []
    worker = SystemSolveWorker(prepared)
    worker.system_result_ready.connect(canonical_results.append)
    worker.result_ready.connect(live_results.append)
    worker.initialized.connect(lambda *values: initialized.append(values))

    worker.run()

    assert canonical_results[0].excitation_port_ids == prepared.request.excitation_port_ids
    assert live_results == [live]
    assert initialized[0][1].tolist() == ["Source A", "Source B"]
