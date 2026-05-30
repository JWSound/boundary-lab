import numpy as np
import os
import sys

from blab.config import SimulationConfig
from blab.solvers.base import SolveRequest
from blab.solvers.julia_local_backend import _resolve_julia_threads, shutdown_julia_workers
from blab.solvers.registry import (
    available_backend_infos,
    backend_info,
    backend_label_to_id,
    create_backend,
    normalize_backend_id,
)


def test_solver_backend_registry_keeps_legacy_ids_available() -> None:
    labels = backend_label_to_id()

    assert labels["Server"] == "server"
    assert labels["Julia CUDA GPU"] == "julia_local"
    assert labels["Bempp OpenCL CPU"] == "local"
    assert normalize_backend_id("bempp_local") == "local"
    assert normalize_backend_id("bempp_server") == "server"
    assert normalize_backend_id("local_julia") == "julia_local"
    assert backend_info("server").capabilities.is_remote is True
    assert "julia_local" in {info.backend_id for info in available_backend_infos()}


def test_local_backend_factory_exposes_contract_metadata() -> None:
    backend = create_backend("local", server_url="http://ignored.example")
    request = SolveRequest(
        config=SimulationConfig(mesh_file="mesh.msh"),
        frequencies_hz=np.array([1000.0], dtype=np.float32),
    )

    assert backend.backend_id == "local"
    assert backend.capabilities.supports_streaming is True
    assert backend.capabilities.is_remote is False
    assert backend.create_session.__name__ == "create_session"
    assert request.frequencies_hz.tolist() == [1000.0]


def test_server_and_julia_backend_factories_expose_contract() -> None:
    server_backend = create_backend(
        "server",
        server_url="http://example.test",
        julia_executable="ignored",
    )
    assert server_backend.backend_id == "server"
    assert server_backend.capabilities.is_remote is True

    julia_backend = create_backend("julia_local")
    assert julia_backend.backend_id == "julia_local"
    assert julia_backend.capabilities.is_remote is False
    assert julia_backend.capabilities.supports_parallel_workers is False


def test_julia_backend_consumes_ndjson_solver_contract(tmp_path) -> None:
    mesh_path = tmp_path / "mesh.msh"
    mesh_path.write_text("mesh", encoding="utf-8")
    fake_solver = tmp_path / "fake_julia_solver.py"
    fake_solver.write_text(
        """
import json
import os
import sys

request_path = sys.argv[sys.argv.index("--request") + 1]
with open(request_path, "r", encoding="utf-8") as handle:
    request = json.load(handle)

print(json.dumps({
    "type": "initialized",
    "polar_angle_deg": [-10.0, 0.0, 10.0],
    "radiator_names": ["Woofer"],
    "sphere_metadata": None,
}), flush=True)
print(json.dumps({
    "type": "result",
    "result": {
        "freq_hz": request["frequencies_hz"][0],
        "horizontal_spl_norm_db": [-1.0, 0.0, -1.5],
        "vertical_spl_norm_db": [-2.0, 0.0, -2.5],
        "impedance": [[6.0, 1.0]],
        "horizontal_spl_db": [91.0, 92.0, 90.5],
        "vertical_spl_db": [90.0, 92.0, 89.5],
        "sphere_spl_norm_db": None,
        "timings": {"assembly_s": 0.1, "solve_s": 0.2, "field_s": 0.3},
        "diagnostics": {"convergence_info": 0, "message": os.environ.get("JULIA_NUM_THREADS")},
    },
}), flush=True)
print(json.dumps({"type": "completed", "solved_count": 1}), flush=True)
""".strip(),
        encoding="utf-8",
    )

    backend = create_backend(
        "julia_local",
        julia_executable=sys.executable,
        solver_script=str(fake_solver),
        julia_threads=3,
        julia_project=None,
        persistent_worker=False,
    )
    session = backend.create_session(
        SolveRequest(
            config=SimulationConfig(mesh_file=str(mesh_path)),
            frequencies_hz=np.array([1000.0], dtype=np.float32),
        )
    )

    assert session.metadata.polar_angle_deg.tolist() == [-10.0, 0.0, 10.0]
    assert session.metadata.radiator_names.tolist() == ["Woofer"]

    results = list(session.solve_stream())
    assert len(results) == 1
    assert results[0].freq_hz == 1000.0
    assert results[0].impedance.tolist() == [[6.0, 1.0]]
    assert results[0].timings.assembly_s == 0.1
    assert results[0].diagnostics.message == "3"


def test_julia_threads_auto_maps_to_cpu_count() -> None:
    assert _resolve_julia_threads("auto") == str(os.cpu_count() or 1)
    assert _resolve_julia_threads(16) == "16"
    assert _resolve_julia_threads("bad") == str(os.cpu_count() or 1)


def test_julia_backend_reuses_persistent_worker(tmp_path) -> None:
    mesh_path = tmp_path / "mesh.msh"
    mesh_path.write_text("mesh", encoding="utf-8")
    starts_path = tmp_path / "starts.txt"
    fake_solver = tmp_path / "fake_julia_worker.py"
    fake_solver.write_text(
        f"""
import json
import pathlib
import sys

starts_path = pathlib.Path({str(starts_path)!r})
starts = int(starts_path.read_text(encoding="utf-8")) if starts_path.exists() else 0
starts_path.write_text(str(starts + 1), encoding="utf-8")

if "--worker" not in sys.argv:
    raise SystemExit("expected --worker")

print(json.dumps({{"type": "ready"}}), flush=True)
for line in sys.stdin:
    message = json.loads(line)
    with open(message["request"], "r", encoding="utf-8") as handle:
        request = json.load(handle)
    print(json.dumps({{
        "type": "initialized",
        "polar_angle_deg": [0.0],
        "radiator_names": ["Woofer"],
        "sphere_metadata": None,
    }}), flush=True)
    print(json.dumps({{
        "type": "result",
        "result": {{
            "freq_hz": request["frequencies_hz"][0],
            "horizontal_spl_norm_db": [0.0],
            "vertical_spl_norm_db": [0.0],
            "impedance": [[6.0, 1.0]],
            "horizontal_spl_db": [90.0],
            "vertical_spl_db": [90.0],
            "sphere_spl_norm_db": None,
            "timings": {{"assembly_s": 0.1, "solve_s": 0.2, "field_s": 0.3}},
            "diagnostics": None,
        }},
    }}), flush=True)
    print(json.dumps({{"type": "completed", "solved_count": 1}}), flush=True)
""".strip(),
        encoding="utf-8",
    )

    try:
        backend = create_backend(
            "julia_local",
            julia_executable=sys.executable,
            solver_script=str(fake_solver),
            julia_project=None,
            persistent_worker=True,
        )
        for freq in (500.0, 1000.0):
            session = backend.create_session(
                SolveRequest(
                    config=SimulationConfig(mesh_file=str(mesh_path)),
                    frequencies_hz=np.array([freq], dtype=np.float32),
                )
            )
            assert len(list(session.solve_stream())) == 1

        assert starts_path.read_text(encoding="utf-8") == "1"
    finally:
        shutdown_julia_workers()


def test_julia_backend_cancel_keeps_persistent_worker_warm(tmp_path) -> None:
    mesh_path = tmp_path / "mesh.msh"
    mesh_path.write_text("mesh", encoding="utf-8")
    starts_path = tmp_path / "cancel_starts.txt"
    fake_solver = tmp_path / "fake_julia_cancel_worker.py"
    fake_solver.write_text(
        f"""
import json
import pathlib
import sys

starts_path = pathlib.Path({str(starts_path)!r})
starts = int(starts_path.read_text(encoding="utf-8")) if starts_path.exists() else 0
starts_path.write_text(str(starts + 1), encoding="utf-8")

print(json.dumps({{"type": "ready"}}), flush=True)
for line in sys.stdin:
    message = json.loads(line)
    with open(message["request"], "r", encoding="utf-8") as handle:
        request = json.load(handle)
    print(json.dumps({{
        "type": "initialized",
        "polar_angle_deg": [0.0],
        "radiator_names": ["Woofer"],
        "sphere_metadata": None,
    }}), flush=True)
    print(json.dumps({{
        "type": "result",
        "result": {{
            "freq_hz": request["frequencies_hz"][0],
            "horizontal_spl_norm_db": [0.0],
            "vertical_spl_norm_db": [0.0],
            "impedance": [[6.0, 1.0]],
            "horizontal_spl_db": [90.0],
            "vertical_spl_db": [90.0],
            "sphere_spl_norm_db": None,
            "timings": {{"assembly_s": 0.1, "solve_s": 0.2, "field_s": 0.3}},
            "diagnostics": None,
        }},
    }}), flush=True)
    if pathlib.Path(request["cancel_path"]).exists():
        print(json.dumps({{"type": "cancelled", "solved_count": 1}}), flush=True)
    else:
        print(json.dumps({{"type": "completed", "solved_count": 1}}), flush=True)
""".strip(),
        encoding="utf-8",
    )

    try:
        backend = create_backend(
            "julia_local",
            julia_executable=sys.executable,
            solver_script=str(fake_solver),
            julia_project=None,
            persistent_worker=True,
        )
        cancelled_session = backend.create_session(
            SolveRequest(
                config=SimulationConfig(mesh_file=str(mesh_path)),
                frequencies_hz=np.array([500.0], dtype=np.float32),
            )
        )
        assert list(cancelled_session.solve_stream(stop_requested=lambda: True)) == []

        completed_session = backend.create_session(
            SolveRequest(
                config=SimulationConfig(mesh_file=str(mesh_path)),
                frequencies_hz=np.array([1000.0], dtype=np.float32),
            )
        )
        assert len(list(completed_session.solve_stream())) == 1
        assert starts_path.read_text(encoding="utf-8") == "1"
    finally:
        shutdown_julia_workers()
