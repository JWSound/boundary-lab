import numpy as np
import os
import sys

from blab.config import SimulationConfig
from blab.solvers.base import SolveRequest
from blab.solvers.beat_engine_backend import (
    DEFAULT_BEAT_ENGINE_CPU_PROJECT,
    DEFAULT_BEAT_ENGINE_CUDA_PROJECT,
    DEFAULT_BEAT_ENGINE_ROCM_PROJECT,
    BeatEngineBackend,
    BeatEngineRocmBackend,
    _friendly_julia_error,
    _resolve_julia_threads,
    shutdown_beat_engine_workers,
)
from blab.solvers.julia_local_backend import JuliaLocalBackend
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
    assert labels["BEAT Engine (CUDA)"] == "beat_cuda"
    assert labels["BEAT Engine (CPU)"] == "beat_cpu"
    assert labels["BEAT Engine (ROCm)"] == "beat_rocm"
    assert labels["Bempp (OpenCL CPU)"] == "local"
    assert normalize_backend_id("bempp_local") == "local"
    assert normalize_backend_id("bempp_server") == "server"
    assert normalize_backend_id("julia_local") == "beat_cuda"
    assert normalize_backend_id("local_julia") == "beat_cuda"
    assert normalize_backend_id("beat") == "beat_cuda"
    assert normalize_backend_id("beat_engine") == "beat_cuda"
    assert normalize_backend_id("beat_cpu") == "beat_cpu"
    assert normalize_backend_id("beat_rocm") == "beat_rocm"
    assert normalize_backend_id("rocm") == "beat_rocm"
    assert normalize_backend_id("amdgpu") == "beat_rocm"
    assert JuliaLocalBackend is BeatEngineBackend
    assert BeatEngineRocmBackend.beat_engine_backend == "rocm"
    assert backend_info("server").capabilities.is_remote is True
    assert backend_info("server").capabilities.supports_symmetry is False
    assert backend_info("local").capabilities.supports_symmetry is False
    assert backend_info("beat_cuda").capabilities.supports_symmetry is True
    assert backend_info("beat_cpu").capabilities.supports_symmetry is True
    assert backend_info("beat_rocm").capabilities.supports_symmetry is True
    assert "beat_cuda" in {info.backend_id for info in available_backend_infos()}
    assert "beat_cpu" in {info.backend_id for info in available_backend_infos()}
    assert "beat_rocm" in {info.backend_id for info in available_backend_infos()}


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
    assert julia_backend.backend_id == "beat_cuda"
    assert julia_backend.julia_project == DEFAULT_BEAT_ENGINE_CUDA_PROJECT
    assert julia_backend.capabilities.is_remote is False
    assert julia_backend.capabilities.supports_parallel_workers is False
    assert julia_backend.capabilities.supports_symmetry is True

    beat_cpu_backend = create_backend("beat_cpu")
    assert beat_cpu_backend.backend_id == "beat_cpu"
    assert beat_cpu_backend.julia_project == DEFAULT_BEAT_ENGINE_CPU_PROJECT
    assert beat_cpu_backend.capabilities.is_remote is False
    assert beat_cpu_backend.capabilities.supports_parallel_workers is False
    assert beat_cpu_backend.capabilities.supports_symmetry is True

    beat_rocm_backend = create_backend("beat_rocm")
    assert beat_rocm_backend.backend_id == "beat_rocm"
    assert beat_rocm_backend.julia_project == DEFAULT_BEAT_ENGINE_ROCM_PROJECT
    assert beat_rocm_backend.beat_engine_backend == "rocm"
    assert beat_rocm_backend.capabilities.is_remote is False
    assert beat_rocm_backend.capabilities.supports_parallel_workers is False
    assert beat_rocm_backend.capabilities.supports_symmetry is True

    assert BeatEngineBackend().julia_project == DEFAULT_BEAT_ENGINE_CUDA_PROJECT
    assert BeatEngineBackend(beat_engine_backend="cpu").julia_project == DEFAULT_BEAT_ENGINE_CPU_PROJECT
    assert BeatEngineBackend(beat_engine_backend="rocm").julia_project == DEFAULT_BEAT_ENGINE_ROCM_PROJECT


def test_bempp_backend_rejects_symmetry() -> None:
    backend = create_backend("local")
    request = SolveRequest(
        config=SimulationConfig(mesh_file="mesh.msh", symmetry="x"),
        frequencies_hz=np.array([1000.0], dtype=np.float32),
    )

    try:
        backend.create_session(request)
    except RuntimeError as exc:
        assert "does not support symmetry" in str(exc)
    else:
        raise AssertionError("Bempp backend accepted a symmetry solve request.")


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

if request.get("beat_engine_backend") != "cuda":
    raise SystemExit("expected cuda BEAT backend")

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


def test_beat_cpu_backend_passes_cpu_selector_to_julia(tmp_path) -> None:
    mesh_path = tmp_path / "mesh.msh"
    mesh_path.write_text("mesh", encoding="utf-8")
    fake_solver = tmp_path / "fake_beat_cpu_solver.py"
    fake_solver.write_text(
        """
import json
import sys

request_path = sys.argv[sys.argv.index("--request") + 1]
with open(request_path, "r", encoding="utf-8") as handle:
    request = json.load(handle)

if request.get("beat_engine_backend") != "cpu":
    raise SystemExit("expected cpu BEAT backend")

print(json.dumps({
    "type": "initialized",
    "polar_angle_deg": [0.0],
    "radiator_names": ["Woofer"],
    "sphere_metadata": None,
}), flush=True)
print(json.dumps({"type": "completed", "solved_count": 0}), flush=True)
""".strip(),
        encoding="utf-8",
    )

    backend = create_backend(
        "beat_cpu",
        julia_executable=sys.executable,
        solver_script=str(fake_solver),
        julia_project=None,
        persistent_worker=False,
    )
    session = backend.create_session(
        SolveRequest(
            config=SimulationConfig(mesh_file=str(mesh_path)),
            frequencies_hz=np.array([1000.0], dtype=np.float32),
        )
    )

    assert session.metadata.radiator_names.tolist() == ["Woofer"]
    assert list(session.solve_stream()) == []


def test_julia_threads_auto_maps_to_cpu_count() -> None:
    assert _resolve_julia_threads("auto") == str(os.cpu_count() or 1)
    assert _resolve_julia_threads(16) == "16"
    assert _resolve_julia_threads("bad") == str(os.cpu_count() or 1)


def test_julia_dependency_load_error_gets_install_hint() -> None:
    message = """Warm BEAT Engine solver exited with code 1.
ArgumentError: Package CUDA not found in current path.
- Run `import Pkg; Pkg.add("CUDA")` to install the CUDA package.
Stacktrace:
 [6] require(into::Module, mod::Symbol)
 @ Base .\\loading.jl:2388"""

    friendly = _friendly_julia_error(
        message,
        julia_project=DEFAULT_BEAT_ENGINE_CUDA_PROJECT,
        beat_engine_backend="cuda",
    )

    assert "BEAT Engine could not load the Julia dependencies for BEAT Engine (CUDA)." in friendly
    assert "julia --project=" in friendly
    assert "src" in friendly
    assert "julia_cuda" in friendly
    assert "Pkg.instantiate()" in friendly
    assert "Julia reported:" in friendly
    assert "Package CUDA not found" in friendly


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
        shutdown_beat_engine_workers()


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
        shutdown_beat_engine_workers()
