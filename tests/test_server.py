import json
import threading
import time
from pathlib import Path
from urllib import error, request

import numpy as np

from blab.config import SimulationConfig
from blab.live import FrequencyResult
from blab.protocol import build_mesh_assets
from blab.server import (
    BackendServerSolver,
    BlabServer,
    JobOrchestrator,
    JobQueueFullError,
    _build_arg_parser,
    normalize_server_solver_id,
)
from blab.solvers.base import SolveMetadata, SolveRequest
from blab.solvers.http_server import HttpServerSession


class FakeSolver:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.polar_angle_deg = np.array([-90.0, 0.0, 90.0], dtype=np.float32)
        self.radiator_names = np.array(["throat"])
        self.sphere_metadata = None

    def solve_stream(self, frequencies, *, stop_requested=None):
        for freq in frequencies:
            if stop_requested is not None and stop_requested():
                return
            yield FrequencyResult(
                freq_hz=float(freq),
                horizontal_spl_norm_db=np.array([-6.0, 0.0, -3.0], dtype=np.float32),
                vertical_spl_norm_db=np.array([-8.0, 0.0, -4.0], dtype=np.float32),
                impedance=np.array([[1.0, 0.2]], dtype=np.float32),
                horizontal_spl_db=np.array([82.0, 88.0, 85.0], dtype=np.float32),
                vertical_spl_db=np.array([80.0, 88.0, 84.0], dtype=np.float32),
            )


def _wait_for_terminal(orchestrator: JobOrchestrator, job_id: str):
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        job = orchestrator.get(job_id)
        if job is not None and job.status in {"completed", "cancelled", "failed"}:
            return job
        time.sleep(0.01)
    raise AssertionError("job did not reach a terminal state")


def test_orchestrator_streams_frequency_results_and_writes_artifact(tmp_path: Path) -> None:
    orchestrator = JobOrchestrator(
        max_running_jobs=1,
        artifact_root=tmp_path,
        solver_factory=FakeSolver,
    )
    try:
        job = orchestrator.submit(
            SimulationConfig(mesh_file="mesh.msh"),
            np.array([200.0, 1000.0], dtype=np.float32),
        )
        completed = _wait_for_terminal(orchestrator, job.job_id)

        events, terminal = orchestrator.events_since(job.job_id, 0)
        event_types = [event["type"] for event in events]

        assert terminal is True
        assert completed.status == "completed"
        assert completed.solved_count == 2
        assert event_types == ["queued", "started", "initialized", "result", "result", "completed"]
        assert completed.result_npz is not None
        assert completed.result_npz.exists()

        with np.load(completed.result_npz) as data:
            assert data["freq_hz"].tolist() == [200.0, 1000.0]
            assert data["horizontal_spl_norm_db"].shape == (2, 3)
            assert data["impedance_real"].shape == (1, 2)
    finally:
        orchestrator.shutdown()


def test_orchestrator_stages_uploaded_mesh_assets_before_solving(tmp_path: Path) -> None:
    source_mesh = tmp_path / "client_mesh.msh"
    source_mesh.write_text("mesh bytes", encoding="utf-8")
    seen_configs = []

    class CapturingSolver(FakeSolver):
        def __init__(self, config: SimulationConfig):
            seen_configs.append(config)
            super().__init__(config)

    orchestrator = JobOrchestrator(
        max_running_jobs=1,
        artifact_root=tmp_path / "jobs",
        solver_factory=CapturingSolver,
    )
    try:
        config = SimulationConfig(mesh_file=str(source_mesh))
        job = orchestrator.submit(
            config,
            np.array([1000.0], dtype=np.float32),
            assets=build_mesh_assets(config),
        )
        completed = _wait_for_terminal(orchestrator, job.job_id)

        assert completed.status == "completed"
        assert seen_configs
        staged_path = Path(seen_configs[0].mesh_file)
        assert staged_path != source_mesh
        assert staged_path.exists()
        assert staged_path.read_text(encoding="utf-8") == "mesh bytes"
        assert staged_path.parent == completed.artifact_dir / "assets"
    finally:
        orchestrator.shutdown()


def test_server_parser_accepts_backend_solver_options() -> None:
    args = _build_arg_parser().parse_args(
        [
            "--solver",
            "beat_cpu",
            "--julia-executable",
            "julia-custom",
            "--julia-threads",
            "4",
            "--log-level",
            "DEBUG",
            "--warm-solver",
            "tiny",
            "--julia-sysimage",
            "blab-beat-cuda.so",
        ]
    )

    assert args.solver == "beat_cpu"
    assert args.julia_executable == "julia-custom"
    assert args.julia_threads == "4"
    assert args.log_level == "DEBUG"
    assert args.warm_solver == "tiny"
    assert args.julia_sysimage == "blab-beat-cuda.so"
    assert args.max_request_mb == 20
    assert args.max_asset_mb == 14
    assert args.max_frequencies == 1000
    assert args.max_queued_jobs == 4
    assert args.job_retention_hours == 24
    assert args.event_stream_window_seconds == 25
    assert _build_arg_parser().parse_args([]).solver == "bempp_cpu"
    assert normalize_server_solver_id("bempp_cpu") == "bempp_cpu"
    assert normalize_server_solver_id("local") == "bempp_cpu"
    assert normalize_server_solver_id("cuda") == "beat_cuda"

    try:
        normalize_server_solver_id("server")
    except ValueError as exc:
        assert "cannot use another solve server" in str(exc)
    else:
        raise AssertionError("server should not be accepted as a server-side backend")


class CapturingBackend:
    def __init__(self):
        self.request = None

    def create_session(self, request):
        self.request = request
        return CapturingSession(request)


class CapturingSession:
    def __init__(self, request):
        self.request = request
        self.metadata = SolveMetadata(
            polar_angle_deg=np.array([0.0], dtype=np.float32),
            radiator_names=np.array(["driver"]),
        )

    def solve_stream(self, *, stop_requested=None):
        yield FrequencyResult(
            freq_hz=float(self.request.frequencies_hz[0]),
            horizontal_spl_norm_db=np.array([0.0], dtype=np.float32),
            vertical_spl_norm_db=np.array([0.0], dtype=np.float32),
            impedance=np.array([[1.0, 0.0]], dtype=np.float32),
            horizontal_spl_db=np.array([90.0], dtype=np.float32),
            vertical_spl_db=np.array([90.0], dtype=np.float32),
        )

    def stop(self) -> None:
        return None


def test_backend_server_solver_initializes_backend_with_job_frequencies() -> None:
    backend = CapturingBackend()
    solver = BackendServerSolver(
        SimulationConfig(mesh_file="mesh.msh"),
        backend=backend,
    )

    solver.initialize(np.array([123.0], dtype=np.float32))

    assert backend.request is not None
    assert backend.request.frequencies_hz.tolist() == [123.0]
    assert solver.polar_angle_deg.tolist() == [0.0]
    assert solver.radiator_names.tolist() == ["driver"]
    results = list(solver.solve_stream(np.array([456.0], dtype=np.float32)))
    assert len(results) == 1
    assert results[0].freq_hz == 123.0


def test_server_health_reports_configured_solver_capabilities(tmp_path: Path) -> None:
    orchestrator = JobOrchestrator(
        max_running_jobs=1,
        artifact_root=tmp_path,
        solver_factory=FakeSolver,
    )
    server = BlabServer(("127.0.0.1", 0), orchestrator, solver_id="beat_cpu")
    try:
        payload = server.health_payload()

        assert payload["status"] == "ok"
        assert payload["solver"] == "beat_cpu"
        assert payload["backend"] == "beat_cpu"
        assert payload["solver_label"] == "BEAT Engine (CPU)"
        assert payload["capabilities"]["supports_symmetry"] is True
    finally:
        orchestrator.shutdown()
        server.server_close()


def test_server_health_reports_bempp_cpu_as_public_solver_name(tmp_path: Path) -> None:
    orchestrator = JobOrchestrator(
        max_running_jobs=1,
        artifact_root=tmp_path,
        solver_factory=FakeSolver,
    )
    server = BlabServer(("127.0.0.1", 0), orchestrator, solver_id="local")
    try:
        payload = server.health_payload()

        assert payload["solver"] == "bempp_cpu"
        assert payload["backend"] == "local"
        assert payload["solver_label"] == "Bempp (OpenCL CPU)"
        assert payload["capabilities"]["supports_symmetry"] is False
    finally:
        orchestrator.shutdown()
        server.server_close()


def _start_test_server(
    tmp_path: Path,
    *,
    access_token: str = "",
    max_request_bytes: int = 20 * 1024 * 1024,
    event_stream_window_seconds: float = 25.0,
    solver_factory=FakeSolver,
) -> tuple[JobOrchestrator, BlabServer, threading.Thread, str]:
    orchestrator = JobOrchestrator(
        max_running_jobs=1,
        artifact_root=tmp_path,
        solver_factory=solver_factory,
    )
    server = BlabServer(
        ("127.0.0.1", 0),
        orchestrator,
        solver_id="beat_cpu",
        access_token=access_token,
        max_request_bytes=max_request_bytes,
        event_stream_window_seconds=event_stream_window_seconds,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return orchestrator, server, thread, f"http://{host}:{port}"


def _stop_test_server(orchestrator: JobOrchestrator, server: BlabServer, thread: threading.Thread) -> None:
    server.shutdown()
    thread.join(timeout=2.0)
    orchestrator.shutdown()
    server.server_close()


def test_server_authentication_is_optional_and_protects_health_when_enabled(tmp_path: Path) -> None:
    orchestrator, server, thread, server_url = _start_test_server(tmp_path, access_token="correct-token")
    try:
        try:
            request.urlopen(f"{server_url}/health", timeout=2)
        except error.HTTPError as exc:
            assert exc.code == 401
            assert exc.headers["WWW-Authenticate"] == "Bearer"
        else:
            raise AssertionError("authenticated server accepted a request without a token")

        health_request = request.Request(
            f"{server_url}/health",
            headers={"Authorization": "Bearer correct-token"},
        )
        with request.urlopen(health_request, timeout=2) as response:
            assert json.loads(response.read())["status"] == "ok"
    finally:
        _stop_test_server(orchestrator, server, thread)

    orchestrator, server, thread, server_url = _start_test_server(tmp_path / "open")
    try:
        with request.urlopen(f"{server_url}/health", timeout=2) as response:
            assert json.loads(response.read())["status"] == "ok"
    finally:
        _stop_test_server(orchestrator, server, thread)


def test_server_rejects_oversized_request_before_reading_json(tmp_path: Path) -> None:
    orchestrator, server, thread, server_url = _start_test_server(
        tmp_path,
        access_token="correct-token",
        max_request_bytes=16,
    )
    oversized = request.Request(
        f"{server_url}/jobs",
        data=b"{" + (b" " * 32) + b"}",
        headers={
            "Authorization": "Bearer correct-token",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        try:
            request.urlopen(oversized, timeout=2)
        except error.HTTPError as exc:
            assert exc.code == 413
        else:
            raise AssertionError("server accepted an oversized request")
    finally:
        _stop_test_server(orchestrator, server, thread)


def test_orchestrator_enforces_asset_frequency_and_queue_limits(tmp_path: Path) -> None:
    class BlockingSolver(FakeSolver):
        def solve_stream(self, frequencies, *, stop_requested=None):
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if stop_requested is not None and stop_requested():
                    return
                time.sleep(0.01)
            yield from super().solve_stream(frequencies, stop_requested=stop_requested)

    orchestrator = JobOrchestrator(
        max_running_jobs=1,
        max_queued_jobs=0,
        max_asset_bytes=4,
        max_frequencies=2,
        artifact_root=tmp_path,
        solver_factory=BlockingSolver,
    )
    try:
        try:
            orchestrator.submit(SimulationConfig(mesh_file="mesh.msh"), np.array([1.0, 2.0, 3.0]))
        except ValueError as exc:
            assert "server limit is 2" in str(exc)
        else:
            raise AssertionError("orchestrator accepted too many frequencies")

        asset = {
            "original_path": "mesh.msh",
            "filename": "mesh.msh",
            "content_base64": "MTIzNDU=",
        }
        try:
            orchestrator.submit(
                SimulationConfig(mesh_file="mesh.msh"),
                np.array([1000.0]),
                assets=[asset],
            )
        except ValueError as exc:
            assert "Uploaded assets exceed" in str(exc)
        else:
            raise AssertionError("orchestrator accepted oversized decoded assets")

        first = orchestrator.submit(SimulationConfig(mesh_file="mesh.msh"), np.array([1000.0]))
        try:
            orchestrator.submit(SimulationConfig(mesh_file="mesh.msh"), np.array([2000.0]))
        except JobQueueFullError:
            pass
        else:
            raise AssertionError("orchestrator accepted a job beyond queue capacity")
        orchestrator.cancel(first.job_id)
    finally:
        orchestrator.shutdown()


def test_orchestrator_purges_expired_job_state_and_artifacts(tmp_path: Path) -> None:
    orchestrator = JobOrchestrator(
        max_running_jobs=1,
        artifact_root=tmp_path,
        job_retention_hours=1,
        solver_factory=FakeSolver,
    )
    try:
        job = orchestrator.submit(SimulationConfig(mesh_file="mesh.msh"), np.array([1000.0]))
        completed = _wait_for_terminal(orchestrator, job.job_id)
        assert completed.artifact_dir.exists()
        assert completed.finished_at is not None

        purged = orchestrator.purge_expired_jobs(now=completed.finished_at + 3601)

        assert purged == 1
        assert orchestrator.get(job.job_id) is None
        assert not completed.artifact_dir.exists()
    finally:
        orchestrator.shutdown()


def test_http_client_reconnects_between_bounded_event_stream_windows(tmp_path: Path) -> None:
    mesh_path = tmp_path / "mesh.msh"
    mesh_path.write_text("mesh", encoding="utf-8")

    class SlowSolver(FakeSolver):
        def solve_stream(self, frequencies, *, stop_requested=None):
            time.sleep(0.08)
            yield from super().solve_stream(frequencies, stop_requested=stop_requested)

    orchestrator, server, thread, server_url = _start_test_server(
        tmp_path / "jobs",
        access_token="correct-token",
        event_stream_window_seconds=0.02,
        solver_factory=SlowSolver,
    )
    try:
        session = HttpServerSession(
            SolveRequest(
                SimulationConfig(mesh_file=str(mesh_path)),
                np.array([1000.0], dtype=np.float32),
            ),
            server_url,
            "correct-token",
        )
        results = list(session.solve_stream())
        assert [result.freq_hz for result in results] == [1000.0]
    finally:
        _stop_test_server(orchestrator, server, thread)
