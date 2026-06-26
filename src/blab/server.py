"""Local/network job server for Boundary Lab BEM solves."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from blab.config import SimulationConfig
from blab.live import LiveSolveDataset, LiveSolver
from blab.protocol import (
    frequency_result_to_dict,
    ndarray_to_wire,
    solve_request_to_job_inputs,
)
from blab.solvers.base import SolveRequest, SolverBackend
from blab.solvers.registry import backend_info, create_backend, normalize_backend_id

TERMINAL_STATES = {"completed", "cancelled", "failed"}
DEFAULT_ARTIFACT_ROOT = Path("runs") / "server_jobs"
ASSET_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
SERVER_SOLVER_BACKENDS = {"bempp_cpu", "beat_cpu", "beat_cuda", "beat_rocm"}
SERVER_SOLVER_BACKEND_IDS = {
    "bempp_cpu": "local",
    "beat_cpu": "beat_cpu",
    "beat_cuda": "beat_cuda",
    "beat_rocm": "beat_rocm",
}
LOGGER = logging.getLogger("blab.server")


class BackendServerSolver:
    """Server-side facade over the shared solver backend contract."""

    def __init__(
        self,
        config: SimulationConfig,
        *,
        backend: SolverBackend,
        status_callback=None,
    ) -> None:
        self.config = config
        self.backend = backend
        self.status_callback = status_callback
        self.session = None

    def set_status_callback(self, status_callback) -> None:
        self.status_callback = status_callback

    def initialize(self, frequencies_hz: np.ndarray) -> None:
        self.session = self.backend.create_session(
            SolveRequest(
                self.config,
                np.asarray(frequencies_hz, dtype=np.float32),
                status_callback=self.status_callback,
            )
        )

    @property
    def polar_angle_deg(self) -> np.ndarray:
        if self.session is None:
            raise RuntimeError("Server solver has not initialized.")
        return self.session.metadata.polar_angle_deg

    @property
    def radiator_names(self) -> np.ndarray:
        if self.session is None:
            raise RuntimeError("Server solver has not initialized.")
        return self.session.metadata.radiator_names

    @property
    def sphere_metadata(self) -> dict[str, np.ndarray] | None:
        if self.session is None:
            raise RuntimeError("Server solver has not initialized.")
        return self.session.metadata.sphere_metadata

    def solve_stream(self, frequencies, *, stop_requested=None):
        if self.session is None:
            self.initialize(np.asarray(frequencies, dtype=np.float32))
        yield from self.session.solve_stream(stop_requested=stop_requested)

@dataclass
class JobRecord:
    job_id: str
    config: SimulationConfig
    frequencies_hz: np.ndarray
    artifact_dir: Path
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    solved_count: int = 0
    event_count: int = 0
    result_npz: Path | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    events: list[dict[str, Any]] = field(default_factory=list)
    dataset: LiveSolveDataset | None = None

    @property
    def total_count(self) -> int:
        return int(self.frequencies_hz.size)

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "solved_count": self.solved_count,
            "total_count": self.total_count,
            "event_count": self.event_count,
            "artifacts": {
                "result_npz": None if self.result_npz is None else f"/jobs/{self.job_id}/artifacts/result.npz",
            },
        }


class JobOrchestrator:
    """Single-node job queue and event store."""

    def __init__(
        self,
        *,
        max_running_jobs: int = 1,
        artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
        solver_factory=LiveSolver,
    ) -> None:
        if max_running_jobs < 1:
            raise ValueError("max_running_jobs must be >= 1.")
        self.artifact_root = Path(artifact_root)
        self.solver_factory = solver_factory
        self._executor = ThreadPoolExecutor(max_workers=max_running_jobs, thread_name_prefix="blab-job")
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)

    def shutdown(self) -> None:
        with self._condition:
            for job in self._jobs.values():
                if job.status not in TERMINAL_STATES:
                    job.cancel_event.set()
            self._condition.notify_all()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def submit(
        self,
        config: SimulationConfig,
        frequencies_hz: np.ndarray,
        assets: list[dict[str, Any]] | None = None,
    ) -> JobRecord:
        frequencies = np.asarray(frequencies_hz, dtype=np.float32)
        if frequencies.size == 0:
            raise ValueError("frequencies_hz must contain at least one frequency.")

        job_id = uuid.uuid4().hex
        artifact_dir = self.artifact_root / job_id
        LOGGER.info(
            "job received job_id=%s frequencies=%s first_hz=%s last_hz=%s assets=%s artifact_dir=%s",
            job_id,
            frequencies.size,
            float(frequencies[0]),
            float(frequencies[-1]),
            0 if assets is None else len(assets),
            artifact_dir,
        )
        if assets:
            config = self._stage_assets(config, assets, artifact_dir / "assets")
        job = JobRecord(
            job_id=job_id,
            config=config,
            frequencies_hz=frequencies,
            artifact_dir=artifact_dir,
        )
        with self._condition:
            self._jobs[job_id] = job
            self._add_event_locked(job, "queued", total_count=job.total_count)
        self._executor.submit(self._run_job, job_id)
        return job

    def _stage_assets(
        self,
        config: SimulationConfig,
        assets: list[dict[str, Any]],
        asset_dir: Path,
    ) -> SimulationConfig:
        asset_dir.mkdir(parents=True, exist_ok=True)
        staged_by_original_path: dict[str, str] = {}
        used_names: set[str] = set()
        total_bytes = 0

        for index, asset in enumerate(assets):
            if not isinstance(asset, dict):
                raise ValueError("Each asset must be an object.")
            original_path = str(asset.get("original_path", ""))
            if not original_path:
                raise ValueError("Each asset must include original_path.")
            encoded = asset.get("content_base64")
            if not isinstance(encoded, str):
                raise ValueError(f"Asset {original_path} must include content_base64.")

            filename = _safe_asset_filename(str(asset.get("filename") or Path(original_path).name), index)
            while filename in used_names:
                filename = f"{index}_{filename}"
            used_names.add(filename)

            try:
                data = base64.b64decode(encoded.encode("ascii"), validate=True)
            except Exception as exc:
                raise ValueError(f"Asset {original_path} is not valid base64.") from exc

            staged_path = asset_dir / filename
            staged_path.write_bytes(data)
            total_bytes += len(data)
            staged_by_original_path[original_path] = str(staged_path)

        LOGGER.info("staged uploaded assets count=%s bytes=%s directory=%s", len(assets), total_bytes, asset_dir)
        return _rewrite_config_mesh_paths(config, staged_by_original_path)

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            LOGGER.info("job cancel requested job_id=%s status=%s", job_id, job.status)
            job.cancel_event.set()
            if job.status == "queued":
                job.status = "cancelled"
                job.finished_at = time.time()
                self._add_event_locked(job, "cancelled")
            else:
                self._add_event_locked(job, "cancelling")
            return True

    def events_since(self, job_id: str, start_index: int) -> tuple[list[dict[str, Any]], bool]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return list(job.events[start_index:]), job.status in TERMINAL_STATES

    def wait_for_event(self, job_id: str, start_index: int, timeout_s: float = 15.0) -> None:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                job = self._jobs.get(job_id)
                if job is None or len(job.events) > start_index or job.status in TERMINAL_STATES:
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                self._condition.wait(timeout=remaining)

    def _run_job(self, job_id: str) -> None:
        with self._condition:
            job = self._jobs[job_id]
            if job.cancel_event.is_set():
                job.status = "cancelled"
                job.finished_at = time.time()
                self._add_event_locked(job, "cancelled")
                LOGGER.info("job cancelled before start job_id=%s", job_id)
                return
            job.status = "running"
            job.started_at = time.time()
            self._add_event_locked(job, "started")
            LOGGER.info("job started job_id=%s frequencies=%s", job_id, job.total_count)

        try:
            solver = self.solver_factory(job.config)
            set_status_callback = getattr(solver, "set_status_callback", None)
            if callable(set_status_callback):
                set_status_callback(
                    lambda message: LOGGER.info("job solver status job_id=%s message=%s", job_id, message)
                )
            initialize = getattr(solver, "initialize", None)
            if callable(initialize):
                LOGGER.info("job initializing solver job_id=%s", job_id)
                initialize(job.frequencies_hz)
            sphere_metadata = solver.sphere_metadata or {}
            dataset = LiveSolveDataset(
                polar_angle_deg=np.asarray(solver.polar_angle_deg, dtype=np.float32),
                radiator_names=np.asarray(solver.radiator_names),
                sphere_r_distance_m=sphere_metadata.get("r_distance_m"),
                sphere_theta_polar_rad=sphere_metadata.get("theta_polar_rad"),
                sphere_phi_azimuth_rad=sphere_metadata.get("phi_azimuth_rad"),
            )
            with self._condition:
                job.dataset = dataset
                self._add_event_locked(
                    job,
                    "initialized",
                    polar_angle_deg=ndarray_to_wire(dataset.polar_angle_deg),
                    radiator_names=np.asarray(dataset.radiator_names).astype(str).tolist(),
                    sphere_metadata={key: ndarray_to_wire(value) for key, value in sphere_metadata.items()}
                    if sphere_metadata
                    else None,
                )
                LOGGER.info(
                    "job initialized job_id=%s angles=%s radiators=%s spherical=%s",
                    job_id,
                    dataset.polar_angle_deg.size,
                    dataset.radiator_names.size,
                    bool(sphere_metadata),
                )

            for result in solver.solve_stream(job.frequencies_hz, stop_requested=job.cancel_event.is_set):
                if job.cancel_event.is_set():
                    break
                dataset.add(result)
                with self._condition:
                    job.solved_count = dataset.solved_count
                    self._add_event_locked(
                        job,
                        "result",
                        result=frequency_result_to_dict(result),
                        solved_count=job.solved_count,
                        total_count=job.total_count,
                    )
                    LOGGER.info(
                        "job result job_id=%s solved=%s/%s freq_hz=%s",
                        job_id,
                        job.solved_count,
                        job.total_count,
                        float(result.freq_hz),
                    )

            with self._condition:
                if job.cancel_event.is_set():
                    job.status = "cancelled"
                    job.finished_at = time.time()
                    self._add_event_locked(job, "cancelled", solved_count=job.solved_count)
                    LOGGER.info("job cancelled job_id=%s solved=%s/%s", job_id, job.solved_count, job.total_count)
                    return

            artifact = self._write_result_artifact(job)
            with self._condition:
                job.result_npz = artifact
                job.status = "completed"
                job.finished_at = time.time()
                self._add_event_locked(
                    job,
                    "completed",
                    solved_count=job.solved_count,
                    artifact=f"/jobs/{job.job_id}/artifacts/result.npz",
                )
                elapsed_s = 0.0 if job.started_at is None else job.finished_at - job.started_at
                LOGGER.info(
                    "job completed job_id=%s solved=%s/%s elapsed_s=%.3f artifact=%s",
                    job_id,
                    job.solved_count,
                    job.total_count,
                    elapsed_s,
                    artifact,
                )
        except Exception as exc:
            with self._condition:
                job.status = "failed"
                job.error = str(exc)
                job.finished_at = time.time()
                self._add_event_locked(job, "failed", error=job.error)
            LOGGER.exception("job failed job_id=%s solved=%s/%s", job_id, job.solved_count, job.total_count)

    def _write_result_artifact(self, job: JobRecord) -> Path | None:
        dataset = job.dataset
        if dataset is None or dataset.solved_count == 0:
            return None

        job.artifact_dir.mkdir(parents=True, exist_ok=True)
        output_path = job.artifact_dir / "result.npz"
        freqs, angles, horizontal, vertical = dataset.as_polar_export_arrays()
        _, _, raw_horizontal, raw_vertical = dataset.as_raw_polar_arrays()
        ordered = dataset.ordered_results()
        impedance = np.stack([item.impedance for item in ordered], axis=1)

        bundle: dict[str, Any] = {
            "freq_hz": freqs,
            "polar_angle_deg": angles,
            "horizontal_spl_db": raw_horizontal,
            "vertical_spl_db": raw_vertical,
            "horizontal_spl_norm_db": horizontal,
            "vertical_spl_norm_db": vertical,
            "impedance_freq_hz": freqs,
            "impedance_radiator_names": dataset.radiator_names,
            "impedance_real": impedance[:, :, 0],
            "impedance_imag": impedance[:, :, 1],
        }
        raw_balloon = dataset.as_balloon_raw_bundle()
        if raw_balloon is not None:
            bundle.update(
                sphere_r_distance_m=raw_balloon["r_distance_m"],
                sphere_theta_polar_rad=raw_balloon["theta_polar_rad"],
                sphere_phi_azimuth_rad=raw_balloon["phi_azimuth_rad"],
                sphere_spl_norm_db=raw_balloon["spl_norm"],
            )

        np.savez_compressed(output_path, **bundle)
        return output_path

    def _add_event_locked(self, job: JobRecord, event_type: str, **payload: Any) -> None:
        event = {
            "index": len(job.events),
            "job_id": job.job_id,
            "type": event_type,
            "timestamp": time.time(),
            **payload,
        }
        job.events.append(event)
        job.event_count = len(job.events)
        self._condition.notify_all()


class BlabServer(ThreadingHTTPServer):
    def __init__(self, server_address, orchestrator: JobOrchestrator, *, solver_id: str = "bempp_cpu"):
        super().__init__(server_address, BlabRequestHandler)
        self.orchestrator = orchestrator
        self.solver_id = normalize_server_solver_id(solver_id)

    def health_payload(self) -> dict[str, Any]:
        backend_id = server_solver_backend_id(self.solver_id)
        info = backend_info(backend_id)
        capabilities = info.capabilities
        return {
            "status": "ok",
            "solver": self.solver_id,
            "backend": info.backend_id,
            "solver_label": info.label,
            "capabilities": {
                "supports_spherical_sampling": capabilities.supports_spherical_sampling,
                "supports_impedance": capabilities.supports_impedance,
                "supports_burton_miller": capabilities.supports_burton_miller,
                "supports_flat_target_normalization": capabilities.supports_flat_target_normalization,
                "supports_channel_resynthesis": capabilities.supports_channel_resynthesis,
                "supports_cancellation": capabilities.supports_cancellation,
                "supports_streaming": capabilities.supports_streaming,
                "supports_symmetry": capabilities.supports_symmetry,
            },
        }


class BlabRequestHandler(BaseHTTPRequestHandler):
    server: BlabServer

    def do_GET(self) -> None:  # noqa: N802 - stdlib override
        parsed = urlparse(self.path)
        parts = [part for part in parsed.path.split("/") if part]

        if parsed.path == "/health":
            self._send_json(self.server.health_payload())
            return

        if len(parts) == 2 and parts[0] == "jobs":
            self._handle_get_job(parts[1])
            return

        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "events":
            query = parse_qs(parsed.query)
            start = int(query.get("since", ["0"])[0])
            self._handle_job_events(parts[1], start)
            return

        if len(parts) == 4 and parts[0] == "jobs" and parts[2] == "artifacts" and parts[3] == "result.npz":
            self._handle_result_artifact(parts[1])
            return

        self._send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint.")

    def do_POST(self) -> None:  # noqa: N802 - stdlib override
        parsed = urlparse(self.path)
        parts = [part for part in parsed.path.split("/") if part]

        if parsed.path == "/jobs":
            self._handle_create_job()
            return

        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "cancel":
            if not self.server.orchestrator.cancel(parts[1]):
                self._send_error(HTTPStatus.NOT_FOUND, "Job not found.")
                return
            self._send_json({"job_id": parts[1], "status": "cancelling"})
            return

        self._send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint.")

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
        return

    def _handle_create_job(self) -> None:
        try:
            LOGGER.info(
                "POST /jobs received client=%s content_length=%s",
                self.client_address[0],
                self.headers.get("Content-Length", "0"),
            )
            payload = self._read_json()
            config, frequencies, assets = solve_request_to_job_inputs(payload)
            job = self.server.orchestrator.submit(config, frequencies, assets)
        except Exception as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return

        LOGGER.info("POST /jobs accepted job_id=%s client=%s", job.job_id, self.client_address[0])
        self._send_json(job.snapshot(), status=HTTPStatus.ACCEPTED)
    def _handle_get_job(self, job_id: str) -> None:
        job = self.server.orchestrator.get(job_id)
        if job is None:
            self._send_error(HTTPStatus.NOT_FOUND, "Job not found.")
            return
        self._send_json(job.snapshot())

    def _handle_job_events(self, job_id: str, start_index: int) -> None:
        if self.server.orchestrator.get(job_id) is None:
            self._send_error(HTTPStatus.NOT_FOUND, "Job not found.")
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        LOGGER.info("job event stream opened job_id=%s client=%s since=%s", job_id, self.client_address[0], start_index)
        next_index = max(0, start_index)
        try:
            while True:
                events, terminal = self.server.orchestrator.events_since(job_id, next_index)
                for event in events:
                    self.wfile.write(json.dumps(event, separators=(",", ":")).encode("utf-8") + b"\n")
                    self.wfile.flush()
                    next_index = event["index"] + 1
                if terminal:
                    LOGGER.info("job event stream completed job_id=%s events_sent_through=%s", job_id, next_index - 1)
                    return
                self.server.orchestrator.wait_for_event(job_id, next_index)
        except (BrokenPipeError, ConnectionResetError):
            LOGGER.warning("job event stream disconnected job_id=%s events_sent_through=%s", job_id, next_index - 1)

    def _handle_result_artifact(self, job_id: str) -> None:
        job = self.server.orchestrator.get(job_id)
        if job is None:
            self._send_error(HTTPStatus.NOT_FOUND, "Job not found.")
            return
        if job.result_npz is None or not job.result_npz.exists():
            self._send_error(HTTPStatus.NOT_FOUND, "Result artifact not available.")
            return

        data = job.result_npz.read_bytes()
        LOGGER.info('serving result artifact job_id=%s bytes=%s', job_id, len(data))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("Request body must contain JSON.")
        raw = self.rfile.read(length)
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Request JSON must be an object.")
        return payload

    def _send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        LOGGER.warning(
            "request failed status=%s message=%s path=%s client=%s",
            status.value,
            message,
            self.path,
            self.client_address[0],
        )
        self._send_json({"error": message}, status=status)


def _build_arg_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Run a Boundary Lab solve job server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host/IP address to bind.")
    parser.add_argument("--port", type=int, default=8765, help="TCP port to bind.")
    parser.add_argument(
        "--solver",
        default="bempp_cpu",
        help="Server-side solver backend: bempp_cpu, beat_cpu, beat_cuda, or beat_rocm.",
    )
    parser.add_argument(
        "--julia-executable",
        default="julia",
        help="Julia executable path for BEAT Engine server solvers.",
    )
    parser.add_argument(
        "--julia-threads",
        default="auto",
        help="Julia thread count for BEAT Engine server solvers.",
    )
    parser.add_argument(
        "--julia-sysimage",
        default=None,
        help="Optional Julia sysimage path for BEAT Engine server solvers.",
    )
    parser.add_argument(
        "--warm-solver",
        choices=("off", "worker", "tiny"),
        default="off",
        help="Warm BEAT Engine at startup: off, worker process only, or a tiny one-frequency solve.",
    )
    parser.add_argument("--max-running-jobs", type=int, default=1, help="Maximum concurrent solve jobs.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Server log level: DEBUG, INFO, WARNING, ERROR, or CRITICAL.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="Directory for completed job artifacts.",
    )
    return parser


def _safe_asset_filename(filename: str, index: int) -> str:
    cleaned = ASSET_FILENAME_PATTERN.sub("_", Path(filename).name).strip("._")
    return cleaned or f"asset_{index}.msh"


def _rewrite_config_mesh_paths(config: SimulationConfig, staged_by_original_path: dict[str, str]) -> SimulationConfig:
    mesh_file = staged_by_original_path.get(config.mesh_file, config.mesh_file)
    meshes = tuple(replace(mesh, file=staged_by_original_path.get(mesh.file, mesh.file)) for mesh in config.meshes)
    return replace(config, mesh_file=mesh_file, meshes=meshes)


def normalize_server_solver_id(solver_id: str) -> str:
    text = str(solver_id or "").strip().lower()
    normalized = normalize_backend_id(text)
    if normalized == "server":
        raise ValueError("The solve server cannot use another solve server as its own backend.")
    if normalized == "local":
        return "bempp_cpu"
    if normalized in SERVER_SOLVER_BACKENDS:
        return normalized
    supported = ", ".join(sorted(SERVER_SOLVER_BACKENDS))
    raise ValueError(f"Unsupported server solver backend '{solver_id}'. Expected one of: {supported}.")


def server_solver_backend_id(solver_id: str) -> str:
    return SERVER_SOLVER_BACKEND_IDS[normalize_server_solver_id(solver_id)]


def create_server_solver_factory(
    solver_id: str,
    *,
    julia_executable: str = "julia",
    julia_threads: str | int = "auto",
    julia_sysimage: str | Path | None = None,
):
    normalized_solver = normalize_server_solver_id(solver_id)
    backend = create_backend(
        server_solver_backend_id(normalized_solver),
        julia_executable=julia_executable,
        julia_threads=julia_threads,
        julia_sysimage=julia_sysimage,
    )

    def _factory(config: SimulationConfig) -> BackendServerSolver:
        return BackendServerSolver(config, backend=backend)

    _factory.backend = backend  # type: ignore[attr-defined]
    return _factory


def _warm_server_solver(solver_factory, mode: str) -> None:
    if mode == "off":
        return
    backend = getattr(solver_factory, "backend", None)
    warm_up = getattr(backend, "warm_up", None)
    if not callable(warm_up):
        LOGGER.info("solver warm-up skipped mode=%s reason=backend does not support warm-up", mode)
        return
    LOGGER.info("solver warm-up starting mode=%s", mode)
    started = time.monotonic()
    warm_up(mode, status_callback=lambda message: LOGGER.info("solver warm-up status message=%s", message))
    LOGGER.info("solver warm-up completed mode=%s elapsed_s=%.3f", mode, time.monotonic() - started)

def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    args = _build_arg_parser(prog=prog).parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    solver_id = normalize_server_solver_id(args.solver)
    orchestrator = JobOrchestrator(
        max_running_jobs=args.max_running_jobs,
        artifact_root=args.artifact_dir,
        solver_factory=create_server_solver_factory(
            solver_id,
            julia_executable=args.julia_executable,
            julia_threads=args.julia_threads,
            julia_sysimage=args.julia_sysimage,
        ),
    )
    server = BlabServer((args.host, args.port), orchestrator, solver_id=solver_id)
    _warm_server_solver(orchestrator.solver_factory, args.warm_solver)
    solver_label = backend_info(server_solver_backend_id(solver_id)).label
    LOGGER.info(
        "Boundary Lab server listening url=http://%s:%s solver=%s label=%s max_running_jobs=%s artifact_dir=%s",
        args.host,
        args.port,
        solver_id,
        solver_label,
        args.max_running_jobs,
        args.artifact_dir,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down Boundary Lab server.")
    finally:
        orchestrator.shutdown()
        try:
            from blab.solvers.beat_engine_backend import shutdown_beat_engine_workers

            shutdown_beat_engine_workers()
        except Exception:
            pass
        server.server_close()


if __name__ == "__main__":
    main()
