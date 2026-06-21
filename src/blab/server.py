"""Local/network job server for Boundary Lab BEM solves."""

from __future__ import annotations

import argparse
import base64
import json
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

TERMINAL_STATES = {"completed", "cancelled", "failed"}
DEFAULT_ARTIFACT_ROOT = Path("runs") / "server_jobs"
ASSET_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


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
            staged_by_original_path[original_path] = str(staged_path)

        return _rewrite_config_mesh_paths(config, staged_by_original_path)

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None:
                return False
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
                return
            job.status = "running"
            job.started_at = time.time()
            self._add_event_locked(job, "started")

        try:
            solver = self.solver_factory(job.config)
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

            with self._condition:
                if job.cancel_event.is_set():
                    job.status = "cancelled"
                    job.finished_at = time.time()
                    self._add_event_locked(job, "cancelled", solved_count=job.solved_count)
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
        except Exception as exc:
            with self._condition:
                job.status = "failed"
                job.error = str(exc)
                job.finished_at = time.time()
                self._add_event_locked(job, "failed", error=job.error)

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
    def __init__(self, server_address, orchestrator: JobOrchestrator):
        super().__init__(server_address, BlabRequestHandler)
        self.orchestrator = orchestrator


class BlabRequestHandler(BaseHTTPRequestHandler):
    server: BlabServer

    def do_GET(self) -> None:  # noqa: N802 - stdlib override
        parsed = urlparse(self.path)
        parts = [part for part in parsed.path.split("/") if part]

        if parsed.path == "/health":
            self._send_json({"status": "ok"})
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
            payload = self._read_json()
            config, frequencies, assets = solve_request_to_job_inputs(payload)
            job = self.server.orchestrator.submit(config, frequencies, assets)
        except Exception as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return

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

        next_index = max(0, start_index)
        while True:
            events, terminal = self.server.orchestrator.events_since(job_id, next_index)
            for event in events:
                self.wfile.write(json.dumps(event, separators=(",", ":")).encode("utf-8") + b"\n")
                self.wfile.flush()
                next_index = event["index"] + 1
            if terminal:
                return
            self.server.orchestrator.wait_for_event(job_id, next_index)

    def _handle_result_artifact(self, job_id: str) -> None:
        job = self.server.orchestrator.get(job_id)
        if job is None:
            self._send_error(HTTPStatus.NOT_FOUND, "Job not found.")
            return
        if job.result_npz is None or not job.result_npz.exists():
            self._send_error(HTTPStatus.NOT_FOUND, "Result artifact not available.")
            return

        data = job.result_npz.read_bytes()
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
        self._send_json({"error": message}, status=status)


def _build_arg_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Run a Boundary Lab solve job server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host/IP address to bind.")
    parser.add_argument("--port", type=int, default=8765, help="TCP port to bind.")
    parser.add_argument("--max-running-jobs", type=int, default=1, help="Maximum concurrent solve jobs.")
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


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    args = _build_arg_parser(prog=prog).parse_args(argv)
    orchestrator = JobOrchestrator(
        max_running_jobs=args.max_running_jobs,
        artifact_root=args.artifact_dir,
    )
    server = BlabServer((args.host, args.port), orchestrator)
    print(f"Boundary Lab server listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Boundary Lab server.")
    finally:
        orchestrator.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
