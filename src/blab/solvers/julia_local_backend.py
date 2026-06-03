"""Local Julia solver backend adapter."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from blab.config import SimulationConfig
from blab.protocol import (
    build_mesh_assets,
    frequency_result_from_dict,
    ndarray_from_wire,
    solve_request_from_config_and_frequencies,
)
from blab.server import _safe_asset_filename
from blab.solvers.base import FrequencyResult, SolveMetadata, SolveRequest, SolverCapabilities


DEFAULT_JULIA_SOLVER_SCRIPT = Path(__file__).with_name("julia_local") / "solver.jl"
DEFAULT_JULIA_PROJECT = DEFAULT_JULIA_SOLVER_SCRIPT.parent
_WORKERS_LOCK = threading.Lock()
_WORKERS: dict[tuple[str, str, str, str], "JuliaWorkerProcess"] = {}


class JuliaLocalSession:
    def __init__(
        self,
        request_payload: SolveRequest,
        *,
        julia_executable: str = "julia",
        solver_script: str | Path = DEFAULT_JULIA_SOLVER_SCRIPT,
        julia_threads: str | int = "auto",
        julia_project: str | Path | None = DEFAULT_JULIA_PROJECT,
        persistent_worker: bool = True,
    ):
        self.request_payload = request_payload
        self.julia_executable = julia_executable.strip() or "julia"
        self.solver_script = Path(solver_script)
        self.julia_threads = julia_threads
        self.julia_project = None if julia_project is None else Path(julia_project)
        self.persistent_worker = persistent_worker
        self._stop = False
        self._temp_dir = tempfile.TemporaryDirectory(prefix="blab-julia-")
        self._process: subprocess.Popen[str] | None = None
        self._worker: JuliaWorkerProcess | None = None
        self._events: Iterator[dict] | None = None
        self._metadata: SolveMetadata | None = None
        self._stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._start_and_initialize()

    @property
    def metadata(self) -> SolveMetadata:
        if self._metadata is None:
            raise RuntimeError("Julia solver session has not initialized.")
        return self._metadata

    def solve_stream(
        self,
        *,
        stop_requested: Callable[[], bool] | None = None,
    ) -> Iterator[FrequencyResult]:
        if self._events is None:
            return

        try:
            for event in self._events:
                if self._stop or (stop_requested is not None and stop_requested()):
                    self.stop()

                event_type = str(event.get("type", ""))
                if event_type == "result":
                    if not self._stop:
                        yield frequency_result_from_dict(event["result"])
                elif event_type == "status":
                    continue
                elif event_type == "cancelled":
                    return
                elif event_type == "completed":
                    return
                elif event_type == "failed":
                    raise RuntimeError(str(event.get("error", "Julia solver failed.")))
        finally:
            self._close()

    def stop(self) -> None:
        self._stop = True
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        if self._worker is not None:
            self._request_worker_cancel()

    def _request_worker_cancel(self) -> None:
        cancel_path = getattr(self, "_cancel_path", None)
        if cancel_path is not None:
            Path(cancel_path).write_text("cancel", encoding="utf-8")

    def _start_and_initialize(self) -> None:
        if not self.solver_script.exists():
            raise RuntimeError(f"Julia solver script does not exist: {self.solver_script}")

        job_dir = Path(self._temp_dir.name)
        request_path = job_dir / "request.json"
        self._cancel_path = job_dir / "cancel"
        config = _stage_config_assets(self.request_payload.config, job_dir / "assets")
        payload = solve_request_from_config_and_frequencies(
            config,
            self.request_payload.frequencies_hz,
            include_assets=False,
        )
        payload["cancel_path"] = str(self._cancel_path)
        request_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

        if self.persistent_worker:
            self._worker = _get_julia_worker(
                julia_executable=self.julia_executable,
                solver_script=self.solver_script,
                julia_threads=self.julia_threads,
                julia_project=self.julia_project,
            )
            self._events = self._worker.submit(request_path)
        else:
            try:
                self._process = subprocess.Popen(
                    _julia_command(
                        self.julia_executable,
                        self.solver_script,
                        request_path,
                        julia_project=self.julia_project,
                    ),
                    cwd=str(job_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=_julia_process_env(self.julia_threads),
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "Julia executable was not found. Set the Julia executable path in Preferences."
                ) from exc

            self._stderr_thread = threading.Thread(target=self._collect_stderr, daemon=True)
            self._stderr_thread.start()
            self._events = self._iter_events()

        for event in self._events:
            event_type = str(event.get("type", ""))
            if event_type == "status":
                continue
            elif event_type == "initialized":
                sphere_metadata = event.get("sphere_metadata") or {}
                self._metadata = SolveMetadata(
                    polar_angle_deg=ndarray_from_wire(event["polar_angle_deg"]),
                    radiator_names=np.asarray(event.get("radiator_names", ["Radiator"])),
                    sphere_metadata={
                        key: ndarray_from_wire(value)
                        for key, value in sphere_metadata.items()
                    },
                )
                return
            elif event_type == "failed":
                raise RuntimeError(str(event.get("error", "Julia solver failed.")))
            elif event_type in {"completed", "cancelled"}:
                raise RuntimeError(f"Julia solver ended before initialization: {event_type}")

        raise RuntimeError(self._process_error("Julia solver ended before initialization."))

    def _iter_events(self) -> Iterator[dict]:
        process = self._process
        if process is None or process.stdout is None:
            return

        for line in process.stdout:
            text = line.strip()
            if not text:
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                self._emit_status(text)
                continue
            if not isinstance(event, dict):
                continue
            yield event

        exit_code = process.wait()
        if exit_code != 0 and not self._stop:
            raise RuntimeError(self._process_error(f"Julia solver exited with code {exit_code}."))

    def _collect_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            text = line.strip()
            if text:
                self._stderr_lines.append(text)
                self._emit_status(text)

    def _process_error(self, fallback: str) -> str:
        detail = "\n".join(self._stderr_lines[-10:])
        return f"{fallback}\n{detail}" if detail else fallback

    def _close(self) -> None:
        events = self._events
        self._events = None
        close = getattr(events, "close", None)
        if close is not None:
            close()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        self._temp_dir.cleanup()

    def _emit_status(self, message: str) -> None:
        if self.request_payload.status_callback is not None:
            self.request_payload.status_callback(message)


class JuliaWorkerProcess:
    def __init__(
        self,
        *,
        julia_executable: str,
        solver_script: Path,
        julia_threads: str | int,
        julia_project: Path | None,
    ):
        self.julia_executable = julia_executable
        self.solver_script = solver_script
        self.julia_threads = julia_threads
        self.julia_project = julia_project
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None

    def submit(self, request_path: Path) -> Iterator[dict]:
        self._lock.acquire()
        try:
            self._ensure_started()
            process = self._process
            if process is None or process.stdin is None:
                raise RuntimeError("Warm Julia solver did not provide stdin.")
            process.stdin.write(json.dumps({"request": str(request_path)}, separators=(",", ":")) + "\n")
            process.stdin.flush()
            return self._iter_events_for_submission()
        except Exception:
            self._lock.release()
            raise

    def terminate(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        if self._lock.locked():
            try:
                self._lock.release()
            except RuntimeError:
                pass

    def _ensure_started(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return

        self._stderr_lines.clear()
        try:
            self._process = subprocess.Popen(
                _julia_worker_command(
                    self.julia_executable,
                    self.solver_script,
                    julia_project=self.julia_project,
                ),
                cwd=str(self.solver_script.parent),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=_julia_process_env(self.julia_threads),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Julia executable was not found. Set the Julia executable path in Preferences."
            ) from exc

        self._stderr_thread = threading.Thread(target=self._collect_stderr, daemon=True)
        self._stderr_thread.start()

        for event in self._read_events():
            event_type = str(event.get("type", ""))
            if event_type == "ready":
                return
            if event_type == "failed":
                raise RuntimeError(str(event.get("error", "Julia solver failed during startup.")))

        raise RuntimeError(self._process_error("Warm Julia solver ended before startup completed."))

    def _iter_events_for_submission(self) -> Iterator[dict]:
        try:
            for event in self._read_events():
                yield event
                if str(event.get("type", "")) in {"completed", "cancelled", "failed"}:
                    return
            raise RuntimeError(self._process_error("Warm Julia solver ended before job completion."))
        finally:
            if self._lock.locked():
                self._lock.release()

    def _read_events(self) -> Iterator[dict]:
        process = self._process
        if process is None or process.stdout is None:
            return

        for line in process.stdout:
            text = line.strip()
            if not text:
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                yield {"type": "status", "message": text}
                continue
            if isinstance(event, dict):
                yield event

        exit_code = process.wait()
        self._process = None
        if exit_code != 0:
            raise RuntimeError(self._process_error(f"Warm Julia solver exited with code {exit_code}."))

    def _collect_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            text = line.strip()
            if text:
                self._stderr_lines.append(text)

    def _process_error(self, fallback: str) -> str:
        detail = "\n".join(self._stderr_lines[-10:])
        return f"{fallback}\n{detail}" if detail else fallback


class JuliaLocalBackend:
    backend_id = "julia_local"
    label = "Julia CUDA GPU"
    capabilities = SolverCapabilities(
        supports_remote_assets=False,
        supports_parallel_workers=False,
        supports_symmetry=True,
        supports_channel_resynthesis=True,
        is_remote=False,
    )

    def __init__(
        self,
        *,
        julia_executable: str = "julia",
        solver_script: str | Path = DEFAULT_JULIA_SOLVER_SCRIPT,
        julia_threads: str | int = "auto",
        julia_project: str | Path | None = DEFAULT_JULIA_PROJECT,
        persistent_worker: bool = True,
    ):
        self.julia_executable = julia_executable
        self.solver_script = Path(solver_script)
        self.julia_threads = julia_threads
        self.julia_project = julia_project
        self.persistent_worker = persistent_worker

    def create_session(self, request: SolveRequest) -> JuliaLocalSession:
        return JuliaLocalSession(
            request,
            julia_executable=self.julia_executable,
            solver_script=self.solver_script,
            julia_threads=self.julia_threads,
            julia_project=self.julia_project,
            persistent_worker=self.persistent_worker,
        )


def _stage_config_assets(config: SimulationConfig, asset_dir: Path) -> SimulationConfig:
    assets = build_mesh_assets(config)
    if not assets:
        return config

    asset_dir.mkdir(parents=True, exist_ok=True)
    staged_by_original_path: dict[str, str] = {}
    used_names: set[str] = set()
    for index, asset in enumerate(assets):
        original_path = str(asset["original_path"])
        filename = _safe_asset_filename(str(asset.get("filename") or Path(original_path).name), index)
        while filename in used_names:
            filename = f"{index}_{filename}"
        used_names.add(filename)
        staged_path = asset_dir / filename
        staged_path.write_bytes(Path(original_path).read_bytes())
        staged_by_original_path[original_path] = str(staged_path)

    meshes = tuple(
        replace(mesh, file=staged_by_original_path.get(mesh.file, mesh.file))
        for mesh in config.meshes
    )
    return replace(
        config,
        mesh_file=staged_by_original_path.get(config.mesh_file, config.mesh_file),
        meshes=meshes,
    )


def _julia_process_env(julia_threads: str | int = "auto") -> dict[str, str]:
    env = os.environ.copy()
    env["JULIA_NUM_THREADS"] = _resolve_julia_threads(julia_threads)
    return env


def _resolve_julia_threads(julia_threads: str | int = "auto") -> str:
    if isinstance(julia_threads, int):
        return str(max(1, julia_threads))

    text = str(julia_threads or "auto").strip().lower()
    if text == "auto":
        return str(os.cpu_count() or 1)

    try:
        return str(max(1, int(text)))
    except ValueError:
        return str(os.cpu_count() or 1)


def _julia_command(
    julia_executable: str,
    solver_script: Path,
    request_path: Path,
    *,
    julia_project: Path | None,
) -> list[str]:
    command = [julia_executable]
    if julia_project is not None:
        command.append(f"--project={julia_project}")
        command.append("--startup-file=no")
    command.extend([str(solver_script), "--request", str(request_path)])
    return command


def _julia_worker_command(
    julia_executable: str,
    solver_script: Path,
    *,
    julia_project: Path | None,
) -> list[str]:
    command = [julia_executable]
    if julia_project is not None:
        command.append(f"--project={julia_project}")
        command.append("--startup-file=no")
    command.extend([str(solver_script), "--worker"])
    return command


def _get_julia_worker(
    *,
    julia_executable: str,
    solver_script: Path,
    julia_threads: str | int,
    julia_project: Path | None,
) -> JuliaWorkerProcess:
    resolved_threads = _resolve_julia_threads(julia_threads)
    key = (
        julia_executable,
        str(solver_script.resolve()),
        "" if julia_project is None else str(julia_project.resolve()),
        resolved_threads,
    )
    with _WORKERS_LOCK:
        worker = _WORKERS.get(key)
        if worker is None:
            worker = JuliaWorkerProcess(
                julia_executable=julia_executable,
                solver_script=solver_script,
                julia_threads=resolved_threads,
                julia_project=julia_project,
            )
            _WORKERS[key] = worker
        return worker


def shutdown_julia_workers() -> None:
    with _WORKERS_LOCK:
        workers = list(_WORKERS.values())
        _WORKERS.clear()
    for worker in workers:
        worker.terminate()
