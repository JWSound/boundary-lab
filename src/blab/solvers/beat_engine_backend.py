"""BEAT Engine solver backend adapter."""

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


DEFAULT_BEAT_ENGINE_SOLVER_SCRIPT = Path(__file__).with_name("julia_local") / "solver.jl"
DEFAULT_BEAT_ENGINE_CPU_PROJECT = DEFAULT_BEAT_ENGINE_SOLVER_SCRIPT.parent
DEFAULT_BEAT_ENGINE_CUDA_PROJECT = Path(__file__).with_name("julia_cuda")
DEFAULT_BEAT_ENGINE_PROJECT = DEFAULT_BEAT_ENGINE_CPU_PROJECT
_DEFAULT_BEAT_ENGINE_PROJECT_SENTINEL = "__default__"
_WORKERS_LOCK = threading.Lock()
_WORKERS: dict[tuple[str, str, str, str], "BeatEngineWorkerProcess"] = {}


BEAT_ENGINE_CUDA_BACKEND = "cuda"
BEAT_ENGINE_CPU_BACKEND = "cpu"
BEAT_ENGINE_BACKENDS = {BEAT_ENGINE_CUDA_BACKEND, BEAT_ENGINE_CPU_BACKEND}


class BeatEngineSession:
    def __init__(
        self,
        request_payload: SolveRequest,
        *,
        julia_executable: str = "julia",
        solver_script: str | Path = DEFAULT_BEAT_ENGINE_SOLVER_SCRIPT,
        julia_threads: str | int = "auto",
        julia_project: str | Path | None = _DEFAULT_BEAT_ENGINE_PROJECT_SENTINEL,
        persistent_worker: bool = True,
        beat_engine_backend: str = BEAT_ENGINE_CUDA_BACKEND,
    ):
        self.request_payload = request_payload
        self.julia_executable = julia_executable.strip() or "julia"
        self.solver_script = Path(solver_script)
        self.julia_threads = julia_threads
        self.persistent_worker = persistent_worker
        self.beat_engine_backend = _normalize_beat_engine_backend(beat_engine_backend)
        if julia_project == _DEFAULT_BEAT_ENGINE_PROJECT_SENTINEL:
            self.julia_project = _default_beat_engine_project(self.beat_engine_backend)
        else:
            self.julia_project = None if julia_project is None else Path(julia_project)
        self._stop = False
        self._temp_dir = tempfile.TemporaryDirectory(prefix="blab-beat-engine-")
        self._process: subprocess.Popen[str] | None = None
        self._worker: BeatEngineWorkerProcess | None = None
        self._events: Iterator[dict] | None = None
        self._metadata: SolveMetadata | None = None
        self._stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._start_and_initialize()

    @property
    def metadata(self) -> SolveMetadata:
        if self._metadata is None:
            raise RuntimeError("BEAT Engine solver session has not initialized.")
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
                    self._emit_status(str(event.get("message", "")))
                    continue
                elif event_type == "cancelled":
                    return
                elif event_type == "completed":
                    return
                elif event_type == "failed":
                    raise RuntimeError(
                        _friendly_julia_error(
                            str(event.get("error", "BEAT Engine solver failed.")),
                            julia_project=self.julia_project,
                            beat_engine_backend=self.beat_engine_backend,
                        )
                    )
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
            raise RuntimeError(f"BEAT Engine solver script does not exist: {self.solver_script}")

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
        payload["beat_engine_backend"] = self.beat_engine_backend
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
                self._emit_status(str(event.get("message", "")))
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
                raise RuntimeError(
                    _friendly_julia_error(
                        str(event.get("error", "BEAT Engine solver failed.")),
                        julia_project=self.julia_project,
                        beat_engine_backend=self.beat_engine_backend,
                    )
                )
            elif event_type in {"completed", "cancelled"}:
                raise RuntimeError(f"BEAT Engine solver ended before initialization: {event_type}")

        raise RuntimeError(self._process_error("BEAT Engine solver ended before initialization."))

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
            raise RuntimeError(self._process_error(f"BEAT Engine solver exited with code {exit_code}."))

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
        message = f"{fallback}\n{detail}" if detail else fallback
        return _friendly_julia_error(
            message,
            julia_project=self.julia_project,
            beat_engine_backend=self.beat_engine_backend,
            detection_text="\n".join(self._stderr_lines),
        )

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


class BeatEngineWorkerProcess:
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
                raise RuntimeError("Warm BEAT Engine solver did not provide stdin.")
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
                raise RuntimeError(
                    _friendly_julia_error(
                        str(event.get("error", "BEAT Engine solver failed during startup.")),
                        julia_project=self.julia_project,
                    )
                )

        raise RuntimeError(self._process_error("Warm BEAT Engine solver ended before startup completed."))

    def _iter_events_for_submission(self) -> Iterator[dict]:
        try:
            for event in self._read_events():
                yield event
                if str(event.get("type", "")) in {"completed", "cancelled", "failed"}:
                    return
            raise RuntimeError(self._process_error("Warm BEAT Engine solver ended before job completion."))
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
            raise RuntimeError(self._process_error(f"Warm BEAT Engine solver exited with code {exit_code}."))

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
        message = f"{fallback}\n{detail}" if detail else fallback
        return _friendly_julia_error(
            message,
            julia_project=self.julia_project,
            detection_text="\n".join(self._stderr_lines),
        )


class BeatEngineBackend:
    backend_id = "beat_cuda"
    label = "BEAT Engine (CUDA)"
    beat_engine_backend = BEAT_ENGINE_CUDA_BACKEND
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
        solver_script: str | Path = DEFAULT_BEAT_ENGINE_SOLVER_SCRIPT,
        julia_threads: str | int = "auto",
        julia_project: str | Path | None = _DEFAULT_BEAT_ENGINE_PROJECT_SENTINEL,
        persistent_worker: bool = True,
        backend_id: str | None = None,
        label: str | None = None,
        beat_engine_backend: str | None = None,
    ):
        self.julia_executable = julia_executable
        self.solver_script = Path(solver_script)
        self.julia_threads = julia_threads
        self.persistent_worker = persistent_worker
        if backend_id is not None:
            self.backend_id = backend_id
        if label is not None:
            self.label = label
        if beat_engine_backend is not None:
            self.beat_engine_backend = _normalize_beat_engine_backend(beat_engine_backend)
        if julia_project == _DEFAULT_BEAT_ENGINE_PROJECT_SENTINEL:
            self.julia_project = _default_beat_engine_project(self.beat_engine_backend)
        else:
            self.julia_project = julia_project
        if self.beat_engine_backend == BEAT_ENGINE_CPU_BACKEND:
            self.capabilities = BeatEngineCpuBackend.capabilities

    def create_session(self, request: SolveRequest) -> BeatEngineSession:
        return BeatEngineSession(
            request,
            julia_executable=self.julia_executable,
            solver_script=self.solver_script,
            julia_threads=self.julia_threads,
            julia_project=self.julia_project,
            persistent_worker=self.persistent_worker,
            beat_engine_backend=self.beat_engine_backend,
        )


class BeatEngineCudaBackend(BeatEngineBackend):
    backend_id = "beat_cuda"
    label = "BEAT Engine (CUDA)"
    beat_engine_backend = BEAT_ENGINE_CUDA_BACKEND


class BeatEngineCpuBackend(BeatEngineBackend):
    backend_id = "beat_cpu"
    label = "BEAT Engine (CPU)"
    beat_engine_backend = BEAT_ENGINE_CPU_BACKEND
    capabilities = SolverCapabilities(
        supports_remote_assets=False,
        supports_parallel_workers=False,
        supports_symmetry=True,
        supports_channel_resynthesis=True,
        is_remote=False,
    )


def _normalize_beat_engine_backend(value: object) -> str:
    text = str(value or BEAT_ENGINE_CUDA_BACKEND).strip().lower()
    aliases = {
        "beat_cuda": BEAT_ENGINE_CUDA_BACKEND,
        "cuda": BEAT_ENGINE_CUDA_BACKEND,
        "gpu": BEAT_ENGINE_CUDA_BACKEND,
        "julia_local": BEAT_ENGINE_CUDA_BACKEND,
        "local_julia": BEAT_ENGINE_CUDA_BACKEND,
        "afterburner": BEAT_ENGINE_CUDA_BACKEND,
        "beat_cpu": BEAT_ENGINE_CPU_BACKEND,
        "cpu": BEAT_ENGINE_CPU_BACKEND,
    }
    backend = aliases.get(text, text)
    if backend not in BEAT_ENGINE_BACKENDS:
        raise ValueError(f"Unknown BEAT Engine backend: {value}")
    return backend


def _default_beat_engine_project(beat_engine_backend: str) -> Path:
    return (
        DEFAULT_BEAT_ENGINE_CPU_PROJECT
        if beat_engine_backend == BEAT_ENGINE_CPU_BACKEND
        else DEFAULT_BEAT_ENGINE_CUDA_PROJECT
    )


def _friendly_julia_error(
    message: str,
    *,
    julia_project: str | Path | None,
    beat_engine_backend: str | None = None,
    detection_text: str | None = None,
) -> str:
    if julia_project is None:
        return message

    text = f"{detection_text or message}\n{message}".lower()
    missing_dependency_markers = (
        "argumenterror: package",
        "not found in current path",
        "run `import pkg; pkg.add",
        "could not load project",
        "failed to precompile",
    )
    julia_load_markers = (
        "loading.jl",
        "require(into::module",
        "require(uuidkey::base.pkgid",
    )
    cuda_load_markers = (
        "cuda.jl could not be loaded",
        "package cuda",
        "using cuda",
        "import cuda",
    )
    looks_like_dependency_error = any(marker in text for marker in missing_dependency_markers)
    looks_like_julia_load_error = any(marker in text for marker in julia_load_markers)
    looks_like_cuda_error = any(marker in text for marker in cuda_load_markers)
    if not (looks_like_dependency_error or looks_like_julia_load_error or looks_like_cuda_error):
        return message

    project_path = Path(julia_project)
    backend_label = _julia_project_backend_label(project_path, beat_engine_backend)
    install_command = f'julia --project={project_path} -e "using Pkg; Pkg.instantiate()"'
    return (
        f"BEAT Engine could not load the Julia dependencies for {backend_label}.\n\n"
        "This usually means the selected BEAT Engine Julia environment has not been installed yet. "
        "From the Boundary Lab repository root, run:\n\n"
        f"{install_command}\n\n"
        f"Julia reported:\n{message}"
    )


def _julia_project_backend_label(project_path: Path, beat_engine_backend: str | None) -> str:
    if beat_engine_backend == BEAT_ENGINE_CUDA_BACKEND or project_path == DEFAULT_BEAT_ENGINE_CUDA_PROJECT:
        return "BEAT Engine (CUDA)"
    if beat_engine_backend == BEAT_ENGINE_CPU_BACKEND or project_path == DEFAULT_BEAT_ENGINE_CPU_PROJECT:
        return "BEAT Engine (CPU)"
    return "the selected BEAT Engine backend"


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
) -> BeatEngineWorkerProcess:
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
            worker = BeatEngineWorkerProcess(
                julia_executable=julia_executable,
                solver_script=solver_script,
                julia_threads=resolved_threads,
                julia_project=julia_project,
            )
            _WORKERS[key] = worker
        return worker


def shutdown_beat_engine_workers() -> None:
    with _WORKERS_LOCK:
        workers = list(_WORKERS.values())
        _WORKERS.clear()
    for worker in workers:
        worker.terminate()


DEFAULT_JULIA_SOLVER_SCRIPT = DEFAULT_BEAT_ENGINE_SOLVER_SCRIPT
DEFAULT_JULIA_PROJECT = DEFAULT_BEAT_ENGINE_PROJECT
JuliaLocalSession = BeatEngineSession
JuliaWorkerProcess = BeatEngineWorkerProcess
JuliaLocalBackend = BeatEngineBackend
shutdown_julia_workers = shutdown_beat_engine_workers

# Legacy aliases retained for old imports and saved tool scripts.
DEFAULT_AFTERBURNER_SOLVER_SCRIPT = DEFAULT_BEAT_ENGINE_SOLVER_SCRIPT
DEFAULT_AFTERBURNER_PROJECT = DEFAULT_BEAT_ENGINE_PROJECT
AfterburnerSession = BeatEngineSession
AfterburnerWorkerProcess = BeatEngineWorkerProcess
AfterburnerBackend = BeatEngineBackend
shutdown_afterburner_workers = shutdown_beat_engine_workers
